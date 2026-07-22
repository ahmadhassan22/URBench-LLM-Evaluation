"""
efbpt_blind30_score.py — CPU-only deterministic scorer for BLIND30.

Measures: "how often did a human need to correct the automatic annotation?"
(gold was created by EDITING predictions — this is a correction-rate study,
NOT independent-annotator agreement; urdu_span may be inflated: prefilled).
No LLM judge. No fuzzy matching. Modifies nothing.
"""
import json, re, math, hashlib, sys
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
OFF   = BASE / "data/strategyqa_official"
EF    = OFF / "efbpt"

CANDS = EF / "blind30_candidates.jsonl"
PREDS = EF / "blind30_predictions.jsonl"
GOLD  = EF / "blind30_gold.jsonl"
REVW  = EF / "blind30_review.jsonl"
A30   = EF / "audit30_answers.jsonl"
DEV50 = OFF / "dev50_seed42_qids.txt"
MAP_F = OFF / "strategyqa_official_mapped_urbench_qid.jsonl"

OUT_S = EF / "blind30_score_summary.txt"
OUT_D = EF / "blind30_score_diffs.jsonl"

INSPECT_QIDS = ["a44c3d9161f5e5c3f41d", "e175b012fc9b5db8da3f", "af2d3c137bd3f5230012"]
STRUCT_PREFIXES = ("parse_failure", "title_missing", "title_duplicate", "title_unknown")

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

# ---------- frozen normalization (same as pilot) ----------
def norm(s):
    s = re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_", " ")).strip().casefold()
    return s.strip(" .,:;\"'")

BOOL_TRUE  = {"yes", "true", "ہاں"}
BOOL_FALSE = {"no", "false", "نہیں"}

def canon_bool(s):
    n = norm(s)
    if n in BOOL_TRUE:  return True
    if n in BOOL_FALSE: return False
    return None

def gia_strict_eq(pv, gv):
    """PRIMARY: normalized exact equality, with boolean canonicalization."""
    if pv is None and gv is None: return True
    bp, bg = canon_bool(pv), canon_bool(gv)
    if bp is not None and bg is not None: return bp == bg
    return norm(pv) == norm(gv)

# ---------- SECONDARY: cautious type-aware equivalence ----------
MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9}
NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)\s*(thousand|million|billion|k|m|b)?\s*([a-z%°]+)?", re.IGNORECASE)

def parse_numbers(s):
    """Return list of (value, unit) found; None if nothing parseable."""
    out = []
    for m in NUM_RE.finditer(str(s)):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if m.group(2): v *= MULT[m.group(2).lower()]
        unit = norm(m.group(3)) if m.group(3) else ""
        out.append((v, unit))
    return out or None

def num_eq(pv, gv):
    a, b = parse_numbers(pv), parse_numbers(gv)
    if a is None or b is None or len(a) != len(b): return False
    for (v1, u1), (v2, u2) in zip(a, b):
        if abs(v1 - v2) > 1e-9: return False
        if u1 and u2 and u1 != u2: return False   # "2 years" != "2 hours"
    return True

DATE_FULL = re.compile(
    r"(\d{1,2})\s+(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)[,\s]+(\d{4})"
    r"|(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\s+(\d{1,2})[,\s]+(\d{4})",
    re.IGNORECASE)
YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")

def parse_date(s):
    s = str(s)
    m = DATE_FULL.search(s)
    if m:
        if m.group(1): d, mo, y = m.group(1), norm(m.group(2))[:3], m.group(3)
        else:          mo, d, y = norm(m.group(4))[:3], m.group(5), m.group(6)
        return ("full", int(d), mo, int(y))
    ys = YEAR_RE.findall(s)
    if len(ys) == 1 and not re.search(r"\d{1,2}\s+\w+|\w+\s+\d{1,2}", s):
        return ("year", int(ys[0]))
    return None

def date_eq(pv, gv):
    a, b = parse_date(pv), parse_date(gv)
    if a is None or b is None: return False
    if a[0] != b[0]: return False        # full-vs-year-only precision must NOT match
    return a == b

def gia_typeaware_eq(pv, gv, atype):
    if gia_strict_eq(pv, gv): return True
    if pv is None or gv is None: return False
    t = (atype or "").upper()
    if t == "BOOLEAN": return False       # strict already canonicalized booleans
    if t == "NUMBER":  return num_eq(pv, gv)
    if t == "DATE":    return date_eq(pv, gv)
    return False                          # ENTITY/LOCATION/SET/SHORT_TEXT: strict only

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-hw), min(1, c+hw))

def prf(ps, gs):
    tp = len(ps & gs)
    p = tp/len(ps) if ps else (1.0 if not gs else 0.0)
    r = tp/len(gs) if gs else 1.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f

def main():
    for f in (CANDS, PREDS, GOLD):
        if not f.exists(): sys.exit(f"FATAL missing {f}")

    cands = {r["urbench_qid"]: r for r in load_jsonl(CANDS)}
    preds = {r["qid"]: r for r in load_jsonl(PREDS)}
    gold  = {r["urbench_qid"]: r for r in load_jsonl(GOLD)}

    # ---- 1-2: integrity ----
    assert len(cands) == len(preds) == len(gold) == 30, "not 30/30/30 rows"
    assert set(cands) == set(preds) == set(gold), "qid sets differ"
    for q in cands:
        cs = [s["id"] for s in cands[q]["typed_plan"]]
        gs = [s["id"] for s in gold[q]["typed_plan"]]
        assert cs == gs, f"step ids differ for {q}: cand {cs} vs gold {gs}"

    # ---- 3: disjointness ----
    a30 = {r["urbench_qid"] for r in load_jsonl(A30)}
    d50 = {l.strip() for l in open(DEV50, encoding="utf-8") if l.strip()}
    ev  = {r["urbench_qid"] for r in load_jsonl(MAP_F) if r.get("is_eval") is True}
    bq  = set(cands)
    assert not (bq & a30) and not (bq & d50) and not (bq & ev), "BLIND30 overlap detected!"

    hashes = {f.name: sha256(f) for f in (CANDS, PREDS, GOLD) }
    if REVW.exists(): hashes[REVW.name] = sha256(REVW)

    # ---- accumulators ----
    S = {}
    def add(k, v): S.setdefault(k, [0, 0]); S[k][0] += v; S[k][1] += 1
    corr = {k: 0 for k in ["entities","urdu_span","type","entity_ref","atype","evidence_ref","gia"]}
    row_corr, diffs, struct_rows = {}, [], []
    contradictions = []

    for q in sorted(cands):
        c, p, g = cands[q], preds[q], gold[q]
        rd = {"qid": q, "fields": []}
        nfix = 0

        rr = p.get("review_reasons", [])
        if any(any(r.startswith(sp) for sp in STRUCT_PREFIXES) for r in rr):
            struct_rows.append(q)

        # entities
        pe = {norm(e.get("canonical_title","")): e for e in p.get("question_entities", []) if e.get("canonical_title")}
        ge = {norm(e.get("canonical_title","")): e for e in g.get("question_entities", []) if e.get("canonical_title")}
        ep, er, ef = prf(set(pe), set(ge))
        add("ent_p", ep); add("ent_r", er); add("ent_f", ef)
        if set(pe) != set(ge):
            n = len(set(pe) ^ set(ge)); corr["entities"] += n; nfix += n
            rd["fields"].append({"field":"entities","pred":sorted(pe),"gold":sorted(ge)})
        for k in set(pe) & set(ge):
            ok = norm(pe[k].get("urdu_span")) == norm(ge[k].get("urdu_span"))
            add("span", 1 if ok else 0)
            if not ok:
                corr["urdu_span"] += 1; nfix += 1
                rd["fields"].append({"field":"urdu_span","entity":k,
                                     "pred":pe[k].get("urdu_span"),"gold":ge[k].get("urdu_span")})

        # steps
        ps = {s.get("id"): s for s in (p.get("typed_plan") or [])}
        for gs_ in g["typed_plan"]:
            sid = gs_["id"]; p_s = ps.get(sid, {})
            pairs = [("type","type",lambda a,b: norm(a)==norm(b)),
                     ("entity_ref","eref_norm",lambda a,b: norm(a)==norm(b)),
                     ("expected_answer_type","atype",lambda a,b: norm(a)==norm(b))]
            # strict entity_ref (raw, whitespace-stripped only)
            add("eref_strict", 1 if (str(p_s.get("entity_ref") or "").strip() ==
                                     str(gs_.get("entity_ref") or "").strip()) else 0)
            for field, key, eq in pairs:
                ok = eq(p_s.get(field), gs_.get(field))
                add(key, 1 if ok else 0)
                if not ok:
                    ck = {"type":"type","entity_ref":"entity_ref","expected_answer_type":"atype"}[field]
                    corr[ck] += 1; nfix += 1
                    rd["fields"].append({"field":field,"step":sid,
                                         "pred":p_s.get(field),"gold":gs_.get(field)})
            pv = set(p_s.get("evidence_ref") or []); gv = set(gs_.get("evidence_ref") or [])
            vp, vr, vf = prf(pv, gv)
            add("ev_p", vp); add("ev_r", vr); add("ev_f", vf)
            if pv != gv:
                corr["evidence_ref"] += 1; nfix += 1
                rd["fields"].append({"field":"evidence_ref","step":sid,
                                     "pred":sorted(pv),"gold":sorted(gv)})
            g_gia, p_gia = gs_.get("gold_intermediate_answer"), p_s.get("gold_intermediate_answer")
            st = gia_strict_eq(p_gia, g_gia)
            ta = st or gia_typeaware_eq(p_gia, g_gia, gs_.get("expected_answer_type"))
            add("gia_strict", 1 if st else 0); add("gia_type", 1 if ta else 0)
            if not st:
                corr["gia"] += 1; nfix += 1
                rd["fields"].append({"field":"gia","step":sid,"pred":p_gia,"gold":g_gia,
                                     "typeaware_equal":ta})

        # consistency audit: final gold step vs official answer
        last = g["typed_plan"][-1]
        gb = canon_bool(last.get("gold_intermediate_answer"))
        official = c.get("answer")
        if gb is None:
            contradictions.append((q, "UNCOMPARABLE", last.get("gold_intermediate_answer"), official))
        elif gb != bool(official):
            contradictions.append((q, "CONTRADICTION", gb, official))

        row_corr[q] = nfix
        if rd["fields"] or q in INSPECT_QIDS:
            if q in INSPECT_QIDS: rd["inspect"] = True
            diffs.append(rd)

    with open(OUT_D, "w", encoding="utf-8") as f:
        for d in diffs: f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- summary ----
    def pct(k):
        n = S[k][1]; return f"{100*S[k][0]/n:5.1f}% (n={n})" if n else "  n/a"
    zero = sum(1 for v in row_corr.values() if v == 0)
    some = 30 - zero
    lo, hi = wilson(some, 30)
    proj_lo, proj_mid, proj_hi = round(lo*1770), round(some/30*1770), round(hi*1770)

    L = []
    L.append("BLIND30 SCORE — correction-rate study (deterministic, no LLM judge)")
    L.append("=" * 68)
    L.append("CAVEAT: gold was created by EDITING predictions. Scores measure how")
    L.append("often a human corrected the automatic annotation — NOT independent-")
    L.append("annotator agreement. urdu_span is likely INFLATED (spans prefilled).")
    L.append("")
    L.append("file hashes (sha256/16): " + json.dumps(hashes))
    L.append(f"disjointness: AUDIT30/DEV50/eval458 overlap = 0/0/0 (asserted)")
    L.append(f"structural failures: {len(struct_rows)}/30  {struct_rows}")
    L.append("")
    L.append("ACCURACY (pred vs human-corrected gold)")
    for lab, k in [("entity precision","ent_p"),("entity recall","ent_r"),("entity F1","ent_f"),
                   ("urdu_span (matched, prefilled!)","span"),
                   ("step type","type"),("entity_ref STRICT","eref_strict"),
                   ("entity_ref normalized","eref_norm"),("answer type","atype"),
                   ("evidence precision","ev_p"),("evidence recall","ev_r"),("evidence F1","ev_f"),
                   ("GIA strict (PRIMARY)","gia_strict"),("GIA type-aware (SECONDARY)","gia_type")]:
        L.append(f"  {lab:34s}: {pct(k)}")
    L.append("")
    L.append("CORRECTION WORKLOAD")
    L.append(f"  rows with ZERO corrections : {zero}/30 = {100*zero/30:.1f}%")
    L.append(f"  rows with >=1 correction   : {some}/30 = {100*some/30:.1f}%")
    L.append(f"  corrections by field       : {corr}")
    L.append(f"  corrections per row        : {[row_corr[q] for q in sorted(row_corr)]}")
    L.append(f"  Wilson 95% CI (rows needing correction): {100*lo:.1f}%–{100*hi:.1f}%")
    L.append(f"  projection to 1,770 rows   : ~{proj_mid} rows (CI {proj_lo}–{proj_hi})")
    L.append("  WARNING: n=30 is small; projection is an operational estimate only.")
    L.append("")
    L.append("CONSISTENCY AUDIT (final gold step vs official answer; nothing modified)")
    if not contradictions: L.append("  none flagged")
    for q, kind, a, b in contradictions:
        mark = " <-- inspect-list" if q in INSPECT_QIDS else ""
        L.append(f"  {q}: {kind} gold_final={a!r} official={b!r}{mark}")
    L.append(f"  inspect-list rows always in diffs: {INSPECT_QIDS}")
    L.append("")
    L.append("ROUTING (descriptive; bands are OPERATIONAL PLANNING, not scientific thresholds)")
    L.append("  field            score   errors  prefilled  corrupts-target?  recommendation")
    def errs(k): return S[k][1] - S[k][0] if k in S else "?"
    L.append(f"  urdu_span        {pct('span')}   {errs('span')}      YES        YES (training span)   NEEDS AUTOMATIC VERIFICATION (prefill inflation)")
    L.append(f"  step type        {pct('type')}   {errs('type')}      no         mild                  SAFE CANDIDATE FOR AUTO-ACCEPT")
    L.append(f"  entities         F1 {pct('ent_f')}  -       partly     YES (core target)     NEEDS HUMAN REVIEW")
    L.append(f"  entity_ref       {pct('eref_norm')}   {errs('eref_norm')}      no         YES (core target)     NEEDS HUMAN REVIEW")
    L.append(f"  answer type      {pct('atype')}   {errs('atype')}      no         mild                  NEEDS AUTOMATIC VERIFICATION")
    L.append(f"  evidence_ref     F1 {pct('ev_f')}  -       no         YES                   NEEDS HUMAN REVIEW")
    L.append(f"  GIA              {pct('gia_strict')}   {errs('gia_strict')}      no         YES                   NEEDS HUMAN REVIEW")
    L.append("")
    L.append("PLANNING BANDS (post hoc, operational only): <=15% rows corrected =")
    L.append("potentially scalable | 15-30% = improve verification / narrow subset |")
    L.append(">30% = do NOT review 1,770 manually; redesign or simplify target.")
    L.append("")
    L.append("BLIND30 is now CONSUMED. Any redesign requires a fresh untouched sample.")

    summary = "\n".join(L)
    OUT_S.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\nsummary:", OUT_S, "\ndiffs:", OUT_D)

if __name__ == "__main__":
    main()