"""
efbpt_blind30_target_ablation.py — CPU-only, post-hoc target-schema ablation.

POST-HOC development analysis: reduced schemas chosen AFTER observing BLIND30.
No reduced-schema result is a blind validation. A revised schema+verifier must
be tested on a NEW untouched sample. Modifies nothing. No LLM judge.
"""
import json, re, math
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
CANDS = EF / "blind30_candidates.jsonl"
PREDS = EF / "blind30_predictions.jsonl"
GOLD  = EF / "blind30_gold.jsonl"
DIFFS = EF / "blind30_score_diffs.jsonl"
OUT_S = EF / "blind30_target_ablation_summary.txt"
OUT_D = EF / "blind30_target_ablation_details.jsonl"

SCHEMAS = {
    "A_FULL":            {"entities","urdu_span","type","entity_ref","atype","evidence_ref","gia"},
    "B_NO_GIA":          {"entities","urdu_span","type","entity_ref","atype","evidence_ref"},
    "C_CORE_ENTITY_PLAN":{"entities","urdu_span","type","entity_ref","atype"},
    "D_ENTITY_BRIDGE":   {"entities","urdu_span","entity_ref"},
    "E_PLAN_SKELETON":   {"type","entity_ref","atype"},
}
MECHANISM = {   # does schema still directly represent Urdu-span -> canonical-entity alignment?
    "A_FULL": "yes (plus extras beyond the mechanism)",
    "B_NO_GIA": "yes (plus evidence, beyond the mechanism)",
    "C_CORE_ENTITY_PLAN": "YES — exactly the stated mechanism + plan typing",
    "D_ENTITY_BRIDGE": "YES — minimal direct expression of the mechanism",
    "E_PLAN_SKELETON": "NO — drops urdu_span/entities; loses the bilingual alignment core",
}

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm(s):
    s = re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_", " ")).strip().casefold()
    return s.strip(" .,:;\"'")

BOOL_TRUE, BOOL_FALSE = {"yes","true","ہاں"}, {"no","false","نہیں"}
def canon_bool(s):
    n = norm(s)
    return True if n in BOOL_TRUE else False if n in BOOL_FALSE else None

def gia_strict_eq(pv, gv):
    if pv is None and gv is None: return True
    bp, bg = canon_bool(pv), canon_bool(gv)
    if bp is not None and bg is not None: return bp == bg
    return norm(pv) == norm(gv)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0,c-hw), min(1,c+hw))

def main():
    cands = {r["urbench_qid"]: r for r in load_jsonl(CANDS)}
    preds = {r["qid"]: r for r in load_jsonl(PREDS)}
    gold  = {r["urbench_qid"]: r for r in load_jsonl(GOLD)}
    _ = load_jsonl(DIFFS)   # read per requirement; recomputation below is authoritative

    # ---------- per-row, per-field error collection ----------
    row_fields = {}          # qid -> set of corrected field keys
    ent_tax = {"missing_only": [], "extra_only": [], "both": []}
    span_tax = {"extra_words": [], "truncated": [], "incorrect": [], "manual": []}
    eref_errors = []
    ev_tax = {"over": [], "under": [], "mixed": [], "empty_disagree": []}
    gia_tax = {"pred_null_gold_not": [], "pred_not_gold_null": [], "wrong_bool": [],
               "wrong_num_date": [], "gold_inside_pred": [], "text_differs": []}
    consistency = []
    details = []

    for q in sorted(cands):
        c, p, g = cands[q], preds[q], gold[q]
        fields = set(); det = {"qid": q, "errors": []}

        pe = {norm(e.get("canonical_title","")): e for e in p.get("question_entities",[]) if e.get("canonical_title")}
        ge = {norm(e.get("canonical_title","")): e for e in g.get("question_entities",[]) if e.get("canonical_title")}
        miss, extra = set(ge)-set(pe), set(pe)-set(ge)
        if miss or extra:
            fields.add("entities")
            key = "both" if (miss and extra) else "missing_only" if miss else "extra_only"
            ent_tax[key].append((q[:8], sorted(miss), sorted(extra)))
            det["errors"].append({"field":"entities","missing":sorted(miss),"extra":sorted(extra)})
        for k in set(pe) & set(ge):
            psp, gsp = pe[k].get("urdu_span",""), ge[k].get("urdu_span","")
            if norm(psp) != norm(gsp):
                fields.add("urdu_span")
                np_, ng_ = norm(psp), norm(gsp)
                if ng_ and ng_ in np_ and len(np_) > len(ng_):
                    span_tax["extra_words"].append((q[:8], k, psp, gsp))
                elif np_ and np_ in ng_ and len(np_) < len(ng_):
                    span_tax["truncated"].append((q[:8], k, psp, gsp))
                elif np_ and ng_:
                    span_tax["incorrect"].append((q[:8], k, psp, gsp))
                else:
                    span_tax["manual"].append((q[:8], k, psp, gsp))
                det["errors"].append({"field":"urdu_span","entity":k,"pred":psp,"gold":gsp})

        ps = {s.get("id"): s for s in (p.get("typed_plan") or [])}
        gsteps = g["typed_plan"]
        for gs in gsteps:
            sid = gs["id"]; p_s = ps.get(sid, {})
            if norm(p_s.get("type")) != norm(gs.get("type")):
                fields.add("type")
                det["errors"].append({"field":"type","step":sid,"pred":p_s.get("type"),"gold":gs.get("type")})
            pr, gr = p_s.get("entity_ref") or "", gs.get("entity_ref") or ""
            if norm(pr) != norm(gr):
                fields.add("entity_ref")
                if bool(pr) != bool(gr): kind = "empty_vs_nonempty"
                else: kind = "different_entity (manual: subject-vs-answer or wrong)"
                eref_errors.append((q, sid, pr, gr, kind))
                det["errors"].append({"field":"entity_ref","step":sid,"pred":pr,"gold":gr,"kind":kind})
            elif str(pr).strip() != str(gr).strip():
                # normalization-only: strict differs, normalized equal — NOT a correction
                eref_errors.append((q, sid, pr, gr, "normalization_only (no correction)"))
            if norm(p_s.get("expected_answer_type")) != norm(gs.get("expected_answer_type")):
                fields.add("atype")
                det["errors"].append({"field":"atype","step":sid,
                                      "pred":p_s.get("expected_answer_type"),"gold":gs.get("expected_answer_type")})
            pv, gv = set(p_s.get("evidence_ref") or []), set(gs.get("evidence_ref") or [])
            if pv != gv:
                fields.add("evidence_ref")
                if bool(pv) != bool(gv): key = "empty_disagree"
                elif pv > gv: key = "over"
                elif pv < gv: key = "under"
                else: key = "mixed"
                ev_tax[key].append((q[:8], sid))
                det["errors"].append({"field":"evidence_ref","step":sid,"pred":sorted(pv),"gold":sorted(gv),"kind":key})
            pg, gg = p_s.get("gold_intermediate_answer"), gs.get("gold_intermediate_answer")
            if not gia_strict_eq(pg, gg):
                fields.add("gia")
                at = (gs.get("expected_answer_type") or "").upper()
                if pg in (None,"null") and gg not in (None,"null"): key="pred_null_gold_not"
                elif gg in (None,"null") and pg not in (None,"null"): key="pred_not_gold_null"
                elif at=="BOOLEAN" and canon_bool(pg) is not None and canon_bool(gg) is not None: key="wrong_bool"
                elif at in ("NUMBER","DATE"): key="wrong_num_date"
                elif pg and gg and norm(gg) and norm(gg) in norm(pg): key="gold_inside_pred"
                else: key="text_differs"
                gia_tax[key].append((q[:8], sid, str(pg)[:60], str(gg)[:60]))
                det["errors"].append({"field":"gia","step":sid,"pred":pg,"gold":gg,"kind":key})

        # consistency: final gold step vs official answer, with rule-based reading
        last = gsteps[-1]
        gb = canon_bool(last.get("gold_intermediate_answer"))
        official = bool(c.get("answer"))
        if gb is None:
            tag = ("final step GIA not boolean/absent -> likely 'final step not intended "
                   "to reproduce official answer' OR annotation gap — manual check")
            consistency.append((q, None, official, tag))
        elif gb != official:
            decomp_len = len(gsteps)
            if decomp_len < 2:
                tag = "malformed/short decomposition — uncertain"
            elif (last.get("type") or "").upper() != "REASON":
                tag = "final step is not a REASON step -> plan may not target the official answer"
            else:
                tag = "candidate literal annotation error OR question-level disagreement — manual check"
            consistency.append((q, gb, official, tag))

        row_fields[q] = fields
        details.append(det)

    with open(OUT_D, "w", encoding="utf-8") as f:
        for d in details: f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---------- schema ablation ----------
    L = []
    L.append("BLIND30 TARGET-SCHEMA ABLATION — POST-HOC (not blind validation)")
    L.append("=" * 68)
    L.append("CAVEAT: schemas selected AFTER seeing BLIND30. Any chosen schema must")
    L.append("be re-validated on a fresh untouched sample before scaling.")
    L.append("")
    best = None
    for name, incl in SCHEMAS.items():
        bad = {q: sorted(row_fields[q] & incl) for q in row_fields if row_fields[q] & incl}
        k = len(bad); zero = 30 - k
        lo, hi = wilson(k, 30)
        proj = (round(lo*1770), round(k/30*1770), round(hi*1770))
        L.append(f"SCHEMA {name}  (fields: {sorted(incl)})")
        L.append(f"  zero-correction rows : {zero}/30 = {100*zero/30:.1f}%")
        L.append(f"  rows needing fix     : {k}/30 = {100*k/30:.1f}%   Wilson95: {100*lo:.1f}%–{100*hi:.1f}%")
        L.append(f"  projection /1770     : ~{proj[1]} (CI {proj[0]}–{proj[2]})  [operational estimate, n=30]")
        L.append(f"  mechanism alignment  : {MECHANISM[name]}")
        L.append(f"  rows + causing fields:")
        for q, fl in sorted(bad.items()): L.append(f"    {q[:8]}: {fl}")
        L.append("")
        if name in ("C_CORE_ENTITY_PLAN","D_ENTITY_BRIDGE"):
            if best is None or k < best[1]: best = (name, k)

    # ---------- taxonomies ----------
    L.append("ENTITY ERROR TAXONOMY")
    for k, v in ent_tax.items(): L.append(f"  {k}: {len(v)}  {v}")
    L.append("")
    L.append("URDU-SPAN TAXONOMY (deterministic containment rules; else manual)")
    for k, v in span_tax.items(): L.append(f"  {k}: {len(v)}  {v}")
    L.append("")
    L.append("ENTITY_REF ERRORS (all, in full)")
    for e in eref_errors: L.append(f"  {e}")
    L.append("")
    L.append("EVIDENCE TAXONOMY (per step)")
    for k, v in ev_tax.items(): L.append(f"  {k}: {len(v)}  qids/steps: {v}")
    L.append("")
    L.append("GIA TAXONOMY")
    for k, v in gia_tax.items():
        L.append(f"  {k}: {len(v)}")
        for e in v[:4]: L.append(f"     e.g. {e}")
    L.append("")
    L.append("CONSISTENCY FLAGS (final gold step vs official answer; nothing changed)")
    for q, gb, off, tag in consistency:
        L.append(f"  {q}: gold_final={gb!r} official={off!r}")
        L.append(f"     reading: {tag}")
    L.append("")

    # ---------- provisional recommendation ----------
    kA = len([q for q in row_fields if row_fields[q] & SCHEMAS["A_FULL"]])
    kC = len([q for q in row_fields if row_fields[q] & SCHEMAS["C_CORE_ENTITY_PLAN"]])
    kD = len([q for q in row_fields if row_fields[q] & SCHEMAS["D_ENTITY_BRIDGE"]])
    L.append("PROVISIONAL RECOMMENDATION (computed; final decision is the human's)")
    L.append(f"  A_FULL correction rate {100*kA/30:.0f}% -> exceeds 30% band: do not scale as-is.")
    def band(k): 
        r = k/30
        return "potentially scalable (<=15%)" if r <= .15 else "improve/narrow (15-30%)" if r <= .30 else "redesign (>30%)"
    L.append(f"  C_CORE_ENTITY_PLAN: {100*kC/30:.0f}% -> {band(kC)}")
    L.append(f"  D_ENTITY_BRIDGE   : {100*kD/30:.0f}% -> {band(kD)}")
    pick = "USE CORE_ENTITY_PLAN" if kC/30 <= .30 else ("USE ENTITY_BRIDGE ONLY" if kD/30 <= .30 else "STOP EFBPT DATA CONSTRUCTION")
    L.append(f"  => computed pick: {pick}")
    L.append("  Rationale factors: evidence_ref/GIA carry most workload and are NOT the")
    L.append("  stated mechanism; dropping them keeps Urdu-span->entity alignment intact.")
    L.append("  Sept-30 deadline favors the smallest schema that preserves the mechanism.")
    L.append("  E_PLAN_SKELETON is excluded: it abandons the bilingual core.")
    L.append("  Chosen schema requires a FRESH validation sample before any scaling.")

    summary = "\n".join(L)
    OUT_S.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\nsummary:", OUT_S, "\ndetails:", OUT_D)

if __name__ == "__main__":
    main()