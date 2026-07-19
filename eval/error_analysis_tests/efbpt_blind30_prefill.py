"""
efbpt_blind30_prefill.py — CPU-only prefill for the frozen BLIND30 sample.

Plain words: takes the 30 frozen blind rows and builds candidate rows with the
EXACT same structure as audit30_candidates.jsonl (evidence pages, typed plan,
hard negatives). Differences from audit30 prefill v2:
  - NO sampling: reads the frozen blind30_rows.jsonl as-is.
  - Title cache MUST exist; this script refuses to rebuild it (no 15-min scan).
  - Refuses to overwrite existing output.
Runs no model. Touches no AUDIT30 or pilot file.
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

BASE   = Path("/mnt/home/user41/URBench")
OFF    = BASE / "data/strategyqa_official"
EF     = OFF / "efbpt"

IN_ROWS     = EF / "blind30_rows.jsonl"
IN_QIDS     = EF / "blind30_qids.txt"
TITLE_CACHE = EF / "title_space_cache.txt"
OUT_CANDS   = EF / "blind30_candidates.jsonl"

MAX_SIBLINGS = 8
MARKERS = {"operation", "no_evidence"}

REASON_PAT = re.compile(
    r"^(is|are|was|were|does|do|did|can|could|would|will)\b.*#\d"
    r"|#\d.*\b(multiplied|divided|plus|minus|times|less than|greater than|more than"
    r"|equal|same as|part of|included|within|before|after|between)\b"
    r"|\b(what is)\s+#\d", re.IGNORECASE)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def para_id_to_title(pid):
    m = re.match(r"^(.*)-(\d+)$", pid)
    return (m.group(1) if m else pid).replace("_", " ").strip()

def step_evidence_candidates(evidence, n_steps):
    per_step = [[] for _ in range(n_steps)]
    if isinstance(evidence, list):
        for annotator in evidence:
            if not isinstance(annotator, list): continue
            for si, step_ev in enumerate(annotator):
                if si >= n_steps: continue
                ids = []
                def walk(x):
                    if isinstance(x, str):
                        if x not in MARKERS: ids.append(x)
                    elif isinstance(x, list):
                        for y in x: walk(y)
                walk(step_ev)
                per_step[si].extend(ids)
    return [sorted(set(s)) for s in per_step]

def suggest_type(step_text):
    return "REASON" if REASON_PAT.search(step_text.strip()) else "RETRIEVE"

def norm_t(t): return re.sub(r"\s+", " ", t).strip().casefold()

def find_siblings(entity, title_index):
    ent_norm = norm_t(entity); toks = ent_norm.split()
    if not toks: return []
    scored = []
    for t in title_index.get(toks[0], ()):
        tn = norm_t(t)
        if tn == ent_norm: continue
        ov = len(set(toks) & set(tn.split()))
        if ov: scored.append((-ov, len(t), t))
    scored.sort()
    return [t for _,__,t in scored[:MAX_SIBLINGS]]

def main():
    # refuse overwrite
    if OUT_CANDS.exists():
        existing = load_jsonl(OUT_CANDS)
        print(f"OUTPUT EXISTS: {OUT_CANDS} ({len(existing)} rows). Refusing to overwrite. Exiting.")
        sys.exit(0)

    # title cache must exist — no rebuild here
    if not TITLE_CACHE.exists():
        print(f"FATAL: title cache missing at {TITLE_CACHE}. Refusing to rebuild "
              "(15-min scan belongs to the audit30 prefill script). STOP and investigate.")
        sys.exit(1)

    rows = load_jsonl(IN_ROWS)
    qids = [l.strip() for l in open(IN_QIDS, encoding="utf-8") if l.strip()]
    assert len(rows) == 30, f"blind30_rows has {len(rows)} rows, expected 30"
    assert [r["urbench_qid"] for r in rows] == sorted(qids), "rows/qids mismatch or order changed"

    with open(TITLE_CACHE, encoding="utf-8") as f:
        titles = set(l.rstrip("\n") for l in f)
    print(f"[titles] cache loaded: {len(titles):,} titles")
    title_index = defaultdict(set)
    for t in titles:
        tk = norm_t(t).split()
        if tk: title_index[tk[0]].add(t)

    recs, no_ev_rows = [], []
    for r in rows:
        decomp = r.get("official_decomposition") or []
        ev_per_step = step_evidence_candidates(r.get("official_evidence"), len(decomp))
        if not any(ev_per_step):
            no_ev_rows.append(r["urbench_qid"])

        ev_pages = []
        for s in ev_per_step:
            for pid in s:
                t = para_id_to_title(pid)
                if t not in ev_pages: ev_pages.append(t)

        q_entities = []
        term = (r.get("term") or "").strip()
        for t in ([term] if term else []):
            q_entities.append({
                "urdu_span": "", "canonical_title": t, "wiki_id": None,
                "source": "term",
                "in_local_title_space": t in titles,
                "hard_negatives": [
                    {"title": s, "source": "disambiguation_sibling",
                     "verified_incorrect": None}
                    for s in find_siblings(t, title_index)],
            })
        q_entities.append({
            "urdu_span": "", "canonical_title": "", "wiki_id": None,
            "source": "manual", "in_local_title_space": None, "hard_negatives": []})

        typed_plan = []
        for i, (step, ev) in enumerate(zip(decomp, ev_per_step), 1):
            typed_plan.append({
                "id": i,
                "type_suggested": suggest_type(step),
                "type": "",
                "question_en": step,
                "entity_ref": "",
                "depends_on": sorted(set(int(m) for m in re.findall(r"#(\d+)", step))),
                "expected_answer_type": "",
                "evidence_candidates": ev,
                "evidence_ref": [],
                "gold_intermediate_answer": None})

        recs.append({
            "urbench_qid": r["urbench_qid"], "official_qid": r["official_qid"],
            "question_ur": r.get("question_ur"), "question_en": r.get("question_en"),
            "term": term, "answer": r.get("answer"),
            "question_entities": q_entities,
            "evidence_pages": [{"title": t, "in_local_title_space": t in titles}
                               for t in ev_pages],
            "typed_plan": typed_plan,
            "flags": {"semantic_mismatch": None, "entity_unresolvable": None,
                      "plan_untypeable": None, "evidence_unlinked": None},
            "verdict": "", "notes": ""})

    with open(OUT_CANDS, "w", encoding="utf-8") as f:
        for rec in recs: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    evp = [p for rec in recs for p in rec["evidence_pages"]]
    miss = [p["title"] for p in evp if not p["in_local_title_space"]]
    print(f"wrote {OUT_CANDS} ({len(recs)} rows)")
    print(f"evidence pages: {len(evp)} total, {len(miss)} NOT in local title space")
    if miss: print("  missing examples:", miss[:8])
    if no_ev_rows:
        print(f"!! {len(no_ev_rows)} rows have ZERO evidence candidates: {no_ev_rows}")
        print("   (blind30 pool was Stage-1 eligible only — has_paragraph_evidence "
              "was NOT a sampling filter, unlike audit30. These rows will still be "
              "annotated; expect them to route to review.)")

if __name__ == "__main__":
    main()