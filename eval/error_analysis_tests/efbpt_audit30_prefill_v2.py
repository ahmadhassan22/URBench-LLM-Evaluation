"""
EFBPT AUDIT30 prefill v2 (CPU-only). Fixes over v1:

  FIX 1 (entity semantics): 'entities' split into
     - question_entities: entities NAMED IN THE QUESTION (seeded from gold term;
       manual additions expected). These get urdu_span + hard negatives. This is
       what EFBPT trains: Urdu span -> canonical English title.
     - evidence_pages: page titles referenced by official evidence (retrieval
       targets, reference only — NO spans, NO negatives).
  FIX 2 (typing rule): dependency != REASON. A step depending on #k can still be
     RETRIEVE (bridge retrieval: "Where was #1 born?"). type_suggested is REASON
     only for compare/boolean/arithmetic patterns; it remains a HINT — the manual
     'type' field decides.
  FIX 3 (title extraction): proper json.loads per line (v1 byte-parser may have
     dropped titles, e.g. Bucharest). Use --rebuild_titles once to regenerate the
     cache; in_local_title_space recomputed from the fixed cache.

Rules unchanged: TRAIN-side only, seed 42, one-annotator evidence candidates,
manual verification mandatory, no GPU / eval458 / DEV50.
"""
import json, re, random, argparse, time
from pathlib import Path
from collections import defaultdict

BASE   = Path("/mnt/home/user41/URBench")
OFF    = BASE / "data/strategyqa_official"
MAP_F  = OFF / "strategyqa_official_mapped_urbench_qid.jsonl"
DEVQ_F = OFF / "dev50_seed42_qids.txt"
META_F = BASE / "rag/index/wikipedia_full_meta.jsonl"
OUT    = OFF / "efbpt"
TITLE_CACHE = OUT / "title_space_cache.txt"

N_SAMPLE, SEED, MAX_SIBLINGS = 30, 42, 8
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
    """HINT only. Dependency != REASON: bridge lookups stay RETRIEVE."""
    return "REASON" if REASON_PAT.search(step_text.strip()) else "RETRIEVE"

def norm_t(t): return re.sub(r"\s+", " ", t).strip().casefold()

def build_titles(rebuild=False):
    if TITLE_CACHE.exists() and not rebuild:
        print(f"[titles] loading cache {TITLE_CACHE}", flush=True)
        with open(TITLE_CACHE, encoding="utf-8") as f:
            return set(l.rstrip("\n") for l in f)
    print(f"[titles] robust json scan of {META_F} (one-time, ~10-15 min)", flush=True)
    titles = set(); t0 = time.time(); n = bad = 0
    with open(META_F, encoding="utf-8") as f:
        for line in f:
            n += 1
            try:
                titles.add(json.loads(line)["title"])
            except Exception:
                bad += 1
            if n % 4_000_000 == 0:
                print(f"[titles]   {n:,} lines, {len(titles):,} unique, "
                      f"{bad} bad ({time.time()-t0:.0f}s)", flush=True)
    print(f"[titles] done: {len(titles):,} unique, {bad} unparseable "
          f"({time.time()-t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(TITLE_CACHE, "w", encoding="utf-8") as f:
        for t in sorted(titles): f.write(t+"\n")
    return titles

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild_titles", action="store_true")
    ap.add_argument("--skip_title_mining", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(MAP_F)
    dev_qids = set(l.strip() for l in open(DEVQ_F) if l.strip())
    train_pool = [r for r in rows if not r["is_eval"] and r["urbench_qid"] not in dev_qids]
    eligible   = [r for r in train_pool if r.get("has_paragraph_evidence")]
    print(f"TRAIN pool = {len(train_pool)}   N_eligible = {len(eligible)} "
          f"({100*len(eligible)/max(len(train_pool),1):.1f}%)")

    rng = random.Random(SEED)
    audit = rng.sample(eligible, N_SAMPLE)
    audit_qids = [r["urbench_qid"] for r in audit]
    assert len(set(audit_qids)) == N_SAMPLE and not (set(audit_qids) & dev_qids)
    with open(OUT/"audit30_qids.txt","w") as f:
        for q in audit_qids: f.write(q+"\n")

    titles = set() if args.skip_title_mining else build_titles(args.rebuild_titles)
    title_index = defaultdict(set)
    for t in titles:
        tk = norm_t(t).split()
        if tk: title_index[tk[0]].add(t)

    recs = []
    for r in audit:
        decomp = r.get("official_decomposition") or []
        ev_per_step = step_evidence_candidates(r.get("official_evidence"), len(decomp))

        # evidence pages: reference-only retrieval targets
        ev_pages = []
        for s in ev_per_step:
            for pid in s:
                t = para_id_to_title(pid)
                if t not in ev_pages: ev_pages.append(t)

        # question entities: seeded from TERM only; audit adds the rest by hand
        q_entities = []
        term = (r.get("term") or "").strip()
        seeds = [term] if term else []
        for t in seeds:
            q_entities.append({
                "urdu_span": "",                       # MANUAL
                "canonical_title": t,
                "wiki_id": None,
                "source": "term",
                "in_local_title_space": (t in titles) if titles else None,
                "hard_negatives": [
                    {"title": s, "source": "disambiguation_sibling",
                     "verified_incorrect": None}       # MANUAL true/false
                    for s in find_siblings(t, title_index)],
            })
        # empty slot template for manual additions
        q_entities.append({
            "urdu_span": "", "canonical_title": "", "wiki_id": None,
            "source": "manual", "in_local_title_space": None, "hard_negatives": []})

        typed_plan = []
        for i,(step,ev) in enumerate(zip(decomp, ev_per_step),1):
            typed_plan.append({
                "id": i,
                "type_suggested": suggest_type(step),   # HINT; dependency != REASON
                "type": "",                             # MANUAL
                "question_en": step,
                "entity_ref": "",                       # MANUAL
                "depends_on": sorted(set(int(m) for m in re.findall(r"#(\d+)", step))),
                "expected_answer_type": "",             # MANUAL
                "evidence_candidates": ev,              # >=1 annotator
                "evidence_ref": [],                     # MANUAL verified subset
                "gold_intermediate_answer": None})      # MANUAL

        recs.append({
            "urbench_qid": r["urbench_qid"], "official_qid": r["official_qid"],
            "question_ur": r.get("question_ur"), "question_en": r.get("question_en"),
            "term": term, "answer": r.get("answer"),
            "question_entities": q_entities,
            "evidence_pages": [{"title": t,
                                "in_local_title_space": (t in titles) if titles else None}
                               for t in ev_pages],
            "typed_plan": typed_plan,
            "flags": {"semantic_mismatch": None, "entity_unresolvable": None,
                      "plan_untypeable": None, "evidence_unlinked": None},
            "verdict": "", "notes": ""})

    with open(OUT/"audit30_candidates.jsonl","w",encoding="utf-8") as f:
        for rec in recs: f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    # coverage stat for evidence pages (relevant to the Bucharest question)
    if titles:
        evp = [p for rec in recs for p in rec["evidence_pages"]]
        miss = [p["title"] for p in evp if not p["in_local_title_space"]]
        print(f"evidence pages: {len(evp)} total, {len(miss)} NOT in local title space")
        if miss: print("  missing examples:", miss[:8])

    with open(OUT/"audit30_manual_report_template.md","w",encoding="utf-8") as f:
        f.write("# AUDIT30 Manual Report\n\n")
        f.write(f"- N_eligible: {len(eligible)} / {len(train_pool)}\n\n")
        f.write("Typing rule: dependency != REASON. RETRIEVE = looks up an external fact "
                "(incl. bridge lookups about #k results). REASON = compares/computes/decides.\n\n")
        f.write("| # | qid | verdict | flags | notes |\n|---|---|---|---|---|\n")
        for i,q in enumerate(audit_qids,1): f.write(f"| {i} | {q} |  |  |  |\n")
        f.write("\n| Gate | Threshold | Measured | Pass? |\n|---|---|---|---|\n")
        f.write("| G-A | CLEAN+USABLE >= 24/30 |  |  |\n")
        f.write("| G-B | semantic_mismatch <= 3/30 |  |  |\n")
        f.write("| G-C | >=90% rows: all question entities resolved (non-manual source or trivially found) |  |  |\n")
        f.write("| G-D | >=90% entities with >=2 verified negatives |  |  |\n")
        f.write("| G-E | retained rows: 100% RETRIEVE steps verified-evidence-linked |  |  |\n")

    print(f"wrote {OUT/'audit30_candidates.jsonl'} ({len(recs)} rows) + qids + template")

if __name__ == "__main__":
    main()