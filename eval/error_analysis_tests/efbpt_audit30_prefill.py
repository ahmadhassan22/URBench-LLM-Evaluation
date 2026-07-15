"""
EFBPT AUDIT30 prefill (CPU-only, no GPU, no model calls, no eval458, no DEV50).

Purpose: mechanical prefill for the 30-row data-audit probe. Computes N_eligible,
samples AUDIT30 (seed 42) from the evidence-eligible TRAIN pool, prefills every
field derivable without judgment, and emits the manual-audit template.

Rules implemented (agreed):
  - TRAIN-side only: non-eval, non-DEV50, has_paragraph_evidence == true.
  - Evidence candidates: a step is a candidate if AT LEAST ONE annotator gave
    paragraph evidence (single-annotator reliability ~93% -> manual verification
    is mandatory before final acceptance; this script only prefills candidates).
  - Hard-negative siblings are mined OFFLINE from the local Wikipedia title space
    (cluster has no Wikidata/Wikipedia network access). All negatives start
    verified_incorrect = null -> must be manually set true/false.
  - wiki_id left null unless --try_wikidata (network) succeeds; fill-rate reported.

Outputs (data/strategyqa_official/efbpt/):
  audit30_qids.txt
  audit30_candidates.jsonl
  audit30_manual_report_template.md
  title_space_cache.txt (cached unique titles, reused on re-runs)
"""
import json, re, random, argparse, time, sys
from pathlib import Path
from collections import defaultdict

BASE   = Path("/mnt/home/user41/URBench")
OFF    = BASE / "data/strategyqa_official"
MAP_F  = OFF / "strategyqa_official_mapped_urbench_qid.jsonl"
DEVQ_F = OFF / "dev50_seed42_qids.txt"
META_F = BASE / "rag/index/wikipedia_full_meta.jsonl"
OUT    = OFF / "efbpt"
TITLE_CACHE = OUT / "title_space_cache.txt"

N_SAMPLE = 30
SEED     = 42
MAX_SIBLINGS = 8

MARKERS = {"operation", "no_evidence"}

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def para_id_to_title(pid):
    """'Vellore_Fort-1' -> 'Vellore Fort'. Strip only a trailing -<int>."""
    m = re.match(r"^(.*)-(\d+)$", pid)
    t = m.group(1) if m else pid
    return t.replace("_", " ").strip()

def step_evidence_candidates(evidence, n_steps):
    """Per decomposition step, paragraph IDs from ANY annotator (candidate rule).
    evidence shape: [annotator][step][...nested lists of pids or markers...]"""
    per_step = [[] for _ in range(n_steps)]
    if not isinstance(evidence, list):
        return per_step
    for annotator in evidence:
        if not isinstance(annotator, list):
            continue
        for si, step_ev in enumerate(annotator):
            if si >= n_steps:
                continue
            ids = []
            def walk(x):
                if isinstance(x, str):
                    if x not in MARKERS:
                        ids.append(x)
                elif isinstance(x, list):
                    for y in x: walk(y)
            walk(step_ev)
            per_step[si].extend(ids)
    return [sorted(set(s)) for s in per_step]

def suggest_type(step_text):
    """Heuristic ONLY — manual typing decides. REASON if references earlier steps."""
    return "REASON" if "#" in step_text else "RETRIEVE"

def norm_t(t):
    return re.sub(r"\s+", " ", t).strip().casefold()

def build_or_load_title_space(skip=False):
    if TITLE_CACHE.exists():
        print(f"[titles] loading cache {TITLE_CACHE}", flush=True)
        with open(TITLE_CACHE, encoding="utf-8") as f:
            return set(l.rstrip("\n") for l in f)
    if skip:
        print("[titles] --skip_title_mining and no cache: sibling mining disabled")
        return set()
    print(f"[titles] one-time scan of {META_F} (~26G, 5-10 min, CPU/IO only)", flush=True)
    titles = set()
    t0 = time.time(); n = 0
    with open(META_F, "rb") as f:
        for line in f:
            n += 1
            # cheap parse: {"title": "....", "text": ...
            i = line.find(b'"title": "')
            if i == -1:
                i = line.find(b'"title":"')
                j = i + 9 if i != -1 else -1
            else:
                j = i + 10
            if j != -1:
                k = line.find(b'", "', j)
                if k == -1:
                    k = line.find(b'","', j)
                if k != -1:
                    try:
                        titles.add(line[j:k].decode("utf-8"))
                    except Exception:
                        pass
            if n % 4_000_000 == 0:
                print(f"[titles]   {n:,} lines, {len(titles):,} unique ({time.time()-t0:.0f}s)", flush=True)
    print(f"[titles] done: {len(titles):,} unique titles ({time.time()-t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(TITLE_CACHE, "w", encoding="utf-8") as f:
        for t in sorted(titles):
            f.write(t + "\n")
    return titles

def find_siblings(entity, title_space, title_index):
    """Confusable-title candidates: share the entity's head token, or containment,
    excluding the entity itself. Sorted by token overlap desc, then length."""
    ent_norm = norm_t(entity)
    ent_toks = ent_norm.split()
    if not ent_toks:
        return []
    head = ent_toks[0]
    cands = title_index.get(head, set())
    scored = []
    for t in cands:
        tn = norm_t(t)
        if tn == ent_norm:
            continue
        toks = set(tn.split())
        overlap = len(set(ent_toks) & toks)
        if overlap == 0:
            continue
        scored.append((-overlap, len(t), t))
    scored.sort()
    return [t for _,__,t in scored[:MAX_SIBLINGS]]

def try_wikidata(title):
    try:
        import urllib.request, urllib.parse
        url = ("https://www.wikidata.org/w/api.php?action=wbsearchentities&search="
               + urllib.parse.quote(title) + "&language=en&format=json&limit=1")
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.load(r)
        hits = d.get("search", [])
        return hits[0]["id"] if hits else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_title_mining", action="store_true")
    ap.add_argument("--try_wikidata", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(MAP_F)
    dev_qids = set(l.strip() for l in open(DEVQ_F) if l.strip())

    # ---- pool + N_eligible ----
    train_pool = [r for r in rows if not r["is_eval"] and r["urbench_qid"] not in dev_qids]
    eligible   = [r for r in train_pool if r.get("has_paragraph_evidence")]
    print(f"TRAIN pool (non-eval, non-DEV50) = {len(train_pool)}")
    print(f"N_eligible (paragraph evidence)  = {len(eligible)} "
          f"({100*len(eligible)/max(len(train_pool),1):.1f}% of TRAIN pool)")
    print(f"NOT eligible                      = {len(train_pool)-len(eligible)}")

    rng = random.Random(SEED)
    audit = rng.sample(eligible, N_SAMPLE)
    audit_qids = [r["urbench_qid"] for r in audit]
    assert len(set(audit_qids)) == N_SAMPLE
    assert not (set(audit_qids) & dev_qids)
    with open(OUT/"audit30_qids.txt", "w") as f:
        for q in audit_qids: f.write(q+"\n")

    # ---- title space for offline sibling mining ----
    title_space = build_or_load_title_space(skip=args.skip_title_mining)
    title_index = defaultdict(set)
    for t in title_space:
        toks = norm_t(t).split()
        if toks:
            title_index[toks[0]].add(t)

    # ---- prefill records ----
    qid_fill = 0; ent_total = 0
    recs = []
    for r in audit:
        decomp = r.get("official_decomposition") or []
        ev_per_step = step_evidence_candidates(r.get("official_evidence"), len(decomp))

        # entity candidates: unique page titles from evidence + gold term
        titles = []
        for s in ev_per_step:
            for pid in s:
                t = para_id_to_title(pid)
                if t not in titles:
                    titles.append(t)
        term = (r.get("term") or "").strip()
        if term and term not in titles:
            titles.append(term)   # term may be Urdu here; manual audit resolves

        entities = []
        for t in titles:
            ent_total += 1
            wid = try_wikidata(t) if args.try_wikidata else None
            if wid: qid_fill += 1
            sibs = find_siblings(t, title_space, title_index) if title_space else []
            entities.append({
                "urdu_span": "",                      # MANUAL
                "canonical_title": t,
                "wiki_id": wid,
                "source": "term" if t == term else "evidence_title",
                "in_local_title_space": t in title_space,
                "hard_negatives": [
                    {"title": s, "source": "disambiguation_sibling",
                     "verified_incorrect": None}       # MANUAL true/false
                    for s in sibs
                ],
            })

        typed_plan = []
        for i, (step, ev) in enumerate(zip(decomp, ev_per_step), 1):
            typed_plan.append({
                "id": i,
                "type_suggested": suggest_type(step),  # heuristic
                "type": "",                            # MANUAL: RETRIEVE|REASON
                "question_en": step,
                "entity_ref": "",                      # MANUAL
                "depends_on": sorted(set(int(m) for m in re.findall(r"#(\d+)", step))),
                "expected_answer_type": "",            # MANUAL: BOOLEAN|ENTITY|LOCATION|DATE|NUMBER|SET|SHORT_TEXT
                "evidence_candidates": ev,             # >=1 annotator (candidate rule)
                "evidence_ref": [],                    # MANUAL: verified subset
                "gold_intermediate_answer": None,      # MANUAL where stated
            })

        recs.append({
            "urbench_qid": r["urbench_qid"],
            "official_qid": r["official_qid"],
            "question_ur": r.get("question_ur"),
            "question_en": r.get("question_en"),
            "term": term,
            "answer": r.get("answer"),
            "entities": entities,
            "typed_plan": typed_plan,
            "flags": {"semantic_mismatch": None, "entity_unresolvable": None,
                      "plan_untypeable": None, "evidence_unlinked": None},
            "verdict": "",     # MANUAL: CLEAN | USABLE-WITH-EDIT | BROKEN
            "notes": "",
        })

    with open(OUT/"audit30_candidates.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    # ---- manual report template ----
    with open(OUT/"audit30_manual_report_template.md", "w", encoding="utf-8") as f:
        f.write("# AUDIT30 Manual Report (fill after hand audit)\n\n")
        f.write(f"- N_eligible: {len(eligible)} / TRAIN pool {len(train_pool)}\n")
        f.write(f"- wiki_id fill-rate (prefill): {qid_fill}/{ent_total}\n\n")
        f.write("## Per-row verdicts\n\n")
        f.write("| # | qid | verdict | flags | notes |\n|---|---|---|---|---|\n")
        for i,q in enumerate(audit_qids,1):
            f.write(f"| {i} | {q} |  |  |  |\n")
        f.write("\n## Gate summary\n\n")
        f.write("| Gate | Threshold | Measured | Pass? |\n|---|---|---|---|\n")
        f.write("| G-A screening | CLEAN+USABLE >= 24/30 |  |  |\n")
        f.write("| G-B feasibility | semantic_mismatch <= 3/30 |  |  |\n")
        f.write("| G-C complete-row coverage | >=90% rows all entities from evidence/term |  |  |\n")
        f.write("| G-D negative coverage | >=90% entities with >=2 verified negatives |  |  |\n")
        f.write("| G-E (retained rows) | 100% RETRIEVE steps verified-evidence-linked |  |  |\n")

    print(f"\nwrote {OUT/'audit30_qids.txt'}")
    print(f"wrote {OUT/'audit30_candidates.jsonl'} ({len(recs)} rows)")
    print(f"wrote {OUT/'audit30_manual_report_template.md'}")
    print(f"wiki_id fill-rate: {qid_fill}/{ent_total}"
          + ("" if args.try_wikidata else "  (wikidata lookup disabled; use --try_wikidata on a networked machine)"))

if __name__ == "__main__":
    main()