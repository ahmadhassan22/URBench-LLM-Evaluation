#!/usr/bin/env python3
"""
d2_retrieve_and_recall.py

DIAGNOSTIC D2, part 1 of 2: retrieval only. NO language model is used here.

Retrieves top-K passages from the FULL Wikipedia index for all 200 DEV200
questions, saves them, and measures TITLE RECALL against the gold evidence
titles in `evidence_paragraph_ids`.

Why this runs first and alone: recall needs no LLM. If retrieval does not
surface the gold evidence pages, we know the bottleneck before spending GPU
hours on generation. And because it is measured without the model, nothing
downstream can shape the number.

Extra diagnostic — MISSED-TITLE PROBE:
For every gold title that the question query failed to retrieve, the title
itself is used as a query against the index. This separates two failures the
earlier RAG error analysis could not tell apart:
  - the page is ABSENT from the index          -> coverage problem
  - the page is present but this query missed it -> query/ranking problem
The old experiment blamed coverage ("Grey seal absent") while running on a
2,152-page filtered index. This settles it on the full 23.96M-chunk index.

Fairness notes, matching DIAGNOSTIC D2 Section C:
  - query = `question_en`, ONE query per question, exactly as the earlier
    experiment did. The only changed variable is the index.
  - embedder unchanged: paraphrase-multilingual-MiniLM-L12-v2, no prefix.
  - retrieval code is IMPORTED from rag/retrieve.py, not reimplemented.

Usage (GPU compute node, NOT the login node; needs ~45G+ RAM for the index):
  python eval/error_analysis_tests/efbpt/d2_retrieve_and_recall.py --test
  python eval/error_analysis_tests/efbpt/d2_retrieve_and_recall.py
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict

BASE = "/mnt/home/user41/URBench"
if BASE not in sys.path:
    sys.path.insert(0, BASE)

DEV_PATH = "data/strategyqa_official/dev200_seed4242.jsonl"
OUT_DIR = "outputs/efbpt/d2"

TOP_K = 10                 # retrieve once at 10; arm R1 slices the first 3
K_REPORT = [1, 3, 5, 10]   # recall reported at these depths
PROBE_K = 5                # depth for the missed-title probe


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def norm_title(t):
    """Compare titles case- and separator-insensitively."""
    return " ".join(str(t).replace("_", " ").strip().lower().split())


def required_titles(evidence_ids):
    """'LendingTree-8' -> 'LendingTree'. rsplit keeps hyphens inside titles
    such as 'Chick-fil-A-1' -> 'Chick-fil-A'."""
    out = []
    for pid in evidence_ids or []:
        if not isinstance(pid, str) or "-" not in pid:
            continue
        out.append(pid.rsplit("-", 1)[0])
    return sorted(set(out))


def load_rows(limit=None):
    if not os.path.exists(DEV_PATH):
        die("DEV200 missing: " + DEV_PATH)
    rows = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "qid": r["urbench_qid"],
                "question_en": r["question_en"],
                "question_ur": r["question_ur"],
                "gold": "yes" if r["answer"] else "no",
                "required": required_titles(r.get("evidence_paragraph_ids")),
                "has_evidence": bool(r.get("has_paragraph_evidence")),
            })
    if limit is None and len(rows) != 200:
        die("DEV200 has %d rows, expected 200" % len(rows))
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="20 rows only")
    ap.add_argument("--test-rows", type=int, default=20)
    ap.add_argument("--skip-probe", action="store_true",
                    help="skip the missed-title probe (faster)")
    args = ap.parse_args()

    from rag.retrieve import Retriever

    rows = load_rows(limit=args.test_rows if args.test else None)
    n = len(rows)

    n_req = [len(r["required"]) for r in rows]
    print("[data] %d rows" % n)
    print("[data] rows with has_paragraph_evidence=True: %d"
          % sum(1 for r in rows if r["has_evidence"]))
    print("[data] required gold titles per question: min=%d median=%d max=%d"
          % (min(n_req), sorted(n_req)[len(n_req) // 2], max(n_req)))
    print("[data] rows with ZERO required titles: %d (excluded from recall)"
          % sum(1 for x in n_req if x == 0))
    print("[data] example required titles, first 5 rows:")
    for r in rows[:5]:
        print("   %s -> %s" % (r["qid"], r["required"]))

    # ---- retrieve ----
    print("\n[retrieve] loading index (this reads ~37GB; expect several minutes)",
          flush=True)
    t0 = time.time()
    retr = Retriever(device="cuda")
    print("[retrieve] retriever ready in %.1fs" % (time.time() - t0), flush=True)

    queries = [r["question_en"] for r in rows]
    print("[retrieve] searching %d queries at top_k=%d ..." % (n, TOP_K), flush=True)
    t0 = time.time()
    hits_all = retr.retrieve(queries, top_k=TOP_K)
    print("[retrieve] done in %.1fs" % (time.time() - t0), flush=True)

    # ---- recall ----
    scored = []
    for r, hits in zip(rows, hits_all):
        req = [norm_title(t) for t in r["required"]]
        rec = OrderedDict()
        for k in K_REPORT:
            got = {norm_title(h["title"]) for h in hits[:k]}
            found = [t for t in req if t in got]
            rec[k] = {
                "n_required": len(req),
                "n_found": len(found),
                "recall": (len(found) / len(req)) if req else None,
                "fully_covered": (len(found) == len(req)) if req else None,
            }
        missed10 = [t for t in r["required"]
                    if norm_title(t) not in {norm_title(h["title"]) for h in hits}]
        scored.append({
            "qid": r["qid"],
            "gold": r["gold"],
            "required": r["required"],
            "missed_at_10": missed10,
            "recall": rec,
            "retrieved": [{"rank": i + 1, "score": h["score"], "title": h["title"],
                           "text": h["text"]} for i, h in enumerate(hits)],
        })

    with_req = [s for s in scored if s["required"]]
    print("\n" + "=" * 78)
    print("TITLE RECALL against gold evidence titles  (n=%d rows with evidence)"
          % len(with_req))
    print("=" * 78)
    print("%6s %14s %18s" % ("k", "mean recall", "fully covered"))
    for k in K_REPORT:
        mr = sum(s["recall"][k]["recall"] for s in with_req) / len(with_req)
        fc = sum(1 for s in with_req if s["recall"][k]["fully_covered"])
        print("%6d %13.1f%% %13d (%.1f%%)"
              % (k, 100 * mr, fc, 100.0 * fc / len(with_req)))

    # how many gold titles are missed entirely at k=10
    all_missed = [t for s in with_req for t in s["missed_at_10"]]
    all_req = [t for s in with_req for t in s["required"]]
    print("\ngold titles required in total : %d" % len(all_req))
    print("gold titles missed at k=10    : %d (%.1f%%)"
          % (len(all_missed), 100.0 * len(all_missed) / max(1, len(all_req))))

    # ---- missed-title probe: absent from index, or just not retrieved? ----
    probe_result = {}
    if not args.skip_probe and all_missed:
        uniq_missed = sorted(set(all_missed))
        print("\n[probe] querying the index with %d unique missed titles ..."
              % len(uniq_missed), flush=True)
        t0 = time.time()
        phits = retr.retrieve(uniq_missed, top_k=PROBE_K)
        print("[probe] done in %.1fs" % (time.time() - t0))
        present, absent = [], []
        for title, hs in zip(uniq_missed, phits):
            got = {norm_title(h["title"]) for h in hs}
            (present if norm_title(title) in got else absent).append(title)
            probe_result[title] = {
                "found_by_title_query": norm_title(title) in got,
                "top_hits": [{"title": h["title"], "score": h["score"]} for h in hs],
            }
        print("\n" + "=" * 78)
        print("MISSED-TITLE PROBE  (is the page absent, or just not retrieved?)")
        print("=" * 78)
        print("unique gold titles missed by the question query : %d" % len(uniq_missed))
        print("  PRESENT in index (found when queried by title): %d (%.1f%%)"
              % (len(present), 100.0 * len(present) / len(uniq_missed)))
        print("  ABSENT / unfindable even by exact title       : %d (%.1f%%)"
              % (len(absent), 100.0 * len(absent) / len(uniq_missed)))
        print("\n  -> pages PRESENT but not retrieved are a QUERY/RANKING failure.")
        print("  -> pages ABSENT are a COVERAGE failure.")
        print("\n  first 15 present-but-missed:", present[:15])
        print("  first 15 absent           :", absent[:15])

    # what did retrieval return instead? title concentration per question
    conc = []
    for s in scored:
        titles = [h["title"] for h in s["retrieved"]]
        c = Counter(titles)
        conc.append(c.most_common(1)[0][1] if c else 0)
    print("\n[diagnostic] of top-%d hits, how many come from the SINGLE most" % TOP_K)
    print("             frequent title (10 = all from one article):")
    print("             mean %.2f  median %d  max %d"
          % (sum(conc) / len(conc), sorted(conc)[len(conc) // 2], max(conc)))
    print("             questions where >=%d of %d hits are one article: %d"
          % (TOP_K - 2, TOP_K, sum(1 for x in conc if x >= TOP_K - 2)))
    print("             (this is the 'partial retrieval' failure mode: one")
    print("              entity crowds out the other in multi-hop questions)")

    # ---- write ----
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_TEST" if args.test else ""

    p1 = os.path.join(OUT_DIR, "d2_retrievals%s.jsonl" % suffix)
    with open(p1, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    n_disk = sum(1 for _ in open(p1, "r", encoding="utf-8"))
    if n_disk != len(scored):
        die("wrote %d but disk has %d lines" % (len(scored), n_disk))
    print("\n[verified] %d lines -> %s" % (n_disk, p1))

    if probe_result:
        p2 = os.path.join(OUT_DIR, "d2_missed_title_probe%s.json" % suffix)
        with open(p2, "w", encoding="utf-8") as f:
            json.dump(probe_result, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print("[verified] probe -> %s" % p2)

    summary = {
        "n_rows": n,
        "n_rows_with_evidence": len(with_req),
        "recall": {str(k): {
            "mean_recall": round(100 * sum(s["recall"][k]["recall"] for s in with_req)
                                 / len(with_req), 2),
            "fully_covered_pct": round(100.0 * sum(
                1 for s in with_req if s["recall"][k]["fully_covered"]) / len(with_req), 2),
        } for k in K_REPORT},
        "gold_titles_required": len(all_req),
        "gold_titles_missed_at_10": len(all_missed),
        "probe_present": len([t for t in probe_result if probe_result[t]["found_by_title_query"]])
                          if probe_result else None,
        "probe_absent": len([t for t in probe_result if not probe_result[t]["found_by_title_query"]])
                         if probe_result else None,
    }
    p3 = os.path.join(OUT_DIR, "d2_recall_summary%s.json" % suffix)
    with open(p3, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("[verified] summary -> %s" % p3)

    if args.test:
        print("\nTEST COMPLETE. 20 rows cannot measure anything. Plumbing check only.")


if __name__ == "__main__":
    main()
