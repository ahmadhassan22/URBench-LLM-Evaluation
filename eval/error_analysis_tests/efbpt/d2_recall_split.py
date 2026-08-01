#!/usr/bin/env python3
"""
d2_recall_split.py

DIAGNOSTIC D2, step 2. Offline. No GPU, no index, no model.

Separates the two walls found in D2 using files already on disk:

  WALL 1  COVERAGE  - the gold page is not in the corpus at all.
  WALL 2  RANKING   - the gold page IS in the corpus but retrieval missed it.

Inputs:
  outputs/efbpt/d2/d2_retrievals.jsonl      (what retrieval returned)
  outputs/efbpt/d2/d2_title_coverage.json   (what the corpus actually contains)

The missed-title probe in d2_retrieve_and_recall.py is NOT used here. It
contradicts the corpus scan (it reported Apple, Animal, Ancient Egypt and
American football as absent when the scan found them present), because a bare
one-word title query competes against 24M chunks and the right page can fall
outside the top 5. The corpus scan is the authority on presence.

Also computes the ORACLE CEILING: if entity linking were perfect and the
system could fetch a page by exact title, how many questions would get full
gold coverage? That is the upper bound on any entity-grounded method, and it
is set purely by corpus coverage.

Usage (login node is fine):
  python eval/error_analysis_tests/efbpt/d2_recall_split.py
"""

import json
import os
import sys

RETR_PATH = "outputs/efbpt/d2/d2_retrievals.jsonl"
COV_PATH = "outputs/efbpt/d2/d2_title_coverage.json"
OUT_PATH = "outputs/efbpt/d2/d2_recall_split.json"

K_REPORT = [1, 3, 5, 10]


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def norm(t):
    return " ".join(str(t).replace("_", " ").strip().lower().split())


def main():
    for p in (RETR_PATH, COV_PATH):
        if not os.path.exists(p):
            die("missing: " + p)

    cov = json.load(open(COV_PATH, "r", encoding="utf-8"))
    present = {norm(t) for t in cov["present_titles"]}
    absent = {norm(t) for t in cov["absent_titles"]}
    print("[coverage] gold titles present in corpus: %d" % len(present))
    print("[coverage] gold titles absent from corpus: %d" % len(absent))
    print("[coverage] coverage: %.2f%%" % cov["coverage_pct"])

    rows = []
    with open(RETR_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("[data] %d questions" % len(rows))

    # sanity: every required title must be classified by the coverage scan
    unknown = set()
    for r in rows:
        for t in r["required"]:
            if norm(t) not in present and norm(t) not in absent:
                unknown.add(t)
    if unknown:
        die("%d required titles are in neither present nor absent lists, "
            "first 5: %s" % (len(unknown), sorted(unknown)[:5]))
    print("[check] every required title is classified by the coverage scan")

    stats = {}
    for k in K_REPORT:
        n_req = n_req_avail = n_req_gone = 0
        n_hit = n_hit_avail = 0
        for r in rows:
            got = {norm(h["title"]) for h in r["retrieved"][:k]}
            for t in r["required"]:
                nt = norm(t)
                n_req += 1
                hit = nt in got
                if hit:
                    n_hit += 1
                if nt in present:
                    n_req_avail += 1
                    if hit:
                        n_hit_avail += 1
                else:
                    n_req_gone += 1
                    if hit:
                        die("title %r marked absent from corpus but was "
                            "retrieved — the two files disagree" % t)
        stats[k] = {
            "required_total": n_req,
            "retrieved_total": n_hit,
            "recall_raw": round(100.0 * n_hit / n_req, 2),
            "required_available": n_req_avail,
            "retrieved_available": n_hit_avail,
            "recall_among_available": round(100.0 * n_hit_avail / n_req_avail, 2),
            "required_missing_from_corpus": n_req_gone,
        }

    print("\n" + "=" * 82)
    print("RECALL, SPLIT BY WHETHER THE GOLD PAGE EXISTS IN THE CORPUS")
    print("=" * 82)
    print("%4s %12s %12s %14s   %s"
          % ("k", "raw recall", "ceiling", "recall|available", "meaning"))
    for k in K_REPORT:
        s = stats[k]
        ceiling = round(100.0 * s["required_available"] / s["required_total"], 2)
        print("%4d %11.2f%% %11.2f%% %13.2f%%"
              % (k, s["recall_raw"], ceiling, s["recall_among_available"]))
    print("\nceiling = share of required titles that are in the corpus at all.")
    print("recall|available = of the pages that ARE there, how many were found.")
    print("The gap between ceiling and raw recall is the RANKING failure.")

    # ---- per-question view ----
    n_all_avail = n_none_avail = 0
    for r in rows:
        avail = [t for t in r["required"] if norm(t) in present]
        if len(avail) == len(r["required"]):
            n_all_avail += 1
        if len(avail) == 0:
            n_none_avail += 1

    print("\n" + "=" * 82)
    print("ORACLE CEILING — perfect entity linking + exact title lookup")
    print("=" * 82)
    print("questions where ALL gold pages exist in the corpus : %d / %d (%.1f%%)"
          % (n_all_avail, len(rows), 100.0 * n_all_avail / len(rows)))
    print("questions where NO gold page exists                : %d / %d (%.1f%%)"
          % (n_none_avail, len(rows), 100.0 * n_none_avail / len(rows)))
    print("\nThe first number is the hard upper bound on ANY method built on this")
    print("corpus: even flawless entity linking cannot fetch a page that is")
    print("absent. Raising it requires rebuilding the corpus, not better search.")

    # ---- how many required titles are literally the question's own subject ----
    print("\n" + "=" * 82)
    print("REQUIRED TITLES PER QUESTION")
    print("=" * 82)
    counts = {}
    for r in rows:
        counts[len(r["required"])] = counts.get(len(r["required"]), 0) + 1
    for k in sorted(counts):
        print("  %d gold title(s): %3d questions" % (k, counts[k]))
    print("\nA question is only fully covered when EVERY one of these is")
    print("retrieved, which is why per-question recall is far below per-title.")

    out = {
        "coverage_pct": cov["coverage_pct"],
        "n_questions": len(rows),
        "by_k": stats,
        "questions_all_gold_available": n_all_avail,
        "questions_no_gold_available": n_none_avail,
        "required_title_count_histogram": counts,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("\n[verified] wrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
