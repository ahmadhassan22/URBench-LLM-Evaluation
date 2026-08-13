#!/usr/bin/env python3
"""
D4 Section D item 4 — invented-fact verification.

READ-ONLY. Writes NO files. All output goes to stdout (the SLURM log).

Checks whether PASS 1 invented facts that are not supported by the
source passages it was given. Selection is the FIRST 10 rows of
d4_extractions.jsonl in file order — no cherry-picking.

fetch_chunks caps at 3 chunks PER TITLE in file order, so requesting
only these rows' titles returns exactly the chunks D4 pass 1 saw.
"""
import sys, os, json, re, string

REPO = "/mnt/home/user41/URBench"
EFBPT = os.path.join(REPO, "eval/error_analysis_tests/efbpt")
sys.path.insert(0, EFBPT)

EXTRACTIONS = os.path.join(REPO, "outputs/efbpt/d4/d4_extractions.jsonl")
ARMX1 = os.path.join(REPO, "outputs/efbpt/d4/d4_armX1.jsonl")
N_ROWS = 10
CHUNKS_PER_TITLE = 3          # must equal d4_extract_facts.py:55
FLAG_RUN = 5                  # print full passages when weakest run < this

from d3_oracle_retrieval import norm, fetch_chunks, META_PATH   # noqa: E402

PUNCT = str.maketrans({c: " " for c in string.punctuation})
STOP = set("""a an the and or but if then than that this these those of in on at
to for from by with as is are was were be been being it its it's he she they
them his her their there here which who whom whose what when where how why not
no yes do does did done have has had can could will would shall should may
might must also into over under about after before during between within
each any all both some such other more most many much very own same so too
one two""".split())


def words(s):
    return [w for w in s.lower().translate(PUNCT).split() if w]


def longest_run(fact, passage):
    """Longest contiguous word run from fact that also appears in passage."""
    fw, pw = words(fact), words(passage)
    if not fw or not pw:
        return 0, ""
    pset = set()
    # index all passage n-grams up to len(fw) for O(n) lookup
    maxn = min(len(fw), len(pw))
    for n in range(1, maxn + 1):
        for i in range(len(pw) - n + 1):
            pset.add(tuple(pw[i:i + n]))
        if n > 30:
            break
    best, best_txt = 0, ""
    for n in range(min(len(fw), maxn, 31), 0, -1):
        for i in range(len(fw) - n + 1):
            if tuple(fw[i:i + n]) in pset:
                return n, " ".join(fw[i:i + n])
    return best, best_txt


def main():
    for p in (EXTRACTIONS, ARMX1):
        if not os.path.exists(p):
            sys.exit("MISSING: %s" % p)

    rows = []
    with open(EXTRACTIONS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_ROWS:
                break
            rows.append(json.loads(line))
    print("selected %d rows (first %d in file order)" % (len(rows), N_ROWS))
    print("qids:", [r["qid"] for r in rows])

    x1 = {}
    with open(ARMX1, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            x1[r["qid"]] = r

    needed = sorted({norm(t) for r in rows for t in r["titles"]})
    print("[scan] fetching up to %d chunks for %d titles ..."
          % (CHUNKS_PER_TITLE, len(needed)), flush=True)
    chunks = fetch_chunks(needed, CHUNKS_PER_TITLE)

    empty = [t for t in needed if not chunks[t]]
    print("titles with ZERO chunks: %d %s" % (len(empty), empty[:10]))
    for t in needed:
        print("   %-45s chunks=%d chars=%d"
              % (t[:45], len(chunks[t]), sum(len(c) for c in chunks[t])))

    summary, flagged = [], []
    for r in rows:
        qid = r["qid"]
        passage = "\n".join(c for t in r["titles"] for c in chunks[norm(t)])
        pwords = set(words(passage))
        print("\n" + "=" * 78)
        print("==== %s ====" % qid)
        print("QUESTION_EN : %s" % r.get("question_en", ""))
        print("GOLD        : %s" % x1.get(qid, {}).get("gold", "?"))
        print("X1 PRED     : %s" % x1.get(qid, {}).get("pred", "?"))
        print("TITLES      : %s" % r["titles"])
        print("PASSAGE     : %d chars, %d chunks"
              % (len(passage), sum(len(chunks[norm(t)]) for t in r["titles"])))
        print("EXTRACTED FACTS (%d):" % len(r["facts"]))
        worst = 99
        for j, fact in enumerate(r["facts"], 1):
            n, run = longest_run(fact, passage)
            worst = min(worst, n)
            miss = sorted({w for w in words(fact)
                           if w not in pwords and w not in STOP and
                           (len(w) > 3 or w.isdigit())})
            print("  %d. %s" % (j, fact))
            print("     longest_run=%d  matched=%r" % (n, run))
            print("     content words NOT in passage: %s" % (miss if miss else "none"))
        summary.append((qid, len(r["facts"]), worst))
        if worst < FLAG_RUN:
            flagged.append(qid)

    print("\n" + "=" * 78)
    print("SUMMARY  (weakest = min longest_run across that row's facts)")
    print("%-22s %8s %10s" % ("qid", "n_facts", "weakest"))
    for qid, nf, w in summary:
        print("%-22s %8d %10d %s" % (qid, nf, w, "<-- FLAGGED" if w < FLAG_RUN else ""))
    print("\nflagged rows (weakest run < %d): %d %s"
          % (FLAG_RUN, len(flagged), flagged))

    print("\n" + "=" * 78)
    print("FULL PASSAGES for flagged rows only")
    for r in rows:
        if r["qid"] not in flagged:
            continue
        print("\n---- %s ----" % r["qid"])
        for t in r["titles"]:
            for k, c in enumerate(chunks[norm(t)]):
                print("\n[%s | chunk %d]\n%s" % (t, k, c))
    print("\n=== done. no files written. ===")


if __name__ == "__main__":
    main()
