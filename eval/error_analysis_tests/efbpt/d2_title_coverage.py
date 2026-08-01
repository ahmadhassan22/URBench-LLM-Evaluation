#!/usr/bin/env python3
"""
d2_title_coverage.py

Settles COVERAGE vs RANKING for DIAGNOSTIC D2, definitively.

The missed-title probe in d2_retrieve_and_recall.py queried the index with a
bare title and checked the top-5. That cannot distinguish "the page is not in
the corpus" from "the page is in the corpus but a one-word query did not rank
it in the top 5". It reported major pages such as Porsche, Douglas Adams and
Rainbow as absent, which is implausible for a 6.4M-article dump. That probe
result must not be used.

This script answers the question directly: it streams the metadata file once
and builds the exact set of article titles present in the corpus. No
embeddings, no index, no model. A title is present or it is not.

Reads ~25.8GB sequentially. Title is parsed from the line prefix rather than
by full JSON decoding, which is far faster over 24M lines. Any line whose
prefix does not match falls back to json.loads, and unparsable lines are
counted and reported rather than silently dropped.

Usage (compute node preferred; no GPU needed):
  python eval/error_analysis_tests/efbpt/d2_title_coverage.py
"""

import json
import os
import re
import sys
import time

BASE = "/mnt/home/user41/URBench"
META_PATH = f"{BASE}/rag/index/wikipedia_full_meta.jsonl"
DEV_PATH = "data/strategyqa_official/dev200_seed4242.jsonl"
OUT_DIR = "outputs/efbpt/d2"

TITLE_RE = re.compile(rb'^\{"title":\s*"((?:[^"\\]|\\.)*)"')


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def norm(t):
    return " ".join(str(t).replace("_", " ").strip().lower().split())


def required_titles(evidence_ids):
    out = []
    for pid in evidence_ids or []:
        if isinstance(pid, str) and "-" in pid:
            out.append(pid.rsplit("-", 1)[0])
    return sorted(set(out))


def main():
    if not os.path.exists(META_PATH):
        die("metadata missing: " + META_PATH)
    if not os.path.exists(DEV_PATH):
        die("DEV200 missing: " + DEV_PATH)

    # ---- gold titles needed ----
    need_raw = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                need_raw.extend(required_titles(r.get("evidence_paragraph_ids")))
    need_raw = sorted(set(need_raw))
    need = {norm(t): t for t in need_raw}
    print("[data] %d unique gold evidence titles across DEV200" % len(need_raw))

    # ---- stream the corpus ----
    print("[scan] streaming %s (~25.8GB) ..." % META_PATH, flush=True)
    titles = set()
    n_lines = 0
    n_fallback = 0
    n_bad = 0
    t0 = time.time()

    with open(META_PATH, "rb") as f:
        for raw in f:
            n_lines += 1
            m = TITLE_RE.match(raw)
            if m:
                # The metadata was written with ensure_ascii=False, so non-ASCII
                # is raw UTF-8, NOT \uXXXX escapes. Applying unicode_escape here
                # would corrupt accented titles (Pantheon with an accent, etc.)
                # and report them as absent. Only JSON's literal escapes for
                # quote and backslash need undoing.
                t = m.group(1).decode("utf-8", "replace")
                t = t.replace('\\"', '"').replace("\\\\", "\\")
                titles.add(norm(t))
            else:
                n_fallback += 1
                try:
                    titles.add(norm(json.loads(raw.decode("utf-8"))["title"]))
                except Exception:
                    n_bad += 1
            if n_lines % 2_000_000 == 0:
                print("[scan]   %d lines, %d unique titles, %.0fs"
                      % (n_lines, len(titles), time.time() - t0), flush=True)

    print("[scan] done: %d lines, %d unique titles, %.0fs"
          % (n_lines, len(titles), time.time() - t0))
    print("[scan] prefix-parse failures (fell back to json): %d" % n_fallback)
    print("[scan] unparsable lines: %d" % n_bad)
    if n_lines != 23963971:
        print("[scan] WARNING: line count %d != index ntotal 23,963,971" % n_lines)

    # ---- answer the question ----
    present = sorted(orig for k, orig in need.items() if k in titles)
    absent = sorted(orig for k, orig in need.items() if k not in titles)

    print("\n" + "=" * 78)
    print("GOLD EVIDENCE TITLE COVERAGE IN THE FULL CORPUS")
    print("=" * 78)
    print("gold titles required : %d" % len(need_raw))
    print("PRESENT in corpus    : %d (%.1f%%)"
          % (len(present), 100.0 * len(present) / len(need_raw)))
    print("ABSENT from corpus   : %d (%.1f%%)"
          % (len(absent), 100.0 * len(absent) / len(need_raw)))
    print("\nIf coverage is high, then low retrieval recall is a RANKING problem,")
    print("not a corpus problem, and better querying can fix it.")
    print("If coverage is low, no query strategy can recover the missing pages.")
    print("\nfirst 40 ABSENT titles:")
    for t in absent[:40]:
        print("   %s" % t)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "corpus_lines": n_lines,
        "corpus_unique_titles": len(titles),
        "gold_titles_required": len(need_raw),
        "gold_titles_present": len(present),
        "gold_titles_absent": len(absent),
        "coverage_pct": round(100.0 * len(present) / len(need_raw), 2),
        "absent_titles": absent,
        "present_titles": present,
    }
    p = os.path.join(OUT_DIR, "d2_title_coverage.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("\n[verified] wrote %s" % p)


if __name__ == "__main__":
    main()
