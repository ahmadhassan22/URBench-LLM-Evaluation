#!/usr/bin/env python3
"""
Build DEV200 = DEV50 (50 existing rows) + 150 new rows.

Frozen rules (docs/EFBPT_PLAN_A_FREEZE.md, section 3):
  - The 150 new rows are drawn from the free Stage-1-RETAINED pool.
  - Free pool = RETAINED  minus  DEV50, eval458, AUDIT30, BLIND30.
  - Plain random draw, seed 4242, NOT stratified.
  - DEV50 must be a strict subset of DEV200.
  - DEV200 must be disjoint from eval458, AUDIT30, BLIND30.

Determinism:
  The free pool is SORTED before sampling. Without sorting, the same seed can
  give different results between runs, because file/set order is not stable.

Usage:
  python efbpt_build_dev200.py --dry-run     # check numbers, write nothing
  python efbpt_build_dev200.py               # actually write the files
  python efbpt_build_dev200.py --force       # overwrite existing output

Run --dry-run first. Read the numbers. Only then run for real.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Config (frozen)
# ----------------------------------------------------------------------------

SEED = 4242
N_NEW = 150
N_TOTAL = 200

REPO = Path(__file__).resolve().parents[3]  # eval/error_analysis_tests/efbpt/ -> repo root

MAPPED      = REPO / "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl"
STAGE1      = REPO / "data/strategyqa_official/efbpt/stage1_report.jsonl"
DEV50_QIDS  = REPO / "data/strategyqa_official/dev50_seed42_qids.txt"
AUDIT30_QIDS= REPO / "data/strategyqa_official/efbpt/audit30_qids.txt"
BLIND30_QIDS= REPO / "data/strategyqa_official/efbpt/blind30_qids.txt"
EVAL458     = REPO / "data/sdfr_splits/strategyqa_eval.jsonl"

OUT_ROWS    = REPO / "data/strategyqa_official/dev200_seed4242.jsonl"
OUT_QIDS    = REPO / "data/strategyqa_official/dev200_seed4242_qids.txt"

# Expected values, verified 2026-07-24. Script stops if reality disagrees.
EXPECT_MAPPED_ROWS   = 2290
EXPECT_STAGE1_ROWS   = 1782
EXPECT_RETAINED      = 1770
EXPECT_DEV50         = 50
EXPECT_AUDIT30       = 30
EXPECT_BLIND30       = 30
EXPECT_EVAL458       = 458
EXPECT_FREE_POOL     = 1712


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def die(msg):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def read_jsonl(path):
    if not path.exists():
        die(f"missing file: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                die(f"{path} line {i}: bad JSON: {e}")
    return rows


def read_qid_file(path):
    if not path.exists():
        die(f"missing file: {path}")
    qids = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(qids) != len(set(qids)):
        die(f"{path} has duplicate qids")
    return qids


def check(label, got, want):
    ok = (got == want)
    mark = "OK " if ok else "BAD"
    print(f"  [{mark}] {label:<44} {got:>6}   (expected {want})")
    return ok


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and verify everything, but write no files")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting existing output files")
    args = ap.parse_args()

    print("=" * 74)
    print("DEV200 BUILDER   seed=%d   new=%d   total=%d" % (SEED, N_NEW, N_TOTAL))
    print("mode: %s" % ("DRY RUN (no files written)" if args.dry_run else "WRITE"))
    print("=" * 74)

    # ---- load inputs -------------------------------------------------------
    print("\n[1] Loading input files")

    mapped_rows = read_jsonl(MAPPED)
    mapped = {}
    for r in mapped_rows:
        q = r.get("urbench_qid")
        if q is None:
            die("a row in the mapped file has no urbench_qid")
        if q in mapped:
            die(f"duplicate urbench_qid in mapped file: {q}")
        mapped[q] = r

    stage1_rows = read_jsonl(STAGE1)
    retained = {r["qid"] for r in stage1_rows if r.get("status") == "RETAINED"}

    dev50    = read_qid_file(DEV50_QIDS)
    audit30  = set(read_qid_file(AUDIT30_QIDS))
    blind30  = set(read_qid_file(BLIND30_QIDS))

    eval458_rows = read_jsonl(EVAL458)
    eval458 = {r["qid"] for r in eval458_rows}

    ok = True
    ok &= check("mapped rows",        len(mapped_rows), EXPECT_MAPPED_ROWS)
    ok &= check("stage1 rows",        len(stage1_rows), EXPECT_STAGE1_ROWS)
    ok &= check("stage1 RETAINED",    len(retained),    EXPECT_RETAINED)
    ok &= check("DEV50 qids",         len(dev50),       EXPECT_DEV50)
    ok &= check("AUDIT30 qids",       len(audit30),     EXPECT_AUDIT30)
    ok &= check("BLIND30 qids",       len(blind30),     EXPECT_BLIND30)
    ok &= check("eval458 qids",       len(eval458),     EXPECT_EVAL458)
    if not ok:
        die("input counts do not match the frozen expectations. STOP. "
            "Do not change the expectations without understanding why.")

    # ---- build the free pool ----------------------------------------------
    print("\n[2] Building free pool")

    dev50_set = set(dev50)
    protected = dev50_set | eval458 | audit30 | blind30
    free_pool = sorted(retained - protected)   # SORTED = deterministic

    print(f"  retained                {len(retained):>6}")
    print(f"  minus DEV50             -{len(dev50_set):>5}")
    print(f"  minus eval458           -{len(eval458 & retained):>5}  (overlap with retained)")
    print(f"  minus AUDIT30           -{len(audit30 & retained):>5}  (overlap with retained)")
    print(f"  minus BLIND30           -{len(blind30 & retained):>5}  (overlap with retained)")
    print(f"  = free pool             {len(free_pool):>6}")

    if not check("free pool size", len(free_pool), EXPECT_FREE_POOL):
        die("free pool size changed. Something upstream moved. STOP and investigate.")

    if len(free_pool) < N_NEW:
        die(f"free pool has only {len(free_pool)} rows, need {N_NEW}")

    # ---- sample ------------------------------------------------------------
    print(f"\n[3] Sampling {N_NEW} rows with seed {SEED}")

    rng = random.Random(SEED)
    new_qids = sorted(rng.sample(free_pool, N_NEW))
    print(f"  sampled {len(new_qids)} qids")
    print(f"  first 3: {new_qids[:3]}")
    print(f"  last  3: {new_qids[-3:]}")

    dev200_qids = sorted(dev50_set | set(new_qids))

    # ---- verify ------------------------------------------------------------
    print("\n[4] Verifying DEV200")

    v = True
    v &= check("DEV200 total (unique)",        len(dev200_qids), N_TOTAL)
    v &= check("DEV50 rows inside DEV200",     len(dev50_set & set(dev200_qids)), EXPECT_DEV50)
    v &= check("overlap with eval458",         len(set(dev200_qids) & eval458), 0)
    v &= check("overlap with AUDIT30",         len(set(dev200_qids) & audit30), 0)
    v &= check("overlap with BLIND30",         len(set(dev200_qids) & blind30), 0)
    v &= check("new rows inside RETAINED",     len(set(new_qids) & retained), N_NEW)
    v &= check("new rows overlapping DEV50",   len(set(new_qids) & dev50_set), 0)

    missing = [q for q in dev200_qids if q not in mapped]
    v &= check("DEV200 qids missing in mapped", len(missing), 0)
    if missing:
        print(f"        missing examples: {missing[:5]}")

    rows = [mapped[q] for q in dev200_qids if q in mapped]

    n_is_eval = sum(1 for r in rows if r.get("is_eval") is True)
    v &= check("rows flagged is_eval=true", n_is_eval, 0)

    n_short = sum(1 for r in rows if len(r.get("official_decomposition") or []) < 2)
    v &= check("rows failing RULE_LEN2", n_short, 0)

    n_no_q = sum(1 for r in rows if not (r.get("question_ur") or "").strip())
    v &= check("rows with empty question_ur", n_no_q, 0)

    if not v:
        die("verification failed. Nothing was written.")

    print("\n  All checks passed.")

    # ---- write -------------------------------------------------------------
    print("\n[5] Writing output")

    if args.dry_run:
        print("  DRY RUN — no files written.")
        print(f"  would write: {OUT_ROWS}")
        print(f"  would write: {OUT_QIDS}")
        print("\nDone (dry run).")
        return

    for p in (OUT_ROWS, OUT_QIDS):
        if p.exists() and not args.force:
            die(f"{p} already exists. Use --force only if you are sure you want "
                f"to overwrite a frozen evaluation set.")

    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)

    with OUT_ROWS.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    OUT_QIDS.write_text("\n".join(dev200_qids) + "\n", encoding="utf-8")

    print(f"  wrote {len(rows)} rows  -> {OUT_ROWS}")
    print(f"  wrote {len(dev200_qids)} qids -> {OUT_QIDS}")
    print("\nDone. DEV200 is now frozen. Do not regenerate it.")


if __name__ == "__main__":
    main()
