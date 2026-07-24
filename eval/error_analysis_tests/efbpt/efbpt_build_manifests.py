#!/usr/bin/env python3
"""
Build the nested EFBPT training manifests: 100 subset-of 250 subset-of 500.

Frozen rules (docs/EFBPT_PLAN_A_FREEZE.md, section 4 + AMENDMENT 1):
  - Pool: Stage-1 RETAINED rows minus ALL protected sets
          (DEV50, DEV200, eval458, AUDIT30, BLIND30).
  - Stratify by: answer label (yes/no) x hop count
    (hop = len(official_decomposition), bucketed as 2, 3, 4, 5+).
  - Seed: 8888. Plain proportional stratified sampling, largest-remainder.
  - Nesting is guaranteed by construction: each stratum is shuffled ONCE,
    and the 100/250/500 manifests take prefixes of the same shuffled order.
    quota(100) <= quota(250) <= quota(500) is enforced per stratum.

Usage:
  python efbpt_build_manifests.py --dry-run   # verify everything, write nothing
  python efbpt_build_manifests.py             # write the three qid files
  python efbpt_build_manifests.py --force     # overwrite existing outputs
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Config (frozen)
# ----------------------------------------------------------------------------

SEED = 8888
SIZES = (100, 250, 500)

REPO = Path(__file__).resolve().parents[3]

MAPPED       = REPO / "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl"
STAGE1       = REPO / "data/strategyqa_official/efbpt/stage1_report.jsonl"
DEV50_QIDS   = REPO / "data/strategyqa_official/dev50_seed42_qids.txt"
DEV200_QIDS  = REPO / "data/strategyqa_official/dev200_seed4242_qids.txt"
AUDIT30_QIDS = REPO / "data/strategyqa_official/efbpt/audit30_qids.txt"
BLIND30_QIDS = REPO / "data/strategyqa_official/efbpt/blind30_qids.txt"
EVAL458      = REPO / "data/sdfr_splits/strategyqa_eval.jsonl"

OUT = {
    100: REPO / "data/strategyqa_official/efbpt/plan_a_qids_100.txt",
    250: REPO / "data/strategyqa_official/efbpt/plan_a_qids_250.txt",
    500: REPO / "data/strategyqa_official/efbpt/plan_a_qids_500.txt",
}

# Expected values, verified 2026-07-24. Script stops if reality disagrees.
EXPECT_RETAINED  = 1770
EXPECT_DEV200    = 200
EXPECT_FREE_POOL = 1562   # 1712 - 150 (the DEV200 rows that live inside RETAINED)


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
    print(f"  [{mark}] {label:<46} {got:>6}   (expected {want})")
    return ok


def hop_bucket(n):
    return "5+" if n >= 5 else str(n)


def largest_remainder(n, caps):
    """Allocate n items across strata proportionally to caps (available
    counts), never exceeding caps. Returns {stratum: quota}."""
    total = sum(caps.values())
    if n > total:
        die(f"cannot allocate {n} from pool of {total}")
    exact = {s: n * c / total for s, c in caps.items()}
    quota = {s: min(int(exact[s]), caps[s]) for s in caps}
    # distribute the remainder by largest fractional part, respecting caps
    remainder = n - sum(quota.values())
    order = sorted(caps, key=lambda s: (exact[s] - int(exact[s])), reverse=True)
    i = 0
    while remainder > 0:
        s = order[i % len(order)]
        if quota[s] < caps[s]:
            quota[s] += 1
            remainder -= 1
        i += 1
        if i > 10 * len(order) + n:
            die("allocation loop failed to converge")
    return quota


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("=" * 76)
    print("MANIFEST BUILDER   seed=%d   sizes=%s" % (SEED, list(SIZES)))
    print("strata: answer label (yes/no) x hop bucket (2,3,4,5+)")
    print("mode: %s" % ("DRY RUN (no files written)" if args.dry_run else "WRITE"))
    print("=" * 76)

    # ---- load --------------------------------------------------------------
    print("\n[1] Loading inputs")

    mapped = {}
    for r in read_jsonl(MAPPED):
        mapped[r["urbench_qid"]] = r

    stage1_rows = read_jsonl(STAGE1)
    retained = {r["qid"] for r in stage1_rows if r.get("status") == "RETAINED"}

    dev50   = set(read_qid_file(DEV50_QIDS))
    dev200  = set(read_qid_file(DEV200_QIDS))
    audit30 = set(read_qid_file(AUDIT30_QIDS))
    blind30 = set(read_qid_file(BLIND30_QIDS))
    eval458 = {r["qid"] for r in read_jsonl(EVAL458)}

    ok = True
    ok &= check("stage1 RETAINED", len(retained), EXPECT_RETAINED)
    ok &= check("DEV200 qids", len(dev200), EXPECT_DEV200)
    ok &= check("DEV50 inside DEV200", len(dev50 & dev200), 50)
    if not ok:
        die("input counts do not match frozen expectations. STOP.")

    # ---- free pool ---------------------------------------------------------
    print("\n[2] Building free pool")

    protected = dev50 | dev200 | audit30 | blind30 | eval458
    free_pool = sorted(retained - protected)   # SORTED = deterministic

    print(f"  retained                 {len(retained):>6}")
    print(f"  minus protected overlap  -{len(retained & protected):>5}")
    print(f"  = free pool              {len(free_pool):>6}")

    if not check("free pool size", len(free_pool), EXPECT_FREE_POOL):
        die("free pool size changed. Something upstream moved. STOP.")

    # ---- stratify ----------------------------------------------------------
    print("\n[3] Stratifying free pool")

    strata = defaultdict(list)
    for q in free_pool:
        r = mapped.get(q)
        if r is None:
            die(f"free-pool qid {q} not found in mapped file")
        label = "yes" if r["answer"] is True else "no"
        hop = hop_bucket(len(r.get("official_decomposition") or []))
        strata[(label, hop)].append(q)

    caps = {s: len(v) for s, v in strata.items()}
    print(f"  {len(caps)} strata:")
    for s in sorted(caps):
        print(f"    {s[0]:>3} / hop {s[1]:<2} : {caps[s]:>5} rows "
              f"({100.0*caps[s]/len(free_pool):.1f}%)")

    # ---- quotas (nested by construction) -----------------------------------
    print("\n[4] Computing nested quotas (largest remainder)")

    q500 = largest_remainder(500, caps)
    q250 = largest_remainder(250, q500)   # capped by q500 -> q250 <= q500
    q100 = largest_remainder(100, q250)   # capped by q250 -> q100 <= q250

    print(f"  {'stratum':<16} {'pool':>6} {'n=100':>6} {'n=250':>6} {'n=500':>6}")
    for s in sorted(caps):
        print(f"  {s[0]+'/hop '+s[1]:<16} {caps[s]:>6} {q100[s]:>6} "
              f"{q250[s]:>6} {q500[s]:>6}")
    print(f"  {'TOTAL':<16} {len(free_pool):>6} {sum(q100.values()):>6} "
          f"{sum(q250.values()):>6} {sum(q500.values()):>6}")

    for s in caps:
        if not (q100[s] <= q250[s] <= q500[s] <= caps[s]):
            die(f"quota monotonicity broken for stratum {s}")

    # ---- sample: shuffle each stratum ONCE, take prefixes ------------------
    print(f"\n[5] Sampling with seed {SEED} (one shuffle per stratum, prefixes)")

    rng = random.Random(SEED)
    manifests = {n: [] for n in SIZES}
    for s in sorted(strata):                 # sorted stratum order = deterministic
        order = sorted(strata[s])            # sorted before shuffle = deterministic
        rng.shuffle(order)
        manifests[100].extend(order[:q100[s]])
        manifests[250].extend(order[:q250[s]])
        manifests[500].extend(order[:q500[s]])

    manifests = {n: sorted(v) for n, v in manifests.items()}
    for n in SIZES:
        print(f"  n={n:<4} first: {manifests[n][0]}   last: {manifests[n][-1]}")

    # ---- verify ------------------------------------------------------------
    print("\n[6] Verifying manifests")

    v = True
    m100, m250, m500 = (set(manifests[n]) for n in SIZES)
    v &= check("manifest 100 size", len(m100), 100)
    v &= check("manifest 250 size", len(m250), 250)
    v &= check("manifest 500 size", len(m500), 500)
    v &= check("nesting: 100 inside 250", len(m100 - m250), 0)
    v &= check("nesting: 250 inside 500", len(m250 - m500), 0)
    v &= check("500 overlap with DEV50",   len(m500 & dev50), 0)
    v &= check("500 overlap with DEV200",  len(m500 & dev200), 0)
    v &= check("500 overlap with eval458", len(m500 & eval458), 0)
    v &= check("500 overlap with AUDIT30", len(m500 & audit30), 0)
    v &= check("500 overlap with BLIND30", len(m500 & blind30), 0)
    v &= check("500 inside RETAINED",      len(m500 & retained), 500)
    if not v:
        die("verification failed. Nothing was written.")
    print("\n  All checks passed.")

    # ---- write -------------------------------------------------------------
    print("\n[7] Writing output")

    if args.dry_run:
        print("  DRY RUN — no files written.")
        for n in SIZES:
            print(f"  would write: {OUT[n]}")
        print("\nDone (dry run).")
        return

    for n in SIZES:
        if OUT[n].exists() and not args.force:
            die(f"{OUT[n]} already exists. Use --force only if you are sure "
                f"you want to overwrite a frozen training manifest.")

    for n in SIZES:
        OUT[n].parent.mkdir(parents=True, exist_ok=True)
        OUT[n].write_text("\n".join(manifests[n]) + "\n", encoding="utf-8")
        print(f"  wrote {len(manifests[n])} qids -> {OUT[n]}")

    print("\nDone. Manifests are now frozen. Do not regenerate them.")


if __name__ == "__main__":
    main()
