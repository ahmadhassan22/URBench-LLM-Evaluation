"""
efbpt_blind30_sampler.py — sample 30 fresh BLIND rows for schema-v2 validation.

Plain words: picks 30 rows from the Stage-1 ELIGIBLE pool (1,770 rows that
passed the length filter — they may still contain other problems), excluding
AUDIT30, DEV50, and eval458. Fixed seed. Prints QIDs only, never questions,
so the sample stays blind. Runs no model. Refuses to overwrite existing output.
"""
import json, random, sys
from pathlib import Path

BASE   = Path("/mnt/home/user41/URBench")
OFF    = BASE / "data/strategyqa_official"
EF     = OFF / "efbpt"

MAPPED = OFF / "strategyqa_official_mapped_urbench_qid.jsonl"
STAGE1 = EF / "stage1_report.jsonl"
A30_C  = EF / "audit30_candidates.jsonl"
A30_A  = EF / "audit30_answers.jsonl"
DEV50  = OFF / "dev50_seed42_qids.txt"

OUT_ROWS = EF / "blind30_rows.jsonl"
OUT_QIDS = EF / "blind30_qids.txt"
OUT_SUM  = EF / "blind30_summary.txt"

SEED = 20260720   # new fixed seed, recorded here and in the summary

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---- 8. refuse to overwrite: if outputs exist, verify and exit ----
if OUT_QIDS.exists():
    qids = [l.strip() for l in open(OUT_QIDS, encoding="utf-8") if l.strip()]
    rows = load_jsonl(OUT_ROWS) if OUT_ROWS.exists() else []
    print(f"FROZEN sample already exists: {len(qids)} qids, {len(rows)} rows.")
    print("Verifying integrity...")
    assert len(qids) == 30 and len(set(qids)) == 30, "existing qid list corrupt!"
    assert len(rows) == 30, "existing rows file corrupt!"
    assert {r["urbench_qid"] for r in rows} == set(qids), "rows/qids mismatch!"
    print("OK — existing blind30 verified. Exiting without changes.")
    sys.exit(0)

# ---- load sources ----
mapped  = load_jsonl(MAPPED)
stage1  = load_jsonl(STAGE1)
a30_c   = {r["urbench_qid"] for r in load_jsonl(A30_C)}
a30_a   = {r["urbench_qid"] for r in load_jsonl(A30_A)}
audit30 = a30_c | a30_a          # union, per safety correction #3
dev50   = {l.strip() for l in open(DEV50, encoding="utf-8") if l.strip()}

# eval458 via is_eval flag — fail loudly if field is absent/different
eval_qids = {r["urbench_qid"] for r in mapped if r.get("is_eval") is True}
retained  = [r["qid"] for r in stage1 if r.get("status") == "RETAINED"]

# ---- 4. pre-sample assertions ----
print(f"AUDIT30: candidates={len(a30_c)}, answers={len(a30_a)}, union={len(audit30)}")
assert len(mapped)  == 2290, f"mapped rows {len(mapped)} != 2290"
assert len(eval_qids) == 458, (
    f"eval rows via is_eval == {len(eval_qids)}, expected 458. "
    "If 0: the is_eval field is missing/differently named — inspect one mapped "
    "row's keys and STOP; do not guess.")
assert len(dev50) == 50, f"DEV50 unique qids {len(dev50)} != 50"
assert len(retained) == 1770, f"stage1 retained {len(retained)} != 1770"
assert len(audit30) == 30, f"AUDIT30 union {len(audit30)} != 30 — investigate!"

# ---- 5. exclude and sample ----
excluded_sets = audit30 | dev50 | eval_qids
pool = sorted(q for q in retained if q not in excluded_sets)  # sorted -> reproducible
rng = random.Random(SEED)
sample = sorted(rng.sample(pool, 30))

# ---- 7. zero-overlap assertions ----
s = set(sample)
assert len(s) == 30
assert not (s & audit30), "overlap with AUDIT30!"
assert not (s & dev50),   "overlap with DEV50!"
assert not (s & eval_qids), "overlap with eval458!"

# ---- 6. write outputs ----
by_qid = {r["urbench_qid"]: r for r in mapped}
with open(OUT_ROWS, "w", encoding="utf-8") as f:
    for q in sample:
        f.write(json.dumps(by_qid[q], ensure_ascii=False) + "\n")
with open(OUT_QIDS, "w", encoding="utf-8") as f:
    f.write("\n".join(sample) + "\n")

summary = "\n".join([
    "BLIND30 SAMPLE — frozen validation set (schema v2 check)",
    "=" * 60,
    f"seed                 : {SEED}",
    f"mapped rows          : {len(mapped)}",
    f"stage1 eligible pool : {len(retained)}",
    f"excluded AUDIT30     : {len(audit30)}",
    f"excluded DEV50       : {len(dev50)}",
    f"excluded eval458     : {len(eval_qids)}",
    f"pool after exclusion : {len(pool)}",
    f"sampled              : {len(sample)}",
    "overlap AUDIT30/DEV50/eval458 : 0 / 0 / 0 (asserted)",
    "NOTE: pool is Stage-1 ELIGIBLE, not verified-clean.",
    "QIDS:", *[f"  {q}" for q in sample],
])
OUT_SUM.write_text(summary + "\n", encoding="utf-8")
print(summary)
print("\nrows:", OUT_ROWS, "\nqids:", OUT_QIDS, "\nsummary:", OUT_SUM)