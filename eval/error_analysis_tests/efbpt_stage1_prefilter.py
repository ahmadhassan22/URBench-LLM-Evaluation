"""
efbpt_stage1_prefilter.py — REPORT-ONLY pre-filter over the TRAIN pool.

WHAT IT DOES (plain words)
--------------------------
1. Loads the official mapped file (all 2,290 rows).
2. Selects the TRAIN pool: rows that are NOT in eval458 (is_eval false)
   and NOT in DEV50. Should be 1,782 rows.
3. Applies exactly ONE exclusion rule:
      RULE_LEN2: official_decomposition has fewer than 2 steps -> exclude.
   No other filters. The "world-knowledge quantity" filter is NOT implemented
   (not yet precisely defined).
4. Writes a report. Does NOT modify or delete any original data.

OUTPUT
------
    data/strategyqa_official/efbpt/stage1_report.jsonl
        one line per TRAIN row: {qid, status: RETAINED|EXCLUDED, reason, n_steps}
    data/strategyqa_official/efbpt/stage1_summary.txt
        counts + the excluded qid list

HOW TO RUN (CPU, seconds)
-------------------------
    cd /mnt/home/user41/URBench
    python eval/error_analysis_tests/efbpt_stage1_prefilter.py
"""

import json
from pathlib import Path

BASE   = Path("/mnt/home/user41/URBench")
OFF    = BASE / "data/strategyqa_official"
EFBPT  = OFF / "efbpt"

MAPPED = OFF / "strategyqa_official_mapped_urbench_qid.jsonl"
DEV50  = OFF / "dev50_seed42.jsonl"
OUT_R  = EFBPT / "stage1_report.jsonl"
OUT_S  = EFBPT / "stage1_summary.txt"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    rows  = load_jsonl(MAPPED)
    dev50 = {r["urbench_qid"] for r in load_jsonl(DEV50)}
    print(f"mapped rows: {len(rows)} | dev50 qids: {len(dev50)}")

    train = [r for r in rows if not r.get("is_eval") and r["urbench_qid"] not in dev50]
    print(f"TRAIN pool: {len(train)} (expected 1782)")
    if len(train) != 1782:
        print("!! WARNING: TRAIN pool size differs from expected 1782 — check is_eval/dev50 logic before trusting this report.")

    retained, excluded = [], []
    with open(OUT_R, "w", encoding="utf-8") as f:
        for r in train:
            n = len(r.get("official_decomposition", []))
            if n < 2:
                rec = {"qid": r["urbench_qid"], "status": "EXCLUDED",
                       "reason": "RULE_LEN2: official_decomposition < 2 steps",
                       "n_steps": n}
                excluded.append(rec)
            else:
                rec = {"qid": r["urbench_qid"], "status": "RETAINED",
                       "reason": None, "n_steps": n}
                retained.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUT_S, "w", encoding="utf-8") as f:
        f.write("STAGE 1 PRE-FILTER — REPORT ONLY (no data modified)\n")
        f.write("=" * 60 + "\n")
        f.write(f"TRAIN pool rows        : {len(train)}\n")
        f.write(f"RETAINED (>=2 steps)   : {len(retained)}\n")
        f.write(f"EXCLUDED (RULE_LEN2)   : {len(excluded)}\n")
        f.write("\nEXCLUDED QIDS:\n")
        for rec in excluded:
            f.write(f"  {rec['qid']}  (steps={rec['n_steps']})\n")

    print(f"RETAINED: {len(retained)} | EXCLUDED (len<2): {len(excluded)}")
    print("report :", OUT_R)
    print("summary:", OUT_S)


if __name__ == "__main__":
    main()