"""
Control test: read BOTH-RIGHT cases (CoT right, SDFR right) same way as flip cases.
Why: 108/135 flips are labeled no-fact/wrong-fact, but fact-coverage script showed
wrong≈right (0.4205 vs 0.4174) — contradiction. This checks if no-fact is common
in successes too, or specific to failures.
"""
import json

COT  = "/mnt/home/user41/URBench/outputs/strategyqa/qwen3_14b/strategyqa_cot_qwen3_14b.jsonl"
SDFR = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_qwen3_14b.jsonl"
OUT  = "/mnt/home/user41/URBench/outputs/sdfr/control_both_right.md"

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

cot = {r["qid"]: r for r in load(COT)}
sdfr = load(SDFR)

both_right = [r for r in sdfr if r["qid"] in cot and cot[r["qid"]]["correct"] and r["correct"]]

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"# StrategyQA CONTROL (CoT right AND SDFR right), n={len(both_right)}\n\n")
    f.write("| # | qid | question | gold | pred | retrieved (q/a/sim) | category |\n")
    f.write("|---|-----|----------|------|------|----------------------|----------|\n")
    for i, r in enumerate(both_right, 1):
        ret = "<br>".join(f"{x['q'][:70]} / {x['a']} / {x['sim']:.3f}" for x in r["retrieved"])
        q = r["question"].replace("|","،")
        f.write(f"| {i} | {r['qid']} | {q} | {r['gold']} | {r['pred']} | {ret} |  |\n")

print(f"Written: {OUT}  ({len(both_right)} rows)")