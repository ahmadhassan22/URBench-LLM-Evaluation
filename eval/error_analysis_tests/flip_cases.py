import json

COT  = "/mnt/home/user41/URBench/outputs/strategyqa/qwen3_14b/strategyqa_cot_qwen3_14b.jsonl"
SDFR = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_qwen3_14b.jsonl"
OUT  = "/mnt/home/user41/URBench/outputs/sdfr/flip_cases.md"

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

cot = {r["qid"]: r for r in load(COT)}
sdfr = load(SDFR)

flips = [r for r in sdfr if r["qid"] in cot and cot[r["qid"]]["correct"] and not r["correct"]]

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"# StrategyQA FLIP cases (CoT right -> SDFR wrong), n={len(flips)}\n\n")
    f.write("| # | qid | question | gold | cot_pred | sdfr_pred | retrieved (q/a/sim) | category |\n")
    f.write("|---|-----|----------|------|----------|-----------|----------------------|----------|\n")
    for i, r in enumerate(flips, 1):
        cp = cot[r["qid"]]["pred_answer"]
        ret = "<br>".join(f"{x['q'][:70]} / {x['a']} / {x['sim']:.3f}" for x in r["retrieved"])
        q = r["question"].replace("|","،")
        f.write(f"| {i} | {r['qid']} | {q} | {r['gold']} | {cp} | {r['pred']} | {ret} |  |\n")

print(f"Written: {OUT}  ({len(flips)} rows)")