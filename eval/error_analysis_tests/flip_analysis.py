"""
Flip analysis: does SDFR-UR break questions CoT got right, or fail on already-hard ones?
"""
import json

COT  = "/mnt/home/user41/URBench/outputs/strategyqa/qwen3_14b/strategyqa_cot_qwen3_14b.jsonl"
SDFR = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_qwen3_14b.jsonl"

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

cot  = {r["qid"]: r["correct"] for r in load(COT)}
sdfr = load(SDFR)

both_right, both_wrong, flip_to_wrong, flip_to_right, no_cot = 0,0,0,0,0

for r in sdfr:
    qid = r["qid"]
    if qid not in cot:
        no_cot += 1; continue
    c, s = cot[qid], r["correct"]
    if c and s: both_right += 1
    elif not c and not s: both_wrong += 1
    elif c and not s: flip_to_wrong += 1
    elif not c and s: flip_to_right += 1

n = len(sdfr) - no_cot
print(f"matched: {n}  (no CoT match: {no_cot})")
print(f"both right:      {both_right}  ({100*both_right/n:.1f}%)")
print(f"both wrong:      {both_wrong}  ({100*both_wrong/n:.1f}%)")
print(f"CoT right→SDFR wrong (FLIPPED, retrieval hurt):  {flip_to_wrong}  ({100*flip_to_wrong/n:.1f}%)")
print(f"CoT wrong→SDFR right (retrieval helped):          {flip_to_right}  ({100*flip_to_right/n:.1f}%)")