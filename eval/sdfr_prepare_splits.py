"""
SDFR-UR Step 1: Prepare retrieval pool + eval splits for all 5 datasets.
Output goes to ~/URBench/data/sdfr_splits/
"""

import json
import os

BASE = os.path.expanduser("~/URBench/data")
OUT  = os.path.join(BASE, "sdfr_splits")
os.makedirs(OUT, exist_ok=True)

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data):>5} examples → {path}")

def split_80_20(data):
    n = len(data)
    cut = int(n * 0.8)
    return data[:cut], data[cut:]

# ── GSM8K ────────────────────────────────────────────────────────────────────
print("\n[GSM8K]")
pool = read_jsonl(f"{BASE}/retrieval_pools/gsm8k_train_en.jsonl")
eval_ = read_jsonl(f"{BASE}/gsm8k_raw/gsm8k_main_train_700_ur.jsonl")
write_jsonl(pool,  f"{OUT}/gsm8k_pool.jsonl")
write_jsonl(eval_, f"{OUT}/gsm8k_eval.jsonl")

# ── BoolQ ────────────────────────────────────────────────────────────────────
print("\n[BoolQ]")
en = read_jsonl(f"{BASE}/boolq_raw/boolq_train_1550_en.jsonl")
ur = read_jsonl(f"{BASE}/boolq_raw/boolq_train_1550_ur_fixed.jsonl")
pool_en, _ = split_80_20(en)
_, eval_ur  = split_80_20(ur)
write_jsonl(pool_en, f"{OUT}/boolq_pool.jsonl")
write_jsonl(eval_ur, f"{OUT}/boolq_eval.jsonl")

# ── CSQA ─────────────────────────────────────────────────────────────────────
print("\n[CSQA]")
en = read_jsonl(f"{BASE}/csqa_raw/csqa_train_1500_en.jsonl")
ur = read_jsonl(f"{BASE}/csqa_raw/csqa_train_1500_ur.jsonl")
pool_en, _ = split_80_20(en)
_, eval_ur  = split_80_20(ur)
write_jsonl(pool_en, f"{OUT}/csqa_pool.jsonl")
write_jsonl(eval_ur, f"{OUT}/csqa_eval.jsonl")

# ── PIQA ─────────────────────────────────────────────────────────────────────
print("\n[PIQA]")
en = read_jsonl(f"{BASE}/piqa_raw/piqa_train_750.jsonl")
ur = read_jsonl(f"{BASE}/piqa_raw/piqa_train_750_ur_fixed.jsonl")
pool_en, _ = split_80_20(en)
_, eval_ur  = split_80_20(ur)
write_jsonl(pool_en, f"{OUT}/piqa_pool.jsonl")
write_jsonl(eval_ur, f"{OUT}/piqa_eval.jsonl")

# ── StrategyQA ───────────────────────────────────────────────────────────────
print("\n[StrategyQA]")
# English source is a JSON array, not JSONL
with open(f"{BASE}/strategyqa_raw/strategyQA_train.json", encoding="utf-8") as f:
    en = json.load(f)
ur = read_jsonl(f"{BASE}/strategyqa_raw/strategyQA_train_ur2_norm.jsonl")
pool_en, _ = split_80_20(en)
_, eval_ur  = split_80_20(ur)
write_jsonl(pool_en, f"{OUT}/strategyqa_pool.jsonl")
write_jsonl(eval_ur, f"{OUT}/strategyqa_eval.jsonl")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n[Done] sdfr_splits contents:")
for f in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, f)
    with open(path, encoding="utf-8") as fh:
        n = sum(1 for l in fh if l.strip())
    print(f"  {f:<35} {n} lines")
