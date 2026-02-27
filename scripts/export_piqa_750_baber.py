import os
import json
import random
from datasets import load_dataset

OUT_JSONL = r"C:\Users\Administrator\Documents\URBench\data\piqa_raw\piqa_train_750.jsonl"
SEED = 42
N = 750

def main():
    ds = load_dataset("baber/piqa")  # ✅ script-free parquet repo
    train = ds["train"]

    random.seed(SEED)
    idxs = random.sample(range(len(train)), N)

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i in idxs:
            ex = train[i]
            obj = {
                "goal": ex["goal"],
                "sol1": ex["sol1"],
                "sol2": ex["sol2"],
                "label": ex["label"],  # 0 = sol1 correct, 1 = sol2 correct
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {N} items to: {OUT_JSONL}")

if __name__ == "__main__":
    main()