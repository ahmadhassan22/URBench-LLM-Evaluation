import os
import json
import random
from datasets import load_dataset

OUT_JSONL = r"C:\Users\Administrator\Documents\URBench\data\gsm8k_raw\gsm8k_main_train_700.jsonl"
SEED = 42
N = 700

def main():
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"]

    random.seed(SEED)
    idxs = random.sample(range(len(train)), N)

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i in idxs:
            ex = train[i]
            # ✅ Keep same schema as dataset (question, answer)
            obj = {
                "question": ex["question"],
                "answer": ex["answer"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {N} items to: {OUT_JSONL}")

if __name__ == "__main__":
    main()