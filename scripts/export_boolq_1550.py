import os
import json
import argparse
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1550)
    ap.add_argument(
        "--output",
        default=r"C:\Users\Administrator\Documents\URBench\data\boolq_raw\boolq_train_1550_en.jsonl",
    )
    args = ap.parse_args()

    ds = load_dataset("google/boolq", split="train")  # official HF dataset

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # BoolQ fields: question, passage, answer (bool)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= args.n:
                break
            obj = {
                "qid": f"BOOLQ_{i:04d}",
                "question": ex["question"],
                "passage": ex["passage"],
                "answer": bool(ex["answer"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Exported {args.n} items to: {args.output}")

if __name__ == "__main__":
    main()