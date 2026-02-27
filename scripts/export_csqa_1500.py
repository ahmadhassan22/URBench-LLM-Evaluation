import os
import json
import argparse
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument(
        "--output",
        default=r"C:\Users\Administrator\Documents\URBench\data\csqa_raw\csqa_train_1500_en.jsonl",
    )
    args = ap.parse_args()

    ds = load_dataset("tau/commonsense_qa", split="train")  # official HF repo
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= args.n:
                break

            obj = {
                "qid": f"CSQA_{i:04d}",
                "id": ex["id"],
                "question": ex["question"],
                "question_concept": ex["question_concept"],
                "choices": {
                    "label": ex["choices"]["label"],   # keep as-is
                    "text": ex["choices"]["text"],     # translate these
                },
                "answerKey": ex["answerKey"],         # keep as-is (A/B/C/D/E)
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Exported {args.n} items to: {args.output}")

if __name__ == "__main__":
    main()