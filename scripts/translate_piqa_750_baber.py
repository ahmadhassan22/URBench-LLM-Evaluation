import os
import json
import time
import argparse
import re
from tqdm import tqdm
from openai import OpenAI

# forbid English letters in Urdu text
EN_LETTERS = re.compile(r"[A-Za-z]")

# normalize Urdu digits → Western digits
DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def to_jsonl_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def count_lines(path: str) -> int:
    """Count total non-empty lines in an existing JSONL file."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=r"C:\Users\Administrator\Documents\URBench\data\piqa_raw\piqa_train_750.jsonl",
    )
    ap.add_argument(
        "--output",
        default=r"C:\Users\Administrator\Documents\URBench\data\piqa_raw\piqa_train_750_ur.jsonl",
    )
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=800)
    args = ap.parse_args()

    client = OpenAI()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ✅ Resume by TOTAL output line count (prevents duplicates)
    done_lines = 0
    if args.resume and os.path.exists(args.output):
        done_lines = count_lines(args.output)

    print(f"[resume] output already has {done_lines} lines. Resuming from item index {done_lines}.")

    instructions = (
        "Translate the JSON values from English to Urdu.\n"
        "Rules:\n"
        "- DO NOT change keys or JSON structure.\n"
        "- Keep 'label' value EXACTLY unchanged.\n"
        "- Translate ONLY these fields: goal, sol1, sol2.\n"
        "- The translated Urdu text MUST NOT contain any English letters a-z or A-Z.\n"
        "- Keep the translated Urdu text length the SAME as the English text, "
        "without any reduction in length, quality, or meaning.\n"
        "- Keep all numbers as Western digits (0-9). Do not write ۰۱۲...\n"
        "- Use pure Urdu only (no English or Roman words).\n"
        "- Do NOT simplify, summarize, or shorten.\n"
        "- Output valid JSON only, same keys: goal, sol1, sol2, label.\n"
    )

    # ✅ append mode: never overwrites old lines
    with open(args.output, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(iter_jsonl(args.input), desc="Translating PIQA → Urdu")):
            # skip items already written (by total lines)
            if idx < done_lines:
                continue

            if not isinstance(item, dict):
                continue

            for attempt in range(6):
                try:
                    resp = client.responses.create(
                        model=args.model,
                        instructions=instructions,
                        input=json.dumps(item, ensure_ascii=False),
                        temperature=0,
                        max_output_tokens=args.max_output_tokens,
                    )

                    out_obj = json.loads(resp.output_text)

                    # HARD GUARDS
                    out_obj["label"] = item["label"]
                    for k in ("goal", "sol1", "sol2"):
                        if not isinstance(out_obj.get(k), str):
                            raise ValueError(f"Missing field {k}")
                        out_obj[k] = out_obj[k].translate(DIGIT_MAP)
                        if EN_LETTERS.search(out_obj[k]):
                            raise ValueError(f"English detected in {k}")

                    out_f.write(to_jsonl_line(out_obj))
                    out_f.flush()
                    break

                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    if attempt == 5:
                        fail = dict(item)
                        fail["_urbench_error"] = str(e)[:500]
                        out_f.write(to_jsonl_line(fail))
                        out_f.flush()
                        break
                    time.sleep(wait)

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("✅ PIQA translation completed.")

if __name__ == "__main__":
    main()