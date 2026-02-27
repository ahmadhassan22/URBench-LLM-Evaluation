import os
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI

# Ensure western digits 0-9 in Urdu output (optional safety)
DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def to_jsonl_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"C:\Users\Administrator\Documents\URBench\data\gsm8k_raw\gsm8k_main_train_700.jsonl")
    ap.add_argument("--output", default=r"C:\Users\Administrator\Documents\URBench\data\gsm8k_raw\gsm8k_main_train_700_ur.jsonl")
    ap.add_argument("--model", default="gpt-4.1")  # or "gpt-4.1-mini"
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=600)
    args = ap.parse_args()

    client = OpenAI()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # resume by line count (since GSM8K has no qid)
    done_lines = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            done_lines = sum(1 for _ in f)

    instructions = (
            "Translate the JSON values from English to Urdu.\n"
            "Rules:\n"
            "- DO NOT change keys or JSON structure.\n"
            "- Keep 'answer' value EXACTLY unchanged.\n"
            "- Translate ONLY the value of 'question' into natural Urdu.\n"
            "- Keep the translated Urdu text length the SAME as the English text, "
            "without any reduction in length, quality, or meaning.\n"
            "- Keep all numbers as Western digits (0-9). Do not write ۰۱۲...\n"
            "- Use pure Urdu (no English or Roman words).\n"
            "- Do NOT simplify, summarize, or shorten the question.\n"
            "- Output valid JSON only, with the exact same keys: question, answer.\n"
    )

    with open(args.output, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(iter_jsonl(args.input), desc="Translating GSM8K(main)")):
            if idx < done_lines:
                continue

            # hard guard
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
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

                    # ✅ HARD GUARDS: ensure schema + answer unchanged
                    out_obj["answer"] = item["answer"]
                    out_obj["question"] = out_obj["question"].translate(DIGIT_MAP)

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

    print("Done.")

if __name__ == "__main__":
    main()