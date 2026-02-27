import os
import json
import time
import argparse
import re
from tqdm import tqdm
from openai import OpenAI

EN_LETTERS = re.compile(r"[A-Za-z]")
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
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def safe_json_load(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=r"C:\Users\Administrator\Documents\URBench\data\boolq_raw\boolq_train_1550_en.jsonl",
    )
    ap.add_argument(
        "--output",
        default=r"C:\Users\Administrator\Documents\URBench\data\boolq_raw\boolq_train_1550_ur.jsonl",
    )
    ap.add_argument("--model", default="gpt-4.1")  # or gpt-4.1-mini
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=1400)
    args = ap.parse_args()

    client = OpenAI()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    done_lines = 0
    if args.resume and os.path.exists(args.output):
        done_lines = count_lines(args.output)

    print(f"[resume] output already has {done_lines} lines. Resuming from item index {done_lines}.")

    instructions = (
    "Translate the JSON values from English to Urdu.\n"
    "Rules:\n"
    "- Output MUST be a single valid JSON object only (no markdown, no extra text).\n"
    "- DO NOT change keys or JSON structure.\n"
    "- Keep 'qid' EXACTLY unchanged.\n"
    "- Convert the boolean 'answer' value as follows:\n"
    "  - If answer is true, output exactly: ہاں\n"
    "  - If answer is false, output exactly: نہیں\n"
    "- Translate ONLY: 'question' and 'passage' into natural Urdu.\n"
    "- Proper nouns MUST be transliterated or written in their established Urdu names.\n"
    "- Urdu output MUST NOT contain any English letters a-z or A-Z.\n"
    "- Keep all numbers as Western digits (0-9), not ۰۱۲...\n"
    "- Keep the translated Urdu text length the SAME as the English text, "
    "without any reduction in length, quality, or meaning.\n"
    "- Do NOT shorten, simplify, summarize, or explain.\n"
    "- Output keys must be exactly: qid, question, passage, answer.\n"
    )

    with open(args.output, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(iter_jsonl(args.input), desc="Translating BoolQ → Urdu")):
            if idx < done_lines:
                continue

            # hard guards
            if not isinstance(item, dict) or "qid" not in item or "question" not in item or "passage" not in item or "answer" not in item:
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

                    out_obj = safe_json_load(resp.output_text)

                    # HARD GUARDS: keep qid + answer unchanged
                    out_obj["qid"] = item["qid"]
                    out_obj["answer"] = "ہاں" if item["answer"] is True else "نہیں"

                    # normalize digits + enforce no English letters
                    for k in ("question", "passage"):
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

    print("✅ BoolQ translation completed.")

if __name__ == "__main__":
    main()