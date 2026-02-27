import os, json, time, argparse, re
from tqdm import tqdm
from openai import OpenAI

EN_LETTERS = re.compile(r"[A-Za-z]")
DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

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
    ap.add_argument("--input", default=r"C:\Users\Administrator\Documents\URBench\data\csqa_raw\csqa_train_1500_en.jsonl")
    ap.add_argument("--output", default=r"C:\Users\Administrator\Documents\URBench\data\csqa_raw\csqa_train_1500_ur.jsonl")
    ap.add_argument("--model", default="gpt-4.1")  # or gpt-4.1-mini
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=900)
    args = ap.parse_args()

    client = OpenAI()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    done = 0
    if args.resume and os.path.exists(args.output):
        done = count_lines(args.output)
    print(f"[resume] output has {done} lines. Resuming from index {done}.")

    # ✅ Minimal instructions (low token overhead)
    instructions = (
    "Translate JSON values to Urdu.\n"
    "Only translate: question, question_concept, choices.text.\n"
    "Do not change any keys, id, qid, choices.label, or answerKey.\n"
    "Ensure the Urdu question ending is grammatically and semantically aligned "
    "with the type of answer options (e.g., place, object, person, action), "
    "without adding new information or biasing toward any option.\n"
    "Use pure Urdu only (no English letters).\n"
    "Use Western digits (0-9).\n"
    "Output ONE valid JSON object only."
    )

    with open(args.output, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(iter_jsonl(args.input), desc="CSQA → Urdu")):
            if idx < done:
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

                    # HARD GUARDS: keep structure + correct answer/labels
                    out_obj["qid"] = item["qid"]
                    out_obj["id"] = item["id"]
                    out_obj["answerKey"] = item["answerKey"]
                    out_obj["choices"]["label"] = item["choices"]["label"]

                    # Normalize digits + block English letters in translated fields only
                    out_obj["question"] = out_obj["question"].translate(DIGIT_MAP)
                    out_obj["question_concept"] = out_obj["question_concept"].translate(DIGIT_MAP)

                    if EN_LETTERS.search(out_obj["question"]) or EN_LETTERS.search(out_obj["question_concept"]):
                        raise ValueError("English detected in question/question_concept")

                    new_texts = []
                    for t in out_obj["choices"]["text"]:
                        t = t.translate(DIGIT_MAP)
                        if EN_LETTERS.search(t):
                            raise ValueError("English detected in choices.text")
                        new_texts.append(t)
                    out_obj["choices"]["text"] = new_texts

                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    out_f.flush()
                    break

                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    if attempt == 5:
                        fail = dict(item)
                        fail["_urbench_error"] = str(e)[:200]
                        out_f.write(json.dumps(fail, ensure_ascii=False) + "\n")
                        out_f.flush()
                        break
                    time.sleep(wait)

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("✅ CSQA translation completed.")

if __name__ == "__main__":
    main()