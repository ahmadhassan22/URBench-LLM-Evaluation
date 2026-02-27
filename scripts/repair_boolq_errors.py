import os
import json
import time
import re
from tqdm import tqdm
from openai import OpenAI

EN_LETTERS = re.compile(r"[A-Za-z]")
DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

EN_PATH = r"C:\Users\Administrator\Documents\URBench\data\boolq_raw\boolq_train_1550_en.jsonl"
UR_PATH = r"C:\Users\Administrator\Documents\URBench\data\boolq_raw\boolq_train_1550_ur.jsonl"
OUT_PATH = r"C:\Users\Administrator\Documents\URBench\data\boolq_raw\boolq_train_1550_ur_fixed.jsonl"


def read_jsonl_safe(path):
    items = []
    bad_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                items.append({"_urbench_error": "EMPTY_LINE"})
                bad_lines.append(idx)
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                items.append({"_urbench_error": "BROKEN_JSON_LINE"})
                bad_lines.append(idx)

    if bad_lines:
        print(f"[warn] Found {len(bad_lines)} malformed/empty lines. First few: {bad_lines[:10]}")
    return items


def has_english(obj):
    # BoolQ: check question + passage
    for k in ("question", "passage"):
        v = obj.get(k, "")
        if isinstance(v, str) and EN_LETTERS.search(v):
            return True
    return False


def is_bad(obj):
    return ("_urbench_error" in obj) or has_english(obj)


def safe_parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def main():
    en_items = read_jsonl_safe(EN_PATH)
    ur_items = read_jsonl_safe(UR_PATH)

    # Align lengths if interrupted writes happened
    if len(ur_items) < len(en_items):
        missing = len(en_items) - len(ur_items)
        print(f"[fix] UR shorter than EN by {missing}. Padding placeholders.")
        for _ in range(missing):
            ur_items.append({"_urbench_error": "MISSING_LINE_PLACEHOLDER"})
    elif len(ur_items) > len(en_items):
        extra = len(ur_items) - len(en_items)
        print(f"[fix] UR longer than EN by {extra}. Truncating extras.")
        ur_items = ur_items[:len(en_items)]

    client = OpenAI()

    instructions = (
        "Translate the JSON values from English to Urdu.\n"
        "Rules:\n"
        "- Output MUST be a single valid JSON object only (no markdown, no extra text).\n"
        "- DO NOT change keys or JSON structure.\n"
        "- Keep 'qid' EXACTLY unchanged.\n"
        "- Convert boolean 'answer' strictly:\n"
        "  - true => ہاں\n"
        "  - false => نہیں\n"
        "- Translate ONLY: question, passage.\n"
        "- Urdu output MUST NOT contain any English letters a-z or A-Z.\n"
        "- IMPORTANT: Any abbreviations/symbols written in Latin letters MUST be transliterated into Urdu script.\n"
        "  Examples: Na => این اے, VGSCs => وی جی ایس سیز, Nav => این اے وی, DNA => ڈی این اے.\n"
        "- Proper nouns MUST be transliterated or written in their established Urdu names.\n"
        "- Keep all numbers as Western digits (0-9), not ۰۱۲...\n"
        "- Do NOT shorten, simplify, summarize, or explain.\n"
        "- Output keys must be exactly: qid, question, passage, answer.\n"
    )

    fixed = []
    bad_count = 0
    repaired = 0

    for en_obj, ur_obj in tqdm(list(zip(en_items, ur_items)), desc="Repairing BoolQ errors"):
        if not is_bad(ur_obj):
            ur_obj.pop("_urbench_error", None)
            # enforce answer mapping anyway (in case older file still has true/false)
            ur_obj["answer"] = "ہاں" if bool(en_obj["answer"]) else "نہیں"
            fixed.append(ur_obj)
            continue

        bad_count += 1

        for attempt in range(8):
            try:
                resp = client.responses.create(
                    model="gpt-4.1",
                    instructions=instructions,
                    input=json.dumps(en_obj, ensure_ascii=False),
                    temperature=0,
                    max_output_tokens=1600,
                )

                out_obj = safe_parse_json(resp.output_text)

                # HARD GUARDS
                out_obj["qid"] = en_obj["qid"]
                out_obj["answer"] = "ہاں" if bool(en_obj["answer"]) else "نہیں"

                for k in ("question", "passage"):
                    if not isinstance(out_obj.get(k), str):
                        raise ValueError(f"Missing field {k}")
                    out_obj[k] = out_obj[k].translate(DIGIT_MAP)
                    if EN_LETTERS.search(out_obj[k]):
                        raise ValueError(f"English detected in {k}")

                out_obj.pop("_urbench_error", None)
                fixed.append(out_obj)
                repaired += 1
                break

            except Exception as e:
                if attempt == 7:
                    keep = dict(ur_obj)
                    keep["_urbench_error"] = f"REPAIR_FAILED: {str(e)[:200]}"
                    fixed.append(keep)
                time.sleep(min(2 ** attempt, 30))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for obj in fixed:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("====================================")
    print(f"Total items           : {len(en_items)}")
    print(f"Bad items detected    : {bad_count}")
    print(f"Successfully repaired : {repaired}")
    print(f"Output written to     : {OUT_PATH}")
    print("====================================")


if __name__ == "__main__":
    main()