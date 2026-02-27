import os
import json
import time
import re
from tqdm import tqdm
from openai import OpenAI

# Detect English letters
EN_LETTERS = re.compile(r"[A-Za-z]")

# Normalize Urdu digits → Western digits
DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

EN_PATH = r"C:\Users\Administrator\Documents\URBench\data\piqa_raw\piqa_train_750.jsonl"
UR_PATH = r"C:\Users\Administrator\Documents\URBench\data\piqa_raw\piqa_train_750_ur.jsonl"
OUT_PATH = r"C:\Users\Administrator\Documents\URBench\data\piqa_raw\piqa_train_750_ur_fixed.jsonl"


# -------------------------
# SAFE JSONL READER
# -------------------------
def read_jsonl_safe(path):
    """
    Reads a JSONL file safely.
    Malformed lines are replaced with a placeholder so indexing is preserved.
    """
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
                # keep alignment by inserting placeholder
                items.append({"_urbench_error": "BROKEN_JSON_LINE"})
                bad_lines.append(idx)

    if bad_lines:
        print(
            f"[warn] Found {len(bad_lines)} malformed/empty lines. "
            f"First few at lines: {bad_lines[:10]}"
        )

    return items


def has_english(obj):
    for k in ("goal", "sol1", "sol2"):
        v = obj.get(k, "")
        if isinstance(v, str) and EN_LETTERS.search(v):
            return True
    return False


def is_bad(obj):
    return ("_urbench_error" in obj) or has_english(obj)


def safe_parse_json(text: str):
    """
    Removes common wrappers and parses JSON safely.
    """
    t = text.strip()

    # Remove fenced code blocks if model returns them
    if t.startswith("```"):
        t = t.strip("`").strip()
        # sometimes starts with 'json'
        if t.lower().startswith("json"):
            t = t[4:].strip()

    return json.loads(t)


def main():
    en_items = read_jsonl_safe(EN_PATH)
    ur_items = read_jsonl_safe(UR_PATH)

    # -------------------------
    # ALIGN LENGTHS (CRITICAL FIX)
    # -------------------------
    if len(ur_items) < len(en_items):
        missing = len(en_items) - len(ur_items)
        print(f"[fix] UR file shorter than EN by {missing} line(s). Padding placeholders.")
        for _ in range(missing):
            ur_items.append({"_urbench_error": "MISSING_LINE_PLACEHOLDER"})

    elif len(ur_items) > len(en_items):
        extra = len(ur_items) - len(en_items)
        print(f"[fix] UR file longer than EN by {extra} line(s). Truncating extras.")
        ur_items = ur_items[:len(en_items)]
    # -------------------------

    client = OpenAI()

    instructions = (
        "Translate the JSON values from English to Urdu.\n"
        "Rules:\n"
        "- Output MUST be a single valid JSON object only (no markdown, no extra text).\n"
        "- DO NOT change keys or JSON structure.\n"
        "- Keep 'label' value EXACTLY unchanged.\n"
        "- Translate ONLY: goal, sol1, sol2.\n"
        "- Urdu output MUST NOT contain any English letters a-z or A-Z.\n"
        "- Transliterate technical terms into Urdu script if needed.\n"
        "- Keep all numbers as Western digits (0-9).\n"
        "- Do NOT shorten, simplify, or explain.\n"
        "- Output keys must be exactly: goal, sol1, sol2, label.\n"
        "- If you cannot comply, still output valid JSON with Urdu text (no English letters).\n"
    )

    fixed = []
    bad_count = 0
    repaired = 0

    for i, (en_obj, ur_obj) in enumerate(
        tqdm(list(zip(en_items, ur_items)), desc="Repairing PIQA errors"),
        start=1
    ):
        if not is_bad(ur_obj):
            ur_obj.pop("_urbench_error", None)
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
                    max_output_tokens=1000,
                )

                out_obj = safe_parse_json(resp.output_text)

                # HARD GUARDS
                out_obj["label"] = en_obj["label"]
                for k in ("goal", "sol1", "sol2"):
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