import json

# Mapping: Eastern Arabic–Indic → Western Arabic
DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹",
    "0123456789"
)

input_file = r"C:\Users\Administrator\Documents\URBench\data\strategyqa_raw\strategyQA_train_ur2.jsonl"
output_file = r"C:\Users\Administrator\Documents\URBench\data\strategyqa_raw\strategyQA_train_ur2_norm.jsonl"
with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        # Convert digits in all string values
        def normalize(value):
            if isinstance(value, str):
                return value.translate(DIGIT_MAP)
            elif isinstance(value, list):
                return [normalize(v) for v in value]
            elif isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            else:
                return value

        obj = normalize(obj)

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Done: Eastern Arabic–Indic digits converted to Western Arabic numerals.")