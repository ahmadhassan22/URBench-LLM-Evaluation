import json
from pathlib import Path

input_file = Path("C:/Users/Administrator/Documents/URBench/data/strategyqa_raw/strategyQA_train_ur.jsonl")
output_file = Path("C:/Users/Administrator/Documents/URBench/data/strategyqa_raw/strategyQA_train_ur2.jsonl")

print("Input file exists:", input_file.exists())
print("Input file path:", input_file.resolve())

count = 0

with input_file.open("r", encoding="utf-8") as fin, \
     output_file.open("w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        if obj.get("answer") is True:
            obj["answer"] = "ہاں"
        elif obj.get("answer") is False:
            obj["answer"] = "نہیں"

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        count += 1

print(f"Done. Processed {count} lines.")
