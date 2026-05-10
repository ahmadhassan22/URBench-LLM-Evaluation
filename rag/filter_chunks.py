import json
import os
import re
from tqdm import tqdm

STRATEGYQA = os.path.expanduser("~/URBench/data/strategyqa_raw/strategyQA_train.json")
CHUNKS_IN  = os.path.expanduser("~/URBench/rag/chunks/wikipedia_chunks.jsonl")
CHUNKS_OUT = os.path.expanduser("~/URBench/rag/chunks/filtered_chunks.jsonl")

def extract_entities(strategyqa_path):
    with open(strategyqa_path) as f:
        data = json.load(f)
    entities = set()
    for item in data:
        for fact in item.get("facts", []):
            matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', fact)
            for m in matches:
                if len(m) >= 4:
                    entities.add(m.lower())
    return entities

def main():
    print("Extracting entities from StrategyQA facts...")
    entities = extract_entities(STRATEGYQA)
    print(f"Total entities: {len(entities)}")

    print("Filtering chunks...")
    kept = 0
    skipped = 0

    with open(CHUNKS_IN, "r", encoding="utf-8") as in_f, \
         open(CHUNKS_OUT, "w", encoding="utf-8") as out_f:

        for line in tqdm(in_f, desc="Filtering", total=23963971):
            chunk = json.loads(line)
            title = chunk["title"].lower()

            if title in entities:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"\nDone!")
    print(f"Kept chunks   : {kept:,}")
    print(f"Skipped chunks: {skipped:,}")

if __name__ == "__main__":
    main()
