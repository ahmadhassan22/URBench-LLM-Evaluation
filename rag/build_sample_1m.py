"""Build a 1M-chunk validation sample that GUARANTEES inclusion of chunks
containing known coverage_gap test entities, plus random fill. This lets us
test whether correct indexing actually retrieves the facts."""
import json, random

FULL   = "/mnt/home/user41/URBench/rag/chunks/wikipedia_chunks.jsonl"
OUT    = "/mnt/home/user41/URBench/rag/chunks/sample_1m.jsonl"
TARGET = 1_000_000

# entities from the coverage_gap failure cases we inspected
MUST_INCLUDE = ["grey seal","pound sterling","genghis khan","julius caesar",
                "casio","petco","mount fuji","boeing","wonder woman",
                "dustin hoffman","dragon ball"]

kept, seen = [], 0
random.seed(0)
with open(FULL, encoding="utf-8") as f:
    for line in f:
        seen += 1
        low = line.lower()
        # always keep chunks mentioning a must-include entity
        if any(e in low for e in MUST_INCLUDE):
            kept.append(line)
        elif random.random() < 0.04:   # ~4% random fill
            kept.append(line)
        if len(kept) >= TARGET:
            break

with open(OUT, "w", encoding="utf-8") as f:
    f.writelines(kept)
print(f"Scanned {seen} lines, wrote {len(kept)} chunks to {OUT}")
