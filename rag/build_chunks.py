import glob
import datasets
import json
import os
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────
CHUNK_SIZE = 200      # words per chunk
OVERLAP    = 50       # overlap between chunks
OUTPUT     = os.path.expanduser("~/URBench/rag/chunks/wikipedia_chunks.jsonl")
# ──────────────────────────────────────────────────────────

def chunk_text(title, text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append({
            "title": title,
            "text": chunk
        })
        start += chunk_size - overlap
    return chunks

def main():
    print("Loading parquet files...")
    files = glob.glob(os.path.expanduser(
        "~/.cache/modelscope/hub/datasets/downloads/[0-9a-f]*"))
    files = sorted([f for f in files
                    if not f.endswith('.json') and not f.endswith('.lock')])
    print(f"Found {len(files)} parquet files")

    ds = datasets.load_dataset(
        'parquet', data_files={'train': files}, split='train')
    print(f"Total articles: {len(ds)}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    total_chunks = 0
    with open(OUTPUT, "w", encoding="utf-8") as out_f:
        for article in tqdm(ds, desc="Chunking articles"):
            title = article["title"]
            text  = article["text"]
            if not text or not text.strip():
                continue
            for chunk in chunk_text(title, text):
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Done! Total chunks written: {total_chunks}")
    print(f"Saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
