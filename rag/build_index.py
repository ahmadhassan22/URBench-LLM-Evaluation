import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────
CHUNKS_FILE  = os.path.expanduser("~/URBench/rag/chunks/filtered_chunks.jsonl")
INDEX_DIR    = os.path.expanduser("~/URBench/rag/index/")
MODEL_NAME   = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE   = 256
# ───────────────────────────────────────────────────────────

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    # load chunks
    print("Loading chunks...")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Total chunks: {len(chunks)}")

    # prepare texts for embedding
    # e5 models need "passage: " prefix for chunks
    texts = ["passage: " + c["title"] + " " + c["text"] for c in chunks]

    # load embedding model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    # embed all chunks in batches
    print(f"\nEmbedding {len(texts)} chunks (batch size {BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        device="cpu"
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # build FAISS index
    print("\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity (since normalized)
    index.add(embeddings.astype(np.float32))
    print(f"Index built. Total vectors: {index.ntotal}")

    # save index and chunks
    print("\nSaving index and chunks...")
    faiss.write_index(index, os.path.join(INDEX_DIR, "wikipedia.index"))

    with open(os.path.join(INDEX_DIR, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n✅ Done!")
    print(f"Index saved to : {INDEX_DIR}wikipedia.index")
    print(f"Chunks saved to: {INDEX_DIR}chunks.jsonl")

if __name__ == "__main__":
    main()
