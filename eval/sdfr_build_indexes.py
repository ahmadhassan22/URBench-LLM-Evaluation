"""
SDFR-UR Step 2: Build FAISS indexes for all 5 retrieval pools.
Encodes the 'question' field of each pool using paraphrase-multilingual-MiniLM-L12-v2.
Output goes to ~/URBench/data/sdfr_indexes/
Runs on CPU — no GPU needed. Expected time: 5-10 minutes total.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE   = os.path.expanduser("~/URBench/data")
SPLITS = os.path.join(BASE, "sdfr_splits")
OUT    = os.path.join(BASE, "sdfr_indexes")
os.makedirs(OUT, exist_ok=True)

MODEL_NAME = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.\n")

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def get_question(item, dataset):
    """Extract the question text from a pool item depending on dataset format."""
    if dataset == "piqa":
        return item["goal"]
    elif dataset == "strategyqa":
        return item.get("question", item.get("input", ""))
    else:
        return item["question"]

def build_index(dataset):
    pool_path = os.path.join(SPLITS, f"{dataset}_pool.jsonl")
    pool = read_jsonl(pool_path)

    texts = [get_question(item, dataset) for item in pool]
    print(f"[{dataset.upper()}] Encoding {len(texts)} examples...", flush=True)

    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)

    # Save embeddings as numpy array
    emb_path = os.path.join(OUT, f"{dataset}_embeddings.npy")
    np.save(emb_path, embeddings)

    # Build FAISS index (inner product = cosine similarity since embeddings are normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    idx_path = os.path.join(OUT, f"{dataset}_faiss.index")
    faiss.write_index(index, idx_path)

    print(f"  embeddings → {emb_path}")
    print(f"  faiss index → {idx_path}  ({index.ntotal} vectors, dim={dim})\n")

for ds in ["gsm8k", "boolq", "csqa", "piqa", "strategyqa"]:
    build_index(ds)

print("[Done] All indexes built.")
print("Contents of sdfr_indexes/:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f:<45} {size/1024/1024:.1f} MB")
