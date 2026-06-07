"""
Build BoolQ large pool FAISS index using PASSAGE embeddings
(not question embeddings — passage is the signal for BoolQ retrieval)
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

POOL_PATH  = '/mnt/home/user41/URBench/data/sdfr_splits/boolq_pool_large_clean.jsonl'
EMBED_PATH = '/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
OUT_EMB    = '/mnt/home/user41/URBench/data/sdfr_indexes/boolq_large_passage_embeddings.npy'
OUT_IDX    = '/mnt/home/user41/URBench/data/sdfr_indexes/boolq_large_passage_faiss.index'

pool = [json.loads(l) for l in open(POOL_PATH) if l.strip()]
print(f"Pool size: {len(pool)}")

embedder = SentenceTransformer(EMBED_PATH)

# Encode PASSAGES (not questions) — truncate to 300 chars for efficiency
texts = [item['passage'][:300] for item in pool]

print("Encoding passages...")
embeddings = embedder.encode(texts, batch_size=128, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True)
np.save(OUT_EMB, embeddings)

dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, OUT_IDX)

print(f"Index built: {index.ntotal} vectors, dim={dim}")
print(f"Embeddings → {OUT_EMB}")
print(f"Index      → {OUT_IDX}")
