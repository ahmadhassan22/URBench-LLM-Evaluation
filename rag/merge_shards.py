"""Merge already-embedded shards into final flat index. Leaner memory handling
with explicit cleanup + progress + memory logging."""
import numpy as np, faiss, glob, gc, os, psutil

SHARD_DIR = "/mnt/home/user41/URBench/rag/index/shards"
INDEX     = "/mnt/home/user41/URBench/rag/index/wikipedia_full.index"

def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

shards = sorted(glob.glob(f"{SHARD_DIR}/shard_*.npy"))
print(f"Found {len(shards)} shards")

dim = np.load(shards[0]).shape[1]
index = faiss.IndexFlatIP(dim)

for i, sp in enumerate(shards):
    emb = np.load(sp)
    index.add(emb)
    del emb
    gc.collect()
    print(f"[{i+1}/{len(shards)}] added {sp} -> total {index.ntotal} | RSS {mem_gb():.1f} GB", flush=True)

faiss.write_index(index, INDEX)
print(f"DONE. Index: {INDEX} | {index.ntotal} vectors")
