"""Full 25G / 23.9M-chunk index build. GPU, no prefix, flat IP index.
Sharded + checkpointed: embeds in shards, saves each shard to disk, so a crash
resumes from the last completed shard instead of restarting."""
import json, os, numpy as np, faiss, time, glob
from sentence_transformers import SentenceTransformer

CHUNKS   = "/mnt/home/user41/URBench/rag/chunks/wikipedia_chunks.jsonl"
SHARD_DIR= "/mnt/home/user41/URBench/rag/index/shards"
INDEX    = "/mnt/home/user41/URBench/rag/index/wikipedia_full.index"
META     = "/mnt/home/user41/URBench/rag/index/wikipedia_full_meta.jsonl"
MODEL    = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH    = 512
SHARD    = 500_000          # chunks per shard
os.makedirs(SHARD_DIR, exist_ok=True)

def main():
    model = SentenceTransformer(MODEL, device="cuda")
    print("Model on GPU. Streaming chunks...")

    buf, shard_id, total = [], 0, 0
    meta_f = open(META, "w", encoding="utf-8")

    def flush(buf, shard_id):
        shard_path = f"{SHARD_DIR}/shard_{shard_id:04d}.npy"
        if os.path.exists(shard_path):
            print(f"  shard {shard_id} exists, skipping embed"); return
        texts = [c["title"] + ". " + c["text"] for c in buf]
        t = time.time()
        emb = model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                           device="cuda", convert_to_numpy=True, show_progress_bar=False)
        np.save(shard_path, emb.astype(np.float32))
        print(f"  shard {shard_id}: {len(buf)} chunks in {(time.time()-t)/60:.1f} min -> {shard_path}", flush=True)

    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            buf.append(c)
            meta_f.write(json.dumps({"title":c["title"],"text":c["text"]}, ensure_ascii=False)+"\n")
            if len(buf) >= SHARD:
                flush(buf, shard_id); total += len(buf); shard_id += 1; buf = []
        if buf:
            flush(buf, shard_id); total += len(buf); shard_id += 1
    meta_f.close()
    print(f"All shards embedded. Total chunks: {total}")

    # merge shards into one flat index
    print("Building flat index from shards...")
    dim = np.load(f"{SHARD_DIR}/shard_0000.npy").shape[1]
    index = faiss.IndexFlatIP(dim)
    for sp in sorted(glob.glob(f"{SHARD_DIR}/shard_*.npy")):
        emb = np.load(sp)
        index.add(emb)
        print(f"  added {sp} -> total {index.ntotal}", flush=True)
    faiss.write_index(index, INDEX)
    print(f"DONE. Index: {INDEX} | {index.ntotal} vectors")

if __name__ == "__main__":
    main()
