"""Embed the 1M sample on GPU (no prefix — MiniLM doesn't use one), build flat
index, then TEST retrieval on known coverage_gap failure questions."""
import json, os, numpy as np, faiss, time
from sentence_transformers import SentenceTransformer

CHUNKS = "/mnt/home/user41/URBench/rag/chunks/sample_1m.jsonl"
INDEX  = "/mnt/home/user41/URBench/rag/index/sample_1m.index"
META   = "/mnt/home/user41/URBench/rag/index/sample_1m_meta.jsonl"
MODEL  = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH  = 512

def main():
    print("Loading chunks..."); t=time.time()
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    print(f"{len(chunks)} chunks in {time.time()-t:.0f}s")

    # NO prefix — must match how queries are embedded in the SDFR/eval scripts
    texts = [c["title"] + ". " + c["text"] for c in chunks]

    print("Loading model on GPU...")
    model = SentenceTransformer(MODEL, device="cuda")

    print(f"Embedding {len(texts)} on GPU (batch {BATCH})...")
    t=time.time()
    emb = model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                       normalize_embeddings=True, device="cuda",
                       convert_to_numpy=True)
    print(f"Embedded in {(time.time()-t)/60:.1f} min | shape {emb.shape}")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))
    faiss.write_index(index, INDEX)
    with open(META, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"title":c["title"],"text":c["text"]}, ensure_ascii=False)+"\n")
    print(f"Index built: {index.ntotal} vectors")

    # ---- RETRIEVAL TEST on known coverage_gap failures ----
    test_qs = [
        "Would a dog respond to bell before Grey seal?",
        "Is a pound sterling valuable?",
        "Are more people today related to Genghis Khan than Julius Caesar?",
        "Can you buy Casio products at Petco?",
    ]
    print("\n=== RETRIEVAL TEST (does correct indexing find the facts?) ===")
    meta = chunks
    for q in test_qs:
        qv = model.encode([q], normalize_embeddings=True, device="cuda", convert_to_numpy=True)
        sims, ids = index.search(qv, 3)
        print(f"\nQ: {q}")
        for s, i in zip(sims[0], ids[0]):
            print(f"  [{s:.3f}] {meta[i]['title']}: {meta[i]['text'][:90]}")

if __name__ == "__main__":
    main()
