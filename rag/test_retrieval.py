import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── CONFIG ────────────────────────────────────────────────
INDEX_PATH  = os.path.expanduser("~/URBench/rag/index/wikipedia.index")
CHUNKS_PATH = os.path.expanduser("~/URBench/rag/index/chunks.jsonl")
MODEL_PATH  = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 3
# ───────────────────────────────────────────────────────────

def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def retrieve(question, model, index, chunks, top_k=TOP_K):
    # e5 needs "query: " prefix for questions
    query = "query: " + question
    embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(embedding.astype(np.float32), top_k)
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_PATH)

    print("Loading index...")
    index = faiss.read_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Index ready. {index.ntotal} vectors.\n")

    # test with 5 real StrategyQA questions in English
    test_questions = [
        "Was Aristotle born before the Roman Empire?",
        "Are more people today related to Genghis Khan than Julius Caesar?",
        "Could the members of The Police perform lawful arrests?",
        "Is the Nile longer than the Amazon River?",
        "Did Einstein win a Nobel Prize before World War 2?",
    ]

    for q in test_questions:
        print(f"Question: {q}")
        results = retrieve(q, model, index, chunks)
        for i, r in enumerate(results):
            print(f"  [{i+1}] Title: {r['title']}")
            print(f"       Text: {r['text'][:150]}...")
        print()

if __name__ == "__main__":
    main()
