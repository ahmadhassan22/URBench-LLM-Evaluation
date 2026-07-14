"""
rag/retrieve.py — Part 2 step 1: retrieval helper over the full Wikipedia index.

Design (locked from handoff 2026-07-09):
  - FAISS IndexFlatIP (cosine via normalized vecs), 23.96M vectors, loaded on CPU RAM.
  - Embedder: paraphrase-multilingual-MiniLM-L12-v2 on cuda, NO prefix, normalize=True.
  - Meta: byte-offset seek per retrieved row. NEVER full-load the 24M-line meta file.
  - Batched search: pass all queries in one index.search() call.

Run this on a GPU COMPUTE node (L20), NOT psn001 (login node OOMs on the 35G index).
Needs ~45G+ RAM for the flat index. Use sbatch with --mem >= 60G for real jobs.
"""

import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Absolute paths + local model dir, matching probe_strategyqa_factretrieval_test50.py
# (compute nodes may have no internet -> HF hub name would fail; use local path).
BASE        = "/mnt/home/user41/URBench"
INDEX_PATH  = f"{BASE}/rag/index/wikipedia_full.index"
META_PATH   = f"{BASE}/rag/index/wikipedia_full_meta.jsonl"
OFFSET_PATH = f"{BASE}/rag/index/wikipedia_full_meta.offsets.npy"   # cached byte offsets
MODEL_NAME  = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Retriever:
    def __init__(self,
                 index_path=INDEX_PATH,
                 meta_path=META_PATH,
                 offset_path=OFFSET_PATH,
                 model_name=MODEL_NAME,
                 device="cuda"):
        self.meta_path = meta_path

        t = time.time()
        print(f"[retriever] loading FAISS index: {index_path} ...", flush=True)
        self.index = faiss.read_index(index_path)          # CPU, ~37GB RAM
        print(f"[retriever] index loaded: ntotal={self.index.ntotal:,}  "
              f"d={self.index.d}  ({time.time()-t:.1f}s)", flush=True)

        t = time.time()
        print(f"[retriever] loading embedder: {model_name} on {device} ...", flush=True)
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[retriever] embedder ready ({time.time()-t:.1f}s)", flush=True)

        self.offsets = self._load_or_build_offsets(offset_path)
        print(f"[retriever] meta offsets ready: {len(self.offsets):,} lines", flush=True)

        # sanity: offsets must cover every vector
        if len(self.offsets) != self.index.ntotal:
            print(f"[retriever] WARNING: offsets ({len(self.offsets):,}) != "
                  f"index.ntotal ({self.index.ntotal:,}). Meta/index mismatch — "
                  f"do NOT trust meta lookups until this is fixed.", flush=True)

    def _load_or_build_offsets(self, offset_path):
        if os.path.exists(offset_path):
            print(f"[retriever] loading cached offsets: {offset_path}", flush=True)
            return np.load(offset_path)

        print(f"[retriever] building byte-offset map from {self.meta_path} "
              f"(one-time scan of the full meta file) ...", flush=True)
        offsets = []
        t = time.time()
        with open(self.meta_path, "rb") as f:
            off = f.tell()
            line = f.readline()
            while line:
                offsets.append(off)
                off = f.tell()
                line = f.readline()
                if len(offsets) % 2_000_000 == 0:
                    print(f"[retriever]   scanned {len(offsets):,} lines "
                          f"({time.time()-t:.0f}s)", flush=True)
        offsets = np.asarray(offsets, dtype=np.uint64)
        np.save(offset_path, offsets)
        print(f"[retriever] offset map built & cached: {len(offsets):,} lines "
              f"({time.time()-t:.0f}s)", flush=True)
        return offsets

    def embed(self, queries):
        """Encode -> unit-normalized float32. NO prefix (MiniLM, not e5)."""
        emb = self.model.encode(
            queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine via IP
            show_progress_bar=False,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def get_meta(self, row_ids):
        """Seek by byte offset and read only the requested meta lines."""
        out = []
        with open(self.meta_path, "rb") as f:
            for rid in row_ids:
                f.seek(int(self.offsets[rid]))
                rec = json.loads(f.readline().decode("utf-8"))
                out.append(rec)   # {"title": ..., "text": ...}
        return out

    def retrieve(self, queries, top_k=20):
        """
        Batched retrieve. Returns list (per query) of hits:
          [{"row": int, "score": float, "title": str, "text": str}, ...]
        """
        if isinstance(queries, str):
            queries = [queries]
        q = self.embed(queries)
        scores, ids = self.index.search(q, top_k)   # (nq, top_k)

        # flat meta lookup for all retrieved rows at once
        flat_ids = ids.reshape(-1)
        metas = self.get_meta(flat_ids.tolist())
        metas = np.array(metas, dtype=object).reshape(ids.shape)

        results = []
        for qi in range(len(queries)):
            hits = []
            for r in range(top_k):
                rid = int(ids[qi, r])
                if rid < 0:
                    continue
                m = metas[qi, r]
                hits.append({
                    "row":   rid,
                    "score": float(scores[qi, r]),
                    "title": m.get("title", ""),
                    "text":  m.get("text", ""),
                })
            results.append(hits)
        return results


# ---------------------------------------------------------------------------
# SMOKE TEST (not a TEST50): prove the plumbing returns the right articles on
# two known coverage_gap questions from the handoff verification set.
# Verifies: index loads, offsets align, no-prefix retrieval surfaces the fact
# article. This is a plumbing check, NOT an accuracy claim.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    r = Retriever(device="cuda")

    probes = [
        "What currency is used in the United Kingdom?",           # -> pound sterling (was FIXED)
        "How fast is a grey seal's reaction speed?",              # -> semantic_drift case
    ]
    res = r.retrieve(probes, top_k=5)
    for q, hits in zip(probes, res):
        print("\n" + "=" * 70)
        print("QUERY:", q)
        for h in hits:
            snippet = h["text"][:120].replace("\n", " ")
            print(f"  {h['score']:.4f}  [{h['title']}]  {snippet}...")