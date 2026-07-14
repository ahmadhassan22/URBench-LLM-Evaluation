"""
PROBE (rerank, n=12, hand-read). Tests the last unbuilt mechanism.

Coverage probe showed: the fact is often IN the pool but BURIED (rank 4-8),
drowned by topically-similar noise. This tests whether a cross-encoder reranker
pulls the fact-bearing chunk to the top.

Pipeline per question (thinking OFF for gen):
  1. union queries = direct(Urdu->EN queries) + translate-first (rescues different entities)
  2. pool = retrieve top-20 per query, dedupe by CHUNK (row), keep top-40 by bi-encoder score
  3. for EACH sub-query: cross-encoder rerank the pool, take top-1
     (mirrors real method: one fact per sub-question; fixes drift + missing entity)
  4. print per-sub-query reranked winner NEXT TO gold facts

YOU hand-score: after rerank, is a gold-fact-bearing chunk now in the top slots?
Compare to the coverage probe: did reranking rescue the BURIED facts (US GDP figure,
Phelps-as-swimmer, Hepburn death) into the top?
Decision: reranked set reliably contains the facts -> drift beaten -> build gate + TEST50.

Memory: vLLM first (0.72), reranker on remainder (~2.3GB), embedder on CPU. Fits 48G L20.
If OOM: run via sbatch --mem=120G, or drop vLLM util to 0.65.
"""
import sys, json, re
BASE = "/mnt/home/user41/URBench"
sys.path.insert(0, f"{BASE}/rag")
from retrieve import Retriever
from vllm import LLM, SamplingParams
from sentence_transformers import CrossEncoder

EVAL        = f"{BASE}/data/sdfr_splits/strategyqa_eval.jsonl"
MODEL_PATH  = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
RERANK_PATH = "/mnt/home/user41/downloaded_models/BAAI/bge-reranker-v2-m3"
N           = 12
RETR_K      = 20    # per query
POOL_CAP    = 40    # rerank candidate cap (by bi-encoder score)

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

TRANSLATE = """Translate this Urdu question to English. Preserve all named entities exactly (people, places, brands). Output ONLY the English translation.

Urdu: {q}"""

DECOMP = """You are helping a retrieval system answer a question by looking facts up in an English encyclopedia (Wikipedia).

The question is:
{q}

List the factual things that must be looked up to answer it. Output ONLY a JSON array of 1-4 short English search queries, each targeting ONE entity or fact (e.g. "United States GDP 2018", "Walt Disney death year"). Do NOT answer the question. Output only the JSON array.
"""

def gen(llm, sp, prompts):
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text.strip() for o in outs]

def parse_queries(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
    except Exception:
        return []

def main():
    rows = read_jsonl(EVAL)[:N]

    # --- generation (vLLM first, grabs its GPU pool) ---
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.72,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    direct = gen(llm, sp, [DECOMP.format(q=x["question"]) for x in rows])
    english = gen(llm, sp, [TRANSLATE.format(q=x["question"]) for x in rows])
    trans   = gen(llm, sp, [DECOMP.format(q=e) for e in english])

    # --- retrieval + rerank (embedder CPU, reranker on GPU remainder) ---
    r = Retriever(device="cpu")
    reranker = CrossEncoder(RERANK_PATH, max_length=512, device="cuda")

    for row, en, d_raw, t_raw in zip(rows, english, direct, trans):
        union = list(dict.fromkeys(parse_queries(d_raw) + parse_queries(t_raw)))
        print("\n" + "=" * 80)
        print("UR :", row["question"])
        print("EN :", en)
        print("QUERIES (union):", union)
        if not union:
            print("  [no queries parsed]")
            continue

        # pool: dedupe by chunk row, keep best bi-encoder score
        pool = {}
        for hits in r.retrieve(union, top_k=RETR_K):
            for h in hits:
                if h["row"] not in pool or h["score"] > pool[h["row"]]["score"]:
                    pool[h["row"]] = h
        cands = sorted(pool.values(), key=lambda h: -h["score"])[:POOL_CAP]
        texts = [c["text"] for c in cands]

        print(f"--- RERANKED top-1 per sub-query (pool size {len(cands)}) ---")
        seen = set()
        for q in union:
            scores = reranker.predict([[q, t] for t in texts])
            best = max(range(len(texts)), key=lambda i: scores[i])
            c = cands[best]
            key = c["row"]
            snip = c["text"][:120].replace("\n", " ")
            dup = "  (dup)" if key in seen else ""
            seen.add(key)
            print(f"  [{q}]")
            print(f"      {scores[best]:+.2f}  [{c['title']}]{dup}  {snip}...")
        print("--- GOLD FACTS (hand-score: present in reranked set above?) ---")
        for fct in row["facts"]:
            print("   -", fct)

if __name__ == "__main__":
    main()