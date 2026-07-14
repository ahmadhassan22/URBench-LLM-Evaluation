"""
PROBE (retrieval coverage, n=12, hand-read). THE MAKE-OR-BREAK TEST.

Stops grading the proxy (query text) and measures the target: does a gold-fact-
bearing chunk actually land in the retrieved pool?

Pipeline per question (thinking OFF for all generation -> clean, no think-leak):
  1. DIRECT queries:     Urdu question       -> English search queries
  2. TRANSLATED queries: Urdu -> English Q    -> English search queries
  3. UNION the queries (direct and translate fail on DIFFERENT entities;
     union gives retrieval its best shot).
  4. Retrieve top-8 per query over the full 24M index, pool + dedupe by title.
  5. Print pooled top chunks NEXT TO gold facts.

YOU hand-score each: is a gold fact present in the pool?  yes / partial / no.
Decision: ~70%+ present -> method has headroom above 65.5%, build rerank+gate.
          ~50% -> retrieval capped near baseline -> pivot to leakage/diagnostic paper.
NOT banked. n=12 qualitative fork.

Embedder on CPU (tiny, ~70 short queries) so vLLM owns the GPU -> no contention.
If it OOMs interactively, run via sbatch --mem=120G.
"""
import sys, json, re
BASE = "/mnt/home/user41/URBench"
sys.path.insert(0, f"{BASE}/rag")
from retrieve import Retriever
from vllm import LLM, SamplingParams

EVAL       = f"{BASE}/data/sdfr_splits/strategyqa_eval.jsonl"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
N          = 12
TOP_K      = 8   # per query
POOL_SHOW  = 8   # unique chunks shown per question

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
    """Batched chat with thinking OFF."""
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text.strip() for o in outs]

def parse_queries(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []

def main():
    rows = read_jsonl(EVAL)[:N]
    r = Retriever(device="cpu")   # embedder on CPU; index on CPU RAM
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=256)

    # direct queries
    direct_raw = gen(llm, sp, [DECOMP.format(q=x["question"]) for x in rows])
    # translate then queries
    english    = gen(llm, sp, [TRANSLATE.format(q=x["question"]) for x in rows])
    trans_raw  = gen(llm, sp, [DECOMP.format(q=e) for e in english])

    for row, en, d_raw, t_raw in zip(rows, english, direct_raw, trans_raw):
        dq = parse_queries(d_raw)
        tq = parse_queries(t_raw)
        union = list(dict.fromkeys(dq + tq))   # dedupe, preserve order

        # retrieve pooled
        pooled = {}
        if union:
            hits_per_q = r.retrieve(union, top_k=TOP_K)
            for hits in hits_per_q:
                for h in hits:
                    key = h["title"]
                    if key not in pooled or h["score"] > pooled[key]["score"]:
                        pooled[key] = h
        top = sorted(pooled.values(), key=lambda h: -h["score"])[:POOL_SHOW]

        print("\n" + "=" * 78)
        print("UR :", row["question"])
        print("EN :", en)
        print("QUERIES (union):", union)
        print("--- POOLED RETRIEVED CHUNKS ---")
        for h in top:
            snip = h["text"][:110].replace("\n", " ")
            print(f"  {h['score']:.3f}  [{h['title']}]  {snip}...")
        print("--- GOLD FACTS (hand-score: is any of this in the pool above?) ---")
        for fct in row["facts"]:
            print("   -", fct)

if __name__ == "__main__":
    main()