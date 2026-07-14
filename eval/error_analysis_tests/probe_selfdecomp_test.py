"""
PROBE (leak-free self-decomposition, n=12, hand-read). THE FORK.

Question: with ONLY the Urdu question (no facts, no gold decomposition, no term),
can Qwen3-14B produce English search queries that would retrieve the needed facts?

This is the gate for the whole leak-free method. If the generated queries are
sensible entity/fact lookups -> method is viable, proceed to retrieve+rerank+gate.
If garbage -> pivot to the leakage-quantification study.

Output: prints, per question, the model's generated queries NEXT TO the gold facts
(gold shown ONLY for your eyeball scoring — it is NOT given to the model).
NOT a TEST50, NOT banked. A qualitative fork decision on ~12 hand-read cases.
"""
import json, re
from vllm import LLM, SamplingParams

BASE       = "/mnt/home/user41/URBench"
EVAL       = f"{BASE}/data/sdfr_splits/strategyqa_eval.jsonl"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
N          = 12

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# Prompt gives ONLY the question. Asks for English lookup queries. Explicitly
# forbids answering (we want retrieval queries, not the model's guess).
PROMPT = """You are helping a retrieval system answer a question by looking facts up in an English encyclopedia (Wikipedia).

The question (in Urdu) is:
{q}

List the factual things that must be looked up to answer it. Output ONLY a JSON array of 1-4 short English search queries. Each query should target ONE entity or fact (e.g. "United States GDP 2018", "Walt Disney death year"). Do NOT answer the question. Do NOT explain. Output only the JSON array.
"""

def extract_json_list(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None, text.strip()
    try:
        return json.loads(m.group(0)), None
    except Exception:
        return None, text.strip()

def main():
    rows = read_jsonl(EVAL)[:N]
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)

    prompts = [PROMPT.format(q=r["question"]) for r in rows]
    # chat template with thinking on (parse after </think>)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, sp)

    for r, o in zip(rows, outs):
        raw = o.outputs[0].text
        queries, fallback = extract_json_list(raw)
        print("\n" + "=" * 74)
        print("Q :", r["question"])
        if queries is not None:
            print("GENERATED QUERIES (leak-free):")
            for i, q in enumerate(queries, 1):
                print(f"   {i}. {q}")
        else:
            print("GENERATED (JSON PARSE FAILED, raw tail):")
            print("   ", fallback[:300])
        print("GOLD FACTS (for YOUR eyeball only — not shown to model):")
        for fct in r["facts"]:
            print("   -", fct)

if __name__ == "__main__":
    main()