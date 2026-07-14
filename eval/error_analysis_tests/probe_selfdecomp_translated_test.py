"""
PROBE (translate-then-decompose, n=12, hand-read). Tests the fix for the
dominant failure found in probe_selfdecomp_test.py: cross-lingual entity
misidentification (kanji->Kannada, hornet->Horn of Africa, ->Titanic).

Only change vs the previous probe: the question is TRANSLATED to English first
(a legit MT step on the legal input — NOT a gold field), then the SAME
decomposition prompt runs on the English question. Clean A/B: input language only.

Watch #2 (kanji), #4 (Hepburn/AirTrain), #8 (hornet) — do the entity errors fix?
NOT banked. Qualitative fork decision, hand-read.
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

TRANSLATE = """Translate this Urdu question to English. Preserve all named entities exactly. Output ONLY the English translation, nothing else.

Urdu: {q}"""

# IDENTICAL decomposition prompt to probe_selfdecomp_test.py, English input.
DECOMP = """You are helping a retrieval system answer a question by looking facts up in an English encyclopedia (Wikipedia).

The question is:
{q}

List the factual things that must be looked up to answer it. Output ONLY a JSON array of 1-4 short English search queries. Each query should target ONE entity or fact (e.g. "United States GDP 2018", "Walt Disney death year"). Do NOT answer the question. Do NOT explain. Output only the JSON array.
"""

def after_think(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()

def extract_json_list(text):
    t = after_think(text)
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return None, t
    try:
        return json.loads(m.group(0)), None
    except Exception:
        return None, t

def main():
    rows = read_jsonl(EVAL)[:N]
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, trust_remote_code=True)
    sp_t = SamplingParams(temperature=0.0, max_tokens=512)
    sp_d = SamplingParams(temperature=0.0, max_tokens=1024)

    # 1) translate all
    tconvs = [[{"role": "user", "content": TRANSLATE.format(q=r["question"])}] for r in rows]
    touts = llm.chat(tconvs, sp_t)
    english = [after_think(o.outputs[0].text) for o in touts]

    # 2) decompose the English
    dconvs = [[{"role": "user", "content": DECOMP.format(q=e)}] for e in english]
    douts = llm.chat(dconvs, sp_d)

    for r, en, o in zip(rows, english, douts):
        queries, fallback = extract_json_list(o.outputs[0].text)
        print("\n" + "=" * 74)
        print("UR :", r["question"])
        print("EN :", en)
        if queries is not None:
            print("GENERATED QUERIES (translate-first):")
            for i, q in enumerate(queries, 1):
                print(f"   {i}. {q}")
        else:
            print("GENERATED (JSON PARSE FAILED, raw tail):")
            print("   ", fallback[:300])
        print("GOLD FACTS (your eyeball only):")
        for fct in r["facts"]:
            print("   -", fct)

if __name__ == "__main__":
    main()