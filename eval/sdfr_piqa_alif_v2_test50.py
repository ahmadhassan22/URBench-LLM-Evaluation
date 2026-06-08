"""
SDFR-UR PIQA Alif v2 - Context-hints approach
Retrieved examples shown as structural context, not as strict templates
Test: 50 examples
"""

import json, os, re, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE        = "/mnt/home/user41/URBench/data"
SPLITS      = f"{BASE}/sdfr_splits"
INDEXES     = f"{BASE}/sdfr_indexes"
MODEL_PATH  = "/mnt/home/user41/downloaded_models/Alif/Alif-1.0-8B-Merged"
EMBED_PATH  = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR  = "/mnt/home/user41/URBench/outputs/sdfr"
OUTPUT_FILE = f"{OUTPUT_DIR}/sdfr_piqa_alif_v2_test50.jsonl"
TOP_K       = 3
TEST_N      = 50

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_answer(text):
    t = text.strip()
    first = t[:30].upper()
    if first.startswith("A"): return "0"
    if first.startswith("B"): return "1"
    if "جواب: A" in t or "جواب:A" in t: return "0"
    if "جواب: B" in t or "جواب:B" in t: return "1"
    if re.search(r'\bA\b', first): return "0"
    if re.search(r'\bB\b', first): return "1"
    return ""

def format_hint(item):
    """Show retrieved example as a brief structural hint only."""
    label_str = "A" if item["label"] == 0 else "B"
    return f"- {item['goal'][:60]} → {label_str}"

def retrieve(embedder, index, pool, query_text, k=TOP_K):
    vec = embedder.encode([query_text], normalize_embeddings=True,
                          convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pool      = read_jsonl(f"{SPLITS}/piqa_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/piqa_eval.jsonl")[:TEST_N]
    print(f"Pool: {len(pool)} | Eval (test): {len(eval_data)}")

    print("Loading FAISS index...")
    index = faiss.read_index(f"{INDEXES}/piqa_faiss.index")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_PATH)

    print("Loading Alif-1.0-8B-Merged...")
    llm       = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192,
                    gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=4,
                                     stop=["overposting", "\n"])

    print("Building prompts...")
    prompts = []
    for item in eval_data:
        neighbors = retrieve(embedder, index, pool, item["goal"], TOP_K)
        hints     = "\n".join(format_hint(n) for n in neighbors)

        raw_prompt = (
            f"آپ کو دو طریقوں میں سے بہتر طریقہ چننا ہے۔ صرف A یا B لکھیں۔\n\n"
            f"ملتے جلتے سوالوں کے جوابات:\n{hints}\n\n"
            f"سوال: {item['goal']}\n"
            f"A. {item['sol1']}\n"
            f"B. {item['sol2']}\n"
            f"جواب:"
        )
        messages = [{"role": "user", "content": raw_prompt}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    print(f"Built {len(prompts)} prompts.")

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    correct, total, results = 0, len(eval_data), []
    for i, (item, output) in enumerate(zip(eval_data, outputs)):
        generated  = output.outputs[0].text.strip()
        pred       = extract_answer(generated)
        gold       = str(item["label"])
        is_correct = (pred == gold)
        if is_correct: correct += 1
        results.append({
            "qid": f"PIQA_{i:04d}", "question": item["goal"],
            "gold": gold, "pred": pred,
            "correct": is_correct, "generated": generated,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty = sum(1 for r in results if r["pred"] == "")
    print(f"\n{'='*50}")
    print(f"SDFR-UR PIQA Alif v2 | top-{TOP_K} hints")
    print(f"Correct: {correct}/{total} | Accuracy: {correct/total*100:.2f}%")
    print(f"Empty predictions: {empty}/{total}")
    print(f"Alif PIQA CoT baseline: 44.93%")
    print(f"SDFR-UR v1: 36.00%")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*50}")
