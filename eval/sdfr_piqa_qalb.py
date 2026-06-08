"""
SDFR-UR: Similarity-Based Dynamic Few-Shot Retrieval for Urdu Reasoning
Dataset: PIQA | Model: Qalb-1.0-8B-Instruct
Manual LLaMA-3.1 prompt format (no chat template)
"""

import json, os, re, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE        = "/mnt/home/user41/URBench/data"
SPLITS      = f"{BASE}/sdfr_splits"
INDEXES     = f"{BASE}/sdfr_indexes"
MODEL_PATH  = "/mnt/home/user41/downloaded_models/Qalb/Qalb-1.0-8B-Instruct"
EMBED_PATH  = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR  = "/mnt/home/user41/URBench/outputs/sdfr"
OUTPUT_FILE = f"{OUTPUT_DIR}/sdfr_piqa_qalb_1.0_8b.jsonl"
TOP_K       = 3

SYS_START = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
USR_START = "<|start_header_id|>user<|end_header_id|>"
AST_START = "<|start_header_id|>assistant<|end_header_id|>"
EOT       = "<|eot_id|>"
NL        = "\n\n"

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
    label_str = "A" if item["label"] == 0 else "B"
    return f"- {item['goal'][:60]} → {label_str}"

def retrieve(embedder, index, pool, query_text, k=TOP_K):
    vec = embedder.encode([query_text], normalize_embeddings=True,
                          convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

def build_prompt(hints, goal, sol1, sol2):
    system = "You are a helpful assistant."
    user   = (f"آپ کو دو طریقوں میں سے بہتر طریقہ چننا ہے۔ صرف A یا B لکھیں۔\n\n"
              f"ملتے جلتے سوالوں کے جوابات:\n{hints}\n\n"
              f"اب یہ نیا سوال حل کریں:\n"
              f"سوال: {goal}\nA. {sol1}\nB. {sol2}\nجواب:")
    return (SYS_START + NL + system + EOT +
            USR_START + NL + user + EOT +
            AST_START + NL)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pool      = read_jsonl(f"{SPLITS}/piqa_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/piqa_eval.jsonl")
    print(f"Pool: {len(pool)} | Eval: {len(eval_data)}")

    print("Loading FAISS index...")
    index = faiss.read_index(f"{INDEXES}/piqa_faiss.index")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_PATH)

    print("Loading Qalb-1.0-8B-Instruct...")
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8,
                                     stop=["<|eot_id|>", "<|end_of_text|>", "\n"])

    print("Building prompts...")
    prompts = []
    for item in eval_data:
        neighbors = retrieve(embedder, index, pool, item["goal"], TOP_K)
        hints     = "\n".join(format_hint(n) for n in neighbors)
        prompts.append(build_prompt(hints, item["goal"],
                                    item["sol1"], item["sol2"]))
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
        results.append({"qid": f"PIQA_{i:04d}", "question": item["goal"],
                        "gold": gold, "pred": pred,
                        "correct": is_correct, "generated": generated})
        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{total}] acc: {correct/(i+1)*100:.2f}%", flush=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty = sum(1 for r in results if r["pred"] == "")
    print(f"\n{'='*50}")
    print(f"SDFR-UR PIQA | Qalb-1.0-8B | top-{TOP_K}")
    print(f"Correct: {correct}/{total} | Accuracy: {correct/total*100:.2f}%")
    print(f"Empty predictions: {empty}/{total}")
    print(f"Qalb PIQA CoT baseline: 51.07%")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*50}")
