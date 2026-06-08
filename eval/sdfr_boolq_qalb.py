"""
SDFR-UR: Similarity-Based Dynamic Few-Shot Retrieval for Urdu Reasoning
Dataset: BoolQ | Model: Qalb-1.0-8B-Instruct
Manual LLaMA-3.1 prompt format (no chat template)
Passage-based retrieval + large clean pool
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
OUTPUT_FILE = f"{OUTPUT_DIR}/sdfr_boolq_qalb_1.0_8b.jsonl"
TOP_K       = 3

SYS_START = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
USR_START = "<|start_header_id|>user<|end_header_id|>"
AST_START = "<|start_header_id|>assistant<|end_header_id|>"
EOT       = "<|eot_id|>"
NL        = "\n\n"

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def normalize_answer(text):
    t = text.strip().lower()
    if t in ["true", "yes", "ہاں", "ہاں۔"]: return "ہاں"
    if t in ["false", "no", "نہیں", "نہیں۔"]: return "نہیں"
    if "ہاں" in text: return "ہاں"
    if "نہیں" in text: return "نہیں"
    return text.strip()

def format_hint(item):
    ans   = item["answer"]
    label = "ہاں" if ans is True or str(ans).lower() == "true" else "نہیں"
    return f"- {item['question'][:60]} → {label}"

def retrieve(embedder, index, pool, query_passage, k=TOP_K):
    vec = embedder.encode([query_passage[:300]], normalize_embeddings=True,
                          convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

def build_prompt(hints, passage, question):
    system = "You are a helpful assistant."
    user   = (f"آپ کو متن پڑھ کر سوال کا جواب صرف 'ہاں' یا 'نہیں' میں دینا ہے۔\n\n"
              f"ملتے جلتے سوالوں کے جوابات:\n{hints}\n\n"
              f"متن: {passage}\n"
              f"سوال: {question}\n"
              f"جواب:")
    return (SYS_START + NL + system + EOT +
            USR_START + NL + user + EOT +
            AST_START + NL)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pool      = read_jsonl(f"{SPLITS}/boolq_pool_large_clean.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/boolq_eval.jsonl")
    print(f"Pool: {len(pool)} | Eval: {len(eval_data)}")

    print("Loading FAISS index (passage-based)...")
    index = faiss.read_index(f"{INDEXES}/boolq_large_passage_faiss.index")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_PATH)

    print("Loading Qalb-1.0-8B-Instruct...")
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8,
                                     stop=["<|eot_id|>", "<|end_of_text|>", "\n"])

    print("Building prompts with passage-based hints...")
    prompts = []
    for item in eval_data:
        neighbors = retrieve(embedder, index, pool, item["passage"], TOP_K)
        hints     = "\n".join(format_hint(n) for n in neighbors)
        passage   = item["passage"][:400]
        prompts.append(build_prompt(hints, passage, item["question"]))
    print(f"Built {len(prompts)} prompts.")

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    correct, total, results = 0, len(eval_data), []
    for i, (item, output) in enumerate(zip(eval_data, outputs)):
        generated  = output.outputs[0].text.strip()
        pred       = normalize_answer(generated)
        gold       = normalize_answer(item["answer"])
        is_correct = (pred == gold)
        if is_correct: correct += 1
        results.append({
            "qid":       item.get("qid", f"BOOLQ_{i:04d}"),
            "question":  item["question"],
            "gold":      gold, "pred": pred,
            "correct":   is_correct, "generated": generated,
        })
        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{total}] acc: {correct/(i+1)*100:.2f}%", flush=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    other = sum(1 for r in results if r["pred"] not in ["ہاں","نہیں"])
    print(f"\n{'='*50}")
    print(f"SDFR-UR BoolQ | Qalb-1.0-8B | top-{TOP_K}")
    print(f"Correct: {correct}/{total} | Accuracy: {correct/total*100:.2f}%")
    print(f"Non-ہاں/نہیں predictions: {other}/{total}")
    print(f"Qalb BoolQ CoT baseline: 55.40%")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*50}")
