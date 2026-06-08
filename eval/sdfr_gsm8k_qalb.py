"""
SDFR-UR: Similarity-Based Dynamic Few-Shot Retrieval for Urdu Reasoning
Dataset: GSM8K | Model: Qalb-1.0-8B-Instruct
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
OUTPUT_FILE = f"{OUTPUT_DIR}/sdfr_gsm8k_qalb_1.0_8b.jsonl"
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
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m: return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if nums: return nums[-1].replace(",", "").strip()
    return ""

def extract_gold(answer_str):
    m = re.search(r"####\s*(-?[\d,]+)", answer_str)
    if m: return m.group(1).replace(",", "").strip()
    return answer_str.strip()

def format_hint(item):
    return f"- {item['question'][:60]} → {extract_gold(item['answer'])}"

def retrieve(embedder, index, pool, query_text, k=TOP_K):
    vec = embedder.encode([query_text], normalize_embeddings=True,
                          convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

def build_prompt(hints, question):
    system = "You are a helpful assistant."
    user   = (f"آپ ایک ریاضی استاد ہیں جو اردو میں سوالات حل کرتے ہیں۔ "
              f"صرف حتمی عددی جواب دیں۔\n\n"
              f"ملتے جلتے سوالوں کے جوابات:\n{hints}\n\n"
              f"اب یہ نیا سوال حل کریں:\n"
              f"سوال: {question}\nجواب:")
    return (SYS_START + NL + system + EOT +
            USR_START + NL + user + EOT +
            AST_START + NL)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pool      = read_jsonl(f"{SPLITS}/gsm8k_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/gsm8k_eval.jsonl")
    print(f"Pool: {len(pool)} | Eval: {len(eval_data)}")

    print("Loading FAISS index...")
    index = faiss.read_index(f"{INDEXES}/gsm8k_faiss.index")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_PATH)

    print("Loading Qalb-1.0-8B-Instruct...")
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256,
                                     stop=["<|eot_id|>", "<|end_of_text|>"])

    print("Building prompts...")
    prompts = []
    for item in eval_data:
        neighbors = retrieve(embedder, index, pool, item["question"], TOP_K)
        hints     = "\n".join(format_hint(n) for n in neighbors)
        prompts.append(build_prompt(hints, item["question"]))
    print(f"Built {len(prompts)} prompts.")

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    correct, total, results = 0, len(eval_data), []
    for i, (item, output) in enumerate(zip(eval_data, outputs)):
        generated  = output.outputs[0].text.strip()
        pred       = extract_answer(generated)
        gold       = extract_gold(item["answer"])
        is_correct = (pred == gold)
        if is_correct: correct += 1
        results.append({"qid": item.get("qid", f"GSM8K_{i:04d}"),
                        "question": item["question"], "gold": gold,
                        "pred": pred, "correct": is_correct,
                        "generated": generated})
        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{total}] acc: {correct/(i+1)*100:.2f}%", flush=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"SDFR-UR GSM8K | Qalb-1.0-8B | top-{TOP_K}")
    print(f"Correct: {correct}/{total} | Accuracy: {correct/total*100:.2f}%")
    print(f"Qalb GSM8K CoT baseline: 38.29%")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*50}")
