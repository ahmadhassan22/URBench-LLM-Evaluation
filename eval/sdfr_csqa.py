"""
SDFR-UR: Similarity-Based Dynamic Few-Shot Retrieval for Urdu Reasoning
Dataset: CSQA | Model: Qwen3-14B | enable_thinking=False
"""

import json, os, re, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE        = "/mnt/home/user41/URBench/data"
SPLITS      = f"{BASE}/sdfr_splits"
INDEXES     = f"{BASE}/sdfr_indexes"
MODEL_PATH  = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH  = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR  = "/mnt/home/user41/URBench/outputs/sdfr"
OUTPUT_FILE = f"{OUTPUT_DIR}/sdfr_csqa_qwen3_14b.jsonl"
TOP_K       = 3

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_answer(text):
    m = re.search(r"\b([A-E])\b", text.strip().upper())
    if m: return m.group(1)
    return text.strip().upper()[:1]

def format_example(item):
    choices = item["choices"]
    opts    = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
    return f"سوال: {item['question']}\n{opts}\nجواب: {item['answerKey']}"

def retrieve(embedder, index, pool, query_text, k=TOP_K):
    vec = embedder.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pool      = read_jsonl(f"{SPLITS}/csqa_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/csqa_eval.jsonl")
    print(f"Pool: {len(pool)} | Eval: {len(eval_data)}")

    print("Loading FAISS index...")
    index = faiss.read_index(f"{INDEXES}/csqa_faiss.index")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_PATH)

    print("Loading Qwen3-14B...")
    llm       = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192,
                    gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8,
                                     stop=["<|im_end|>"])

    print("Building prompts with dynamic retrieval...")
    prompts = []
    for item in eval_data:
        neighbors  = retrieve(embedder, index, pool, item["question"], TOP_K)
        few_shot   = "\n\n".join(format_example(n) for n in neighbors)
        choices    = item["choices"]
        opts       = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
        system     = "آپ ایک سوال جواب کا نظام ہیں۔ دیے گئے اختیارات میں سے صحیح جواب کا حرف (A، B، C، D، یا E) لکھیں۔"
        raw_prompt = f"{system}\n\n{few_shot}\n\nسوال: {item['question']}\n{opts}\nجواب:"
        messages   = [{"role": "user", "content": raw_prompt}]
        prompt     = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(prompt)
    print(f"Built {len(prompts)} prompts.")

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    correct, total, results = 0, len(eval_data), []
    for i, (item, output) in enumerate(zip(eval_data, outputs)):
        generated  = output.outputs[0].text.strip()
        pred       = extract_answer(generated)
        gold       = item["answerKey"].strip().upper()
        is_correct = (pred == gold)
        if is_correct: correct += 1
        results.append({"qid": item.get("qid", f"CSQA_{i:04d}"),
                        "question": item["question"], "gold": gold,
                        "pred": pred, "correct": is_correct, "generated": generated})
        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{total}] acc: {correct/(i+1)*100:.2f}%", flush=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"SDFR-UR CSQA | Qwen3-14B | top-{TOP_K} retrieval")
    print(f"Correct: {correct}/{total} | Accuracy: {correct/total*100:.2f}%")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*50}")
