"""
SDFR-UR BoolQ Alif v2 - Context-hints approach
Passage-based retrieval, hints format instead of full examples
Full run: 310 examples
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
OUTPUT_FILE = f"{OUTPUT_DIR}/sdfr_boolq_alif_v2_1.0_8b.jsonl"
TOP_K       = 3

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
    """Brief hint: topic + answer only, no full passage."""
    ans   = item["answer"]
    label = "ہاں" if ans is True or str(ans).lower() == "true" else "نہیں"
    return f"- {item['question'][:60]} → {label}"

def retrieve(embedder, index, pool, query_passage, k=TOP_K):
    vec = embedder.encode([query_passage[:300]], normalize_embeddings=True,
                          convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pool      = read_jsonl(f"{SPLITS}/boolq_pool_large_clean.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/boolq_eval.jsonl")
    print(f"Pool: {len(pool)} | Eval: {len(eval_data)}")

    print("Loading FAISS index (passage-based)...")
    index = faiss.read_index(f"{INDEXES}/boolq_large_passage_faiss.index")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_PATH)

    print("Loading Alif-1.0-8B-Merged...")
    llm       = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192,
                    gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8,
                                     stop=["overposting", "\n", "مجھے"])

    print("Building prompts with passage-based hints...")
    prompts = []
    for item in eval_data:
        neighbors = retrieve(embedder, index, pool, item["passage"], TOP_K)
        hints     = "\n".join(format_hint(n) for n in neighbors)
        passage   = item["passage"][:400]

        raw_prompt = (
            f"آپ کو متن پڑھ کر سوال کا جواب صرف 'ہاں' یا 'نہیں' میں دینا ہے۔\n\n"
            f"ملتے جلتے سوالوں کے جوابات:\n{hints}\n\n"
            f"متن: {passage}\n"
            f"سوال: {item['question']}\n"
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
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{total}] acc: {correct/(i+1)*100:.2f}%", flush=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    other = sum(1 for r in results if r["pred"] not in ["ہاں","نہیں"])
    print(f"\n{'='*50}")
    print(f"SDFR-UR BoolQ Alif v2 | Alif-1.0-8B | top-{TOP_K}")
    print(f"Correct: {correct}/{total} | Accuracy: {correct/total*100:.2f}%")
    print(f"Non-ہاں/نہیں predictions: {other}/{total}")
    print(f"Alif BoolQ CoT baseline: 71.57%")
    print(f"SDFR-UR v1: 57.74%")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*50}")
