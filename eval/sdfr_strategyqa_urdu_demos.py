"""SDFR-UR StrategyQA, Urdu-demos variant. Retrieval unchanged (same English FAISS index).
Only the demo TEXT shown to the model is swapped to Urdu (human-quality translation).
qid-lookup for pool_ur so the 1 missing translation can't misalign retrieval."""
import json, os, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE       = "/mnt/home/user41/URBench/data"
SPLITS     = f"{BASE}/sdfr_splits"
INDEXES    = f"{BASE}/sdfr_indexes"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_urdu_demos_qwen3_14b.jsonl"
TOP_K = 3

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def normalize_answer(text):
    t = text.strip().lower()
    if t in ["true","yes","ہاں","ہاں۔"]: return "ہاں"
    if t in ["false","no","نہیں","نہیں۔"]: return "نہیں"
    if "ہاں" in text: return "ہاں"
    if "نہیں" in text: return "نہیں"
    return text.strip()

def format_example_urdu(q_text, ans):
    label = "ہاں" if ans is True or str(ans).lower() in ["true","yes"] else "نہیں"
    return f"سوال: {q_text}\nجواب: {label}"

def retrieve(embedder, index, query_text, k=TOP_K):
    vec = embedder.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(vec, k)
    return ids[0], scores[0]

if __name__ == "__main__":
    os.makedirs("/mnt/home/user41/URBench/outputs/sdfr", exist_ok=True)

    pool_en = read_jsonl(f"{SPLITS}/strategyqa_pool.jsonl")            # index-aligned to FAISS
    pool_ur_by_qid = {r["qid"]: r for r in read_jsonl(f"{SPLITS}/strategyqa_pool_urdu.jsonl")}
    eval_data = read_jsonl(f"{SPLITS}/strategyqa_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    n_fallback = 0  # count neighbors with no Urdu translation (the 1 missing)

    index = faiss.read_index(f"{INDEXES}/strategyqa_faiss.index")
    embedder = SentenceTransformer(EMBED_PATH)
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=16, stop=["<|im_end|>"])

    prompts, retrieved_log = [], []
    for item in eval_data:
        ids, sims = retrieve(embedder, index, item["question"], TOP_K)
        demo_lines, log_row = [], []
        for i, s in zip(ids, sims):
            en_item = pool_en[i]
            ur_item = pool_ur_by_qid.get(en_item["qid"])
            if ur_item is None:
                q_text = en_item["question"]   # fallback: 1 missing qid keeps English
                n_fallback += 1
            else:
                q_text = ur_item["question"]
            ans = en_item.get("answer")
            demo_lines.append(format_example_urdu(q_text, ans))
            log_row.append({"q_ur": q_text, "q_en": en_item["question"], "a": ans, "sim": float(s)})
        retrieved_log.append(log_row)
        few_shot = "\n\n".join(demo_lines)
        system = "آپ ایک سوال جواب کا نظام ہیں۔ ہر سوال کا جواب صرف 'ہاں' یا 'نہیں' میں دیں۔"
        raw = f"{system}\n\n{few_shot}\n\nسوال: {item['question']}\nجواب:"
        messages = [{"role": "user", "content": raw}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    outputs = llm.generate(prompts, sp)

    correct, results = 0, []
    for i, (item, out) in enumerate(zip(eval_data, outputs)):
        gen = out.outputs[0].text.strip()
        pred = normalize_answer(gen)
        gold_raw = item.get("answer", False)
        gold = "ہاں" if gold_raw is True or str(gold_raw).lower() in ["true","yes","ہاں"] else "نہیں"
        is_correct = (pred == gold)
        correct += is_correct
        results.append({"qid": item.get("qid", f"SQA_{i:04d}"), "question": item["question"],
                         "gold": gold, "pred": pred, "correct": is_correct, "generated": gen,
                         "retrieved": retrieved_log[i]})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Correct: {correct}/{len(eval_data)} | Accuracy: {correct/len(eval_data)*100:.2f}%")
    print(f"Demo-neighbor English fallbacks (missing Urdu): {n_fallback}")
    print(f"Output: {OUTPUT_FILE}")