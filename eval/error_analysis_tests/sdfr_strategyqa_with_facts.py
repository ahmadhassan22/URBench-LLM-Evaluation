"""SDFR-UR StrategyQA + gold facts for the eval question (DIAGNOSTIC ONLY — facts are
gold-annotated, not available at real inference; this isolates the facts-vs-nofacts confound)."""
import json, os, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE       = "/mnt/home/user41/URBench/data"
SPLITS     = f"{BASE}/sdfr_splits"
INDEXES    = f"{BASE}/sdfr_indexes"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_withfacts_qwen3_14b.jsonl"
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

def format_example(item):
    ans = item.get("answer", False)
    label = "ہاں" if ans is True or str(ans).lower() in ["true","yes"] else "نہیں"
    return f"سوال: {item['question']}\nجواب: {label}"

def retrieve(embedder, index, pool, query_text, k=TOP_K):
    vec = embedder.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]], scores[0]

if __name__ == "__main__":
    os.makedirs("/mnt/home/user41/URBench/outputs/sdfr", exist_ok=True)
    pool = read_jsonl(f"{SPLITS}/strategyqa_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/strategyqa_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    index = faiss.read_index(f"{INDEXES}/strategyqa_faiss.index")
    embedder = SentenceTransformer(EMBED_PATH)
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=16, stop=["<|im_end|>"])

    prompts = []
    for item in eval_data:
        neighbors, _ = retrieve(embedder, index, pool, item["question"], TOP_K)
        few_shot = "\n\n".join(format_example(n) for n in neighbors)
        facts = "\n".join(f"- {f}" for f in item.get("facts", []))   # ONLY new line vs baseline SDFR
        system = "آپ ایک سوال جواب کا نظام ہیں۔ ہر سوال کا جواب صرف 'ہاں' یا 'نہیں' میں دیں۔"
        raw = f"{system}\n\n{few_shot}\n\nحقائق:\n{facts}\nسوال: {item['question']}\nجواب:"
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
                         "gold": gold, "pred": pred, "correct": is_correct})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Correct: {correct}/{len(eval_data)} | Accuracy: {correct/len(eval_data)*100:.2f}%")