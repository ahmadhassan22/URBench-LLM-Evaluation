"""SDFR-UR GSM8K FAIR — thinking ON, baseline CoT prompt, retrieved demos prepended.
Only difference from baseline = demos. Matches cot_gsm8k_baseline_fair.py exactly otherwise."""
import json, os, re, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE       = "/mnt/home/user41/URBench/data"
SPLITS     = f"{BASE}/sdfr_splits"
INDEXES    = f"{BASE}/sdfr_indexes"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_gsm8k_fair_qwen3_14b.jsonl"
TOP_K = 3

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_gold(answer_str):
    m = re.search(r"####\s*(-?[\d,]+)", answer_str)
    if m: return m.group(1).replace(",", "").strip()
    return answer_str.strip()

def extract_answer(text):
    # parse after the Urdu final-answer marker; fallback to last number
    m = re.search(r"حتمی\s*جواب\s*[:：]\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "").strip()
    m2 = re.search(r"####\s*(-?[\d,]+)", text)
    if m2: return m2.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""

def format_example(item):
    return f"مسئلہ: {item['question']}\nحتمی جواب: {extract_gold(item['answer'])}"

def retrieve(embedder, index, pool, q, k=TOP_K):
    vec = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

if __name__ == "__main__":
    pool = read_jsonl(f"{SPLITS}/gsm8k_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/gsm8k_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    index = faiss.read_index(f"{INDEXES}/gsm8k_faiss.index")
    embedder = SentenceTransformer(EMBED_PATH)
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    prompts = []
    for item in eval_data:
        neighbors = retrieve(embedder, index, pool, item["question"], TOP_K)
        few_shot = "\n\n".join(format_example(n) for n in neighbors)
        instr = ('آپ کو ایک ریاضی کا لفظی مسئلہ دیا گیا ہے۔\n'
                 'پہلے مرحلہ وار سوچیں اور مسئلہ حل کریں۔\n'
                 'اس کے بعد "حتمی جواب:" کے بعد صرف ایک عددی جواب لکھیں۔\n'
                 'حتمی جواب میں کوئی یونٹ، الفاظ یا اضافی متن شامل نہ کریں۔')
        q_block = f"مسئلہ: {item['question']}\n\nسوچنے کے مراحل:\nحتمی جواب:"
        raw = f"{instr}\n\n{few_shot}\n\n{q_block}"
        messages = [{"role": "user", "content": raw}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=True))

    outputs = llm.generate(prompts, sp)

    correct, results = 0, []
    for i, (item, out) in enumerate(zip(eval_data, outputs)):
        gen = out.outputs[0].text.strip()
        gen_final = gen.split("</think>")[-1]
        pred = extract_answer(gen_final)
        gold = extract_gold(item["answer"])
        ok = (pred == gold)
        correct += ok
        results.append({"qid": item.get("qid", f"GSM8K_{i:04d}"), "question": item["question"],
                         "gold": gold, "pred": pred, "correct": ok, "generated": gen})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SDFR-fair Correct: {correct}/{len(eval_data)} | Accuracy: {correct/len(eval_data)*100:.2f}%")
