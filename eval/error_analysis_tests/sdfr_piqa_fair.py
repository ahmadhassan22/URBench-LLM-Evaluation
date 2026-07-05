"""SDFR-UR PIQA FAIR — thinking ON, baseline CoT prompt (0/1 format), retrieved demos prepended.
Only difference from baseline = demos."""
import json, os, re, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE       = "/mnt/home/user41/URBench/data"
SPLITS     = f"{BASE}/sdfr_splits"
INDEXES    = f"{BASE}/sdfr_indexes"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_piqa_fair_qwen3_14b.jsonl"
TOP_K = 3

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_answer(text):
    m = re.search(r"حتمی\s*جواب\s*[:：]\s*([01])", text)
    if m: return m.group(1)
    m2 = re.findall(r"\b([01])\b", text)
    return m2[-1] if m2 else ""

def format_example(item):
    return (f"مقصد: {item['goal']}\n"
            f"حل 1: {item['sol1']}\nحل 2: {item['sol2']}\n"
            f"حتمی جواب: {item['label']}")

def retrieve(embedder, index, pool, q, k=TOP_K):
    vec = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    _, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]]

if __name__ == "__main__":
    pool = read_jsonl(f"{SPLITS}/piqa_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/piqa_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    index = faiss.read_index(f"{INDEXES}/piqa_faiss.index")
    embedder = SentenceTransformer(EMBED_PATH)
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    prompts = []
    for item in eval_data:
        neighbors = retrieve(embedder, index, pool, item["goal"], TOP_K)
        few_shot = "\n\n".join(format_example(n) for n in neighbors)
        instr = ('مقصد اور دو ممکنہ حل دیے گئے ہیں۔\n'
                 'پہلے مرحلہ وار اور تفصیل سے سوچیں کہ کون سا حل زیادہ منطقی ہے اور عام فہم کے مطابق ہے۔\n'
                 'پھر آخر میں صرف ایک ہندسہ لکھیں: 0 یا 1۔\n'
                 '0 کا مطلب ہے حل 1 درست ہے، 1 کا مطلب ہے حل 2 درست ہے۔')
        q_block = (f"مقصد: {item['goal']}\n"
                   f"حل 1: {item['sol1']}\nحل 2: {item['sol2']}\n\n"
                   f"سوچنے کے مراحل:\nحتمی جواب:")
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
        gold = str(item["label"])
        ok = (pred == gold)
        correct += ok
        results.append({"qid": f"PIQA_{i:04d}", "question": item["goal"],
                         "gold": gold, "pred": pred, "correct": ok, "generated": gen})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SDFR-fair Correct: {correct}/{len(eval_data)} | Accuracy: {correct/len(eval_data)*100:.2f}%")
