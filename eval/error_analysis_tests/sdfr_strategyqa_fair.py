"""SDFR-UR StrategyQA — FAIR: same no-facts CoT prompt as cot_strategyqa_nofacts_baseline_fair.py,
thinking ON, max_tokens=2048, identical parsing. ONLY difference vs baseline = retrieved demos.
Replaces the old handicapped run (thinking OFF, max_tokens=16, answer-only)."""
import json, os, re, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE        = "/mnt/home/user41/URBench/data"
SPLITS      = f"{BASE}/sdfr_splits"
INDEXES     = f"{BASE}/sdfr_indexes"
MODEL_PATH  = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH  = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_fair_qwen3_14b.jsonl"
TOP_K       = 3

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm_gold(a):
    if isinstance(a, bool): return "ہاں" if a else "نہیں"
    s = str(a).strip().lower()
    if s in ("true", "yes", "ہاں"):  return "ہاں"
    if s in ("false", "no", "نہیں"): return "نہیں"
    return str(a).strip()

def extract_answer(text):
    if "حتمی جواب" in text:
        text = text.split("حتمی جواب")[-1]
    i_haan = text.rfind("ہاں")
    i_nahi = text.rfind("نہیں")
    if i_haan == -1 and i_nahi == -1:
        tl = text.lower()
        if "yes" in tl and "no" not in tl: return "ہاں"
        if "no" in tl and "yes" not in tl: return "نہیں"
        return ""
    return "ہاں" if i_haan > i_nahi else "نہیں"

def demo_label(item):
    ans = item.get("answer", item.get("expected_answer", False))
    return "ہاں" if ans is True or str(ans).lower() in ("true", "yes") else "نہیں"

def retrieve(embedder, index, pool, query_text, k=TOP_K):
    vec = embedder.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(vec, k)
    return [pool[i] for i in ids[0]], scores[0]

if __name__ == "__main__":
    pool      = read_jsonl(f"{SPLITS}/strategyqa_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/strategyqa_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]
    print(f"Pool: {len(pool)} | Eval: {len(eval_data)}")

    index    = faiss.read_index(f"{INDEXES}/strategyqa_faiss.index")
    embedder = SentenceTransformer(EMBED_PATH)
    llm      = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok      = llm.get_tokenizer()
    sp       = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    prompts, retrieved_log = [], []
    for item in eval_data:
        neighbors, sims = retrieve(embedder, index, pool, item["question"], TOP_K)
        retrieved_log.append([{"q": n["question"], "a": demo_label(n), "sim": float(s)}
                              for n, s in zip(neighbors, sims)])
        # demos as Q -> final answer (answer-only; that is what the method retrieves)
        few_shot = "\n\n".join(f"سوال: {n['question']}\nحتمی جواب: {demo_label(n)}" for n in neighbors)
        # SAME instruction block as the no-facts baseline; demos are the ONLY addition
        instr = ("آپ کو ایک سوال دیا گیا ہے۔\n"
                 "پہلے مرحلہ وار اور منطقی طور پر سوچیں۔\n"
                 "اس کے بعد آخر میں صرف ایک لفظ میں جواب دیں: ہاں یا نہیں۔\n"
                 "کوئی وضاحت، جملہ، یا اضافی متن شامل نہ کریں۔")
        q_block = f"سوال: {item['question']}\nسوچنے کے مراحل:\nحتمی جواب:"
        raw = f"{instr}\n\n{few_shot}\n\n{q_block}"
        messages = [{"role": "user", "content": raw}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=True))

    outputs = llm.generate(prompts, sp)

    correct, answered, trunc, results = 0, 0, 0, []
    for i, (item, out) in enumerate(zip(eval_data, outputs)):
        gen = out.outputs[0].text.strip()
        is_trunc = ("<think>" in gen and "</think>" not in gen)
        if is_trunc: trunc += 1
        gen_final = gen.split("</think>")[-1]
        pred = extract_answer(gen_final)
        gold = norm_gold(item.get("answer"))
        if pred: answered += 1
        ok = (pred == gold) and pred != ""
        correct += ok
        results.append({"qid": item.get("qid", f"SQA_{i:04d}"), "question": item["question"],
                        "gold": gold, "pred": pred, "correct": ok,
                        "truncated": is_trunc, "generated": gen,
                        "retrieved": retrieved_log[i]})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(eval_data)
    print(f"Correct: {correct}/{n} | Overall: {correct/n*100:.2f}% | "
          f"Answered: {answered}/{n} ({answered/n*100:.1f}%) | "
          f"AnsAcc: {correct/answered*100:.2f}% | Truncated: {trunc}")
