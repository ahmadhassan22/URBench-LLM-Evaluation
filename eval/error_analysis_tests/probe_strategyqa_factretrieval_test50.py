"""PROBE (TEST50 only): 3-condition StrategyQA diagnostic to check if fact-retrieval
has headroom above the fair no-facts baseline, BEFORE building entity-aware retrieval.

Conditions (Qwen3-14B, thinking ON, max_tokens=2048, identical prompt except facts source):
  1. NO facts           -> reproduces fair baseline floor (~65%)
  2. GOLD facts          -> perfect-retrieval ceiling (~80%)
  3. RETRIEVED facts     -> top-k facts pulled by similarity from the POOL questions'
                            gold facts (optimistic sim of a real fact-retrieval method)

Note: condition 3 retrieves from StrategyQA's OWN gold facts pool -> generous.
A real corpus is messier. If cond 3 can't beat cond 1 here, the real method is in trouble.
"""
import json, os, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE       = "/mnt/home/user41/URBench/data"
SPLITS     = f"{BASE}/sdfr_splits"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT        = "/mnt/home/user41/URBench/outputs/sdfr/probe_strategyqa_factretrieval_test50.jsonl"
TOP_K_FACTS = 3

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm_gold(a):
    if isinstance(a, bool): return "ہاں" if a else "نہیں"
    s = str(a).strip().lower()
    if s in ("true","yes","ہاں"):  return "ہاں"
    if s in ("false","no","نہیں"): return "نہیں"
    return str(a).strip()

def extract_answer(text):
    if "حتمی جواب" in text: text = text.split("حتمی جواب")[-1]
    ih, ina = text.rfind("ہاں"), text.rfind("نہیں")
    if ih == -1 and ina == -1:
        tl = text.lower()
        if "yes" in tl and "no" not in tl: return "ہاں"
        if "no" in tl and "yes" not in tl: return "نہیں"
        return ""
    return "ہاں" if ih > ina else "نہیں"

def build_prompt(tok, question, facts_list):
    if facts_list:
        facts_block = "حقائق:\n" + "\n".join(f"- {f}" for f in facts_list) + "\n"
        instr = ("آپ کو ایک سوال اور اس سے متعلق حقائق دیے گئے ہیں۔\n"
                 "پہلے مرحلہ وار اور منطقی طور پر حقائق کی بنیاد پر سوچیں۔\n"
                 "اس کے بعد آخر میں صرف ایک لفظ میں جواب دیں: ہاں یا نہیں۔\n"
                 "کوئی وضاحت، جملہ، یا اضافی متن شامل نہ کریں۔")
        raw = f"{instr}\n{facts_block}سوال: {question}\nسوچنے کے مراحل:\nحتمی جواب:"
    else:
        instr = ("آپ کو ایک سوال دیا گیا ہے۔\n"
                 "پہلے مرحلہ وار اور منطقی طور پر سوچیں۔\n"
                 "اس کے بعد آخر میں صرف ایک لفظ میں جواب دیں: ہاں یا نہیں۔\n"
                 "کوئی وضاحت، جملہ، یا اضافی متن شامل نہ کریں۔")
        raw = f"{instr}\nسوال: {question}\nسوچنے کے مراحل:\nحتمی جواب:"
    return tok.apply_chat_template([{"role":"user","content":raw}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=True)

if __name__ == "__main__":
    eval_data = read_jsonl(f"{SPLITS}/strategyqa_eval.jsonl")[:50]
    pool      = read_jsonl(f"{SPLITS}/strategyqa_pool.jsonl")

    # Build a fact corpus from POOL questions' gold facts (flatten, keep provenance)
    fact_corpus = []
    for item in pool:
        for f in item.get("facts", []):
            if f and f.strip():
                fact_corpus.append(f.strip())
    fact_corpus = list(dict.fromkeys(fact_corpus))  # dedupe, preserve order
    print(f"Eval: {len(eval_data)} | Fact corpus size: {len(fact_corpus)}")

    embedder = SentenceTransformer(EMBED_PATH)
    fact_vecs = embedder.encode(fact_corpus, normalize_embeddings=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(fact_vecs.shape[1]); index.add(fact_vecs)

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp  = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    def run(condition):
        prompts, metas = [], []
        for item in eval_data:
            q = item["question"]
            if condition == "nofacts":
                facts = []
            elif condition == "gold":
                facts = item.get("facts", [])
            else:  # retrieved
                qv = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
                _, ids = index.search(qv, TOP_K_FACTS)
                facts = [fact_corpus[i] for i in ids[0]]
            prompts.append(build_prompt(tok, q, facts))
            metas.append(facts)
        outs = llm.generate(prompts, sp)
        correct = 0; rows = []
        for item, out, facts in zip(eval_data, outs, metas):
            gen = out.outputs[0].text.strip()
            pred = extract_answer(gen.split("</think>")[-1])
            gold = norm_gold(item.get("answer"))
            ok = (pred == gold) and pred != ""
            correct += ok
            rows.append({"condition":condition, "q":item["question"], "gold":gold,
                         "pred":pred, "correct":ok, "facts_used":facts})
        return correct, rows

    all_rows = []; summary = {}
    for cond in ["nofacts", "gold", "retrieved"]:
        c, rows = run(cond); all_rows += rows
        summary[cond] = f"{c}/50 = {c/50*100:.1f}%"
        print(f"[{cond:10s}] {summary[cond]}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in all_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== PROBE SUMMARY (TEST50, noisy — read the gaps, not absolute values) ===")
    print(f"  1. NO facts   (baseline floor): {summary['nofacts']}")
    print(f"  2. GOLD facts (ceiling):        {summary['gold']}")
    print(f"  3. RETRIEVED  (optimistic sim): {summary['retrieved']}")
    print("Interpretation: if (3) is near (1) -> naive fact-retrieval fails, entity-aware")
    print("method has headroom to prove itself. If (3) near (2) -> gap is easy, method may")
    print("be unnecessary. If (3) < (1) -> retrieval hurts (matches RAG), hard challenge.")
