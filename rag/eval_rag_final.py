import json
import os
import re
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

# ─── CONFIG ────────────────────────────────────────────────
INDEX_PATH    = os.path.expanduser("~/URBench/rag/index/wikipedia.index")
CHUNKS_PATH   = os.path.expanduser("~/URBench/rag/index/chunks.jsonl")
MODEL_PATH    = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH    = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STRATEGYQA_UR = os.path.expanduser("~/URBench/data/strategyqa_raw/strategyQA_train_ur2_norm.jsonl")
STRATEGYQA_EN = os.path.expanduser("~/URBench/data/strategyqa_raw/strategyQA_train.json")
OUTPUT_FILE   = os.path.expanduser("~/URBench/rag/outputs/rag_strategyqa_qwen3_14b_final.jsonl")
TOP_K         = 3
MAX_MODEL_LEN = 4096
# ───────────────────────────────────────────────────────────

def load_chunks(path):
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def retrieve(question_en, embed_model, index, chunks, top_k=TOP_K):
    query = "query: " + question_en
    emb = embed_model.encode([query], normalize_embeddings=True)
    _, indices = index.search(emb.astype(np.float32), top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def build_rag_prompt(question_ur, passages):
    # format retrieved passages as facts — same structure as zero_shot.txt
    facts_text = ""
    for p in passages:
        facts_text += f"- {p['title']}: {p['text'][:200]}\n"

    prompt = (
        "آپ کو ایک سوال اور متعلقہ حقائق دیے گئے ہیں۔\n"
        "صرف انہی حقائق کی بنیاد پر فیصلہ کریں کہ جواب \"ہاں\" ہے یا \"نہیں\"۔\n"
        "صرف ایک لفظ میں جواب دیں: ہاں یا نہیں۔\n"
        "کوئی وضاحت، جملہ، یا اضافی متن شامل نہ کریں۔\n\n"
        f"حقائق:\n{facts_text}\n"
        f"سوال: {question_ur}\n"
        "جواب:"
    )
    return prompt

def format_qwen3_prompts(tokenizer, raw_prompts):
    formatted = []
    for raw_prompt in raw_prompts:
        messages = [{"role": "user", "content": raw_prompt}]
        f = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        formatted.append(f)
    return formatted

def normalize_answer(text):
    if not text:
        return None
    t = text.strip().replace('"', "").replace("'", "").strip()
    if not t:
        return None
    cleaned = re.sub(r"[^\u0600-\u06FF\s]", " ", t)
    cleaned = " ".join(cleaned.split())
    if cleaned == "ہاں": return "ہاں"
    if cleaned == "نہیں": return "نہیں"
    if cleaned.startswith("ہا"): return "ہاں"
    if cleaned.startswith("نہ"): return "نہیں"
    has_haan = ("ہاں" in cleaned) or ("ہا" in cleaned)
    has_nahi = ("نہیں" in cleaned) or cleaned.startswith("نہ")
    if has_nahi and not has_haan: return "نہیں"
    if has_haan and not has_nahi: return "ہاں"
    tl = t.lower()
    if "yes" in tl and "no" not in tl: return "ہاں"
    if "no" in tl and "yes" not in tl: return "نہیں"
    return None

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_PATH)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Index: {index.ntotal} vectors")

    print("Loading StrategyQA...")
    items = []
    with open(STRATEGYQA_UR) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    with open(STRATEGYQA_EN) as f:
        en_data = json.load(f)
    en_map = {item["qid"]: item["question"] for item in en_data}

    print(f"Total questions: {len(items)}")

    print("Loading Qwen3-14B...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_num_seqs=4,
        enforce_eager=True,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        stop=["\n"]
    )

    # resume support
    done_qids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                done_qids.add(json.loads(line)["qid"])
    print(f"Already done: {len(done_qids)}")

    # filter remaining
    remaining = [item for item in items if item["qid"] not in done_qids]
    print(f"Remaining: {len(remaining)}")

    correct = 0
    answered = 0
    total = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for item in tqdm(remaining, desc="RAG Final"):
            qid = item["qid"]
            question_ur = item["question"]
            question_en = en_map.get(qid, question_ur)
            gold = item["answer"]

            # retrieve using English question
            passages = retrieve(question_en, embed_model, index, chunks)

            # build prompt
            raw_prompt = build_rag_prompt(question_ur, passages)
            formatted = format_qwen3_prompts(tokenizer, [raw_prompt])

            # generate
            output = llm.generate(formatted, sampling)
            raw = output[0].outputs[0].text.strip()
            predicted = normalize_answer(raw)

            total += 1
            if predicted:
                answered += 1
                if predicted == gold:
                    correct += 1

            out_f.write(json.dumps({
                "qid": qid,
                "question_ur": question_ur,
                "question_en": question_en,
                "gold": gold,
                "predicted": predicted,
                "raw_output": raw,
                "passages": [{"title": p["title"], "text": p["text"][:150]} for p in passages]
            }, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\n{'='*50}")
    print(f"Total     : {total}")
    print(f"Answered  : {answered} ({100*answered/total:.2f}%)")
    print(f"Correct   : {correct}")
    print(f"Accuracy  : {100*correct/total:.2f}%")
    if answered:
        print(f"Ans. Acc. : {100*correct/answered:.2f}%")
    print(f"Saved to  : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
