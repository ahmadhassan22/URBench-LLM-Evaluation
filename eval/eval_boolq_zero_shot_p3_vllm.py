import json
import os
import re
from vllm import LLM, SamplingParams

BOOLQ_PATH = "../data/boolq_raw/boolq_train_1550_ur_fixed.jsonl"
PROMPT_PATH = "../prompts/boolq/zero_shot_p3.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/boolq/qwen3_14b_p3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "boolq_zero_shot_p3_qwen3_14b.jsonl"

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 8
SAFETY_TOKENS = 64

def load_boolq(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data

def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_boolq_output(text):
    if text is None:
        return None
    t = text.strip().replace('"', "").replace("'", "").strip()
    if "ہاں" in t:
        return "ہاں"
    if "نہیں" in t or "نھیں" in t or "نہيں" in t:
        return "نہیں"
    low = t.lower()
    if "yes" in low:
        return "ہاں"
    if "no" in low:
        return "نہیں"
    return None

def build_prompt(template: str, passage: str, question: str) -> str:
    return template.format(passage=passage, question=question)

def format_qwen3_prompt(tokenizer, raw_prompt: str) -> str:
    messages = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )

def truncate_passage_to_fit(template, passage, question, tokenizer, max_model_len):
    budget = max_model_len - MAX_NEW_TOKENS - SAFETY_TOKENS
    if budget < 256:
        budget = 256
    full_prompt = build_prompt(template, passage, question)
    full_ids = tokenizer.encode(full_prompt)
    orig_len = len(full_ids)
    if orig_len <= budget:
        return full_prompt, False, orig_len, orig_len
    passage_ids = tokenizer.encode(passage)
    min_passage_tokens = 64
    empty_prompt = build_prompt(template, "", question)
    overhead = len(tokenizer.encode(empty_prompt))
    passage_budget = max(budget - overhead, min_passage_tokens)
    passage_ids_trunc = passage_ids[:passage_budget]
    passage_trunc = tokenizer.decode(passage_ids_trunc)
    final_prompt = build_prompt(template, passage_trunc, question)
    final_len = len(tokenizer.encode(final_prompt))
    while final_len > budget and len(passage_ids_trunc) > min_passage_tokens:
        passage_ids_trunc = passage_ids_trunc[:max(min_passage_tokens, len(passage_ids_trunc) - 32)]
        passage_trunc = tokenizer.decode(passage_ids_trunc)
        final_prompt = build_prompt(template, passage_trunc, question)
        final_len = len(tokenizer.encode(final_prompt))
    return final_prompt, True, orig_len, final_len

def main():
    print("Loading BoolQ dataset...")
    examples = load_boolq(BOOLQ_PATH)
    print(f"Total examples: {len(examples)}")

    template = load_prompt_template(PROMPT_PATH)

    print(f"Loading model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()

    prompts = []
    trunc_count = 0
    for ex in examples:
        p, truncated, orig_len, final_len = truncate_passage_to_fit(
            template, ex["passage"], ex["question"], tokenizer, MAX_MODEL_LEN)
        if truncated:
            trunc_count += 1
        p = format_qwen3_prompt(tokenizer, p)
        prompts.append(p)

    print(f"Truncated: {trunc_count}/{len(examples)}")

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0,
        max_tokens=MAX_NEW_TOKENS, stop=["\n"]
    )

    outputs = llm.generate(prompts, sampling)

    correct = 0
    total = 0
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_boolq_output(raw)
            gold = ex["answer"].strip()
            is_correct = None
            if pred is not None:
                total += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1
            record = {
                "qid": ex["qid"], "question": ex["question"],
                "passage": ex["passage"], "gold_answer": gold,
                "pred_answer": pred, "correct": is_correct,
                "prompt": prompt, "raw_output": raw,
                "model_name": MODEL_NAME, "prompt_type": "zero_shot_p3",
                "thinking_mode": "disabled", "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = (correct / total * 100) if total > 0 else 0.0
    print("\n=== BoolQ Zero-shot P3 — Qwen3-14B ===")
    print(f"Model:      {MODEL_NAME}")
    print(f"Used items: {len(examples)}")
    print(f"Scored on:  {total} (pred != None)")
    print(f"Accuracy:   {acc:.2f}%")
    print(f"Outputs ->  {out_path}")

if __name__ == "__main__":
    main()