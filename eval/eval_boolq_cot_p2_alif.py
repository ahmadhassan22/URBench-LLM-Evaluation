import json
import os
import re
from vllm import LLM, SamplingParams

BOOLQ_PATH   = "../data/boolq_raw/boolq_train_1550_ur_fixed.jsonl"
PROMPT_PATH  = "../prompts/boolq/cot_p2.txt"

MODEL_NAME   = "/mnt/home/user41/downloaded_models/Alif/Alif-1.0-8B-Merged"
OUTPUT_DIR   = "/mnt/home/user41/URBench/outputs/boolq/alif_1.0_8b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE  = "boolq_cot_p2_alif_test50.jsonl"

MAX_EXAMPLES       = 50
MAX_MODEL_LEN      = 4096
MAX_PASSAGE_CHARS  = 3000
MAX_QUESTION_CHARS = 512

def truncate_text(s, max_chars):
    if s is None: return ""
    s = s.strip()
    return s if len(s) <= max_chars else s[:max_chars]

def load_boolq(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))
    return data

def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompts(template, examples):
    prompts = []
    for ex in examples:
        passage  = truncate_text(ex["passage"],  MAX_PASSAGE_CHARS)
        question = truncate_text(ex["question"], MAX_QUESTION_CHARS)
        prompt = template.replace("{passage}", passage).replace("{question}", question)
        prompts.append(prompt)
    return prompts

def format_prompts(tokenizer, raw_prompts):
    formatted = []
    for raw in raw_prompts:
        messages = [{"role": "user", "content": raw}]
        formatted.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
    return formatted

def normalize_boolq_output(text):
    if text is None: return None
    t = text.strip().replace('"', "").replace("'", "").strip()
    if "ہاں" in t: return "ہاں"
    if "نہیں" in t or "نھیں" in t or "نہيں" in t: return "نہیں"
    low = t.lower()
    if "yes" in low: return "ہاں"
    if "no"  in low: return "نہیں"
    return None

def main():
    print("Loading BoolQ dataset...")
    examples = load_boolq(BOOLQ_PATH)
    if MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"Total examples: {len(examples)}")

    template = load_prompt_template(PROMPT_PATH)
    prompts  = build_prompts(template, examples)

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
    prompts   = format_prompts(tokenizer, prompts)

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=512,
        stop=["overposting", "\n\n\n"]
    )

    print("Generating answers (Alif CoT)...")
    outputs = llm.generate(prompts, sampling)

    correct  = 0
    total    = 0
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw  = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_boolq_output(raw)
            gold = ex["answer"].strip()
            is_correct = None
            if pred is not None:
                total += 1
                is_correct = (pred == gold)
                if is_correct: correct += 1
            record = {
                "qid":             ex["qid"],
                "question":        truncate_text(ex["question"], MAX_QUESTION_CHARS),
                "passage":         truncate_text(ex["passage"],  MAX_PASSAGE_CHARS),
                "gold_answer":     gold,
                "pred_answer":     pred,
                "correct":         is_correct,
                "prompt":          prompt,
                "raw_output":      raw,
                "model_name":      MODEL_NAME,
                "prompt_type":     "cot",
                "thinking_mode":   "disabled",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = (correct / total * 100) if total > 0 else 0.0
    print(f"\n=== BoolQ CoT — Alif-1.0-8B-Merged ===")
    print(f"Model:      {MODEL_NAME}")
    print(f"Used items: {len(examples)}")
    print(f"Scored on:  {total} (pred != None)")
    print(f"Accuracy:   {acc:.2f}%")
    print(f"Outputs ->  {out_path}")

if __name__ == "__main__":
    main()