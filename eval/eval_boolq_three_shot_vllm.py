import json
import os
from vllm import LLM, SamplingParams

# ==========================
# CONFIG – EDIT PATHS IF NEEDED
# ==========================

# Path to your BoolQ Urdu JSONL file
BOOLQ_PATH = "../data/boolq_raw/boolq_train_1550_ur_fixed.jsonl"

# Path to your 3-shot BoolQ prompt template
# (Should contain the literal strings {passage} and {question})
PROMPT_PATH = "../prompts/boolq/three_shot.txt"

# Model to use (DeepSeek 7B)
MODEL_NAME = "/mnt/home/user41/downloaded_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Output directory
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/boolq/deepseek_r1_distill_qwen_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = None  # None = all

# vLLM / context settings
MAX_MODEL_LEN = 3072          # same setting style as your zero-shot
MAX_NEW_TOKENS = 8            # BoolQ answer is short
SAFETY_TOKENS = 64            # extra headroom for special tokens / formatting

# ADDED: DeepSeek detection flag
IS_DEEPSEEK = "deepseek" in MODEL_NAME.lower()


# ==========================
# HELPER FUNCTIONS
# ==========================

def load_boolq(path):
    """Load BoolQ Urdu JSONL: expects qid, question, passage, answer."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append(obj)
    return data


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ADDED: DeepSeek answer extraction function for BoolQ
def extract_deepseek_boolq_answer(text):
    """Extract 'ہاں' or 'نہیں' from DeepSeek's output that might contain reasoning."""
    if not text:
        return text
    
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check for exact matches
        if line == "ہاں" or line == "نہیں":
            return line
        
        # Check for truncated versions
        if line.startswith("ہا"):
            return "ہاں"
        if line.startswith("نہ"):
            return "نہیں"
        
        # Check for patterns like "Answer: ہاں" or "جواب: نہیں"
        if "ہاں" in line:
            return "ہاں"
        if "نہیں" in line or "نھیں" in line or "نہيں" in line:
            return "نہیں"
    
    # English fallback
    t_lower = text.lower()
    if "yes" in t_lower:
        return "ہاں"
    if "no" in t_lower:
        return "نہیں"
    
    return text


def normalize_boolq_output(text):
    """
    Map model output to exactly 'ہاں' or 'نہیں'.
    We keep it simple and robust to tiny variations.
    """
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
    return (
        template
        .replace("{passage}", passage)
        .replace("{question}", question)
    )


def truncate_passage_to_fit(template: str, passage: str, question: str, tokenizer, max_model_len: int):
    """
    If template+passage+question is too long, truncate ONLY the passage
    so the final prompt fits within max_model_len - MAX_NEW_TOKENS - SAFETY_TOKENS.
    Returns (final_prompt, was_truncated, orig_len, final_len)
    """
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

    passage_budget = budget - overhead
    if passage_budget < min_passage_tokens:
        passage_budget = min_passage_tokens

    passage_ids_trunc = passage_ids[:passage_budget]
    passage_trunc = tokenizer.decode(passage_ids_trunc)

    final_prompt = build_prompt(template, passage_trunc, question)
    final_len = len(tokenizer.encode(final_prompt))

    while final_len > budget and len(passage_ids_trunc) > min_passage_tokens:
        passage_ids_trunc = passage_ids_trunc[: max(min_passage_tokens, len(passage_ids_trunc) - 32)]
        passage_trunc = tokenizer.decode(passage_ids_trunc)
        final_prompt = build_prompt(template, passage_trunc, question)
        final_len = len(tokenizer.encode(final_prompt))

    return final_prompt, True, orig_len, final_len


# ==========================
# MAIN
# ==========================

def main():
    print("Loading BoolQ dataset...")
    examples = load_boolq(BOOLQ_PATH)
    print(f"Total examples in file: {len(examples)}")
    print(f"DeepSeek model detected: {IS_DEEPSEEK}")

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]
        print(f"Using only first {len(examples)} examples for this run.")

    print("Loading 3-shot template...")
    template = load_prompt_template(PROMPT_PATH)

    print(f"Loading model with vLLM: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        max_num_seqs=2,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()

    print("Building prompts (with truncation if needed)...")
    prompts = []
    trunc_count = 0
    for ex in examples:
        p, truncated, orig_len, final_len = truncate_passage_to_fit(
            template=template,
            passage=ex["passage"],
            question=ex["question"],
            tokenizer=tokenizer,
            max_model_len=MAX_MODEL_LEN,
        )
        if truncated:
            trunc_count += 1
        prompts.append(p)

    print(f"Truncated passages for {trunc_count}/{len(examples)} examples to fit context window.")

    # CHANGED: Different sampling params for DeepSeek
    if IS_DEEPSEEK:
        print("Using DeepSeek-specific generation parameters")
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,  # Higher for reasoning
            # No stop parameter
        )
    else:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
            stop=["\n"],
        )

    print("Generating answers...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    total = 0

    out_path = os.path.join(OUTPUT_DIR, "boolq_three_shot_deepseek_r1_distill_qwen_7b.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            
            # ADDED: For DeepSeek, try to extract answer from full output
            if IS_DEEPSEEK:
                raw_for_pred = extract_deepseek_boolq_answer(raw)
            else:
                raw_for_pred = raw
                
            pred = normalize_boolq_output(raw_for_pred)
            gold = ex["answer"].strip()

            is_correct = None
            if pred is not None:
                total += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1

            record = {
                "qid": ex["qid"],
                "question": ex["question"],
                "passage": ex["passage"],
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": is_correct,
                "prompt": prompt,
                "raw_output": raw,
                "model_name": MODEL_NAME,
                "prompt_type": "three_shot",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = (correct / total * 100) if total > 0 else 0.0
    print("\n=== BoolQ 3-shot with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"DeepSeek mode:     {IS_DEEPSEEK}")
    print(f"Used items:        {len(examples)}")
    print(f"Scored on:         {total} (pred != None)")
    print(f"Accuracy:          {acc:.2f}%")
    print(f"Outputs ->         {out_path}")


if __name__ == "__main__":
    main()