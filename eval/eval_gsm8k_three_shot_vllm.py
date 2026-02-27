import json
import os
import re
from decimal import Decimal, InvalidOperation

from vllm import LLM, SamplingParams

# ==========================
# CONFIG
# ==========================

# Path to your GSM8K Urdu JSONL file
GSM8K_PATH = "../data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl"

# 3-shot prompt template (UPDATED to enforce #### N)
PROMPT_PATH = "../prompts/gsm8k/three_shot_llama.txt"

# CHANGED: Model name to DeepSeek
MODEL_NAME = "/mnt/home/user41/downloaded_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# CHANGED: Output dir to DeepSeek
OUTPUT_DIR = "../outputs/gsm8k/deepseek_r1_distill_qwen_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# For full run, keep None
MAX_EXAMPLES = None

# ADDED: DeepSeek detection flag
IS_DEEPSEEK = "deepseek" in MODEL_NAME.lower()


# ==========================
# HELPERS
# ==========================

def load_gsm8k(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
    return data


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompts(template, examples):
    prompts = []
    for ex in examples:
        q = ex["question"]
        prompt = template.format(question=q)
        prompts.append(prompt)
    return prompts


# FIXED: DeepSeek answer extraction function for GSM8K
def extract_deepseek_gsm8k_answer(text):
    """
    Extract the final answer number from DeepSeek's output.
    Returns just the number as a string.
    """
    if not text:
        return None
    
    # First try to find #### pattern (standard GSM8K format)
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    
    # Try "Answer:" or "جواب:" patterns
    m = re.search(r"(Answer|جواب)\s*:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(2)
    
    # Look at the last few lines for the answer
    lines = text.strip().split('\n')
    for line in reversed(lines[-10:]):  # Check last 10 lines
        line = line.strip()
        if not line:
            continue
        
        # Find numbers in this line
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", ""))
        if numbers:
            # Return the last number found
            return numbers[-1]
    
    return None  # Return None if no number found


def _normalize_number_string(num_str: str):
    try:
        d = Decimal(num_str)
    except InvalidOperation:
        return None

    if d == d.to_integral():
        return str(d.to_integral())
    return format(d.normalize(), "f")


def extract_number_from_hashes(text: str):
    if not text:
        return None
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return _normalize_number_string(m.group(1))
    return None


def normalize_gsm8k_output(text: str):
    # IMPORTANT: ONLY accept #### N format (no last-number fallback)
    return extract_number_from_hashes(text)


def extract_gold_number(answer_text: str):
    # Gold already contains #### N
    return extract_number_from_hashes(answer_text)


# ==========================
# MAIN
# ==========================

def main():
    print("Loading GSM8K dataset...")
    examples = load_gsm8k(GSM8K_PATH)
    print(f"Total examples in file: {len(examples)}")
    print(f"DeepSeek model detected: {IS_DEEPSEEK}")

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]
        print(f"Using only first {len(examples)} examples for this run.")

    print("Loading 3-shot template...")
    template = load_prompt_template(PROMPT_PATH)

    print("Building prompts...")
    prompts = build_prompts(template, examples)

    print(f"Loading model with vLLM: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=3072,
        gpu_memory_utilization=0.75,
        max_num_seqs=1,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
    )

    # CHANGED: Different sampling params for DeepSeek
    if IS_DEEPSEEK:
        print("Using DeepSeek-specific generation parameters")
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,  # Much higher for math reasoning
            # No stop parameter
        )
    else:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,   # enough for reasoning + final #### N
        )

    print("Generating answers (3-shot)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    total = len(examples)

    # CHANGED: Output filename
    out_path = os.path.join(
        OUTPUT_DIR,
        "gsm8k_three_shot_deepseek_r1_distill_qwen_7b.jsonl",
    )

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, (ex, prompt, out) in enumerate(zip(examples, prompts, outputs)):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            
            # FIXED: For DeepSeek, extract just the number, then normalize
            if IS_DEEPSEEK:
                extracted_number = extract_deepseek_gsm8k_answer(raw)
                if extracted_number is None:
                    pred_num = None
                else:
                    # Wrap in #### format for the normalizer
                    wrapped = f"#### {extracted_number}"
                    pred_num = normalize_gsm8k_output(wrapped)
            else:
                pred_num = normalize_gsm8k_output(raw)

            gold_text = ex.get("answer", "")
            gold_num = extract_gold_number(gold_text)

            is_correct = None
            if pred_num is not None and gold_num is not None:
                answered += 1
                is_correct = (pred_num == gold_num)
                if is_correct:
                    correct += 1

            record = {
                "idx": idx,
                "question": ex["question"],
                "gold_answer_text": gold_text,
                "gold_number": gold_num,
                "pred_raw": raw,
                "pred_number": pred_num,
                "correct": is_correct,
                "prompt": prompt,
                "model_name": MODEL_NAME,
                "prompt_type": "three_shot",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc_overall = (correct / total * 100) if total > 0 else 0.0
    acc_answered = (correct / answered * 100) if answered > 0 else 0.0

    print("\n=== GSM8K 3-shot with vLLM ===")
    print(f"Model:              {MODEL_NAME}")
    print(f"DeepSeek mode:      {IS_DEEPSEEK}")
    print(f"Used items:         {total}")
    print(f"Answered (numeric): {answered} ({answered / total * 100:.2f}%)")
    print(f"Accuracy overall:   {acc_overall:.2f}%")
    print(f"Accuracy answered:  {acc_answered:.2f}%")
    print(f"Outputs ->          {out_path}")


if __name__ == "__main__":
    main()