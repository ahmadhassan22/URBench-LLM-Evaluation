import json
import os
from vllm import LLM, SamplingParams

# ==========================
# CONFIG – EDIT PATHS IF NEEDED
# ==========================

# Path to your BoolQ Urdu JSONL file
BOOLQ_PATH = "../data/boolq_raw/boolq_train_1550_ur_fixed.jsonl"

# Path to your CoT BoolQ prompt template
PROMPT_PATH = "../prompts/boolq/cot.txt"

# DeepSeek 7B (new model path)
MODEL_NAME = "/mnt/home/user41/downloaded_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Updated output directory (new model)
OUTPUT_DIR = "../outputs/boolq/deepseek_r1_distill_qwen_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = None

MAX_PASSAGE_CHARS = 3000
MAX_QUESTION_CHARS = 512

# ADDED: DeepSeek detection flag
IS_DEEPSEEK = "deepseek" in MODEL_NAME.lower()


# ==========================
# HELPER FUNCTIONS
# ==========================

def truncate_text(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def load_boolq(path):
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


def build_prompts(template, examples):
    prompts = []
    for ex in examples:
        passage = truncate_text(ex["passage"], MAX_PASSAGE_CHARS)
        question = truncate_text(ex["question"], MAX_QUESTION_CHARS)
        prompt = (
            template
            .replace("{passage}", passage)
            .replace("{question}", question)
        )
        prompts.append(prompt)
    return prompts


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

    print("Loading CoT template...")
    template = load_prompt_template(PROMPT_PATH)

    print("Building prompts...")
    prompts = build_prompts(template, examples)

    print(f"Loading model with vLLM: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        enforce_eager=True,
        disable_log_stats=True,
    )

    # CHANGED: Different sampling params for DeepSeek
    if IS_DEEPSEEK:
        print("Using DeepSeek-specific generation parameters")
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,  # Higher for reasoning
            # No stop parameter
        )
    else:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            stop=None,
        )

    print("Generating answers (CoT)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    total = 0

    out_path = os.path.join(
        OUTPUT_DIR,
        "boolq_cot_deepseek_r1_distill_qwen_7b.jsonl",
    )
    fout = open(out_path, "w", encoding="utf-8")

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

        logged_passage = truncate_text(ex["passage"], MAX_PASSAGE_CHARS)
        logged_question = truncate_text(ex["question"], MAX_QUESTION_CHARS)

        record = {
            "qid": ex["qid"],
            "question": logged_question,
            "passage": logged_passage,
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": is_correct,
            "prompt": prompt,
            "raw_output": raw,
            "model_name": MODEL_NAME,
            "prompt_type": "cot",
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    fout.close()

    acc = (correct / total * 100) if total > 0 else 0.0

    print("\n=== BoolQ CoT with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"DeepSeek mode:     {IS_DEEPSEEK}")
    print(f"Used items:        {len(examples)}")
    print(f"Scored on:         {total} (pred != None)")
    print(f"Accuracy:          {acc:.2f}%")
    print(f"Outputs ->         {out_path}")


if __name__ == "__main__":
    main()