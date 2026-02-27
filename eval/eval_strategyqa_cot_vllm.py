import json
import os
import re

from vllm import LLM, SamplingParams

# ==========================
# CONFIG
# ==========================

STRATEGYQA_PATH = "../data/strategyqa_raw/strategyQA_train_ur2_norm.jsonl"
PROMPT_PATH = "../prompts/strategyqa/cot.txt"

# CHANGED: Model name to DeepSeek
MODEL_NAME = "/mnt/home/user41/downloaded_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# CHANGED: Output dir to DeepSeek
OUTPUT_DIR = "../outputs/strategyqa/deepseek_r1_distill_qwen_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = None

# ADDED: DeepSeek detection flag
IS_DEEPSEEK = "deepseek" in MODEL_NAME.lower()


# ==========================
# HELPERS
# ==========================

def load_strategyqa(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_facts(facts):
    return "\n".join(f"- {fact}" for fact in facts)


def build_prompts(template, examples):
    prompts = []
    for ex in examples:
        prompt = template.format(
            facts=format_facts(ex.get("facts", [])),
            question=ex["question"],
        )
        prompts.append(prompt)
    return prompts


# ADDED: DeepSeek answer extraction function for StrategyQA
def extract_deepseek_strategyqa_answer(text):
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
        cleaned = re.sub(r"[^\u0600-\u06FF\s]", " ", line)
        cleaned = " ".join(cleaned.split())
        
        if cleaned.startswith("ہا"):
            return "ہاں"
        if cleaned.startswith("نہ"):
            return "نہیں"
        
        # Check for patterns like "Answer: ہاں" or "جواب: نہیں"
        m = re.search(r"(Answer|جواب)\s*:\s*([ہن][اہیں]*)", line, re.IGNORECASE)
        if m:
            ans = m.group(2).strip()
            if ans.startswith("ہا"):
                return "ہاں"
            if ans.startswith("نہ"):
                return "نہیں"
    
    # English fallback for the whole text
    t_lower = text.lower()
    if "yes" in t_lower and "no" not in t_lower:
        return "ہاں"
    if "no" in t_lower and "yes" not in t_lower:
        return "نہیں"
    
    return text


def normalize_strategyqa_output(text: str):
    if not text:
        return None

    t = text.strip().replace('"', "").replace("'", "").strip()
    if not t:
        return None

    cleaned = re.sub(r"[^\u0600-\u06FF\s]", " ", t)
    cleaned = " ".join(cleaned.split())

    if cleaned == "ہاں":
        return "ہاں"
    if cleaned == "نہیں":
        return "نہیں"

    # tolerate common truncations
    if cleaned.startswith("ہا"):
        return "ہاں"
    if cleaned.startswith("نہ"):
        return "نہیں"

    has_haan = ("ہاں" in cleaned) or ("ہا" in cleaned)
    has_nahi = ("نہیں" in cleaned) or cleaned.startswith("نہ")

    if has_nahi and not has_haan:
        return "نہیں"
    if has_haan and not has_nahi:
        return "ہاں"

    tl = t.lower()
    if "yes" in tl and "no" not in tl:
        return "ہاں"
    if "no" in tl and "yes" not in tl:
        return "نہیں"

    return None


# ==========================
# MAIN
# ==========================

def main():
    print("Loading StrategyQA dataset...")
    examples = load_strategyqa(STRATEGYQA_PATH)
    print(f"Total examples in file: {len(examples)}")
    print(f"DeepSeek model detected: {IS_DEEPSEEK}")

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]
        print(f"Using only first {len(examples)} examples for this run.")

    template = load_prompt_template(PROMPT_PATH)
    prompts = build_prompts(template, examples)

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
            max_tokens=128,  # allow reasoning
        )

    print("Generating answers (CoT)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0

    # CHANGED: Output filename
    out_path = os.path.join(
        OUTPUT_DIR,
        "strategyqa_cot_deepseek_r1_distill_qwen_7b.jsonl",
    )

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            
            # ADDED: For DeepSeek, try to extract answer from full output
            if IS_DEEPSEEK:
                raw_for_pred = extract_deepseek_strategyqa_answer(raw)
            else:
                raw_for_pred = raw
                
            pred = normalize_strategyqa_output(raw_for_pred)
            gold = ex.get("answer", "").strip()

            is_correct = None
            if pred is not None:
                answered += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1

            fout.write(json.dumps({
                "qid": ex.get("qid", ""),
                "question": ex["question"],
                "facts": ex.get("facts", []),
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": is_correct,
                "prompt": prompt,
                "raw_output": raw,
                "prompt_type": "cot",
                "model_name": MODEL_NAME,
            }, ensure_ascii=False) + "\n")

    total = len(examples)
    print("\n=== StrategyQA CoT with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"DeepSeek mode:     {IS_DEEPSEEK}")
    print(f"Used items:        {total}")
    print(f"Answered (ہاں/نہیں): {answered} ({answered / total * 100:.2f}%)")
    print(f"Accuracy overall:  {correct / total * 100:.2f}%")
    print(f"Accuracy answered: {correct / answered * 100:.2f}%")
    print(f"Outputs ->         {out_path}")


if __name__ == "__main__":
    main()