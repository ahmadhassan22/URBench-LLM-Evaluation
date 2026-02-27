import json
import os
import re

from vllm import LLM, SamplingParams

# ==========================
# CONFIG
# ==========================

PIQA_PATH = "../data/piqa_raw/piqa_train_750_ur_fixed.jsonl"
PROMPT_PATH = "../prompts/piqa/three_shot.txt"

# CHANGED: Model name to DeepSeek
MODEL_NAME = "/mnt/home/user41/downloaded_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# CHANGED: Output dir to DeepSeek
OUTPUT_DIR = "../outputs/piqa/deepseek_r1_distill_qwen_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = None

# ADDED: DeepSeek detection flag
IS_DEEPSEEK = "deepseek" in MODEL_NAME.lower()


# ==========================
# HELPER FUNCTIONS
# ==========================

def load_piqa(path):
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
        goal = ex["goal"]
        sol1 = ex["sol1"]
        sol2 = ex["sol2"]

        prompt = (
            template
            .replace("{goal}", goal)
            .replace("{sol1}", sol1)
            .replace("{sol2}", sol2)
        )
        prompts.append(prompt)

    return prompts


# ADDED: DeepSeek answer extraction function for PIQA
def extract_deepseek_piqa_answer(text):
    """Extract 0 or 1 from DeepSeek's output that might contain reasoning."""
    if not text:
        return text
    
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        
        if line == "0" or line == "1":
            return line
        
        words = line.split()
        for word in words:
            word_clean = word.strip('.:;,!?')
            if word_clean == "0" or word_clean == "1":
                return word_clean
    
    m = re.search(r"(Answer|جواب)\s*:\s*([01])", text, re.IGNORECASE)
    if m:
        return m.group(2)
    
    return text


def normalize_piqa_output(text: str):
    if not text:
        return None

    t = text.strip()
    if not t:
        return None

    t = " ".join(t.split())

    m = re.search(r"\b([01])\b", t)
    if m:
        return m.group(1)

    first = t[0]
    if first in ("0", "1"):
        return first

    return None


# ==========================
# MAIN
# ==========================

def main():
    print("Loading PIQA dataset...")
    examples = load_piqa(PIQA_PATH)
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
            max_tokens=256,  # Higher for reasoning
            # No stop parameter
        )
    else:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
            stop=["\n"],
        )

    print("Generating answers (3-shot)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    total = len(examples)

    # CHANGED: Output filename
    out_path = os.path.join(
        OUTPUT_DIR,
        "piqa_three_shot_deepseek_r1_distill_qwen_7b.jsonl",
    )

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            
            # ADDED: For DeepSeek, try to extract answer from full output
            if IS_DEEPSEEK:
                raw_for_pred = extract_deepseek_piqa_answer(raw)
            else:
                raw_for_pred = raw
                
            pred = normalize_piqa_output(raw_for_pred)

            gold = str(ex.get("label", ex.get("answer", ""))).strip()

            is_correct = None
            if pred is not None and gold in ("0", "1"):
                answered += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1

            record = {
                "qid": ex.get("qid", ""),
                "goal": ex["goal"],
                "sol1": ex["sol1"],
                "sol2": ex["sol2"],
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": is_correct,
                "prompt": prompt,
                "raw_output": raw,
                "model_name": MODEL_NAME,
                "prompt_type": "three_shot",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc_overall = (correct / total * 100) if total > 0 else 0.0
    acc_answered = (correct / answered * 100) if answered > 0 else 0.0

    print("\n=== PIQA 3-shot with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"DeepSeek mode:     {IS_DEEPSEEK}")
    print(f"Used items:        {total}")
    print(f"Answered (0/1):    {answered} ({answered / total * 100:.2f}%)")
    print(f"Accuracy overall:  {acc_overall:.2f}%")
    print(f"Accuracy answered: {acc_answered:.2f}%")
    print(f"Outputs ->         {out_path}")


if __name__ == "__main__":
    main()