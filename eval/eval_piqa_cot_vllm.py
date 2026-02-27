import json
import os
import re
import glob

from vllm import LLM, SamplingParams

# ==========================
# CONFIG
# ==========================

PIQA_PATH = "../data/piqa_raw/piqa_train_750_ur_fixed.jsonl"
PROMPT_PATH = "../prompts/piqa/cot.txt"

# CHANGED: Model name to DeepSeek
MODEL_NAME = "/mnt/home/user41/downloaded_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# CHANGED: Output dir to DeepSeek
OUTPUT_DIR = "../outputs/piqa/deepseek_r1_distill_qwen_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = None

# ADDED: DeepSeek detection flag
IS_DEEPSEEK = "deepseek" in MODEL_NAME.lower()

# ADDED: Output filename
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "piqa_cot_deepseek_r1_distill_qwen_7b.jsonl"
)


# ==========================
# HELPER FUNCTIONS
# ==========================

# ADDED: Function to get already processed QIDs
def get_processed_qids(output_path):
    """Extract QIDs from already processed examples in output file."""
    if not os.path.exists(output_path):
        return set()
    
    processed = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Try different possible ID fields
                    qid = data.get('qid') or data.get('id') or data.get('goal')[:50]
                    if qid:
                        processed.add(str(qid))
                except:
                    continue
    except Exception as e:
        print(f"Warning: Could not read existing output file: {e}")
    
    return processed


def load_piqa(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # ADDED: Ensure each example has a unique identifier
            if 'qid' not in obj:
                obj['qid'] = f"piqa_{len(data)}"
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
    all_examples = load_piqa(PIQA_PATH)
    print(f"Total examples in file: {len(all_examples)}")
    print(f"DeepSeek model detected: {IS_DEEPSEEK}")

    # ADDED: Check for already processed examples
    processed_qids = get_processed_qids(OUTPUT_FILE)
    if processed_qids:
        print(f"Found {len(processed_qids)} already processed examples, skipping...")
        examples = [ex for ex in all_examples if str(ex.get('qid', '')) not in processed_qids]
        print(f"Remaining to process: {len(examples)}")
    else:
        examples = all_examples
        print(f"No existing output found, processing all {len(examples)} examples")

    if MAX_EXAMPLES is not None and len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
        print(f"Limited to first {len(examples)} examples for this run.")

    if not examples:
        print("All examples already processed! Exiting.")
        return

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
            # no hard stop; let it finish
        )

    print("Generating answers (CoT)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    total_processed = 0

    # CHANGED: Open in append mode
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
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
            
            total_processed += 1

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
                "prompt_type": "cot",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()  # ADDED: Ensure data is written immediately

    # Calculate stats including previously processed examples
    all_processed_count = len(processed_qids) + total_processed
    total_items = len(all_examples)
    
    # For accuracy, we need to read the full output file
    final_correct = 0
    final_answered = 0
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('correct') is True:
                    final_correct += 1
                if data.get('pred_answer') is not None:
                    final_answered += 1
            except:
                continue

    acc_overall = (final_correct / total_items * 100) if total_items > 0 else 0.0
    acc_answered = (final_correct / final_answered * 100) if final_answered > 0 else 0.0

    print("\n=== PIQA CoT with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"DeepSeek mode:     {IS_DEEPSEEK}")
    print(f"Total items:       {total_items}")
    print(f"Processed this run: {total_processed}")
    print(f"Total processed:   {all_processed_count}")
    print(f"Answered (0/1):    {final_answered} ({final_answered/total_items*100:.2f}%)")
    print(f"Accuracy overall:  {acc_overall:.2f}%")
    print(f"Accuracy answered: {acc_answered:.2f}%")
    print(f"Outputs ->         {OUTPUT_FILE}")


if __name__ == "__main__":
    main()