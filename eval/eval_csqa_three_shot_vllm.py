import json
import os
import re
from collections import defaultdict

from vllm import LLM, SamplingParams

# ==========================
# CONFIG
# ==========================

CSQA_PATH = "../data/csqa_raw/csqa_train_1500_ur.jsonl"
PROMPT_PATH = "../prompts/csqa/three_shot.txt"

# Updated model (DeepSeek 7B)
MODEL_NAME = "/mnt/home/user41/downloaded_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Updated output directory
OUTPUT_DIR = "../outputs/csqa/deepseek_r1_distill_qwen_7b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = None

# ==========================
# DETECT IF MODEL IS DEEPSEEK
# ==========================
IS_DEEPSEEK = "deepseek" in MODEL_NAME.lower()


# ==========================
# HELPER FUNCTIONS
# ==========================

def load_csqa(path):
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


def extract_choice_mapping(choices_field):
    mapping = {}

    if isinstance(choices_field, dict) and "label" in choices_field and "text" in choices_field:
        labels = choices_field["label"]
        texts = choices_field["text"]
        for lab, txt in zip(labels, texts):
            mapping[str(lab).strip()] = str(txt).strip()
        return mapping

    labels_default = ["A", "B", "C", "D", "E"]
    if isinstance(choices_field, list):
        for idx, ch in enumerate(choices_field):
            default_lab = labels_default[idx] if idx < len(labels_default) else str(idx)
            if isinstance(ch, dict):
                lab = (ch.get("label") or default_lab).strip()
                txt = str(ch.get("text", "")).strip()
            else:
                lab = default_lab
                txt = str(ch).strip()
            mapping[lab] = txt

    return mapping


def build_prompts(template, examples):
    prompts = []

    for ex in examples:
        question = ex["question"]
        choices = ex["choices"]
        mapping = extract_choice_mapping(choices)

        fmt = defaultdict(str)
        fmt["question"] = question

        for lab, txt in mapping.items():
            fmt[lab] = txt

        prompt = template.format_map(fmt)
        prompts.append(prompt)

    return prompts


URDU_LETTER_MAP = {
    "الف": "A",
    "ب": "B",
    "ج": "C",
    "د": "D",
    "ہ": "E",
    "ھ": "E",
}

DIGIT_TO_LABEL = {
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
    "۱": "A", "۲": "B", "۳": "C", "۴": "D", "۵": "E",
    "١": "A", "٢": "B", "٣": "C", "٤": "D", "٥": "E",
}


def normalize_csqa_output(text, choices):
    if not text:
        return None

    t = text.strip()
    if not t:
        return None

    t = " ".join(t.split())

    for ch in t:
        u = ch.upper()
        if u in "ABCDE":
            return u

    for ch in t:
        if ch in DIGIT_TO_LABEL:
            return DIGIT_TO_LABEL[ch]

    for urdu_letter, lab in URDU_LETTER_MAP.items():
        if urdu_letter in t:
            return lab

    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if isinstance(labels, list) and isinstance(texts, list):
            low_full = t.lower()
            for lab, txt in zip(labels, texts):
                if not isinstance(txt, str):
                    continue
                tt = txt.strip()
                if not tt:
                    continue
                if tt in t or tt.lower() in low_full:
                    lab_str = str(lab).upper()
                    if lab_str in "ABCDE":
                        return lab_str

    return None


def extract_deepseek_answer(text):
    """Extract answer from DeepSeek's output that might contain reasoning."""
    if not text:
        return text
    
    # Look for answer pattern after thinking/reasoning
    lines = text.split('\n')
    for line in reversed(lines):  # Check from bottom (answer usually at end)
        line = line.strip()
        if not line:
            continue
        
        # Check for single letter answer
        if len(line) == 1 and line.upper() in "ABCDE":
            return line.upper()
        
        # Check for patterns like "Answer: A" or "جواب: A" or "A"
        words = line.split()
        for word in words:
            word_clean = word.strip('.:;,!?')
            if len(word_clean) == 1 and word_clean.upper() in "ABCDE":
                return word_clean.upper()
    
    return text  # Return original if no clear answer found


# ==========================
# MAIN
# ==========================

def main():
    print("Loading CSQA dataset...")
    examples = load_csqa(CSQA_PATH)
    print(f"Total examples in file: {len(examples)}")

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]
        print(f"Using only first {len(examples)} examples for this run.")

    print("Loading 3-shot template...")
    template = load_prompt_template(PROMPT_PATH)

    print("Building prompts...")
    prompts = build_prompts(template, examples)

    print(f"Loading model with vLLM: {MODEL_NAME}")
    print(f"DeepSeek model detected: {IS_DEEPSEEK}")
    
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

    # Different sampling params for DeepSeek vs other models
    if IS_DEEPSEEK:
        print("Using DeepSeek-specific generation parameters (no stop, higher max_tokens)")
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,  # Much higher to allow reasoning + answer
            # No stop parameter - let it generate fully
        )
    else:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
            stop=["\n"],
        )

    print("Generating answers (3-shot)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0

    out_path = os.path.join(
        OUTPUT_DIR,
        "csqa_three_shot_deepseek_r1_distill_qwen_7b.jsonl",
    )

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            
            # For DeepSeek, try to extract answer from full output
            if IS_DEEPSEEK:
                raw_for_pred = extract_deepseek_answer(raw)
            else:
                raw_for_pred = raw
                
            pred = normalize_csqa_output(raw_for_pred, ex["choices"])
            gold = ex["answerKey"].strip().upper()

            is_correct = None
            if pred is not None:
                answered += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1
            else:
                is_correct = False

            record = {
                "qid": ex["qid"],
                "question": ex["question"],
                "question_concept": ex.get("question_concept", ""),
                "choices": ex["choices"],
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": is_correct,
                "prompt": prompt,
                "raw_output": raw,  # Store full raw output
                "model_name": MODEL_NAME,
                "prompt_type": "three_shot",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    used = len(examples)
    answer_rate = (answered / used * 100) if used > 0 else 0.0
    acc_overall = (correct / used * 100) if used > 0 else 0.0
    acc_answered = (correct / answered * 100) if answered > 0 else 0.0

    print("\n=== CSQA 3-shot with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"DeepSeek mode:     {IS_DEEPSEEK}")
    print(f"Used items:        {used}")
    print(f"Answered (A–E):    {answered} ({answer_rate:.2f}%)")
    print(f"Accuracy overall:  {acc_overall:.2f}%  (correct / {used})")
    print(f"Accuracy answered: {acc_answered:.2f}%  (correct / {answered})")
    print(f"Outputs ->         {out_path}")


if __name__ == "__main__":
    main()