import json
import os
import re
from collections import defaultdict

from vllm import LLM, SamplingParams

# ==========================
# CONFIG
# ==========================

CSQA_PATH = "../data/csqa_raw/csqa_train_1500_ur.jsonl"
PROMPT_PATH = "../prompts/csqa/cot.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/csqa/llama3.1_70b_awq"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "csqa_cot_llama3.1_70b_awq.jsonl"

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096


# ==========================
# HELPERS
# ==========================

def load_csqa(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
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
            lab = labels_default[idx] if idx < len(labels_default) else str(idx)
            if isinstance(ch, dict):
                txt = str(ch.get("text", "")).strip()
            else:
                txt = str(ch).strip()
            mapping[lab] = txt
    return mapping


def build_prompts(template, examples):
    prompts = []
    for ex in examples:
        question = ex["question"]
        mapping = extract_choice_mapping(ex["choices"])

        fmt = defaultdict(str)
        fmt["question"] = question

        for lab, txt in mapping.items():
            fmt[lab] = txt

        prompt = template.format_map(fmt)
        prompts.append(prompt)

    return prompts


def format_llama_prompts(tokenizer, raw_prompts: list) -> list:
    formatted = []
    for raw_prompt in raw_prompts:
        messages = [{"role": "user", "content": raw_prompt}]
        f = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted.append(f)
    return formatted


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


def _map_letter_or_digit_to_label(text):
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

    return None


def normalize_csqa_cot_output(text, choices):
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    marker_match = re.search(r"(Answer\s*:)(.*)", t, flags=re.IGNORECASE | re.DOTALL)
    if not marker_match:
        marker_match = re.search(r"(جواب\s*:)(.*)", t, flags=re.IGNORECASE | re.DOTALL)

    if marker_match:
        after = marker_match.group(2).strip()
        label = _map_letter_or_digit_to_label(after)
        if label:
            return label

    label = _map_letter_or_digit_to_label(t)
    if label:
        return label

    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if isinstance(labels, list) and isinstance(texts, list):
            low_full = t.lower()
            for lab, txt in zip(labels, texts):
                if isinstance(txt, str):
                    tt = txt.strip()
                    if tt and (tt in t or tt.lower() in low_full):
                        if isinstance(lab, str) and lab.upper() in "ABCDE":
                            return lab.upper()

    return None


# ==========================
# MAIN
# ==========================

def main():
    print("Loading CSQA dataset...")
    examples = load_csqa(CSQA_PATH)
    print(f"Total examples in file: {len(examples)}")

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]

    print("Loading CoT template...")
    template = load_prompt_template(PROMPT_PATH)

    print("Building prompts...")
    prompts = build_prompts(template, examples)

    print(f"Loading model with vLLM: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        enforce_eager=True,
        disable_log_stats=True,
        quantization="awq",
    )

    tokenizer = llm.get_tokenizer()
    prompts = format_llama_prompts(tokenizer, prompts)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        stop=None
    )

    print("Generating answers (CoT)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    total_answered = 0

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_csqa_cot_output(raw, ex["choices"])
            gold = ex["answerKey"].strip().upper()

            if pred is not None:
                total_answered += 1
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
                "raw_output": raw,
                "model_name": MODEL_NAME,
                "prompt_type": "cot",
                "thinking_mode": "N/A",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    used_items = len(examples)
    overall_acc = (correct / used_items * 100) if used_items > 0 else 0.0
    answered_acc = (correct / total_answered * 100) if total_answered > 0 else 0.0

    print("\n=== CSQA CoT with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"Used items:        {used_items}")
    print(f"Answered (A–E):    {total_answered} ({(total_answered/used_items*100):.2f}%)")
    print(f"Accuracy overall:  {overall_acc:.2f}%")
    print(f"Accuracy answered: {answered_acc:.2f}%")
    print(f"Outputs ->         {out_path}")


if __name__ == "__main__":
    main()