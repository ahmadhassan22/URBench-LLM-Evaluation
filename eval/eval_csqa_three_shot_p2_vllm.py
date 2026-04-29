import json
import os
from collections import defaultdict
from vllm import LLM, SamplingParams

CSQA_PATH = "../data/csqa_raw/csqa_train_1500_ur.jsonl"
PROMPT_PATH = "../prompts/csqa/three_shot_p2.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/csqa/qwen3_14b_p2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "csqa_three_shot_p2_qwen3_14b.jsonl"

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096

def load_csqa(path):
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
        mapping = extract_choice_mapping(ex["choices"])
        fmt = defaultdict(str)
        fmt["question"] = question
        for lab, txt in mapping.items():
            fmt[lab] = txt
        prompt = template.format_map(fmt)
        prompts.append(prompt)
    return prompts

def format_qwen3_prompts(tokenizer, raw_prompts):
    formatted = []
    for raw_prompt in raw_prompts:
        messages = [{"role": "user", "content": raw_prompt}]
        f = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        formatted.append(f)
    return formatted

URDU_LETTER_MAP = {"الف": "A", "ب": "B", "ج": "C", "د": "D", "ہ": "E", "ھ": "E"}
DIGIT_TO_LABEL = {
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
    "۱": "A", "۲": "B", "۳": "C", "۴": "D", "۵": "E",
    "١": "A", "٢": "B", "٣": "C", "٤": "D", "٥": "E",
}

def normalize_csqa_output(text, choices):
    if not text:
        return None
    t = " ".join(text.strip().split())
    for ch in t:
        if ch.upper() in "ABCDE":
            return ch.upper()
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
                if tt and (tt in t or tt.lower() in low_full):
                    lab_str = str(lab).upper()
                    if lab_str in "ABCDE":
                        return lab_str
    return None

def main():
    print("Loading CSQA dataset...")
    examples = load_csqa(CSQA_PATH)
    print(f"Total examples: {len(examples)}")

    template = load_prompt_template(PROMPT_PATH)
    prompts = build_prompts(template, examples)

    print(f"Loading model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME, tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN, trust_remote_code=True,
        gpu_memory_utilization=0.90, max_num_seqs=1,
        enforce_eager=True, disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()
    prompts = format_qwen3_prompts(tokenizer, prompts)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16, stop=["\n"])

    print("Generating answers (3-shot)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_csqa_output(raw, ex["choices"])
            gold = ex["answerKey"].strip().upper()
            if pred is not None:
                answered += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1
            else:
                is_correct = False
            record = {
                "qid": ex["qid"], "question": ex["question"],
                "question_concept": ex.get("question_concept", ""),
                "choices": ex["choices"], "gold_answer": gold,
                "pred_answer": pred, "correct": is_correct,
                "prompt": prompt, "raw_output": raw,
                "model_name": MODEL_NAME, "prompt_type": "three_shot_p2",
                "thinking_mode": "disabled", "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    used = len(examples)
    print("\n=== CSQA 3-shot P2 — Qwen3-14B ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"Used items:        {used}")
    print(f"Answered (A–E):    {answered} ({answered/used*100:.2f}%)")
    print(f"Accuracy overall:  {correct/used*100:.2f}%  (correct / {used})")
    print(f"Accuracy answered: {correct/answered*100:.2f}%  (correct / {answered})" if answered else "Accuracy answered: N/A")
    print(f"Outputs ->         {out_path}")

if __name__ == "__main__":
    main()