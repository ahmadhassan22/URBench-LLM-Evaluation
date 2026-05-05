import json
import os
import re
from decimal import Decimal, InvalidOperation
from vllm import LLM, SamplingParams

GSM8K_PATH = "../data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl"
PROMPT_PATH = "../prompts/gsm8k/zero_shot_p2.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/gsm8k/qwen3_14b_p2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "gsm8k_zero_shot_p2_qwen3_14b.jsonl"

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096

def load_gsm8k(path):
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

def build_prompts(template, examples):
    return [template.format(question=ex["question"]) for ex in examples]

def format_qwen3_prompts(tokenizer, raw_prompts):
    formatted = []
    for raw_prompt in raw_prompts:
        messages = [{"role": "user", "content": raw_prompt}]
        f = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False)
        formatted.append(f)
    return formatted

def _normalize_number_string(num_str):
    try:
        d = Decimal(num_str)
    except InvalidOperation:
        return None
    if d == d.to_integral():
        return str(d.to_integral())
    return format(d.normalize(), "f")

def extract_number_from_hashes(text):
    if not text:
        return None
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return _normalize_number_string(m.group(1))
    return None

def extract_last_number(text):
    if not text:
        return None
    s = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not matches:
        return None
    return _normalize_number_string(matches[-1])

def normalize_gsm8k_output(text):
    n = extract_number_from_hashes(text)
    if n is not None:
        return n
    return extract_last_number(text)

def extract_gold_number(answer_text):
    if not answer_text:
        return None
    s = answer_text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not matches:
        return None
    return _normalize_number_string(matches[-1])

def main():
    print("Loading GSM8K dataset...")
    examples = load_gsm8k(GSM8K_PATH)
    print(f"Total examples: {len(examples)}")

    template = load_prompt_template(PROMPT_PATH)
    prompts = build_prompts(template, examples)

    print(f"Loading model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME, tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN, trust_remote_code=True,
        gpu_memory_utilization=0.90, max_num_seqs=4,
        enforce_eager=True, disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()
    prompts = format_qwen3_prompts(tokenizer, prompts)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256, stop=None)

    print("Generating answers (zero-shot)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    total = len(examples)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, (ex, prompt, out) in enumerate(zip(examples, prompts, outputs)):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred_num = normalize_gsm8k_output(raw)
            gold_num = extract_gold_number(ex.get("answer", ""))
            is_correct = None
            if pred_num is not None and gold_num is not None:
                answered += 1
                is_correct = (pred_num == gold_num)
                if is_correct:
                    correct += 1
            record = {
                "idx": idx, "question": ex["question"],
                "gold_answer_text": ex.get("answer", ""),
                "gold_number": gold_num, "pred_raw": raw,
                "pred_number": pred_num, "correct": is_correct,
                "prompt": prompt, "model_name": MODEL_NAME,
                "prompt_type": "zero_shot_p2", "thinking_mode": "disabled",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n=== GSM8K Zero-shot P2 — Qwen3-14B ===")
    print(f"Model:              {MODEL_NAME}")
    print(f"Used items:         {total}")
    print(f"Answered (numeric): {answered} ({answered/total*100:.2f}%)")
    print(f"Accuracy overall:   {correct/total*100:.2f}%")
    print(f"Accuracy answered:  {correct/answered*100:.2f}%" if answered else "Accuracy answered: N/A")
    print(f"Outputs ->          {out_path}")

if __name__ == "__main__":
    main()