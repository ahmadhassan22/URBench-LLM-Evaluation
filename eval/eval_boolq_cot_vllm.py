import json
import os
from vllm import LLM, SamplingParams

# ==========================
# CONFIG – EDIT PATHS IF NEEDED
# ==========================

BOOLQ_PATH = "../data/boolq_raw/boolq_train_1550_ur_fixed.jsonl"
PROMPT_PATH = "../prompts/boolq/cot.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/boolq/llama3.1_70b_awq"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "boolq_cot_llama3.1_70b_awq.jsonl"

MAX_EXAMPLES = None

MAX_MODEL_LEN = 4096
MAX_PASSAGE_CHARS = 3000
MAX_QUESTION_CHARS = 512


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
            data.append(json.loads(line))
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


def format_llama_prompts(tokenizer, raw_prompts: list) -> list:
    formatted = []
    for raw_prompt in raw_prompts:
        messages = [
            {"role": "user", "content": raw_prompt}
        ]
        f = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted.append(f)
    return formatted


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
    total = 0

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_boolq_output(raw)
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
                "thinking_mode": "N/A",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = (correct / total * 100) if total > 0 else 0.0

    print("\n=== BoolQ CoT with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"Used items:        {len(examples)}")
    print(f"Scored on:         {total} (pred != None)")
    print(f"Accuracy:          {acc:.2f}%")
    print(f"Outputs ->         {out_path}")


if __name__ == "__main__":
    main()