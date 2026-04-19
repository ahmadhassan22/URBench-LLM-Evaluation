import json
import os
import re
from vllm import LLM, SamplingParams

PIQA_PATH = "../data/piqa_raw/piqa_train_750_ur_fixed.jsonl"
PROMPT_PATH = "../prompts/piqa/zero_shot.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/piqa/llama3.1_70b_awq"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "piqa_zero_shot_llama3.1_70b_awq.jsonl"

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096

def load_piqa(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompts(template, examples):
    prompts = []
    for ex in examples:
        prompt = (
            template
            .replace("{goal}", ex["goal"])
            .replace("{sol1}", ex["sol1"])
            .replace("{sol2}", ex["sol2"])
        )
        prompts.append(prompt)
    return prompts

def format_llama_prompts(tokenizer, raw_prompts):
    formatted = []
    for raw_prompt in raw_prompts:
        messages = [{"role": "user", "content": raw_prompt}]
        f = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(f)
    return formatted

def normalize_piqa_output(text):
    if not text:
        return None
    t = text.strip()
    if t == "0" or t == "1":
        return t
    if t and t[0] in ("0", "1"):
        return t[0]
    m = re.search(r"\b([01])\b", t)
    if m:
        return m.group(1)
    return None

def main():
    print("Loading PIQA dataset...")
    examples = load_piqa(PIQA_PATH)
    print(f"Total examples in file: {len(examples)}")

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]

    template = load_prompt(PROMPT_PATH)
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

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8, stop=["\n"])

    print("Generating answers...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_piqa_output(raw)
            gold = str(ex["label"])

            is_correct = None
            if pred is not None:
                answered += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1

            record = {
                "goal": ex["goal"], "sol1": ex["sol1"], "sol2": ex["sol2"],
                "gold_answer": gold, "pred_answer": pred, "correct": is_correct,
                "raw_output": raw, "model_name": MODEL_NAME,
                "prompt_type": "zero_shot", "thinking_mode": "N/A",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(examples)
    print("\n=== PIQA Zero-shot with vLLM ===")
    print(f"Model:             {MODEL_NAME}")
    print(f"Used items:        {total}")
    print(f"Answered (0/1):    {answered} ({answered/total*100:.2f}%)")
    print(f"Accuracy overall:  {correct/total*100:.2f}%")
    print(f"Accuracy answered: {correct/answered*100:.2f}%" if answered else "Accuracy answered: N/A")
    print(f"Outputs ->         {out_path}")

if __name__ == "__main__":
    main()