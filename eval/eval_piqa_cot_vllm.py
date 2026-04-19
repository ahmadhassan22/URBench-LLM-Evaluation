import json
import os
import re
from vllm import LLM, SamplingParams

PIQA_PATH = "../data/piqa_raw/piqa_train_750_ur_fixed.jsonl"
PROMPT_PATH = "../prompts/piqa/cot.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/piqa/llama3.1_70b_awq"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "piqa_cot_llama3.1_70b_awq.jsonl")

def get_processed_qids(output_path):
    if not os.path.exists(output_path):
        return set()
    processed = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    qid = data.get("qid") or data.get("goal", "")[:50]
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
            if "qid" not in obj:
                obj["qid"] = f"piqa_{len(data)}"
            data.append(obj)
    return data

def load_prompt_template(path):
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
    t = " ".join(text.strip().split())
    m = re.search(r"\b([01])\b", t)
    if m:
        return m.group(1)
    if t and t[0] in ("0", "1"):
        return t[0]
    return None

def main():
    print("Loading PIQA dataset...")
    all_examples = load_piqa(PIQA_PATH)
    print(f"Total examples in file: {len(all_examples)}")

    processed_qids = get_processed_qids(OUTPUT_FILE)
    if processed_qids:
        print(f"Found {len(processed_qids)} already processed, skipping...")
        examples = [ex for ex in all_examples if str(ex.get("qid", "")) not in processed_qids]
        print(f"Remaining to process: {len(examples)}")
    else:
        examples = all_examples

    if MAX_EXAMPLES is not None and len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]

    if not examples:
        print("All examples already processed! Exiting.")
        return

    template = load_prompt_template(PROMPT_PATH)
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

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, stop=None)

    print("Generating answers (CoT)...")
    outputs = llm.generate(prompts, sampling)

    total_processed = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_piqa_output(raw)
            gold = str(ex.get("label", ex.get("answer", ""))).strip()

            is_correct = None
            if pred is not None and gold in ("0", "1"):
                is_correct = (pred == gold)

            total_processed += 1
            record = {
                "qid": ex.get("qid", ""), "goal": ex["goal"],
                "sol1": ex["sol1"], "sol2": ex["sol2"],
                "gold_answer": gold, "pred_answer": pred, "correct": is_correct,
                "prompt": prompt, "raw_output": raw, "model_name": MODEL_NAME,
                "prompt_type": "cot", "thinking_mode": "N/A",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    # Final scoring from full output file
    total_items = len(all_examples)
    final_correct = 0
    final_answered = 0
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("correct") is True:
                    final_correct += 1
                if data.get("pred_answer") is not None:
                    final_answered += 1
            except:
                continue

    print("\n=== PIQA CoT with vLLM ===")
    print(f"Model:              {MODEL_NAME}")
    print(f"Total items:        {total_items}")
    print(f"Processed this run: {total_processed}")
    print(f"Total processed:    {len(processed_qids) + total_processed}")
    print(f"Answered (0/1):     {final_answered} ({final_answered/total_items*100:.2f}%)")
    print(f"Accuracy overall:   {final_correct/total_items*100:.2f}%")
    print(f"Accuracy answered:  {final_correct/final_answered*100:.2f}%" if final_answered else "Accuracy answered: N/A")
    print(f"Outputs ->          {OUTPUT_FILE}")

if __name__ == "__main__":
    main()