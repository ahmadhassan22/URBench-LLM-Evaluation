import json
import os
import re
from vllm import LLM, SamplingParams

STRATEGYQA_PATH = "../data/strategyqa_raw/strategyQA_train_ur2_norm.jsonl"
PROMPT_PATH = "../prompts/strategyqa/zero_shot_p2.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/strategyqa/qwen3_14b_p2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "strategyqa_zero_shot_p2_qwen3_14b.jsonl"

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096

def load_strategyqa(path):
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

def format_facts(facts_list):
    if not facts_list:
        return ""
    return "\n".join(f"- {fact}" for fact in facts_list)

def build_prompts(template, examples):
    prompts = []
    for ex in examples:
        prompt = template.format(
            facts=format_facts(ex.get("facts", [])),
            question=ex["question"],
        )
        prompts.append(prompt)
    return prompts

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

def normalize_strategyqa_output(text):
    if not text:
        return None
    t = text.strip().replace('"', "").replace("'", "").strip()
    if not t:
        return None
    cleaned = re.sub(r"[^\u0600-\u06FF\s]", " ", t)
    cleaned = " ".join(cleaned.split())
    if cleaned == "ہاں": return "ہاں"
    if cleaned == "نہیں": return "نہیں"
    if cleaned.startswith("ہا"): return "ہاں"
    if cleaned.startswith("نہ"): return "نہیں"
    has_haan = ("ہاں" in cleaned) or ("ہا" in cleaned)
    has_nahi = ("نہیں" in cleaned) or cleaned.startswith("نہ")
    if has_nahi and not has_haan: return "نہیں"
    if has_haan and not has_nahi: return "ہاں"
    tl = t.lower()
    if "yes" in tl and "no" not in tl: return "ہاں"
    if "no" in tl and "yes" not in tl: return "نہیں"
    return None

def main():
    print("Loading StrategyQA dataset...")
    examples = load_strategyqa(STRATEGYQA_PATH)
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

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16, stop=["\n"])

    print("Generating answers (zero-shot)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    total = len(examples)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_strategyqa_output(raw)
            gold = str(ex.get("answer", "")).strip()
            is_correct = None
            if pred is not None and gold in ("ہاں", "نہیں"):
                answered += 1
                is_correct = (pred == gold)
                if is_correct: correct += 1
            record = {
                "qid": ex.get("qid", ""), "term": ex.get("term", ""),
                "description": ex.get("description", ""), "question": ex["question"],
                "facts": ex.get("facts", []), "gold_answer": gold,
                "pred_answer": pred, "correct": is_correct,
                "prompt": prompt, "raw_output": raw, "model_name": MODEL_NAME,
                "prompt_type": "zero_shot_p2", "thinking_mode": "disabled",
                "context_max_len": MAX_MODEL_LEN,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n=== StrategyQA Zero-shot P2 — Qwen3-14B ===")
    print(f"Model:               {MODEL_NAME}")
    print(f"Used items:          {total}")
    print(f"Answered (ہاں/نہیں): {answered} ({answered/total*100:.2f}%)")
    print(f"Accuracy overall:    {correct/total*100:.2f}%")
    print(f"Accuracy answered:   {correct/answered*100:.2f}%" if answered else "Accuracy answered: N/A")
    print(f"Outputs ->           {out_path}")

if __name__ == "__main__":
    main()