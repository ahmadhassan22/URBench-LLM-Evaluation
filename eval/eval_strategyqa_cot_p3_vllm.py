import json
import os
import re
from vllm import LLM, SamplingParams

STRATEGYQA_PATH = "../data/strategyqa_raw/strategyQA_train_ur2_norm.jsonl"
PROMPT_PATH = "../prompts/strategyqa/cot_p3.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_DIR  = "/mnt/home/user41/URBench/outputs/strategyqa/qwen3_14b_p3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "strategyqa_cot_p3_qwen3_14b.jsonl"

MAX_EXAMPLES = None
MAX_MODEL_LEN = 4096

def load_strategyqa(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_facts(facts):
    return "\n".join(f"- {fact}" for fact in facts)

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
            enable_thinking=True)
        formatted.append(f)
    return formatted

def normalize_strategyqa_output(text):
    """
    Smart extraction — works whether think block is complete or cut off.
    Priority:
    1. Look after </think> tag for clean answer
    2. Look at the very end of output for ہاں/نہیں
    3. Search entire text for ہاں/نہیں
    4. Fall back to yes/no in English
    """
    if not text:
        return None

    # Step 1: extract after </think> if it exists
    after_think = ""
    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip()

    # Step 2: search after </think> first
    if after_think:
        result = _find_answer(after_think)
        if result:
            return result

    # Step 3: search last 200 chars of full text
    tail = text[-200:].strip()
    result = _find_answer(tail)
    if result:
        return result

    # Step 4: search entire text
    result = _find_answer(text)
    return result

def _find_answer(text):
    if not text:
        return None
    t = text.strip()

    # Check for حتمی جواب marker
    for marker in ["حتمی جواب", "جواب"]:
        if marker in t:
            idx = t.rfind(marker)
            after = t[idx + len(marker):].strip().lstrip(":").strip()
            if "ہاں" in after:
                return "ہاں"
            if "نہیں" in after or "نہيں" in after:
                return "نہیں"

    # Direct Urdu check
    cleaned = re.sub(r"[^\u0600-\u06FF\s]", " ", t)
    cleaned = " ".join(cleaned.split())
    if cleaned.endswith("ہاں"):
        return "ہاں"
    if cleaned.endswith("نہیں"):
        return "نہیں"
    if "ہاں" in cleaned and "نہیں" not in cleaned:
        return "ہاں"
    if "نہیں" in cleaned and "ہاں" not in cleaned:
        return "نہیں"

    # English fallback
    tl = t.lower()
    if "yes" in tl and "no" not in tl:
        return "ہاں"
    if "no" in tl and "yes" not in tl:
        return "نہیں"

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
        gpu_memory_utilization=0.90, max_num_seqs=2,
        enforce_eager=True, disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()
    prompts = format_qwen3_prompts(tokenizer, prompts)

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0,
        max_tokens=512, stop=None
    )

    print("Generating answers (CoT)...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    answered = 0
    total = len(examples)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in zip(examples, prompts, outputs):
            raw = out.outputs[0].text.strip() if out.outputs else ""
            pred = normalize_strategyqa_output(raw)
            gold = ex.get("answer", "").strip()
            is_correct = None
            if pred is not None:
                answered += 1
                is_correct = (pred == gold)
                if is_correct:
                    correct += 1
            fout.write(json.dumps({
                "qid": ex.get("qid", ""), "question": ex["question"],
                "facts": ex.get("facts", []), "gold_answer": gold,
                "pred_answer": pred, "correct": is_correct,
                "prompt": prompt, "raw_output": raw,
                "prompt_type": "cot_p2", "model_name": MODEL_NAME,
                "thinking_mode": "enabled", "context_max_len": MAX_MODEL_LEN,
            }, ensure_ascii=False) + "\n")

    print("\n=== StrategyQA CoT P3 — Qwen3-14B ===")
    print(f"Model:               {MODEL_NAME}")
    print(f"Used items:          {total}")
    print(f"Answered (ہاں/نہیں): {answered} ({answered/total*100:.2f}%)")
    print(f"Accuracy overall:    {correct/total*100:.2f}%")
    print(f"Accuracy answered:   {correct/answered*100:.2f}%" if answered else "Accuracy answered: N/A")
    print(f"Outputs ->           {out_path}")

if __name__ == "__main__":
    main()