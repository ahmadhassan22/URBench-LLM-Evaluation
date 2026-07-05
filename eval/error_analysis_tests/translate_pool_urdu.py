"""Translate StrategyQA pool questions (English) to Urdu using Qwen3-14B, offline, one-time."""
import json, os
from vllm import LLM, SamplingParams

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
SPLITS     = "/mnt/home/user41/URBench/data/sdfr_splits"
POOL_IN    = f"{SPLITS}/strategyqa_pool.jsonl"
POOL_OUT   = f"{SPLITS}/strategyqa_pool_urdu.jsonl"

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

pool = read_jsonl(POOL_IN)
print(f"Translating {len(pool)} pool questions...")

llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=100, stop=["<|im_end|>"])

prompts = []
for item in pool:
    msg = f"Translate this English question into natural Urdu. Output ONLY the Urdu translation, nothing else.\n\nEnglish: {item['question']}\nUrdu:"
    messages = [{"role": "user", "content": msg}]
    prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

outputs = llm.generate(prompts, sp)

with open(POOL_OUT, "w", encoding="utf-8") as f:
    for item, out in zip(pool, outputs):
        translated = out.outputs[0].text.strip()
        row = dict(item)
        row["question_en"] = item["question"]
        row["question"] = translated
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Done: {POOL_OUT}")