"""PIQA CoT baseline FAIR — thinking ON, 2048 tokens, IDENTICAL to sdfr_piqa_fair.py minus demos."""
import json, os, re
from vllm import LLM, SamplingParams

SPLITS     = "/mnt/home/user41/URBench/data/sdfr_splits"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/cot_piqa_baseline_fair_qwen3_14b.jsonl"

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_answer(text):
    m = re.search(r"حتمی\s*جواب\s*[:：]\s*([01])", text)
    if m: return m.group(1)
    m2 = re.findall(r"\b([01])\b", text)
    return m2[-1] if m2 else ""

if __name__ == "__main__":
    eval_data = read_jsonl(f"{SPLITS}/piqa_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    prompts = []
    for item in eval_data:
        instr = ('مقصد اور دو ممکنہ حل دیے گئے ہیں۔\n'
                 'پہلے مرحلہ وار اور تفصیل سے سوچیں کہ کون سا حل زیادہ منطقی ہے اور عام فہم کے مطابق ہے۔\n'
                 'پھر آخر میں صرف ایک ہندسہ لکھیں: 0 یا 1۔\n'
                 '0 کا مطلب ہے حل 1 درست ہے، 1 کا مطلب ہے حل 2 درست ہے۔')
        q_block = (f"مقصد: {item['goal']}\n"
                   f"حل 1: {item['sol1']}\nحل 2: {item['sol2']}\n\n"
                   f"سوچنے کے مراحل:\nحتمی جواب:")
        raw = f"{instr}\n\n{q_block}"
        messages = [{"role": "user", "content": raw}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=True))

    outputs = llm.generate(prompts, sp)

    correct, results = 0, []
    for i, (item, out) in enumerate(zip(eval_data, outputs)):
        gen = out.outputs[0].text.strip()
        gen_final = gen.split("</think>")[-1]
        pred = extract_answer(gen_final)
        gold = str(item["label"])
        ok = (pred == gold)
        correct += ok
        results.append({"qid": f"PIQA_{i:04d}", "question": item["goal"],
                         "gold": gold, "pred": pred, "correct": ok, "generated": gen})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"CoT-baseline-fair Correct: {correct}/{len(eval_data)} | Accuracy: {correct/len(eval_data)*100:.2f}%")
