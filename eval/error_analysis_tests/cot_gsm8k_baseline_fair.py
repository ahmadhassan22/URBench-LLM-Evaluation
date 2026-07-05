"""GSM8K CoT baseline FAIR — thinking ON, 1024 tokens, IDENTICAL to sdfr_gsm8k_fair.py minus demos."""
import json, os, re
from vllm import LLM, SamplingParams

SPLITS     = "/mnt/home/user41/URBench/data/sdfr_splits"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/cot_gsm8k_baseline_fair_qwen3_14b.jsonl"

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_gold(answer_str):
    m = re.search(r"####\s*(-?[\d,]+)", answer_str)
    if m: return m.group(1).replace(",", "").strip()
    return answer_str.strip()

def extract_answer(text):
    m = re.search(r"حتمی\s*جواب\s*[:：]\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "").strip()
    m2 = re.search(r"####\s*(-?[\d,]+)", text)
    if m2: return m2.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""

if __name__ == "__main__":
    eval_data = read_jsonl(f"{SPLITS}/gsm8k_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    prompts = []
    for item in eval_data:
        instr = ('آپ کو ایک ریاضی کا لفظی مسئلہ دیا گیا ہے۔\n'
                 'پہلے مرحلہ وار سوچیں اور مسئلہ حل کریں۔\n'
                 'اس کے بعد "حتمی جواب:" کے بعد صرف ایک عددی جواب لکھیں۔\n'
                 'حتمی جواب میں کوئی یونٹ، الفاظ یا اضافی متن شامل نہ کریں۔')
        q_block = f"مسئلہ: {item['question']}\n\nسوچنے کے مراحل:\nحتمی جواب:"
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
        gold = extract_gold(item["answer"])
        ok = (pred == gold)
        correct += ok
        results.append({"qid": item.get("qid", f"GSM8K_{i:04d}"), "question": item["question"],
                         "gold": gold, "pred": pred, "correct": ok, "generated": gen})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"CoT-baseline-fair Correct: {correct}/{len(eval_data)} | Accuracy: {correct/len(eval_data)*100:.2f}%")
