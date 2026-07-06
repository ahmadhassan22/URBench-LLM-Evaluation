"""StrategyQA CoT baseline — NO FACTS, thinking ON, Qwen3-14B fair regime.
Same regime as sdfr_strategyqa_fair.py (to be built). Only diff there = retrieved demos.
Facts removed from BOTH the block AND the instruction lines that referenced them."""
import json, os, re
from vllm import LLM, SamplingParams

SPLITS      = "/mnt/home/user41/URBench/data/sdfr_splits"
MODEL_PATH  = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/cot_strategyqa_nofacts_baseline_fair_qwen3_14b.jsonl"

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm_gold(a):
    if isinstance(a, bool): return "ہاں" if a else "نہیں"
    s = str(a).strip().lower()
    if s in ("true", "yes", "ہاں"):  return "ہاں"
    if s in ("false", "no", "نہیں"): return "نہیں"
    return str(a).strip()

def extract_answer(text):
    # only look after final حتمی جواب marker if present
    if "حتمی جواب" in text:
        text = text.split("حتمی جواب")[-1]
    i_haan = text.rfind("ہاں")
    i_nahi = text.rfind("نہیں")
    if i_haan == -1 and i_nahi == -1:
        tl = text.lower()
        if "yes" in tl and "no" not in tl: return "ہاں"
        if "no" in tl and "yes" not in tl: return "نہیں"
        return ""
    return "ہاں" if i_haan > i_nahi else "نہیں"

if __name__ == "__main__":
    eval_data = read_jsonl(f"{SPLITS}/strategyqa_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    prompts = []
    for item in eval_data:
        instr = ("آپ کو ایک سوال دیا گیا ہے۔\n"
                 "پہلے مرحلہ وار اور منطقی طور پر سوچیں۔\n"
                 "اس کے بعد آخر میں صرف ایک لفظ میں جواب دیں: ہاں یا نہیں۔\n"
                 "کوئی وضاحت، جملہ، یا اضافی متن شامل نہ کریں۔")
        q_block = f"سوال: {item['question']}\nسوچنے کے مراحل:\nحتمی جواب:"
        raw = f"{instr}\n{q_block}"
        messages = [{"role": "user", "content": raw}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=True))

    outputs = llm.generate(prompts, sp)

    correct, answered, trunc, results = 0, 0, 0, []
    for i, (item, out) in enumerate(zip(eval_data, outputs)):
        gen = out.outputs[0].text.strip()
        is_trunc = ("<think>" in gen and "</think>" not in gen)
        if is_trunc: trunc += 1
        gen_final = gen.split("</think>")[-1]
        pred = extract_answer(gen_final)
        gold = norm_gold(item.get("answer"))
        if pred: answered += 1
        ok = (pred == gold) and pred != ""
        correct += ok
        results.append({"qid": item.get("qid", f"SQA_{i:04d}"), "question": item["question"],
                        "gold": gold, "pred": pred, "correct": ok,
                        "truncated": is_trunc, "generated": gen})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(eval_data)
    print(f"Correct: {correct}/{n} | Overall: {correct/n*100:.2f}% | "
          f"Answered: {answered}/{n} ({answered/n*100:.1f}%) | "
          f"AnsAcc: {correct/answered*100:.2f}% | Truncated: {trunc}")
