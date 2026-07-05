"""CSQA CoT baseline — thinking ON, IDENTICAL regime to sdfr_csqa_thinking.py, demos removed."""
import json, os, re
from vllm import LLM, SamplingParams

SPLITS     = "/mnt/home/user41/URBench/data/sdfr_splits"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUTPUT_FILE = "/mnt/home/user41/URBench/outputs/sdfr/cot_csqa_baseline_fair_qwen3_14b.jsonl"

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_answer(text):
    m = re.search(r"[Aa]nswer\s*[:：]\s*([A-E])", text)
    if m: return m.group(1).upper()
    m2 = re.findall(r"\b([A-E])\b", text.upper())
    return m2[-1] if m2 else ""

if __name__ == "__main__":
    eval_data = read_jsonl(f"{SPLITS}/csqa_eval.jsonl")
    if os.environ.get("TEST50"): eval_data = eval_data[:50]

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|im_end|>"])

    prompts = []
    for item in eval_data:
        ch = item["choices"]
        A,B,C,D,E = [ch["text"][ch["label"].index(x)] if x in ch["label"] else "" for x in ["A","B","C","D","E"]]
        instr = ("آپ کو ایک سوال اور پانچ اختیارات (A، B، C، D، E) دیے گئے ہیں۔\n"
                 "پہلے مختصر طور پر اردو میں سوچیں کہ کون سا جواب درست ہے۔\n"
                 "آخر میں صرف یہ شکل استعمال کریں: Answer: X\n"
                 "جہاں X صرف A، B، C، D یا E ہو۔")
        q_block = (f"سوال: {item['question']}\n"
                   f"A: {A}\nB: {B}\nC: {C}\nD: {D}\nE: {E}\n\n"
                   f"Step-by-step reasoning (اردو میں):\n\nAnswer:")
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
        gold = item["answerKey"].strip().upper()
        ok = (pred == gold)
        correct += ok
        results.append({"qid": item.get("qid", f"CSQA_{i:04d}"), "question": item["question"],
                         "gold": gold, "pred": pred, "correct": ok, "generated": gen})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Correct: {correct}/{len(eval_data)} | Accuracy: {correct/len(eval_data)*100:.2f}%")
