"""PROBE (TEST50): does GenRead (generate-then-read) + a confidence gate beat the
no-facts baseline on StrategyQA, WITHOUT any corpus/retrieval?
Qwen3-14B, thinking ON, max_tokens=2048.

Conditions:
  1. nofacts   (temp 0)                       -> baseline floor
  2. gold      (gold facts in prompt)          -> ceiling
  3. genread   (model generates facts, answers)-> parametric-knowledge fact source
  4. gated     : ask at temp0 AND temp0.7; if answers AGREE -> keep nofacts answer
                 (confident); if DISAGREE -> use genread answer (uncertain)
"""
import json, os
from vllm import LLM, SamplingParams

SPLITS     = "/mnt/home/user41/URBench/data/sdfr_splits"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
OUT        = "/mnt/home/user41/URBench/outputs/sdfr/probe_genread_gate_test50.jsonl"

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm_gold(a):
    if isinstance(a, bool): return "ہاں" if a else "نہیں"
    s = str(a).strip().lower()
    if s in ("true","yes","ہاں"):  return "ہاں"
    if s in ("false","no","نہیں"): return "نہیں"
    return str(a).strip()

def extract_answer(text):
    if "حتمی جواب" in text: text = text.split("حتمی جواب")[-1]
    ih, ina = text.rfind("ہاں"), text.rfind("نہیں")
    if ih == -1 and ina == -1:
        tl = text.lower()
        if "yes" in tl and "no" not in tl: return "ہاں"
        if "no" in tl and "yes" not in tl: return "نہیں"
        return ""
    return "ہاں" if ih > ina else "نہیں"

def p_answer(tok, question, facts_list):
    if facts_list:
        block = "حقائق:\n" + "\n".join(f"- {f}" for f in facts_list) + "\n"
        instr = ("آپ کو ایک سوال اور اس سے متعلق حقائق دیے گئے ہیں۔\n"
                 "پہلے مرحلہ وار اور منطقی طور پر حقائق کی بنیاد پر سوچیں۔\n"
                 "اس کے بعد آخر میں صرف ایک لفظ میں جواب دیں: ہاں یا نہیں۔")
        raw = f"{instr}\n{block}سوال: {question}\nسوچنے کے مراحل:\nحتمی جواب:"
    else:
        instr = ("آپ کو ایک سوال دیا گیا ہے۔\n"
                 "پہلے مرحلہ وار اور منطقی طور پر سوچیں۔\n"
                 "اس کے بعد آخر میں صرف ایک لفظ میں جواب دیں: ہاں یا نہیں۔")
        raw = f"{instr}\nسوال: {question}\nسوچنے کے مراحل:\nحتمی جواب:"
    return tok.apply_chat_template([{"role":"user","content":raw}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)

def p_genfacts(tok, question):
    # step 1 prompt: ask model to WRITE the facts needed (in Urdu), no answer yet
    instr = ("درج ذیل سوال کا جواب دینے کے لیے جو ضروری حقائق درکار ہیں وہ لکھیں۔\n"
             "صرف حقائق کی فہرست دیں، حتمی جواب نہ دیں۔\n"
             "ہر حقیقت کو ایک الگ سطر میں '- ' سے شروع کریں۔")
    raw = f"{instr}\nسوال: {question}\nحقائق:"
    return tok.apply_chat_template([{"role":"user","content":raw}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)

def parse_genfacts(text):
    tail = text.split("</think>")[-1]
    facts = [ln.strip("-• ").strip() for ln in tail.splitlines()
             if ln.strip().startswith(("-","•")) and len(ln.strip()) > 3]
    return facts[:6]

if __name__ == "__main__":
    data = read_jsonl(f"{SPLITS}/strategyqa_eval.jsonl")[:50]
    llm  = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
    tok  = llm.get_tokenizer()
    sp0  = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])
    sp7  = SamplingParams(temperature=0.7, max_tokens=2048, stop=["<|im_end|>"])

    golds = [norm_gold(d.get("answer")) for d in data]

    # cond 1: nofacts temp0
    o_nf0 = llm.generate([p_answer(tok, d["question"], []) for d in data], sp0)
    pred_nf0 = [extract_answer(o.outputs[0].text.split("</think>")[-1]) for o in o_nf0]

    # cond 1b: nofacts temp0.7 (for the gate's agreement check)
    o_nf7 = llm.generate([p_answer(tok, d["question"], []) for d in data], sp7)
    pred_nf7 = [extract_answer(o.outputs[0].text.split("</think>")[-1]) for o in o_nf7]

    # cond 2: gold
    o_gold = llm.generate([p_answer(tok, d["question"], d.get("facts", [])) for d in data], sp0)
    pred_gold = [extract_answer(o.outputs[0].text.split("</think>")[-1]) for o in o_gold]

    # cond 3: genread — step1 generate facts, step2 answer from them
    o_gf = llm.generate([p_genfacts(tok, d["question"]) for d in data], sp0)
    gen_facts = [parse_genfacts(o.outputs[0].text) for o in o_gf]
    o_gr = llm.generate([p_answer(tok, d["question"], gf) for d, gf in zip(data, gen_facts)], sp0)
    pred_gr = [extract_answer(o.outputs[0].text.split("</think>")[-1]) for o in o_gr]

    # cond 4: gated — agree(temp0,temp0.7)? keep nofacts : use genread
    pred_gate, gate_fired = [], 0
    for a0, a7, agr in zip(pred_nf0, pred_nf7, pred_gr):
        if a0 == a7 and a0 != "":
            pred_gate.append(a0)          # confident -> parametric
        else:
            pred_gate.append(agr); gate_fired += 1   # uncertain -> genread

    def acc(preds): return sum(p==g and p!="" for p,g in zip(preds,golds))
    n = len(data)
    rows = []
    for i,d in enumerate(data):
        rows.append({"q":d["question"],"gold":golds[i],"nf0":pred_nf0[i],"nf7":pred_nf7[i],
                     "gold_pred":pred_gold[i],"genread":pred_gr[i],"gated":pred_gate[i],
                     "gen_facts":gen_facts[i]})
    with open(OUT,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    print("\n=== PROBE (TEST50, noisy — read gaps) ===")
    print(f"  1. nofacts  (floor):   {acc(pred_nf0)}/{n} = {acc(pred_nf0)/n*100:.1f}%")
    print(f"  2. gold     (ceiling): {acc(pred_gold)}/{n} = {acc(pred_gold)/n*100:.1f}%")
    print(f"  3. genread:            {acc(pred_gr)}/{n} = {acc(pred_gr)/n*100:.1f}%")
    print(f"  4. gated:              {acc(pred_gate)}/{n} = {acc(pred_gate)/n*100:.1f}% (gate fired {gate_fired}/{n})")
    print("Read: does 3 or 4 beat 1? does gate fire on a sensible fraction (not 0, not all)?")
