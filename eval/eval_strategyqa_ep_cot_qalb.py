import json
import os
import re

from vllm import LLM, SamplingParams

DATA_PATH    = "../data/strategyqa_raw/strategyQA_train_ur2_norm.jsonl"
PROMPT_PATH  = "../prompts/strategyqa/english_pivoted/ep_cot.txt"
MODEL_NAME   = "/mnt/home/user41/downloaded_models/Qalb/Qalb-1.0-8B-Instruct"
OUTPUT_DIR   = "../outputs/strategyqa/qalb_1.0_8b"
OUTPUT_FILE  = "strategyqa_ep_cot_qalb_test50.jsonl"
MAX_EXAMPLES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

examples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line.strip()))

if MAX_EXAMPLES:
    examples = examples[:MAX_EXAMPLES]

print(f"Total examples: {len(examples)}")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    template = f.read()

prompts = []
for ex in examples:
    question = ex.get("question", "")
    prompt = template.replace("{question}", question)
    prompts.append(prompt)

print(f"Loading model: {MODEL_NAME}")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    max_model_len=2048,
    max_num_seqs=1,
    disable_log_stats=True,
    enforce_eager=True,
)

sampling = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
)

print("Generating answers (Qalb English-Pivoted CoT)...")
outputs = llm.generate(prompts, sampling)

def extract_answer(text):
    marker = "حتمی جواب"
    if marker in text:
        segment = text.split(marker)[-1]
    else:
        segment = text
    segment = segment.strip().lstrip(":").strip()
    if re.search(r"ہاں", segment[:20]):
        return "ہاں"
    if re.search(r"نہیں", segment[:20]):
        return "نہیں"
    if re.search(r"ہاں", text):
        return "ہاں"
    if re.search(r"نہیں", text):
        return "نہیں"
    return None

correct  = 0
answered = 0
total    = len(examples)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

with open(out_path, "w", encoding="utf-8") as f_out:
    for ex, out in zip(examples, outputs):
        raw        = out.outputs[0].text
        pred       = extract_answer(raw)
        gold_label = str(ex.get("answer"))
        is_correct = (pred == gold_label)
        if pred is not None:
            answered += 1
            if is_correct:
                correct += 1
        record = {
            "idx":        ex.get("qid", ""),
            "question":   ex.get("question", ""),
            "gold":       gold_label,
            "pred":       pred,
            "pred_raw":   raw,
            "correct":    is_correct,
            "prompt_type": "ep_cot",
            "model_name": MODEL_NAME,
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n=== StrategyQA English-Pivoted CoT — Qalb-1.0-8B-Instruct ===")
print(f"Model:               {MODEL_NAME}")
print(f"Used items:          {total}")
print(f"Answered (ہاں/نہیں): {answered} ({answered/total*100:.2f}%)")
print(f"Accuracy overall:    {correct/total*100:.2f}%")
if answered:
    print(f"Accuracy answered:   {correct/answered*100:.2f}%")
print(f"Outputs ->           {out_path}")
