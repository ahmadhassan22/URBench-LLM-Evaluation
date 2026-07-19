"""
c_probe_run_TEST5.py  — capture Qwen3-14B reasoning traces for 5 entity-rich rows.

WHAT THIS TESTS
---------------
(c) = does Qwen3-14B corrupt named entities INSIDE its own Urdu reasoning trace?
This script feeds 5 Urdu multi-hop questions to the model with thinking ON and
saves the FULL <think> trace + final answer. It does NOT score anything.
You + ChatGPT read the traces afterward and check each testable entity.

WHY TEST5 FIRST
---------------
Confirms trace capture works and traces are readable BEFORE running all 28 rows.
Never bank a 5-row result — this is a go/no-go on the pipeline, not evidence.

MATCHES YOUR FAIR REGIME
------------------------
Same model path, bfloat16, enable_thinking=True, temperature=0.0, split on </think>
— identical to sdfr_gsm8k_fair.py. Only difference: no retrieval, no few-shot,
we ask the bare Urdu question and keep the whole trace.

HOW TO RUN
----------
    cd /mnt/home/user41/URBench
    python eval/error_analysis_tests/c_probe_run_TEST5.py

OUTPUT
------
    data/strategyqa_official/efbpt/c_probe_TEST5_traces.jsonl   (machine)
    data/strategyqa_official/efbpt/c_probe_TEST5_traces.txt     (you read this)
"""

import json
from pathlib import Path
from vllm import LLM, SamplingParams

BASE       = Path("/mnt/home/user41/URBench")
EFBPT      = BASE / "data/strategyqa_official/efbpt"
TESTSET    = EFBPT / "c_probe_testset.jsonl"
OUT_JSONL  = EFBPT / "c_probe_TEST5_traces.jsonl"
OUT_TXT    = EFBPT / "c_probe_TEST5_traces.txt"

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"

# 5 entity-rich rows chosen for hard transliteration cases.
# These qids are from the audit set (Wilson, Roewe, Vlad, Sartre, ZRK Kumanovo).
TEST5_QIDS = [
    "3f59cf2d6b48378dbefe",  # Woodrow Wilson / Taft / Harding
    "f9fcf86196d1847b2f0b",  # Roewe 550 / 2008 Olympics
    "2295eaf3cdbecc17ca0a",  # Vlad the Impaler / Bucharest
    "46d5eda453734f1d4f98",  # Jean-Paul Sartre / Elizabeth I
    "3d80646ef0844fac8da5",  # Spice Girls / ZRK Kumanovo / Handball
]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    rows = read_jsonl(TESTSET)
    by_qid = {r["urbench_qid"]: r for r in rows}

    selected = []
    for qid in TEST5_QIDS:
        if qid in by_qid:
            selected.append(by_qid[qid])
        else:
            print(f"!! qid not found in testset: {qid}")

    if not selected:
        print("!! No rows selected. Check TEST5_QIDS against c_probe_testset.jsonl.")
        raise SystemExit(1)

    print(f"Selected {len(selected)} rows for TEST5.")

    # Bare Urdu question, thinking ON. No retrieval, no few-shot.
    # Simple instruction: answer the yes/no question, reason step by step in Urdu.
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])

    prompts = []
    for r in selected:
        instr = ('درج ذیل سوال کا جواب دیں۔\n'
                 'پہلے مرحلہ وار سوچیں، پھر "حتمی جواب:" کے بعد صرف "ہاں" یا "نہیں" لکھیں۔')
        q_block = f"سوال: {r['question_ur']}\n\nحتمی جواب:"
        raw = f"{instr}\n\n{q_block}"
        messages = [{"role": "user", "content": raw}]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True))

    outputs = llm.generate(prompts, sp)

    txt = open(OUT_TXT, "w", encoding="utf-8")
    txt.write("(c) PROBE — TEST5 reasoning traces (Qwen3-14B, thinking ON)\n")
    txt.write("=" * 95 + "\n")
    txt.write("For each row: read the THINK trace, check if each testable entity is named CORRECTLY.\n")
    txt.write("Corruption = trace refers to a DIFFERENT entity than the gold canonical title.\n")
    txt.write("=" * 95 + "\n\n")

    results = []
    for r, out in zip(selected, outputs):
        gen = out.outputs[0].text.strip()
        # split think vs final, same convention as your fair scripts
        if "</think>" in gen:
            think = gen.split("</think>")[0].replace("<think>", "").strip()
            final = gen.split("</think>")[-1].strip()
        else:
            think = "[[no </think> marker found]]"
            final = gen

        testable = [e for e in r["entities"] if e["testable"]]

        results.append({
            "urbench_qid": r["urbench_qid"],
            "question_ur": r["question_ur"],
            "question_en": r["question_en"],
            "gold_answer": r.get("verdict"),  # note: verdict field, not gold yes/no
            "testable_entities": testable,
            "think_trace": think,
            "final_answer": final,
            "raw_full": gen,
        })

        txt.write("#" * 95 + "\n")
        txt.write(f"qid: {r['urbench_qid']}\n")
        txt.write(f"UR: {r['question_ur']}\n")
        txt.write(f"EN: {r['question_en']}\n")
        txt.write("TESTABLE ENTITIES (check each of these in the trace):\n")
        for e in testable:
            txt.write(f"    - {e['canonical_title']}   (urdu span: {e['urdu_span']})\n")
        txt.write("\n--- THINK TRACE ---\n")
        txt.write(think + "\n")
        txt.write("\n--- FINAL ANSWER ---\n")
        txt.write(final + "\n\n")

    txt.close()

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDONE.")
    print("READ THIS :", OUT_TXT)
    print("MACHINE   :", OUT_JSONL)
    print("\nRead the 5 traces. For each testable entity, decide: named correctly, or corrupted?")
    print("Send the .txt back for calibration before running all 28.")