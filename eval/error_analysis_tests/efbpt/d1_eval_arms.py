#!/usr/bin/env python3
"""
d1_eval_arms.py

DIAGNOSTIC D1 (as amended 2026-07-31): is the ceiling caused by missing
KNOWLEDGE or by inability to handle the URDU question?

Six passes over the same 200 DEV200 rows. Base Qwen3-14B, NO adapters.

  A  urdu question,    no facts          baseline; MUST reproduce C0 (57.50%)
  B  urdu question,    gold English facts
  C  urdu question,    English facts from ANOTHER row      control
  E  english question, gold English facts
  F  english question, no facts
  G  no question,      gold English facts   leakage probe (Reading 6)

Arm G operationalises Section C Reading 6 ("check answer leakage"): if the
model can answer well from the facts alone, with no question at all, then any
gain in B or E is partly the facts giving the answer away rather than the model
reasoning. Declared here before any arm is scored.

Reading order, fixed in advance:
  1. Arm A must land within ~2pp of 57.50%, or nothing is interpreted.
  2. If C is close to B, facts are not being read; stop.
  3. If G is far above the floor, gains in B/E are partly leakage.
  4. Only then read F-A, B-A, E-B, and E.

Decoding, extractor and scoring are imported from efbpt_eval_dev200.py so they
cannot drift from AMENDMENT 4 / AMENDMENT 5.

Usage (inside a SLURM job, from ~/URBench):
  python eval/error_analysis_tests/efbpt/d1_eval_arms.py --test
  python eval/error_analysis_tests/efbpt/d1_eval_arms.py
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from efbpt_eval_dev200 import (          # noqa: E402
    MODEL_PATH,
    DEV_PATH,
    SYSTEM_MESSAGE,
    MAX_NEW_TOKENS,
    extract_answer,
    load_instruction,
    die,
)

OUT_DIR = "outputs/efbpt/d1/arms"

# Urdu header for the facts block, built from codepoints so a lookalike
# character cannot slip in. حقائق  = "facts".
FACTS_HEADER = "".join(chr(c) for c in [0x062D, 0x0642, 0x0627, 0x0626, 0x0642])

CORRUPT_SHIFT = 97          # same deterministic scheme as the counterfactual run
C0_REFERENCE = 57.50        # arm A must reproduce this
A_TOLERANCE = 2.0

ARMS = ["A", "B", "C", "E", "F", "G"]

ARM_SPEC = {
    #        question field   facts
    "A": ("question_ur", "none"),
    "B": ("question_ur", "gold"),
    "C": ("question_ur", "donor"),
    "E": ("question_en", "gold"),
    "F": ("question_en", "none"),
    "G": (None,          "gold"),
}


def load_rows(limit=None):
    if not os.path.exists(DEV_PATH):
        die("DEV200 missing: " + DEV_PATH)
    rows = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            a = r["answer"]
            if not isinstance(a, bool):
                die("qid %s: answer is %r, expected bool" % (r["urbench_qid"], a))
            for k in ("question_ur", "question_en", "urbench_facts"):
                if not r.get(k):
                    die("qid %s: missing or empty %s" % (r["urbench_qid"], k))
            rows.append({
                "qid": r["urbench_qid"],
                "question_ur": r["question_ur"],
                "question_en": r["question_en"],
                "facts": list(r["urbench_facts"]),
                "gold": "yes" if a else "no",
            })
    if limit is None and len(rows) != 200:
        die("DEV200 has %d rows, expected 200" % len(rows))
    return rows[:limit] if limit else rows


def facts_block(facts):
    return FACTS_HEADER + ":\n" + "\n".join("- " + f.strip() for f in facts)


def build_user(instruction, row, donor_facts, arm):
    qfield, fmode = ARM_SPEC[arm]
    parts = [instruction]
    if fmode == "gold":
        parts.append(facts_block(row["facts"]))
    elif fmode == "donor":
        parts.append(facts_block(donor_facts))
    if qfield is not None:
        parts.append(row[qfield])
    return "\n\n".join(parts)


def score(records):
    n = len(records)
    correct = sum(1 for r in records if r["pred"] == r["gold"])
    unparsed = sum(1 for r in records if r["pred"] is None)
    truncated = sum(1 for r in records if r["truncated"])
    pred_yes = sum(1 for r in records if r["pred"] == "yes")
    floor_no = sum(1 for r in records if r["gold"] == "no")
    return OrderedDict([
        ("n", n),
        ("accuracy", round(100.0 * correct / n, 2)),
        ("floor_always_no", round(100.0 * floor_no / n, 2)),
        ("unparsed_rate", round(100.0 * unparsed / n, 2)),
        ("truncation_rate", round(100.0 * truncated / n, 2)),
        ("predicted_yes_rate", round(100.0 * pred_yes / n, 2)),
        ("correct", correct),
    ])


def generate_all(model, tok, prompts, batch_size, eos_ids):
    n = len(prompts)
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    results = [None] * n
    t0 = time.time()
    done = 0
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt",
                  padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            new_ids = out[j][in_len:]
            keep = [t for t in new_ids.tolist() if t != tok.pad_token_id]
            n_new = len(keep)
            text = tok.decode(new_ids, skip_special_tokens=True)
            trunc = (n_new >= MAX_NEW_TOKENS) and (
                len(new_ids) > 0 and new_ids[-1].item() not in eos_ids)
            results[i] = (text, n_new, trunc)
        done += len(idxs)
        el = time.time() - t0
        print("    %d/%d  (%.1fs, %.1fs/row)" % (done, n, el, el / done), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="20 rows, all arms. Never banked.")
    ap.add_argument("--test-rows", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    instruction = load_instruction()
    print("[ok] instruction verified")
    print("[ok] facts header codepoints: %s"
          % " ".join("%04X" % ord(c) for c in FACTS_HEADER))

    rows = load_rows(limit=args.test_rows if args.test else None)
    n = len(rows)
    n_yes = sum(1 for r in rows if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, n - n_yes) / n
    print("[data] %d rows | yes=%d no=%d | floor %.2f%%" % (n, n_yes, n - n_yes, floor))

    # donor facts for arm C
    donors = [rows[(i + CORRUPT_SHIFT) % n]["facts"] for i in range(n)]
    n_changed = sum(1 for i in range(n) if donors[i] != rows[i]["facts"])
    print("[data] arm C: %d/%d rows get a genuinely different fact set"
          % (n_changed, n))
    if n_changed != n:
        die("arm C donor assignment left %d row(s) with their own facts"
            % (n - n_changed))

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0:
        eos_ids.add(im_end)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("[load] base model, 4-bit nf4, no adapters ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []

    for arm in ARMS:
        print("\n" + "=" * 70)
        print("ARM %s   question=%s   facts=%s"
              % (arm, ARM_SPEC[arm][0] or "NONE", ARM_SPEC[arm][1]))
        print("=" * 70, flush=True)

        prompts = []
        for i, r in enumerate(rows):
            user = build_user(instruction, r, donors[i], arm)
            msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user}]
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))

        print("  prompt example 0 (repr, first 700 chars):")
        print("  " + repr(prompts[0][:700]), flush=True)

        gens = generate_all(model, tok, prompts, args.batch_size, eos_ids)

        records = []
        for r, (text, n_new, trunc) in zip(rows, gens):
            records.append({
                "qid": r["qid"], "arm": arm, "gold": r["gold"],
                "pred": extract_answer(text), "truncated": trunc,
                "n_gen_tokens": n_new, "generation": text,
            })

        s = score(records)
        s["arm"] = arm
        summary.append(s)

        suffix = "_TEST" if args.test else ""
        out_path = os.path.join(OUT_DIR, "d1_arm%s%s.jsonl" % (arm, suffix))
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n_disk = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
        if n_disk != len(records):
            die("wrote %d records but %s has %d lines"
                % (len(records), out_path, n_disk))
        print("  [verified] %d lines on disk" % n_disk)
        print("  accuracy %.2f%%   unparsed %.2f%%   trunc %.2f%%   predYes %.2f%%"
              % (s["accuracy"], s["unparsed_rate"], s["truncation_rate"],
                 s["predicted_yes_rate"]))
        print("  raw -> %s" % out_path)
        print("  sample generation (row 0, repr, first 300 chars):")
        print("  " + repr(records[0]["generation"][:300]), flush=True)

    # ---------------- summary ----------------
    acc = {s["arm"]: s["accuracy"] for s in summary}
    unp = [s["unparsed_rate"] for s in summary]

    print("\n" + "=" * 78)
    print("SUMMARY" + ("  (TEST — NOT A RESULT)" if args.test else ""))
    print("=" * 78)
    print("%-4s %-16s %-8s %8s %9s %8s %9s"
          % ("arm", "question", "facts", "acc%", "unparsed%", "trunc%", "predYes%"))
    for s in summary:
        q, fm = ARM_SPEC[s["arm"]]
        print("%-4s %-16s %-8s %8.2f %9.2f %8.2f %9.2f"
              % (s["arm"], q or "NONE", fm, s["accuracy"], s["unparsed_rate"],
                 s["truncation_rate"], s["predicted_yes_rate"]))
    print("\nmajority-class floor: %.2f%%" % floor)

    sum_path = os.path.join(OUT_DIR, "d1_summary%s.json" % ("_TEST" if args.test else ""))
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("summary -> %s" % sum_path)

    if args.test:
        print("\nTEST COMPLETE. Nothing here is a result.")
        return

    # ---------------- pre-declared reading order ----------------
    print("\n" + "=" * 78)
    print("PRE-DECLARED CHECKS (DIAGNOSTIC D1, Section E/F and AMENDMENT 1 C)")
    print("=" * 78)

    ok = True

    d = abs(acc["A"] - C0_REFERENCE)
    print("\n1. VALIDITY: arm A = %.2f%% vs C0 reference %.2f%% (diff %.2f pp)"
          % (acc["A"], C0_REFERENCE, d))
    if d > A_TOLERANCE:
        ok = False
        print("   *** FAIL: arm A does not reproduce C0 within %.1f pp." % A_TOLERANCE)
        print("   *** The setup differs from the main evaluation. Do NOT interpret")
        print("   *** any arm until this is explained.")
    else:
        print("   PASS")

    spread = max(unp) - min(unp)
    print("\n2. VALIDITY: unparsed-rate spread across arms = %.2f pp" % spread)
    if spread > 5.0:
        ok = False
        print("   *** FAIL: AMENDMENT 5D voids the accuracy comparison. Diagnose")
        print("   *** the parsing gap before reading any number.")
    else:
        print("   PASS")

    print("\n3. CONTROL: are facts being read at all?")
    print("   B (gold facts) = %.2f%%   C (wrong facts) = %.2f%%   B-C = %+.2f pp"
          % (acc["B"], acc["C"], acc["B"] - acc["C"]))
    if acc["B"] - acc["C"] < 3.0:
        ok = False
        print("   *** C is close to B: the model is not using the facts.")
        print("   *** Per the frozen reading, nothing else is interpretable.")
    else:
        print("   PASS: gold facts beat wrong facts, so facts are being used.")

    print("\n4. LEAKAGE: arm G (facts only, no question) = %.2f%% vs floor %.2f%%"
          % (acc["G"], floor))
    print("   G - floor = %+.2f pp" % (acc["G"] - floor))
    if acc["G"] - floor > 10.0:
        print("   *** WARNING: the facts alone predict the answer well above the")
        print("   *** floor. Gains in B and E are partly leakage, not reasoning.")
    else:
        print("   OK: little evidence of answer leakage from the facts alone.")

    if not ok:
        print("\nOne or more validity checks FAILED. The contrasts below are")
        print("printed for diagnosis only and must NOT be reported as results.")

    print("\n" + "-" * 78)
    print("CONTRASTS")
    print("-" * 78)
    print("  F - A  (cost of the Urdu question, no facts)      = %+.2f pp"
          % (acc["F"] - acc["A"]))
    print("  B - A  (value of knowledge, Urdu question)        = %+.2f pp"
          % (acc["B"] - acc["A"]))
    print("  E - B  (cost of the Urdu question, facts given)   = %+.2f pp"
          % (acc["E"] - acc["B"]))
    print("  E      (full-English ceiling)                     =  %.2f%%" % acc["E"])

    print("\n" + "-" * 78)
    print("WHICH PRE-DECLARED READING FITS")
    print("-" * 78)
    if acc["E"] >= 80.0 and (acc["E"] - acc["B"]) <= 5.0 and acc["B"] >= 80.0:
        print("  Reading 2: KNOWLEDGE is the bottleneck; question language costs")
        print("  little. Next method should target getting knowledge to the model.")
    elif acc["E"] >= 80.0 and (acc["E"] - acc["B"]) >= 15.0:
        print("  Reading 3: the URDU QUESTION is the bottleneck even when knowledge")
        print("  is supplied. The method must attack Urdu comprehension.")
    elif acc["E"] <= 70.0:
        print("  Reading 4: neither knowledge nor English phrasing is sufficient.")
        print("  The model cannot combine given facts on this task. This closes")
        print("  retrieval-style approaches as a family.")
    else:
        print("  No pre-declared reading fits cleanly. Report the numbers as they")
        print("  are and do NOT invent a new reading after the fact.")

    print("\n  F - A is reported regardless: it quantifies how much accuracy")
    print("  URBench's Urdu question translation costs relative to the original")
    print("  English question. This is a standalone Urdu-NLP result.")


if __name__ == "__main__":
    main()
