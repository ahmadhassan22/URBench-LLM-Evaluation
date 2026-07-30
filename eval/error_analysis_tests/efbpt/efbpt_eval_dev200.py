#!/usr/bin/env python3
"""
efbpt_eval_dev200.py

Evaluates C0 (base) + C1/C2/C3 x seeds 13/42/2026 on DEV200.

Contract: docs/EFBPT_PLAN_A_FREEZE.md
  - AMENDMENT 4: fixed system message + frozen Urdu instruction
  - AMENDMENT 5: frozen answer extractor, scoring, mandatory reporting
  - Section 8.3: thinking OFF, temperature 0, max_tokens 1024

Inference uses transformers with the SAME 4-bit nf4 quantization used in
training, so there is no train/eval numerical mismatch. Slower than vLLM,
chosen deliberately for exactness.

The base model is loaded ONCE. All 9 adapters are attached to it and switched
with set_adapter(); C0 runs inside disable_adapter().

Usage (inside a SLURM job, from ~/URBench):
  python eval/error_analysis_tests/efbpt/efbpt_eval_dev200.py --test
  python eval/error_analysis_tests/efbpt/efbpt_eval_dev200.py
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict

import torch

# ----------------------------------------------------------------------------
# Frozen constants
# ----------------------------------------------------------------------------

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
DEV_PATH = "data/strategyqa_official/dev200_seed4242.jsonl"
INSTRUCTION_PATH = "prompts/efbpt/plan_a_instruction_ur.txt"
INSTRUCTION_MD5 = "f3b58d766fe3ec2573ff4f24761cf0c9"
SYSTEM_MESSAGE = "You are a helpful assistant. Answer the user's question."

ADAPTER_ROOT = "outputs/efbpt/plan_a/adapters"
OUT_DIR = "outputs/efbpt/plan_a/dev200"

SEEDS = [13, 42, 2026]
CONDITIONS = ["C1", "C2", "C3"]

EXPECTED_ROWS = 200
MAX_NEW_TOKENS = 1024      # Section 8.3, frozen
TEMPERATURE = 0.0          # Section 8.3, frozen -> greedy decoding

# Urdu match strings built from codepoints, never typed, so a substituted
# lookalike character (this project already found a Cyrillic substitution)
# cannot silently break the extractor.
HAAN = "".join(chr(c) for c in [0x06C1, 0x0627, 0x06BA])                    # yes
NAHIN = "".join(chr(c) for c in [0x0646, 0x06C1, 0x06CC, 0x06BA])           # no
MARKER = "".join(chr(c) for c in [0x062D, 0x062A, 0x0645, 0x06CC, 0x0020,
                                  0x062C, 0x0648, 0x0627, 0x0628])          # final answer

RE_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"(yes|no)"', re.IGNORECASE)
RE_EN_WORD = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# AMENDMENT 5 Section B — frozen extractor. Ordered; first match wins.
# ----------------------------------------------------------------------------

def extract_answer(text):
    """Returns 'yes', 'no', or None (unparsed). Identical for C0-C3."""

    # Rule 1: JSON answer field, LAST match.
    m = None
    for m in RE_JSON_ANSWER.finditer(text):
        pass
    if m is not None:
        return m.group(1).lower()

    # Rule 2: Urdu. Keep only text after the LAST marker occurrence.
    seg = text
    if MARKER in seg:
        seg = seg[seg.rfind(MARKER) + len(MARKER):]
    i_haan = seg.rfind(HAAN)
    i_nahin = seg.rfind(NAHIN)
    if i_haan != -1 or i_nahin != -1:
        return "yes" if i_haan > i_nahin else "no"

    # Rule 3: English, word-bounded, LAST match.
    m = None
    for m in RE_EN_WORD.finditer(text):
        pass
    if m is not None:
        return m.group(1).lower()

    # Rule 4: unparsed.
    return None


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_instruction():
    if not os.path.exists(INSTRUCTION_PATH):
        die("instruction file missing: " + INSTRUCTION_PATH)
    raw = open(INSTRUCTION_PATH, "rb").read()
    got = hashlib.md5(raw).hexdigest()
    if got != INSTRUCTION_MD5:
        die("instruction MD5 mismatch: expected %s, got %s. The evaluation "
            "prompt would not match the training prompt." % (INSTRUCTION_MD5, got))
    return raw.decode("utf-8")


def load_dev(limit=None):
    if not os.path.exists(DEV_PATH):
        die("DEV200 missing: " + DEV_PATH)
    rows = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is None and len(rows) != EXPECTED_ROWS:
        die("DEV200 has %d rows, expected %d" % (len(rows), EXPECTED_ROWS))

    out = []
    for r in rows:
        a = r["answer"]
        if not isinstance(a, bool):
            die("qid %s: answer is %r, expected a boolean" % (r["urbench_qid"], a))
        out.append({
            "qid": r["urbench_qid"],
            "question_ur": r["question_ur"],
            "gold": "yes" if a else "no",   # AMENDMENT 2 mapping
        })
    if limit is not None:
        out = out[:limit]
    return out


def score(records):
    n = len(records)
    correct = sum(1 for r in records if r["pred"] == r["gold"])
    unparsed = sum(1 for r in records if r["pred"] is None)
    truncated = sum(1 for r in records if r["truncated"])
    pred_yes = sum(1 for r in records if r["pred"] == "yes")
    return OrderedDict([
        ("n", n),
        ("accuracy", round(100.0 * correct / n, 2)),
        ("unparsed_rate", round(100.0 * unparsed / n, 2)),
        ("truncation_rate", round(100.0 * truncated / n, 2)),
        ("predicted_yes_rate", round(100.0 * pred_yes / n, 2)),
        ("correct", correct),
        ("unparsed", unparsed),
        ("truncated", truncated),
    ])


# ----------------------------------------------------------------------------
# Batched greedy generation
# ----------------------------------------------------------------------------

def generate_all(model, tok, prompts, batch_size, eos_ids):
    """Returns list of (text, n_new_tokens, truncated) in the original order."""
    n = len(prompts)
    order = sorted(range(n), key=lambda i: len(prompts[i]))   # group similar lengths
    results = [None] * n
    t0 = time.time()
    done = 0

    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        batch = [prompts[i] for i in idxs]

        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                # temperature 0 == greedy
                pad_token_id=tok.pad_token_id,
            )

        for j, i in enumerate(idxs):
            new_ids = out[j][in_len:]
            # strip trailing pad
            keep = [t for t in new_ids.tolist() if t != tok.pad_token_id]
            n_new = len(keep)
            text = tok.decode(new_ids, skip_special_tokens=True)
            truncated = (n_new >= MAX_NEW_TOKENS) and (
                len(new_ids) > 0 and new_ids[-1].item() not in eos_ids
            )
            results[i] = (text, n_new, truncated)

        done += len(idxs)
        el = time.time() - t0
        print("    %d/%d  (%.1fs elapsed, %.1fs/row)" % (done, n, el, el / done),
              flush=True)

    return results


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="TEST: few rows, C0 and C3_seed13 only. Never banked.")
    ap.add_argument("--test-rows", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    instruction = load_instruction()
    print("[ok] instruction verified, MD5 %s" % INSTRUCTION_MD5)
    print("[ok] extractor strings (hex): HAAN=%s | NAHIN=%s | MARKER=%s"
          % (" ".join("%04X" % ord(c) for c in HAAN),
             " ".join("%04X" % ord(c) for c in NAHIN),
             " ".join("%04X" % ord(c) for c in MARKER)))

    rows = load_dev(limit=args.test_rows if args.test else None)
    n_yes = sum(1 for r in rows if r["gold"] == "yes")
    n_no = len(rows) - n_yes
    floor = 100.0 * max(n_yes, n_no) / len(rows)
    print("[data] %d rows | gold yes=%d no=%d | majority-class floor = %.2f%%"
          % (len(rows), n_yes, n_no, floor))

    # ---- configuration list ----
    configs = [("C0", None, None)]
    if args.test:
        configs.append(("C3", 13, os.path.join(ADAPTER_ROOT, "C3_seed13")))
    else:
        for cond in CONDITIONS:
            for seed in SEEDS:
                path = os.path.join(ADAPTER_ROOT, "%s_seed%d" % (cond, seed))
                if not os.path.isdir(path):
                    die("adapter missing: " + path)
                if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
                    die("adapter_model.safetensors missing in " + path)
                configs.append((cond, seed, path))
    print("[plan] %d configurations" % len(configs))

    # ---- tokenizer + prompts (identical for every configuration) ----
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"      # required for correct decoder-only batching

    prompts = []
    for r in rows:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": instruction + "\n\n" + r["question_ur"]},
        ]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))
    print("[prompt] example 0 (repr):")
    print(repr(prompts[0]))

    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0:
        eos_ids.add(im_end)

    # ---- base model, loaded ONCE, same 4-bit config as training ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("[load] loading base model in 4-bit nf4 (same as training) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.eval()

    # ---- attach every adapter once ----
    adapter_tags = [(c, s, p) for (c, s, p) in configs if p is not None]
    peft_model = None
    for i, (cond, seed, path) in enumerate(adapter_tags):
        tag = "%s_seed%d" % (cond, seed)
        if i == 0:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=tag)
        else:
            peft_model.load_adapter(path, adapter_name=tag)
        print("[load] adapter attached: %s" % tag)
    active = peft_model if peft_model is not None else model
    active.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []

    for idx, (cond, seed, path) in enumerate(configs, start=1):
        tag = cond if seed is None else "%s_seed%d" % (cond, seed)
        print("\n" + "=" * 70)
        print("[%d/%d] %s" % (idx, len(configs), tag))
        print("=" * 70, flush=True)

        if path is None:
            if peft_model is not None:
                with peft_model.disable_adapter():
                    print("  [adapter] DISABLED (pure base model)")
                    gens = generate_all(active, tok, prompts, args.batch_size, eos_ids)
            else:
                gens = generate_all(active, tok, prompts, args.batch_size, eos_ids)
        else:
            peft_model.set_adapter(tag)
            print("  [adapter] active = %r" % peft_model.active_adapter)
            if peft_model.active_adapter != tag:
                die("adapter switch failed: wanted %s, got %s"
                    % (tag, peft_model.active_adapter))
            gens = generate_all(active, tok, prompts, args.batch_size, eos_ids)

        records = []
        for r, (text, n_new, truncated) in zip(rows, gens):
            records.append({
                "qid": r["qid"],
                "gold": r["gold"],
                "pred": extract_answer(text),
                "truncated": truncated,
                "n_gen_tokens": n_new,
                "generation": text,
            })

        s = score(records)
        s["condition"] = cond
        s["seed"] = seed
        s["tag"] = tag
        summary.append(s)

        suffix = "_TEST" if args.test else ""
        out_path = os.path.join(OUT_DIR, "dev200_%s%s.jsonl" % (tag, suffix))
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n_on_disk = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
        if n_on_disk != len(records):
            die("wrote %d records but %s has %d lines on disk"
                % (len(records), out_path, n_on_disk))
        print("  [verified] %d lines on disk" % n_on_disk)

        print("  accuracy        %6.2f%%" % s["accuracy"])
        print("  unparsed        %6.2f%%  (%d)" % (s["unparsed_rate"], s["unparsed"]))
        print("  truncated       %6.2f%%  (%d)" % (s["truncation_rate"], s["truncated"]))
        print("  predicted yes   %6.2f%%" % s["predicted_yes_rate"])
        print("  raw -> %s" % out_path)
        print("  sample generation (row 0, repr, first 400 chars):")
        print("  " + repr(records[0]["generation"][:400]), flush=True)

    # ---- summary ----
    print("\n" + "=" * 78)
    print("SUMMARY" + ("  (TEST — NOT A RESULT)" if args.test else ""))
    print("=" * 78)
    print("%-14s %10s %10s %10s %10s" %
          ("config", "acc%", "unparsed%", "trunc%", "predYes%"))
    for s in summary:
        print("%-14s %10.2f %10.2f %10.2f %10.2f" %
              (s["tag"], s["accuracy"], s["unparsed_rate"],
               s["truncation_rate"], s["predicted_yes_rate"]))
    print("\nmajority-class floor: %.2f%%" % floor)

    sum_path = os.path.join(OUT_DIR, "dev200_summary%s.json"
                            % ("_TEST" if args.test else ""))
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("summary -> %s" % sum_path)

    if args.test:
        print("\nTEST COMPLETE. Nothing here is a result.")
        return

    # ---- AMENDMENT 3C gate, step 1 (accuracy only) ----
    print("\n" + "=" * 78)
    print("GATE CHECK — AMENDMENT 3C step 1 (accuracy)")
    print("=" * 78)

    by = {}
    for s in summary:
        if s["seed"] is not None:
            by.setdefault(s["condition"], {})[s["seed"]] = s["accuracy"]
    means = {c: sum(by[c].values()) / len(by[c]) for c in by}
    for c in CONDITIONS:
        print("%s mean accuracy: %.2f%%  (seeds: %s)"
              % (c, means[c], {k: by[c][k] for k in SEEDS}))

    unp = [s["unparsed_rate"] for s in summary]
    spread = max(unp) - min(unp)
    print("\nunparsed-rate spread across all configs: %.2f pp" % spread)
    if spread > 5.0:
        print("*** WARNING: unparsed rates differ by more than 5pp. Under "
              "AMENDMENT 5D the accuracy comparison is NOT valid and the gate "
              "is VOID. Diagnose the parsing gap before reading any number. ***")

    beats_c1 = sum(1 for s in SEEDS if by["C3"][s] > by["C1"][s])
    beats_c2 = sum(1 for s in SEEDS if by["C3"][s] > by["C2"][s])
    print("\nC3 beats C1 in %d/3 paired seeds" % beats_c1)
    print("C3 beats C2 in %d/3 paired seeds" % beats_c2)

    passed = (means["C3"] > means["C1"] and means["C3"] > means["C2"]
              and beats_c1 >= 2 and beats_c2 >= 2)
    print("\nGATE STEP 1: %s" % ("PASS" if passed else "FAIL"))
    if not passed:
        print("Per AMENDMENT 3C: stop, log, do NOT run the faithfulness probe.")


if __name__ == "__main__":
    main()
