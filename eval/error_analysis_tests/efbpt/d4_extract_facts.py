#!/usr/bin/env python3
"""
d4_extract_facts.py

DIAGNOSTIC D4: can the model EXTRACT the needed facts from correct articles?

Two passes over the same 71-question subset used by D3:

  PASS 1  EXTRACTION  english question + the same chunks D3 arm O2 used
                      -> short atomic English facts, one per line
  PASS 2  ANSWERING   identical to D1 arm B: frozen Urdu prompt + facts block
                      built from pass 1's lines

Arms A, O1, O2, B are REUSED from D3/D1 outputs on the same qids.

Everything frozen is imported, never re-implemented:
  - prompt constants + AMENDMENT 5 extractor  <- efbpt_eval_dev200.py
  - facts block format                        <- d1_eval_arms.py
  - Devanagari secondary scorer               <- d1_score_dual.py
  - chunk fetching by exact title             <- same logic as D3

Usage (GPU node, from ~/URBench):
  python eval/error_analysis_tests/efbpt/d4_extract_facts.py --test
  python eval/error_analysis_tests/efbpt/d4_extract_facts.py
"""

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from efbpt_eval_dev200 import (          # noqa: E402
    MODEL_PATH, DEV_PATH, SYSTEM_MESSAGE, MAX_NEW_TOKENS,
    extract_answer, load_instruction, die,
)
from d1_eval_arms import facts_block      # noqa: E402
from d1_score_dual import extract_answer_secondary   # noqa: E402
from d3_oracle_retrieval import (         # noqa: E402
    norm, required_titles, fetch_chunks, META_PATH, COV_PATH,
)

D3_DIR = "outputs/efbpt/d3"
D1_ARM_DIR = "outputs/efbpt/d1/arms"
OUT_DIR = "outputs/efbpt/d4"

CHUNKS_PER_TITLE = 3          # equal information access to D3 arm O2
EXTRACT_MAX_NEW = 512

EXTRACT_SYSTEM = (
    "You are a careful research assistant. You extract facts from provided "
    "passages. You never invent information that is not in the passages."
)

EXTRACT_INSTRUCTION = (
    "Read the passages below, then extract the facts needed to answer the "
    "question.\n"
    "Rules:\n"
    "1. Output ONLY short English factual sentences, one per line.\n"
    "2. Each fact must come from the passages. If a needed fact is not in the "
    "passages, do not invent it.\n"
    "3. No numbering, no bullets, no explanation, no answer to the question "
    "itself. Facts only.\n"
    "4. At most 6 facts."
)


def load_rows():
    rows = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "qid": r["urbench_qid"],
                "question_ur": r["question_ur"],
                "question_en": r["question_en"],
                "gold": "yes" if r["answer"] else "no",
                "required": required_titles(r.get("evidence_paragraph_ids")),
            })
    if len(rows) != 200:
        die("DEV200 has %d rows, expected 200" % len(rows))
    return rows


def clean_fact_lines(text):
    """Pass-1 output -> list of fact strings. Strips numbering/bullets that
    rule 3 forbids but a model may still emit; drops empty lines."""
    facts = []
    for line in text.split("\n"):
        s = line.strip()
        s = re.sub(r"^[-*\u2022]\s*", "", s)
        s = re.sub(r"^\d+[.)]\s*", "", s)
        s = s.strip()
        if s:
            facts.append(s)
    return facts


def score(records, key):
    n = len(records)
    correct = sum(1 for r in records if r[key] == r["gold"])
    unparsed = sum(1 for r in records if r[key] is None)
    return OrderedDict([
        ("n", n),
        ("accuracy", round(100.0 * correct / n, 2)),
        ("unparsed_rate", round(100.0 * unparsed / n, 2)),
        ("correct", correct),
    ])


def generate_all(model, tok, prompts, batch_size, eos_ids, max_new):
    n = len(prompts)
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    out = [None] * n
    t0 = time.time()
    done = 0
    for s in range(0, n, batch_size):
        idxs = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt",
                  padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            new_ids = g[j][in_len:]
            keep = [t for t in new_ids.tolist() if t != tok.pad_token_id]
            trunc = (len(keep) >= max_new) and (
                len(new_ids) > 0 and new_ids[-1].item() not in eos_ids)
            out[i] = (tok.decode(new_ids, skip_special_tokens=True),
                      len(keep), trunc)
        done += len(idxs)
        el = time.time() - t0
        print("    %d/%d  (%.1fs, %.1fs/row)" % (done, n, el, el / done),
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="first 8 questions only")
    ap.add_argument("--test-rows", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    instruction = load_instruction()
    print("[ok] Urdu instruction verified")

    cov = json.load(open(COV_PATH, "r", encoding="utf-8"))
    present = {norm(t) for t in cov["present_titles"]}
    absent = {norm(t) for t in cov["absent_titles"]}

    rows = load_rows()
    subset = []
    for r in rows:
        if not r["required"]:
            continue
        for t in r["required"]:
            if norm(t) not in present and norm(t) not in absent:
                die("title %r classified by neither coverage list" % t)
        if all(norm(t) in present for t in r["required"]):
            subset.append(r)
    print("[subset] %d / 200 questions (expect 71)" % len(subset))
    if not args.test and len(subset) != 71:
        die("subset size %d != 71; D3 and D4 would not be comparable"
            % len(subset))
    if args.test:
        subset = subset[:args.test_rows]

    n_yes = sum(1 for r in subset if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(subset) - n_yes) / len(subset)
    print("[subset] yes=%d no=%d floor %.2f%%"
          % (n_yes, len(subset) - n_yes, floor))

    needed = sorted({norm(t) for r in subset for t in r["required"]})
    print("[scan] fetching up to %d chunks for %d titles ..."
          % (CHUNKS_PER_TITLE, len(needed)), flush=True)
    chunks = fetch_chunks(needed, CHUNKS_PER_TITLE)
    missing = [t for t in needed if not chunks[t]]
    if missing:
        die("titles with no chunk despite coverage=present: %s" % missing[:5])

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if im is not None and im >= 0:
        eos_ids.add(im)

    # ---- pass 1 prompts (English extraction) ----
    def extract_prompt(row):
        passages = []
        for t in row["required"]:
            for c in chunks[norm(t)][:CHUNKS_PER_TITLE]:
                passages.append("[%s] %s" % (t, c))
        user = (EXTRACT_INSTRUCTION + "\n\nQuestion: " + row["question_en"]
                + "\n\nPassages:\n" + "\n\n".join(passages))
        msgs = [{"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    p1 = [extract_prompt(r) for r in subset]
    lens = sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in p1)
    print("[len] pass-1 prompts: min=%d median=%d max=%d"
          % (lens[0], lens[len(lens) // 2], lens[-1]))
    if lens[-1] + EXTRACT_MAX_NEW > 40960:
        die("pass-1 prompt would exceed the context window")

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
    suffix = "_TEST" if args.test else ""

    # ---- PASS 1 ----
    print("\n" + "=" * 70)
    print("PASS 1 — EXTRACTION (English -> English)")
    print("=" * 70, flush=True)
    print("  prompt example 0 (repr, first 600 chars):")
    print("  " + repr(p1[0][:600]), flush=True)
    gen1 = generate_all(model, tok, p1, args.batch_size, eos_ids, EXTRACT_MAX_NEW)

    extracted = []
    n_empty = 0
    for r, (text, n_new, trunc) in zip(subset, gen1):
        facts = clean_fact_lines(text)
        if not facts:
            n_empty += 1
        extracted.append({
            "qid": r["qid"], "question_en": r["question_en"],
            "titles": r["required"], "raw_extraction": text,
            "facts": facts, "n_facts": len(facts), "truncated": trunc,
        })
    nf = sorted(e["n_facts"] for e in extracted)
    print("  facts per question: min=%d median=%d max=%d  empty=%d"
          % (nf[0], nf[len(nf) // 2], nf[-1], n_empty))
    print("  sample extraction (row 0):")
    for f in extracted[0]["facts"][:6]:
        print("    - %s" % f)

    pex = os.path.join(OUT_DIR, "d4_extractions%s.jsonl" % suffix)
    with open(pex, "w", encoding="utf-8") as f:
        for e in extracted:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    nd = sum(1 for _ in open(pex, "r", encoding="utf-8"))
    if nd != len(extracted):
        die("extractions: wrote %d, disk has %d" % (len(extracted), nd))
    print("  [verified] %d lines -> %s" % (nd, pex))

    # ---- PASS 2 (identical to D1 arm B in structure) ----
    print("\n" + "=" * 70)
    print("PASS 2 — ANSWERING (arm X1, frozen Urdu prompt)")
    print("=" * 70, flush=True)

    def answer_prompt(row, facts):
        # empty extraction -> the model gets no facts block, same as arm A;
        # recorded per-row so the effect is visible, never hidden
        if facts:
            user = "\n\n".join([instruction, facts_block(facts),
                                row["question_ur"]])
        else:
            user = "\n\n".join([instruction, row["question_ur"]])
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    p2 = [answer_prompt(r, e["facts"]) for r, e in zip(subset, extracted)]
    print("  prompt example 0 (repr, first 500 chars):")
    print("  " + repr(p2[0][:500]), flush=True)
    gen2 = generate_all(model, tok, p2, args.batch_size, eos_ids, MAX_NEW_TOKENS)

    recs = []
    for r, e, (text, n_new, trunc) in zip(subset, extracted, gen2):
        recs.append({
            "qid": r["qid"], "arm": "X1", "gold": r["gold"],
            "pred": extract_answer(text),
            "pred_secondary": extract_answer_secondary(text),
            "truncated": trunc, "n_gen_tokens": n_new,
            "n_facts_given": e["n_facts"], "generation": text,
        })
    px = os.path.join(OUT_DIR, "d4_armX1%s.jsonl" % suffix)
    with open(px, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    nd = sum(1 for _ in open(px, "r", encoding="utf-8"))
    if nd != len(recs):
        die("armX1: wrote %d, disk has %d" % (len(recs), nd))
    print("  [verified] %d lines -> %s" % (nd, px))

    # ---- reuse D3/D1 arms on the same qids ----
    qids = {r["qid"] for r in subset}
    ref = {}
    sources = {"O1": os.path.join(D3_DIR, "d3_armO1.jsonl"),
               "O2": os.path.join(D3_DIR, "d3_armO2.jsonl"),
               "A": os.path.join(D1_ARM_DIR, "d1_armA.jsonl"),
               "B": os.path.join(D1_ARM_DIR, "d1_armB.jsonl")}
    for arm, path in sources.items():
        if not os.path.exists(path):
            die("missing reference output: " + path)
        rr = []
        seen = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["qid"] in qids:
                    if "pred_secondary" not in r:
                        r["pred_secondary"] = extract_answer_secondary(
                            r["generation"])
                    rr.append(r)
                    seen.add(r["qid"])
        if seen != qids:
            die("arm %s covers %d/%d subset qids" % (arm, len(seen), len(qids)))
        ref[arm] = rr
        print("[reuse] arm %s: %d rows" % (arm, len(rr)))
    ref["X1"] = recs

    # ---- report ----
    print("\n" + "=" * 84)
    print("D4 RESULT%s   n=%d, floor %.2f%%"
          % ("  (TEST — NOT A RESULT)" if args.test else "", len(subset), floor))
    print("=" * 84)
    label = {"A": "none", "O1": "1 wiki chunk/title", "O2": "3 wiki chunks/title",
             "X1": "EXTRACTED facts (pass 1)", "B": "clean gold facts"}
    acc, unp = {}, []
    for arm in ("A", "O1", "O2", "X1", "B"):
        s = score(ref[arm], "pred")
        acc[arm] = s["accuracy"]
        unp.append(s["unparsed_rate"])
        print("%-4s %-28s acc %6.2f%%   unparsed %5.2f%%"
              % (arm, label[arm], s["accuracy"], s["unparsed_rate"]))
    spread = max(unp) - min(unp)
    print("\nunparsed spread: %.2f pp" % spread)
    if spread > 5.0:
        print("*** AMENDMENT 5D: spread > 5pp, accuracy comparison is VOID.")

    if args.test:
        print("\nTEST COMPLETE. Not a result.")
        return

    # ---- McNemar X1 vs A and the recovered fraction ----
    a_by = {r["qid"]: r for r in ref["A"]}
    b_pairs = c_pairs = 0
    for r in recs:
        ok_x = r["pred"] == r["gold"]
        ok_a = a_by[r["qid"]]["pred"] == a_by[r["qid"]]["gold"]
        if ok_x and not ok_a:
            b_pairs += 1
        elif ok_a and not ok_x:
            c_pairs += 1
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b_pairs, c_pairs), b_pairs + c_pairs, 0.5).pvalue \
            if (b_pairs + c_pairs) else 1.0
    except Exception:
        p = None
    gap = acc["B"] - acc["A"]
    got = acc["X1"] - acc["A"]
    print("\n" + "-" * 84)
    print("PRE-DECLARED READING (D4 Section D)")
    print("-" * 84)
    print("  gap (B - A)              = %+.2f pp" % gap)
    print("  X1 - A                   = %+.2f pp" % got)
    if gap > 0:
        print("  fraction of gap recovered = %.0f%%" % (100.0 * got / gap))
    print("  paired X1 vs A: X1_only=%d  A_only=%d  p=%s"
          % (b_pairs, c_pairs, ("%.4f" % p) if p is not None else "scipy missing"))
    if p is not None and gap > 0:
        if got >= 0.6 * gap and p < 0.05:
            print("\n  Reading 1: EXTRACTION WORKS. Method validated end-to-end at")
            print("  the oracle-linking level. Remaining research: entity linking")
            print("  and a corpus replacement.")
        elif p >= 0.05:
            print("\n  Reading 2: extraction gain not significant. Page-derived")
            print("  fact pipelines close; the fact SOURCE itself must change.")
        else:
            print("\n  Reading 3: partial recovery, significant. Decide with the")
            print("  supervisor, not unilaterally.")

    print("\n  Section D item 4 reminder: hand-check 10 extraction rows in")
    print("  %s for invented facts before quoting any number." % pex)

    summ = {"n": len(subset), "floor": floor, "accuracy": acc,
            "unparsed_spread": spread, "x1_vs_a": {"b": b_pairs, "c": c_pairs,
            "p": p}}
    ps = os.path.join(OUT_DIR, "d4_summary%s.json" % suffix)
    with open(ps, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("\n[verified] wrote %s" % ps)


if __name__ == "__main__":
    main()
