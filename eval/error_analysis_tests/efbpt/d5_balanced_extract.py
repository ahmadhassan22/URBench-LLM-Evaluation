#!/usr/bin/env python3
"""
d5_balanced_extract.py

DIAGNOSTIC D5: does BALANCED extraction beat free-allocation extraction?

Declared in docs/EFBPT_PLAN_A_FREEZE.md -> DIAGNOSTIC D5 before execution.
Step-0 existence gate PASSED: 39.7% of multi-title rows had >= 1 title
receiving zero facts (strict rule 50.0%).

ARMS (same 71 qids as D4):
  X1  reused from outputs/efbpt/d4/d4_armX1.jsonl — 6 facts, free allocation
  X2  BALANCED — one pass-1 call PER TITLE, at most 2 facts per title
  X3  free allocation, total budget MATCHED to X2's realized count per row

X3 is the control that separates "more facts" from "balanced facts".
Without it, X2 vs X1 alone cannot distinguish the two.

Everything frozen is imported from the D4 script, never re-implemented, so
decoding, prompts, facts-block format and scoring are identical by
construction.

Usage (GPU node, from ~/URBench):
  python eval/error_analysis_tests/efbpt/d5_balanced_extract.py --test
  python eval/error_analysis_tests/efbpt/d5_balanced_extract.py
"""

import argparse
import hashlib
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
    MODEL_PATH, SYSTEM_MESSAGE, MAX_NEW_TOKENS,
    extract_answer, load_instruction, die,
)
from d1_eval_arms import facts_block      # noqa: E402
from d1_score_dual import extract_answer_secondary   # noqa: E402
from d3_oracle_retrieval import norm, fetch_chunks, COV_PATH   # noqa: E402
from d4_extract_facts import (            # noqa: E402
    load_rows, clean_fact_lines, generate_all, score,
    EXTRACT_SYSTEM, EXTRACT_INSTRUCTION, CHUNKS_PER_TITLE, EXTRACT_MAX_NEW,
)
from d5_step0_gate import attribute       # noqa: E402

D4_DIR = "outputs/efbpt/d4"
OUT_DIR = "outputs/efbpt/d5"
FACTS_PER_TITLE = 2          # declared in the freeze

# Instruction template. Asserted below to reproduce D4's EXTRACT_INSTRUCTION
# byte-for-byte at n=6, which guarantees arm X3 at n=6 is the same prompt X1
# used — the only differences between arms are the budget and the passages.
INSTRUCTION_TMPL = (
    "Read the passages below, then extract the facts needed to answer the "
    "question.\n"
    "Rules:\n"
    "1. Output ONLY short English factual sentences, one per line.\n"
    "2. Each fact must come from the passages. If a needed fact is not in the "
    "passages, do not invent it.\n"
    "3. No numbering, no bullets, no explanation, no answer to the question "
    "itself. Facts only.\n"
    "4. At most %d facts."
)


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def verify_instruction_template():
    built = INSTRUCTION_TMPL % 6
    if built != EXTRACT_INSTRUCTION:
        die("INSTRUCTION_TMPL at n=6 does not reproduce D4's "
            "EXTRACT_INSTRUCTION.\n  built: %r\n  d4   : %r"
            % (built, EXTRACT_INSTRUCTION))
    print("[ok] instruction template reproduces D4 exactly at n=6 (md5 %s)"
          % md5(built))


def mcnemar(recs_x, recs_y, label_x, label_y):
    """Paired exact McNemar on primary preds. Returns (b, c, p)."""
    y_by = {r["qid"]: r for r in recs_y}
    b = c = 0
    for r in recs_x:
        o = y_by[r["qid"]]
        ok_x = r["pred"] == r["gold"]
        ok_y = o["pred"] == o["gold"]
        if ok_x and not ok_y:
            b += 1
        elif ok_y and not ok_x:
            c += 1
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    except Exception:
        p = None
    print("  %-8s vs %-8s   b=%-3d c=%-3d  p=%s"
          % (label_x, label_y, b, c,
             ("%.4f" % p) if p is not None else "n/a"))
    return b, c, p


def coverage(fact_map):
    """fact_map: {qid: (titles, facts)}. Manipulation check only — for X2 a
    high value is forced by construction and is NEVER a result."""
    slots = filled = 0
    zero_rows = multi_rows = 0
    for qid, (titles, facts) in fact_map.items():
        if len(titles) < 2:
            continue
        multi_rows += 1
        cnt = {t: 0 for t in titles}
        for f in facts:
            _, decl = attribute(f, titles)
            for t in decl:
                cnt[t] += 1
        slots += len(titles)
        filled += sum(1 for t in titles if cnt[t] > 0)
        if any(cnt[t] == 0 for t in titles):
            zero_rows += 1
    return OrderedDict([
        ("multi_title_rows", multi_rows),
        ("title_slots", slots),
        ("slots_with_ge1_fact_pct", round(100.0 * filled / max(slots, 1), 1)),
        ("rows_with_a_zero_fact_title", zero_rows),
        ("rows_with_a_zero_fact_title_pct",
         round(100.0 * zero_rows / max(multi_rows, 1), 1)),
    ])


def write_jsonl(path, recs, what):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    nd = sum(1 for _ in open(path, "r", encoding="utf-8"))
    if nd != len(recs):
        die("%s: wrote %d, disk has %d" % (what, len(recs), nd))
    print("  [verified] %d lines -> %s" % (nd, path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="first 8 questions only")
    ap.add_argument("--test-rows", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true",
                    help="required to replace existing non-TEST outputs")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    verify_instruction_template()
    instruction = load_instruction()
    print("[ok] Urdu instruction verified")

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_TEST" if args.test else ""

    # ---- no-overwrite guard (freeze section 8; D4 lacked this) ----
    outs = [os.path.join(OUT_DIR, n % suffix) for n in
            ("d5_extractions_X2%s.jsonl", "d5_extractions_X3%s.jsonl",
             "d5_armX2%s.jsonl", "d5_armX3%s.jsonl", "d5_summary%s.json")]
    if not args.test and not args.overwrite:
        exist = [p for p in outs if os.path.exists(p)]
        if exist:
            die("refusing to overwrite existing outputs (pass --overwrite "
                "only if you mean it):\n  " + "\n  ".join(exist))

    # ---- same subset construction as D4 ----
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
        die("subset size %d != 71; D4 and D5 would not be comparable"
            % len(subset))
    if args.test:
        subset = subset[:args.test_rows]

    n_yes = sum(1 for r in subset if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(subset) - n_yes) / len(subset)
    print("[subset] yes=%d no=%d floor %.2f%%"
          % (n_yes, len(subset) - n_yes, floor))

    # ---- reuse arm X1 and confirm qid identity ----
    x1_path = os.path.join(D4_DIR, "d4_armX1.jsonl")
    if not os.path.exists(x1_path):
        die("missing " + x1_path)
    x1 = [json.loads(l) for l in open(x1_path, encoding="utf-8") if l.strip()]
    qids = [r["qid"] for r in subset]
    x1 = [r for r in x1 if r["qid"] in set(qids)]
    if not args.test and {r["qid"] for r in x1} != set(qids):
        die("arm X1 does not cover the D5 subset exactly")
    print("[reuse] arm X1: %d rows" % len(x1))

    x1_facts = {}
    for l in open(os.path.join(D4_DIR, "d4_extractions.jsonl"),
                  encoding="utf-8"):
        e = json.loads(l)
        if e["qid"] in set(qids):
            x1_facts[e["qid"]] = (e["titles"], e["facts"])

    # ---- chunks (identical fetch to D4) ----
    needed = sorted({norm(t) for r in subset for t in r["required"]})
    print("[scan] fetching up to %d chunks for %d titles ..."
          % (CHUNKS_PER_TITLE, len(needed)), flush=True)
    chunks = fetch_chunks(needed, CHUNKS_PER_TITLE)
    missing = [t for t in needed if not chunks[t]]
    if missing:
        die("titles with no chunk despite coverage=present: %s" % missing[:5])

    # ---- tokenizer / model (identical config to D4) ----
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

    def extract_prompt(row, titles, n_facts):
        passages = []
        for t in titles:
            for c in chunks[norm(t)][:CHUNKS_PER_TITLE]:
                passages.append("[%s] %s" % (t, c))
        user = ((INSTRUCTION_TMPL % n_facts) + "\n\nQuestion: "
                + row["question_en"] + "\n\nPassages:\n" + "\n\n".join(passages))
        msgs = [{"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    def answer_prompt(row, facts):
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

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("[load] base model, 4-bit nf4, no adapters ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    # ================= PASS 1a — X2 BALANCED, one call per title =========
    print("\n" + "=" * 70)
    print("PASS 1a — X2 BALANCED EXTRACTION (%d facts per title)"
          % FACTS_PER_TITLE)
    print("=" * 70, flush=True)
    jobs = [(r, t) for r in subset for t in r["required"]]
    print("  %d title-calls across %d questions" % (len(jobs), len(subset)))
    p1a = [extract_prompt(r, [t], FACTS_PER_TITLE) for r, t in jobs]
    print("  prompt example 0 (repr, first 600 chars):")
    print("  " + repr(p1a[0][:600]), flush=True)
    gen1a = generate_all(model, tok, p1a, args.batch_size, eos_ids,
                         EXTRACT_MAX_NEW)

    per_title = {}
    trunc_x2 = 0
    for (r, t), (text, n_new, trunc) in zip(jobs, gen1a):
        f = clean_fact_lines(text)[:FACTS_PER_TITLE]
        per_title.setdefault(r["qid"], []).append(
            {"title": t, "facts": f, "raw": text, "truncated": trunc})
        trunc_x2 += int(trunc)

    x2_extract, x2_facts = [], {}
    for r in subset:
        blocks = per_title[r["qid"]]
        facts = [f for b in blocks for f in b["facts"]]
        x2_facts[r["qid"]] = (r["required"], facts)
        x2_extract.append({
            "qid": r["qid"], "question_en": r["question_en"],
            "titles": r["required"], "per_title": blocks,
            "facts": facts, "n_facts": len(facts),
            "truncated": any(b["truncated"] for b in blocks),
        })
    nf = sorted(e["n_facts"] for e in x2_extract)
    print("  X2 facts/question: min=%d median=%d max=%d  empty=%d  "
          "truncated title-calls=%d"
          % (nf[0], nf[len(nf) // 2], nf[-1],
             sum(1 for e in x2_extract if e["n_facts"] == 0), trunc_x2))
    write_jsonl(os.path.join(OUT_DIR, "d5_extractions_X2%s.jsonl" % suffix),
                x2_extract, "X2 extractions")

    # ================= PASS 1b — X3 FREE, budget matched to X2 ===========
    print("\n" + "=" * 70)
    print("PASS 1b — X3 FREE EXTRACTION, budget matched to X2 per row")
    print("=" * 70, flush=True)
    budgets = {e["qid"]: max(e["n_facts"], 1) for e in x2_extract}
    bs = sorted(budgets.values())
    print("  budget per row: min=%d median=%d max=%d"
          % (bs[0], bs[len(bs) // 2], bs[-1]))
    p1b = [extract_prompt(r, r["required"], budgets[r["qid"]]) for r in subset]
    gen1b = generate_all(model, tok, p1b, args.batch_size, eos_ids,
                         EXTRACT_MAX_NEW)

    x3_extract, x3_facts = [], {}
    for r, (text, n_new, trunc) in zip(subset, gen1b):
        f = clean_fact_lines(text)[:budgets[r["qid"]]]
        x3_facts[r["qid"]] = (r["required"], f)
        x3_extract.append({
            "qid": r["qid"], "question_en": r["question_en"],
            "titles": r["required"], "budget": budgets[r["qid"]],
            "raw_extraction": text, "facts": f, "n_facts": len(f),
            "truncated": trunc,
        })
    nf = sorted(e["n_facts"] for e in x3_extract)
    print("  X3 facts/question: min=%d median=%d max=%d"
          % (nf[0], nf[len(nf) // 2], nf[-1]))
    write_jsonl(os.path.join(OUT_DIR, "d5_extractions_X3%s.jsonl" % suffix),
                x3_extract, "X3 extractions")

    tot2 = sum(e["n_facts"] for e in x2_extract)
    tot3 = sum(e["n_facts"] for e in x3_extract)
    print("  total facts: X2=%d  X3=%d  (matched design; residual gap %+d)"
          % (tot2, tot3, tot3 - tot2))

    # ================= PASS 2 — answering, identical to D4 ===============
    def run_pass2(arm, extract_recs):
        print("\n" + "=" * 70)
        print("PASS 2 — ANSWERING (arm %s, frozen Urdu prompt)" % arm)
        print("=" * 70, flush=True)
        p2 = [answer_prompt(r, e["facts"])
              for r, e in zip(subset, extract_recs)]
        gen2 = generate_all(model, tok, p2, args.batch_size, eos_ids,
                            MAX_NEW_TOKENS)
        recs = []
        for r, e, (text, n_new, trunc) in zip(subset, extract_recs, gen2):
            recs.append({
                "qid": r["qid"], "arm": arm, "gold": r["gold"],
                "pred": extract_answer(text),
                "pred_secondary": extract_answer_secondary(text),
                "truncated": trunc, "n_gen_tokens": n_new,
                "n_facts_given": e["n_facts"], "generation": text,
            })
        write_jsonl(os.path.join(OUT_DIR, "d5_arm%s%s.jsonl" % (arm, suffix)),
                    recs, "arm " + arm)
        return recs

    x2 = run_pass2("X2", x2_extract)
    x3 = run_pass2("X3", x3_extract)

    # ================= REPORT ============================================
    print("\n" + "=" * 84)
    print("D5 RESULT%s   n=%d, floor %.2f%%"
          % ("  (TEST — NOT A RESULT)" if args.test else "", len(subset), floor))
    print("=" * 84)
    arms = OrderedDict([("X1", x1), ("X2", x2), ("X3", x3)])
    label = {"X1": "free, <=6 facts (D4)",
             "X2": "BALANCED, %d facts/title" % FACTS_PER_TITLE,
             "X3": "free, budget matched to X2"}
    acc, unp = {}, []
    for a, recs in arms.items():
        s = score(recs, "pred")
        s2 = score(recs, "pred_secondary")
        acc[a] = s["accuracy"]
        unp.append(s["unparsed_rate"])
        print("%-4s %-30s acc %6.2f%%  unparsed %5.2f%%   "
              "(secondary %6.2f%% / %5.2f%%)"
              % (a, label[a], s["accuracy"], s["unparsed_rate"],
                 s2["accuracy"], s2["unparsed_rate"]))
    spread = max(unp) - min(unp)
    print("\nunparsed spread across X1/X2/X3: %.2f pp" % spread)
    if spread > 5.0:
        print("*** VALIDITY: spread > 5pp, the accuracy comparison is VOID.")

    print("\nCOVERAGE (manipulation check for X2 — forced by construction,")
    print("never reported as a result):")
    for a, fm in (("X1", x1_facts), ("X2", x2_facts), ("X3", x3_facts)):
        c = coverage(fm)
        print("  %-3s slots with >=1 fact %5.1f%%   rows with a zero-fact "
              "title %5.1f%% (%d/%d)"
              % (a, c["slots_with_ge1_fact_pct"],
                 c["rows_with_a_zero_fact_title_pct"],
                 c["rows_with_a_zero_fact_title"], c["multi_title_rows"]))

    if args.test:
        print("\nTEST COMPLETE. Not a result.")
        return

    print("\n" + "-" * 84)
    print("PRE-DECLARED READINGS (D5)")
    print("-" * 84)
    print("PRIMARY:")
    b, c, p = mcnemar(x2, x1, "X2", "X1")
    print("SECONDARY (mechanism):")
    b31, c31, p31 = mcnemar(x3, x1, "X3", "X1")
    b23, c23, p23 = mcnemar(x2, x3, "X2", "X3")

    print("\n  X2 - X1 = %+.2f pp   (headroom to gold facts B=83.10 was "
          "%+.2f pp)" % (acc["X2"] - acc["X1"], 83.10 - acc["X1"]))
    if p is None:
        print("  scipy unavailable; p not computed")
    elif p < 0.05 and acc["X2"] > acc["X1"]:
        print("\n  READING A: balance HELPS. Adopt the X2 form.")
    elif p < 0.05 and acc["X2"] < acc["X1"]:
        print("\n  READING B: balance HURTS. Keep the X1 form.")
    else:
        print("\n  READING C: no detectable effect at n=71. This was the")
        print("  EXPECTED outcome given the small headroom. Report it as a")
        print("  finding, keep the X1 form, move to entity linking.")

    summ = {"n": len(subset), "floor": floor, "accuracy": acc,
            "unparsed_spread": spread,
            "facts_total": {"X2": tot2, "X3": tot3},
            "coverage": {"X1": coverage(x1_facts), "X2": coverage(x2_facts),
                         "X3": coverage(x3_facts)},
            "x2_vs_x1": {"b": b, "c": c, "p": p},
            "x3_vs_x1": {"b": b31, "c": c31, "p": p31},
            "x2_vs_x3": {"b": b23, "c": c23, "p": p23}}
    ps = os.path.join(OUT_DIR, "d5_summary%s.json" % suffix)
    with open(ps, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("\n[verified] wrote %s" % ps)


if __name__ == "__main__":
    main()
