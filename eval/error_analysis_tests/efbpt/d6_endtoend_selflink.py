#!/usr/bin/env python3
"""
d6_endtoend_selflink.py

DIAGNOSTIC D6: end-to-end Urdu multi-hop with SELF-GENERATED entities.
Declared in docs/EFBPT_PLAN_A_FREEZE.md -> DIAGNOSTIC D6 before execution.

Arm E1, four stages, NO gold information at any stage:
  0  base Qwen3-14B reads question_ur -> [{urdu_span, canonical_title}, ...]
  1  each generated title -> norm() -> EXACT lookup in the corpus
  2  up to CHUNKS_PER_TITLE=3 chunks per matched title
  3  D4 pass-1 extraction (<=6 facts) then the frozen Urdu answer prompt

Controls A (59.15%) and X1 (76.06%) are reused from disk, never re-run.

Everything after stage 1 is IMPORTED from d4_extract_facts.py so decoding,
chunk count, extraction prompt, facts-block format and scorers are identical
to arm X1 by construction.

Usage (GPU node, from ~/URBench):
  python eval/error_analysis_tests/efbpt/d6_endtoend_selflink.py --test
  python eval/error_analysis_tests/efbpt/d6_endtoend_selflink.py
"""

import argparse
import hashlib
import json
import os
import re
import sys
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

OUT_DIR = "outputs/efbpt/d6"
D1_ARM_DIR = "outputs/efbpt/d1/arms"
D4_DIR = "outputs/efbpt/d4"
ENTITY_MAX_NEW = 512
MAX_UNIQUE_TITLES = 2000     # compute guardrail only, not a design cap

ENTITY_SYSTEM = ("You are a careful assistant that identifies real-world "
                 "entities in questions. You answer only in the requested "
                 "format.")

ENTITY_INSTRUCTION = (
    "Read the question below. List the real-world entities it mentions.\n"
    "For each entity give:\n"
    "  urdu_span       - the exact span copied from the question\n"
    "  canonical_title - the English Wikipedia article title for that entity\n"
    "Rules:\n"
    "1. Output ONLY a JSON array. No explanation, no markdown, no code "
    "fences.\n"
    "2. Do not answer the question.\n"
    "3. If you are unsure of an entity, still give your best English "
    "Wikipedia title.\n"
    "Format:\n"
    '[{"urdu_span": "...", "canonical_title": "..."}]'
)


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# stage 0 parsing.  Never raises.  Returns (pairs, status) where status is
# one of ok / json_error / empty / no_array.  Unparseable rows are KEPT and
# answered with no facts, never dropped (freeze section H).
# --------------------------------------------------------------------------
_OBJ_RE = re.compile(
    r'\{[^{}]*?"urdu_span"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*?'
    r'"canonical_title"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*?\}',
    re.S)


def parse_entities(text):
    if text is None:
        return [], "empty"
    t = text.strip()
    if not t:
        return [], "empty"
    # strip any code fence wrapper without touching inner content
    if "```" in t:
        parts = t.split("```")
        cand = max(parts, key=len)
        t = cand[cand.find("["):] if "[" in cand else cand
    i, j = t.find("["), t.rfind("]")
    if i != -1 and j > i:
        try:
            arr = json.loads(t[i:j + 1])
            out = []
            if isinstance(arr, list):
                for e in arr:
                    if not isinstance(e, dict):
                        continue
                    ti = e.get("canonical_title")
                    sp = e.get("urdu_span", "")
                    if isinstance(ti, str) and ti.strip():
                        out.append({"urdu_span": sp if isinstance(sp, str)
                                    else "", "canonical_title": ti.strip()})
                if out:
                    return out, "ok"
                return [], "empty"
        except Exception:
            pass
    # regex fallback for malformed JSON
    out = []
    for m in _OBJ_RE.finditer(t):
        ti = m.group(2).strip()
        if ti:
            out.append({"urdu_span": m.group(1), "canonical_title": ti})
    if out:
        return out, "json_error"
    return [], "no_array"


def dedupe_titles(pairs):
    seen, out = set(), []
    for p in pairs:
        n = norm(p["canonical_title"])
        if n and n not in seen:
            seen.add(n)
            out.append(p["canonical_title"])
    return out


def mcnemar(recs_x, recs_y, lx, ly):
    y = {r["qid"]: r for r in recs_y}
    b = c = 0
    for r in recs_x:
        o = y[r["qid"]]
        ok_x, ok_y = r["pred"] == r["gold"], o["pred"] == o["gold"]
        if ok_x and not ok_y:
            b += 1
        elif ok_y and not ok_x:
            c += 1
    p = None
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    except Exception:
        pass
    print("  %-4s vs %-4s   b=%-3d c=%-3d  p=%s"
          % (lx, ly, b, c, ("%.4f" % p) if p is not None else "n/a"))
    return b, c, p


def acc_of(recs):
    return 100.0 * sum(1 for r in recs if r["pred"] == r["gold"]) / len(recs)


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


def load_arm(path, qids, label):
    """Load a control arm. Arms written before AMENDMENT 5 have no
    pred_secondary field; backfill it from the stored generation so the
    dual scoring in freeze section C can run. Primary preds are never
    touched."""
    if not os.path.exists(path):
        die("missing control arm: %s" % path)
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["qid"] in qids]
    filled = copied = 0
    for r in recs:
        if "pred_secondary" in r:
            continue
        if isinstance(r.get("generation"), str):
            r["pred_secondary"] = extract_answer_secondary(r["generation"])
            filled += 1
        else:
            r["pred_secondary"] = r.get("pred")
            copied += 1
    note = ""
    if filled or copied:
        note = ("   [backfilled pred_secondary: %d from generation, %d "
                "copied from primary]" % (filled, copied))
    print("[reuse] %s: %d rows%s" % (label, len(recs), note))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test-rows", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    print("[freeze] stage-0 instruction md5 %s" % md5(ENTITY_INSTRUCTION))
    print("[freeze] extraction instruction md5 %s (imported from D4)"
          % md5(EXTRACT_INSTRUCTION))
    instruction = load_instruction()
    print("[ok] Urdu answer instruction verified")

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_TEST" if args.test else ""
    outs = [os.path.join(OUT_DIR, n % suffix) for n in
            ("d6_entities%s.jsonl", "d6_extractions%s.jsonl",
             "d6_armE1%s.jsonl", "d6_summary%s.json")]
    if not args.test and not args.overwrite:
        ex = [p for p in outs if os.path.exists(p)]
        if ex:
            die("refusing to overwrite existing outputs (use --overwrite only "
                "if you mean it):\n  " + "\n  ".join(ex))

    # ---- subset: identical construction to D4/D5 ----
    cov = json.load(open(COV_PATH, encoding="utf-8"))
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
        die("subset size %d != 71; D6 would not be comparable to D4"
            % len(subset))
    if args.test:
        subset = subset[:args.test_rows]
    qids = {r["qid"] for r in subset}
    n_yes = sum(1 for r in subset if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(subset) - n_yes) / len(subset)
    print("[subset] yes=%d no=%d floor %.2f%%"
          % (n_yes, len(subset) - n_yes, floor))

    armA = load_arm(os.path.join(D1_ARM_DIR, "d1_armA.jsonl"), qids, "arm A")
    armX1 = load_arm(os.path.join(D4_DIR, "d4_armX1.jsonl"), qids, "arm X1")
    if not args.test and (len(armA) != len(subset) or len(armX1) != len(subset)):
        die("control arms do not cover the subset exactly")

    # ---- model ----
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

    def chat(system, user):
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("[load] base model, 4-bit nf4, NO adapters ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    # ================= STAGE 0 — entity generation (Urdu question only) ===
    print("\n" + "=" * 70)
    print("STAGE 0 — SELF-GENERATED ENTITIES (question_ur only, no gold)")
    print("=" * 70, flush=True)
    p0 = [chat(ENTITY_SYSTEM, ENTITY_INSTRUCTION + "\n\nQuestion: "
               + r["question_ur"]) for r in subset]
    print("  prompt example 0 (repr, first 700 chars):")
    print("  " + repr(p0[0][:700]), flush=True)
    gen0 = generate_all(model, tok, p0, args.batch_size, eos_ids,
                        ENTITY_MAX_NEW)

    ent_recs, status_ct = [], OrderedDict(
        [("ok", 0), ("json_error", 0), ("empty", 0), ("no_array", 0)])
    for r, (text, n_new, trunc) in zip(subset, gen0):
        pairs, st = parse_entities(text)
        status_ct[st] = status_ct.get(st, 0) + 1
        ent_recs.append({"qid": r["qid"], "question_ur": r["question_ur"],
                         "raw": text, "parse_status": st, "entities": pairs,
                         "titles": dedupe_titles(pairs), "truncated": trunc})
    ne = sorted(len(e["titles"]) for e in ent_recs)
    print("  parse status: " + "  ".join("%s=%d" % (k, v)
                                         for k, v in status_ct.items()))
    print("  entities/question: min=%d median=%d max=%d  zero-entity rows=%d"
          % (ne[0], ne[len(ne) // 2], ne[-1], sum(1 for x in ne if x == 0)))
    write_jsonl(os.path.join(OUT_DIR, "d6_entities%s.jsonl" % suffix),
                ent_recs, "entities")

    # ================= STAGE 1 — EXACT corpus lookup ======================
    print("\n" + "=" * 70)
    print("STAGE 1 — EXACT normalized lookup in the corpus (no fuzzy match)")
    print("=" * 70, flush=True)
    gen_titles = {e["qid"]: e["titles"] for e in ent_recs}
    needed = sorted({norm(t) for ts in gen_titles.values() for t in ts})
    if len(needed) > MAX_UNIQUE_TITLES:
        die("compute guardrail: %d unique generated titles exceeds %d; "
            "inspect stage 0 output before proceeding"
            % (len(needed), MAX_UNIQUE_TITLES))
    print("  %d unique generated titles to look up" % len(needed))
    chunks = fetch_chunks(needed, CHUNKS_PER_TITLE) if needed else {}
    matched = {t for t in needed if chunks.get(t)}
    print("  matched in corpus: %d / %d = %.1f%%"
          % (len(matched), len(needed),
             100.0 * len(matched) / max(len(needed), 1)))

    # ================= STAGE 2+3 — extraction then answering ==============
    print("\n" + "=" * 70)
    print("STAGE 2+3 — FACT EXTRACTION (D4 prompt, <=6 facts)")
    print("=" * 70, flush=True)
    ex_idx, ex_prompts = [], []
    for i, r in enumerate(subset):
        hits = [t for t in gen_titles[r["qid"]] if chunks.get(norm(t))]
        if not hits:
            continue
        passages = ["[%s] %s" % (t, c) for t in hits
                    for c in chunks[norm(t)][:CHUNKS_PER_TITLE]]
        user = (EXTRACT_INSTRUCTION + "\n\nQuestion: " + r["question_en"]
                + "\n\nPassages:\n" + "\n\n".join(passages))
        ex_idx.append(i)
        ex_prompts.append(chat(EXTRACT_SYSTEM, user))
    print("  %d / %d rows have at least one matched title"
          % (len(ex_idx), len(subset)))

    facts_by_i = {}
    if ex_prompts:
        gen1 = generate_all(model, tok, ex_prompts, args.batch_size, eos_ids,
                            EXTRACT_MAX_NEW)
        for i, (text, n_new, trunc) in zip(ex_idx, gen1):
            facts_by_i[i] = (clean_fact_lines(text)[:6], text, trunc)

    ext_recs = []
    for i, r in enumerate(subset):
        hits = [t for t in gen_titles[r["qid"]] if chunks.get(norm(t))]
        f, raw, trunc = facts_by_i.get(i, ([], "", False))
        ext_recs.append({"qid": r["qid"], "question_en": r["question_en"],
                         "generated_titles": gen_titles[r["qid"]],
                         "matched_titles": hits, "facts": f, "n_facts": len(f),
                         "raw_extraction": raw, "truncated": trunc})
    nf = sorted(e["n_facts"] for e in ext_recs)
    print("  facts/question: min=%d median=%d max=%d  zero-fact rows=%d"
          % (nf[0], nf[len(nf) // 2], nf[-1], sum(1 for x in nf if x == 0)))
    write_jsonl(os.path.join(OUT_DIR, "d6_extractions%s.jsonl" % suffix),
                ext_recs, "extractions")

    print("\n" + "=" * 70)
    print("STAGE 3b — ANSWERING (frozen Urdu prompt, identical to D4)")
    print("=" * 70, flush=True)
    p2 = []
    for r, e in zip(subset, ext_recs):
        if e["facts"]:
            user = "\n\n".join([instruction, facts_block(e["facts"]),
                                r["question_ur"]])
        else:
            user = "\n\n".join([instruction, r["question_ur"]])
        p2.append(chat(SYSTEM_MESSAGE, user))
    gen2 = generate_all(model, tok, p2, args.batch_size, eos_ids,
                        MAX_NEW_TOKENS)
    e1 = []
    for r, e, (text, n_new, trunc) in zip(subset, ext_recs, gen2):
        e1.append({"qid": r["qid"], "arm": "E1", "gold": r["gold"],
                   "pred": extract_answer(text),
                   "pred_secondary": extract_answer_secondary(text),
                   "truncated": trunc, "n_gen_tokens": n_new,
                   "n_facts_given": e["n_facts"],
                   "n_matched_titles": len(e["matched_titles"]),
                   "generation": text})
    write_jsonl(os.path.join(OUT_DIR, "d6_armE1%s.jsonl" % suffix), e1,
                "arm E1")

    # ================= REPORT ============================================
    print("\n" + "=" * 84)
    print("D6 RESULT%s   n=%d, floor %.2f%%"
          % ("  (TEST — NOT A RESULT)" if args.test else "",
             len(subset), floor))
    print("=" * 84)
    arms = OrderedDict([("A", armA), ("E1", e1), ("X1", armX1)])
    lab = {"A": "no facts (floor)", "E1": "SELF-LINKED end-to-end",
           "X1": "GOLD titles, oracle linking (ceiling)"}
    accs, unp = {}, []
    for a, recs in arms.items():
        s, s2 = score(recs, "pred"), score(recs, "pred_secondary")
        accs[a] = s["accuracy"]
        unp.append(s["unparsed_rate"])
        print("%-3s %-40s acc %6.2f%%  unparsed %5.2f%%  (secondary %6.2f%%)"
              % (a, lab[a], s["accuracy"], s["unparsed_rate"], s2["accuracy"]))
    spread = max(unp) - min(unp)
    print("\nunparsed spread across A/E1/X1: %.2f pp" % spread)
    void = spread >= 5.0
    if void:
        print("*** VALIDITY (freeze D): spread >= 5pp. The accuracy")
        print("*** comparison is VOID. Only descriptive statistics may be")
        print("*** quoted. No reading fires.")

    # ---- mandatory diagnostics (freeze section G) ----
    print("\n" + "-" * 84)
    print("MANDATORY DIAGNOSTICS (freeze G) — descriptive, never gates")
    print("-" * 84)
    print("G1 entities/question       min=%d median=%d max=%d"
          % (ne[0], ne[len(ne) // 2], ne[-1]))
    print("G2 generated titles matched in corpus  %d/%d = %.1f%%"
          % (len(matched), len(needed),
             100.0 * len(matched) / max(len(needed), 1)))
    zero_qids = {e["qid"] for e in ext_recs if e["n_facts"] == 0}
    print("G3 rows with ZERO facts    %d/%d = %.1f%%"
          % (len(zero_qids), len(subset),
             100.0 * len(zero_qids) / max(len(subset), 1)))
    if zero_qids:
        sub_e1 = [r for r in e1 if r["qid"] in zero_qids]
        sub_a = [r for r in armA if r["qid"] in zero_qids]
        print("   matched control on the SAME qids: E1 %.2f%% vs A %.2f%% "
              "(n=%d)" % (acc_of(sub_e1), acc_of(sub_a), len(sub_e1)))
    got_qids = [e["qid"] for e in ext_recs if e["n_facts"] > 0]
    if got_qids:
        s_e1 = [r for r in e1 if r["qid"] in set(got_qids)]
        s_a = [r for r in armA if r["qid"] in set(got_qids)]
        print("   rows WITH facts, same qids:       E1 %.2f%% vs A %.2f%% "
              "(n=%d)" % (acc_of(s_e1), acc_of(s_a), len(s_e1)))
    gold_by = {r["qid"]: {norm(t) for t in r["required"]} for r in subset}
    inter = sum(len({norm(t) for t in e["generated_titles"]}
                    & gold_by[e["qid"]]) for e in ext_recs)
    goldn = sum(len(gold_by[e["qid"]]) for e in ext_recs)
    print("G4 generated-vs-gold title overlap %d/%d = %.1f%%  "
          "(DESCRIPTIVE ONLY, freeze A: NOT a success criterion)"
          % (inter, goldn, 100.0 * inter / max(goldn, 1)))
    print("G5 facts/question          min=%d median=%d max=%d  empty=%d"
          % (nf[0], nf[len(nf) // 2], nf[-1], sum(1 for x in nf if x == 0)))
    print("\nG6 first 10 rows in file order:")
    for r, e, a in list(zip(subset, ext_recs, e1))[:10]:
        print("  ---- %s  gold=%s pred=%s" % (r["qid"], a["gold"], a["pred"]))
        print("     Q_ur : %s" % r["question_ur"][:110])
        print("     gen  : %s" % e["generated_titles"])
        print("     match: %s" % e["matched_titles"])
        for k, f in enumerate(e["facts"], 1):
            print("       %d. %s" % (k, f[:150]))

    if args.test:
        print("\nTEST COMPLETE. Not a result.")
        return
    if void:
        print("\nNo reading fires (validity condition failed).")
        return

    # ---- pre-declared readings (freeze section F) ----
    print("\n" + "-" * 84)
    print("PRE-DECLARED READINGS (freeze D6 section F)")
    print("-" * 84)
    print("PRIMARY:")
    b, c, p = mcnemar(e1, armA, "E1", "A")
    print("SECONDARY:")
    b2, c2, p2 = mcnemar(e1, armX1, "E1", "X1")
    d = accs["E1"] - accs["A"]
    half = (accs["X1"] - accs["A"]) / 2.0
    print("\n  d = E1 - A = %+.2f pp    half the oracle gap = %+.2f pp" % (d, half))
    if p is None:
        reading = "UNDETERMINED (scipy unavailable, p not computed)"
    elif d > 0 and p < 0.05:
        reading = ("READING 1 — WORKING METHOD. End-to-end Urdu multi-hop "
                   "works and is statistically supported.")
    elif d >= half:
        reading = ("READING 2 — WORKING BUT UNDERPOWERED. Recovers >= half "
                   "the oracle gap; report with the section E power table. "
                   "Do NOT claim significance. Do NOT re-run on another "
                   "subset.")
    elif d > 0:
        reading = ("READING 3 — PARTIAL, INSUFFICIENT. Helps but recovers "
                   "< half the gap. Name the bottleneck from the G "
                   "diagnostics, do not guess.")
    else:
        reading = ("READING 4 — FAIL. Self-generated titles do not deliver "
                   "usable knowledge. The retrieval-free pipeline is closed.")
    print("\n  " + reading)

    summ = {"n": len(subset), "floor": floor, "accuracy": accs,
            "unparsed_spread": spread, "void": void,
            "entities": {"parse_status": status_ct,
                         "unique_titles": len(needed),
                         "matched_titles": len(matched)},
            "rows_with_facts": len(got_qids),
            "gold_overlap": {"intersect": inter, "gold_total": goldn},
            "e1_vs_a": {"b": b, "c": c, "p": p},
            "e1_vs_x1": {"b": b2, "c": c2, "p": p2},
            "reading": reading}
    ps = os.path.join(OUT_DIR, "d6_summary%s.json" % suffix)
    with open(ps, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    print("\n[verified] wrote %s" % ps)


if __name__ == "__main__":
    main()
