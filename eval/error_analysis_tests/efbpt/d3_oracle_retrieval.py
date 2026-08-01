#!/usr/bin/env python3
"""
d3_oracle_retrieval.py

DIAGNOSTIC D3: can the model use RAW WIKIPEDIA PARAGRAPHS when they are the
CORRECT ones?

Restricted to the DEV200 questions where every required gold evidence title
exists in the corpus. Retrieval is bypassed: pages are fetched by EXACT TITLE
from the metadata file. This simulates perfect entity linking and perfect
retrieval, i.e. the ceiling of any entity-grounded method.

  O1  1 chunk per gold title  (article lead)
  O2  3 chunks per gold title
  A   no facts        REUSED from D1 arm A, same qids, not re-run
  B   clean gold facts REUSED from D1 arm B, same qids, not re-run

Prompt assembly and the facts block are IMPORTED from d1_eval_arms.py so the
O arms are structurally identical to D1 arms B/C. The answer extractor is
IMPORTED from efbpt_eval_dev200.py (AMENDMENT 5) and the Devanagari secondary
from d1_score_dual.py (D1 AMENDMENT 2).

Usage (GPU node, from ~/URBench):
  python eval/error_analysis_tests/efbpt/d3_oracle_retrieval.py --test
  python eval/error_analysis_tests/efbpt/d3_oracle_retrieval.py
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
from d1_eval_arms import facts_block      # noqa: E402  identical block format
from d1_score_dual import extract_answer_secondary   # noqa: E402

META_PATH = "/mnt/home/user41/URBench/rag/index/wikipedia_full_meta.jsonl"
COV_PATH = "outputs/efbpt/d2/d2_title_coverage.json"
D1_ARM_DIR = "outputs/efbpt/d1/arms"
OUT_DIR = "outputs/efbpt/d3"

ARMS = OrderedDict([("O1", 1), ("O2", 3)])   # arm -> chunks per title

TITLE_RE = re.compile(rb'^\{"title":\s*"((?:[^"\\]|\\.)*)"')


def norm(t):
    return " ".join(str(t).replace("_", " ").strip().lower().split())


def required_titles(evidence_ids):
    out = []
    for pid in evidence_ids or []:
        if isinstance(pid, str) and "-" in pid:
            out.append(pid.rsplit("-", 1)[0])
    return sorted(set(out))


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
                "gold": "yes" if r["answer"] else "no",
                "required": required_titles(r.get("evidence_paragraph_ids")),
            })
    if len(rows) != 200:
        die("DEV200 has %d rows, expected 200" % len(rows))
    return rows


def fetch_chunks(needed_norm, max_per_title):
    """One pass over the metadata file. Returns {norm_title: [chunk_text,...]}
    preserving file order, so index 0 is the article lead."""
    got = {t: [] for t in needed_norm}
    remaining = set(needed_norm)
    n_lines = 0
    t0 = time.time()
    with open(META_PATH, "rb") as f:
        for raw in f:
            n_lines += 1
            m = TITLE_RE.match(raw)
            if not m:
                continue
            t = m.group(1).decode("utf-8", "replace")
            t = t.replace('\\"', '"').replace("\\\\", "\\")
            nt = norm(t)
            if nt in remaining:
                rec = json.loads(raw.decode("utf-8"))
                got[nt].append(rec["text"])
                if len(got[nt]) >= max_per_title:
                    remaining.discard(nt)
                    if not remaining:
                        break
            if n_lines % 4_000_000 == 0:
                print("[scan]   %d lines, %d titles still short (%.0fs)"
                      % (n_lines, len(remaining), time.time() - t0), flush=True)
    print("[scan] read %d lines in %.0fs" % (n_lines, time.time() - t0))
    return got


def score(records, key):
    n = len(records)
    correct = sum(1 for r in records if r[key] == r["gold"])
    unparsed = sum(1 for r in records if r[key] is None)
    pred_yes = sum(1 for r in records if r[key] == "yes")
    floor_no = sum(1 for r in records if r["gold"] == "no")
    return OrderedDict([
        ("n", n),
        ("accuracy", round(100.0 * correct / n, 2)),
        ("floor_always_no", round(100.0 * floor_no / n, 2)),
        ("unparsed_rate", round(100.0 * unparsed / n, 2)),
        ("predicted_yes_rate", round(100.0 * pred_yes / n, 2)),
        ("correct", correct),
    ])


def generate_all(model, tok, prompts, batch_size, eos_ids):
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
            g = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            new_ids = g[j][in_len:]
            keep = [t for t in new_ids.tolist() if t != tok.pad_token_id]
            trunc = (len(keep) >= MAX_NEW_TOKENS) and (
                len(new_ids) > 0 and new_ids[-1].item() not in eos_ids)
            out[i] = (tok.decode(new_ids, skip_special_tokens=True),
                      len(keep), trunc)
        done += len(idxs)
        el = time.time() - t0
        print("    %d/%d  (%.1fs, %.1fs/row)" % (done, n, el, el / done), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="first 8 questions only")
    ap.add_argument("--test-rows", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    instruction = load_instruction()
    print("[ok] instruction verified")

    cov = json.load(open(COV_PATH, "r", encoding="utf-8"))
    present = {norm(t) for t in cov["present_titles"]}
    absent = {norm(t) for t in cov["absent_titles"]}
    print("[coverage] present=%d absent=%d" % (len(present), len(absent)))

    rows = load_rows()

    # ---- the 71: every required title present in the corpus ----
    subset = []
    for r in rows:
        if not r["required"]:
            continue
        for t in r["required"]:
            if norm(t) not in present and norm(t) not in absent:
                die("title %r classified by neither list" % t)
        if all(norm(t) in present for t in r["required"]):
            subset.append(r)
    print("[subset] questions with ALL gold titles present: %d / %d"
          % (len(subset), len(rows)))
    if len(subset) != 71:
        print("[subset] WARNING: expected 71 per the D2 split, got %d"
              % len(subset))

    if args.test:
        subset = subset[:args.test_rows]
        print("[subset] TEST: using first %d" % len(subset))

    n_yes = sum(1 for r in subset if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(subset) - n_yes) / len(subset)
    print("[subset] yes=%d no=%d  majority floor %.2f%%"
          % (n_yes, len(subset) - n_yes, floor))

    # ---- fetch chunks by exact title ----
    needed = sorted({norm(t) for r in subset for t in r["required"]})
    maxk = max(ARMS.values())
    print("\n[scan] fetching up to %d chunks for each of %d titles from %s"
          % (maxk, len(needed), META_PATH), flush=True)
    chunks = fetch_chunks(needed, maxk)

    missing = [t for t in needed if not chunks[t]]
    if missing:
        die("%d title(s) marked present in the corpus returned NO chunk, "
            "first 5: %s. The coverage file and the metadata disagree."
            % (len(missing), missing[:5]))
    short = [t for t in needed if len(chunks[t]) < maxk]
    print("[scan] all %d titles found. %d have fewer than %d chunks "
          "(short articles; that is fine)." % (len(needed), len(short), maxk))
    lens = sorted(len(chunks[t]) for t in needed)
    print("[scan] chunks per title: min=%d median=%d max=%d"
          % (lens[0], lens[len(lens) // 2], lens[-1]))

    # ---- tokenizer ----
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

    def build_prompt(row, k):
        facts = []
        for t in row["required"]:
            for c in chunks[norm(t)][:k]:
                facts.append("%s: %s" % (t, c))
        user = "\n\n".join([instruction, facts_block(facts), row["question_ur"]])
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    # ---- length check BEFORE loading the model ----
    for arm, k in ARMS.items():
        toks = [len(tok(build_prompt(r, k), add_special_tokens=False)["input_ids"])
                for r in subset]
        toks.sort()
        print("[len] %s prompt tokens: min=%d median=%d max=%d"
              % (arm, toks[0], toks[len(toks) // 2], toks[-1]))
        if toks[-1] + MAX_NEW_TOKENS > 40960:
            die("%s prompts would exceed the model context window" % arm)

    # ---- model ----
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("\n[load] base model, 4-bit nf4, no adapters ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_TEST" if args.test else ""
    results = OrderedDict()

    for arm, k in ARMS.items():
        print("\n" + "=" * 70)
        print("ARM %s — %d chunk(s) per gold title" % (arm, k))
        print("=" * 70, flush=True)
        prompts = [build_prompt(r, k) for r in subset]
        print("  prompt example 0 (repr, first 600 chars):")
        print("  " + repr(prompts[0][:600]), flush=True)

        gens = generate_all(model, tok, prompts, args.batch_size, eos_ids)

        recs = []
        for r, (text, n_new, trunc) in zip(subset, gens):
            recs.append({
                "qid": r["qid"], "arm": arm, "gold": r["gold"],
                "pred": extract_answer(text),
                "pred_secondary": extract_answer_secondary(text),
                "truncated": trunc, "n_gen_tokens": n_new,
                "titles": r["required"], "generation": text,
            })
        results[arm] = recs

        p = os.path.join(OUT_DIR, "d3_arm%s%s.jsonl" % (arm, suffix))
        with open(p, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        nd = sum(1 for _ in open(p, "r", encoding="utf-8"))
        if nd != len(recs):
            die("wrote %d but disk has %d lines: %s" % (len(recs), nd, p))
        s = score(recs, "pred")
        print("  [verified] %d lines on disk" % nd)
        print("  accuracy %.2f%%  unparsed %.2f%%" % (s["accuracy"], s["unparsed_rate"]))
        print("  sample (row 0, first 300 chars): %s"
              % repr(recs[0]["generation"][:300]), flush=True)

    # ---- reuse D1 arms A and B on the SAME qids ----
    qids = {r["qid"] for r in subset}
    for arm in ("A", "B"):
        p = os.path.join(D1_ARM_DIR, "d1_arm%s.jsonl" % arm)
        if not os.path.exists(p):
            die("missing D1 output: " + p)
        recs = []
        seen = set()
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["qid"] in qids:
                    r["pred_secondary"] = extract_answer_secondary(r["generation"])
                    recs.append(r)
                    seen.add(r["qid"])
        if seen != qids:
            die("D1 arm %s covers %d of the %d subset qids"
                % (arm, len(seen), len(qids)))
        results[arm] = recs
        print("[reuse] D1 arm %s: %d matching rows" % (arm, len(recs)))

    # ---- report ----
    print("\n" + "=" * 84)
    print("D3 RESULT%s   n=%d questions, majority floor %.2f%%"
          % ("  (TEST — NOT A RESULT)" if args.test else "", len(subset), floor))
    print("=" * 84)
    print("%-4s %-28s %9s %9s %11s"
          % ("arm", "facts", "acc%", "unpars%", "SEC acc%"))
    label = {"O1": "1 wiki chunk per title", "O2": "3 wiki chunks per title",
             "A": "none (D1, reused)", "B": "clean gold facts (D1, reused)"}
    acc = {}
    unp = []
    for arm in ("A", "O1", "O2", "B"):
        pri = score(results[arm], "pred")
        sec = score(results[arm], "pred_secondary")
        acc[arm] = pri["accuracy"]
        unp.append(pri["unparsed_rate"])
        print("%-4s %-28s %9.2f %9.2f %11.2f"
              % (arm, label[arm], pri["accuracy"], pri["unparsed_rate"],
                 sec["accuracy"]))

    spread = max(unp) - min(unp)
    print("\nunparsed spread across arms: %.2f pp" % spread)
    if spread > 5.0:
        print("*** AMENDMENT 5D: spread > 5pp, the accuracy comparison is VOID.")

    if args.test:
        print("\nTEST COMPLETE. Not a result.")
        return

    print("\n" + "-" * 84)
    print("CONTRASTS")
    print("-" * 84)
    print("  B - A    value of clean gold facts on this subset = %+.2f pp"
          % (acc["B"] - acc["A"]))
    print("  O1 - A   value of 1 correct wiki chunk per title  = %+.2f pp"
          % (acc["O1"] - acc["A"]))
    print("  O2 - O1  effect of more correct context           = %+.2f pp"
          % (acc["O2"] - acc["O1"]))
    print("  B - O2   how far paragraphs fall short of facts   = %+.2f pp"
          % (acc["B"] - acc["O2"]))
    gain = max(acc["O1"], acc["O2"]) - acc["A"]
    total = acc["B"] - acc["A"]
    if total > 0:
        print("\n  fraction of the gold-facts gain recovered by oracle pages: %.0f%%"
              % (100.0 * gain / total))

    print("\n" + "-" * 84)
    print("PRE-DECLARED READING (D3 Section D)")
    print("-" * 84)
    best = max(acc["O1"], acc["O2"])
    if abs(best - acc["B"]) <= 5.0:
        print("  Reading 1: correct Wikipedia paragraphs work nearly as well as")
        print("  clean gold facts. The bottleneck is entity linking and corpus")
        print("  coverage. Retrieval is a viable method; the pathway is confirmed.")
    elif abs(best - acc["A"]) <= 5.0:
        print("  Reading 2: the model cannot use raw paragraphs even when they are")
        print("  the correct ones. Fetching PAGES is a dead end regardless of")
        print("  coverage or linking. The method must EXTRACT facts from pages.")
    else:
        print("  Reading 3: paragraphs partially work. Treat fact extraction as an")
        print("  improvement to a working pipeline, not a replacement for it.")

    summ = {"n": len(subset), "floor": floor, "accuracy": acc,
            "unparsed_spread": spread}
    p = os.path.join(OUT_DIR, "d3_summary%s.json" % suffix)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("\n[verified] wrote %s" % p)


if __name__ == "__main__":
    main()
