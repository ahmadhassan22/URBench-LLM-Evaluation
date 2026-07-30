#!/usr/bin/env python3
"""
efbpt_counterfactual_entities.py

Does C3's entity block actually drive the answer, or is it decorative?

Three arms on the same 200 DEV200 questions, per seed:

  A  FREE      - C3 generates everything itself. Already measured by
                 efbpt_eval_dev200.py; not re-run here.
  B  SELF      - force-decode C3's OWN previously generated entity block,
                 then let it generate steps + answer. Control: forcing by
                 itself should change nothing.
  C  CORRUPT   - force-decode the SAME urdu_spans but with canonical_title
                 values taken from a DIFFERENT question. Then generate.

Only the mention -> identity mapping differs between B and C. The Urdu spans
are untouched, so span extraction (measured at 97.9% correct) is held fixed
and only entity identity is corrupted.

Reading the result:
  C much worse than B  -> the entity block causally drives the answer.
                          Binding quality is the bottleneck.
  C about equal to B   -> the block is inert; EFBPT's premise is refuted.

The answer extractor is IMPORTED from efbpt_eval_dev200.py so it cannot drift
from the frozen AMENDMENT 5 version.

Usage (inside a SLURM job, from ~/URBench):
  python eval/error_analysis_tests/efbpt/efbpt_counterfactual_entities.py --test
  python eval/error_analysis_tests/efbpt/efbpt_counterfactual_entities.py
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

# Frozen pieces reused verbatim. Importing rather than copying guarantees the
# extractor and prompt are identical to the main evaluation.
from efbpt_eval_dev200 import (          # noqa: E402
    MODEL_PATH,
    ADAPTER_ROOT,
    SYSTEM_MESSAGE,
    SEEDS,
    MAX_NEW_TOKENS,
    extract_answer,
    load_instruction,
    load_dev,
    die,
)

DEV200_RAW_DIR = "outputs/efbpt/plan_a/dev200"
OUT_DIR = "outputs/efbpt/plan_a/counterfactual"

# How far to shift when stealing another question's titles. A fixed offset
# keeps the experiment deterministic and reproducible.
CORRUPT_SHIFT = 97


def parse_entities(generation):
    """Return the entities list from a C3 generation, or None if unusable."""
    try:
        obj = json.loads(generation)
    except Exception:
        return None
    ents = obj.get("entities")
    if not isinstance(ents, list) or len(ents) == 0:
        return None
    out = []
    for e in ents:
        if not isinstance(e, dict):
            return None
        t = e.get("canonical_title")
        s = e.get("urdu_span")
        if not isinstance(t, str) or not isinstance(s, str):
            return None
        out.append({"canonical_title": t, "urdu_span": s})
    return out


def entities_prefix(entities):
    """Force-decode prefix: the entity block plus the opening of the next key.

    Matches the frozen serialization exactly (compact separators, key order
    entities -> steps -> answer), so the forced text is in-distribution.
    """
    body = json.dumps(entities, ensure_ascii=False, separators=(",", ":"))
    return '{"entities":' + body + ','


def corrupt(entities, donor_entities):
    """Keep this row's Urdu spans; replace identities with the donor's."""
    donor_titles = [e["canonical_title"] for e in donor_entities]
    out = []
    for i, e in enumerate(entities):
        out.append({
            "canonical_title": donor_titles[i % len(donor_titles)],
            "urdu_span": e["urdu_span"],
        })
    return out


def score(records):
    n = len(records)
    if n == 0:
        return OrderedDict([("n", 0)])
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
        batch = [prompts[i] for i in idxs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
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
            truncated = (n_new >= MAX_NEW_TOKENS) and (
                len(new_ids) > 0 and new_ids[-1].item() not in eos_ids)
            results[i] = (text, n_new, truncated)
        done += len(idxs)
        el = time.time() - t0
        print("    %d/%d  (%.1fs, %.1fs/row)" % (done, n, el, el / done), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="20 rows, seed 13 only. Never banked.")
    ap.add_argument("--test-rows", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    instruction = load_instruction()
    print("[ok] instruction verified")

    rows = load_dev(limit=args.test_rows if args.test else None)
    by_qid = {r["qid"]: r for r in rows}
    print("[data] %d rows" % len(rows))

    seeds = [13] if args.test else SEEDS

    # ---- load previous C3 generations (arm A) ----
    prev = {}
    for seed in seeds:
        path = os.path.join(DEV200_RAW_DIR, "dev200_C3_seed%d.jsonl" % seed)
        if not os.path.exists(path):
            die("missing arm-A output: " + path)
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    d[rec["qid"]] = rec
        prev[seed] = d
        print("[load] arm A, seed %d: %d rows" % (seed, len(d)))

    # ---- tokenizer ----
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

    def base_prompt(row):
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": instruction + "\n\n" + row["question_ur"]},
        ]
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    # ---- model ----
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("[load] base model, 4-bit nf4 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    peft_model = None
    for i, seed in enumerate(seeds):
        tag = "C3_seed%d" % seed
        path = os.path.join(ADAPTER_ROOT, tag)
        if not os.path.isdir(path):
            die("adapter missing: " + path)
        if i == 0:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=tag)
        else:
            peft_model.load_adapter(path, adapter_name=tag)
        print("[load] adapter attached: %s" % tag)
    peft_model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []

    for seed in seeds:
        tag = "C3_seed%d" % seed
        peft_model.set_adapter(tag)
        if peft_model.active_adapter != tag:
            die("adapter switch failed for " + tag)

        # ---- build the usable subset: rows whose arm-A entity block parses ----
        usable = []
        for r in rows:
            rec = prev[seed].get(r["qid"])
            if rec is None:
                continue
            ents = parse_entities(rec["generation"])
            if ents is None:
                continue
            usable.append((r, ents))
        print("\n[seed %d] usable rows: %d / %d" % (seed, len(usable), len(rows)))

        if len(usable) < 2:
            die("too few usable rows for seed %d" % seed)

        # donor assignment: fixed shift, deterministic
        n_use = len(usable)
        donors = [usable[(i + CORRUPT_SHIFT) % n_use][1] for i in range(n_use)]

        n_changed = 0
        for i in range(n_use):
            own = [e["canonical_title"] for e in usable[i][1]]
            new = [d["canonical_title"] for d in donors[i]]
            if own != new[:len(own)]:
                n_changed += 1
        print("[seed %d] rows whose titles actually change under corruption: "
              "%d / %d" % (seed, n_changed, n_use))

        for arm in ["B_self", "C_corrupt"]:
            print("\n" + "=" * 70)
            print("[seed %d] ARM %s" % (seed, arm))
            print("=" * 70, flush=True)

            prompts, metas = [], []
            for i, (r, ents) in enumerate(usable):
                use_ents = ents if arm == "B_self" else corrupt(ents, donors[i])
                pfx = entities_prefix(use_ents)
                prompts.append(base_prompt(r) + pfx)
                metas.append((r, pfx, use_ents))

            print("  forced prefix example (first 200 chars):")
            print("  " + repr(metas[0][1][:200]), flush=True)

            gens = generate_all(peft_model, tok, prompts, args.batch_size, eos_ids)

            records = []
            for (r, pfx, use_ents), (cont, n_new, trunc) in zip(metas, gens):
                full = pfx + cont          # extractor sees the whole object
                records.append({
                    "qid": r["qid"],
                    "gold": r["gold"],
                    "pred": extract_answer(full),
                    "truncated": trunc,
                    "n_gen_tokens": n_new,
                    "forced_titles": [e["canonical_title"] for e in use_ents],
                    "generation": full,
                })

            s = score(records)
            s["seed"] = seed
            s["arm"] = arm
            summary.append(s)

            suffix = "_TEST" if args.test else ""
            out_path = os.path.join(
                OUT_DIR, "cf_%s_seed%d%s.jsonl" % (arm, seed, suffix))
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            n_disk = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
            if n_disk != len(records):
                die("wrote %d but disk has %d lines: %s"
                    % (len(records), n_disk, out_path))
            print("  [verified] %d lines on disk" % n_disk)
            print("  accuracy %.2f%%  (floor %.2f%%)  unparsed %.2f%%  trunc %.2f%%"
                  % (s["accuracy"], s["floor_always_no"],
                     s["unparsed_rate"], s["truncation_rate"]))
            print("  raw -> %s" % out_path, flush=True)

    # ---- summary + paired B vs C ----
    print("\n" + "=" * 78)
    print("SUMMARY" + ("  (TEST — NOT A RESULT)" if args.test else ""))
    print("=" * 78)
    print("%-16s %6s %9s %9s %9s %9s" %
          ("arm", "n", "acc%", "floor%", "unpars%", "trunc%"))
    for s in summary:
        print("%-16s %6d %9.2f %9.2f %9.2f %9.2f"
              % ("seed%d %s" % (s["seed"], s["arm"]), s["n"], s["accuracy"],
                 s["floor_always_no"], s["unparsed_rate"], s["truncation_rate"]))

    sum_path = os.path.join(OUT_DIR, "cf_summary%s.json"
                            % ("_TEST" if args.test else ""))
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nsummary -> %s" % sum_path)

    print("\n" + "=" * 78)
    print("PAIRED B vs C (same qid, same seed)")
    print("=" * 78)
    tot = OrderedDict([("both_correct", 0), ("B_only", 0),
                       ("C_only", 0), ("both_wrong", 0)])
    for seed in seeds:
        suffix = "_TEST" if args.test else ""
        b = {}
        c = {}
        for arm, d in (("B_self", b), ("C_corrupt", c)):
            p = os.path.join(OUT_DIR, "cf_%s_seed%d%s.jsonl" % (arm, seed, suffix))
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        d[rec["qid"]] = rec
        cnt = OrderedDict([("both_correct", 0), ("B_only", 0),
                           ("C_only", 0), ("both_wrong", 0)])
        for q in b:
            if q not in c:
                continue
            ok_b = b[q]["pred"] == b[q]["gold"]
            ok_c = c[q]["pred"] == c[q]["gold"]
            key = ("both_correct" if ok_b and ok_c else
                   "B_only" if ok_b else
                   "C_only" if ok_c else "both_wrong")
            cnt[key] += 1
            tot[key] += 1
        print("seed %-5d %s" % (seed, dict(cnt)))
    print("POOLED    %s" % dict(tot))
    print("\nDiscordant pairs: B_only=%d  C_only=%d" % (tot["B_only"], tot["C_only"]))
    print("If B_only is much larger than C_only, corrupting the entity identity")
    print("damaged the answer, so the entity block is causally used.")

    if args.test:
        print("\nTEST COMPLETE. Nothing here is a result.")


if __name__ == "__main__":
    main()
