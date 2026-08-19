#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l0_part_bc.py  —  L0 PARTS B and C.

Declared in docs/EFBPT_PLAN_A_FREEZE.md -> EXPERIMENT L0 (+ AMENDMENT 1).

Part B  free-generation BASELINE. D6's stage-0 prompt on the gold questions,
        base model, decoding imported from D6 so it cannot drift. A pair is
        CORRECT if any generated canonical_title matches the gold title
        under norm().
Part C  RETRIEVAL CEILING. One scan of the corpus for its unique titles, then
        every title encoded with paraphrase-multilingual-MiniLM-L12-v2 and
        scored against the Urdu spans. Reports recall@1/5/10/20/50/100.

The census (buckets, COREF list, thresholds) is IMPORTED from
l0_part_a_census.py, never re-implemented, so Parts B/C use exactly the
classification the human confirmed.

No FAISS index is built. Titles are streamed in batches and only a running
top-K per span is kept, so peak memory is a few hundred MB instead of the
~20 GB a flat 6.4M-vector index would need.

READ-ONLY on all repository data. Writes one JSON summary to outputs/efbpt/l0/.
"""
import argparse, json, os, re, sys, unicodedata
from collections import OrderedDict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from efbpt_eval_dev200 import MODEL_PATH, die                    # noqa: E402
from d3_oracle_retrieval import META_PATH, norm                  # noqa: E402
from d4_extract_facts import generate_all                        # noqa: E402
from d6_endtoend_selflink import (                               # noqa: E402
    ENTITY_SYSTEM, ENTITY_INSTRUCTION, ENTITY_MAX_NEW,
    parse_entities, dedupe_titles, md5,
)
from l0_part_a_census import GOLD, classify, translit            # noqa: E402

EMBED_PATH = ("/mnt/home/user41/downloaded_models/sentence-transformers/"
              "paraphrase-multilingual-MiniLM-L12-v2")
QUESTION_SOURCES = [
    "data/strategyqa_official/dev200_seed4242.jsonl",
    "data/strategyqa_official/efbpt/blind30_rows.jsonl",
]
OUT_DIR = "outputs/efbpt/l0"
KS = (1, 5, 10, 20, 50, 100)
TOPK = max(KS)
TITLE_RE = re.compile(rb'^\{"title":\s*"((?:[^"\\]|\\.)*)"')
EXPECTED_TITLES = 6402346          # freeze C: printed, never asserted
PICK_ACC = 0.85                    # freeze F: declared assumption
MARGIN_PP = 6.0                    # freeze F
ENC_BATCH = 20000                  # titles per encode/score batch


def load_pairs():
    """Gold pairs + census buckets, using the imported (human-confirmed) rules."""
    pairs, seen = [], set()
    for path, idf, entf in GOLD:
        if not os.path.exists(path):
            die("MISSING: " + path)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for e in row.get(entf, []):
                sp, ti = e.get("urdu_span", ""), e.get("canonical_title", "")
                key = (row[idf], sp)
                if not ti.strip() or key in seen:
                    continue
                seen.add(key)
                b, _ = classify(sp, ti)
                pairs.append({"qid": row[idf], "span": sp, "title": ti,
                              "bucket": b, "q_ur": row.get("question_ur", "")})
    return pairs


def resolve_questions(pairs):
    """blind30_gold carries no question_ur. Its questions live in
    blind30_rows.jsonl (verified 2026-08-18: DEV200 has zero id overlap with
    blind30, so both sources are searched)."""
    need = {p["qid"] for p in pairs if not p["q_ur"]}
    found = {}
    for path in QUESTION_SOURCES:
        if not need or not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("question_ur"):
                continue
            for k in ("urbench_qid", "official_qid", "urdu_qid_original"):
                if r.get(k) in need:
                    found[r[k]] = r["question_ur"]
    for p in pairs:
        if not p["q_ur"]:
            p["q_ur"] = found.get(p["qid"], "")
    return len(need), len(found)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def report_by_bucket(label, ok_fn, pairs):
    """freeze E: every number reported over ALL, LINKABLE and SEMANTIC."""
    out = OrderedDict()
    for name, sel in (("ALL", lambda p: True),
                      ("LINKABLE", lambda p: p["bucket"] != "COREF"),
                      ("SEMANTIC", lambda p: p["bucket"] == "SEMANTIC"),
                      ("TRANSLIT", lambda p: p["bucket"] == "TRANSLIT")):
        sub = [p for p in pairs if sel(p)]
        ok = sum(1 for p in sub if ok_fn(p))
        out[name] = {"n": len(sub), "correct": ok, "pct": round(pct(ok, len(sub)), 2)}
        print("    %-9s %4d/%-4d = %5.1f%%" % (name, ok, len(sub), pct(ok, len(sub))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="20 questions, 200k titles — pipeline check only")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_TEST" if args.test else ""

    print("=" * 78)
    print("L0 PARTS B + C")
    print("=" * 78)
    print("[freeze] stage-0 instruction md5 %s  (imported from D6)"
          % md5(ENTITY_INSTRUCTION))
    if md5(ENTITY_INSTRUCTION) != "4675bc6b29aaca764b72c84da246bb9a":
        die("stage-0 prompt has drifted from the D6 md5; Part B would not be "
            "the declared baseline")

    pairs = load_pairs()
    need, got = resolve_questions(pairs)
    ct = {b: sum(1 for p in pairs if p["bucket"] == b)
          for b in ("TRANSLIT", "SEMANTIC", "COREF")}
    link = ct["TRANSLIT"] + ct["SEMANTIC"]
    print("  pairs %d | TRANSLIT %d  SEMANTIC %d  COREF %d | LINKABLE %d"
          % (len(pairs), ct["TRANSLIT"], ct["SEMANTIC"], ct["COREF"], link))
    print("  question_ur missing inline for %d qids, resolved from DEV200: %d"
          % (need, got))
    if link != 287 and not args.test:
        print("  *** WARNING: LINKABLE is %d, census recorded 287. Investigate "
              "before trusting the gate." % link)

    b_ok = [p for p in pairs if p["q_ur"]]
    if len(b_ok) != len(pairs):
        print("  *** %d pairs have NO Urdu question and are excluded from Part B."
              % (len(pairs) - len(b_ok)))
        print("  *** The GATE will be computed over the Part-B-measurable set only.")
    questions = OrderedDict((p["qid"], p["q_ur"]) for p in b_ok)
    if args.test:
        keep = set(list(questions)[:20])
        questions = OrderedDict((k, v) for k, v in questions.items() if k in keep)
        b_ok = [p for p in b_ok if p["qid"] in keep]

    # ================= PART B — free-generation baseline =================
    print("\n" + "-" * 78)
    print("PART B — free-generation baseline (%d questions)" % len(questions))
    print("-" * 78, flush=True)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    eos = {i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|im_end|>")) if i and i >= 0}
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": ENTITY_SYSTEM},
         {"role": "user", "content": ENTITY_INSTRUCTION + "\n\nQuestion: " + q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for q in questions.values()]
    gen = generate_all(model, tok, prompts, args.batch_size, eos, ENTITY_MAX_NEW)

    gen_titles, status = {}, {"ok": 0, "json_error": 0, "empty": 0, "no_array": 0}
    for qid, (text, _n, _t) in zip(questions, gen):
        ps, st = parse_entities(text)
        status[st] = status.get(st, 0) + 1
        gen_titles[qid] = {norm(t) for t in dedupe_titles(ps)}
    print("  parse status: " + "  ".join("%s=%d" % kv for kv in status.items()))

    for p in pairs:
        p["b_correct"] = norm(p["title"]) in gen_titles.get(p["qid"], set())
    print("  BASELINE accuracy (freeze E, all three views):")
    b_stats = report_by_bucket("B", lambda p: p["b_correct"], b_ok)

    del model
    torch.cuda.empty_cache()

    # ================= PART C — retrieval ceiling ========================
    print("\n" + "-" * 78)
    print("PART C — retrieval ceiling over the corpus title universe")
    print("-" * 78, flush=True)
    titles, seen = [], set()
    limit = 200000 if args.test else None
    with open(META_PATH, "rb") as f:
        for raw in f:
            m = TITLE_RE.match(raw)
            if not m:
                continue
            t = m.group(1).decode("utf-8", "replace").replace('\\"', '"')
            n = norm(t)
            if n and n not in seen:
                seen.add(n)
                titles.append(t)
                if limit and len(titles) >= limit:
                    break
    print("  unique titles observed : %d" % len(titles))
    if not args.test:
        d = abs(len(titles) - EXPECTED_TITLES) / EXPECTED_TITLES
        print("  recorded in d2         : %d  (difference %.2f%%)"
              % (EXPECTED_TITLES, 100 * d))
        if d > 0.05:
            print("  *** DISCREPANCY > 5%% between observed and recorded counts.")

    gold_present = {p["qid"] + "|" + p["span"]: norm(p["title"]) in seen
                    for p in pairs}
    absent = sum(1 for v in gold_present.values() if not v)
    print("  gold titles ABSENT from the corpus universe: %d/%d = %.1f%%"
          % (absent, len(pairs), pct(absent, len(pairs))))
    print("  (recall for these is 0 by construction — a corpus failure, not a")
    print("   retriever failure; freeze section E)")

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(EMBED_PATH)
    spans = sorted({p["span"] for p in pairs})
    qv = emb.encode(spans, normalize_embeddings=True, convert_to_numpy=True,
                    batch_size=128).astype(np.float32)
    nq = len(spans)
    best_s = np.full((nq, TOPK), -1e9, dtype=np.float32)
    best_i = np.full((nq, TOPK), -1, dtype=np.int64)
    for off in range(0, len(titles), ENC_BATCH):
        chunk = titles[off:off + ENC_BATCH]
        tv = emb.encode(chunk, normalize_embeddings=True, convert_to_numpy=True,
                        batch_size=512).astype(np.float32)
        S = qv @ tv.T
        idx = np.broadcast_to(np.arange(off, off + len(chunk)), (nq, len(chunk)))
        cs = np.concatenate([best_s, S], axis=1)
        ci = np.concatenate([best_i, idx], axis=1)
        part = np.argpartition(-cs, TOPK - 1, axis=1)[:, :TOPK]
        best_s = np.take_along_axis(cs, part, 1)
        best_i = np.take_along_axis(ci, part, 1)
        if (off // ENC_BATCH) % 25 == 0:
            print("    %d/%d titles" % (min(off + ENC_BATCH, len(titles)),
                                        len(titles)), flush=True)
    order = np.argsort(-best_s, axis=1)
    best_i = np.take_along_axis(best_i, order, 1)
    span_rank = {s: [norm(titles[j]) for j in best_i[i] if j >= 0]
                 for i, s in enumerate(spans)}

    recalls = OrderedDict()
    for k in KS:
        for p in pairs:
            p["r%d" % k] = norm(p["title"]) in span_rank[p["span"]][:k]
        print("  recall@%-3d (freeze E, all three views):" % k)
        recalls[k] = report_by_bucket("R%d" % k, lambda p, k=k: p["r%d" % k], pairs)

    # ================= GATE (freeze F) ===================================
    print("\n" + "=" * 78)
    print("GATE (freeze section F)")
    print("=" * 78)
    gate_set = [p for p in b_ok if p["bucket"] != "COREF"]
    B = pct(sum(1 for p in gate_set if p["b_correct"]), len(gate_set))
    R10 = pct(sum(1 for p in gate_set if p["r10"]), len(gate_set))
    delivered = R10 * PICK_ACC
    print("  computed on the Part-B-measurable LINKABLE set, n=%d" % len(gate_set))
    print("  B    = free-generation baseline      = %.2f%%" % B)
    print("  R10  = recall@10                     = %.2f%%" % R10)
    print("  R10 x %.2f (declared pick-accuracy)  = %.2f%%" % (PICK_ACC, delivered))
    print("  gate needs R10 x %.2f >= B + %.0fpp  = %.2f%%"
          % (PICK_ACC, MARGIN_PP, B + MARGIN_PP))
    if args.test:
        print("\n  TEST RUN — gate NOT applied. Not a result.")
        verdict = "TEST"
    elif delivered >= B + MARGIN_PP:
        verdict = ("GATE PASS — BUILD L1. The ceiling clears the baseline by "
                   "enough that a detectable improvement is possible.")
    elif delivered < B:
        verdict = ("GATE FAIL — DO NOT BUILD. The ceiling is at or below the "
                   "baseline. Multilingual sentence embeddings do not retrieve "
                   "entity titles from Urdu spans well enough to support "
                   "constrained linking. Report as a finding.")
    else:
        verdict = ("MARGINAL — DECIDE JOINTLY. Report the full recall curve; "
                   "no unilateral build.")
    print("\n  " + verdict)

    summ = {"pairs": len(pairs), "buckets": ct, "linkable": link,
            "part_b_measurable": len(b_ok), "gate_n": len(gate_set),
            "unique_titles": len(titles), "gold_absent_from_corpus": absent,
            "baseline": b_stats, "recall": recalls,
            "B": round(B, 2), "R10": round(R10, 2),
            "delivered": round(delivered, 2), "verdict": verdict}
    path = os.path.join(OUT_DIR, "l0_summary%s.json" % suffix)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    print("\n[verified] wrote %s" % path)


if __name__ == "__main__":
    main()
