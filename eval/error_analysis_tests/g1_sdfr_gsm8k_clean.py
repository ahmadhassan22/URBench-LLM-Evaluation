#!/usr/bin/env python3
"""
g1_sdfr_gsm8k_clean.py

EXPERIMENT G1 — SDFR-UR on GSM8K with a DECONTAMINATED retrieval pool.
Declared in docs/EFBPT_PLAN_A_FREEZE.md -> EXPERIMENT G1 before execution.

Identical to sdfr_gsm8k_fair.py in every respect EXCEPT pool construction:
same eval items and order, same embedder, same FAISS index, same TOP_K,
same prompt template, same extractor, same decoding.

The FAISS index on disk holds all 7473 pool vectors. Rows are therefore NOT
deleted from the pool list — that would desynchronise pool[i] from the index
ids and silently corrupt every retrieval. Instead the index is left untouched,
K_SEARCH neighbours are retrieved, contaminated and near-duplicate ones are
dropped by pool position, and the first TOP_K survivors are used. Retrieval
geometry is thus byte-identical to the contaminated run.

Arm B0 is reused from disk and never re-run (freeze section C).
"""
import argparse, difflib, json, os, re, sys, unicodedata
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

BASE        = "/mnt/home/user41/URBench/data"
SPLITS      = f"{BASE}/sdfr_splits"
INDEXES     = f"{BASE}/sdfr_indexes"
EN_SOURCE   = f"{BASE}/gsm8k_raw/gsm8k_main_train_700.jsonl"
MODEL_PATH  = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EMBED_PATH  = ("/mnt/home/user41/downloaded_models/sentence-transformers/"
               "paraphrase-multilingual-MiniLM-L12-v2")
B0_FILE     = ("/mnt/home/user41/URBench/outputs/sdfr/"
               "cot_gsm8k_baseline_fair_qwen3_14b.jsonl")
OUTPUT_FILE = ("/mnt/home/user41/URBench/outputs/sdfr/"
               "g1_sdfr_gsm8k_clean_qwen3_14b.jsonl")

TOP_K        = 3        # unchanged from sdfr_gsm8k_fair.py
K_SEARCH     = 50       # over-retrieve so TOP_K survivors always exist
SIM_THRESH   = 0.90     # freeze C.3 near-duplicate guard
POOL_EXPECT  = 7473
REMOVE_EXPECT = 700
POOL_AFTER_EXPECT = 6773   # freeze C.2


def die(msg):
    sys.exit("FATAL: " + msg)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_gold(answer_str):
    m = re.search(r"####\s*(-?[\d,]+)", answer_str)
    if m: return m.group(1).replace(",", "").strip()
    return answer_str.strip()


def extract_answer(text):
    # parse after the Urdu final-answer marker; fallback to last number
    m = re.search(r"حتمی\s*جواب\s*[:：]\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "").strip()
    m2 = re.search(r"####\s*(-?[\d,]+)", text)
    if m2: return m2.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def format_example(item):
    return f"مسئلہ: {item['question']}\nحتمی جواب: {extract_gold(item['answer'])}"


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())


def build_prompt(tok, item, few_shot):
    instr = ('آپ کو ایک ریاضی کا لفظی مسئلہ دیا گیا ہے۔\n'
             'پہلے مرحلہ وار سوچیں اور مسئلہ حل کریں۔\n'
             'اس کے بعد "حتمی جواب:" کے بعد صرف ایک عددی جواب لکھیں۔\n'
             'حتمی جواب میں کوئی یونٹ، الفاظ یا اضافی متن شامل نہ کریں۔')
    q_block = f"مسئلہ: {item['question']}\n\nسوچنے کے مراحل:\nحتمی جواب:"
    raw = f"{instr}\n\n{few_shot}\n\n{q_block}"
    return tok.apply_chat_template([{"role": "user", "content": raw}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)


def mcnemar(b, c):
    from scipy.stats import binomtest
    return binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="first 50 items only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    suffix = "_TEST" if args.test else ""
    out_path = OUTPUT_FILE.replace(".jsonl", suffix + ".jsonl")
    if not args.test and os.path.exists(out_path) and not args.overwrite:
        die("refusing to overwrite %s (pass --overwrite only if you mean it)"
            % out_path)

    # ---------- load ----------
    pool = read_jsonl(f"{SPLITS}/gsm8k_pool.jsonl")
    eval_data = read_jsonl(f"{SPLITS}/gsm8k_eval.jsonl")
    en_src = read_jsonl(EN_SOURCE)
    if len(pool) != POOL_EXPECT:
        die("pool is %d rows, expected %d" % (len(pool), POOL_EXPECT))
    if not (len(eval_data) == len(en_src) == 700):
        die("eval=%d en_source=%d, both must be 700"
            % (len(eval_data), len(en_src)))

    # ---------- freeze C.1/C.2 : decontamination ----------
    # The eval `question` field is Urdu, so the freeze's exact (question,
    # answer) match is performed against the positionally-aligned ENGLISH
    # source file, whose answers match the eval answers 700/700.
    bad = [i for i, (e, s) in enumerate(zip(eval_data, en_src))
           if e["answer"] != s["answer"]]
    if bad:
        die("english source not positionally aligned at %d rows, first: %s"
            % (len(bad), bad[:5]))

    src_pairs = {(s["question"], s["answer"]) for s in en_src}
    removed = {i for i, p in enumerate(pool)
               if (p["question"], p["answer"]) in src_pairs}
    # independent cross-check: answer-string alone identifies the same rows
    src_answers = {s["answer"] for s in en_src}
    removed_by_answer = {i for i, p in enumerate(pool)
                         if p["answer"] in src_answers}
    if removed != removed_by_answer:
        die("pair-match and answer-match disagree (%d vs %d rows)"
            % (len(removed), len(removed_by_answer)))
    if len(removed) != REMOVE_EXPECT:
        die("removed %d pool records, freeze requires exactly %d"
            % (len(removed), REMOVE_EXPECT))
    pool_after = len(pool) - len(removed)
    if pool_after != POOL_AFTER_EXPECT:
        die("pool after removal is %d, freeze requires %d"
            % (pool_after, POOL_AFTER_EXPECT))

    print("=" * 78)
    print("G1 — DECONTAMINATION (freeze section C)")
    print("=" * 78)
    print("  pool before removal      : %d" % len(pool))
    print("  records removed          : %d" % len(removed))
    print("  pool after removal       : %d  (freeze requires %d)"
          % (pool_after, POOL_AFTER_EXPECT))
    print("  cross-check (answer-only): %d rows, AGREES" % len(removed_by_answer))
    print("  eval items whose own source is still reachable: 0  (all %d removed)"
          % len(removed), flush=True)

    if args.test:
        eval_data, en_src = eval_data[:50], en_src[:50]

    # ---------- retrieve ----------
    index = faiss.read_index(f"{INDEXES}/gsm8k_faiss.index")
    if index.ntotal != POOL_EXPECT:
        die("index ntotal %d != pool %d; pool[i] would desynchronise"
            % (index.ntotal, POOL_EXPECT))
    embedder = SentenceTransformer(EMBED_PATH)
    qvecs = embedder.encode([e["question"] for e in eval_data],
                            normalize_embeddings=True, convert_to_numpy=True,
                            batch_size=64)
    _, cand = index.search(qvecs, K_SEARCH)

    pool_norm = [norm(p["question"]) for p in pool]
    n_contam = n_nearsim = 0
    rows_contam = rows_nearsim = 0
    max_sims, mean_sims, neighbours = [], [], []
    for row, src in zip(cand, en_src):
        target = norm(src["question"])
        keep, sims, hit_c, hit_s = [], [], False, False
        for i in row:
            if i in removed:
                n_contam += 1; hit_c = True; continue
            r = difflib.SequenceMatcher(None, pool_norm[i], target).ratio()
            if r >= SIM_THRESH:
                n_nearsim += 1; hit_s = True; continue
            keep.append(i); sims.append(r)
            if len(keep) == TOP_K:
                break
        if len(keep) != TOP_K:
            die("only %d clean neighbours within K_SEARCH=%d; raise K_SEARCH"
                % (len(keep), K_SEARCH))
        rows_contam += hit_c; rows_nearsim += hit_s
        max_sims.append(max(sims)); mean_sims.append(sum(sims) / len(sims))
        neighbours.append(keep)

    print("\n  contaminated neighbours dropped : %d  (affecting %d/%d items)"
          % (n_contam, rows_contam, len(eval_data)))
    print("  near-duplicate neighbours dropped: %d  (affecting %d/%d items)"
          % (n_nearsim, rows_nearsim, len(eval_data)))
    print("  post-filter similarity to English source: max %.3f, mean %.3f"
          % (max(max_sims), sum(mean_sims) / len(mean_sims)), flush=True)

    # ---------- generate ----------
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])
    prompts = [build_prompt(tok, e,
                            "\n\n".join(format_example(pool[i]) for i in ns))
               for e, ns in zip(eval_data, neighbours)]
    outputs = llm.generate(prompts, sp)

    results = []
    for i, (item, out) in enumerate(zip(eval_data, outputs)):
        gen = out.outputs[0].text.strip()
        pred = extract_answer(gen.split("</think>")[-1])
        gold = extract_gold(item["answer"])
        results.append({"qid": item.get("qid", f"GSM8K_{i:04d}"),
                        "question": item["question"], "gold": gold,
                        "pred": pred, "correct": pred == gold,
                        "generated": gen,
                        "neighbours": [int(j) for j in neighbours[i]]})

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    n_disk = sum(1 for _ in open(out_path, encoding="utf-8"))
    if n_disk != len(results):
        die("wrote %d rows, disk has %d" % (len(results), n_disk))
    print("\n  [verified] %d lines -> %s" % (n_disk, out_path))

    # ---------- diagnostics (freeze G) ----------
    gold_in_prompt = sum(r["gold"] in p for r, p in zip(results, prompts))
    s1_unparsed = sum(r["pred"] == "" for r in results)
    print("\n" + "-" * 78)
    print("MANDATORY DIAGNOSTICS (freeze G) — descriptive, never gates")
    print("-" * 78)
    print("G1 pool %d -> %d, removed %d" % (len(pool), pool_after, len(removed)))
    print("G2 near-duplicate drops %d across %d items" % (n_nearsim, rows_nearsim))
    print("G3 similarity to English source: max %.3f  mean %.3f"
          % (max(max_sims), sum(mean_sims) / len(mean_sims)))
    print("G4 gold string still in final prompt: %d/%d = %.2f%%  "
          "(DESCRIPTIVE: unrelated problems may share an answer)"
          % (gold_in_prompt, len(results), 100.0 * gold_in_prompt / len(results)))
    print("G5 unparseable: S1 %d/%d" % (s1_unparsed, len(results)))
    print("\nG6 first 10 items:")
    for r, ns in list(zip(results, neighbours))[:10]:
        print("  ---- %s  gold=%s pred=%s" % (r["qid"], r["gold"], r["pred"]))
        for k, j in enumerate(ns, 1):
            print("     D%d [%d]: %s" % (k, j, pool[j]["question"][:100]))
        print("     tail: %s" % r["generated"][-200:].replace("\n", " "))

    if args.test:
        print("\nTEST COMPLETE. Not a result.")
        return

    # ---------- readings (freeze F) ----------
    b0 = {r["qid"]: r for r in read_jsonl(B0_FILE)}
    if set(b0) != {r["qid"] for r in results}:
        die("qid sets differ between B0 and S1; run is VOID (freeze D)")
    b0_unparsed = sum(r["pred"] == "" for r in b0.values())
    spread = abs(100.0 * s1_unparsed / len(results)
                 - 100.0 * b0_unparsed / len(b0))
    b0_acc = 100.0 * sum(r["correct"] for r in b0.values()) / len(b0)
    s1_acc = 100.0 * sum(r["correct"] for r in results) / len(results)
    b = sum(r["correct"] and not b0[r["qid"]]["correct"] for r in results)
    c = sum(b0[r["qid"]]["correct"] and not r["correct"] for r in results)
    p = mcnemar(b, c)
    d = s1_acc - b0_acc

    print("\n" + "=" * 78)
    print("G1 RESULT   n=%d" % len(results))
    print("=" * 78)
    print("B0  baseline CoT (reused)      acc %6.2f%%  unparsed %d" % (b0_acc, b0_unparsed))
    print("S1  SDFR, DECONTAMINATED pool  acc %6.2f%%  unparsed %d" % (s1_acc, s1_unparsed))
    print("\nunparseable-rate spread: %.2f pp" % spread)
    if spread > 5.0:
        print("*** VALIDITY (freeze D): spread > 5pp. Comparison VOID.")
        return
    print("paired McNemar S1 vs B0: b=%d  c=%d  p=%.6g" % (b, c, p))
    print("d = S1 - B0 = %+.2f pp" % d)
    if d > 0 and p < 0.05:
        rd = ("READING 1 — METHOD CONFIRMED. Report with the contamination "
              "history in freeze section A.")
    elif p >= 0.05:
        rd = ("READING 2 — NO EFFECT. The original GSM8K gain was leakage. "
              "SDFR-UR is NOT claimed as a method on GSM8K. Substantive at "
              "n=700, not an underpowered null.")
    else:
        rd = "READING 3 — METHOD HARMS. Reported as a finding; not claimed."
    print("\n  " + rd)

    with open(out_path.replace(".jsonl", "_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n": len(results), "b0_acc": b0_acc, "s1_acc": s1_acc,
                   "d_pp": d, "b": b, "c": c, "p": p, "spread": spread,
                   "pool_after": pool_after, "removed": len(removed),
                   "contam_drops": n_contam, "nearsim_drops": n_nearsim,
                   "gold_in_prompt": gold_in_prompt,
                   "unparsed": {"B0": b0_unparsed, "S1": s1_unparsed},
                   "reading": rd}, f, indent=2)
        f.flush(); os.fsync(f.fileno())


if __name__ == "__main__":
    main()
