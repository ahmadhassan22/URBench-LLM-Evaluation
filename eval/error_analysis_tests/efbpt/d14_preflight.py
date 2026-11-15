#!/usr/bin/env python3
"""
d14_preflight.py — D14 CPU preflight (freeze F8, 9.1-9.4, 9.10-9.12). Writes nothing.

  1 pinned MD5s; the EN prompt strings and the translation system message are in the freeze text
  2 row counts 1500 / 750 / 700; Urdu and English rows aligned by position
  3 the UR prompt of every row, built by the fair script's own statements; its three parts
  4 the sealed fair outputs: prompt field (none is stored), row alignment, and the task scorers
    reproduce 190/300, 108/150 and 621/700 (FATAL)
  5 no gold answer in any prompt
  6 context length per arm (tokenizers only)
With --mode TEST|FULL and --with-readings the arms that need the sealed R1 readings and the
sealed teacher facts are built too (after those files exist).

The scorers are called here on the sealed fair outputs only. No D14 arm file is opened.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d14_preflight.py
  python eval/error_analysis_tests/efbpt/d14_preflight.py --mode FULL --with-readings
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_common as C  # noqa: E402
import d14_score as SC  # noqa: E402
from d14_common import die  # noqa: E402


def check_freeze_text():
    txt = open(C.rp(C.FREEZE_PATH), encoding="utf-8").read()
    n = 0
    for task, d in C.EN.items():
        for k in ("instruction", "trailer"):
            lit = json.dumps(d[k], ensure_ascii=False)
            if "`%s`" % lit not in txt:
                die("EN %s %s is not in the freeze text as written: %s" % (task, k, lit))
            n += 1
        item = json.dumps(C.item_block(task, ["{%s}" % x for x in
                                             {"csqa": ("q", "A", "B", "C", "D", "E"), "piqa": ("goal", "sol1", "sol2"), "gsm8k": ("question",)}[task]]),
                          ensure_ascii=False)
        if "`%s`" % item not in txt:
            die("EN %s item block layout is not in the freeze text: %s" % (task, item))
        n += 1
    if "`%s`" % json.dumps(C.R1_SYSTEM, ensure_ascii=False) not in txt:
        die("the translation system message is not in the freeze text as written")
    for pat in (C.D14_WORD_RE.pattern, C.D14_LABEL_RE.pattern.split("|")[0], C.D14_LABEL_RE.pattern.split("|")[1], SC.EN_MARKER_PATTERN, SC.UR_MARKER_PATTERN):
        if "`%s`" % pat not in txt:
            die("pattern %r is not in the freeze text" % pat)
    print("[ok] %d EN prompt strings, the translation system message and 5 patterns are in the freeze text as written" % n)


def check_rows(task):
    t = C.TASKS[task]
    ur, en = C.read_jsonl(t["ur"]), C.read_jsonl(t["en"])
    rows = C.load_rows(task)
    print("[rows] %s: %d Urdu rows, %d English rows (expected %d)" % (task, len(ur), len(en), t["n"]))
    if t["id_field"]:
        bad = [i for i, (a, b) in enumerate(zip(ur, en)) if a.get(t["id_field"]) != b.get(t["id_field"])]
        if tuple(bad) != tuple(t["id_mismatch_ok"]):
            die("%s: '%s' differs at positions %s, expected only %s" % (task, t["id_field"], bad, list(t["id_mismatch_ok"])))
        print("  '%s' equal at %d of %d positions; differs at 0-based %s (the known position of docs/KNOWN_DATA_ISSUES.md)"
              % (t["id_field"], len(ur) - len(bad), len(ur), bad))
        if "qid" in ur[0]:
            nq = sum(1 for a, b in zip(ur, en) if a["qid"] != b["qid"])
            if nq:
                die("%s: qid differs at %d positions" % (task, nq))
            print("  'qid' equal at every position")
    else:
        print("  no id field; aligned by position (same row count)")
    for side in ("ur", "en"):
        texts = [C.field_texts(task, r[side]) for r in rows]
        empty = sum(1 for tx in texts if any(not x.strip() for x in tx))
        nl = sum(1 for tx in texts if any("\n" in x or "\r" in x for x in tx))
        sp = sum(1 for tx in texts if any(x != x.strip() for x in tx))
        print("  %s fields: rows with an empty field %d, with a line break in a field %d, with a leading/trailing space %d" % (side, empty, nl, sp))
        if empty:
            die("%s: a %s row has an empty text field" % (task, side))
        if side == "ur" and nl:
            die("%s: an Urdu field holds a line break (the UR prompt parts could not be told apart)" % task)
    gold, diff = SC.load_gold(task)
    print("  gold (Urdu source row, fair-script rule): %d rows; the English source row gives another gold at %d positions%s"
          % (len(gold), len(diff), (" (0-based %s)" % diff) if diff else ""))
    if any(g == "" for g in gold.values()):
        die("%s: an empty gold" % task)
    return rows, ur


def check_prompts(task, rows, ur_src):
    t = C.TASKS[task]
    mt = C.fair_max_tokens(task)
    trailer = C.ur_trailer(task)
    users = [C.ur_user(task, r) for r in rows]
    nlab = len(t["labels"])
    for r, u in zip(rows, users):
        instr, items, tr = C.ur_parts(task, r)
        if u != instr + "\n\n" + items + "\n\n" + tr or len(items.split("\n")) != nlab:
            die("%s %s: the UR prompt is not instruction, %d item lines, trailer" % (task, r["rid"], nlab))
        if C.ur_user(task, r, "X") != instr + "\n\n" + items + "\n\nX\n\n" + tr:
            die("%s %s: a block does not land between the item lines and the trailer" % (task, r["rid"]))
        og = C.og_block(task, r)
        if C.parse_translation(task, og) != og or len(C.ur_block(task, r).split("\n")) != nlab:
            die("%s %s: an item block does not have %d label lines" % (task, r["rid"], nlab))
    print("[prompt] %s: %d UR prompts built by the statements of %s (max_tokens %d read from its source); trailer %r"
          % (task, len(users), t["fair"], mt, trailer))
    # sealed fair output: prompt field and row alignment
    sealed = C.read_jsonl(t["sealed"])
    lo, hi = t["eval_range"]
    has_prompt = sum(1 for r in sealed if "prompt" in r)
    qf = {"csqa": "question", "piqa": "goal", "gsm8k": "question"}[task]
    same_q = sum(1 for i, r in enumerate(sealed) if r["question"] == ur_src[lo + i][qf])
    print("  sealed fair output %s: %d rows; rows with a stored prompt field: %d (nothing to compare, freeze 9.1); "
          "stored question equal to the Urdu source row at positions %d..%d: %d of %d"
          % (t["sealed"], len(sealed), has_prompt, lo, hi - 1, same_q, len(sealed)))
    if same_q != len(sealed):
        die("%s: the sealed fair output is not aligned with source positions %d..%d" % (task, lo, hi - 1))
    if has_prompt:
        bad = sum(1 for i, r in enumerate(sealed) if "prompt" in r and users[lo + i] not in r["prompt"])
        if bad:
            die("%s: %d rebuilt UR prompts differ from the stored prompts" % (task, bad))
    # no gold answer in a prompt (9.12)
    if task == "gsm8k":
        leak = 0
        for r, src, u in zip(rows, ur_src, users):
            ans = src["answer"]
            for p in (u, C.en_user(task, C.og_block(task, r)), C.ur_user(task, r, C.tr_label_line() + "\n" + C.og_block(task, r))):
                leak += int(ans in p)
        if leak:
            die("gsm8k: the gold answer text is inside %d prompts" % leak)
        print("  gsm8k: the gold 'answer' text of a row is in none of its prompts")
    else:
        print("  %s: the builders receive the text fields only (%s); the gold field never reaches a prompt" % (task, ", ".join(C.TEXT_FIELDS[task])))
    return users


def check_lengths(task, rows, mode, with_readings, s_tok, t_tok):
    t = C.TASKS[task]
    T = C.teacher_frozen()
    r1 = facts = None
    if with_readings:
        r1, info = C.load_r1(task, mode, rows)
        print("  [r1] %s %s: sealed %s MD5 %s, reading_failed %d" % (task, mode, info["file"], info["md5"], info["reading_failed"]))
        if t["has_tr"]:
            facts, finfo = C.load_facts(task, mode, rows, r1)
            print("  [facts] %s %s: sealed %s MD5 %s, rows without facts %d" % (task, mode, finfo["file"], finfo["md5"], finfo["rows_without_facts"]))
    for arm in C.arms_of(task):
        spec = C.ARMS[arm]
        if spec["needs_r1"] and not with_readings:
            continue
        us = C.build_users(task, arm, rows, r1, facts)
        ok = [u for u in us if u["status"] == "ok"]
        if spec["model"] == "S":
            n = [C.n_tokens(s_tok, C.student_chat(s_tok, u["user"])) for u in ok]
            limit, mx = C.STUDENT_MAX_MODEL_LEN, t["max_tokens"]
        elif spec["model"] == "T":
            n = [len(C.teacher_render(t_tok, [{"role": "user", "content": u["user"]}])[1]) for u in ok]
            limit, mx = T["vllm_args"]["max_model_len"], C.TEACHER_ARM_MAX_TOKENS
        else:
            print("  [len] %-5s rows %d  reading_failed %d  (API arm: no local context limit)" % (arm, len(us), len(us) - len(ok)))
            continue
        n.sort()
        too = sum(1 for x in n if x + mx > limit)
        print("  [len] %-5s rows %d  reading_failed %d  prompt tokens min %d median %d max %d  (+%d new <= %d)  too_long %d"
              % (arm, len(us), len(us) - len(ok), n[0], n[len(n) // 2], n[-1], mx, limit, too))
    if with_readings and t["has_tr"]:
        blocks = [b for b in r1.values() if b is not None]
        n = sorted(len(C.teacher_render(t_tok, [{"role": "system", "content": T["facts_system"]}, {"role": "user", "content": C.facts_user(b)}])[1]) for b in blocks)
        print("  [len] facts prompts %d  tokens min %d median %d max %d  (+%d new <= %d)  too_long %d"
              % (len(n), n[0], n[len(n) // 2], n[-1], T["facts_max_tokens"], T["vllm_args"]["max_model_len"],
                 sum(1 for x in n if x + T["facts_max_tokens"] > T["vllm_args"]["max_model_len"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="FULL", choices=C.MODES)
    ap.add_argument("--with-readings", action="store_true", help="also build the arms that need the sealed R1 readings / teacher facts")
    ap.add_argument("--no-tokenizers", action="store_true", help="skip the context-length step")
    args = ap.parse_args()
    print("=" * 78)
    print("D14 PREFLIGHT (CPU, writes nothing)   mode %s   with readings: %s" % (args.mode, args.with_readings))
    print("=" * 78, flush=True)
    C.verify_pins()
    check_freeze_text()
    print("[label] translation label line %r (d11_ablate_ro RO_LABEL without its two leading words)" % C.tr_label_line())
    print("[label] facts block of ['x'] = %r (d1_eval_arms.facts_block)" % C.facts_block(["x"]))
    T = C.teacher_frozen()
    A = C.api_settings()
    print("[teacher] %s  vLLM %s  args %s  sampling %s" % (T["path"], T["vllm_version"], dict(T["vllm_args"]), dict(T["sampling"])))
    print("[teacher] facts: max_tokens %d, at most %d facts, verdict filter %r / %r" % (T["facts_max_tokens"], T["max_facts"], T["verdict_re"], T["yesno_re"]))
    print("[api] settings %s" % json.dumps(A, ensure_ascii=False))
    for p in (C.STUDENT_PATH, T["path"]):
        if not os.path.exists(os.path.join(p, "config.json")):
            die("model weights missing: " + p)
    print("[ok] model directories present")

    data = OrderedDict()
    for task in C.TASKS:
        rows, ur_src = check_rows(task)
        check_prompts(task, rows, ur_src)
        data[task] = rows

    n = SC.self_check_en()
    print("[score] %d hand-made scorer cases pass" % n)
    for task, t in C.TASKS.items():
        c, k, pd, gd = SC.score_fair(task)
        print("[score] %s: the task scorer on the sealed fair output gives %d / %d (freeze: %d / %d); recomputed pred differs from the stored pred on %d rows; "
              "gold rule on the source row differs from the stored gold on %d rows" % (task, c, k, t["sealed_correct"], t["eval_range"][1] - t["eval_range"][0], pd, gd))
        if c != t["sealed_correct"] or pd or gd:
            die("%s: the scorer does not reproduce the sealed fair output" % task)

    if not args.no_tokenizers:
        s_tok, t_tok = C.load_tok(C.STUDENT_PATH), C.load_tok(T["path"])
        for task in C.TASKS:
            rows = data[task][:C.TEST_N] if args.mode == "TEST" else data[task]
            print("[len] %s (%s, %d rows)" % (task, args.mode, len(rows)))
            check_lengths(task, rows, args.mode, args.with_readings, s_tok, t_tok)
    for task in C.TASKS:
        r = data[task][0]
        C.log_full_prompt("%s UR, row 0" % task, C.ur_user(task, r))
        C.log_full_prompt("%s S_og, row 0" % task, C.ur_user(task, r, C.tr_label_line() + "\n" + C.og_block(task, r)))
        C.log_full_prompt("%s EN with the original item block, row 0" % task, C.en_user(task, C.og_block(task, r)))
        C.log_full_prompt("%s translation USER, row 0" % task, C.ur_block(task, r))
    print("[preflight] PASS", flush=True)


if __name__ == "__main__":
    main()
