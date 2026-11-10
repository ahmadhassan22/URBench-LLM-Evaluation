#!/usr/bin/env python3
"""
d11_a6_guard_survey.py — D11 RTG Amendment 6: CPU preflight of the amended leak guard on eval458.

No model, nothing written, no answer scored (no generation exists for these arms).
1. Runs the real student preflight (d11_student.preflight, amended guard) for the given eval458
   arms: all prompts are built, every check runs, and any FATAL hit stops the script there.
2. If it passes, tabulates every guard needle INDEPENDENTLY of the guard code — one needle at a
   time over every prompt: each question_en, each gold fact, each evidence id (its own regex with
   the boundaries of d11_common.needles_for), each sealed reading — and lists every needle with
   at least one hit as (arm, kind, needle, number of prompts, class).
   class: FATAL (question_en, readings, evidence ids with >= 3 letters, or a non-specific
   evidence id outside the logged A6.3 hits), NON-SPECIFIC (A6.3: evidence id with < 3 letters,
   inside a teacher fact, logged), V6 (gold-fact restatement in a facts arm: reported, as before).
3. Prints every A6.3 hit: prompt row, needle, the rows owning the evidence id, the fact sentence,
   whether the reading contains the needle.

  python eval/error_analysis_tests/efbpt/d11_a6_guard_survey.py [--arms TR,TR_wrong_eval]
"""
import argparse
import os
import re
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import d11_student as S  # noqa: E402
from d11_common import load_set_rows, needles_for, load_readings, md5_file, EVAL458_READERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="TR,TR_wrong_eval")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",")]
    S.log_full_prompt = lambda label, prompt: print("[full prompt not echoed by the survey] %s (%d chars)" % (label, len(prompt)))
    _, sub, arms, prompts, _, _, _, _, pre = S.preflight("eval458", None, False, S.EVAL_CHOSEN, arms)

    rows = load_set_rows("eval458")
    needles = needles_for(rows, "eval458")
    readings = sorted({rd for k in sorted(EVAL458_READERS) for rd in load_readings(k, "eval458", rows)[0].values() if rd.strip()})
    non_ids = [e for e in needles["evidence_ids"] if sum(1 for ch in e if ch.isalpha()) < S.A6_MIN_LETTERS]
    print("\n" + "=" * 100)
    print("A6 GUARD SURVEY — eval458, arms %s; d11_student.py md5 %s; Amendment 6 md5 %s" % (arms, md5_file(os.path.join(_HERE, "d11_student.py")), S.D11_AMEND6_MD5))
    print("=" * 100)
    print("needles: %d question_en, %d gold facts, %d evidence ids (%d with >= %d letters = specific, %d non-specific), %d sealed readings"
          % (len(needles["question_en"]), len(needles["facts"]), len(needles["evidence_ids"]), len(needles["evidence_ids"]) - len(non_ids), S.A6_MIN_LETTERS,
             len(non_ids), len(readings)))
    print("non-specific evidence ids of the item set (fewer than %d letters): %s" % (S.A6_MIN_LETTERS, non_ids))
    table, n_fatal = [], 0
    for arm in arms:
        ps = [p for p, _, _, _ in prompts[arm]]
        logged = pre["leak_guard"][arm].get("A6_3_nonspecific_evidence_hits") or []
        logged_rows = {(h["needle"], h["prompt_row"]) for h in logged}
        for kind, items in (("question_en", needles["question_en"]), ("gold fact", needles["facts"]), ("reading", readings)):
            for s in items:
                n = sum(1 for p in ps if s in p)
                if n:
                    cls = "V6 (reported)" if kind == "gold fact" else "FATAL"
                    n_fatal += cls == "FATAL"
                    table.append((arm, kind, s, n, cls))
        for e in needles["evidence_ids"]:
            rx = re.compile(r"(?<![0-9A-Za-z])(?:%s)(?![0-9])" % re.escape(e))
            hit = [i for i, p in enumerate(ps) if rx.search(p)]
            if hit:
                if e in non_ids and all((e, i) in logged_rows for i in hit):
                    cls = "NON-SPECIFIC (A6.3: inside a teacher fact; logged)"
                else:
                    cls = "FATAL"
                    n_fatal += 1
                table.append((arm, "evidence id", e, len(hit), cls))
    print("\n%-14s %-12s %-70s %8s  %s" % ("arm", "kind", "needle", "prompts", "class"))
    for arm, kind, s, n, cls in table:
        print("%-14s %-12s %-70s %8d  %s" % (arm, kind, repr(s)[:70], n, cls))
    if not table:
        print("(no needle occurs in any prompt)")
    print("\nprompts built: " + ", ".join("%s %d" % (a, len(prompts[a])) for a in arms))
    print("FATAL hits: %d" % n_fatal)
    print("\nA6.3 hits (logged in S_START of each arm):")
    for arm in arms:
        for h in pre["leak_guard"][arm].get("A6_3_nonspecific_evidence_hits") or []:
            print("  %s prompt row %d (qid %s) <- facts of row %d (qid %s)" % (arm, h["prompt_row"], h["qid"], h["facts_source_row"], h["facts_source_qid"]))
            print("     needle %r (%d letters); rows owning the evidence id: %s" % (h["needle"], h["needle_letters"], [(o["row"], o["qid"]) for o in h["owner_rows"]]))
            print("     needle in the reading of the source row: %s" % h["needle_in_reading"])
            print("     fact sentence: %r" % h["fact"])
    if n_fatal:
        print("\n*** FATAL hits remain: submit nothing ***")
        sys.exit(1)
    print("\nA6 GUARD SURVEY: CLEAN (no fatal hit)")


if __name__ == "__main__":
    main()
