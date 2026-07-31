#!/usr/bin/env python3
"""
d1_score_dual.py

Offline dual scoring for DIAGNOSTIC D1. No GPU, no model, no re-run.
Reads the saved generations and scores each arm twice:

  PRIMARY   - the frozen AMENDMENT 5 extractor, imported unchanged.
  SECONDARY - the same extractor plus Rule 2b (Devanagari), per
              DIAGNOSTIC D1 AMENDMENT 2 Section C.

Both are always reported together. The difference between them, per arm, IS
the Devanagari script-drift rate (AMENDMENT 2 Section D item 4).

Which score may be interpreted is decided by the rules frozen in AMENDMENT 2
Section D, applied here mechanically so the choice is not made by eye after
seeing the numbers.

Usage (from ~/URBench, CPU is fine):
  python eval/error_analysis_tests/efbpt/d1_score_dual.py
  python eval/error_analysis_tests/efbpt/d1_score_dual.py --test
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from efbpt_eval_dev200 import (          # noqa: E402
    extract_answer,                      # PRIMARY, frozen AMENDMENT 5
    HAAN, NAHIN, MARKER,
    RE_JSON_ANSWER, RE_EN_WORD,
    die,
)

ARMS = ["A", "B", "C", "E", "F", "G"]
ARM_LABEL = {
    "A": ("urdu q", "no facts"),
    "B": ("urdu q", "gold facts"),
    "C": ("urdu q", "wrong facts"),
    "E": ("english q", "gold facts"),
    "F": ("english q", "no facts"),
    "G": ("no question", "gold facts"),
}

IN_DIR = "outputs/efbpt/d1/arms"

C0_REFERENCE = 57.50
A_TOLERANCE = 2.0
SPREAD_LIMIT = 5.0

# Devanagari strings built from codepoints, exactly as listed in
# DIAGNOSTIC D1 AMENDMENT 2 Section C. Never typed.
DEV_YES = [
    "".join(chr(c) for c in [0x0939, 0x093E, 0x0902]),   # हां
    "".join(chr(c) for c in [0x0939, 0x093E, 0x0901]),   # हाँ
]
DEV_NO = [
    "".join(chr(c) for c in [0x0928, 0x0939, 0x0940, 0x0902]),   # नहीं
]


def extract_answer_secondary(text):
    """AMENDMENT 5 extractor with Rule 2b (Devanagari) inserted after Rule 2."""

    # Rule 1: JSON answer field, LAST match.
    m = None
    for m in RE_JSON_ANSWER.finditer(text):
        pass
    if m is not None:
        return m.group(1).lower()

    seg = text
    if MARKER in seg:
        seg = seg[seg.rfind(MARKER) + len(MARKER):]

    # Rule 2: Urdu (Perso-Arabic).
    i_haan = seg.rfind(HAAN)
    i_nahin = seg.rfind(NAHIN)
    if i_haan != -1 or i_nahin != -1:
        return "yes" if i_haan > i_nahin else "no"

    # Rule 2b: Devanagari. Greatest index wins.
    d_yes = max(seg.rfind(s) for s in DEV_YES)
    d_no = max(seg.rfind(s) for s in DEV_NO)
    if d_yes != -1 or d_no != -1:
        return "yes" if d_yes > d_no else "no"

    # Rule 3: English, word-bounded, LAST match.
    m = None
    for m in RE_EN_WORD.finditer(text):
        pass
    if m is not None:
        return m.group(1).lower()

    return None


def score(records, key):
    n = len(records)
    correct = sum(1 for r in records if r[key] == r["gold"])
    unparsed = sum(1 for r in records if r[key] is None)
    pred_yes = sum(1 for r in records if r[key] == "yes")
    return OrderedDict([
        ("n", n),
        ("accuracy", round(100.0 * correct / n, 2)),
        ("unparsed_rate", round(100.0 * unparsed / n, 2)),
        ("predicted_yes_rate", round(100.0 * pred_yes / n, 2)),
        ("correct", correct),
        ("unparsed", unparsed),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="score the _TEST files")
    args = ap.parse_args()
    suffix = "_TEST" if args.test else ""

    print("[ok] Devanagari strings (hex):")
    for s in DEV_YES:
        print("     yes  %s" % " ".join("%04X" % ord(c) for c in s))
    for s in DEV_NO:
        print("     no   %s" % " ".join("%04X" % ord(c) for c in s))

    data = {}
    for arm in ARMS:
        path = os.path.join(IN_DIR, "d1_arm%s%s.jsonl" % (arm, suffix))
        if not os.path.exists(path):
            die("missing: " + path)
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["pred_primary"] = extract_answer(r["generation"])
                r["pred_secondary"] = extract_answer_secondary(r["generation"])
                # the stored pred must match a fresh primary run, or the saved
                # file and the frozen extractor have diverged
                if r["pred_primary"] != r["pred"]:
                    die("arm %s qid %s: stored pred %r != recomputed primary %r"
                        % (arm, r["qid"], r["pred"], r["pred_primary"]))
                recs.append(r)
        data[arm] = recs
        print("[load] arm %s: %d rows" % (arm, len(recs)))

    n = len(data["A"])
    n_no = sum(1 for r in data["A"] if r["gold"] == "no")
    floor = 100.0 * max(n_no, n - n_no) / n

    prim = {a: score(data[a], "pred_primary") for a in ARMS}
    seco = {a: score(data[a], "pred_secondary") for a in ARMS}

    print("\n" + "=" * 88)
    print("DUAL SCORING" + ("  (TEST — NOT A RESULT)" if args.test else ""))
    print("=" * 88)
    print("%-4s %-12s %-12s | %8s %9s | %8s %9s | %7s"
          % ("arm", "question", "facts",
             "PRI acc", "PRI unpr", "SEC acc", "SEC unpr", "drift"))
    for a in ARMS:
        q, fm = ARM_LABEL[a]
        drift = prim[a]["unparsed_rate"] - seco[a]["unparsed_rate"]
        print("%-4s %-12s %-12s | %8.2f %9.2f | %8.2f %9.2f | %7.2f"
              % (a, q, fm,
                 prim[a]["accuracy"], prim[a]["unparsed_rate"],
                 seco[a]["accuracy"], seco[a]["unparsed_rate"], drift))
    print("\nmajority-class floor: %.2f%%   n per arm: %d" % (floor, n))
    print("drift = rows recovered by the Devanagari rule, in pp "
          "(AMENDMENT 2 Section D item 4)")

    # ---- which score may be interpreted (AMENDMENT 2 Section D) ----
    ps = max(prim[a]["unparsed_rate"] for a in ARMS) - \
         min(prim[a]["unparsed_rate"] for a in ARMS)
    ss = max(seco[a]["unparsed_rate"] for a in ARMS) - \
         min(seco[a]["unparsed_rate"] for a in ARMS)

    print("\n" + "=" * 88)
    print("WHICH SCORE IS INTERPRETABLE (AMENDMENT 2 Section D)")
    print("=" * 88)
    print("primary   unparsed spread: %.2f pp" % ps)
    print("secondary unparsed spread: %.2f pp" % ss)

    if ps <= SPREAD_LIMIT:
        use, acc, label = "PRIMARY", {a: prim[a]["accuracy"] for a in ARMS}, \
            "rule 1: primary spread within limit"
    elif ss <= SPREAD_LIMIT:
        use, acc, label = "SECONDARY", {a: seco[a]["accuracy"] for a in ARMS}, \
            "rule 2: primary VOID under AMENDMENT 5D, secondary within limit"
    else:
        print("\nrule 3: BOTH spreads exceed %.1f pp. NEITHER comparison is "
              "valid." % SPREAD_LIMIT)
        print("Report the numbers, diagnose the remaining parsing gap, and draw")
        print("NO conclusion about knowledge vs language.")
        return

    print("-> using %s (%s)" % (use, label))

    if args.test:
        print("\nTEST FILES — these are not results. 20 rows cannot measure "
              "anything.")
        return

    # ---- validity + pre-declared reading ----
    print("\n" + "=" * 88)
    print("PRE-DECLARED CHECKS, on the %s score" % use)
    print("=" * 88)

    ok = True
    d = abs(acc["A"] - C0_REFERENCE)
    print("\n1. arm A = %.2f%% vs C0 %.2f%% (diff %.2f pp) -> %s"
          % (acc["A"], C0_REFERENCE, d, "PASS" if d <= A_TOLERANCE else "FAIL"))
    if d > A_TOLERANCE:
        ok = False
        print("   Setup differs from the main evaluation. Interpret nothing "
              "until explained.")

    print("\n2. facts actually used?  B = %.2f%%   C = %.2f%%   B-C = %+.2f pp"
          % (acc["B"], acc["C"], acc["B"] - acc["C"]))
    if acc["B"] - acc["C"] < 3.0:
        ok = False
        print("   FAIL: wrong facts score as well as right facts. The model is")
        print("   not using the facts. Nothing else is interpretable.")
    else:
        print("   PASS")

    print("\n3. leakage?  G = %.2f%%   floor = %.2f%%   G-floor = %+.2f pp"
          % (acc["G"], floor, acc["G"] - floor))
    if acc["G"] - floor > 10.0:
        print("   WARNING: facts alone predict the answer well above the floor.")
        print("   Gains in B and E are partly leakage, not reasoning.")
    else:
        print("   OK")

    if not ok:
        print("\nValidity checks FAILED. Contrasts below are for diagnosis only.")

    print("\n" + "-" * 88)
    print("CONTRASTS (%s score)" % use)
    print("-" * 88)
    print("  F - A   cost of the Urdu question, no facts      = %+.2f pp"
          % (acc["F"] - acc["A"]))
    print("  B - A   value of knowledge, Urdu question        = %+.2f pp"
          % (acc["B"] - acc["A"]))
    print("  E - B   cost of the Urdu question, facts given   = %+.2f pp"
          % (acc["E"] - acc["B"]))
    print("  E       full-English ceiling                     =  %.2f%%" % acc["E"])

    print("\n" + "-" * 88)
    print("PRE-DECLARED READING (D1 AMENDMENT 1 Section C)")
    print("-" * 88)
    if acc["E"] >= 80.0 and acc["B"] >= 80.0 and (acc["E"] - acc["B"]) <= 5.0:
        print("  Reading 2: KNOWLEDGE is the bottleneck. Question language costs")
        print("  little. Next method should target getting knowledge to the model.")
    elif acc["E"] >= 80.0 and (acc["E"] - acc["B"]) >= 15.0:
        print("  Reading 3: the URDU QUESTION is the bottleneck even when")
        print("  knowledge is supplied. The method must attack Urdu comprehension.")
    elif acc["E"] <= 70.0:
        print("  Reading 4: neither knowledge nor English phrasing is sufficient.")
        print("  The model cannot combine given facts on this task. This closes")
        print("  retrieval-style approaches as a family.")
    else:
        print("  No pre-declared reading fits cleanly. Report the numbers as they")
        print("  are. Do NOT invent a new reading after the fact.")

    print("\n  F - A is reported regardless: how much accuracy URBench's Urdu")
    print("  question translation costs versus the original English question.")

    out = {"floor": floor, "n": n, "primary": prim, "secondary": seco,
           "primary_spread": ps, "secondary_spread": ss, "used": use}
    p = os.path.join(IN_DIR, "d1_dual_scores%s.json" % suffix)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
