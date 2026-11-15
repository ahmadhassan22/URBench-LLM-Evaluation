#!/usr/bin/env python3
"""
d13_sensitivity.py — D13 sensitivity check of the scorer. WRITTEN AFTER THE D13 RESULTS WERE SEEN.

It changes no verdict: the D13 readout (docs/handoff/D13_READOUT.md) and its wording stand as
they are. This script only shows how the all-rows line of four comparisons moves if answers
that the frozen extractor E0 leaves unparsed, but that carry a verdict word E0 does not read,
are counted.

Scorer E0x = E0 (efbpt_eval_dev200.extract_answer), and only where E0 is unparsed, in this order:
  1. the answer's first three words contain the Urdu word "جی"            -> yes
  2. else Devanagari: the last of "नहीं" (no) and "हाँ" / "हां" (yes)      -> that label
  3. else Latin: the last match of \\b(nahi|nahin|nahee?n)\\b (no) or
     \\b(haan|ji haan|jee haan)\\b (yes), case-insensitive                 -> that label
A word = a maximal run of letters and combining marks (the word rule of d13_analysis.py).
A refused row stays unparsed.

Comparisons (eval458, all rows): the three F2 pairs (Llama8B_TR vs Llama8B_A, Gemma9B_TR vs
Gemma9B_A, Qwen8B_TR vs Qwen8B_A) and TR vs A_r2_eval. For each: the all-rows line under E0
and under E0x, and how many rows E0x newly parses per arm.

Everything is read through the committed D13 readout code (d13_readout.py, imported): the same
seal checks, the same E0, the same paired line. No model is run. No sealed file is changed.

  --self-test   hand-made E0x cases only; no arm file is opened
Output: docs/handoff/D13_SENSITIVITY.md
"""
import argparse
import os
import re
import sys
import unicodedata
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUT_PATH = "docs/handoff/D13_SENSITIVITY.md"
SET = "eval458"
PAIRS = [("Llama8B_TR", "Llama8B_A"), ("Gemma9B_TR", "Gemma9B_A"), ("Qwen8B_TR", "Qwen8B_A"), ("TR", "A_r2_eval")]
REF_ARMS = ("TR", "A_r2_eval")                     # sealed D11 arms; the others are sealed D13 arms

JI = "".join(chr(c) for c in (0x062C, 0x06CC))                       # جی
DEV_NO = "".join(chr(c) for c in (0x0928, 0x0939, 0x0940, 0x0902))    # नहीं
DEV_YES = ("".join(chr(c) for c in (0x0939, 0x093E, 0x0901)),          # हाँ
           "".join(chr(c) for c in (0x0939, 0x093E, 0x0902)))          # हां
RE_LATIN_NO = re.compile(r"\b(nahi|nahin|nahee?n)\b", re.IGNORECASE)
RE_LATIN_YES = re.compile(r"\b(haan|ji haan|jee haan)\b", re.IGNORECASE)
RULES = ("ji", "devanagari", "latin")


def words(s):
    """Maximal runs of letters and combining marks (as d13_analysis.words)."""
    out, cur = [], []
    for ch in s:
        if unicodedata.category(ch)[0] in ("L", "M"):
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def e0x_extra(text):
    """(label, rule) for a text that E0 leaves unparsed; (None, None) if no rule reads it."""
    if JI in words(text)[:3]:
        return "yes", "ji"
    pos = [(text.rfind(DEV_NO), "no")] + [(text.rfind(w), "yes") for w in DEV_YES]
    pos = [(p, lab) for p, lab in pos if p >= 0]
    if pos:
        return max(pos)[1], "devanagari"
    last = None
    for rx, lab in ((RE_LATIN_NO, "no"), (RE_LATIN_YES, "yes")):
        for m in rx.finditer(text):
            if last is None or m.start() > last[0]:
                last = (m.start(), lab)
    if last:
        return last[1], "latin"
    return None, None


def self_test():
    ur_yes = "".join(chr(c) for c in (0x06C1, 0x0627, 0x06BA))
    cases = [
        (JI + "، یہ ممکن ہے", ("yes", "ji")),
        ("یہ بات " + JI + " ہے", ("yes", "ji")),                         # third word
        ("یہ ایک بات " + JI + " ہے", (None, None)),                      # fourth word: no rule
        ("जी " + DEV_NO + " यह संभव " + DEV_YES[0], ("yes", "devanagari")),   # the last Devanagari word decides
        (DEV_YES[1] + " ... " + DEV_NO, ("no", "devanagari")),
        ("Nahi, yeh mumkin nahin hai", ("no", "latin")),
        ("pehle nahi, phir Ji Haan", ("yes", "latin")),
        ("naheen", ("no", "latin")), ("nahen", ("no", "latin")), ("haanji", (None, None)),
        ("It depends.", (None, None)), ("", (None, None)),
        (JI + " " + DEV_NO, ("yes", "ji")),                              # rule order: 1 before 2
    ]
    for text, want in cases:
        got = e0x_extra(text)
        if got != want:
            raise SystemExit("[FATAL] self-test: e0x_extra(%r) = %r, expected %r" % (text, got, want))
    assert ur_yes
    return len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    n_cases = self_test()
    print("[self-test] PASS: %d hand-made E0x cases" % n_cases)
    if args.self_test:
        return

    import d13_readout as R                  # the committed D13 readout code: seal checks, E0, paired line
    C = R.C
    rows = R.load_set_rows(SET)
    gold = OrderedDict((r["qid"], r["gold"]) for r in rows)

    arms, files = OrderedDict(), OrderedDict()
    for arm in sorted({a for p in PAIRS for a in p}, key=lambda a: [x for p in PAIRS for x in p].index(a)):
        if arm in REF_ARMS:
            scored, info = R.load_ref(SET, arm, rows)
            path = info["file"]
            raw = R.read_jsonl(path)
            gens = [r["generation"] if "generation" in r else r["raw"] for r in raw]
        else:
            scored, info = R.load_d13(arm, SET, rows)
            path = C.arm_paths(arm, SET, "FULL")["rows"]
            raw = C.read_rows(path)
            gens = [r["generation"] for r in raw]
        if [r["qid"] for r in raw] != list(gold):
            R.die("%s: rows are not the eval458 qids in order" % path)
        out = []
        for s, g in zip(scored, gens):
            x = dict(s)
            x["e0x"], x["rule"] = s["e0"], None
            if s["e0"] is None and not s["refused"]:
                x["e0x"], x["rule"] = e0x_extra(g)
            out.append(x)
        arms[arm] = out
        files[arm] = (path, R.md5_file(path))

    def allrows(x, y, key):
        return R.line(x, y, key, lambda a, b: True)

    D = []
    D.append("# D13 sensitivity check of the scorer — written after the results")
    D.append("")
    D.append("**This check was written after the D13 results were seen. It changes no verdict.** The D13 readout (`docs/handoff/D13_READOUT.md`) and its wording stand as they are; "
             "the frozen extractor E0 remains the scorer of record. The lines below only show how the all-rows line of four comparisons moves under a wider scorer.")
    D.append("")
    D.append("Written by `eval/error_analysis_tests/efbpt/d13_sensitivity.py` (MD5 `%s`). No model was run and no sealed file was changed. "
             "The arm files are read through the committed D13 readout code (same seal checks, same E0, same paired line)." % R.md5_file(os.path.join(_HERE, "d13_sensitivity.py")))
    D.append("")
    D.append("Scorer E0x = E0, and only where E0 is unparsed, in this order: (1) the answer's first three words contain the Urdu word \"%s\" -> yes; "
             "(2) else Devanagari: the last of \"%s\" (no) and \"%s\" / \"%s\" (yes); (3) else Latin: the last match of `%s` (no) or `%s` (yes), case-insensitive. "
             "A word is a maximal run of letters and combining marks. A row E0 parses keeps its E0 label." % (JI, DEV_NO, DEV_YES[0], DEV_YES[1], RE_LATIN_NO.pattern, RE_LATIN_YES.pattern))
    D.append("")
    D.append("Set: eval458, %d rows, all rows. Unparsed = wrong. b = first arm right and second wrong; exact two-sided McNemar." % len(rows))
    D.append("")
    D.append("## 1. Sealed files read")
    D.append("")
    D.append("| arm | file | MD5 |")
    D.append("|---|---|---|")
    for arm, (path, md5) in files.items():
        D.append("| %s | `%s` | `%s` |" % (arm, path, md5))
    D.append("")
    D.append("## 2. Rows that E0x newly parses, per arm")
    D.append("")
    D.append("| arm | E0 unparsed | newly parsed by E0x | by rule 1 (\"%s\") | by rule 2 (Devanagari) | by rule 3 (Latin) | newly parsed and right | newly parsed and wrong | E0x unparsed | E0 acc %% (correct) | E0x acc %% (correct) |" % JI)
    D.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, recs in arms.items():
        new = [r for r in recs if r["e0"] is None and r["e0x"] is not None]
        by = OrderedDict((k, sum(1 for r in new if r["rule"] == k)) for k in RULES)
        right = sum(1 for r in new if r["e0x"] == r["gold"])
        c0 = sum(1 for r in recs if r["e0"] == r["gold"])
        cx = sum(1 for r in recs if r["e0x"] == r["gold"])
        D.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %.2f (%d) | %.2f (%d) |"
                 % (arm, sum(1 for r in recs if r["e0"] is None), len(new), by["ji"], by["devanagari"], by["latin"], right, len(new) - right,
                    sum(1 for r in recs if r["e0x"] is None), 100.0 * c0 / len(recs), c0, 100.0 * cx / len(recs), cx))
    D.append("")
    D.append("## 3. All-rows line under E0 and under E0x")
    D.append("")
    D.append("| comparison | scorer | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |")
    D.append("|---|---|---|---|---|---|---|---|---|")
    for a, b in PAIRS:
        for key, name in (("e0", "E0"), ("e0x", "E0x")):
            l = allrows(arms[a], arms[b], key)
            D.append("| %s vs %s | %s | %d | %.2f (%d) | %.2f (%d) | %+.2f | %d | %d | %s |"
                     % (a, b, name, l["n"], l["acc_x"], l["cx"], l["acc_y"], l["cy"], l["delta"], l["b"], l["c"], R.fp(l["p"])))
    D.append("")
    D.append("The E0 rows are the L1 (all rows) numbers of the committed readouts: the three F2 pairs in `docs/handoff/D13_READOUT.md`, TR vs A_r2_eval in `docs/handoff/D11_EVAL458_READOUT_LINES.md`. "
             "In the D13 readout the three F2 tests were decided on other lines (the unparsed and truncation rules fired); those deciding lines and the Holm step are not recomputed here.")
    D.append("")
    D.append("No wording is attached to the E0x rows. They are a sensitivity check, not a result.")
    text = "\n".join(D) + "\n"
    with open(os.path.join(C.ROOT, OUT_PATH), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("[done] wrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
