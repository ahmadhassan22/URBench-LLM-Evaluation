#!/usr/bin/env python3
"""
d14_score.py — the D14 scorers (freeze F5, 9.2, 9.11).

Imported only by the preflight (on the sealed fair outputs) and by the readout.
No runner imports this module.

The functions `extract_answer` and `extract_gold` are the function definitions of
the pinned fair scripts, taken from their source and run unchanged. The EN
variant for PIQA and GSM8K is the same source with the Urdu marker pattern
replaced by `(?i:Final\\s*answer)`; CSQA has one function for every arm.
"""
import ast
import json
import os
import re
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_common as C  # noqa: E402
from d14_common import die  # noqa: E402

UR_MARKER_PATTERN = "حتمی\\s*جواب"          # as it stands in the source of the PIQA and GSM8K fair scripts
EN_MARKER_PATTERN = "(?i:Final\\s*answer)"   # freeze 9.11
_FN = {}


def _functions(task):
    if task in _FN:
        return _FN[task]
    path = C.TASKS[task]["fair"]
    src = C._source(path)
    tree = ast.parse(src)
    seg = OrderedDict((n.name, ast.get_source_segment(src, n)) for n in tree.body if isinstance(n, ast.FunctionDef))
    if "extract_answer" not in seg:
        die("%s has no extract_answer" % path)
    out = OrderedDict()

    def run(code, name):
        ns = {"re": re}
        exec(compile(code, path, "exec"), ns)
        return ns[name]

    out["ur"] = run(seg["extract_answer"], "extract_answer")
    if task == "csqa":
        if UR_MARKER_PATTERN in seg["extract_answer"]:
            die("CSQA extract_answer unexpectedly holds the Urdu marker")
        out["en"] = out["ur"]
    else:
        if seg["extract_answer"].count(UR_MARKER_PATTERN) != 1:
            die("%s: extract_answer does not hold the Urdu marker pattern exactly once" % path)
        out["en"] = run(seg["extract_answer"].replace(UR_MARKER_PATTERN, EN_MARKER_PATTERN), "extract_answer")
    if task == "gsm8k":
        if "extract_gold" not in seg:
            die("%s has no extract_gold" % path)
        out["gold"] = run(seg["extract_gold"], "extract_gold")
    _FN[task] = out
    return out


def extract(task, variant, text):
    """The task's extract_answer ('ur' = unchanged, 'en' = the EN variant) on the answer text."""
    return _functions(task)[variant](text)


def gold_of(task, src_row):
    """The gold rule of the fair script on a row of the Urdu source file (freeze 9.2)."""
    if task == "csqa":
        return src_row["answerKey"].strip().upper()
    if task == "piqa":
        return str(src_row["label"])
    return _functions("gsm8k")["gold"](src_row["answer"])


def load_gold(task):
    """{rid: gold} from the Urdu source file, and the number of rows whose English source row gives another gold."""
    t = C.TASKS[task]
    ur, en = C.read_jsonl(t["ur"]), C.read_jsonl(t["en"])
    gold = OrderedDict(("%s_%04d" % (task, i), gold_of(task, r)) for i, r in enumerate(ur))
    diff = [i for i, (a, b) in enumerate(zip(ur, en)) if gold_of(task, a) != gold_of(task, b)]
    return gold, diff


def answer_text(model, generation):
    """F5: after </think> for Qwen3-14B (the fair script's strip and split); the whole output otherwise."""
    if model == "S":
        return generation.strip().split("</think>")[-1]
    return generation


def score_fair(task):
    """Recompute the sealed fair output of a task: (correct, rows, rows where the recomputed pred differs from the
    stored pred, rows where the gold rule on the source row differs from the stored gold)."""
    t = C.TASKS[task]
    out = C.read_jsonl(t["sealed"])
    ur = C.read_jsonl(t["ur"])
    lo, hi = t["eval_range"]
    if len(out) != hi - lo:
        die("%s: the sealed fair output has %d rows, the evaluation range has %d" % (task, len(out), hi - lo))
    correct = pred_diff = gold_diff = 0
    for i, r in enumerate(out):
        pred = extract(task, "ur", answer_text("S", r["generated"]))
        correct += int(pred == r["gold"])
        pred_diff += int(pred != r["pred"])
        gold_diff += int(gold_of(task, ur[lo + i]) != r["gold"])
    return correct, len(out), pred_diff, gold_diff


def self_check_en():
    """Hand-made cases for the EN variant and for the unchanged functions (no D14 file is read)."""
    cases = [
        ("piqa", "en", "Solution 1 uses 2 cups.\nFinal answer: 0", "0"),
        ("piqa", "en", "final ANSWER : 1", "1"),
        ("piqa", "ur", "Final answer: 0 ... 1", "1"),           # the unchanged function has no English marker: last digit
        ("piqa", "en", "no digit here", ""),
        ("gsm8k", "en", "3 + 4 = 7\nFinal answer: 1,250", "1250"),
        ("gsm8k", "en", "Final Answer: 12.5 dollars", "12.5"),
        ("gsm8k", "ur", "so 7 then 9", "9"),
        ("csqa", "en", "B is wrong. Answer: C", "C"),
        ("csqa", "ur", "Answer: C", "C"),
    ]
    for task, variant, text, want in cases:
        got = extract(task, variant, text)
        if got != want:
            die("scorer self-check: %s/%s on %r gave %r, expected %r" % (task, variant, text, got, want))
    return len(cases)


if __name__ == "__main__":
    C.verify_pins()
    n = self_check_en()
    print("[ok] %d hand-made scorer cases" % n)
    for task in C.TASKS:
        c, k, pd, gd = score_fair(task)
        print(json.dumps(OrderedDict([("task", task), ("sealed_fair_correct", c), ("rows", k), ("pred_differs", pd), ("gold_differs", gd)])))
