#!/usr/bin/env python3
"""
d15_score.py — the D15 scorer (freeze N5, 11.10).

Imported only by the preflight (on the sealed CSQA fair output) and by the readout. No runner
imports this module.

extract_answer = the function definition of the pinned CSQA fair script, taken from its source,
with the letter class [A-E] replaced by [A-D] (asserted to occur exactly twice) and otherwise run
unchanged; the same function for every D15 arm. The same code path with the class left at [A-E]
must give 190 / 300 on the sealed CSQA fair output (checked by score_fair; fatal).
"""
import ast
import os
import re
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d15_common as C  # noqa: E402
import d14_common as C14  # noqa: E402
from d15_common import die  # noqa: E402

CLASS_FAIR = "[A-E]"
CLASS_D15 = "[A-D]"
_FN = {}


def extractor(letter_class=CLASS_D15):
    """The fair extract_answer with its letter class set to `letter_class` ([A-E] = unchanged, [A-D] = D15)."""
    if letter_class in _FN:
        return _FN[letter_class]
    src = C14._source(C.FAIR_SCRIPT)
    tree = ast.parse(src)
    seg = [ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_answer"]
    if len(seg) != 1:
        die("%s: expected one extract_answer" % C.FAIR_SCRIPT)
    code = seg[0]
    if code.count(CLASS_FAIR) != 2:
        die("extract_answer holds %r %d times, expected exactly twice" % (CLASS_FAIR, code.count(CLASS_FAIR)))
    if letter_class != CLASS_FAIR:
        code = code.replace(CLASS_FAIR, letter_class)
    ns = {"re": re}
    exec(compile(code, C.FAIR_SCRIPT + " (letter class %s)" % letter_class, "exec"), ns)
    _FN[letter_class] = ns["extract_answer"]
    _FN["src_" + letter_class] = code
    return _FN[letter_class]


def answer_text(model, generation):
    """N5: after </think> for Qwen3-14B (the fair script's strip and split); the whole output otherwise."""
    if model == "S":
        return C.after_think(generation)
    return generation or ""


def extract(model, generation):
    return extractor(CLASS_D15)(answer_text(model, generation))


def score_fair():
    """The sealed CSQA fair output with the unrestricted class: (correct, rows, rows where the recomputed pred differs
    from the stored pred)."""
    fn = extractor(CLASS_FAIR)
    out = C.C14.read_jsonl(C.FAIR_SEALED)
    correct = diff = 0
    for r in out:
        pred = fn(answer_text("S", r["generated"]))
        correct += int(pred == r["gold"])
        diff += int(pred != r["pred"])
    return correct, len(out), diff


def self_check():
    """Hand-made cases for the A-D class (no D15 file is read)."""
    cases = [
        ("S", "<think>E is wrong</think>\nAnswer: C", "C"),
        ("S", "<think>x</think>جواب واضح ہے۔ Answer: E", ""),      # E is not a D15 letter, and no other single letter
        ("S", "<think>x</think>A or B ... D", "D"),
        ("T", "The answer is clear.\nAnswer: B", "B"),
        ("T", "answer : d", "D"),                                 # found by the fallback on the upper-cased text, as in the fair script
        ("T", "Answer: E. So not A.", "A"),                       # E is outside the class; the fallback takes the last A-D letter
        ("M", "no letter here", ""),
        ("M", "Answer：A", "A"),
        ("S", "<think>Answer: B", "B"),                           # truncated thinking: the whole text, as the fair script
    ]
    for model, text, want in cases:
        got = extract(model, text)
        if got != want:
            die("scorer self-check: %s on %r gave %r, expected %r" % (model, text, got, want))
    if extractor(CLASS_FAIR)("Answer: E") != "E" or extractor(CLASS_D15)("Answer: E") != "":
        die("scorer self-check: the class change is not effective")
    return len(cases) + 1


if __name__ == "__main__":
    C.verify_pins()
    print("[ok] %d hand-made scorer cases" % self_check())
    c, n, d = score_fair()
    print("[fair] sealed CSQA fair output with %s: %d / %d (stored pred differs on %d rows)" % (CLASS_FAIR, c, n, d))
    if (c, n, d) != (C.FAIR_SEALED_CORRECT, 300, 0):
        die("the scorer does not reproduce %d / 300" % C.FAIR_SEALED_CORRECT)
    print(OrderedDict([("d15_source", _FN["src_" + CLASS_D15])])["d15_source"])
