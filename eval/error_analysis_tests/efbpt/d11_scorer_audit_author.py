#!/usr/bin/env python3
"""
d11_scorer_audit_author.py — D11 RTG Amendment 6 note N6: the author's independent labelling of a
30-item subset of the scorer audit. The labels select nothing.

Login node, CPU. DEV200 audit items only; nothing under an eval458 directory is opened. Edits no frozen
file; the items are parsed with d11_scorer_audit_adjudicators.parse_items (items file MD5 pinned there).

Subset (N6): the 6 items where the adjudicators A (PRIMARY), B (qwen3.8-max), C (deepseek-v4-pro)
disagree (28, 40, 41, 78, 95, 101; checked against the three label files) + 24 of the other 99.
One generator: rng = random.Random(2026); drawn = rng.sample(<the other 99, ascending>, 24);
order = [the 6, ascending] + sorted(drawn); rng.shuffle(order). Sheet number k = order[k - 1].

  --make            writes docs/handoff/D11_SCORER_AUDIT_AUTHOR_SHEET.md (three lines of rubric; per item:
                    sheet number, Urdu question, the text shown in the items file, an empty "Label:" line;
                    no item number, arm, gold or label) and the map sheet number -> item number in
                    outputs/efbpt/d11/_audit/author_sheet_map.json. The map is never printed. Existing
                    files are never overwritten with different content.
  --labels FILE     FILE (or - for stdin) holds one line per item: "<sheet number> <label>", label =
                    yes / no / none; empty lines and lines starting with # are ignored. All 30 sheet
                    numbers are required, each once. Prints the N6 comparison: on the 24 random items,
                    agreement and Cohen's kappa of the author with PRIMARY and with the majority label;
                    the 6 disagreement items one by one. Prints only; writes nothing.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import d11_scorer_audit_adjudicators as J  # noqa: E402  (chdir to the repo root)
from d11_common import md5_file, die  # noqa: E402

SEED = 2026
DISAGREE = (28, 40, 41, 78, 95, 101)
N_RANDOM = 24
SHEET_PATH = "docs/handoff/D11_SCORER_AUDIT_AUTHOR_SHEET.md"
MAP_PATH = "outputs/efbpt/d11/_audit/author_sheet_map.json"


def load_labels():
    lab = OrderedDict()
    for name, path in (("PRIMARY", J.PRIMARY_PATH), ("B", J.LABEL_FILES["qwen"]), ("C", J.LABEL_FILES["deepseek"]), ("majority", J.LABEL_FILES["majority"])):
        lab[name] = {int(k): v for k, v in json.load(open(path, encoding="utf-8")).items()}
        if sorted(lab[name]) != list(range(1, J.N_ITEMS + 1)):
            die("%s does not label items 1..%d" % (path, J.N_ITEMS))
    return lab


def sheet_order(lab):
    dis = [i for i in range(1, J.N_ITEMS + 1) if len({lab[k][i] for k in ("PRIMARY", "B", "C")}) > 1]
    if dis != list(DISAGREE):
        die("the items where the adjudicators disagree are %s, note N6 lists %s" % (dis, list(DISAGREE)))
    rng = random.Random(SEED)
    others = [i for i in range(1, J.N_ITEMS + 1) if i not in DISAGREE]
    drawn = rng.sample(others, N_RANDOM)
    order = list(DISAGREE) + sorted(drawn)
    rng.shuffle(order)
    return order


def write_once(path, text):
    if os.path.exists(path):
        if open(path, encoding="utf-8").read() != text:
            die("%s exists with different content; not overwritten" % path)
        return "unchanged"
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    return "written"


def make(order):
    items = {n: (q, blocks) for n, q, blocks in J.parse_items()}
    out = ["# D11 scorer audit — author sheet (Amendment 6 note N6)", "",
           "- yes / no = what the answer says.",
           "- none = no clear answer, or both.",
           "- An opening %s whose own sentence denies the question = no." % J.HAAN, ""]
    for k, n in enumerate(order, 1):
        q, blocks = items[n]
        out += ["## %d" % k, "", "Question: %s" % q, ""]
        fz = "```"
        while fz in "".join(t for _, t in blocks):
            fz += "`"
        for cap, text in blocks:
            out += [cap, fz, text, fz, ""]
        out += ["Label:", ""]
    sheet = "\n".join(out)
    if re.search(r"(?m)^## Item ", sheet):
        die("the sheet would show an item number")
    st = write_once(SHEET_PATH, sheet)
    print("[%s] %s (%d items; MD5 %s)" % (st, SHEET_PATH, len(order), md5_file(SHEET_PATH)))
    m = OrderedDict([("what", "D11 scorer audit, author sheet (Amendment 6 note N6): sheet number -> item number of %s. Not shown to the author before the labels are in." % J.ITEMS_PATH),
                     ("seed", SEED), ("disagreement_items", list(DISAGREE)), ("n_random", N_RANDOM), ("items_md5", J.ITEMS_MD5),
                     ("map", OrderedDict((str(k), n) for k, n in enumerate(order, 1)))])
    st = write_once(MAP_PATH, json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    print("[%s] %s (not printed)" % (st, MAP_PATH))


def read_author(path, n):
    lines = (sys.stdin if path == "-" else open(path, encoding="utf-8")).read().split("\n")
    got = OrderedDict()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.match(r"^(\d+)\s+(\S+)$", ln)
        if not m:
            die("label line %r is not '<sheet number> <label>'" % ln)
        k, v = int(m.group(1)), m.group(2).lower()
        if not 1 <= k <= n:
            die("sheet number %d is not in 1..%d" % (k, n))
        if v not in J.LABELS:
            die("sheet number %d: label %r is not yes / no / none" % (k, v))
        if k in got:
            die("sheet number %d is labelled twice" % k)
        got[k] = v
    missing = [k for k in range(1, n + 1) if k not in got]
    if missing:
        die("no label for sheet number(s) %s; all %d are required before anything is compared" % (missing, n))
    return got


def compare(order, lab, path):
    if not os.path.exists(MAP_PATH):
        die("%s missing: run --make first" % MAP_PATH)
    m = json.load(open(MAP_PATH, encoding="utf-8"))
    if [m["map"][str(k)] for k in range(1, len(order) + 1)] != order:
        die("%s is not the map recomputed now" % MAP_PATH)
    author = read_author(path, len(order))
    by_item = {order[k - 1]: v for k, v in author.items()}
    rnd = sorted(i for i in by_item if i not in DISAGREE)
    print("# D11 scorer audit — author labels (Amendment 6 note N6). They select nothing.")
    print("")
    print("Sheet %s; %d items = %d where the adjudicators disagree + %d drawn with random.Random(%d). Author labels: yes %d, no %d, none %d."
          % (SHEET_PATH, len(order), len(DISAGREE), len(rnd), SEED, sum(1 for v in by_item.values() if v == "yes"),
             sum(1 for v in by_item.values() if v == "no"), sum(1 for v in by_item.values() if v == "none")))
    print("")
    print("## The %d random items" % len(rnd))
    print("")
    print("| pair | items agreeing | agreement % | Cohen's kappa |")
    print("|---|---|---|---|")
    a = [by_item[i] for i in rnd]
    for name, disp in (("PRIMARY", "author vs PRIMARY (adjudicator A)"), ("majority", "author vs majority of A, B, C")):
        b = [lab[name][i] for i in rnd]
        po, kappa = J.cohen(a, b)
        print("| %s | %d / %d | %.2f | %.3f |" % (disp, round(po * len(rnd)), len(rnd), 100.0 * po, kappa))
    diff = [i for i in rnd if by_item[i] != lab["PRIMARY"][i] or by_item[i] != lab["majority"][i]]
    print("")
    print("Random items where the author differs from PRIMARY or from the majority: %s."
          % ("; ".join("item %d (author %s, PRIMARY %s, majority %s)" % (i, by_item[i], lab["PRIMARY"][i], lab["majority"][i]) for i in diff) or "none"))
    print("")
    print("## The %d items where the adjudicators disagree" % len(DISAGREE))
    print("")
    print("| item | author | PRIMARY (adjudicator A) | B (%s) | C (%s) | majority of A, B, C |" % (J.MODELS["qwen"], J.MODELS["deepseek"]))
    print("|---|---|---|---|---|---|")
    for i in DISAGREE:
        print("| %d | %s | %s | %s | %s | %s |" % (i, by_item[i], lab["PRIMARY"][i], lab["B"][i], lab["C"][i], lab["majority"][i]))
    print("")
    print("Author = PRIMARY on %d of %d; author = majority on %d of %d."
          % (sum(1 for i in DISAGREE if by_item[i] == lab["PRIMARY"][i]), len(DISAGREE), sum(1 for i in DISAGREE if by_item[i] == lab["majority"][i]), len(DISAGREE)))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make", action="store_true", help="write the sheet and the (unprinted) map")
    g.add_argument("--labels", metavar="FILE", help="lines '<sheet number> <label>' (- = stdin); print the N6 comparison")
    args = ap.parse_args()
    J.check_inputs()
    lab = load_labels()
    order = sheet_order(lab)
    if len(order) != len(DISAGREE) + N_RANDOM or len(set(order)) != len(order):
        die("the sheet does not have %d distinct items" % (len(DISAGREE) + N_RANDOM))
    if args.make:
        make(order)
    else:
        compare(order, lab, args.labels)


if __name__ == "__main__":
    main()
