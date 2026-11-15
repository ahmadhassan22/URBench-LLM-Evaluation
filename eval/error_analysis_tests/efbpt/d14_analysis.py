#!/usr/bin/env python3
"""
d14_analysis.py — D14 analysis pack (descriptive). Run after the D14 readout exists.

It decides nothing and attaches no wording. Output: docs/handoff/D14_ANALYSIS.md
  A  the reading effect for 4 tasks x 3 models: StrategyQA from docs/handoff/D13_ANALYSIS.md §A
     (lines copied), CSQA / PIQA / GSM8K from docs/handoff/D14_READOUT.md (lines copied)
  B  translation loss: the original English item against the R1 translation (*_og vs *_r1),
     lines copied from the readout, with the overlap counts
  C  S_tr vs S_ro: the line copied from the readout, the overlap counts, and the split by
     whether the row has teacher facts
  D  15 random rows per task (seed 1010) where S_ro is right and S_ur wrong, and 15 where S_tr
     is wrong and S_ur right

The arm files are scored with the readout's own functions (d14_readout.evaluate, imported).
PASS CHECK (fatal): the accuracy and the correct count of every arm recomputed here equal the
per-arm table of the committed readout.
"""
import json
import os
import random
import re
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_common as C  # noqa: E402
import d14_score as SC  # noqa: E402
import d14_readout as R  # noqa: E402
import d14_tables as TB  # noqa: E402  (the readout table parser)
from d14_common import die  # noqa: E402

OUT_PATH = "docs/handoff/D14_ANALYSIS.md"
READOUT = "docs/handoff/D14_READOUT.md"
D13_ANALYSIS = ("docs/handoff/D13_ANALYSIS.md", "971b584e41d8126f2c6c6e127be4c633")
SEED = 1010
N_EXAMPLES = 15
STRATEGYQA = [("qwen3.8-max", "API_tr_R1 vs API_direct_ur_eval"), ("Llama-3.1-70B", "T_tr_R1 vs T_direct_F_eval"), ("Qwen3-14B", "RO_eval vs A_r2_eval")]
D14_PAIRS = [("qwen3.8-max", "M_r1", "M_ur"), ("Llama-3.1-70B", "T_r1", "T_ur"), ("Qwen3-14B", "S_ro", "S_ur")]
MODEL_NAME = {"S": "Qwen3-14B", "T": "Llama-3.1-70B", "M": "qwen3.8-max"}


def short(s, n=220):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."


def cell(s):
    return str(s).replace("|", "\\|")


def overlap(x, y):
    both = sum(1 for a, b in zip(x, y) if a["correct"] and b["correct"])
    xo = sum(1 for a, b in zip(x, y) if a["correct"] and not b["correct"])
    yo = sum(1 for a, b in zip(x, y) if b["correct"] and not a["correct"])
    return both, xo, yo, len(x) - both - xo - yo


def main():
    C.verify_pins()
    if not os.path.exists(C.rp(READOUT)):
        die("%s does not exist: the analysis runs after the readout" % READOUT)
    if C.md5_file(D13_ANALYSIS[0]) != D13_ANALYSIS[1]:
        die("%s MD5 differs from the pinned value" % D13_ANALYSIS[0])
    ro_text = open(C.rp(READOUT), encoding="utf-8").read()
    ro_lines = ro_text.split("\n")
    secs = TB.sections(ro_text)

    # ---- arms, scored with the readout's functions; PASS CHECK against the readout's per-arm table
    gold, ev, recs, missing, rows = {}, OrderedDict(), OrderedDict(), {}, {}
    for task in C.TASKS:
        gold[task], _ = SC.load_gold(task)
        rows[task] = C.load_rows(task)
        recs[task], _info, missing[task] = R.load_task(task)
        ev[task] = OrderedDict((arm, R.evaluate(task, arm, rr, gold[task])) for arm, rr in recs[task].items())
    checked = 0
    for task, name in R.TASK_NAME.items():
        head, trows = TB.tables(TB.find(secs, "### %s (n = " % name))[0]
        table = {r[0]: r[1] for r in trows}
        for arm, e in ev[task].items():
            s = R.summary(e)
            want = "%s (%d)" % (R.pct(s["correct"], s["n"]), s["correct"])
            if table.get(arm) != want:
                die("PASS CHECK: %s %s recomputed %s, the readout has %s" % (name, arm, want, table.get(arm)))
            checked += 1
    print("[check] PASS: accuracy and correct count of %d arm files equal the readout" % checked)

    def ro_line(task_name, pair, line_prefix):
        """The readout line (1-based number, cells) of a comparison row in any of its comparison tables."""
        hits = [(i + 1, TB.cells(l)) for i, l in enumerate(ro_lines)
                if l.startswith("| %s | %s | %s" % (task_name, pair, line_prefix)) and len(TB.cells(l)) == 10]
        if not hits:
            die("readout: no line for %s %s %s" % (task_name, pair, line_prefix))
        return hits[0]

    D = []
    D.append("# D14 analysis pack (descriptive)")
    D.append("")
    D.append("Written by `%sd14_analysis.py` (MD5 `%s`) after the readout `%s` (MD5 `%s`). It decides nothing and attaches no wording; the readout is the record. "
             "PASS CHECK: the accuracy and the correct count of %d arm files recomputed here equal the per-arm table of the readout."
             % (C.EFBPT, C.md5_file(C.EFBPT + "d14_analysis.py"), READOUT, C.md5_file(READOUT), checked))
    D.append("")
    D.append("Arms: S = Qwen3-14B, T = Llama-3.1-70B-AWQ, M = qwen3.8-max. `_ur` = Urdu prompt; S_ro = Urdu prompt plus the R1 translation; S_og = Urdu prompt plus the original English item; "
             "S_tr = Urdu prompt plus teacher facts; T_r1, M_r1 = English prompt with the R1 translation; T_og, M_og = English prompt with the original English item. "
             "b = first arm right and second wrong; p = exact two-sided McNemar.")
    D.append("")

    # ---- A
    D.append("## A. Reading effect, 4 tasks x 3 models (all rows)")
    D.append("")
    D.append("First arm = the model with the English translation; second arm = the same model on the Urdu item. StrategyQA rows are the L1 lines of `%s` §A (eval458, yes/no answers, scorer E0); "
             "the other rows are the LA lines of `%s` §3. The StrategyQA arms use the D13 prompts, which differ from the D14 prompts; for Qwen3-14B the reading is added to the Urdu prompt in every task." % (D13_ANALYSIS[0], READOUT))
    D.append("")
    D.append("| task | model | first arm vs second arm | n | acc first % | acc second % | delta pp | b | c | raw p | source line |")
    D.append("|---|---|---|---|---|---|---|---|---|---|---|")
    d13 = open(C.rp(D13_ANALYSIS[0]), encoding="utf-8").read().split("\n")
    for model, pair in STRATEGYQA:
        hits = [(i + 1, TB.cells(l)) for i, l in enumerate(d13) if l.startswith("| %s | L1 label |" % pair)]
        if len(hits) != 1:
            die("%s: %d L1 lines for %s" % (D13_ANALYSIS[0], len(hits), pair))
        ln, c = hits[0]
        D.append("| StrategyQA | %s | %s | %s | %s | %s | %s | %s | %s | %s | D13_ANALYSIS.md:%d |" % (model, pair, c[2], c[3], c[4], c[5], c[6], c[7], c[8], ln))
    for task, name in R.TASK_NAME.items():
        for model, a, b in D14_PAIRS:
            if a in missing[task] or b in missing[task]:
                D.append("| %s | %s | %s vs %s | missing | - | - | - | - | - | - | - |" % (name, model, a, b))
                continue
            ln, c = ro_line(name, "%s vs %s" % (a, b), "LA")
            D.append("| %s | %s | %s vs %s | %s | %s | %s | %s | %s | %s | %s | D14_READOUT.md:%d |"
                     % (name, model, a, b, c[3], TB.split2(c[4])[0], TB.split2(c[5])[0], c[6], c[7], c[8], c[9], ln))
    D.append("")

    # ---- B
    D.append("## B. Translation loss: the original English item against the R1 translation (all rows)")
    D.append("")
    D.append("First arm = the model with the original English item; second arm = the same model with the R1 translation. Lines copied from `%s` §5; the four overlap counts are computed here. "
             "A reading_failed row has no R1 translation and is wrong in the second arm." % READOUT)
    D.append("")
    D.append("| task | model | first arm vs second arm | n | acc first % | acc second % | delta pp | b | c | raw p | both right | first only | second only | both wrong | reading_failed rows | source line |")
    D.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for task, name in R.TASK_NAME.items():
        for a, b in (("S_og", "S_ro"), ("T_og", "T_r1"), ("M_og", "M_r1")):
            if a in missing[task] or b in missing[task]:
                D.append("| %s | %s | %s vs %s | missing | - | - | - | - | - | - | - | - | - | - | - | - |" % (name, MODEL_NAME[a[0]], a, b))
                continue
            ln, c = ro_line(name, "%s vs %s" % (a, b), "LA")
            o = overlap(ev[task][a], ev[task][b])
            if (o[1], o[2]) != (int(c[7]), int(c[8])):
                die("B: overlap of %s %s vs %s does not equal the readout's b and c" % (name, a, b))
            D.append("| %s | %s | %s vs %s | %s | %s | %s | %s | %s | %s | %s | %d | %d | %d | %d | %d | D14_READOUT.md:%d |"
                     % (name, MODEL_NAME[a[0]], a, b, c[3], TB.split2(c[4])[0], TB.split2(c[5])[0], c[6], c[7], c[8], c[9], o[0], o[1], o[2], o[3],
                        sum(1 for e in ev[task][b] if e["status"] == "reading_failed"), ln))
    D.append("")

    # ---- C
    D.append("## C. Teacher facts against the translation alone: S_tr vs S_ro (all rows)")
    D.append("")
    D.append("Both arms are Qwen3-14B on the Urdu prompt; S_tr adds teacher facts, S_ro adds the R1 translation. The first line of each task is copied from `%s` §5; the split by facts is computed here." % READOUT)
    D.append("")
    D.append("| task | rows | n | S_tr acc % (correct) | S_ro acc % (correct) | delta pp | b | c | raw p | both right | S_tr only | S_ro only | both wrong | source |")
    D.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for task, name in R.TASK_NAME.items():
        if not C.TASKS[task]["has_tr"]:
            continue
        ln, c = ro_line(name, "S_tr vs S_ro", "LA")
        x, y = ev[task]["S_tr"], ev[task]["S_ro"]
        o = overlap(x, y)
        D.append("| %s | all rows | %s | %s | %s | %s | %s | %s | %s | %d | %d | %d | %d | D14_READOUT.md:%d |" % (name, c[3], c[4], c[5], c[6], c[7], c[8], c[9], o[0], o[1], o[2], o[3], ln))
        nf = [r.get("n_facts") for r in recs[task]["S_tr"]]
        for label, keep in (("rows with at least one fact", lambda k: bool(k)), ("rows without facts (S_tr prompt = S_ur prompt) or reading_failed", lambda k: not k)):
            idx = [i for i, k in enumerate(nf) if keep(k)]
            xs, ys = [x[i] for i in idx], [y[i] for i in idx]
            l = R.line(xs, ys, lambda a, b: True)
            o = overlap(xs, ys)
            D.append("| %s | %s | %d | %s (%d) | %s (%d) | %s | %d | %d | %s | %d | %d | %d | %d | computed here |"
                     % (name, label, l["n"], R.pct(l["cx"], l["n"]), l["cx"], R.pct(l["cy"], l["n"]), l["cy"], R.delta_pp(l), l["b"], l["c"], R.fmt_p(l["p"]), o[0], o[1], o[2], o[3]))
    D.append("")
    D.append("S_tr against S_ur and S_og on the same split (computed here):")
    D.append("")
    D.append("| task | rows | n | S_ur acc % (correct) | S_ro acc % (correct) | S_og acc % (correct) | S_tr acc % (correct) |")
    D.append("|---|---|---|---|---|---|---|")
    for task, name in R.TASK_NAME.items():
        if not C.TASKS[task]["has_tr"]:
            continue
        nf = [r.get("n_facts") for r in recs[task]["S_tr"]]
        for label, keep in (("all rows", lambda k: True), ("rows with at least one fact", lambda k: bool(k)), ("rows without facts or reading_failed", lambda k: not k)):
            idx = [i for i, k in enumerate(nf) if keep(k)]
            cs = [sum(1 for i in idx if ev[task][a][i]["correct"]) for a in ("S_ur", "S_ro", "S_og", "S_tr")]
            D.append("| %s | %s | %d | %s |" % (name, label, len(idx), " | ".join("%s (%d)" % (R.pct(k, len(idx)), k) for k in cs)))
    D.append("")

    # ---- D
    r1 = {}
    facts = {}
    for task in C.TASKS:
        r1[task], _ = C.load_r1(task, "FULL", rows[task])
        if C.TASKS[task]["has_tr"]:
            facts[task], _ = C.load_facts(task, "FULL", rows[task], r1[task])
    D.append("## D. Example rows (seed %d)" % SEED)
    D.append("")
    D.append("Each list is a random sample without replacement (`random.Random(%d).sample`) of the row ids of the set, in row order of the set. Texts are cut to 220 characters, white space collapsed. "
             "Predictions are the scorer's reading of each answer; an empty prediction is unparsed." % SEED)
    D.append("")
    for task, name in R.TASK_NAME.items():
        e = ev[task]
        sets = [("S_ro right and S_ur wrong", "S_ro", [i for i in range(len(rows[task])) if e["S_ro"][i]["correct"] and not e["S_ur"][i]["correct"]])]
        if C.TASKS[task]["has_tr"]:
            sets.append(("S_tr wrong and S_ur right", "S_tr", [i for i in range(len(rows[task])) if not e["S_tr"][i]["correct"] and e["S_ur"][i]["correct"]]))
        for title, arm, idx in sets:
            pick = sorted(random.Random(SEED).sample(idx, min(N_EXAMPLES, len(idx))))
            D.append("### D. %s — %s: %d of %d rows" % (name, title, len(pick), len(idx)))
            D.append("")
            extra = "R1 translation" if arm == "S_ro" else "teacher facts"
            D.append("| row | Urdu item | %s | gold | S_ur | %s |" % (extra, arm))
            D.append("|---|---|---|---|---|---|")
            for i in pick:
                row = rows[task][i]
                rid = row["rid"]
                if arm == "S_ro":
                    add = r1[task][rid] or "(reading_failed)"
                else:
                    add = " / ".join(facts[task][rid]) if facts[task][rid] else ("(reading_failed)" if r1[task][rid] is None else "(no facts)")
                D.append("| %s | %s | %s | %s | %s | %s |" % (rid, cell(short(C.ur_block(task, row).replace("\n", " / "))), cell(short(add.replace("\n", " / "), 300)),
                                                         cell(gold[task][rid]), cell(e["S_ur"][i]["pred"] or "(unparsed)"), cell(e[arm][i]["pred"] or "(unparsed)")))
            D.append("")
    text = "\n".join(D) + "\n"
    with open(C.rp(OUT_PATH), "w", encoding="utf-8") as f:
        f.write(text)
    print("[done] wrote %s (%d lines)" % (OUT_PATH, text.count("\n")))


if __name__ == "__main__":
    main()
