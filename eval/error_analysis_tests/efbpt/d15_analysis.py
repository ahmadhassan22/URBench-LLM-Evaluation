#!/usr/bin/env python3
"""
d15_analysis.py — D15 analysis pack (descriptive). Run after the D15 readout exists.

It decides nothing and attaches no wording. Output: docs/handoff/D15_ANALYSIS.md
  A  the reading effect over all five sets: StrategyQA (D13), CSQA / PIQA / GSM8K (D14): the 12
     lines of docs/handoff/D14_ANALYSIS.md §A copied as they stand (with their source lines);
     UrduMMLU (D15): the set-A LA lines of docs/handoff/D15_READOUT.md, copied
  B  the 10 subdomains with the largest and the 10 with the smallest S_ro - S_ur delta, from the
     per-subdomain table of the readout (cells copied, sorted here)
  C  per group (URDU, REGIONAL, OTHER), 15 random rows of set A (a new random.Random(1515) for each
     list, candidates in run order) where S_ro is right and S_ur wrong, and 15 where S_ro is wrong
     and S_ur right: the Urdu item, the R1 translation, both answers and the gold key

The arm files are scored with the readout's own functions (d15_readout.evaluate, imported).
PASS CHECK (fatal): the accuracy and the correct count of every arm on set A and on all sampled
rows, recomputed here, equal the per-arm tables of the committed readout.
"""
import os
import random
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d15_common as C  # noqa: E402
import d15_readout as R  # noqa: E402
import d14_tables as TB  # noqa: E402  (the readout table parser)
from d15_common import die  # noqa: E402

OUT_PATH = "docs/handoff/D15_ANALYSIS.md"
READOUT = "docs/handoff/D15_READOUT.md"
D14_ANALYSIS = ("docs/handoff/D14_ANALYSIS.md", "bea30d202e0da0745974a6b25e3522dc")
SEED = 1515
N_EXAMPLES = 15
D15_PAIRS = [("qwen3.8-max", "M_r1", "M_ue"), ("Llama-3.1-70B", "T_r1", "T_ue"), ("Qwen3-14B", "S_ro", "S_ur"),
             ("qwen3.8-max", "M_r1", "M_ur"), ("Llama-3.1-70B", "T_r1", "T_ur")]


def short(s, n=260):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."


def cell(s):
    return str(s).replace("|", "\\|")


def main():
    C.verify_pins()
    if not os.path.exists(C.rp(READOUT)):
        die("%s does not exist: the analysis runs after the readout" % READOUT)
    if C.md5_file(D14_ANALYSIS[0]) != D14_ANALYSIS[1]:
        die("%s MD5 differs from the pinned value" % D14_ANALYSIS[0])
    ro_text = open(C.rp(READOUT), encoding="utf-8").read()
    ro_lines = ro_text.split("\n")
    secs = TB.sections(ro_text)

    rows, rids, trans, tinfo, facts, finfo, recs, info, missing, led = R.load_all("FULL")
    gold = C.load_gold()
    meta = C.row_meta()
    ev = OrderedDict((arm, R.evaluate(arm, rr, gold)) for arm, rr in recs.items())
    r1_ok = OrderedDict((rid, trans["R1"][rid]["status"] == "ok") for rid in rids)
    m_status = OrderedDict((arm, OrderedDict((r["rid"], r["status"]) for r in recs[arm])) for arm in C.arms_of("M") if arm in recs)
    A = set(R.set_a(rids, r1_ok, m_status))
    checked = 0
    for key, keep in (("### set A (n = ", A), ("### all sampled rows (n = ", set(rids))):
        head, trows = TB.tables(TB.find(secs, key))[0]
        table = {r[0]: r[2] for r in trows}
        for arm, e in ev.items():
            s = R.summary(R.restrict(e, keep))
            want = "%s (%d)" % (R.pct(s["correct"], s["n"]), s["correct"])
            if table.get(arm) != want:
                die("PASS CHECK: %s %s recomputed %s, the readout has %s" % (key, arm, want, table.get(arm)))
            checked += 1
    print("[check] PASS: accuracy and correct count of %d arm tables (set A and all rows) equal the readout" % checked)

    D = []
    D.append("# D15 analysis pack (descriptive)")
    D.append("")
    D.append("Written by `%sd15_analysis.py` (MD5 `%s`) after the readout `%s` (MD5 `%s`). It decides nothing and attaches no wording; the readout is the record. "
             "PASS CHECK: the accuracy and the correct count of every arm on set A and on all sampled rows (%d values) recomputed here equal the readout."
             % (C.EFBPT, C.md5_file(C.EFBPT + "d15_analysis.py"), READOUT, C.md5_file(READOUT), checked))
    D.append("")
    D.append("b = first arm right and second wrong; p = exact two-sided McNemar. UrduMMLU rows are on set A (n = %d of %d sampled rows; 100 rows per subdomain, so the UrduMMLU "
             "accuracy is the mean over subjects, not the dataset-weighted accuracy)." % (len(A), len(rids)))
    D.append("")

    # ---- A
    D.append("## A. Reading effect over all five sets")
    D.append("")
    D.append("First arm = the model with the English translation; second arm = the same model on the native item. StrategyQA, CSQA, PIQA and GSM8K: the 12 lines of `%s` §A "
             "copied as they stand (their source column points into D13_ANALYSIS.md and D14_READOUT.md; StrategyQA eval458 with the D13 prompts; CSQA, PIQA, GSM8K all rows; "
             "these four Urdu sets were translated from English). UrduMMLU: native Urdu, set A, LA lines of `%s`. For T and M the D15 test (H1) compares with the English instruction "
             "and the Urdu item (`_ue`); the D14 form (against the Urdu prompt, `_ur`) is shown too." % (D14_ANALYSIS[0], READOUT))
    D.append("")
    D.append("| task | model | first arm vs second arm | n | acc first % | acc second % | delta pp | b | c | raw p | source line |")
    D.append("|---|---|---|---|---|---|---|---|---|---|---|")
    d14a = open(C.rp(D14_ANALYSIS[0]), encoding="utf-8").read()
    head, arows = TB.tables(TB.find(TB.sections(d14a), "## A. Reading effect"))[0]
    if head != ["task", "model", "first arm vs second arm", "n", "acc first %", "acc second %", "delta pp", "b", "c", "raw p", "source line"] or len(arows) != 12:
        die("%s §A: unexpected table (%d rows)" % (D14_ANALYSIS[0], len(arows)))
    for r in arows:
        D.append("| " + " | ".join(r) + " |")
    for model, a, b in D15_PAIRS:
        if a in missing or b in missing:
            D.append("| UrduMMLU | %s | %s vs %s | missing | - | - | - | - | - | - | - |" % (model, a, b))
            continue
        hits = [(i + 1, TB.cells(l)) for i, l in enumerate(ro_lines) if l.startswith("| set A | %s vs %s | LA" % (a, b)) and len(TB.cells(l)) == 10]
        if not hits:
            die("readout: no set-A LA line for %s vs %s" % (a, b))
        ln, c = hits[0]
        D.append("| UrduMMLU | %s | %s vs %s%s | %s | %s | %s | %s | %s | %s | %s | D15_READOUT.md:%d |"
                 % (model, a, b, " (D14 form)" if b.endswith("_ur") and a != "S_ro" else "", c[3], TB.split2(c[4])[0], TB.split2(c[5])[0], c[6], c[7], c[8], c[9], ln))
    D.append("")

    # ---- B
    head, srows = TB.tables(TB.find(secs, "## 10. Per subdomain"))[0]
    col = head.index("S_ro − S_ur")
    def dval(r):
        return float(r[col].replace("−", "-"))
    srt = sorted(srows, key=lambda r: (-dval(r), r[0]))
    D.append("## B. Subdomains by S_ro − S_ur (set A)")
    D.append("")
    D.append("Cells copied from `%s` §10 (delta on all rows of the subdomain in set A; about 100 rows each, so one subdomain delta moves by 1 pp per row). Sorted here; ties by name." % READOUT)
    D.append("")
    for title, part in (("10 largest", srt[:10]), ("10 smallest", list(reversed(srt[-10:])))):
        D.append("### %s" % title)
        D.append("")
        D.append("| " + " | ".join(head) + " |")
        D.append("|" + "|".join("---" for _ in head) + "|")
        for r in part:
            D.append("| " + " | ".join(r) + " |")
        D.append("")

    # ---- C
    D.append("## C. Rows where the reading changes the answer of Qwen3-14B (set A)")
    D.append("")
    D.append("Per group, 15 random rows (a new `random.Random(%d)` for each list; candidates in run order) where S_ro is right and S_ur wrong, and 15 where S_ro is wrong and S_ur right "
             "(all of them if fewer). Item = the Urdu item block; R1 = the R1 English translation; answers = the predictions read by the readout's scorer (empty = unparsed)." % SEED)
    D.append("")
    by = {arm: OrderedDict((e["rid"], e) for e in ev[arm]) for arm in ("S_ro", "S_ur")}
    item = OrderedDict((r["rid"], C.ur_block(r["item"])) for r in rows)
    for g in C.GROUPS:
        for lab, want_ro in (("S_ro right, S_ur wrong", True), ("S_ro wrong, S_ur right", False)):
            cands = [rid for rid in rids if rid in A and meta[rid]["group"] == g
                     and by["S_ro"][rid]["correct"] == want_ro and by["S_ur"][rid]["correct"] == (not want_ro)]
            pick = random.Random(SEED).sample(cands, N_EXAMPLES) if len(cands) > N_EXAMPLES else list(cands)
            pick = [rid for rid in rids if rid in set(pick)]
            D.append("### %s — %s (%d candidates, %d shown)" % (g, lab, len(cands), len(pick)))
            D.append("")
            D.append("| row | subdomain | Urdu item | R1 translation | gold | S_ro | S_ur |")
            D.append("|---|---|---|---|---|---|---|")
            for rid in pick:
                D.append("| %s | %s | %s | %s | %s | %s | %s |" % (rid, meta[rid]["subdomain"], cell(short(item[rid].replace("\n", " / "))),
                                                             cell(short(trans["R1"][rid]["block"].replace("\n", " / "))), gold[rid],
                                                             by["S_ro"][rid]["pred"] or "(unparsed)", by["S_ur"][rid]["pred"] or "(unparsed)"))
            D.append("")
    with open(C.rp(OUT_PATH), "w", encoding="utf-8") as f:
        f.write("\n".join(D) + "\n")
    print("[done] wrote %s (%d lines)" % (OUT_PATH, len(D)))


if __name__ == "__main__":
    main()
