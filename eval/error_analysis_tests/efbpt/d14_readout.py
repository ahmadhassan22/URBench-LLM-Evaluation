#!/usr/bin/env python3
"""
d14_readout.py — the one D14 readout (freeze F5, F6, F7, 9.11, 9.12).

  --self-test   no D14 arm file is opened: the scorers and this script's own row evaluation
                reproduce 190/300, 108/150 and 621/700 on the sealed fair outputs; hand-made
                Holm, McNemar, deciding-line, wording and prediction cases. Every check fatal.
  (no flag)     runs the self-test, then the readout. It refuses to run unless this script,
                d14_score.py and d14_common.py are committed, unchanged and on origin/main,
                every arm of every task is sealed (or is an API arm listed as missing because
                the API budget ended), and docs/handoff/D14_READOUT.md does not exist yet
                (the readout is run once).

Scoring: the task's extract_answer and gold rule from its fair script, unchanged, on the answer
text (after </think> for Qwen3-14B; the whole output otherwise); the EN variant for T_r1, T_og,
M_r1, M_og on PIQA and GSM8K. Empty prediction = unparsed = wrong; a reading_failed, refused
or too_long row is unparsed = wrong. Truncated = finish reason "length".

Lines: LA all rows; LP rows parsed in both arms; LT rows truncated in neither arm; LPT both.
Deciding line: LA; LP if the unparsed rates differ by more than 5.0 pp; LT if the truncated
rates differ by more than 5.0 pp; LPT if both. Exact two-sided McNemar, b = first arm right
and second wrong. Holm within a family (m = 9 for G1, m = 2 for G2).

Output: docs/handoff/D14_READOUT.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_common as C  # noqa: E402
import d14_score as SC  # noqa: E402
from d14_common import die  # noqa: E402

DOC = "docs/handoff/D14_READOUT.md"
ALPHA = 0.05
RULE_PP = 5.0
TASK_NAME = OrderedDict([("csqa", "CSQA"), ("piqa", "PIQA"), ("gsm8k", "GSM8K")])
LINES = ("LA", "LP", "LT", "LPT")
LINE_TEXT = {"LA": "all rows", "LP": "rows parsed in both arms", "LT": "rows truncated in neither arm",
             "LPT": "rows parsed in both arms and truncated in neither"}

G1 = [(t, a, b) for t in TASK_NAME for a, b in (("S_ro", "S_ur"), ("T_r1", "T_ur"), ("M_r1", "M_ur"))]     # F6: 9 tests
G2 = [(t, "S_tr", "S_ur") for t in ("csqa", "piqa")]                                                        # F6: 2 tests
DESCRIPTIVE = [(t, a, b) for t in TASK_NAME for a, b in (
    ("S_og", "S_ur"), ("T_og", "T_ur"), ("M_og", "M_ur"), ("S_og", "S_ro"), ("T_og", "T_r1"), ("M_og", "M_r1"),
    ("S_tr", "S_ro"), ("S_tr", "T_r1")) if not (a == "S_tr" and not C.TASKS[t]["has_tr"])]
PREDICTIONS = (      # F7, read on LA: (text, task, first, second, low, high)
    [("S_ro - S_ur in [0, +8]", "csqa", "S_ro", "S_ur", 0.0, 8.0), ("S_ro - S_ur in [-2, +8]", "piqa", "S_ro", "S_ur", -2.0, 8.0),
     ("S_ro - S_ur in [-2, +4]", "gsm8k", "S_ro", "S_ur", -2.0, 4.0)]
    + [("%s - %s in [0, +10]" % (a, b), t, a, b, 0.0, 10.0) for a, b in (("T_r1", "T_ur"), ("M_r1", "M_ur")) for t in TASK_NAME]
    + [("S_tr - S_ur in [-3, +6]", t, "S_tr", "S_ur", -3.0, 6.0) for t in ("csqa", "piqa")]
    + [("%s is at least %s minus 2" % (a, b), t, a, b, -2.0, None) for a, b in (("S_og", "S_ro"), ("T_og", "T_r1"), ("M_og", "M_r1")) for t in TASK_NAME])


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def exact_p(b, c):
    """Exact two-sided McNemar: binomial test on the discordant pairs; 1 when there is none."""
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0


def holm(ps, m):
    """D13 freeze §3 / D14 9.11: step-down on the raw p-values of a family of m tests. ps may hold None (a missing
    test); the available p-values are ranked among themselves and the i-th smallest gets the factor m - i + 1."""
    have = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: (ps[i], i))
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(have):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def wording(delta_rows, holm_p):
    """F6: 'helps' only if delta > 0 and Holm p < 0.05; 'hurts' only if delta < 0 and Holm p < 0.05."""
    if holm_p < ALPHA and delta_rows > 0:
        return "helps"
    if holm_p < ALPHA and delta_rows < 0:
        return "hurts"
    return "no clear effect"


def evaluate(task, arm, recs, gold):
    """One dict per row: rid, status, pred, unparsed, correct, truncated."""
    spec = C.ARMS[arm]
    out = []
    for r in recs:
        if r["status"] == "ok":
            pred = SC.extract(task, spec["scorer"], SC.answer_text(spec["model"], r["generation"]))
        else:
            pred = ""
        out.append(OrderedDict([("rid", r["rid"]), ("status", r["status"]), ("pred", pred), ("unparsed", pred == ""),
                                ("correct", pred != "" and pred == gold[r["rid"]]), ("truncated", bool(r.get("truncated")))]))
    return out


def summary(ev):
    n = len(ev)
    st = OrderedDict()
    for e in ev:
        st[e["status"]] = st.get(e["status"], 0) + 1
    return OrderedDict([("n", n), ("correct", sum(e["correct"] for e in ev)), ("unparsed", sum(e["unparsed"] for e in ev)),
                        ("truncated", sum(e["truncated"] for e in ev)), ("status", st)])


def line(x, y, keep):
    if len(x) != len(y):
        die("paired arms have different lengths")
    n = cx = cy = b = c = 0
    for ex, ey in zip(x, y):
        if ex["rid"] != ey["rid"]:
            die("paired rows are not aligned")
        if not keep(ex, ey):
            continue
        n += 1
        cx += ex["correct"]
        cy += ey["correct"]
        b += int(ex["correct"] and not ey["correct"])
        c += int(ey["correct"] and not ex["correct"])
    return OrderedDict([("n", n), ("cx", cx), ("cy", cy), ("b", b), ("c", c), ("p", exact_p(b, c))])


def compare(x, y):
    """The four lines and the deciding line of one comparison (first arm x, second arm y)."""
    n = len(x)
    L = OrderedDict([
        ("LA", line(x, y, lambda a, b: True)),
        ("LP", line(x, y, lambda a, b: not a["unparsed"] and not b["unparsed"])),
        ("LT", line(x, y, lambda a, b: not a["truncated"] and not b["truncated"])),
        ("LPT", line(x, y, lambda a, b: not a["unparsed"] and not b["unparsed"] and not a["truncated"] and not b["truncated"]))])
    ux, uy = sum(e["unparsed"] for e in x), sum(e["unparsed"] for e in y)
    tx, ty = sum(e["truncated"] for e in x), sum(e["truncated"] for e in y)
    fire_u = abs(ux - uy) * 100.0 > RULE_PP * n
    fire_t = abs(tx - ty) * 100.0 > RULE_PP * n
    dec = "LPT" if (fire_u and fire_t) else "LP" if fire_u else "LT" if fire_t else "LA"
    return OrderedDict([("lines", L), ("deciding", dec), ("unparsed", (ux, uy)), ("truncated", (tx, ty)), ("n", n)])


def pct(k, n):
    return "%.2f" % (100.0 * k / n) if n else "n/a"


def delta_pp(l):
    return "%+.2f" % (100.0 * (l["cx"] - l["cy"]) / l["n"]) if l["n"] else "n/a"


def fmt_p(p):
    if p is None:
        return "n/a"
    return "%.4f" % p if p >= 0.0001 else "%.2e" % p


# ----------------------------------------------------------------------------
# Self-test (opens no D14 arm file)
# ----------------------------------------------------------------------------
def self_test():
    out = []
    # S1: the scorers and evaluate() on the sealed fair outputs
    parts = []
    for task, t in C.TASKS.items():
        c, k, pd, gd = SC.score_fair(task)
        sealed = C.read_jsonl(t["sealed"])
        lo, hi = t["eval_range"]
        gold_all, _ = SC.load_gold(task)
        recs = [OrderedDict([("rid", "%s_%04d" % (task, lo + i)), ("status", "ok"), ("generation", r["generated"]), ("truncated", False)])
                for i, r in enumerate(sealed)]
        s = summary(evaluate(task, "S_ur", recs, gold_all))
        if (c, k) != (t["sealed_correct"], hi - lo) or pd or gd or s["correct"] != t["sealed_correct"] or s["n"] != hi - lo:
            die("self-test S1: %s gives %d/%d (scorer) and %d/%d (row evaluation), expected %d/%d" % (task, c, k, s["correct"], s["n"], t["sealed_correct"], hi - lo))
        parts.append("%s %d/%d" % (TASK_NAME[task], c, k))
    out.append("S1 PASS: the task scorers and this script's row evaluation reproduce the sealed fair outputs (%s); the recomputed prediction equals the stored one on every row." % ", ".join(parts))
    n = SC.self_check_en()
    out.append("S2 PASS: %d hand-made scorer cases (the EN marker for PIQA and GSM8K, the unchanged functions, CSQA)." % n)

    # S3: Holm
    def close(a, b):
        return all((x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-12) for x, y in zip(a, b))
    cases = [(([0.01, 0.04, 0.03], 3), [0.03, 0.06, 0.06]), (([0.5, 0.6], 2), [1.0, 1.0]), (([0.02, 0.02], 2), [0.04, 0.04]),
             (([0.001, 0.02], 9), [0.009, 0.16]), (([0.04, None, 0.001], 3), [0.08, None, 0.003]), (([0.03, 0.001, 0.02], 3), [0.04, 0.003, 0.04]),
             (([], 2), [])]
    for (ps, m), want in cases:
        if not close(holm(ps, m), want):
            die("self-test S3: holm(%s, %d) = %s, expected %s" % (ps, m, holm(ps, m), want))
    out.append("S3 PASS: Holm step-down, %d hand-made cases (running maximum, the cap at 1, ties, a family larger than the tests present, a missing test, input order kept)." % len(cases))
    # S4: McNemar
    for (b, c), want in (((5, 0), 0.0625), ((0, 0), 1.0), ((10, 2), 158.0 / 4096.0), ((3, 3), 1.0), ((0, 6), 0.03125)):
        if abs(exact_p(b, c) - want) > 1e-12:
            die("self-test S4: exact_p(%d, %d) = %r, expected %r" % (b, c, exact_p(b, c), want))
    out.append("S4 PASS: exact two-sided McNemar, 5 hand-made cases.")

    # S5: lines, deciding line, wording, predictions on hand-made rows
    def rows(n, wrong=(), unparsed=(), truncated=()):
        return [OrderedDict([("rid", "r%03d" % i), ("status", "ok"), ("pred", "" if i in unparsed else "x"), ("unparsed", i in unparsed),
                             ("correct", i not in wrong and i not in unparsed), ("truncated", i in truncated)]) for i in range(n)]
    x = rows(100, wrong=range(0, 10))                                   # 90 right
    y = rows(100, wrong=range(5, 25))                                   # 80 right
    cp = compare(x, y)
    la = cp["lines"]["LA"]
    if (la["n"], la["cx"], la["cy"], la["b"], la["c"], cp["deciding"]) != (100, 90, 80, 15, 5, "LA") or abs(la["p"] - exact_p(15, 5)) > 1e-15:
        die("self-test S5: LA counts are wrong: %s" % dict(la))
    if delta_pp(la) != "+10.00" or pct(la["cx"], la["n"]) != "90.00":
        die("self-test S5: formatting")
    y5 = rows(100, wrong=range(5, 25), unparsed=range(95, 100))         # unparsed rates differ by exactly 5.0 pp: LA
    y6 = rows(100, wrong=range(5, 25), unparsed=range(94, 100))         # 6.0 pp: LP
    t6 = rows(100, wrong=range(5, 25), truncated=range(0, 6))           # truncated 6.0 pp: LT
    b6 = rows(100, wrong=range(5, 25), unparsed=range(94, 100), truncated=range(0, 6))
    got = [compare(x, z)["deciding"] for z in (y5, y6, t6, b6)]
    if got != ["LA", "LP", "LT", "LPT"]:
        die("self-test S5: deciding lines %s" % got)
    lp = compare(x, y6)["lines"]["LP"]
    if (lp["n"], lp["cx"], lp["cy"]) != (94, 84, 74):
        die("self-test S5: LP counts %s" % dict(lp))
    lt = compare(x, t6)["lines"]["LT"]
    if (lt["n"], lt["cx"], lt["cy"]) != (94, 90, 75):
        die("self-test S5: LT counts %s" % dict(lt))
    lpt = compare(x, b6)["lines"]["LPT"]
    if (lpt["n"], lpt["cx"], lpt["cy"]) != (88, 84, 69):
        die("self-test S5: LPT counts %s" % dict(lpt))
    gold = {"csqa_0000": "B", "csqa_0001": "C", "csqa_0002": "A", "csqa_0003": "D"}
    recs = [OrderedDict([("rid", "csqa_0000"), ("status", "ok"), ("generation", "<think>A</think>\nAnswer: B"), ("truncated", False)]),
            OrderedDict([("rid", "csqa_0001"), ("status", "reading_failed"), ("generation", ""), ("truncated", False)]),
            OrderedDict([("rid", "csqa_0002"), ("status", "ok"), ("generation", "<think>Answer: A"), ("truncated", True)]),
            OrderedDict([("rid", "csqa_0003"), ("status", "ok"), ("generation", "<think>x</think>no letter"), ("truncated", False)])]
    ev = evaluate("csqa", "S_ro", recs, gold)
    if [(e["pred"], e["unparsed"], e["correct"], e["truncated"]) for e in ev] != [("B", False, True, False), ("", True, False, False), ("A", False, True, True), ("", True, False, False)]:
        die("self-test S5: row evaluation %s" % [dict(e) for e in ev])
    ev_m = evaluate("piqa", "M_r1", [OrderedDict([("rid", "p0"), ("status", "refused"), ("generation", ""), ("truncated", False)]),
                                     OrderedDict([("rid", "p1"), ("status", "ok"), ("generation", "Solution 1 is 2 cups. Final answer: 0"), ("truncated", False)])],
                    {"p0": "1", "p1": "0"})
    if [(e["pred"], e["correct"]) for e in ev_m] != [("", False), ("0", True)]:
        die("self-test S5: EN-arm evaluation %s" % [dict(e) for e in ev_m])
    w = [wording(5, 0.01), wording(-5, 0.01), wording(5, 0.05), wording(0, 0.001), wording(-5, 0.2)]
    if w != ["helps", "hurts", "no clear effect", "no clear effect", "no clear effect"]:
        die("self-test S5: wording %s" % w)
    if [pred_mark(0.0, 0.0, 8.0), pred_mark(8.0, 0.0, 8.0), pred_mark(-0.07, 0.0, 8.0), pred_mark(-2.0, -2.0, None), pred_mark(-2.13, -2.0, None)] != ["hit", "hit", "miss", "hit", "miss"]:
        die("self-test S5: prediction marks")
    out.append("S5 PASS: hand-made rows: LA / LP / LT / LPT counts and p, the deciding-line rule at and above 5.0 pp for unparsed and for truncated, "
               "a reading_failed row and a refused row (unparsed, wrong), the answer after </think>, the EN marker on an API arm, the wording rules, the prediction marks.")
    return out


def pred_mark(delta, low, high):
    return "hit" if (delta >= low and (high is None or delta <= high)) else "miss"


# ----------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------
def git(*a):
    return subprocess.check_output(("git",) + a, cwd=C.ROOT).decode().strip()


def require_committed_and_pushed():
    files = [C.EFBPT + n for n in ("d14_readout.py", "d14_score.py", "d14_common.py")]
    dirty = git("status", "--porcelain", "--", *files)
    if dirty:
        die("the readout scripts are not committed and clean: " + dirty.replace("\n", "; "))
    for f in files:
        last = git("log", "-1", "--format=%H", "--", f)
        if not last:
            die("%s is not committed" % f)
        rc = subprocess.call(["git", "merge-base", "--is-ancestor", last, "origin/main"], cwd=C.ROOT)
        if rc != 0:
            die("%s: its last commit %s is not on origin/main (push first)" % (f, last[:7]))
    return git("rev-parse", "HEAD")


def load_task(task):
    """({arm: records}, {arm: seal info}, [missing arms]) of one task; fatal if an arm is neither sealed nor a
    missing API arm after an API stop."""
    led = C.ledger_read() if os.path.exists(C.rp(C.LEDGER)) else None
    recs, info, missing = OrderedDict(), OrderedDict(), []
    rids = [r["rid"] for r in C.load_rows(task)]
    for arm in C.arms_of(task):
        pth = C.arm_paths(task, arm, "FULL")
        if not C.is_sealed(pth):
            if C.ARMS[arm]["model"] == "M" and led is not None and led["stopped"]:
                missing.append(arm)
                continue
            die("%s %s is not sealed (%s); nothing is scored before every arm of a task is sealed" % (task, arm, pth["seal"]))
        seal = json.load(open(C.rp(pth["seal"]), encoding="utf-8"), object_pairs_hook=OrderedDict)
        if seal.get("mode") != "FULL" or seal.get("scored") is not False:
            die("%s: not a FULL, unscored seal" % pth["seal"])
        rr = C.read_rows(pth["rows"])
        if [r["rid"] for r in rr] != rids:
            die("%s: rows are not the %d rows of the task in order" % (pth["rows"], len(rids)))
        recs[arm] = rr
        info[arm] = OrderedDict([("file", pth["rows"]), ("md5", C.md5_file(pth["rows"])), ("seal", pth["seal"]), ("seal_md5", C.md5_file(pth["seal"])),
                                 ("slurm", seal.get("slurm_job_id")), ("end_time", seal.get("end_time")), ("seal_obj", seal)])
    return recs, info, missing


# ----------------------------------------------------------------------------
# The document
# ----------------------------------------------------------------------------
def comparison_rows(task, a, b, cp):
    rows = []
    for k in LINES:
        l = cp["lines"][k]
        rows.append("| %s | %s vs %s | %s%s | %d | %s (%d) | %s (%d) | %s | %d | %d | %s |"
                    % (TASK_NAME[task], a, b, k, " (deciding)" if k == cp["deciding"] else "", l["n"], pct(l["cx"], l["n"]), l["cx"],
                       pct(l["cy"], l["n"]), l["cy"], delta_pp(l), l["b"], l["c"], fmt_p(l["p"])))
    return rows


def family_section(name, tests, m, ev, missing):
    cps, ps = [], []
    for task, a, b in tests:
        if a in missing[task] or b in missing[task]:
            cps.append(None)
            ps.append(None)
        else:
            cp = compare(ev[task][a], ev[task][b])
            cps.append(cp)
            ps.append(cp["lines"][cp["deciding"]]["p"])
    adj = holm(ps, m)
    out = ["| task | test | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p | Holm p | wording |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for (task, a, b), cp, hp in zip(tests, cps, adj):
        if cp is None:
            out.append("| %s | %s vs %s | missing | - | - | - | - | - | - | - | - | missing (an arm was not run) |" % (TASK_NAME[task], a, b))
            verdicts.append((task, a, b, None))
            continue
        l = cp["lines"][cp["deciding"]]
        w = wording(l["cx"] - l["cy"], hp)
        verdicts.append((task, a, b, w))
        out.append("| %s | %s vs %s | %s | %d | %s (%d) | %s (%d) | %s | %d | %d | %s | %s | %s |"
                   % (TASK_NAME[task], a, b, cp["deciding"], l["n"], pct(l["cx"], l["n"]), l["cx"], pct(l["cy"], l["n"]), l["cy"],
                      delta_pp(l), l["b"], l["c"], fmt_p(l["p"]), fmt_p(hp), w))
    out += ["", "Every line of every %s test (unparsed and truncated counts are on all rows, first arm / second arm):" % name, "",
            "| task | test | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |", "|---|---|---|---|---|---|---|---|---|---|"]
    notes = []
    for (task, a, b), cp in zip(tests, cps):
        if cp is None:
            continue
        out += comparison_rows(task, a, b, cp)
        notes.append("- %s %s vs %s: unparsed %d / %d (%s%% / %s%%), truncated %d / %d (%s%% / %s%%); deciding line %s (%s)."
                     % (TASK_NAME[task], a, b, cp["unparsed"][0], cp["unparsed"][1], pct(cp["unparsed"][0], cp["n"]), pct(cp["unparsed"][1], cp["n"]),
                        cp["truncated"][0], cp["truncated"][1], pct(cp["truncated"][0], cp["n"]), pct(cp["truncated"][1], cp["n"]),
                        cp["deciding"], LINE_TEXT[cp["deciding"]]))
    return out + [""] + notes, verdicts


def build_doc(self_lines, head):
    gold, gold_diff, ev, recs, info, missing = {}, {}, OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for task in C.TASKS:
        gold[task], gold_diff[task] = SC.load_gold(task)
        recs[task], info[task], missing[task] = load_task(task)
        ev[task] = OrderedDict((arm, evaluate(task, arm, rr, gold[task])) for arm, rr in recs[task].items())
    D = []
    D.append("# D14 readout — reading effect and teacher facts on CSQA, PIQA and GSM8K")
    D.append("")
    D.append("Written by `%sd14_readout.py` (MD5 `%s`, git HEAD `%s`), one run, %s. Freeze: `%s` (MD5 `%s`). Every arm file was sealed before this script opened it; "
             "no D14 arm was scored before this run. No arm is run again after this readout."
             % (C.EFBPT, C.md5_file(C.EFBPT + "d14_readout.py"), head, time.strftime("%Y-%m-%d %H:%M:%S"), C.FREEZE_PATH, C.FREEZE_MD5))
    D.append("")
    D.append("Scoring (freeze F5, 9.11): the task's `extract_answer` and gold rule from its fair script, unchanged, on the answer text (after `</think>` for Qwen3-14B; the whole output otherwise); "
             "for T_r1, T_og, M_r1, M_og on PIQA and GSM8K the same function with \"Final answer\" in place of the Urdu marker. Empty prediction = unparsed = wrong. "
             "A reading_failed, refused or too_long row has no answer: unparsed = wrong. Truncated = reached max_tokens.")
    D.append("")
    D.append("Arms: S = Qwen3-14B (student), T = Llama-3.1-70B-AWQ (teacher), M = qwen3.8-max (commercial). `_ur` = Urdu prompt; S_ro = Urdu prompt plus the R1 English translation; "
             "S_og = Urdu prompt plus the original English item; S_tr = Urdu prompt plus teacher facts; T_r1, M_r1 = English prompt with the R1 translation; T_og, M_og = English prompt with the original English item.")
    D.append("")
    D.append("## 0. Self-test (run before any D14 arm was opened; every check fatal)")
    D.append("")
    D += ["- " + s for s in self_lines]
    D.append("")

    D.append("## 1. Sealed files")
    D.append("")
    D.append("| task | arm | file | MD5 | rows | row status | seal |")
    D.append("|---|---|---|---|---|---|---|")
    for task in C.TASKS:
        for arm in C.arms_of(task):
            if arm in missing[task]:
                D.append("| %s | %s | - | - | - | - | MISSING: not sealed, the API budget ended |" % (TASK_NAME[task], arm))
                continue
            i = info[task][arm]
            st = i["seal_obj"]["status_counts"]
            D.append("| %s | %s | `%s` | `%s` | %d | %s | %s (md5 %s; %s; %s) |"
                     % (TASK_NAME[task], arm, i["file"], i["md5"], i["seal_obj"]["n_rows"], ", ".join("%s %d" % kv for kv in st.items()),
                        i["seal"], i["seal_md5"], ("slurm %s" % i["slurm"]) if i["slurm"] else "login node, no slurm job", i["end_time"]))
    D.append("")
    miss_all = [(t, a) for t in C.TASKS for a in missing[t]]
    D.append("Missing arms: %s." % (", ".join("%s %s" % (TASK_NAME[t], a) for t, a in miss_all) if miss_all else "none; all 29 arm files are sealed"))
    D.append("")

    D.append("## 2. Per arm")
    D.append("")
    for task in C.TASKS:
        n = C.TASKS[task]["n"]
        D.append("### %s (n = %d)" % (TASK_NAME[task], n))
        D.append("")
        D.append("| arm | acc % (correct) | unparsed (%) | truncated (%) | reading_failed | refused | too_long |")
        D.append("|---|---|---|---|---|---|---|")
        for arm, e in ev[task].items():
            s = summary(e)
            D.append("| %s | %s (%d) | %d (%s) | %d (%s) | %d | %d | %d |"
                     % (arm, pct(s["correct"], s["n"]), s["correct"], s["unparsed"], pct(s["unparsed"], s["n"]), s["truncated"], pct(s["truncated"], s["n"]),
                        s["status"].get("reading_failed", 0), s["status"].get("refused", 0), s["status"].get("too_long", 0)))
        D.append("")

    D.append("## 3. Family G1 — reading effect (9 tests, Holm at 0.05)")
    D.append("")
    D.append("First arm = the model with the English translation R1; second arm = the same model on the Urdu item. Delta = first minus second on the deciding line. "
             "Wording (freeze F6): \"helps\" only if delta > 0 and Holm p < 0.05; \"hurts\" only if delta < 0 and Holm p < 0.05; otherwise \"no clear effect\".")
    D.append("")
    sec, v1 = family_section("G1", G1, 9, ev, missing)
    D += sec
    D.append("")
    D.append("G1 wording:")
    D.append("")
    for task, a, b, w in v1:
        D.append("- %s, %s vs %s: %s." % (TASK_NAME[task], a, b, ("the English translation: " + w) if w else "missing"))
    D.append("")

    D.append("## 4. Family G2 — method (2 tests, Holm at 0.05)")
    D.append("")
    D.append("First arm = S_tr (Qwen3-14B with teacher facts); second arm = S_ur (Qwen3-14B on the Urdu item alone). Same wording rule.")
    D.append("")
    sec, v2 = family_section("G2", G2, 2, ev, missing)
    D += sec
    D.append("")
    D.append("G2 wording:")
    D.append("")
    for task, a, b, w in v2:
        D.append("- %s, %s vs %s: %s." % (TASK_NAME[task], a, b, ("the teacher facts: " + w) if w else "missing"))
    D.append("")

    D.append("## 5. Descriptive comparisons (no claim, raw p only)")
    D.append("")
    D.append("| task | comparison | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |")
    D.append("|---|---|---|---|---|---|---|---|---|---|")
    for task, a, b in DESCRIPTIVE:
        if a in missing[task] or b in missing[task]:
            D.append("| %s | %s vs %s | missing | - | - | - | - | - | - | - |" % (TASK_NAME[task], a, b))
            continue
        D += comparison_rows(task, a, b, compare(ev[task][a], ev[task][b]))
    D.append("")

    D.append("## 6. Predictions (freeze F7, read on LA = all rows)")
    D.append("")
    D.append("| prediction | task | first % | second % | delta pp | result |")
    D.append("|---|---|---|---|---|---|")
    hits = total = 0
    for text, task, a, b, lo, hi in PREDICTIONS:
        if a in missing[task] or b in missing[task]:
            D.append("| %s | %s | - | - | - | missing |" % (text, TASK_NAME[task]))
            continue
        sa, sb = summary(ev[task][a]), summary(ev[task][b])
        d = 100.0 * (sa["correct"] - sb["correct"]) / sa["n"]
        mark = pred_mark(d, lo, hi)
        hits += mark == "hit"
        total += 1
        D.append("| %s | %s | %s | %s | %+.2f | %s |" % (text, TASK_NAME[task], pct(sa["correct"], sa["n"]), pct(sb["correct"], sb["n"]), d, mark))
    D.append("")
    D.append("Predictions hit: %d of %d." % (hits, total))
    D.append("")

    D.append("## 7. S_ur against the sealed fair baseline on the shared rows (agreement, reported only)")
    D.append("")
    D.append("The sealed fair outputs are the evaluation splits: the last rows of the CSQA and PIQA source files and all GSM8K rows. S_ur uses the same prompt, engine arguments and sampling; "
             "the rows were generated in other batches, so the outputs need not be identical.")
    D.append("")
    D.append("| task | shared rows | sealed fair acc % (correct) | S_ur acc % (correct) on these rows | both right | fair only | S_ur only | both wrong | same prediction |")
    D.append("|---|---|---|---|---|---|---|---|---|")
    for task, t in C.TASKS.items():
        sealed = C.read_jsonl(t["sealed"])
        lo, hi = t["eval_range"]
        e = ev[task]["S_ur"][lo:hi]
        fc = [bool(r["correct"]) for r in sealed]
        sc = [bool(x["correct"]) for x in e]
        same = sum(1 for r, x in zip(sealed, e) if r["pred"] == x["pred"])
        D.append("| %s | %d | %s (%d) | %s (%d) | %d | %d | %d | %d | %d |"
                 % (TASK_NAME[task], hi - lo, pct(sum(fc), len(fc)), sum(fc), pct(sum(sc), len(sc)), sum(sc),
                    sum(1 for a, b in zip(fc, sc) if a and b), sum(1 for a, b in zip(fc, sc) if a and not b),
                    sum(1 for a, b in zip(fc, sc) if b and not a), sum(1 for a, b in zip(fc, sc) if not a and not b), same))
    D.append("")

    D.append("## 8. Readings and teacher facts")
    D.append("")
    D.append("| task | rows | R1 file MD5 | requests | reading_failed | accepted readings that reached max_tokens |")
    D.append("|---|---|---|---|---|---|")
    for task in C.TASKS:
        pth = C.r1_paths(task, "FULL")
        if not C.is_sealed(pth):
            die("the R1 file of %s is not sealed" % task)
        s = json.load(open(C.rp(pth["seal"]), encoding="utf-8"))
        D.append("| %s | %d | `%s` | %d | %d | %d |" % (TASK_NAME[task], s["n_rows"], C.md5_file(pth["rows"]), s["requests"], s["reading_failed"],
                                                     s["accepted_readings_that_reached_max_tokens"]))
    D.append("")
    D.append("A reading_failed row is not generated in S_ro, S_tr, T_r1 and M_r1 and counts as unparsed = wrong there.")
    D.append("")
    D.append("| task | rows | facts file MD5 | replies parsed | facts kept | lines removed by the D10 verdict filter | lines removed by the D14 filter | rows with a line removed by the D14 filter | rows without facts |")
    D.append("|---|---|---|---|---|---|---|---|---|")
    for task in C.TASKS:
        if not C.TASKS[task]["has_tr"]:
            continue
        pth = C.facts_paths(task, "FULL")
        if not C.is_sealed(pth):
            die("the facts file of %s is not sealed" % task)
        s = json.load(open(C.rp(pth["seal"]), encoding="utf-8"))
        D.append("| %s | %d | `%s` | %d | %d | %d | %d | %d | %d |" % (TASK_NAME[task], s["n_rows"], C.md5_file(pth["rows"]), s["parse_ok"], s["facts_total"],
                                                                 s["removed_verdict_lines"], s["removed_d14_lines"], s["rows_with_a_d14_removed_line"], s["rows_without_facts"]))
    D.append("")
    D.append("A row without facts has no facts block: its S_tr prompt equals its S_ur prompt (freeze 9.6).")
    D.append("")
    D.append("Gold: taken from the Urdu source row with the fair script's rule. Positions where the English source row gives another gold: %s."
             % "; ".join("%s %d" % (TASK_NAME[t], len(gold_diff[t])) for t in C.TASKS))
    D.append("")

    D.append("## 9. Seconds and tokens")
    D.append("")
    D.append("Seconds of a vLLM row = the seconds of its chunk divided by the rows generated in the chunk (up to 64 prompts per call); seconds of an API row = the latency of its answered request.")
    D.append("")
    D.append("| task | arm | rows generated | seconds total | seconds per row | prompt tokens per row | new tokens per row |")
    D.append("|---|---|---|---|---|---|---|")
    for task in C.TASKS:
        for arm, rr in recs[task].items():
            g = [r for r in rr if r["status"] == "ok"]
            secs = [r["seconds"] for r in g if isinstance(r.get("seconds"), (int, float))]
            pt = [r["n_prompt_tokens"] for r in g if isinstance(r.get("n_prompt_tokens"), (int, float))]
            gt = [r["n_gen_tokens"] for r in g if isinstance(r.get("n_gen_tokens"), (int, float))]
            D.append("| %s | %s | %d | %.0f | %.2f | %.1f | %.1f |" % (TASK_NAME[task], arm, len(g), sum(secs), sum(secs) / len(secs) if secs else 0.0,
                                                                   sum(pt) / len(pt) if pt else 0.0, sum(gt) / len(gt) if gt else 0.0))
    D.append("")
    if os.path.exists(C.rp(C.LEDGER)):
        led = C.ledger_read()
        D.append("API ledger at the time of the readout (TEST-5 and retries included): %d calls of at most %d; %d tokens of at most %d (prompt %d, completion %d); stopped: %s."
                 % (led["calls"], led["cap_calls"], led["tokens"], led["cap_tokens"], led["prompt_tokens"], led["completion_tokens"], led["stopped"] or "no"))
        D.append("")
    return "\n".join(D) + "\n", v1, v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="run the self-test only; no D14 arm file is opened")
    args = ap.parse_args()
    print("=" * 78)
    print("D14 READOUT%s" % (" — SELF-TEST ONLY (no D14 arm file is opened)" if args.self_test else ""))
    print("=" * 78, flush=True)
    C.verify_pins()
    lines = self_test()
    for s in lines:
        print("[self-test] " + s)
    if args.self_test:
        print("[self-test] PASS")
        return
    if os.path.exists(C.rp(DOC)):
        die("%s exists: the readout is run once" % DOC)
    head = require_committed_and_pushed()
    print("[ok] d14_readout.py, d14_score.py and d14_common.py are committed, unchanged and on origin/main (HEAD %s)" % head, flush=True)
    text, v1, v2 = build_doc(lines, head)
    with open(C.rp(DOC), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("[done] wrote %s (%d lines)" % (DOC, text.count("\n")))


if __name__ == "__main__":
    main()
