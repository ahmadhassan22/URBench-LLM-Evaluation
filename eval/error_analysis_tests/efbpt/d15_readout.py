#!/usr/bin/env python3
"""
d15_readout.py — the one D15 readout (freeze N5-N9, 11.10-11.12).

  --self-test   no D15 arm, translation or facts file is opened: this script's row evaluation with
                the unrestricted class [A-E] reproduces 190 / 300 on the sealed CSQA fair output;
                hand-made scorer, Holm, McNemar, line, deciding-line, set-A, wording, prediction
                and N9-reading cases. Every check fatal.
  (no flag)     runs the self-test, then the readout. It refuses to run unless this script,
                d15_score.py and d15_common.py are committed, unchanged and on origin/main; every
                arm, the three translations and the teacher facts are sealed (an M arm may be
                missing only if the API ledger shows a stop); and docs/handoff/D15_READOUT.md does
                not exist yet (the readout is run once).

Scoring: the CSQA fair extract_answer with the class [A-D] (d15_score) on the answer text (after
</think> for Qwen3-14B; the whole output otherwise). Empty = unparsed = wrong; a reading_failed,
refused or too_long row is unparsed = wrong. Truncated = finish reason "length".

Set A: sampled rows with a valid R1 translation and an answered (not refused) record in M_ur,
M_ue and M_r1. Every test is decided on set A with the D14 F6 deciding-line rule (LA; LP / LT /
LPT when the unparsed / truncated rates differ by more than 5.0 pp); exact two-sided McNemar,
b = first arm right and second wrong; Holm inside each family (m = 3 H1, 2 H2, 1 H3). The
statistics functions are those of d14_readout (imported, unchanged).

Output: docs/handoff/D15_READOUT.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d15_common as C  # noqa: E402
import d15_score as SC  # noqa: E402
import d14_readout as DR  # noqa: E402  (exact_p, holm, wording, line, compare, pct, delta_pp, fmt_p, pred_mark)
from d15_common import die  # noqa: E402

DOC = "docs/handoff/D15_READOUT.md"
LINES = DR.LINES
LINE_TEXT = DR.LINE_TEXT
MODEL_NAME = {"S": "Qwen3-14B", "T": "Llama-3.1-70B-AWQ", "M": "qwen3.8-max"}

H1 = [("S_ro", "S_ur"), ("T_r1", "T_ue"), ("M_r1", "M_ue")]          # N7: 3 tests
H2 = [("S_self", "S_ur"), ("T_self", "T_ue")]                         # N7: 2 tests
H3 = [("S_tr", "S_ur")]                                               # N7: 1 test
LOCAL = {"S_self": "SL", "T_self": "TL"}                              # N6: the arm's local translation
DESCRIPTIVE = [("T_ue", "T_ur"), ("M_ue", "M_ur"), ("T_r1", "T_ur"), ("M_r1", "M_ur"), ("S_tr", "S_ro"), ("S_self", "S_ro"), ("T_self", "T_r1")]
SPLIT_PAIRS = H1 + H2
SUBDOMAIN_PAIRS = H1 + H2 + H3
PREDICTIONS = [   # N8, read on LA of set A: (text, first, second, low, high)
    ("S_ro - S_ur in [-2, +6]", "S_ro", "S_ur", -2.0, 6.0),
    ("T_r1 - T_ue in [-2, +8]", "T_r1", "T_ue", -2.0, 8.0),
    ("M_r1 - M_ue in [-4, +3]", "M_r1", "M_ue", -4.0, 3.0),
    ("S_self - S_ur in [-4, +4]", "S_self", "S_ur", -4.0, 4.0),
    ("T_self - T_ue in [-3, +6]", "T_self", "T_ue", -3.0, 6.0),
    ("S_tr - S_ur in [-2, +6]", "S_tr", "S_ur", -2.0, 6.0),
]

exact_p, holm, wording, compare, pct, delta_pp, fmt_p, pred_mark = (DR.exact_p, DR.holm, DR.wording, DR.compare, DR.pct, DR.delta_pp,
                                                                     DR.fmt_p, DR.pred_mark)


# ----------------------------------------------------------------------------
# Row evaluation, set A, readings
# ----------------------------------------------------------------------------
def evaluate(arm, recs, gold, fn=None):
    """One dict per row: rid, status, pred, unparsed, correct, truncated. fn = the extractor (default: [A-D])."""
    model = C.ARMS[arm]["model"] if arm in C.ARMS else "S"
    ex = fn or SC.extractor(SC.CLASS_D15)
    out = []
    for r in recs:
        pred = ex(SC.answer_text(model, r["generation"])) if r["status"] == "ok" else ""
        out.append(OrderedDict([("rid", r["rid"]), ("status", r["status"]), ("pred", pred), ("unparsed", pred == ""),
                                ("correct", pred != "" and pred == gold[r["rid"]]), ("truncated", bool(r.get("truncated")))]))
    return out


def summary(ev):
    st = Counter(e["status"] for e in ev)
    return OrderedDict([("n", len(ev)), ("correct", sum(e["correct"] for e in ev)), ("unparsed", sum(e["unparsed"] for e in ev)),
                        ("truncated", sum(e["truncated"] for e in ev)), ("status", st)])


def set_a(rids, r1_ok, m_status):
    """N6: rows with a valid R1 reading and status 'ok' in every sealed M arm (m_status = {arm: {rid: status}})."""
    return [rid for rid in rids if r1_ok[rid] and all(ms[rid] == "ok" for ms in m_status.values())]


def restrict(ev, keep):
    return [e for e in ev if e["rid"] in keep]


def reading_n9(words, need):
    """N9: 'holds' if at least `need` tests say helps; 'not shown' if none; otherwise 'mixed'."""
    h = sum(1 for w in words if w == "helps")
    return "holds" if h >= need else "not shown" if h == 0 else "mixed"


# ----------------------------------------------------------------------------
# Self-test (opens no D15 arm, translation or facts file)
# ----------------------------------------------------------------------------
def self_test():
    out = []
    sealed = C.C14.read_jsonl(C.FAIR_SEALED)
    recs = [OrderedDict([("rid", "f%03d" % i), ("status", "ok"), ("generation", r["generated"]), ("truncated", False)]) for i, r in enumerate(sealed)]
    gold = {"f%03d" % i: r["gold"] for i, r in enumerate(sealed)}
    s = summary(evaluate("S_ur", recs, gold, SC.extractor(SC.CLASS_FAIR)))
    c, n, d = SC.score_fair()
    if (s["correct"], s["n"]) != (C.FAIR_SEALED_CORRECT, 300) or (c, n, d) != (C.FAIR_SEALED_CORRECT, 300, 0):
        die("self-test S1: %d / %d (row evaluation), %d / %d (scorer, %d preds differ); expected %d / 300" % (s["correct"], s["n"], c, n, d, C.FAIR_SEALED_CORRECT))
    out.append("S1 PASS: with the unrestricted class [A-E], this script's row evaluation and the scorer reproduce %d / 300 on the sealed CSQA fair output; "
               "the recomputed prediction equals the stored one on every row." % C.FAIR_SEALED_CORRECT)
    out.append("S2 PASS: %d hand-made scorer cases for the class [A-D] (d15_score.self_check)." % SC.self_check())

    def close(a, b):
        return all((x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-12) for x, y in zip(a, b))
    cases = [(([0.01, 0.04, 0.03], 3), [0.03, 0.06, 0.06]), (([0.5, 0.6], 2), [1.0, 1.0]), (([0.02, 0.02], 2), [0.04, 0.04]),
             (([0.04, None, 0.001], 3), [0.08, None, 0.003]), (([0.03, 0.001, 0.02], 3), [0.04, 0.003, 0.04]), (([0.03], 1), [0.03]),
             (([0.6], 1), [0.6]), (([None, 0.01], 2), [None, 0.02])]
    for (ps, m), want in cases:
        if not close(holm(ps, m), want):
            die("self-test S3: holm(%s, %d) = %s, expected %s" % (ps, m, holm(ps, m), want))
    out.append("S3 PASS: Holm step-down, %d hand-made cases (m = 3, 2 and 1; ties; a missing test; the cap at 1; input order kept)." % len(cases))
    for (b, c_), want in (((5, 0), 0.0625), ((0, 0), 1.0), ((10, 2), 158.0 / 4096.0), ((3, 3), 1.0), ((0, 6), 0.03125)):
        if abs(exact_p(b, c_) - want) > 1e-12:
            die("self-test S4: exact_p(%d, %d) = %r, expected %r" % (b, c_, exact_p(b, c_), want))
    out.append("S4 PASS: exact two-sided McNemar, 5 hand-made cases.")

    def rows(n, wrong=(), unparsed=(), truncated=()):
        return [OrderedDict([("rid", "r%03d" % i), ("status", "ok"), ("pred", "" if i in unparsed else "x"), ("unparsed", i in unparsed),
                             ("correct", i not in wrong and i not in unparsed), ("truncated", i in truncated)]) for i in range(n)]
    x = rows(100, wrong=range(0, 10))
    y = rows(100, wrong=range(5, 25))
    cp = compare(x, y)
    la = cp["lines"]["LA"]
    if (la["n"], la["cx"], la["cy"], la["b"], la["c"], cp["deciding"]) != (100, 90, 80, 15, 5, "LA") or delta_pp(la) != "+10.00":
        die("self-test S5: LA counts %s" % dict(la))
    got = [compare(x, z)["deciding"] for z in (rows(100, wrong=range(5, 25), unparsed=range(95, 100)), rows(100, wrong=range(5, 25), unparsed=range(94, 100)),
                                                rows(100, wrong=range(5, 25), truncated=range(0, 6)),
                                                rows(100, wrong=range(5, 25), unparsed=range(94, 100), truncated=range(0, 6)))]
    if got != ["LA", "LP", "LT", "LPT"]:
        die("self-test S5: deciding lines %s" % got)
    keep = {"r%03d" % i for i in range(50)}
    sub = compare(restrict(x, keep), restrict(y, keep))["lines"]["LA"]
    if (sub["n"], sub["cx"], sub["cy"]) != (50, 40, 30):
        die("self-test S5: restricted counts %s" % dict(sub))
    out.append("S5 PASS: hand-made rows: LA counts, delta and p; the deciding-line rule at and above 5.0 pp (LA, LP, LT, LPT); a comparison restricted to a row set.")

    g = {"um_00001": "B", "um_00002": "C", "um_00003": "A", "um_00004": "D"}
    rr = [OrderedDict([("rid", "um_00001"), ("status", "ok"), ("generation", "<think>A</think>\nAnswer: B"), ("truncated", False)]),
          OrderedDict([("rid", "um_00002"), ("status", "reading_failed"), ("generation", ""), ("truncated", False)]),
          OrderedDict([("rid", "um_00003"), ("status", "ok"), ("generation", "<think>Answer: A"), ("truncated", True)]),
          OrderedDict([("rid", "um_00004"), ("status", "ok"), ("generation", "<think>x</think>Answer: E"), ("truncated", False)])]
    ev = evaluate("S_ro", rr, g)
    if [(e["pred"], e["unparsed"], e["correct"]) for e in ev] != [("B", False, True), ("", True, False), ("A", False, True), ("", True, False)]:
        die("self-test S6: row evaluation %s" % [dict(e) for e in ev])
    em = evaluate("M_r1", [OrderedDict([("rid", "um_00001"), ("status", "refused"), ("generation", ""), ("truncated", False)]),
                           OrderedDict([("rid", "um_00002"), ("status", "ok"), ("generation", "The answer is C.\nAnswer: C"), ("truncated", False)])], g)
    if [(e["pred"], e["correct"]) for e in em] != [("", False), ("C", True)]:
        die("self-test S6: API row evaluation %s" % [dict(e) for e in em])
    rids = ["a", "b", "c", "d", "e"]
    A = set_a(rids, {"a": True, "b": False, "c": True, "d": True, "e": True},
              {"M_ur": {"a": "ok", "b": "ok", "c": "refused", "d": "ok", "e": "ok"}, "M_ue": dict.fromkeys(rids, "ok"),
               "M_r1": {"a": "ok", "b": "reading_failed", "c": "ok", "d": "ok", "e": "refused"}})
    if A != ["a", "d"]:
        die("self-test S6: set A %s" % A)
    out.append("S6 PASS: hand-made rows: the answer after </think>, a truncated thinking row, E outside the class, a reading_failed and a refused row "
               "(unparsed, wrong); set A drops a row without a valid R1 reading and a row refused in any M arm.")
    w = [wording(5, 0.01), wording(-5, 0.01), wording(5, 0.05), wording(0, 0.001), wording(-5, 0.2)]
    if w != ["helps", "hurts", "no clear effect", "no clear effect", "no clear effect"]:
        die("self-test S7: wording %s" % w)
    if [pred_mark(-2.0, -2.0, 6.0), pred_mark(6.0, -2.0, 6.0), pred_mark(6.01, -2.0, 6.0), pred_mark(-4.5, -4.0, 3.0)] != ["hit", "hit", "miss", "miss"]:
        die("self-test S7: prediction marks")
    n9 = [reading_n9(["helps", "helps", "no clear effect"], 2), reading_n9(["helps", "helps", "helps"], 2), reading_n9(["no clear effect", "hurts", None], 2),
          reading_n9(["helps", "hurts", "no clear effect"], 2), reading_n9(["helps", "helps"], 2), reading_n9(["helps", "no clear effect"], 2),
          reading_n9(["hurts", "no clear effect"], 2)]
    if n9 != ["holds", "holds", "not shown", "mixed", "holds", "mixed", "not shown"]:
        die("self-test S7: N9 readings %s" % n9)
    out.append("S7 PASS: wording rules, prediction marks, and the N9 reading (H1: at least 2 of 3 'helps' = holds, none = not shown, otherwise mixed; "
               "H2: both = holds, none = not shown, otherwise mixed).")
    return out


# ----------------------------------------------------------------------------
# Gates and loading
# ----------------------------------------------------------------------------
def git(*a):
    return subprocess.check_output(("git",) + a, cwd=C.ROOT).decode().strip()


def require_committed_and_pushed():
    files = [C.EFBPT + n for n in ("d15_readout.py", "d15_score.py", "d15_common.py")]
    dirty = git("status", "--porcelain", "--", *files)
    if dirty:
        die("the readout scripts are not committed and clean: " + dirty.replace("\n", "; "))
    for f in files:
        last = git("log", "-1", "--format=%H", "--", f)
        if not last:
            die("%s is not committed" % f)
        if subprocess.call(["git", "merge-base", "--is-ancestor", last, "origin/main"], cwd=C.ROOT) != 0:
            die("%s: its last commit %s is not on origin/main (push first)" % (f, last[:7]))
    return git("rev-parse", "HEAD")


def load_seal(pth, mode="FULL"):
    seal = json.load(open(C.rp(pth["seal"]), encoding="utf-8"), object_pairs_hook=OrderedDict)
    if seal.get("mode") != mode or seal.get("scored") is not False:
        die("%s: not a %s, unscored seal" % (pth["seal"], mode))
    return seal


def load_all(mode="FULL"):
    rows = C.load_rows(mode)
    rids = [r["rid"] for r in rows]
    led = C.ledger_read() if os.path.exists(C.rp(C.LEDGER)) else None
    trans, tinfo = OrderedDict(), OrderedDict()
    for name in C.TRANSLATIONS:
        pth = C.tr_paths(name, mode)
        if not C.is_sealed(pth):
            die("the %s translations are not sealed" % name)
        seal = load_seal(pth, mode)
        rr = C.read_rows(pth["rows"])
        if [r["rid"] for r in rr] != rids:
            die("%s: rows are not the sample in run order" % pth["rows"])
        C.load_reading(name, mode, rows)                      # validity of every sealed block
        trans[name] = OrderedDict((r["rid"], r) for r in rr)
        tinfo[name] = OrderedDict([("file", pth["rows"]), ("md5", C.md5_file(pth["rows"])), ("seal", pth["seal"]), ("seal_md5", C.md5_file(pth["seal"])), ("seal_obj", seal)])
    fpth = C.facts_paths(mode)
    if not C.is_sealed(fpth):
        die("the teacher facts are not sealed")
    fseal = load_seal(fpth, mode)
    r1map = OrderedDict((rid, trans["R1"][rid]["block"] if trans["R1"][rid]["status"] == "ok" else None) for rid in rids)
    facts, _ = C.load_facts(mode, rows, r1map)
    recs, info, missing = OrderedDict(), OrderedDict(), []
    for arm in C.ARMS:
        pth = C.arm_paths(arm, mode)
        if not C.is_sealed(pth):
            if C.ARMS[arm]["model"] == "M" and led is not None and led["stopped"]:
                missing.append(arm)
                continue
            die("%s is not sealed (%s); nothing is scored before every arm is sealed" % (arm, pth["seal"]))
        seal = load_seal(pth, mode)
        rr = C.read_rows(pth["rows"])
        if [r["rid"] for r in rr] != rids:
            die("%s: rows are not the sample in run order" % pth["rows"])
        recs[arm] = rr
        info[arm] = OrderedDict([("file", pth["rows"]), ("md5", C.md5_file(pth["rows"])), ("seal", pth["seal"]), ("seal_md5", C.md5_file(pth["seal"])),
                                 ("slurm", seal.get("slurm_job_id")), ("end_time", seal.get("end_time")), ("seal_obj", seal)])
    return rows, rids, trans, tinfo, facts, OrderedDict([("file", fpth["rows"]), ("md5", C.md5_file(fpth["rows"])), ("seal_obj", fseal)]), recs, info, missing, led


# ----------------------------------------------------------------------------
# Document pieces
# ----------------------------------------------------------------------------
def line_cells(l):
    return "%d | %s (%d) | %s (%d) | %s | %d | %d | %s" % (l["n"], pct(l["cx"], l["n"]), l["cx"], pct(l["cy"], l["n"]), l["cy"], delta_pp(l), l["b"], l["c"], fmt_p(l["p"]))


def every_line(label, a, b, cp):
    return ["| %s | %s vs %s | %s%s | %s |" % (label, a, b, k, " (deciding)" if k == cp["deciding"] else "", line_cells(cp["lines"][k])) for k in LINES]


def family(name, tests, m, ev, A, allr, local_ok, missing):
    """Family table on set A (deciding line, raw p, Holm p, wording), the lines beside it, every line."""
    cps, ps = [], []
    for a, b in tests:
        if a in missing or b in missing:
            cps.append(None)
            ps.append(None)
            continue
        cp = compare(restrict(ev[a], A), restrict(ev[b], A))
        cps.append(cp)
        ps.append(cp["lines"][cp["deciding"]]["p"])
    adj = holm(ps, m)
    T = ["| test | deciding line (set A) | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p | Holm p | wording |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = []
    beside = ["| test | row set | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for (a, b), cp, hp in zip(tests, cps, adj):
        if cp is None:
            T.append("| %s vs %s | missing | - | - | - | - | - | - | - | - | missing (an arm was not run) |" % (a, b))
            verdicts.append((a, b, None, None, None))
            continue
        l = cp["lines"][cp["deciding"]]
        w = wording(l["cx"] - l["cy"], hp)
        verdicts.append((a, b, w, 100.0 * (l["cx"] - l["cy"]) / l["n"] if l["n"] else 0.0, hp))
        T.append("| %s vs %s | %s | %s | %s | %s |" % (a, b, cp["deciding"], line_cells(l), fmt_p(hp), w))
        ca = compare(restrict(ev[a], allr), restrict(ev[b], allr))
        beside.append("| %s vs %s | all sampled rows | %s | %s |" % (a, b, ca["deciding"], line_cells(ca["lines"][ca["deciding"]])))
        for arm in (a, b):
            if arm in LOCAL:
                keep = A & local_ok[LOCAL[arm]]
                cl = compare(restrict(ev[a], keep), restrict(ev[b], keep))
                beside.append("| %s vs %s | set A rows with a valid %s translation | %s | %s |" % (a, b, LOCAL[arm], cl["deciding"], line_cells(cl["lines"][cl["deciding"]])))
    lines = ["| row set | test | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |", "|---|---|---|---|---|---|---|---|---|---|"]
    notes = []
    for (a, b), cp in zip(tests, cps):
        if cp is None:
            continue
        lines += every_line("set A", a, b, cp)
        notes.append("- %s vs %s on set A: unparsed %d / %d (%s%% / %s%%), truncated %d / %d (%s%% / %s%%); deciding line %s (%s)."
                     % (a, b, cp["unparsed"][0], cp["unparsed"][1], pct(cp["unparsed"][0], cp["n"]), pct(cp["unparsed"][1], cp["n"]),
                        cp["truncated"][0], cp["truncated"][1], pct(cp["truncated"][0], cp["n"]), pct(cp["truncated"][1], cp["n"]),
                        cp["deciding"], LINE_TEXT[cp["deciding"]]))
    for (a, b), cp in zip(tests, cps):
        if cp is None:
            continue
        lines += every_line("all sampled rows", a, b, compare(restrict(ev[a], allr), restrict(ev[b], allr)))
        for arm in (a, b):
            if arm in LOCAL:
                keep = A & local_ok[LOCAL[arm]]
                lines += every_line("set A, valid %s" % LOCAL[arm], a, b, compare(restrict(ev[a], keep), restrict(ev[b], keep)))
    return T, beside, lines, notes, verdicts


def build_doc(self_lines, head, mode="FULL", dummy_gold=False):
    """dummy_gold (the --dry-test on the TEST-5 files): every gold is '-', so no row can be correct and nothing is scored."""
    rows, rids, trans, tinfo, facts, finfo, recs, info, missing, led = load_all(mode)
    gold = OrderedDict((rid, "-") for rid in rids) if dummy_gold else C.load_gold()
    meta = OrderedDict((rid, m) for rid, m in C.row_meta().items() if rid in set(rids))
    ev = OrderedDict((arm, evaluate(arm, rr, gold)) for arm, rr in recs.items())
    allr = set(rids)
    r1_ok = OrderedDict((rid, trans["R1"][rid]["status"] == "ok") for rid in rids)
    local_ok = OrderedDict((n, {rid for rid in rids if trans[n][rid]["status"] == "ok"}) for n in ("TL", "SL"))
    m_status = OrderedDict((arm, OrderedDict((r["rid"], r["status"]) for r in recs[arm])) for arm in C.arms_of("M") if arm in recs)
    A_list = set_a(rids, r1_ok, m_status)
    A = set(A_list)
    groups = OrderedDict((g, {rid for rid in rids if meta[rid]["group"] == g}) for g in C.GROUPS)
    domains = OrderedDict((d, {rid for rid in rids if meta[rid]["domain"] == d}) for d in sorted({m["domain"] for m in meta.values()}))
    subs = OrderedDict((s, {rid for rid in rids if meta[rid]["subdomain"] == s}) for s in sorted({m["subdomain"] for m in meta.values()}))

    D = []
    D.append("# D15 readout — reading effect and teacher facts on native Urdu questions (UrduMMLU)")
    D.append("")
    D.append("Written by `%sd15_readout.py` (MD5 `%s`, git HEAD `%s`), one run, %s. Freeze: `%s` (MD5 `%s`). Sample: `%s` (MD5 `%s`), %d rows. "
             "Every arm, translation and facts file was sealed before this script opened it; no D15 arm was scored before this run. No arm is run again after this readout."
             % (C.EFBPT, C.md5_file(C.EFBPT + "d15_readout.py"), head, time.strftime("%Y-%m-%d %H:%M:%S"), C.FREEZE_PATH, C.FREEZE_MD5, C.SAMPLE_PATH, C.SAMPLE_MD5, len(rids)))
    D.append("")
    D.append("Data: UrduMMLU (MBZUAI/UrduMMLU, CC BY 4.0), revision `%s`, `%s` (MD5 `%s`): native Urdu questions, never translated from English. "
             "**The sample is equal per subject (100 rows from each of the 26 subdomains), so every overall accuracy here is the mean over subjects, not the "
             "dataset-weighted accuracy of UrduMMLU.**" % (C.DATA_REVISION, C.DATA, C.PINS[C.DATA]))
    D.append("")
    D.append("Scoring (freeze N5, 11.10): the CSQA fair `extract_answer` with the letter class `[A-D]`, on the answer text (after `</think>` for Qwen3-14B; the whole output otherwise). "
             "Empty prediction = unparsed = wrong; a reading_failed, refused or too_long row is unparsed = wrong. Truncated = reached max_tokens.")
    D.append("")
    D.append("Arms: S = Qwen3-14B (student), T = Llama-3.1-70B-AWQ (teacher), M = qwen3.8-max (commercial). `_ur` = the Urdu prompt (CSQA fair prompt for four options); "
             "`_ue` = the English instruction with the Urdu item; T_r1 / M_r1 = the English prompt with the R1 translation (qwen3.8-max); S_ro = the Urdu prompt plus the R1 "
             "translation; S_self = the Urdu prompt plus Qwen3-14B's own translation (SL); T_self = the English prompt with Llama's own translation (TL); S_tr = the Urdu prompt "
             "plus teacher facts (from the R1 item). M_r1 is self-translation for M (R1 is the same model).")
    D.append("")
    D.append("## 0. Self-test (run before any D15 arm was opened; every check fatal)")
    D.append("")
    D += ["- " + s for s in self_lines]
    D.append("")

    D.append("## 1. Sealed files")
    D.append("")
    D.append("| unit | file | MD5 | rows | row status | seal |")
    D.append("|---|---|---|---|---|---|")
    for name, i in tinfo.items():
        s = i["seal_obj"]
        D.append("| translation %s | `%s` | `%s` | %d | %s | %s (md5 %s; %s) |" % (name, i["file"], i["md5"], s["n_rows"], ", ".join("%s %d" % kv for kv in s["status_counts"].items()),
                                                                          i["seal"], i["seal_md5"], ("slurm %s" % s.get("slurm_job_id")) if s.get("slurm_job_id") else "login node"))
    fs = finfo["seal_obj"]
    D.append("| teacher facts | `%s` | `%s` | %d | %s | slurm %s |" % (finfo["file"], finfo["md5"], fs["n_rows"], ", ".join("%s %d" % kv for kv in fs["status_counts"].items()), fs.get("slurm_job_id")))
    for arm in C.ARMS:
        if arm in missing:
            D.append("| %s | - | - | - | - | MISSING: not sealed, the API budget ended |" % arm)
            continue
        i = info[arm]
        st = i["seal_obj"]["status_counts"]
        D.append("| %s | `%s` | `%s` | %d | %s | %s (md5 %s; %s; %s) |" % (arm, i["file"], i["md5"], i["seal_obj"]["n_rows"], ", ".join("%s %d" % kv for kv in st.items()),
                                                                     i["seal"], i["seal_md5"], ("slurm %s" % i["slurm"]) if i["slurm"] else "login node", i["end_time"]))
    D.append("")
    D.append("Missing arms: %s." % (", ".join(missing) if missing else "none; all 11 arm files, the 3 translation files and the facts file are sealed"))
    D.append("")

    D.append("## 2. Sample, set A, dropped and failed rows")
    D.append("")
    D.append("Set A (freeze N6) = sampled rows with a valid R1 translation and an answered (not refused) call in %s. n(sampled) = %d, n(set A) = %d, dropped = %d."
             % (", ".join(m_status), len(rids), len(A), len(rids) - len(A)))
    D.append("")
    D.append("| rows | sampled | set A | dropped | R1 reading_failed | M_ur refused | M_ue refused | M_r1 refused | TL not valid | SL not valid | rows without facts |")
    D.append("|---|---|---|---|---|---|---|---|---|---|---|")
    splits = [("all", allr)] + [("group " + g, s) for g, s in groups.items()] + [("domain " + d, s) for d, s in domains.items()]
    for lab, s in splits:
        D.append("| %s | %d | %d | %d | %d | %s | %s | %s | %d | %d | %d |"
                 % (lab, len(s), len(A & s), len(s - A), sum(1 for rid in s if not r1_ok[rid]),
                    *[(str(sum(1 for rid in s if m_status[a][rid] == "refused")) if a in m_status else "missing") for a in ("M_ur", "M_ue", "M_r1")],
                    sum(1 for rid in s if rid not in local_ok["TL"]), sum(1 for rid in s if rid not in local_ok["SL"]), sum(1 for rid in s if not facts[rid])))
    D.append("")
    D.append("A row is counted once under 'dropped' even if it has more than one cause; 'not valid' = reading_failed or too_long.")
    D.append("")

    D.append("## 3. Per arm")
    D.append("")
    for lab, keep in (("set A (n = %d)" % len(A), A), ("all sampled rows (n = %d)" % len(rids), allr)):
        D.append("### %s" % lab)
        D.append("")
        D.append("| arm | model | acc % (correct) | unparsed (%) | truncated (%) | reading_failed | refused | too_long |")
        D.append("|---|---|---|---|---|---|---|---|")
        for arm, e in ev.items():
            s = summary(restrict(e, keep))
            D.append("| %s | %s | %s (%d) | %d (%s) | %d (%s) | %d | %d | %d |"
                     % (arm, MODEL_NAME[C.ARMS[arm]["model"]], pct(s["correct"], s["n"]), s["correct"], s["unparsed"], pct(s["unparsed"], s["n"]),
                        s["truncated"], pct(s["truncated"], s["n"]), s["status"].get("reading_failed", 0), s["status"].get("refused", 0), s["status"].get("too_long", 0)))
        D.append("")

    wording_note = ("Wording (D14 F6): \"helps\" only if delta > 0 and Holm p < 0.05; \"hurts\" only if delta < 0 and Holm p < 0.05; otherwise \"no clear effect\". "
                    "Delta = first arm minus second arm on the deciding line of set A.")
    fam_out = OrderedDict()
    for key, title, tests, m, what in (("H1", "## 4. Family H1 — reading effect on native Urdu (3 tests, Holm at 0.05)", H1, 3,
                                        "First arm = the model with the R1 English translation; second arm = the same model on the Urdu item (S: Urdu prompt; T and M: English instruction with the Urdu item)."),
                                       ("H2", "## 5. Family H2 — self-translation, no outside model (2 tests, Holm at 0.05)", H2, 2,
                                        "First arm = the model with its own English translation (S_self: SL in the Urdu prompt; T_self: TL in the English prompt); second arm = S_ur / T_ue. "
                                        "A row without a valid own translation counts as wrong in the first arm (N6); the line on the rows with a valid own translation is printed beside it."),
                                       ("H3", "## 6. Family H3 — method (1 test)", H3, 1,
                                        "First arm = S_tr (Qwen3-14B with teacher facts from the R1 item); second arm = S_ur.")):
        D.append(title)
        D.append("")
        D.append(what + " " + wording_note)
        D.append("")
        T, beside, lines, notes, verdicts = family(key, tests, m, ev, A, allr, local_ok, missing)
        fam_out[key] = verdicts
        D += T
        D.append("")
        D.append("Printed beside each test (not deciding):")
        D.append("")
        D += beside
        D.append("")
        D.append("Every line (%s):" % key)
        D.append("")
        D += lines
        D.append("")
        D += notes
        D.append("")
        D.append("%s wording:" % key)
        D.append("")
        for a, b, w, d, hp in verdicts:
            D.append("- %s vs %s: %s." % (a, b, ("%s (delta %+.2f pp, Holm p %s)" % (w, d, fmt_p(hp))) if w else "missing"))
        D.append("")

    D.append("## 7. Reading of the outcome (freeze N9, fixed before any run)")
    D.append("")
    w1 = [v[2] for v in fam_out["H1"]]
    w2 = [v[2] for v in fam_out["H2"]]
    r1n = reading_n9(w1, 2)
    r2n = reading_n9(w2, 2)
    h1_list = "; ".join("%s vs %s: %s" % (a, b, w or "missing") for a, b, w, _, _ in fam_out["H1"])
    h2_list = "; ".join("%s vs %s: %s" % (a, b, w or "missing") for a, b, w, _, _ in fam_out["H2"])
    if r1n == "holds":
        D.append("- H1: %d of 3 tests say \"helps\". **The reading effect holds on native Urdu text.** (%s)" % (w1.count("helps"), h1_list))
    elif r1n == "not shown":
        D.append("- H1: none of the 3 tests says \"helps\". **The effect measured on the translated tasks (D13, D14) is not shown on native text and is reported as tied, "
                 "at least in part, to translated benchmarks.** (%s)" % h1_list)
    else:
        D.append("- H1: **mixed** (%d of 3 say \"helps\"). One by one: %s." % (w1.count("helps"), h1_list))
    if r2n == "holds":
        D.append("- H2: both tests say \"helps\". **The gain needs no stronger translator.** (%s)" % h2_list)
    elif r2n == "not shown":
        D.append("- H2: neither test says \"helps\". **That the gain needs no stronger translator is not shown.** (%s)" % h2_list)
    else:
        D.append("- H2: **mixed** (%d of 2 say \"helps\"). One by one: %s." % (w2.count("helps"), h2_list))
    D.append("- H3: S_tr vs S_ur: %s." % (fam_out["H3"][0][2] or "missing"))
    D.append("- Whatever the outcome, D11, D13 and D14 stand as they are.")
    D.append("")

    D.append("## 8. Descriptive comparisons on set A (no wording, raw p only)")
    D.append("")
    D.append("| row set | comparison | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |")
    D.append("|---|---|---|---|---|---|---|---|---|---|")
    for a, b in DESCRIPTIVE:
        if a in missing or b in missing:
            D.append("| set A | %s vs %s | missing | - | - | - | - | - | - | - |" % (a, b))
            continue
        D += every_line("set A", a, b, compare(restrict(ev[a], A), restrict(ev[b], A)))
    D.append("")

    D.append("## 9. Group and domain splits (set A; descriptive, raw p only)")
    D.append("")
    D.append("Groups from the subdomain names only (freeze 11.2): URDU = urdu literature, urdu language, urdu grammar; REGIONAL = pakistan studies, islamic studies; OTHER = the other 21. "
             "Inside a split the deciding-line rule is applied to the rows of the split.")
    D.append("")
    D.append("| split | test | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |")
    D.append("|---|---|---|---|---|---|---|---|---|---|")
    split_delta = OrderedDict()
    for lab, s in [("group " + g, s) for g, s in groups.items()] + [("domain " + d, s) for d, s in domains.items()]:
        keep = A & s
        for a, b in SPLIT_PAIRS:
            if a in missing or b in missing:
                D.append("| %s | %s vs %s | missing | - | - | - | - | - | - | - |" % (lab, a, b))
                continue
            cp = compare(restrict(ev[a], keep), restrict(ev[b], keep))
            l = cp["lines"][cp["deciding"]]
            split_delta[(lab, a, b)] = cp["lines"]["LA"]
            D.append("| %s | %s vs %s | %s | %s |" % (lab, a, b, cp["deciding"], line_cells(l)))
    D.append("")

    D.append("## 10. Per subdomain (set A; deltas in pp on all rows of the subdomain in set A, no p)")
    D.append("")
    D.append("| subdomain | group | domain | n (set A) | S_ur %% | %s |" % " | ".join("%s − %s" % (a, b) for a, b in SUBDOMAIN_PAIRS))
    D.append("|---|---|---|---|---|%s|" % "|".join("---" for _ in SUBDOMAIN_PAIRS))
    for sname, s in subs.items():
        keep = A & s
        one = next(iter(s))
        cells = []
        for a, b in SUBDOMAIN_PAIRS:
            if a in missing or b in missing:
                cells.append("missing")
                continue
            cells.append(delta_pp(compare(restrict(ev[a], keep), restrict(ev[b], keep))["lines"]["LA"]))
        su = summary(restrict(ev["S_ur"], keep))
        D.append("| %s | %s | %s | %d | %s | %s |" % (sname, meta[one]["group"], meta[one]["domain"], len(keep), pct(su["correct"], su["n"]), " | ".join(cells)))
    D.append("")

    D.append("## 11. Predictions (freeze N8, written before any run; read on LA of set A)")
    D.append("")
    D.append("| prediction | first % | second % | delta pp | result |")
    D.append("|---|---|---|---|---|")
    hits = total = 0
    for text, a, b, lo, hi in PREDICTIONS:
        if a in missing or b in missing:
            D.append("| %s | - | - | - | missing |" % text)
            continue
        la = compare(restrict(ev[a], A), restrict(ev[b], A))["lines"]["LA"]
        if not la["n"]:
            D.append("| %s | - | - | - | no rows in set A |" % text)
            continue
        d = 100.0 * (la["cx"] - la["cy"]) / la["n"]
        mark = pred_mark(d, lo, hi)
        hits += mark == "hit"
        total += 1
        D.append("| %s | %s | %s | %+.2f | %s |" % (text, pct(la["cx"], la["n"]), pct(la["cy"], la["n"]), d, mark))
    for a, b in H1:
        if a in missing or b in missing:
            D.append("| %s − %s: the delta in URDU is not above the delta in OTHER | - | - | - | missing |" % (a, b))
            continue
        du, do = split_delta[("group URDU", a, b)], split_delta[("group OTHER", a, b)]
        if not du["n"] or not do["n"]:
            D.append("| %s − %s: the delta in URDU is not above the delta in OTHER | - | - | - | no rows (n URDU %d, n OTHER %d) |" % (a, b, du["n"], do["n"]))
            continue
        vu = 100.0 * (du["cx"] - du["cy"]) / du["n"]
        vo = 100.0 * (do["cx"] - do["cy"]) / do["n"]
        mark = "hit" if vu <= vo else "miss"
        hits += mark == "hit"
        total += 1
        D.append("| %s − %s: the delta in URDU is not above the delta in OTHER | URDU %+.2f | OTHER %+.2f | %+.2f | %s |" % (a, b, vu, vo, vu - vo, mark))
    D.append("")
    D.append("Predictions hit: %d of %d." % (hits, total))
    D.append("")

    D.append("## 12. Translations and teacher facts")
    D.append("")
    D.append("| translation | model | rows | valid | reading_failed | too_long | requests | accepted readings that reached max_tokens | contains \"correct\" | contains \"answer is\" | mark on one option | any flag |")
    D.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in C.TRANSLATIONS:
        rr = list(trans[name].values())
        st = Counter(r["status"] for r in rr)
        fl = Counter()
        for r in rr:
            for k, v in r["flags"].items():
                fl[k] += int(v)
            fl["any"] += int(any(r["flags"].values()))
        s = tinfo[name]["seal_obj"]
        D.append("| %s | %s | %d | %d | %d | %d | %s | %d | %d | %d | %d | %d |"
                 % (name, C.TRANSLATIONS[name], len(rr), st.get("ok", 0), st.get("reading_failed", 0), st.get("too_long", 0), s.get("requests", len(rr)),
                    s.get("accepted_readings_that_reached_max_tokens", 0), fl["correct"], fl["answer_is"], fl["mark_on_one_option"], fl["any"]))
    D.append("")
    D.append("Flags are reported, not a check (freeze 11.5). \"correct\" also matches questions such as \"Which of the following is correct?\".")
    D.append("")
    D.append("| teacher facts | rows | replies parsed | facts kept | lines removed by the D10 verdict filter | lines removed by the D14 filter | rows with a D14-removed line | rows without facts |")
    D.append("|---|---|---|---|---|---|---|---|")
    D.append("| `%s` | %d | %d | %d | %d | %d | %d | %d |" % (finfo["md5"], fs["n_rows"], fs["parse_ok"], fs["facts_total"], fs["removed_verdict_lines"], fs["removed_d14_lines"],
                                                       fs["rows_with_a_d14_removed_line"], fs["rows_without_facts"]))
    D.append("")
    D.append("A row without facts has no facts block: its S_tr prompt equals its S_ur prompt.")
    D.append("")

    D.append("## 13. Seconds and tokens")
    D.append("")
    D.append("Seconds of a vLLM row = the seconds of its chunk divided by the rows generated in the chunk (up to 64 prompts per call); seconds of an API row = the latency of its answered request (R1: of all its requests).")
    D.append("")
    D.append("| unit | rows generated | seconds total | seconds per row | prompt tokens per row | new tokens per row |")
    D.append("|---|---|---|---|---|---|")
    units = [("translation " + n, list(trans[n].values())) for n in C.TRANSLATIONS] + [(arm, rr) for arm, rr in recs.items()]
    for lab, rr in units:
        g_ = [r for r in rr if r["status"] in ("ok", "reading_failed") and r.get("n_gen_tokens") is not None]
        secs = [r["seconds"] for r in g_ if isinstance(r.get("seconds"), (int, float))]
        pt = [r["n_prompt_tokens"] for r in g_ if isinstance(r.get("n_prompt_tokens"), (int, float))]
        gt = [r["n_gen_tokens"] for r in g_ if isinstance(r.get("n_gen_tokens"), (int, float))]
        D.append("| %s | %d | %.0f | %.2f | %.1f | %.1f |" % (lab, len(g_), sum(secs), sum(secs) / len(secs) if secs else 0.0,
                                                          sum(pt) / len(pt) if pt else 0.0, sum(gt) / len(gt) if gt else 0.0))
    D.append("")
    if led is not None:
        D.append("API ledger at the time of the readout (TEST-5 and retries included): %d calls of at most %d; %d tokens of at most %d (prompt %d, completion %d); stopped: %s."
                 % (led["calls"], led["cap_calls"], led["tokens"], led["cap_tokens"], led["prompt_tokens"], led["completion_tokens"], led["stopped"] or "no"))
        used = [(b.get("note"), b.get("used_by_key")) for b in led.get("balance", []) if b.get("used_by_key") is not None]
        if used:
            D.append("")
            D.append("Amount used by the key (gateway billing query; the key has no limit, so no balance is given): %s."
                     % "; ".join("%s: %.4f" % (n, u) for n, u in used))
        D.append("")
    return "\n".join(D) + "\n", fam_out, r1n, r2n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="run the self-test only; no D15 arm file is opened")
    ap.add_argument("--dry-test", default=None, metavar="OUT",
                    help="build the document from the TEST-5 files with every gold set to '-' (nothing can be correct; nothing is scored) and write it to OUT")
    args = ap.parse_args()
    print("=" * 78)
    print("D15 READOUT%s" % (" — SELF-TEST ONLY (no D15 arm file is opened)" if args.self_test else ""))
    print("=" * 78, flush=True)
    C.verify_pins()
    lines = self_test()
    for s in lines:
        print("[self-test] " + s)
    if args.self_test:
        print("[self-test] PASS")
        return
    if args.dry_test:
        if os.path.abspath(args.dry_test) == os.path.abspath(C.rp(DOC)):
            die("the dry test never writes the readout document")
        text, fam, r1n, r2n = build_doc(lines, "dry-test", mode="TEST", dummy_gold=True)
        with open(args.dry_test, "w", encoding="utf-8") as f:
            f.write(text)
        print("[dry-test] built the document from the TEST-5 files with a dummy gold ('-'): %d lines written to %s; nothing scored" % (text.count("\n"), args.dry_test))
        return
    if os.path.exists(C.rp(DOC)):
        die("%s exists: the readout is run once" % DOC)
    head = require_committed_and_pushed()
    print("[ok] d15_readout.py, d15_score.py and d15_common.py are committed, unchanged and on origin/main (HEAD %s)" % head, flush=True)
    text, fam, r1n, r2n = build_doc(lines, head)
    with open(C.rp(DOC), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("[done] wrote %s (%d lines); N9: H1 %s, H2 %s" % (DOC, text.count("\n"), r1n, r2n))


if __name__ == "__main__":
    main()
