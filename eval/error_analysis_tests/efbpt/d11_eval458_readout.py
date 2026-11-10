#!/usr/bin/env python3
"""
d11_eval458_readout.py — D11 RTG Amendment 5: the eval458 readout, fixed before scoring.

Contract: docs/EFBPT_D11_RTG_AMENDMENT_5.md (MD5 pinned below), with A2.2/A2.6/A2.7,
A3.1/A3.2 and A4.4. CPU only, no model. Edits no frozen file. Re-scores the sealed
generation files itself:
  E0 = the frozen extractor (efbpt_eval_dev200.extract_answer), imported unchanged;
  E3 = d11_scorer_audit.e3 (validated scorer, A5.1), imported; the MD5 of
       d11_scorer_audit.py is pinned to the audit commit e3fa2ca.

  --set dev200    SELF-TEST on the sealed DEV200 arms A_r2, TR, TR_L1, TR_wrong (MD5 pinned).
                  Asserts the sealed E0 numbers (TR vs A_r2: 68.5 / 60.0, b 39 / c 22,
                  p 0.0396) and the A4.4 E3 table (71.5 / 55.0, b 57 / c 24; TR_L1 60 / 30;
                  TR_wrong 37 / 46). Nothing under an eval458 directory is opened.
  --set eval458   the held-out readout. Refuses to run unless A_r2_eval, TR and
                  TR_wrong_eval are sealed (S_SEAL_<ARM>.json); TR_L1_eval is joined if
                  sealed. Per-arm seals are verified (md5, bytes, lines), the PASSED A2.3
                  regression record of every arm is required, gold is joined from the
                  pinned item set, and every sealed prompt is compared with the prompt
                  rebuilt from the sealed teacher facts (d11_student.preflight, imported).

For every comparison, four lines (A5.3), each with n, accuracy of both arms, delta, b, c
and the exact two-sided McNemar p (binomial on the discordant pairs, as d5 mcnemar):
  L1  E0, all rows                         the pre-registered gate
  L2  E3, all rows                         validated scorer (A4.4 tier)
  L3  E0, rows parsed in both arms         A3.1
  L4  rows where E0 and E3 give the same label in both arms (scorer-unambiguous; descriptive)
Per arm, under E0 and E3: unparsed, predicted-yes rate, accuracy on gold-yes and on gold-no
rows, balanced accuracy. Comparisons: primary TR vs floor; content control TR vs TR_wrong
(A5.5); TR_wrong vs floor (reported); secondary TR_L1 vs floor.

Verdict (A5.4; gate = delta >= +5.0 pp and p < 0.05):
  tier-1 line = L1 if |unparsed_E0(TR) - unparsed_E0(floor)| <= 5.0 pp, else L3 (A3.1)
  tier-1 line passes                 -> CONFIRMED
  else L2 passes                     -> CONFIRMED UNDER THE VALIDATED SCORER; PRE-REGISTERED GATE MISSED
  else                               -> NOT CONFIRMED
  secondary (A3.2: p < 0.05 with a positive delta, no pp threshold): tested in the tier that
  confirmed the primary (its own L1 / L3 by A3.1 if tier 1; its L2 if tier 2); else descriptive.

Output: the text printed is written to docs/handoff/D11_<SET>_READOUT_LINES.md (with the
secondary arm) or D11_<SET>_READOUT_LINES_PRIMARY.md (eval458 before TR_L1_eval is sealed).
An existing file is never overwritten with different content.

  --per-arm-dir DIR   testing hook, dev200 only: read the DEV200 arms from a directory in the
                      eval458 per-arm format (raw rows + S_SEAL_<ARM>.json), i.e. through the
                      loader that eval458 uses.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from d11_common import (  # noqa: E402  (chdir to the repo root)
    load_set_rows, md5_file, die, read_jsonl, file_entry, out_dir,
    D10_A_R2, D11_FREEZE_MD5, D11_AMEND1_MD5, D11_AMEND2_MD5, REGRESS_N,
)
from efbpt_eval_dev200 import extract_answer  # noqa: E402  E0, frozen

AMEND5_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_5.md"
AMEND5_MD5 = "45c118de9804390d12853f231e7632a0"
AUDIT_SCRIPT = os.path.join(_HERE, "d11_scorer_audit.py")
AUDIT_SCRIPT_MD5 = "246cf6c9687b3fa67324722cb5cd16b2"      # d11_scorer_audit.py at the audit commit e3fa2ca (A5.1)
if md5_file(AUDIT_SCRIPT) != AUDIT_SCRIPT_MD5:
    die("d11_scorer_audit.py MD5 %s != pinned %s (A5.1: E3 is the function at commit e3fa2ca)" % (md5_file(AUDIT_SCRIPT), AUDIT_SCRIPT_MD5))
from d11_scorer_audit import e3 as extract_e3  # noqa: E402  E3, validated scorer

GATE_PP, ALPHA, V1_PP = 5, 0.05, 5.0                        # freeze §8 / A2.2 gate; A3.1 unparsed difference
ROLES = OrderedDict([
    ("dev200", OrderedDict([("floor", "A_r2"), ("primary", "TR"), ("control", "TR_wrong"), ("secondary", "TR_L1")])),
    ("eval458", OrderedDict([("floor", "A_r2_eval"), ("primary", "TR"), ("control", "TR_wrong_eval"), ("secondary", "TR_L1_eval")])),
])
DEV200_SEALED = OrderedDict([
    ("A_r2", D10_A_R2),
    ("TR", ("outputs/efbpt/d11/student/dev200/d11_armTR.jsonl", "bab987cc492012f4e32dbcd348be9668")),
    ("TR_wrong", ("outputs/efbpt/d11/student/dev200/d11_armTR_wrong.jsonl", "9519b9adf215ac90eb6de743ec1dca57")),
    ("TR_L1", ("outputs/efbpt/d11/student/dev200/d11_armTR_L1.jsonl", "b8db599f8c4b427ddcb2d772fd8f09aa")),
])
OUT_FULL = "docs/handoff/D11_%s_READOUT_LINES.md"
OUT_PRIMARY = "docs/handoff/D11_%s_READOUT_LINES_PRIMARY.md"


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def row_rec(qid, gold, generation, truncated, prompt):
    return OrderedDict([("qid", qid), ("gold", gold), ("generation", generation), ("truncated", truncated), ("prompt", prompt),
                        ("e0", extract_answer(generation)), ("e3", extract_e3(generation))])


def load_dev200_sealed(rows):
    """The sealed DEV200 arm files (scored at the time; E0 must reproduce the sealed pred)."""
    arms, prov = OrderedDict(), OrderedDict()
    for arm, (path, want) in DEV200_SEALED.items():
        if "eval458" in path:
            die("refusing an eval458 path in the DEV200 self-test")
        got = md5_file(path)
        if got != want:
            die("sealed %s MD5 %s != pinned %s" % (path, got, want))
        recs = read_jsonl(path)
        if [r["qid"] for r in recs] != [r["qid"] for r in rows]:
            die("%s: qids are not DEV200 in order" % path)
        out = []
        for r, row in zip(recs, rows):
            if r["gold"] != row["gold"]:
                die("%s qid %s: gold differs from the item set" % (path, r["qid"]))
            x = row_rec(r["qid"], row["gold"], r["generation"], r["truncated"], r["prompt"])
            if x["e0"] != r["pred"]:
                die("%s qid %s: E0 does not reproduce the sealed pred" % (path, r["qid"]))
            out.append(x)
        arms[arm] = out
        prov[arm] = OrderedDict([("file", path), ("md5", got), ("seal", "MD5 pinned (readout §1)")])
    return arms, prov


def sealed_per_arm(d, arm):
    return os.path.exists(os.path.join(d, "S_SEAL_%s.json" % arm))


def load_per_arm(d, set_name, arm_names, rows):
    """The per-arm path of Amendment 2: d11_arm<ARM>.jsonl (raw rows: no pred / correct / gold) with
    S_SEAL_<ARM>.json and S_START_<ARM>.json. Seals verified; gold joined from the item set."""
    arms, prov = OrderedDict(), OrderedDict()
    for arm in arm_names:
        seal_p, arm_p, start_p = (os.path.join(d, n % arm) for n in ("S_SEAL_%s.json", "d11_arm%s.jsonl", "S_START_%s.json"))
        if not os.path.exists(seal_p):
            die("%s missing: arm %s is not sealed" % (seal_p, arm))
        seal = json.load(open(seal_p, encoding="utf-8"), object_pairs_hook=OrderedDict)
        for k, want in (("arm", arm), ("set", set_name), ("complete", True), ("n_rows", len(rows)),
                        ("d11_freeze_md5", D11_FREEZE_MD5), ("d11_amendment1_md5", D11_AMEND1_MD5), ("d11_amendment2_md5", D11_AMEND2_MD5)):
            if seal.get(k) != want:
                die("%s: %s is %r, expected %r" % (seal_p, k, seal.get(k), want))
        if not {arm_p, start_p} <= set(seal["files"]):
            die("%s does not cover the arm file and its S_START" % seal_p)
        for p, want in seal["files"].items():
            if not os.path.exists(p):
                die("%s: sealed file missing: %s" % (seal_p, p))
            got = file_entry(p)
            if (got["md5"], got["bytes"], got["lines"]) != (want["md5"], want["bytes"], want["lines"]):
                die("%s: %s differs from the seal" % (seal_p, p))
        start = json.load(open(start_p, encoding="utf-8"), object_pairs_hook=OrderedDict)
        reg = start.get("regression") or {}
        if reg.get("n_identical") != REGRESS_N or reg.get("n_rows") != REGRESS_N:
            die("%s: no PASSED A2.3 regression record (arm %s)" % (start_p, arm))
        with open(arm_p, "rb") as f:
            lines = [l for l in f.read().split(b"\n") if l]
        recs = [json.loads(l.decode("utf-8")) for l in lines]
        if [r["qid"] for r in recs] != [r["qid"] for r in rows]:
            die("%s: rows are not the %d %s qids in order" % (arm_p, len(rows), set_name))
        out = []
        for r, row in zip(recs, rows):
            if r["arm"] != arm:
                die("%s qid %s: arm field is %r" % (arm_p, r["qid"], r["arm"]))
            if any(k in r for k in ("pred", "correct", "gold")):
                die("%s qid %s: a per-arm row carries a scoring field (A3.3)" % (arm_p, r["qid"]))
            out.append(row_rec(r["qid"], row["gold"], r["generation"], r["truncated"], r["prompt"]))
        arms[arm] = out
        prov[arm] = OrderedDict([("file", arm_p), ("md5", seal["files"][arm_p]["md5"]),
                                 ("seal", "%s (md5 %s; slurm %s; %s; %d files verified)" % (seal_p, md5_file(seal_p), seal.get("slurm_job_id"), seal.get("end_time"), len(seal["files"])))])
    return arms, prov


def verify_prompts(set_name, arms):
    """Every sealed prompt equals the prompt rebuilt now from the sealed teacher facts (teacher seals,
    readings, derangement and leak guards are re-verified by d11_student.preflight, imported unchanged)."""
    import d11_student as S
    _, sub, names, prompts, _, _, _, _, _ = S.preflight(set_name, None, False, "TR" if set_name == "eval458" else None, list(arms))
    n = 0
    for arm in arms:
        if len(prompts[arm]) != len(arms[arm]):
            die("verify-prompts: %s has %d rows, %d prompts rebuilt" % (arm, len(arms[arm]), len(prompts[arm])))
        for r, (p, _, _, _) in zip(arms[arm], prompts[arm]):
            if r["prompt"] != p:
                die("verify-prompts: %s qid %s: the sealed prompt differs from the prompt rebuilt from the sealed teacher facts" % (arm, r["qid"]))
            n += 1
    return n


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def exact_p(b, c):
    from scipy.stats import binomtest
    return binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0


def line(x, y, key_x, key_y, keep):
    """Paired comparison on the rows where keep(rx, ry). key_* = 'e0' or 'e3'. b = x right & y wrong."""
    n = cx = cy = b = c = 0
    for rx, ry in zip(x, y):
        if rx["qid"] != ry["qid"]:
            die("paired rows are not aligned")
        if not keep(rx, ry):
            continue
        ox, oy = rx[key_x] == rx["gold"], ry[key_y] == ry["gold"]
        n += 1
        cx += ox
        cy += oy
        b += ox and not oy
        c += oy and not ox
    return OrderedDict([("n", n), ("cx", cx), ("cy", cy), ("acc_x", 100.0 * cx / n if n else float("nan")), ("acc_y", 100.0 * cy / n if n else float("nan")),
                        ("delta", 100.0 * (b - c) / n if n else float("nan")), ("b", b), ("c", c), ("p", exact_p(b, c))])


def four_lines(x, y):
    return OrderedDict([
        ("L1", line(x, y, "e0", "e0", lambda a, b: True)),
        ("L2", line(x, y, "e3", "e3", lambda a, b: True)),
        ("L3", line(x, y, "e0", "e0", lambda a, b: a["e0"] is not None and b["e0"] is not None)),
        ("L4", line(x, y, "e0", "e0", lambda a, b: a["e0"] == a["e3"] and b["e0"] == b["e3"])),
    ])


LINE_LABEL = OrderedDict([("L1", "L1  E0, all rows (pre-registered gate)"), ("L2", "L2  E3, all rows (validated scorer)"),
                          ("L3", "L3  E0, rows parsed in both arms (A3.1)"), ("L4", "L4  rows where E0 = E3 in both arms (descriptive)")])


def passes_gate(ln):
    """delta >= +5.0 pp and p < 0.05; the pp test in integers: 100 (b - c) >= 5 n."""
    return ln["n"] > 0 and 100 * (ln["b"] - ln["c"]) >= GATE_PP * ln["n"] and ln["p"] < ALPHA


def arm_stats(recs, key):
    n = len(recs)
    yes = [r for r in recs if r["gold"] == "yes"]
    no = [r for r in recs if r["gold"] == "no"]
    ay = 100.0 * sum(1 for r in yes if r[key] == "yes") / len(yes)
    an = 100.0 * sum(1 for r in no if r[key] == "no") / len(no)
    return OrderedDict([("n", n), ("correct", sum(1 for r in recs if r[key] == r["gold"])), ("acc", 100.0 * sum(1 for r in recs if r[key] == r["gold"]) / n),
                        ("unparsed", sum(1 for r in recs if r[key] is None)), ("unparsed_pct", 100.0 * sum(1 for r in recs if r[key] is None) / n),
                        ("yes_rate", 100.0 * sum(1 for r in recs if r[key] == "yes") / n),
                        ("acc_gold_yes", ay), ("n_gold_yes", len(yes)), ("acc_gold_no", an), ("n_gold_no", len(no)), ("balanced", (ay + an) / 2.0)])


def fp(p):
    return "%.4g" % p


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def report(set_name, arms, prov, n_prompts_verified):
    role = ROLES[set_name]
    floor, prim, ctrl, sec = role["floor"], role["primary"], role["control"], role["secondary"]
    have_sec = sec in arms
    selftest = set_name == "dev200"
    n = len(arms[floor])
    L = []
    w = L.append
    w("# D11 %s readout lines (Amendment 5)%s" % (set_name, " — SELF-TEST, not a decision" if selftest else ""))
    w("")
    w("Written by `eval/error_analysis_tests/efbpt/d11_eval458_readout.py --set %s` (MD5 `%s`); Amendment 5 `%s`; E3 from `d11_scorer_audit.py` `%s` (audit commit e3fa2ca); "
      "E0 = the frozen extractor. n = %d (gold yes %d / no %d). b = the first-named arm right and the second wrong; c = the reverse; p = exact two-sided McNemar; unparsed = wrong."
      % (set_name, md5_file(os.path.abspath(__file__)), AMEND5_MD5, AUDIT_SCRIPT_MD5, n,
         sum(1 for r in arms[floor] if r["gold"] == "yes"), sum(1 for r in arms[floor] if r["gold"] == "no")))
    w("")
    w("| arm | file | MD5 | seal |")
    w("|---|---|---|---|")
    for a, pv in prov.items():
        w("| %s | `%s` | `%s` | %s |" % (a, pv["file"], pv["md5"], pv["seal"]))
    w("")
    w("Prompts: %s." % ("%d sealed prompts equal the prompts rebuilt from the sealed teacher facts" % n_prompts_verified if n_prompts_verified is not None else "not re-verified in this run (--verify-prompts)"))
    if not have_sec:
        w("")
        w("Secondary arm %s: NOT SEALED at this readout (A2.4: joined when sealed)." % sec)
    w("")

    # ---- per-arm table (A5.3) ----
    w("## Per arm, under E0 and E3 (A5.3)")
    w("")
    w("| arm | scorer | acc % (correct) | unparsed (%) | predicted yes % | acc on gold-yes % | acc on gold-no % | balanced acc % |")
    w("|---|---|---|---|---|---|---|---|")
    stats = OrderedDict()
    for a in arms:
        for key, name in (("e0", "E0"), ("e3", "E3")):
            s = arm_stats(arms[a], key)
            stats[(a, key)] = s
            w("| %s | %s | %.2f (%d) | %d (%.2f) | %.2f | %.2f | %.2f | %.2f |"
              % (a, name, s["acc"], s["correct"], s["unparsed"], s["unparsed_pct"], s["yes_rate"], s["acc_gold_yes"], s["acc_gold_no"], s["balanced"]))
    w("")
    w("Balanced accuracy = mean of the gold-yes and gold-no accuracies (descriptive; separates knowledge from a shift to \"no\"). "
      "Rows where E0 and E3 differ: " + "; ".join("%s %d" % (a, sum(1 for r in arms[a] if r["e0"] != r["e3"])) for a in arms) + ".")
    w("")

    # ---- comparisons (A5.3, A5.5) ----
    comps = OrderedDict()
    comps["primary"] = (prim, floor, "PRIMARY (gate)")
    comps["content_control"] = (prim, ctrl, "CONTENT CONTROL (A5.5)")
    comps["wrong_vs_floor"] = (ctrl, floor, "wrong facts vs floor (reported, A5.5)")
    if have_sec:
        comps["secondary"] = (sec, floor, "SECONDARY (A2.2, A3.2)")
    res = OrderedDict()
    w("## Comparisons, four lines each (A5.3)")
    for cid, (x, y, title) in comps.items():
        fl = four_lines(arms[x], arms[y])
        res[cid] = fl
        w("")
        w("**%s vs %s — %s**" % (x, y, title))
        w("")
        w("| line | n | acc %s | acc %s | Δ pp | b | c | exact two-sided p |" % (x, y))
        w("|---|---|---|---|---|---|---|---|")
        for k, ln in fl.items():
            w("| %s | %d | %.2f | %.2f | %+.2f | %d | %d | %s |" % (LINE_LABEL[k], ln["n"], ln["acc_x"], ln["acc_y"], ln["delta"], ln["b"], ln["c"], fp(ln["p"])))
    w("")

    # ---- A3.1 and verdict tier (A5.4) ----
    def tier1_line(x):
        d = abs(stats[(x, "e0")]["unparsed_pct"] - stats[(floor, "e0")]["unparsed_pct"])
        return ("L1" if d <= V1_PP else "L3"), d

    w("## A3.1 and verdict tier (A5.4)%s" % (" — SELF-TEST on DEV200: the tier logic is exercised; the DEV200 decision is the freeze §7 GO" if selftest else ""))
    w("")
    k1, d1 = tier1_line(prim)
    w("- A3.1 primary: |unparsed E0 %s − %s| = |%.2f − %.2f| = %.2f pp %s 5.0 → the pre-registered gate is read on **%s**%s."
      % (prim, floor, stats[(prim, "e0")]["unparsed_pct"], stats[(floor, "e0")]["unparsed_pct"], d1, "≤" if d1 <= V1_PP else ">", k1,
         "" if k1 == "L1" else " (rows parsed in both arms); L1 is reported beside it"))
    dc = abs(stats[(ctrl, "e0")]["unparsed_pct"] - stats[(floor, "e0")]["unparsed_pct"])
    w("- A3.1 control: unparsed E0 %s %.2f %% vs %s %.2f %% (difference %.2f pp%s); it never changes the primary."
      % (ctrl, stats[(ctrl, "e0")]["unparsed_pct"], floor, stats[(floor, "e0")]["unparsed_pct"], dc, "; > 5.0: read the control on L3 as well" if dc > V1_PP else ""))
    g1 = res["primary"][k1]
    g2 = res["primary"]["L2"]
    p1, p2 = passes_gate(g1), passes_gate(g2)
    w("- Gate (≥ +5.0 pp and p < 0.05) on %s: Δ %+.2f pp, b %d / c %d, p = %s → **%s**." % (k1, g1["delta"], g1["b"], g1["c"], fp(g1["p"]), "PASS" if p1 else "MISSED"))
    w("- Validated scorer, L2: Δ %+.2f pp, b %d / c %d, p = %s → %s." % (g2["delta"], g2["b"], g2["c"], fp(g2["p"]), "passes" if p2 else "does not pass"))
    if p1:
        tier, verdict = 1, "CONFIRMED"
    elif p2:
        tier, verdict = 2, "CONFIRMED UNDER THE VALIDATED SCORER; PRE-REGISTERED GATE MISSED"
    else:
        tier, verdict = 0, "NOT CONFIRMED"
    w("")
    w("**VERDICT%s: %s**" % (" (self-test label only)" if selftest else "", verdict))
    w("")
    if tier == 2:
        w("Reason for the second tier: the scorer audit (Amendment 4; A5.1).")
    if tier == 0 and g1["delta"] > 0:
        w("A2.7: positive but gate missed → reported as not confirmed; pooled DEV200 + eval458 only as description.")
    if not selftest:
        w("A2.7: no second eval458 run in any case.")
    if have_sec:
        ks, ds = tier1_line(sec)
        if tier == 1:
            ln, where = res["secondary"][ks], "%s (A3.1: unparsed difference %.2f pp)" % (ks, ds)
        elif tier == 2:
            ln, where = res["secondary"]["L2"], "L2"
        else:
            ln, where = res["secondary"]["L1"], "L1"
        if tier:
            sig = ln["p"] < ALPHA and ln["b"] > ln["c"]
            w("- Secondary %s vs %s (A3.2: p < 0.05 with a positive delta; tested in tier %d on %s): Δ %+.2f pp, b %d / c %d, p = %s → **%s**."
              % (sec, floor, tier, where, ln["delta"], ln["b"], ln["c"], fp(ln["p"]), "SIGNIFICANT" if sig else "not significant"))
        else:
            w("- Secondary %s vs %s: DESCRIPTIVE (the primary is not confirmed; fixed-sequence test): L1 Δ %+.2f pp, b %d / c %d, p = %s."
              % (sec, floor, ln["delta"], ln["b"], ln["c"], fp(ln["p"])))
    else:
        w("- Secondary %s: pending (not sealed)." % sec)
    cc = res["content_control"]
    w("- Content control %s vs %s (A5.5): L1 Δ %+.2f pp (b %d / c %d, p = %s); L2 Δ %+.2f pp (b %d / c %d, p = %s)."
      % (prim, ctrl, cc["L1"]["delta"], cc["L1"]["b"], cc["L1"]["c"], fp(cc["L1"]["p"]), cc["L2"]["delta"], cc["L2"]["b"], cc["L2"]["c"], fp(cc["L2"]["p"])))
    wf = res["wrong_vs_floor"]
    w("- %s vs %s (reported): L1 Δ %+.2f pp (p = %s); L2 Δ %+.2f pp (p = %s). A5.5: under E3 on DEV200 this is −4.5 pp (p 0.38), so \"wrong facts harm the baseline\" is not claimed under E3."
      % (ctrl, floor, wf["L1"]["delta"], fp(wf["L1"]["p"]), wf["L2"]["delta"], fp(wf["L2"]["p"])))
    w("")

    # ---- A2.6 predictions ----
    w("## A2.6 predictions%s" % (" (written for eval458; shown on DEV200 only to exercise the table)" if selftest else ""))
    w("")
    w("| prediction | observed, E0 | E0 | observed, E3 (descriptive) | E3 |")
    w("|---|---|---|---|---|")

    def hm(ok):
        return "pending" if ok is None else ("hit" if ok else "miss")

    def pred_row(label, f):
        (o0, h0), (o3, h3) = f("e0"), f("e3")
        w("| %s | %s | %s | %s | %s |" % (label, o0, hm(h0), o3, hm(h3)))

    def acc(a, key):
        return stats[(a, key)]["acc"]

    def rng(lo, hi):
        return lambda v: lo <= v <= hi

    def diff_row(label, x, test):
        def f(key):
            if x not in arms:
                return "—", None
            d = acc(x, key) - acc(floor, key)
            return "%+.2f" % d, test(round(d, 6))
        pred_row(label, f)

    pred_row("%s accuracy in [55, 63]" % floor, lambda key: ("%.2f" % acc(floor, key), 55.0 <= round(acc(floor, key), 6) <= 63.0))
    diff_row("%s − %s in [+5, +10]" % (prim, floor), prim, rng(5.0, 10.0))
    diff_row("%s − %s ≤ −3" % (ctrl, floor), ctrl, lambda v: v <= -3.0)
    diff_row("%s − %s in [+4, +9]" % (sec, floor), sec, rng(4.0, 9.0))

    def gain(key):
        gn = stats[(prim, key)]["acc_gold_no"] - stats[(floor, key)]["acc_gold_no"]
        gy = stats[(prim, key)]["acc_gold_yes"] - stats[(floor, key)]["acc_gold_yes"]
        return "gold-no %+.2f, gold-yes %+.2f" % (gn, gy), gn > gy
    pred_row("gain on gold-no rows > gain on gold-yes rows (%s vs %s)" % (prim, floor), gain)
    w("")
    return "\n".join(L), res, stats, verdict


def dev200_asserts(res, stats):
    """A5.2: the readout must reproduce the sealed E0 numbers and the A4.4 E3 table on DEV200."""
    def chk(what, got, want):
        if got != want:
            die("DEV200 self-test: %s is %r, expected %r" % (what, got, want))
    p = res["primary"]
    chk("L1 TR vs A_r2 (correct TR, correct A_r2, b, c)", (p["L1"]["cx"], p["L1"]["cy"], p["L1"]["b"], p["L1"]["c"]), (137, 120, 39, 22))
    chk("L1 TR vs A_r2 p (4 d.p.)", round(p["L1"]["p"], 4), 0.0396)
    chk("L1 accuracies", (round(p["L1"]["acc_x"], 2), round(p["L1"]["acc_y"], 2)), (68.5, 60.0))
    chk("L2 TR vs A_r2 (acc TR, acc A_r2, b, c)", (round(p["L2"]["acc_x"], 2), round(p["L2"]["acc_y"], 2), p["L2"]["b"], p["L2"]["c"]), (71.5, 55.0, 57, 24))
    chk("L2 TR_L1 vs A_r2 (acc, b, c)", (round(res["secondary"]["L2"]["acc_x"], 2), res["secondary"]["L2"]["b"], res["secondary"]["L2"]["c"]), (70.0, 60, 30))
    chk("L2 TR_wrong vs A_r2 (acc, b, c)", (round(res["wrong_vs_floor"]["L2"]["acc_x"], 2), res["wrong_vs_floor"]["L2"]["b"], res["wrong_vs_floor"]["L2"]["c"]), (50.5, 37, 46))
    chk("L1 TR_L1 vs A_r2 (b, c)", (res["secondary"]["L1"]["b"], res["secondary"]["L1"]["c"]), (45, 31))
    chk("L1 TR_wrong vs A_r2 (b, c)", (res["wrong_vs_floor"]["L1"]["b"], res["wrong_vs_floor"]["L1"]["c"]), (30, 49))
    chk("L1 TR vs TR_wrong (b, c)", (res["content_control"]["L1"]["b"], res["content_control"]["L1"]["c"]), (53, 17))
    chk("L3 TR vs A_r2 (n, b, c)", (p["L3"]["n"], p["L3"]["b"], p["L3"]["c"]), (189, 34, 22))
    chk("unparsed E0 / E3 (A_r2, TR, TR_L1, TR_wrong)", tuple((stats[(a, "e0")]["unparsed"], stats[(a, "e3")]["unparsed"]) for a in ("A_r2", "TR", "TR_L1", "TR_wrong")),
        ((8, 10), (3, 4), (2, 2), (11, 11)))
    return ("DEV200 self-test PASS: L1 TR vs A_r2 = 68.5 / 60.0, b 39 / c 22, p 0.0396 (sealed); L2 = 71.5 / 55.0, b 57 / c 24 (A4.4); "
            "TR_L1 and TR_wrong L1 / L2 pairs, TR vs TR_wrong L1, L3 n / b / c (supplement S2) and the unparsed counts all as sealed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(ROLES))
    ap.add_argument("--verify-prompts", action="store_true", help="rebuild every prompt from the sealed teacher facts and compare (always on for eval458)")
    ap.add_argument("--per-arm-dir", default=None, help="testing hook, dev200 only: read the DEV200 arms through the per-arm loader")
    ap.add_argument("--out", default=None, help="output path (default docs/handoff/D11_<SET>_READOUT_LINES[_PRIMARY].md)")
    args = ap.parse_args()
    if md5_file(AMEND5_PATH) != AMEND5_MD5:
        die("%s MD5 %s != pinned %s" % (AMEND5_PATH, md5_file(AMEND5_PATH), AMEND5_MD5))
    role = ROLES[args.set]
    names = [role["floor"], role["primary"], role["control"], role["secondary"]]
    if args.set == "dev200":
        for p in (args.per_arm_dir or "", args.out or ""):
            if "eval458" in p:
                die("refusing an eval458 path with --set dev200")
        rows = load_set_rows("dev200")
        if args.per_arm_dir:
            arms, prov = load_per_arm(args.per_arm_dir, "dev200", names, rows)
        else:
            arms, prov = load_dev200_sealed(rows)
    else:
        if args.per_arm_dir:
            die("--per-arm-dir is a DEV200 testing hook")
        d = out_dir("student", "eval458", False)
        missing = [a for a in names[:3] if not sealed_per_arm(d, a)]
        if missing:
            die("eval458 readout refused: not sealed: %s (needs %s; %s is joined if sealed)" % (", ".join(missing), ", ".join(names[:3]), names[3]))
        if not sealed_per_arm(d, names[3]):
            names = names[:3]
        rows = load_set_rows("eval458")
        arms, prov = load_per_arm(d, "eval458", names, rows)
    arms = OrderedDict((a, arms[a]) for a in names)          # floor, primary, control, secondary
    n_verified = verify_prompts(args.set, arms) if (args.verify_prompts or args.set == "eval458") else None

    text, res, stats, verdict = report(args.set, arms, prov, n_verified)
    if args.set == "dev200":
        text += "\n" + dev200_asserts(res, stats) + "\n"
    full = role["secondary"] in arms
    out = args.out or (OUT_FULL if full else OUT_PRIMARY) % args.set.upper()
    print("\n" + text)
    if os.path.exists(out):
        if open(out, encoding="utf-8").read() != text:
            die("%s exists with different content; not overwritten" % out)
        print("[unchanged] %s already holds exactly this text" % out)
    else:
        with open(out, "x", encoding="utf-8") as f:
            f.write(text)
        print("[written] %s" % out)


if __name__ == "__main__":
    main()
