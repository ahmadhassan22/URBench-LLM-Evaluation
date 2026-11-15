#!/usr/bin/env python3
"""
d13_readout.py — D13 readout: baselines and transfer on eval458 (decides) and DEV200
(descriptive). One script for stage 1 (Groups 1-3, family F1 and the descriptive
comparisons) and stage 2 (Group 4, family F2); it runs both together and only when all 19
D13 arm files are sealed.

Contract (MD5 pinned below): docs/EFBPT_D13_BASELINES_TRANSFER_FREEZE.md §3, §4, §5, §7.1,
§7.3, §7.6 and D13 Amendments 1, 2, 3. CPU only, no model, no API, the key file is never
opened. It edits no frozen file and no sealed file. It scores the sealed generation files
itself:
  E0   = the frozen extractor (efbpt_eval_dev200.extract_answer), imported unchanged;
  E3   = d11_scorer_audit.e3 (validated scorer), imported; script MD5 pinned;
  E0fa = freeze §7.1, for the arms T_tr_R1 and T_tr_L1 only: the first word-bounded yes/no
         after the LAST 'Final answer' (case-insensitive); if the marker is absent, E0.
         A row that has the marker and no yes/no after it is unparsed (the rule names E0
         only for an absent marker); such rows are counted in the readout.
  "label" below = E0, or E0fa for the two T_tr arms.
Conventions (freeze): exact two-sided McNemar (binomial on the discordant pairs, as
d5_balanced_extract.mcnemar); unparsed = wrong; b = the first-named arm right and the second
wrong, c = the reverse; delta = accuracy of the first-named arm minus the second, in pp.

Lines of a comparison (D11 A5.3; freeze §3; Amendment 1 A1.2; Amendment 2):
  L1  label, all rows
  L2  E3, all rows
  L3  label, rows parsed in both arms
  L4  rows where plain E0 and E3 give the same label in both arms (descriptive)
  L5  label, the 408 eval458 rows left after dropping the first 50 rows of
      data/sdfr_splits/strategyqa_eval.jsonl (eval458 only)
  L6  label, rows truncated in neither arm
  L7  label, rows parsed in both arms and truncated in neither
  A3.3 line (comparisons with API_tr_R1 on eval458): label, the 456 rows the provider answered
The two refused rows of API_tr_R1 on eval458 (Amendment 3) have no answer: they are unparsed
under every scorer, wrong on every line, and not truncated.
Deciding line of a test (Amendment 2 A2.2): the unparsed rule fires when the unparsed rates
(label) of the two arms differ by more than 5.0 pp; the truncation rule fires when the
truncated rates differ by more than 5.0 pp. Neither: L1. Unparsed only: L3. Truncation only:
L6. Both: L7. The L1 numbers are always printed beside the deciding line.
Families (freeze §3): F1 = TR vs the 7 Group 1-2 arms, F2 = S_TR vs S_A for 3 students; Holm
step-down at 0.05 inside each family on the raw p of the deciding lines.
Wording (freeze §4): "TR beats X" only if delta > 0 and Holm p < 0.05; "X beats TR" only if
delta < 0 and Holm p < 0.05; else "not distinguishable". Transfer "holds" if at least 2 of 3
students have delta > 0 with Holm p < 0.05; "partial" if 1; "not shown" if 0.
Predictions (freeze §5) are read on L1 of eval458.

SELF-TEST, every run, all fatal, before any D13 arm is opened:
  S1  the sealed DEV200 reference arms give the E0 numbers of freeze §1, and TR vs A_r2 gives
      the L1-L4 rows of docs/handoff/D11_DEV200_READOUT_LINES.md (MD5 pinned);
  S2  the sealed D11 eval458 arms give the E0 numbers of freeze §1, the per-arm rows and the
      L1-L4 rows of docs/handoff/D11_EVAL458_READOUT_LINES.md (MD5 pinned) for TR vs A_r2_eval
      (and TR_L1_eval vs A_r2_eval);
  S3  hand-made Holm step-down cases;
  S4  hand-made E0fa cases;
  S5  hand-made rows: lines L1-L7, the deciding-line rule, a refused row, the wording rules
      and the title share.

  --selftest-only        run S1-S5 and stop. No D13 arm, retrieval file or seal is opened.
  --synthetic-root DIR   testing hook: write synthetic arm files (made-up generations, real
                         qids) under DIR in the D13 file format, then run the whole readout on
                         them. Output DIR/D13_READOUT_SYNTHETIC.md. No D13 arm is opened.
  (no flag)              the readout. Refuses to run unless this script and every d13 script
                         are committed, unchanged and contained in origin/main, and all 19 arm
                         files are sealed. Writes docs/handoff/D13_READOUT.md; an existing file
                         is never overwritten with different content.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d13_common as C  # noqa: E402  (chdir to the repo root; nothing in it scores)
from d13_common import die, md5_file, read_jsonl, load_set_rows, file_entry  # noqa: E402
from efbpt_eval_dev200 import extract_answer, RE_EN_WORD, HAAN, NAHIN, MARKER  # noqa: E402  E0, frozen

AUDIT_SCRIPT = os.path.join(_HERE, "d11_scorer_audit.py")
AUDIT_SCRIPT_MD5 = "246cf6c9687b3fa67324722cb5cd16b2"       # freeze: E3 = d11_scorer_audit.e3 at this MD5
if md5_file(AUDIT_SCRIPT) != AUDIT_SCRIPT_MD5:
    die("d11_scorer_audit.py MD5 %s != pinned %s" % (md5_file(AUDIT_SCRIPT), AUDIT_SCRIPT_MD5))
from d11_scorer_audit import e3 as extract_e3  # noqa: E402  E3, validated scorer

AMENDMENTS = OrderedDict([
    ("d13_amendment_1", ("docs/EFBPT_D13_AMENDMENT_1.md", "26bf37ea7e17146d44916b83cfa6e5d5")),
    ("d13_amendment_2", ("docs/EFBPT_D13_AMENDMENT_2.md", "12809c115ae8b6fba6bf6afe7fb6c51c")),
    ("d13_amendment_3", ("docs/EFBPT_D13_AMENDMENT_3.md", "344590e857e9f3ac075519120892e5d6")),
])
D11_DEV200_LINES = ("docs/handoff/D11_DEV200_READOUT_LINES.md", "888b477fd42c9d2886e384655bab2443")
D11_EVAL458_LINES = ("docs/handoff/D11_EVAL458_READOUT_LINES.md", "9f95a95ecd0a92fe6362b438c3f900a4")   # freeze, first page
OUT_PATH = "docs/handoff/D13_READOUT.md"
REAL_OUT_ROOT = "outputs/efbpt/d13"

ALPHA = 0.05
RULE_PP = 5                                   # D11 A3.1 / Amendment 1 A1.3: more than 5.0 pp
FA_ARMS = ("T_tr_R1", "T_tr_L1")              # freeze §7.1: the label is E0fa
REFUSED_ARM = ("API_tr_R1", "eval458")        # Amendment 3
REFUSED_QIDS = ("90538ec7aa1100a34829", "78f2f99d04b9acd8bada")
N_ANSWERED = 456
L5_YES_NO = (197, 211)                        # freeze §3, checked 2026-11-12

# sealed reference arms (freeze §1): name in the readout -> key of d13_common.SEALED, E0 correct
REFS = OrderedDict([
    ("dev200", OrderedDict([("A_r2", ("dev200/A_r2", 120)), ("TR", ("dev200/TR", 137)), ("TR_L1", ("dev200/TR_L1", 134)),
                            ("T_direct_F", ("dev200/T_direct_F", 111)), ("API_direct_ur", ("dev200/API_direct_ur", 138)),
                            ("RO", ("dev200/RO", 115))])),
    ("eval458", OrderedDict([("A_r2_eval", ("eval458/A_r2_eval", 251)), ("TR", ("eval458/TR", 315)),
                             ("TR_L1_eval", ("eval458/TR_L1_eval", 296))])),
])
REF_ACC = {("dev200", "A_r2"): "60.00", ("dev200", "TR"): "68.50", ("dev200", "TR_L1"): "67.00", ("dev200", "T_direct_F"): "55.50",
           ("dev200", "API_direct_ur"): "69.00", ("dev200", "RO"): "57.50",
           ("eval458", "A_r2_eval"): "54.80", ("eval458", "TR"): "68.78", ("eval458", "TR_L1_eval"): "64.63"}

# freeze §3
F1 = [("TR", x) for x in ("T_direct_F_eval", "T_tr_R1", "T_tr_L1", "API_direct_ur_eval", "API_tr_R1", "RAG_ur", "RAG_en")]
F2 = [("Llama8B_TR", "Llama8B_A"), ("Gemma9B_TR", "Gemma9B_A"), ("Qwen8B_TR", "Qwen8B_A")]
DESC_EVAL = [("TR_L1_eval", "T_direct_F_eval"), ("TR_L1_eval", "T_tr_L1"), ("TR", "RO_eval"), ("RO_eval", "A_r2_eval"),
             ("RAG_ur", "A_r2_eval"), ("RAG_en", "A_r2_eval")]
DESC_DEV = [("TR", "T_direct_F"), ("TR", "T_tr_R1"), ("TR", "T_tr_L1"), ("TR", "API_direct_ur"), ("TR", "API_tr_R1"),
            ("TR", "RAG_ur"), ("TR", "RAG_en"), ("RAG_ur", "A_r2"), ("RAG_en", "A_r2"), ("TR_L1", "T_tr_L1")]
EVAL_ORDER = ["A_r2_eval", "TR", "TR_L1_eval"] + list(C.ARMS)
DEV_ORDER = list(REFS["dev200"]) + [a for a, s in C.ARMS.items() if "dev200" in s["sets"]]

RE_FINAL_ANSWER = re.compile(r"final answer", re.IGNORECASE)


# ----------------------------------------------------------------------------
# scorers
# ----------------------------------------------------------------------------
def e0fa(text):
    """freeze §7.1. Returns (label, marker_present)."""
    last = None
    for last in RE_FINAL_ANSWER.finditer(text):
        pass
    if last is None:
        return extract_answer(text), False
    m = RE_EN_WORD.search(text, last.end())
    return (m.group(1).lower() if m else None), True


def mk(arm, qid, gold, generation, truncated, refused=False, seconds=None, tokens=None):
    """One scored row. 'main' is the label of the arm: E0, or E0fa for the two T_tr arms."""
    if refused:
        if generation:
            die("%s qid %s: a refused row carries an answer" % (arm, qid))
        e0 = e3 = main = None
        marker = None
    else:
        e0, e3 = extract_answer(generation), extract_e3(generation)
        if arm in FA_ARMS:
            main, marker = e0fa(generation)
        else:
            main, marker = e0, None
    for v in (e0, e3, main):
        if v not in ("yes", "no", None):
            die("%s qid %s: a scorer returned %r" % (arm, qid, v))
    if gold not in ("yes", "no"):
        die("%s qid %s: gold is %r" % (arm, qid, gold))
    return {"qid": qid, "gold": gold, "e0": e0, "e3": e3, "main": main, "marker": marker, "trunc": bool(truncated),
            "refused": bool(refused), "seconds": seconds, "tokens": tokens}


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def exact_p(b, c):
    from scipy.stats import binomtest
    return binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0


def line(x, y, key, keep):
    """Paired comparison on the rows where keep(rx, ry). b = x right and y wrong."""
    if len(x) != len(y):
        die("paired arms have different lengths")
    n = cx = cy = b = c = 0
    for rx, ry in zip(x, y):
        if rx["qid"] != ry["qid"] or rx["gold"] != ry["gold"]:
            die("paired rows are not aligned")
        if not keep(rx, ry):
            continue
        ox, oy = rx[key] == rx["gold"], ry[key] == ry["gold"]
        n += 1
        cx += ox
        cy += oy
        b += ox and not oy
        c += oy and not ox
    nan = float("nan")
    return OrderedDict([("n", n), ("cx", cx), ("cy", cy), ("acc_x", 100.0 * cx / n if n else nan), ("acc_y", 100.0 * cy / n if n else nan),
                        ("delta", 100.0 * (b - c) / n if n else nan), ("b", b), ("c", c), ("p", exact_p(b, c))])


def all_lines(x, y, l5_qids=None, answered_line=False):
    out = OrderedDict()
    out["L1"] = line(x, y, "main", lambda a, b: True)
    out["L2"] = line(x, y, "e3", lambda a, b: True)
    out["L3"] = line(x, y, "main", lambda a, b: a["main"] is not None and b["main"] is not None)
    out["L4"] = line(x, y, "e0", lambda a, b: a["e0"] == a["e3"] and b["e0"] == b["e3"])
    if l5_qids is not None:
        out["L5"] = line(x, y, "main", lambda a, b: a["qid"] in l5_qids)
    out["L6"] = line(x, y, "main", lambda a, b: not a["trunc"] and not b["trunc"])
    out["L7"] = line(x, y, "main", lambda a, b: a["main"] is not None and b["main"] is not None and not a["trunc"] and not b["trunc"])
    if answered_line:
        out["A3.3"] = line(x, y, "main", lambda a, b: not a["refused"] and not b["refused"])
    return out


LINE_LABEL = OrderedDict([
    ("L1", "L1  label, all rows"), ("L2", "L2  E3, all rows"), ("L3", "L3  label, rows parsed in both arms"),
    ("L4", "L4  rows where E0 = E3 in both arms (descriptive)"), ("L5", "L5  label, the 408 rows (first 50 probe rows dropped)"),
    ("L6", "L6  label, rows truncated in neither arm"), ("L7", "L7  label, parsed in both and truncated in neither"),
    ("A3.3", "A3.3  label, the 456 rows the provider answered"),
])


def n_unparsed(recs):
    return sum(1 for r in recs if r["main"] is None)


def n_trunc(recs):
    return sum(1 for r in recs if r["trunc"])


def deciding(x, y):
    """Amendment 2 A2.2. 'more than 5.0 pp' in integers: 100 * |difference in rows| > 5 * n."""
    n = len(x)
    unp = 100 * abs(n_unparsed(x) - n_unparsed(y)) > RULE_PP * n
    tru = 100 * abs(n_trunc(x) - n_trunc(y)) > RULE_PP * n
    return ("L7" if unp and tru else "L3" if unp else "L6" if tru else "L1"), unp, tru


def holm(ps):
    """freeze §3: the adjusted p of the i-th smallest of m is the largest of min(1, (m - j + 1) * p_(j)) over j <= i."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: (ps[i], i))
    adj, run = [None] * m, 0.0
    for rank, i in enumerate(order):             # rank 0 is the smallest: factor m
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def wording(x, y, delta_rows, holm_p):
    """freeze §4. delta_rows = b - c on the deciding line."""
    if holm_p < ALPHA and delta_rows > 0:
        return "%s beats %s" % (x, y)
    if holm_p < ALPHA and delta_rows < 0:
        return "%s beats %s" % (y, x)
    return "not distinguishable"


def transfer_word(k):
    return "holds" if k >= 2 else "partial" if k == 1 else "not shown"


def arm_stats(recs, key):
    n = len(recs)
    yes = [r for r in recs if r["gold"] == "yes"]
    no = [r for r in recs if r["gold"] == "no"]
    ay = 100.0 * sum(1 for r in yes if r[key] == "yes") / len(yes)
    an = 100.0 * sum(1 for r in no if r[key] == "no") / len(no)
    correct = sum(1 for r in recs if r[key] == r["gold"])
    unp = sum(1 for r in recs if r[key] is None)
    return OrderedDict([("n", n), ("correct", correct), ("acc", 100.0 * correct / n), ("unparsed", unp), ("unparsed_pct", 100.0 * unp / n),
                        ("yes_rate", 100.0 * sum(1 for r in recs if r[key] == "yes") / n),
                        ("acc_gold_yes", ay), ("acc_gold_no", an), ("balanced", (ay + an) / 2.0)])


def stat_cells(s):
    return ["%.2f (%d)" % (s["acc"], s["correct"]), "%d (%.2f)" % (s["unparsed"], s["unparsed_pct"]), "%.2f" % s["yes_rate"],
            "%.2f" % s["acc_gold_yes"], "%.2f" % s["acc_gold_no"], "%.2f" % s["balanced"]]


def fp(p):
    return "%.4g" % p


def line_cells(ln):
    return ["%d" % ln["n"], "%.2f" % ln["acc_x"], "%.2f" % ln["acc_y"], "%+.2f" % ln["delta"], "%d" % ln["b"], "%d" % ln["c"], fp(ln["p"])]


def title_share(rows, retr, tmap):
    """freeze §7.3. Gold titles of a row = row['required']; a gold title is found if its corpus
    title (title map) equals the title of at least one of the 5 chunks."""
    n_gold = n_present = n_found = rows_gold = rows_any = 0
    for row, e in zip(rows, retr):
        if row["qid"] != e["qid"]:
            die("title share: retrieval rows are not aligned with the set")
        top = {h["title"] for h in e["hits"]}
        gold = [t for t in row["required"] if t]
        hit = False
        for t in gold:
            d = tmap.get(t)
            if d is None:
                die("title share: gold title %r is not in the title map" % t)
            n_gold += 1
            if d["present"]:
                n_present += 1
                if d["corpus_title"] in top:
                    n_found += 1
                    hit = True
        rows_gold += bool(gold)
        rows_any += hit
    return OrderedDict([("n_rows", len(rows)), ("n_gold", n_gold), ("n_present", n_present), ("n_found", n_found),
                        ("rows_with_gold", rows_gold), ("rows_any_found", rows_any)])


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_ref(set_name, name, rows):
    """A sealed reference arm of freeze §1 (D10 / D11), MD5 pinned in d13_common.SEALED."""
    key, _ = REFS[set_name][name]
    path, want = C.SEALED[key]
    C.verify_sealed((key,))
    recs = read_jsonl(path)
    if [r["qid"] for r in recs] != [r["qid"] for r in rows]:
        die("%s: qids are not %s in order" % (path, set_name))
    out = []
    for r, row in zip(recs, rows):
        if "gold" in r and r["gold"] != row["gold"]:
            die("%s qid %s: gold differs from the item set" % (path, r["qid"]))
        gen = r["generation"] if "generation" in r else r["raw"]           # the sealed API file stores the answer as 'raw'
        trunc = r["truncated"] if "truncated" in r else (r.get("finish_reason") == "length")
        x = mk(name, r["qid"], row["gold"], gen, trunc, tokens=r.get("n_gen_tokens", r.get("completion_tokens")))
        if "pred" in r and x["e0"] != r["pred"]:
            die("%s qid %s: E0 does not reproduce the sealed pred" % (path, r["qid"]))
        out.append(x)
    return out, OrderedDict([("file", path), ("md5", want), ("seal", "sealed reference, MD5 pinned (freeze §1)")])


def load_d13(arm, set_name, rows):
    """A sealed D13 arm: seal verified (fields, MD5, bytes, lines), rows in item order, no scoring field; gold joined here."""
    pth = C.arm_paths(arm, set_name, "FULL")
    for p in (pth["seal"], pth["rows"], pth["start"]):
        if not os.path.exists(p):
            die("%s on %s is not sealed: %s missing" % (arm, set_name, p))
    seal = json.load(open(pth["seal"], encoding="utf-8"), object_pairs_hook=OrderedDict)
    for k, want in (("experiment", "D13"), ("arm", arm), ("set", set_name), ("mode", "FULL"), ("complete", True),
                    ("n_rows", len(rows)), ("scored", False), ("d13_freeze_md5", C.D13_FREEZE_MD5)):
        if seal.get(k) != want:
            die("%s: %s is %r, expected %r" % (pth["seal"], k, seal.get(k), want))
    if not {pth["rows"], pth["start"]} <= set(seal["files"]):
        die("%s does not cover the row file and the START file" % pth["seal"])
    for p, want in seal["files"].items():
        got = file_entry(p)
        if (got["md5"], got["bytes"], got["lines"]) != (want["md5"], want["bytes"], want["lines"]):
            die("%s: %s differs from the seal" % (pth["seal"], p))
    recs = C.read_rows(pth["rows"])
    if [r["qid"] for r in recs] != [r["qid"] for r in rows]:
        die("%s: rows are not the %d %s qids in order" % (pth["rows"], len(rows), set_name))
    is_ref = (arm, set_name) == REFUSED_ARM
    out, refused = [], []
    for r, row in zip(recs, rows):
        if r.get("arm") != arm or r.get("set") != set_name:
            die("%s qid %s: arm / set fields are %r / %r" % (pth["rows"], r["qid"], r.get("arm"), r.get("set")))
        if any(k in r for k in ("pred", "correct", "gold")):
            die("%s qid %s: a D13 row carries a scoring field (freeze §7.6)" % (pth["rows"], r["qid"]))
        ref = bool(r.get("refused"))
        if ref and not is_ref:
            die("%s qid %s: a refused row outside %s on %s" % (pth["rows"], r["qid"], REFUSED_ARM[0], REFUSED_ARM[1]))
        if not ref and r.get("ok") is False:
            die("%s qid %s: a row that is neither ok nor refused" % (pth["rows"], r["qid"]))
        if ref:
            refused.append(r["qid"])
        out.append(mk(arm, r["qid"], row["gold"], r["generation"], r["truncated"], refused=ref, seconds=r.get("seconds"), tokens=r.get("n_gen_tokens")))
    if is_ref:
        if sorted(refused) != sorted(REFUSED_QIDS) or seal.get("n_ok") != N_ANSWERED or seal.get("n_refused") != len(REFUSED_QIDS) \
                or sorted(seal.get("refused_qids") or []) != sorted(REFUSED_QIDS):
            die("%s on %s: the refused rows are not the two rows of Amendment 3 (rows %s; seal n_ok %r, n_refused %r)"
                % (arm, set_name, refused, seal.get("n_ok"), seal.get("n_refused")))
    if seal.get("rows_truncated") != n_trunc(out):
        die("%s: rows_truncated %r != %d truncated rows in the file" % (pth["seal"], seal.get("rows_truncated"), n_trunc(out)))
    start = json.load(open(pth["start"], encoding="utf-8"), object_pairs_hook=OrderedDict)
    prov = OrderedDict([("file", pth["rows"]), ("md5", seal["files"][pth["rows"]]["md5"]),
                        ("seal", "%s (md5 %s; %s; %s; %d files verified)" % (pth["seal"], md5_file(pth["seal"]), ("slurm %s" % seal["slurm_job_id"]) if seal.get("slurm_job_id") else "login node, no slurm job",
                                           seal.get("end_time"), len(seal["files"])))])
    return out, prov, seal, start


def load_title_map(set_name):
    pth = C.retrieval_paths("RAG_ur", set_name, "FULL")
    seal = json.load(open(pth["title_seal"], encoding="utf-8"), object_pairs_hook=OrderedDict)
    want = seal["files"].get(pth["title_map"])
    if want is None or file_entry(pth["title_map"])["md5"] != want["md5"]:
        die("%s: the title map does not match its seal" % set_name)
    recs = read_jsonl(pth["title_map"])
    return OrderedDict((r["gold_title"], r) for r in recs), OrderedDict([("file", pth["title_map"]), ("md5", want["md5"]), ("n_gold_titles", len(recs)), ("n_present", sum(1 for r in recs if r["present"]))])


def load_l5_qids(rows):
    """freeze §3 L5: the eval458 rows after dropping the first 50 rows of the pinned file (in file order)."""
    path, want = C.FIRST50_FILE
    if md5_file(path) != want:
        die("%s MD5 %s != pinned %s" % (path, md5_file(path), want))
    qids = [r["qid"] for r in read_jsonl(path)]
    if qids != [r["qid"] for r in rows]:
        die("%s does not hold the eval458 qids in the order of the eval458 loader (freeze §3)" % path)
    keep = set(qids[C.FIRST50_N:])
    kept = [r for r in rows if r["qid"] in keep]
    yn = (sum(1 for r in kept if r["gold"] == "yes"), sum(1 for r in kept if r["gold"] == "no"))
    if len(kept) != C.L5_ROWS or yn != L5_YES_NO:
        die("L5: %d rows remain (%d yes / %d no); freeze §3 says %d (%d / %d)" % (len(kept), yn[0], yn[1], C.L5_ROWS, L5_YES_NO[0], L5_YES_NO[1]))
    return keep


def require_committed_and_pushed():
    """This script and every d13 script: committed, unchanged, and HEAD contained in origin/main."""
    C.require_committed()
    me = os.path.relpath(os.path.abspath(__file__), C.ROOT)

    def git(*a):
        return subprocess.run(["git"] + list(a), cwd=C.ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if git("ls-files", "--error-unmatch", me).returncode != 0:
        die("readout refused: %s is not committed" % me)
    if git("merge-base", "--is-ancestor", "HEAD", "origin/main").returncode != 0:
        die("readout refused: HEAD is not contained in origin/main (the script must be pushed before it reads a D13 arm)")
    head = git("rev-parse", "HEAD").stdout.decode().strip()
    print("[ok] %s is committed, unchanged and HEAD %s is contained in origin/main" % (me, head))
    return head


# ----------------------------------------------------------------------------
# self-test (no D13 arm is opened)
# ----------------------------------------------------------------------------
def doc_rows(path, want_md5, start_prefix, row_prefix):
    if md5_file(path) != want_md5:
        die("%s MD5 %s != pinned %s" % (path, md5_file(path), want_md5))
    lines = open(path, encoding="utf-8").read().split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(start_prefix)]
    if len(idx) != 1:
        die("%s: %d lines start with %r" % (path, len(idx), start_prefix))
    out = []
    for l in lines[idx[0] + 1:]:
        if l.startswith(row_prefix):
            out.append([c.strip() for c in l.strip().strip("|").split("|")])
        elif out:
            break
    return out


def chk(what, got, want):
    if got != want:
        die("SELF-TEST FAILED: %s is %r, expected %r" % (what, got, want))


def selftest_refs(set_name, rows, doc, pairs, per_arm_names):
    """S1 / S2: E0 numbers of freeze §1; L1-L4 (and per-arm rows) of the committed D11 readout lines."""
    arms = OrderedDict()
    for name in REFS[set_name]:
        arms[name], _ = load_ref(set_name, name, rows)
        s = arm_stats(arms[name], "e0")
        chk("%s %s E0 correct (freeze §1)" % (set_name, name), s["correct"], REFS[set_name][name][1])
        chk("%s %s E0 accuracy (freeze §1)" % (set_name, name), "%.2f" % s["acc"], REF_ACC[(set_name, name)])
    n_cells = 0
    for x, y in pairs:
        want = doc_rows(doc[0], doc[1], "**%s vs %s — " % (x, y), "| L")
        if len(want) != 4:
            die("%s: %d line rows under %s vs %s, expected 4" % (doc[0], len(want), x, y))
        got = all_lines(arms[x], arms[y])
        for k, w in zip(("L1", "L2", "L3", "L4"), want):
            if not w[0].startswith(k):
                die("%s: row %r is not %s" % (doc[0], w[0], k))
            chk("%s %s vs %s %s (n, acc, acc, delta, b, c, p)" % (set_name, x, y, k), line_cells(got[k]), w[1:])
            n_cells += 7
    for name in per_arm_names:
        for key, sc in (("e0", "E0"), ("e3", "E3")):
            want = doc_rows(doc[0], doc[1], "| arm | scorer |", "| %s | %s |" % (name, sc))
            if len(want) != 1:
                die("%s: no per-arm row for %s %s" % (doc[0], name, sc))
            chk("%s per-arm row %s %s" % (set_name, name, sc), stat_cells(arm_stats(arms[name], key)), want[0][2:])
            n_cells += 6
    return arms, n_cells


def selftest_handmade():
    # S3 Holm
    def close(a, b):
        return len(a) == len(b) and all(abs(u - v) < 1e-12 for u, v in zip(a, b))
    cases = [
        ([0.01, 0.04, 0.03, 0.005], [0.03, 0.06, 0.06, 0.02]),      # 4*0.005, 3*0.01, 2*0.03, then 0.04 lifted to 0.06
        ([0.5, 0.6, 0.7], [1.0, 1.0, 1.0]),                         # capped at 1, stays at 1
        ([0.02], [0.02]),
        ([0.007] * 7, [0.049] * 7),
        ([0.001, 0.02, 0.04], [0.003, 0.04, 0.04]),
        ([0.04, 0.001, 0.02], [0.04, 0.003, 0.04]),                 # same as above, input order kept
        ([0.008, 0.008, 0.9, 0.9, 0.9, 0.9, 0.9], [0.056, 0.056, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ]
    for ps, want in cases:
        if not close(holm(ps), want):
            die("SELF-TEST FAILED: Holm %s -> %s, expected %s" % (ps, holm(ps), want))
    # S4 E0fa
    fa_cases = [
        ("Some reasoning.\nFinal answer: yes", "yes", True),
        ("Final answer: no. There is no doubt, yes indeed.", "no", True),          # plain E0 reads the last word: yes
        ("Final answer: yes\nNote: no further information is needed.", "yes", True),
        ("final ANSWER: No", "no", True),
        ("Final answer: yes. Wait.\nFinal answer: no", "no", True),                # the LAST marker
        ("**Final answer:** Yes", "yes", True),
        ("Final answer: yesterday is not a word here, no", "no", True),            # word-bounded
        ("no, I think. Final answer: unknown", None, True),                        # marker, nothing after it: unparsed
        ("Yes, it is.", "yes", False),                                             # marker absent: E0
        ("x " + HAAN + " x", "yes", False),                                        # marker absent: E0 (Urdu rule)
        ("x y z", None, False),
    ]
    for text, want, marker in fa_cases:
        chk("E0fa(%r)" % text, e0fa(text), (want, marker))
    chk("plain E0 on the 'no doubt, yes' case", extract_answer("Final answer: no. There is no doubt, yes indeed."), "yes")
    chk("mk(T_tr_R1) uses E0fa", mk("T_tr_R1", "q", "no", "Final answer: no. There is no doubt, yes indeed.", False)["main"], "no")
    chk("mk(RAG_ur) uses E0", mk("RAG_ur", "q", "no", "Final answer: no. There is no doubt, yes indeed.", False)["main"], "yes")
    chk("mk: Urdu marker cut", mk("RAG_ur", "q", "yes", NAHIN + " x " + MARKER + ": " + HAAN, False)["main"], "yes")

    # S5 lines, deciding rule, refused row, wording, title share
    def arm(name, spec):
        """spec: list of (gold, text, truncated, refused)"""
        return [mk(name, "q%02d" % i, g, t, tr, refused=rf) for i, (g, t, tr, rf) in enumerate(spec)]
    Y, N, U = HAAN, NAHIN, "x y z"
    golds = ["yes"] * 10 + ["no"] * 10
    x = arm("X", [(g, (Y if g == "yes" else N), False, False) for g in golds])                     # 20 right
    ytxt = [Y, Y, Y, Y, Y, N, N, U, U, U] + [N, N, N, N, N, N, Y, Y, U, N]
    ytr = [False] * 4 + [True] + [False] * 13 + [False, True]
    y = arm("Y", [(g, t, tr, False) for g, t, tr in zip(golds, ytxt, ytr)])
    ls = all_lines(x, y, l5_qids={"q%02d" % i for i in range(5, 20)})
    chk("hand L1 (n, cx, cy, b, c)", tuple(ls["L1"][k] for k in ("n", "cx", "cy", "b", "c")), (20, 20, 12, 8, 0))
    chk("hand L1 p", round(ls["L1"]["p"], 10), round(2 * 0.5 ** 8, 10))
    chk("hand L3 (n, cx, cy, b, c)", tuple(ls["L3"][k] for k in ("n", "cx", "cy", "b", "c")), (16, 16, 12, 4, 0))
    chk("hand L5 (n, cx, cy)", tuple(ls["L5"][k] for k in ("n", "cx", "cy")), (15, 15, 7))
    chk("hand L6 (n, cx, cy)", tuple(ls["L6"][k] for k in ("n", "cx", "cy")), (18, 18, 10))
    chk("hand L7 (n, cx, cy)", tuple(ls["L7"][k] for k in ("n", "cx", "cy")), (14, 14, 10))
    chk("hand deciding: 4 unparsed of 20 = 20 pp, 2 truncated = 10 pp -> both", deciding(x, y), ("L7", True, True))
    chk("hand deciding: equal arms", deciding(x, x), ("L1", False, False))
    one_unp = arm("Z", [(g, (U if i == 0 else Y if g == "yes" else N), False, False) for i, g in enumerate(golds)])
    chk("hand deciding: 1 of 20 = exactly 5.0 pp does not fire", deciding(x, one_unp), ("L1", False, False))
    two_unp = arm("Z", [(g, (U if i < 2 else Y if g == "yes" else N), False, False) for i, g in enumerate(golds)])
    chk("hand deciding: 2 of 20 = 10 pp -> L3", deciding(x, two_unp), ("L3", True, False))
    two_tr = arm("Z", [(g, (Y if g == "yes" else N), i < 2, False) for i, g in enumerate(golds)])
    chk("hand deciding: truncation only -> L6", deciding(x, two_tr), ("L6", False, True))
    r = arm("R", [(g, ("" if i in (0, 10) else Y if g == "yes" else N), False, i in (0, 10)) for i, g in enumerate(golds)])
    lr = all_lines(x, r, answered_line=True)
    chk("hand refused: wrong on L1, unparsed, not truncated", (lr["L1"]["cy"], n_unparsed(r), n_trunc(r), lr["L2"]["cy"]), (18, 2, 0, 18))
    chk("hand refused: dropped from L3 and from the A3.3 line", (lr["L3"]["n"], lr["A3.3"]["n"], lr["A3.3"]["cy"], lr["L4"]["n"]), (18, 18, 18, 20))
    chk("wording +", wording("TR", "X", 30, 0.01), "TR beats X")
    chk("wording -", wording("TR", "X", -30, 0.01), "X beats TR")
    chk("wording p = 0.05 is not < 0.05", wording("TR", "X", 30, 0.05), "not distinguishable")
    chk("wording delta 0", wording("TR", "X", 0, 0.01), "not distinguishable")
    chk("transfer words", [transfer_word(k) for k in (0, 1, 2, 3)], ["not shown", "partial", "holds", "holds"])
    rows = [{"qid": "a", "required": ["T1", "T2"]}, {"qid": "b", "required": ["T3"]}, {"qid": "c", "required": []}, {"qid": "d", "required": ["T4"]}]
    tmap = {"T1": {"present": True, "corpus_title": "T1"}, "T2": {"present": False, "corpus_title": None},
            "T3": {"present": True, "corpus_title": "t3"}, "T4": {"present": True, "corpus_title": "T4"}}
    retr = [{"qid": "a", "hits": [{"title": "T1"}, {"title": "Z"}]}, {"qid": "b", "hits": [{"title": "T3"}]},
            {"qid": "c", "hits": [{"title": "T1"}]}, {"qid": "d", "hits": [{"title": "T4"}, {"title": "T4"}]}]
    chk("title share", tuple(title_share(rows, retr, tmap).values()), (4, 4, 3, 2, 3, 2))
    return len(cases), len(fa_cases)


def selftest():
    """Returns (report lines, reference arms by set, item rows by set). Opens the sealed D10 / D11 reference files only."""
    out = []
    rows = OrderedDict((s, load_set_rows(s)) for s in C.SETS)
    for s in C.SETS:
        shape = (len(rows[s]), sum(1 for r in rows[s] if r["gold"] == "yes"), sum(1 for r in rows[s] if r["gold"] == "no"))
        if shape != C.SET_SHAPE[s]:
            die("%s: rows / yes / no = %s, expected %s" % (s, shape, C.SET_SHAPE[s]))
    refs = OrderedDict()
    refs["dev200"], n1 = selftest_refs("dev200", rows["dev200"], D11_DEV200_LINES, [("TR", "A_r2"), ("TR_L1", "A_r2")], ("A_r2", "TR", "TR_L1"))
    out.append("S1 PASS: the 6 sealed DEV200 reference arms give the E0 numbers of freeze §1 (A_r2 60.00 (120), TR 68.50 (137), TR_L1 67.00 (134), "
               "T_direct_F 55.50 (111), API_direct_ur 69.00 (138), RO 57.50 (115)); TR vs A_r2 and TR_L1 vs A_r2 give the L1-L4 rows and the per-arm "
               "E0 / E3 rows of `%s` (MD5 `%s`), %d cells compared as printed." % (D11_DEV200_LINES[0], D11_DEV200_LINES[1], n1))
    refs["eval458"], n2 = selftest_refs("eval458", rows["eval458"], D11_EVAL458_LINES, [("TR", "A_r2_eval"), ("TR_L1_eval", "A_r2_eval")], ("A_r2_eval", "TR", "TR_L1_eval"))
    out.append("S2 PASS: the 3 sealed D11 eval458 arms give the E0 numbers of freeze §1 (A_r2_eval 54.80 (251), TR 68.78 (315), TR_L1_eval 64.63 (296)); "
               "TR vs A_r2_eval and TR_L1_eval vs A_r2_eval give the L1-L4 rows and the per-arm E0 / E3 rows of `%s` (MD5 `%s`), %d cells compared as printed."
               % (D11_EVAL458_LINES[0], D11_EVAL458_LINES[1], n2))
    nh, nf = selftest_handmade()
    out.append("S3 PASS: Holm step-down, %d hand-made cases (ties, the cap at 1, the running maximum, input order kept)." % nh)
    out.append("S4 PASS: E0fa, %d hand-made cases (last marker, first yes/no after it, case, word bounds, marker without an answer = unparsed, no marker = E0)." % nf)
    out.append("S5 PASS: hand-made rows: L1, L3, L5, L6, L7 counts and p, the deciding-line rule at and above 5.0 pp, a refused row "
               "(wrong, unparsed, not truncated, left out of L3 and of the A3.3 line), the wording rules, the title share.")
    return out, refs, rows


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def report(arms, prov, seals, starts, rows, retrieval, selftest_lines, head, synthetic):
    L = []
    w = L.append
    ev, dv = arms["eval458"], arms["dev200"]
    w("# D13 readout — baselines and transfer%s" % (" — SYNTHETIC DATA, A TEST OF THE SCRIPT, NOT A RESULT" if synthetic else ""))
    w("")
    w("Written by `eval/error_analysis_tests/efbpt/d13_readout.py` (MD5 `%s`, git HEAD `%s`), one run, stage 1 (Groups 1-3) and stage 2 (Group 4) together: "
      "all 19 D13 arm files were sealed before it started. Contract: freeze `%s`; Amendments 1 `%s`, 2 `%s`, 3 `%s`. "
      "E0 = the frozen extractor; E3 = `d11_scorer_audit.e3` (script MD5 `%s`); E0fa (freeze §7.1) for T_tr_R1 and T_tr_L1. "
      "\"label\" = E0, or E0fa for the two T_tr arms. b = the first-named arm right and the second wrong; c = the reverse; p = exact two-sided McNemar; "
      "unparsed = wrong; Δ = first-named minus second, in pp. eval458 decides (n = 458, gold yes 221 / no 237); DEV200 is descriptive (n = 200, 87 / 113)."
      % (md5_file(os.path.abspath(__file__)), head, C.D13_FREEZE_MD5, AMENDMENTS["d13_amendment_1"][1], AMENDMENTS["d13_amendment_2"][1],
         AMENDMENTS["d13_amendment_3"][1], AUDIT_SCRIPT_MD5))
    w("")
    w("The D11 verdict CONFIRMED stays as it is whatever this readout shows (freeze, first page). No arm is run again after this readout.")
    w("")
    w("## 0. Self-test (run before any D13 arm was opened; every check fatal)")
    w("")
    for s in selftest_lines:
        w("- " + s)
    w("")

    # ---- 1. sealed files ----
    w("## 1. Sealed files")
    w("")
    w("| set | arm | file | MD5 | seal |")
    w("|---|---|---|---|---|")
    for s, order in (("eval458", EVAL_ORDER), ("dev200", DEV_ORDER)):
        for a in order:
            pv = prov[s][a]
            w("| %s | %s | `%s` | `%s` | %s |" % (s, a, pv["file"], pv["md5"], pv["seal"]))
    w("")
    w("API_tr_R1 on eval458 (Amendment 3): 456 ok rows and 2 rows refused by the provider's input inspection (qids %s). The refused rows have no answer: "
      "unparsed under every scorer, wrong on every line, not truncated." % ", ".join(REFUSED_QIDS))
    w("")

    # ---- 2. per arm ----
    stats = {}

    def per_arm(set_name, order, title):
        w("### %s" % title)
        w("")
        w("| arm | scorer | acc % (correct) | unparsed (%) | predicted yes % | acc on gold-yes % | acc on gold-no % | balanced acc % |")
        w("|---|---|---|---|---|---|---|---|")
        for a in order:
            recs = arms[set_name][a]
            scorers = [("main", "E0fa (label)"), ("e0", "E0 (plain)"), ("e3", "E3")] if a in FA_ARMS else [("e0", "E0 (label)"), ("e3", "E3")]
            for key, name in scorers:
                s = arm_stats(recs, key)
                stats[(set_name, a, key)] = s
                w("| %s | %s | %s |" % (a, name, " | ".join(stat_cells(s))))
            stats[(set_name, a, "main")] = arm_stats(recs, "main")
        w("")
        w("Rows where plain E0 and E3 differ: " + "; ".join("%s %d" % (a, sum(1 for r in arms[set_name][a] if r["e0"] != r["e3"])) for a in order) + ".")
        w("")

    w("## 2. Per arm")
    w("")
    w("Balanced accuracy = mean of the gold-yes and gold-no accuracies. The sealed reference arms (A_r2_eval, TR, TR_L1_eval; on DEV200 A_r2, TR, TR_L1, "
      "T_direct_F, API_direct_ur, RO) are shown beside the D13 arms; their numbers are the committed D10 / D11 numbers.")
    w("")
    per_arm("eval458", EVAL_ORDER, "eval458 (n = 458)")
    per_arm("dev200", DEV_ORDER, "DEV200 (n = 200, descriptive)")

    # ---- 3. E0fa ----
    w("## 3. E0fa for T_tr_R1 and T_tr_L1 (freeze §7.1; Amendment 2 A2.1)")
    w("")
    w("E0fa replaces E0 for these two arms on L1, L3, L5, L6 and L7. L2 is E3 and L4 uses plain E0 and E3, as for every arm.")
    w("")
    w("| set | arm | E0fa acc % (correct) | plain E0 acc % (correct) | E3 acc % (correct) | rows without the marker | marker present, no yes/no after it (unparsed) | rows where E0fa and E0 differ |")
    w("|---|---|---|---|---|---|---|---|")
    for s in ("eval458", "dev200"):
        for a in FA_ARMS:
            recs = arms[s][a]
            sm, s0, s3 = (arm_stats(recs, k) for k in ("main", "e0", "e3"))
            w("| %s | %s | %.2f (%d) | %.2f (%d) | %.2f (%d) | %d | %d | %d |"
              % (s, a, sm["acc"], sm["correct"], s0["acc"], s0["correct"], s3["acc"], s3["correct"],
                 sum(1 for r in recs if r["marker"] is False), sum(1 for r in recs if r["marker"] is True and r["main"] is None),
                 sum(1 for r in recs if r["main"] != r["e0"])))
    w("")

    # ---- comparisons ----
    l5 = load_l5_qids(rows["eval458"])

    def comparison(set_name, x, y, title, with_rule):
        ax, ay = arms[set_name][x], arms[set_name][y]
        is_ref = set_name == REFUSED_ARM[1] and REFUSED_ARM[0] in (x, y)
        ls = all_lines(ax, ay, l5_qids=l5 if set_name == "eval458" else None, answered_line=is_ref)
        if set_name == "eval458" and ls["L5"]["n"] != C.L5_ROWS:
            die("L5 of %s vs %s has %d rows" % (x, y, ls["L5"]["n"]))
        if is_ref and ls["A3.3"]["n"] != N_ANSWERED:
            die("the A3.3 line of %s vs %s has %d rows" % (x, y, ls["A3.3"]["n"]))
        key, unp, tru = deciding(ax, ay)
        n = len(ax)
        w("")
        w("**%s vs %s — %s**" % (x, y, title))
        w("")
        w("| line | n | acc %s | acc %s | Δ pp | b | c | exact two-sided p |" % (x, y))
        w("|---|---|---|---|---|---|---|---|")
        for k, ln in ls.items():
            w("| %s | %s |" % (LINE_LABEL[k], " | ".join(line_cells(ln))))
        w("")
        ux, uy, tx, ty = n_unparsed(ax), n_unparsed(ay), n_trunc(ax), n_trunc(ay)
        txt = ("Unparsed (label): %s %d (%.2f %%), %s %d (%.2f %%), difference %.2f pp %s 5.0. Truncated: %s %d (%.2f %%), %s %d (%.2f %%), difference %.2f pp %s 5.0."
               % (x, ux, 100.0 * ux / n, y, uy, 100.0 * uy / n, 100.0 * abs(ux - uy) / n, ">" if unp else "≤",
                  x, tx, 100.0 * tx / n, y, ty, 100.0 * ty / n, 100.0 * abs(tx - ty) / n, ">" if tru else "≤"))
        if with_rule == "test":
            txt += " Deciding line (Amendment 2 A2.2): **%s**%s." % (key, "" if key == "L1" else "; L1 is reported beside it")
        elif with_rule == "desc":
            txt += " Descriptive, no claim; the A2.2 rule would read it on %s." % key
        else:
            txt += " Descriptive, raw p only."
        w(txt)
        return ls, key

    def family_table(name, tests, res):
        ps = [res[t][0][res[t][1]]["p"] for t in tests]
        adj = holm(ps)
        w("| test | deciding line | n | Δ pp | b | c | raw p | Holm p | L1 beside it: Δ pp, b / c, raw p | reading |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        out = OrderedDict()
        for t, hp in zip(tests, adj):
            ls, key = res[t]
            ln, l1 = ls[key], ls["L1"]
            if name == "F1":
                word = "**" + wording(t[0], t[1], ln["b"] - ln["c"], hp) + "**"
            else:
                word = "**%s**; Δ > 0 and Holm p < 0.05: %s" % (wording(t[0], t[1], ln["b"] - ln["c"], hp), "yes" if (ln["b"] > ln["c"] and hp < ALPHA) else "no")
            w("| %s vs %s | %s | %d | %+.2f | %d | %d | %s | %s | %+.2f, %d / %d, %s | %s |"
              % (t[0], t[1], key, ln["n"], ln["delta"], ln["b"], ln["c"], fp(ln["p"]), fp(hp), l1["delta"], l1["b"], l1["c"], fp(l1["p"]), word))
            out[t] = (ln, hp)
        w("")
        return out

    w("## 4. eval458 — family F1: TR vs the large model alone and vs traditional RAG (7 tests, Holm at 0.05)")
    res_f1 = OrderedDict()
    for i, t in enumerate(F1, 1):
        res_f1[t] = comparison("eval458", t[0], t[1], "F1 test %d of 7" % i, "test")
    w("")
    w("### F1 summary (freeze §3 Holm, §4 wording)")
    w("")
    out_f1 = family_table("F1", F1, res_f1)
    w("Wording allowed by freeze §4: \"TR beats X\" only if Δ > 0 and Holm p < 0.05; \"X beats TR\" only if Δ < 0 and Holm p < 0.05; everything else \"not distinguishable\".")
    a33 = res_f1[("TR", "API_tr_R1")][0]["A3.3"]
    w("")
    w("Amendment 3 A3.3, TR vs API_tr_R1 on the %d rows the provider answered: acc %.2f vs %.2f, Δ %+.2f pp, b %d / c %d, raw p = %s (reported beside the test; the test counts the 2 refused rows as wrong)."
      % (a33["n"], a33["acc_x"], a33["acc_y"], a33["delta"], a33["b"], a33["c"], fp(a33["p"])))
    w("")

    w("## 5. eval458 — family F2: transfer to other students, S_TR vs S_A (3 tests, Holm at 0.05)")
    res_f2 = OrderedDict()
    for i, t in enumerate(F2, 1):
        res_f2[t] = comparison("eval458", t[0], t[1], "F2 test %d of 3" % i, "test")
    w("")
    w("### F2 summary")
    w("")
    out_f2 = family_table("F2", F2, res_f2)
    k_tr = sum(1 for t in F2 if out_f2[t][0]["b"] > out_f2[t][0]["c"] and out_f2[t][1] < ALPHA)
    w("Students with Δ > 0 and Holm p < 0.05: %d of 3. Freeze §4 (\"holds\" if at least 2 of 3, \"partial\" if 1, \"not shown\" if 0): transfer **%s**." % (k_tr, transfer_word(k_tr)))
    w("")

    w("## 6. eval458 — descriptive comparisons (freeze §3: no claim, no Holm, no wording)")
    res_desc = OrderedDict()
    for t in DESC_EVAL:
        res_desc[t] = comparison("eval458", t[0], t[1], "descriptive", "desc")
    w("")

    # ---- 7. predictions ----
    w("## 7. Predictions (freeze §5), read on L1 of eval458")
    w("")
    w("| prediction | observed | |")
    w("|---|---|---|")
    n = len(rows["eval458"])

    def cor(a):
        return stats[("eval458", a, "main")]["correct"]

    def hm(ok):
        return "hit" if ok else "miss"

    def acc_row(a, lo, hi):
        c = cor(a)
        w("| %s accuracy in [%d, %d] | %.2f | %s |" % (a, lo, hi, 100.0 * c / n, hm(lo * n <= 100 * c <= hi * n)))

    def diff_row(x, y, lo, hi):
        d = cor(x) - cor(y)
        w("| %s − %s in [%+d, %+d] | %+.2f | %s |" % (x, y, lo, hi, 100.0 * d / n, hm(lo * n <= 100 * d <= hi * n)))

    acc_row("T_direct_F_eval", 48, 60)
    acc_row("API_direct_ur_eval", 62, 72)
    acc_row("T_tr_R1", 60, 76)
    d = cor("T_tr_L1") - cor("T_tr_R1")
    w("| T_tr_L1 within 4 pp of T_tr_R1 | T_tr_L1 − T_tr_R1 = %+.2f | %s |" % (100.0 * d / n, hm(100 * abs(d) <= 4 * n)))
    diff_row("RAG_ur", "A_r2_eval", -6, 4)
    diff_row("RAG_en", "A_r2_eval", -6, 4)
    diff_row("RO_eval", "A_r2_eval", -3, 5)
    for x, y in F2:
        d = cor(x) - cor(y)
        w("| %s − %s > 0 | %+.2f | %s |" % (x, y, 100.0 * d / n, hm(d > 0)))
    t = ("TR", "API_direct_ur_eval")
    word = wording(t[0], t[1], out_f1[t][0]["b"] - out_f1[t][0]["c"], out_f1[t][1])
    w("| expected reading: TR and API_direct_ur_eval not distinguishable | %s | %s |" % (word, hm(word == "not distinguishable")))
    w("")
    w("T_tr_R1 and T_tr_L1 are read with E0fa (their L1 label). The expected reading is the F1 wording of that test.")
    w("")

    # ---- 8. DEV200 ----
    w("## 8. DEV200 — descriptive (freeze §3: lines L1-L4, raw p only, no wording; L6 and L7 added by Amendments 1 and 2; L5 is defined on eval458 only)")
    for t in DESC_DEV:
        comparison("dev200", t[0], t[1], "DEV200, descriptive", "dev")
    w("")

    # ---- 9. title share ----
    w("## 9. Retrieval title share (freeze §2, §7.3; D8 §3 title rule)")
    w("")
    w("Gold titles = the titles of the row's evidence ids. A gold title is found if its corpus title (sealed title map) equals the title of at least one of the 5 retrieved chunks.")
    w("")
    w("| set | arm | query | found / all gold titles | found / gold titles present in the corpus | rows with at least one found / rows | rows with at least one found / rows that have a gold title | retrieval s per query |")
    w("|---|---|---|---|---|---|---|---|")

    def frac(a, b):
        return "%d / %d (%.2f %%)" % (a, b, 100.0 * a / b) if b else "%d / 0" % a
    for s in ("eval458", "dev200"):
        for a in C.RAG_ARMS:
            ts, info = retrieval[(s, a)]
            w("| %s | %s | %s | %s | %s | %s | %s | %s |" % (s, a, info.get("query", "—"), frac(ts["n_found"], ts["n_gold"]), frac(ts["n_found"], ts["n_present"]),
                                                          frac(ts["rows_any_found"], ts["n_rows"]), frac(ts["rows_any_found"], ts["rows_with_gold"]), info.get("seconds_per_query", "—")))
    w("")
    for s in ("eval458", "dev200"):
        tm = retrieval[(s, "title_map")]
        w("Title map %s: `%s` (MD5 `%s`), %d distinct gold titles, %d present in the corpus." % (s, tm["file"], tm["md5"], tm["n_gold_titles"], tm["n_present"]))
    w("")

    # ---- 10. time, tokens, truncation ----
    w("## 10. Time, generated tokens and truncated rows per arm (freeze C6; Amendment 1)")
    w("")
    w("| set | arm | rows | truncated rows (%) | seconds per row (mean) | generated tokens per row (mean) | seconds total | generated tokens total |")
    w("|---|---|---|---|---|---|---|---|")

    def mean(v):
        return "%.2f" % (sum(v) / len(v)) if v else "—"
    for s, order in (("eval458", EVAL_ORDER), ("dev200", DEV_ORDER)):
        for a in order:
            recs = [r for r in arms[s][a] if not r["refused"]]
            secs = [r["seconds"] for r in recs if isinstance(r["seconds"], (int, float))]
            toks = [r["tokens"] for r in recs if isinstance(r["tokens"], (int, float))]
            nt = n_trunc(arms[s][a])
            w("| %s | %s | %d | %d (%.2f) | %s | %s | %s | %s |" % (s, a, len(arms[s][a]), nt, 100.0 * nt / len(arms[s][a]), mean(secs), mean(toks),
                                                                  "%.0f" % sum(secs) if secs else "—", "%d" % sum(toks) if toks else "—"))
    w("")
    w("Sealed reference arms carry no per-row time (—). API arms: seconds = latency of the request, tokens = completion tokens reported by the provider; "
      "API_tr_R1 on eval458: means and totals over the 456 answered rows. Truncated = the row reached the token limit (1024 new tokens for the local models; finish reason \"length\" for the API).")
    w("")

    # ---- 11. guard log ----
    w("## 11. Guard log from the START files (freeze §7.5; reported, not a check of this script)")
    w("")
    for s in ("eval458", "dev200"):
        for a in [x for x in (EVAL_ORDER if s == "eval458" else DEV_ORDER) if x in C.ARMS]:
            st = starts[s].get(a) or {}
            g = st.get("guards") or {}
            bits = []
            if "readings_equal_own_question_en" in g:
                bits.append("readings equal to the row's own question_en: %s of %s (reader %s)" % (g.get("readings_equal_own_question_en"), g.get("readings_checked"), g.get("reader")))
            hits = g.get("logged_hits_in_retrieved_text")
            if hits is not None:
                bits.append("strings inside retrieved text that equal an evidence id or a gold fact (logged, not fatal): %d" % len(hits)
                            + ("" if not hits else " — " + "; ".join("qid %s rank %s %s %r in \"%s\"" % (h.get("qid"), h.get("rank"), h.get("kind"), h.get("needle"), h.get("title")) for h in hits)))
            lay = (g.get(a) or {}).get("layouts") if isinstance(g.get(a), dict) else None
            if isinstance(lay, dict):
                bits.append("prompt layouts: %s (the user message is the sealed Qwen3-14B user message of the row; freeze §7.4)" % ", ".join("%s %s" % (k, v) for k, v in sorted(lay.items())))
            reg = st.get("regression")
            if isinstance(reg, dict):
                bits.append("C1 regression %s (job %s)" % (reg.get("verdict"), reg.get("slurm_job_id")))
            c2 = st.get("c2")
            if isinstance(c2, dict):
                rp = st.get("c2_repeat_vs_test5") or {}
                bits.append("C2 %s (job %s); repeat of the 5 S_TR rows in the full job identical to TEST-5: %s of %s" % (c2.get("verdict"), c2.get("slurm_job_id"), rp.get("n_identical"), rp.get("n_rows")))
            runs = st.get("runs")
            if isinstance(runs, list) and len(runs) > 1:
                bits.append("%d runs (resumed; finished rows were never generated again)" % len(runs))
            w("- %s %s: %s." % (s, a, "; ".join(bits) if bits else "no guard note"))
    w("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# synthetic testing hook: made-up generations in the D13 file format; no D13 arm is opened
# ----------------------------------------------------------------------------
def synth_build(root):
    import random
    C.OUT_ROOT = os.path.join(root, "outputs_d13")
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    for set_name in C.SETS:
        ctx = C.make_ctx(set_name, "FULL")
        rows = ctx["rows_all"]
        for arm, spec in C.ARMS.items():
            if set_name not in spec["sets"]:
                continue
            rng = random.Random("synthetic/%s/%s" % (arm, set_name))
            p_right, p_unp, p_tr = rng.uniform(0.45, 0.8), rng.choice([0.0, 0.02, 0.12]), rng.choice([0.0, 0.03, 0.15])
            pth = C.arm_paths(arm, set_name, "FULL")
            os.makedirs(pth["dir"], exist_ok=True)
            is_ref = (arm, set_name) == REFUSED_ARM
            recs = []
            for row in rows:
                right = rng.random() < p_right
                ans = row["gold"] if right else ("no" if row["gold"] == "yes" else "yes")
                other = "no" if ans == "yes" else "yes"
                if arm in FA_ARMS:
                    gen = rng.choice(["Reasoning. Final answer: %s" % ans, "Final answer: %s. There is %s doubt." % (ans, other), ans.capitalize() + ", it is."])
                else:
                    u = {"yes": HAAN, "no": NAHIN}
                    gen = rng.choice([u[ans], MARKER + ": " + u[ans], u[other] + "، x " + MARKER + " " + u[ans], u[ans] + " " + " ".join(["x"] * 14) + " " + u[ans]])
                if rng.random() < p_unp:
                    gen = "x y z"
                rec = OrderedDict([("qid", row["qid"]), ("arm", arm), ("set", set_name), ("truncated", rng.random() < p_tr),
                                   ("n_gen_tokens", rng.randint(1, 1024)), ("seconds", round(rng.uniform(0.2, 40), 3)), ("prompt", "synthetic"), ("generation", gen)])
                if is_ref and row["qid"] in REFUSED_QIDS:
                    rec.update([("ok", False), ("truncated", False), ("n_gen_tokens", None), ("generation", ""), ("refused", True)])
                elif spec["kind"] == "api":
                    rec["ok"] = True
                recs.append(rec)
            with open(pth["rows"], "wb") as f:
                for r in recs:
                    f.write(C.record_line(r))
            guards = OrderedDict()
            if spec.get("input") == "reading":
                guards.update([("reader", spec["reader"]), ("readings_checked", len(rows)), ("readings_equal_own_question_en", 3)])
            if arm in C.RAG_ARMS:
                guards["logged_hits_in_retrieved_text"] = [OrderedDict([("qid", rows[0]["qid"]), ("rank", 2), ("kind", "evidence_id"), ("needle", "B-1"), ("title", "Synthetic")])]
            C.write_json(pth["start"], OrderedDict([("experiment", "D13"), ("arm", arm), ("set", set_name), ("mode", "FULL"), ("n_rows", len(rows)),
                                                    ("guards", guards), ("scored", False), ("runs", [{"start_time": "synthetic"}])]))
            extra = OrderedDict([("n_ok", N_ANSWERED), ("n_refused", len(REFUSED_QIDS)), ("refused_qids", list(REFUSED_QIDS))]) if is_ref else None
            C.seal_arm(arm, set_name, "FULL", rows, extra)
        # retrieval files and title map
        gold = sorted({t for row in rows for t in row["required"] if t})
        rng = random.Random("synthetic/titles/" + set_name)
        tmap = OrderedDict()
        for t in gold:
            present = rng.random() < 0.7
            tmap[t] = OrderedDict([("gold_title", t), ("corpus_title", t if present else None), ("present", present), ("rule", "exact" if present else "absent_no_match"), ("norm_candidates", [])])
        pth = C.retrieval_paths("RAG_ur", set_name, "FULL")
        os.makedirs(pth["dir"], exist_ok=True)
        with open(pth["title_map"], "wb") as f:
            for t in gold:
                f.write(C.record_line(tmap[t]))
        C.write_json(pth["title_seal"], OrderedDict([("experiment", "D13"), ("set", set_name), ("files", OrderedDict([(pth["title_map"], file_entry(pth["title_map"]))]))]))
        for arm in C.RAG_ARMS:
            pth = C.retrieval_paths(arm, set_name, "FULL")
            with open(pth["rows"], "wb") as f:
                for row in rows:
                    titles = ["Synthetic %d" % i for i in range(C.RAG_TOP_K)]
                    pres = [t for t in row["required"] if t and tmap[t]["present"]]
                    if pres and rng.random() < 0.3:
                        titles[rng.randrange(C.RAG_TOP_K)] = pres[0]
                    hits = [OrderedDict([("rank", i + 1), ("row", i), ("score", 0.5), ("title", t), ("text", "synthetic text")]) for i, t in enumerate(titles)]
                    f.write(C.record_line(OrderedDict([("qid", row["qid"]), ("arm", arm), ("set", set_name), ("query", C.rag_query(arm, row, ctx)), ("top_k", C.RAG_TOP_K),
                                                       ("hits", hits), ("entries", [C.rag_entry(h) for h in hits])])))
            C.write_json(pth["seal"], OrderedDict([("experiment", "D13"), ("arm", arm), ("set", set_name), ("n_rows", len(rows)), ("top_k", C.RAG_TOP_K),
                                                   ("index_ntotal", C.RAG_NTOTAL), ("retrieve_py_md5", C.RETRIEVE_PY[1]), ("query", "question_ur" if arm == "RAG_ur" else "R1 reading"),
                                                   ("seconds_per_query", 1.0), ("files", OrderedDict([(pth["rows"], file_entry(pth["rows"]))]))]))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true", help="run the self-test and stop; no D13 arm is opened")
    ap.add_argument("--synthetic-root", default=None, help="testing hook: write synthetic arm files under this directory and run the readout on them")
    args = ap.parse_args()
    print("=" * 78)
    print("D13 readout%s" % ("   SELF-TEST ONLY" if args.selftest_only else "   SYNTHETIC TEST — NOT A RESULT" if args.synthetic_root else ""))
    print("=" * 78, flush=True)
    C.verify_contracts()
    for key, (path, want) in AMENDMENTS.items():
        if md5_file(path) != want:
            die("%s MD5 %s != pinned %s" % (path, md5_file(path), want))
        print("[md5] %-18s %s  %s" % (key, want, path))

    st_lines, refs, rows = selftest()
    print("\n".join("[self-test] " + s for s in st_lines), flush=True)
    if args.selftest_only:
        print("[self-test] all checks passed. No D13 arm, retrieval file or seal was opened.")
        return

    if args.synthetic_root:
        root = os.path.abspath(args.synthetic_root)
        if root == C.ROOT or root.startswith(C.ROOT + os.sep):
            die("--synthetic-root must be outside the repository")
        synth_build(root)
        head, out_path = "synthetic", os.path.join(root, "D13_READOUT_SYNTHETIC.md")
    else:
        if C.OUT_ROOT != REAL_OUT_ROOT:
            die("unexpected output root %r" % C.OUT_ROOT)
        head, out_path = require_committed_and_pushed(), OUT_PATH

    # every D13 arm file must be sealed before anything is scored (freeze C5; stage 1 and stage 2 together)
    missing = [a + "/" + s for a, spec in C.ARMS.items() for s in spec["sets"] if not os.path.exists(C.arm_paths(a, s, "FULL")["seal"])]
    if missing:
        die("readout refused: not sealed: %s" % ", ".join(missing))

    arms, prov, seals, starts = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for s in C.SETS:
        arms[s], prov[s], seals[s], starts[s] = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        for name in REFS[s]:
            arms[s][name] = refs[s][name]
            key = REFS[s][name][0]
            prov[s][name] = OrderedDict([("file", C.SEALED[key][0]), ("md5", C.SEALED[key][1]), ("seal", "sealed reference, MD5 pinned (freeze §1)")])
        for a, spec in C.ARMS.items():
            if s in spec["sets"]:
                arms[s][a], prov[s][a], seals[s][a], starts[s][a] = load_d13(a, s, rows[s])
    n_files = sum(1 for a, spec in C.ARMS.items() for s in spec["sets"])
    if n_files != 19:
        die("%d D13 arm files, freeze §2 says 19" % n_files)
    print("[ok] 19 D13 arm files: seals verified (fields, MD5, bytes, lines), rows in item order, no scoring field", flush=True)

    retrieval = OrderedDict()
    for s in C.SETS:
        ctx = C.make_ctx(s, "FULL")
        tmap, tinfo = load_title_map(s)
        retrieval[(s, "title_map")] = tinfo
        for a in C.RAG_ARMS:
            recs, rinfo = C.load_retrieval(a, ctx)
            used = ((starts[s][a].get("guards") or {}).get("retrieval") or {}).get("md5")
            if used is not None and used != rinfo["md5"]:
                die("%s %s: the arm was run on a retrieval file with MD5 %s, the sealed retrieval file has %s" % (s, a, used, rinfo["md5"]))
            rseal = json.load(open(rinfo["seal"], encoding="utf-8"))
            retrieval[(s, a)] = (title_share(rows[s], recs, tmap), OrderedDict([("query", rseal.get("query")), ("seconds_per_query", rseal.get("seconds_per_query"))]))

    text = report(arms, prov, seals, starts, rows, retrieval, st_lines, head, bool(args.synthetic_root))
    print("\n" + text)
    if os.path.exists(out_path):
        if open(out_path, encoding="utf-8").read() != text:
            die("%s exists with different content; not overwritten" % out_path)
        print("[unchanged] %s already holds exactly this text" % out_path)
    else:
        with open(out_path, "x", encoding="utf-8") as f:
            f.write(text)
        print("[written] %s" % out_path)


if __name__ == "__main__":
    main()
