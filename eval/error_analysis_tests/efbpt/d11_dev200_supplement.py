#!/usr/bin/env python3
"""
d11_dev200_supplement.py — D11 DEV200 supplement (descriptive), CPU only.

Reads SEALED DEV200 files only (MD5 pinned below, asserted); no model, no
generation, no network. Nothing under any eval458 directory is opened: every
input path is checked. Nothing frozen is edited: the frozen extractor, the
paired exact McNemar, the D11-R norm() / row-M1 and the DEV200 loader are
imported.

  S1 scorer sensitivity: frozen extractor vs a first-mention variant
  S2 TR vs A_r2 on rows parsed in both arms / parsed and untruncated in both
  S3 TR and TR_L1 paired against the D10 arm TF_v1
  S4 reader agreement (R1 vs R2 readings after norm()) as a deployable proxy
  S5 the first 15 TR-vs-A_r2 WIN rows in the loss-row format of readout §8

Output: docs/handoff/D11_DEV200_SUPPLEMENT.md (refuses to overwrite).
Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d11_dev200_supplement.py [--out PATH]
"""
import argparse
import os
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from d11_common import load_set_rows, read_jsonl, md5_file, die  # noqa: E402  (chdir to the repo root)
from efbpt_eval_dev200 import extract_answer, HAAN, NAHIN, MARKER, RE_JSON_ANSWER, RE_EN_WORD  # noqa: E402  frozen
from d5_balanced_extract import mcnemar  # noqa: E402  paired exact two-sided McNemar
from d11r_score import norm, m1_row  # noqa: E402  D11-R norm() and row-M1

OUT_DEFAULT = "docs/handoff/D11_DEV200_SUPPLEMENT.md"
READOUT = "docs/handoff/D11_FULL_READOUT.md"

# sealed DEV200 inputs (MD5s as in docs/handoff/D11_FULL_READOUT.md §1)
ARMS = OrderedDict([
    ("A_r2", ("outputs/efbpt/d10/student/d10_armA_r2.jsonl", "7ccec942c8f0eee08156d258baaf806d")),
    ("TF_v1", ("outputs/efbpt/d10/student/d10_armTF.jsonl", "c54b7553baa5ffa77204bed1dab37250")),
    ("TR", ("outputs/efbpt/d11/student/dev200/d11_armTR.jsonl", "bab987cc492012f4e32dbcd348be9668")),
    ("TR2", ("outputs/efbpt/d11/student/dev200/d11_armTR2.jsonl", "17f1c43d6406414d66aa28d769640ab6")),
    ("TR_L1", ("outputs/efbpt/d11/student/dev200/d11_armTR_L1.jsonl", "b8db599f8c4b427ddcb2d772fd8f09aa")),
    ("TR_wrong", ("outputs/efbpt/d11/student/dev200/d11_armTR_wrong.jsonl", "9519b9adf215ac90eb6de743ec1dca57")),
    ("GF", ("outputs/efbpt/d11o/student/d11o_armGF.jsonl", "448fec4127bb8060c10422d9aa42699f")),
    ("TO1", ("outputs/efbpt/d11o/student/d11o_armTO1.jsonl", "6548b408a68a8e94bb65b9ca6d023fdd")),
])
READINGS = OrderedDict([
    ("R1", ("outputs/efbpt/d11r/api_qwen3.8-max/readings_dev200.jsonl", "1f5b2db3c1403f63b3525a1562edd176")),
    ("R2", ("outputs/efbpt/d11r/api_deepseek-v4-pro/readings_dev200.jsonl", "cd77944e45a98292777382b9d9f04aef")),
])
T_FACTS_R1 = ("outputs/efbpt/d11/teacher/dev200/R1/t_facts.jsonl", "160406e12205635e69f5c95140f9936f")
N_WIN_ROWS = 15
GEN_CHARS = 110


def sealed(path, want):
    if "eval458" in path:
        die("refusing to open an eval458 path: %s" % path)
    got = md5_file(path)
    if got != want:
        die("%s MD5 %s != pinned %s" % (path, got, want))
    return read_jsonl(path)


def extract(text, first):
    """The frozen AMENDMENT 5 extractor, rule for rule. first=False is the frozen
    rule (asserted equal to efbpt_eval_dev200.extract_answer on every generation
    read here). first=True differs in ONE place: in Rule 2, where the frozen rule
    takes the LAST ہاں/نہیں mention of the segment, the FIRST is taken. Rule 1
    (JSON, last), the marker cut (after the last marker), the substring matching
    and Rule 3 (English, last) are unchanged."""
    m = None
    for m in RE_JSON_ANSWER.finditer(text):
        pass
    if m is not None:
        return m.group(1).lower()
    seg = text
    if MARKER in seg:
        seg = seg[seg.rfind(MARKER) + len(MARKER):]
    if first:
        i_haan, i_nahin = seg.find(HAAN), seg.find(NAHIN)
        if i_haan != -1 or i_nahin != -1:
            if i_nahin == -1:
                return "yes"
            if i_haan == -1:
                return "no"
            return "yes" if i_haan < i_nahin else "no"
    else:
        i_haan, i_nahin = seg.rfind(HAAN), seg.rfind(NAHIN)
        if i_haan != -1 or i_nahin != -1:
            return "yes" if i_haan > i_nahin else "no"
    m = None
    for m in RE_EN_WORD.finditer(text):
        pass
    if m is not None:
        return m.group(1).lower()
    return None


def acc(recs, key="pred"):
    return 100.0 * sum(1 for r in recs if r[key] == r["gold"]) / len(recs)


def paired(x, y, qids=None):
    """b = x right & y wrong, c = y right & x wrong, exact two-sided p (d5 mcnemar), on qids (default all)."""
    if qids is not None:
        x = [r for r in x if r["qid"] in qids]
        y = [r for r in y if r["qid"] in qids]
    b, c, p = mcnemar(x, y, "x", "y")
    n = len(x)
    return OrderedDict([("n", n), ("acc_x", acc(x) if n else None), ("acc_y", acc(y) if n else None),
                        ("delta", 100.0 * (b - c) / n if n else None), ("b", b), ("c", c), ("p", p)])


def fp(p):
    return "%.4g" % p


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def gen_cell(rec):
    return "**%s** — %s" % (rec["pred"] if rec["pred"] is not None else "None (unparsed)", cell(" ".join(rec["generation"].split())[:GEN_CHARS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    if "eval458" in args.out:
        die("refusing an eval458 output path")
    if os.path.exists(args.out):
        die("%s exists; this script does not overwrite" % args.out)

    rows = load_set_rows("dev200")
    qids = [r["qid"] for r in rows]
    by_q = OrderedDict((r["qid"], r) for r in rows)
    arms = OrderedDict()
    for a, (path, want) in ARMS.items():
        recs = sealed(path, want)
        if [r["qid"] for r in recs] != qids:
            die("%s: qids are not DEV200 in order" % path)
        for r in recs:
            if r["gold"] != by_q[r["qid"]]["gold"]:
                die("%s qid %s: gold differs from the item set" % (path, r["qid"]))
            if extract_answer(r["generation"]) != r["pred"] or extract(r["generation"], False) != r["pred"]:
                die("%s qid %s: the frozen extractor does not reproduce the sealed pred" % (path, r["qid"]))
            if r["correct"] != (r["pred"] == r["gold"]) or r["unparsed"] != (r["pred"] is None):
                die("%s qid %s: correct / unparsed fields inconsistent" % (path, r["qid"]))
        arms[a] = recs
    readings = OrderedDict()
    for k, (path, want) in READINGS.items():
        recs = sealed(path, want)
        if [r["qid"] for r in recs] != qids:
            die("%s: qids are not DEV200 in order" % path)
        readings[k] = OrderedDict((r["qid"], r["reading"]) for r in recs)
    t_facts = sealed(*T_FACTS_R1)
    if [e["qid"] for e in t_facts] != qids:
        die("%s: qids are not DEV200 in order" % T_FACTS_R1[0])
    facts = OrderedDict((e["qid"], e["facts"]) for e in t_facts)
    for e in t_facts:
        if e["teacher_input_reading"] != readings["R1"][e["qid"]]:
            die("teacher R1 input differs from the R1 reading for qid %s" % e["qid"])

    A, TR = arms["A_r2"], arms["TR"]
    a_by = {r["qid"]: r for r in A}
    t_by = {r["qid"]: r for r in TR}
    wins = [q for q in qids if t_by[q]["correct"] and not a_by[q]["correct"]]
    losses = [q for q in qids if a_by[q]["correct"] and not t_by[q]["correct"]]
    base = paired(TR, A)
    if (len(wins), len(losses), base["b"], base["c"]) != (39, 22, 39, 22):
        die("TR vs A_r2 is not b 39 / c 22 on the sealed files")

    L = []
    w = L.append
    w("# D11 DEV200 supplement (descriptive)")
    w("")
    w("Written by `eval/error_analysis_tests/efbpt/d11_dev200_supplement.py` (CPU, no model, no generation re-run) from sealed DEV200 files only; "
      "nothing under any eval458 directory was opened. Descriptive: no gate, no decision and no eval458 rule depends on anything here. "
      "Conventions as in `docs/handoff/D11_FULL_READOUT.md`: b = the first-named arm right and the second wrong, c = the reverse, "
      "p = exact two-sided McNemar (`d5_balanced_extract.mcnemar`), unparsed = wrong, n = 200 (87 gold yes / 113 gold no) unless a subset is named.")
    w("")
    w("Inputs (MD5 asserted): " + "; ".join("%s `%s`" % (a, m[:8]) for a, (_, m) in ARMS.items())
      + "; readings R1 `%s`, R2 `%s`; teacher facts R1 `%s`. " % (READINGS["R1"][1][:8], READINGS["R2"][1][:8], T_FACTS_R1[1][:8])
      + "The frozen extractor reproduces the sealed `pred` on all %d generations read." % (len(arms) * len(qids)))
    w("")

    # ------------------------------------------------------------------ S1
    w("## S1. Scorer sensitivity: frozen extractor vs first-mention variant")
    w("")
    w("The variant is the frozen extractor with one change: in Rule 2, where the frozen rule takes the **last** ہاں/نہیں mention of the answer segment, "
      "the **first** is taken. Rule 1 (JSON), the cut after the last \"حتمی جواب\" marker, Rule 3 (English) and the plain substring matching are unchanged "
      "(so a word such as یہاں or وہاں, which contains ہاں, counts as a mention under both rules). The set of unparsed rows is therefore the same under both.")
    w("")
    w("| arm | frozen acc % (correct) | first-mention acc % (correct) | variant − frozen pp | rows where the two labels differ | of those: frozen right / variant right | unparsed |")
    w("|---|---|---|---|---|---|---|")
    var = OrderedDict()
    for a, recs in arms.items():
        vr = [dict(qid=r["qid"], gold=r["gold"], pred=extract(r["generation"], True)) for r in recs]
        var[a] = vr
        diff = [(r, v) for r, v in zip(recs, vr) if r["pred"] != v["pred"]]
        if any((r["pred"] is None) != (v["pred"] is None) for r, v in zip(recs, vr)):
            die("%s: the variant changed the unparsed set" % a)
        cf, cv = sum(1 for r in recs if r["correct"]), sum(1 for v in vr if v["pred"] == v["gold"])
        w("| %s | %.2f (%d) | %.2f (%d) | %+.1f | %d | %d / %d | %d |"
          % (a, acc(recs), cf, acc(vr), cv, acc(vr) - acc(recs), len(diff),
             sum(1 for r, _ in diff if r["correct"]), sum(1 for _, v in diff if v["pred"] == v["gold"]), sum(1 for r in recs if r["pred"] is None)))
    w("")
    pv = paired(var["TR"], var["A_r2"])
    w("| TR vs A_r2 | acc TR | acc A_r2 | Δ pp | b | c | exact two-sided p |")
    w("|---|---|---|---|---|---|---|")
    w("| frozen extractor (the sealed result) | %.2f | %.2f | %+.1f | %d | %d | %s |" % (base["acc_x"], base["acc_y"], base["delta"], base["b"], base["c"], fp(base["p"])))
    w("| first-mention variant | %.2f | %.2f | %+.1f | %d | %d | %s |" % (pv["acc_x"], pv["acc_y"], pv["delta"], pv["b"], pv["c"], fp(pv["p"])))
    w("")
    both_same = sum(1 for q in qids if t_by[q]["pred"] == extract(t_by[q]["generation"], True) and a_by[q]["pred"] == extract(a_by[q]["generation"], True))
    w("Rows whose label is the same under both rules in both arms: %d of 200. The frozen extractor is the pre-registered scorer; the variant is a sensitivity reading only." % both_same)
    w("")

    # ------------------------------------------------------------------ S2
    w("## S2. TR vs A_r2 on parsed and untruncated rows")
    w("")
    parsed = {q for q in qids if t_by[q]["pred"] is not None and a_by[q]["pred"] is not None}
    clean = {q for q in parsed if not t_by[q]["truncated"] and not a_by[q]["truncated"]}
    w("| rows | n | acc TR | acc A_r2 | Δ pp (on these rows) | b | c | exact two-sided p |")
    w("|---|---|---|---|---|---|---|---|")
    for label, sel in (("all rows (the sealed result)", None), ("parsed in both arms", parsed), ("parsed and untruncated in both arms", clean)):
        e = paired(TR, A, sel)
        w("| %s | %d | %.2f | %.2f | %+.2f | %d | %d | %s |" % (label, e["n"], e["acc_x"], e["acc_y"], e["delta"], e["b"], e["c"], fp(e["p"])))
    w("")
    w("Rows dropped: A_r2 unparsed %d, TR unparsed %d (either %d); A_r2 truncated %d, TR truncated %d; unparsed or truncated in either arm %d."
      % (sum(1 for r in A if r["pred"] is None), sum(1 for r in TR if r["pred"] is None), 200 - len(parsed),
         sum(1 for r in A if r["truncated"]), sum(1 for r in TR if r["truncated"]), 200 - len(clean)))
    w("")
    w("| discordant rows | n | other arm unparsed | other arm truncated | unparsed or truncated | both |")
    w("|---|---|---|---|---|---|")
    for label, qs, other in (("39 wins (TR right, A_r2 wrong): A_r2 was", wins, a_by), ("22 losses (A_r2 right, TR wrong): TR was", losses, t_by)):
        u = sum(1 for q in qs if other[q]["pred"] is None)
        t = sum(1 for q in qs if other[q]["truncated"])
        e = sum(1 for q in qs if other[q]["pred"] is None or other[q]["truncated"])
        w("| %s | %d | %d | %d | %d | %d |" % (label, len(qs), u, t, e, u + t - e))
    w("")

    # ------------------------------------------------------------------ S3
    w("## S3. Paired against the D10 arm TF_v1")
    w("")
    w("TF_v1 = the D10 teacher-facts arm (same teacher, same student, same layout B; teacher input = the Urdu question, no reading stage).")
    w("")
    w("| comparison | acc first | acc TF_v1 | Δ pp | b | c | exact two-sided p |")
    w("|---|---|---|---|---|---|---|")
    for a in ("TR", "TR_L1"):
        e = paired(arms[a], arms["TF_v1"])
        w("| %s vs TF_v1 | %.2f | %.2f | %+.1f | %d | %d | %s |" % (a, e["acc_x"], e["acc_y"], e["delta"], e["b"], e["c"], fp(e["p"])))
    w("")

    # ------------------------------------------------------------------ S4
    w("## S4. Reader agreement (R1 vs R2) as a deployable proxy")
    w("")
    w("Agreement = the R1 (api_qwen3.8-max) and R2 (api_deepseek-v4-pro) readings are equal after `norm()` (`d11r_score.norm`: \"_\" → space, strip, lower-case, "
      "collapse whitespace). It needs no gold. Row-M1 = share of the gold question's proper nouns recovered by the R1 reading (`d11r_score.m1_row`; "
      "45 rows have no gold proper noun). TR uses the R1 reading in both groups.")
    w("")
    m1 = OrderedDict((q, m1_row(by_q[q]["question_en"], readings["R1"][q])[0]) for q in qids)
    agree = {q for q in qids if norm(readings["R1"][q]) == norm(readings["R2"][q])}
    w("| R1 vs R2 reading | n | R1 row-M1 = 1 | row-M1 < 1 | no proper noun | share row-M1 = 1 (of n) | share row-M1 = 1 (of rows with a proper noun) | TR acc % | A_r2 acc % | Δ pp | wins | losses | exact two-sided p |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, sel in (("agree after norm()", agree), ("disagree", set(qids) - agree), ("all", set(qids))):
        full = sum(1 for q in sel if m1[q] is not None and m1[q] == 1.0)
        part = sum(1 for q in sel if m1[q] is not None and m1[q] < 1.0)
        none = sum(1 for q in sel if m1[q] is None)
        e = paired(TR, A, sel)
        w("| %s | %d | %d | %d | %d | %.1f%% | %.1f%% | %.2f | %.2f | %+.2f | %d | %d | %s |"
          % (label, len(sel), full, part, none, 100.0 * full / len(sel), 100.0 * full / (full + part) if full + part else float("nan"),
             e["acc_x"], e["acc_y"], e["delta"], e["b"], e["c"], fp(e["p"])))
    w("")
    w("Post-hoc split of one run; the two groups differ in question difficulty as well as in reading quality (see the A_r2 column).")
    w("")

    # ------------------------------------------------------------------ S5
    w("## S5. The first %d of %d TR-vs-A_r2 WIN rows (TR right, A_r2 wrong)" % (N_WIN_ROWS, len(wins)))
    w("")
    w("Format of readout §8.1: qid, gold, Urdu question, gold question_en, the R1 reading (row-M1 in parentheses; \"-\" = no gold proper noun), the first two teacher "
      "facts, A_r2 answer, TR answer; answers show the frozen extractor's `pred` in bold and the first ~%d characters of the generation (whitespace collapsed). "
      "Rows in DEV200 file order." % GEN_CHARS)
    w("")
    head = "| # | qid | gold | Urdu question | gold question_en | R1 reading (row-M1) | fact 1 | fact 2 | A_r2 | TR |"
    w(head)
    w("|---|---|---|---|---|---|---|---|---|---|")
    win_lines = []
    for i, q in enumerate(wins[:N_WIN_ROWS], 1):
        r = by_q[q]
        f = list(facts[q]) + ["", ""]
        line = "| %d | `%s` | %s | %s | %s | %s (%s) | %s | %s | %s | %s |" % (
            i, q, r["gold"], cell(r["question_ur"]), cell(r["question_en"]), cell(readings["R1"][q]),
            "-" if m1[q] is None else "%.2f" % m1[q], cell(f[0]), cell(f[1]), gen_cell(a_by[q]), gen_cell(t_by[q]))
        win_lines.append(line)
        w(line)
    w("")
    ro = open(READOUT, encoding="utf-8").read().split("\n")
    i0 = next(i for i, l in enumerate(ro) if l.startswith("### 8.2 "))
    ro_rows = [l for l in ro[i0:i0 + 40] if l.startswith("| ") and l.split("|")[1].strip().isdigit()][:N_WIN_ROWS]
    same = sum(1 for x, y in zip(win_lines, ro_rows) if x == y)
    w("Cross-check: %d of these %d rows are character-identical to the rows of readout §8.2 (`%s`)." % (same, len(win_lines), READOUT))
    w("")

    text = "\n".join(L)
    if "eval458" in "".join(p for p, _ in list(ARMS.values()) + list(READINGS.values()) + [T_FACTS_R1]):
        die("an input path names eval458")
    with open(args.out, "x", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("\n[written] %s (%d lines)" % (args.out, len(L)))


if __name__ == "__main__":
    main()
