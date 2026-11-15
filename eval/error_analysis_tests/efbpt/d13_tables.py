#!/usr/bin/env python3
"""
d13_tables.py — thesis tables for D13, written from two committed readouts only:
  docs/handoff/D13_READOUT.md               (MD5 pinned)
  docs/handoff/D11_EVAL458_READOUT_LINES.md (MD5 pinned)
No sealed output is opened, nothing is scored and nothing is computed: every cell is a string
copied from a source line. Output under docs/thesis_tables/: LaTeX booktabs tables (no vertical
lines) and the same data as CSV files for figures:
  d13_systems         all eval458 systems: E0, E3, unparsed, truncated
  d13_f1_tests        family F1: deciding line, raw and Holm p, the readout wording
  d13_transfer        family F2: L1 and the deciding line per student
  d13_reading_effect  each model on the Urdu question and on the English reading
  d13_rag_titles      retrieval title share
  d13_cost            seconds and generated tokens per row per arm

  (no flag)  write the files, then run the check
  --check    check only: every number in every .tex and .csv occurs as a number in one of the
             two source files (signs ignored). Fatal if one does not.
"""
import argparse
import csv
import hashlib
import io
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(ROOT)
D13 = ("docs/handoff/D13_READOUT.md", "8ff667881213f108871a09229a881653")
D11 = ("docs/handoff/D11_EVAL458_READOUT_LINES.md", "9f95a95ecd0a92fe6362b438c3f900a4")
OUT_DIR = "docs/thesis_tables"
NUM = re.compile(r"(?<![\w.])[+\-−]?\d+(?:\.\d+)?(?:e-?\d+)?(?!\w)")
F1 = ["T_direct_F_eval", "T_tr_R1", "T_tr_L1", "API_direct_ur_eval", "API_tr_R1", "RAG_ur", "RAG_en"]
F2 = [("Llama8B_TR", "Llama8B_A"), ("Gemma9B_TR", "Gemma9B_A"), ("Qwen8B_TR", "Qwen8B_A")]
SYSTEMS = ["A_r2_eval", "TR", "TR_L1_eval", "TR_wrong_eval", "T_direct_F_eval", "T_tr_R1", "T_tr_L1", "API_direct_ur_eval", "API_tr_R1", "RAG_ur", "RAG_en", "RO_eval",
           "Llama8B_A", "Llama8B_TR", "Gemma9B_A", "Gemma9B_TR", "Qwen8B_A", "Qwen8B_TR"]
READING = [("API model", "API_direct_ur_eval", "API_tr_R1"), ("70B teacher", "T_direct_F_eval", "T_tr_R1"), ("70B teacher, own reading", "T_direct_F_eval", "T_tr_L1"),
           ("14B student, reading as one line", "A_r2_eval", "RO_eval")]


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def split2(s):
    """'54.80 (251)' -> ('54.80', '251')"""
    m = re.match(r"^(\S+) \(([^)]+)\)$", s)
    if not m:
        die("cannot split %r" % s)
    return m.group(1), m.group(2)


def parse():
    for path, want in (D13, D11):
        if md5(path) != want:
            die("%s MD5 %s != pinned %s" % (path, md5(path), want))
    R = {"arm": {}, "cmp": {}, "summ": {}, "titles": [], "time": []}
    sec = sub = cur = None
    for l in open(D13[0], encoding="utf-8").read().split("\n"):
        if l.startswith("## "):
            sec, sub, cur = l[3:].split(".")[0].strip(), None, None
        elif l.startswith("### "):
            sub = l[4:]
        elif sec == "2" and sub and sub.startswith("eval458") and l.startswith("| ") and not l.startswith(("| arm", "|--")):
            c = cells(l)
            sc = "E0" if "(label)" in c[1] else "E3" if c[1] == "E3" else None
            if sc:
                R["arm"][(c[0], sc)] = split2(c[2]) + split2(c[3])
        elif sec in ("4", "5", "6") and re.match(r"\*\*(\S+) vs (\S+) — ", l):
            m = re.match(r"\*\*(\S+) vs (\S+) — ", l)
            cur = (m.group(1), m.group(2))
            R["cmp"][cur] = {}
        elif cur and re.match(r"\| (L\d|A3\.3) ", l):
            c = cells(l)
            R["cmp"][cur][c[0].split()[0]] = c[1:]
        elif sub and sub.startswith(("F1 summary", "F2 summary")) and l.startswith("| ") and " vs " in l.split("|")[1]:
            c = cells(l)
            x, y = c[0].split(" vs ")
            R["summ"][(x, y)] = c[1:8] + [re.search(r"\*\*(.+?)\*\*", c[9]).group(1)]
        elif sec == "9" and l.startswith(("| eval458", "| dev200")):
            R["titles"].append(cells(l))
        elif sec == "10" and l.startswith(("| eval458", "| dev200")):
            R["time"].append(cells(l))
    for l in open(D11[0], encoding="utf-8").read().split("\n"):          # TR_wrong_eval is in the D11 readout only
        if l.startswith("| TR_wrong_eval | E0 |") or l.startswith("| TR_wrong_eval | E3 |"):
            c = cells(l)
            R["arm"][(c[0], c[1])] = split2(c[2]) + split2(c[3])
    return R


def tex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("Δ", r"$\Delta$"), ("−", "-")):
        s = s.replace(a, b)
    return s


def num(s):
    """signed numbers in math mode, so that the minus sign is typeset"""
    s = str(s)
    return "$%s$" % s if re.match(r"^[+\-]\d", s) else tex(s)


def write_table(name, caption, header, rows, align):
    os.makedirs(OUT_DIR, exist_ok=True)
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(header)
    for r in rows:
        wr.writerow(r)
    open(os.path.join(OUT_DIR, name + ".csv"), "w", encoding="utf-8").write(buf.getvalue())
    L = ["% written by eval/error_analysis_tests/efbpt/d13_tables.py from docs/handoff/D13_READOUT.md and docs/handoff/D11_EVAL458_READOUT_LINES.md",
         r"\begin{table}[t]", r"\centering", r"\small", r"\caption{%s}" % caption, r"\label{tab:%s}" % name.replace("_", "-"),
         r"\begin{tabular}{%s}" % align, r"\toprule", " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(num(c) for c in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(os.path.join(OUT_DIR, name + ".tex"), "w", encoding="utf-8").write("\n".join(L))


def build():
    R = parse()
    tr_by = {c[1]: c[3] for c in R["time"] if c[0] == "eval458"}           # 'n (pct)'
    rows = []
    for a in SYSTEMS:
        e0, e3 = R["arm"][(a, "E0")], R["arm"][(a, "E3")]
        t = split2(tr_by[a]) if a in tr_by else ("", "")
        rows.append([a, e0[0], e0[1], e3[0], e3[1], e0[2], e0[3], t[0], t[1]])
    write_table("d13_systems", "All systems on the held-out set. E0 is the frozen extractor, E3 the validated scorer; unparsed answers count as wrong. "
                "TR\\_wrong\\_eval is from the first held-out readout, which records no truncation count.",
                ["System", "E0 acc %", "E0 correct", "E3 acc %", "E3 correct", "Unparsed (E0)", "Unparsed %", "Truncated", "Truncated %"], rows, "lrrrrrrrr")

    rows = [["TR vs " + y] + R["summ"][("TR", y)] for y in F1]
    write_table("d13_f1_tests", "Family F1: TR against the large model alone and against traditional RAG, on the deciding line of each test; exact two-sided McNemar, Holm step-down.",
                ["Test", "Line", "n", "Δ pp", "b", "c", "Raw p", "Holm p", "Reading"], rows, "llrrrrrrl")

    rows = []
    for x, y in F2:
        l1, s = R["cmp"][(x, y)]["L1"], R["summ"][(x, y)]
        rows.append([x + " vs " + y, l1[1], l1[2], l1[3], l1[4], l1[5], l1[6], s[0], s[1], s[2], s[3], s[4], s[5], s[6]])
    write_table("d13_transfer", "Family F2: the same teacher facts with three other students. L1 is the frozen extractor on all rows; the deciding line follows the unparsed and truncation rules.",
                ["Test", "L1 acc with facts", "L1 acc alone", "L1 Δ pp", "L1 b", "L1 c", "L1 p", "Deciding line", "n", "Δ pp", "b", "c", "Raw p", "Holm p"], rows, "lrrrrrrlrrrrrr")

    rows = [[m, a, R["arm"][(a, "E0")][0], R["arm"][(a, "E3")][0], b, R["arm"][(b, "E0")][0], R["arm"][(b, "E3")][0]] for m, a, b in READING]
    write_table("d13_reading_effect", "Each model on the Urdu question and on the English reading of the same question (accuracy in percent, all rows).",
                ["Model", "Arm, Urdu question", "E0 acc %", "E3 acc %", "Arm, English reading", "E0 acc %", "E3 acc %"], rows, "llrrlrr")

    rows = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in R["titles"]]
    write_table("d13_rag_titles", "Share of gold evidence titles among the retrieved chunks.",
                ["Set", "Arm", "Query", "Found / all gold titles", "Found / gold titles in the corpus", "Rows with at least one found / rows"], rows, "lllrrr")

    rows = []
    for c in R["time"]:
        t = split2(c[3])
        rows.append([c[0], c[1], c[2], t[0], t[1], c[4], c[5], c[6], c[7]])
    write_table("d13_cost", "Truncated rows, seconds and generated tokens per row per arm. A dash means the sealed file records no time.",
                ["Set", "Arm", "Rows", "Truncated", "Truncated %", "Seconds per row", "Tokens per row", "Seconds total", "Tokens total"], rows, "llrrrrrrr")


def check():
    def toks(text):
        return {t.replace("−", "-").lstrip("+-") for t in NUM.findall(text)}
    src = set()
    for path, want in (D13, D11):
        if md5(path) != want:
            die("%s MD5 %s != pinned %s" % (path, md5(path), want))
        src |= toks(open(path, encoding="utf-8").read())
    names = sorted(f for f in os.listdir(OUT_DIR) if f.endswith((".tex", ".csv")))
    if len(names) != 12:
        die("%d .tex / .csv files under %s, expected 12" % (len(names), OUT_DIR))
    n = 0
    for f in names:
        text = open(os.path.join(OUT_DIR, f), encoding="utf-8").read()
        if f.endswith(".tex"):
            if "|" in text.split(r"\begin{tabular}")[1].split("}")[0]:
                die("%s: vertical line in the column spec" % f)
            text = re.sub(r"\\begin\{table\}\[t\]", " ", text)
        for t in sorted(toks(text)):
            n += 1
            if t not in src:
                die("CHECK FAILED: %s holds the number %s, which is in neither source file" % (f, t))
    print("[check] PASS: %d files, %d distinct number tokens (per file), every one occurs in %s or %s" % (len(names), n, D13[0], D11[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        build()
    check()


if __name__ == "__main__":
    main()
