#!/usr/bin/env python3
"""
d14_tables.py — thesis tables for D14, written from the committed readout only:
  docs/handoff/D14_READOUT.md
No sealed output is opened, nothing is scored and nothing is computed: every cell is a string
copied from a line of the readout. Output under docs/thesis_tables/: LaTeX booktabs tables (no
vertical lines) and the same data as CSV files:
  d14_arms            every arm of every task: accuracy, unparsed, truncated
  d14_reading_effect  family G1: the model on the Urdu item and with the English translation
  d14_method          family G2: Qwen3-14B with teacher facts against the Urdu item alone
  d14_original        the original English item against the R1 translation (descriptive)
  d14_cost            seconds and tokens per row per arm

  (no flag)  write the files, then run the check
  --check    check only: every number in every d14 .tex and .csv occurs as a number in the
             readout (signs ignored). Fatal if one does not.
  --readout PATH   testing hook: read another readout file and write under --out-dir
"""
import argparse
import csv
import io
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(ROOT)
READOUT = "docs/handoff/D14_READOUT.md"
OUT_DIR = "docs/thesis_tables"
NUM = re.compile(r"(?<![\w.])[+\-−]?\d+(?:\.\d+)?(?:e-?\d+)?(?!\w)")
TASKS = ("CSQA", "PIQA", "GSM8K")
MODEL = {"S": "Qwen3-14B", "T": "Llama-3.1-70B", "M": "qwen3.8-max"}
FILES = ("d14_arms", "d14_reading_effect", "d14_method", "d14_original", "d14_cost")


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def split2(s):
    """'54.80 (251)' -> ('54.80', '251')"""
    m = re.match(r"^(\S+) \(([^)]+)\)$", s)
    if not m:
        die("cannot split %r" % s)
    return m.group(1), m.group(2)


def sections(text):
    """{heading line: [lines]} for the '## ' and '### ' headings, in order."""
    out, cur = {}, None
    for l in text.split("\n"):
        if l.startswith("## ") or l.startswith("### "):
            cur = l.strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(l)
    return out


def tables(lines):
    """The pipe tables of a section: a list of (header cells, [row cells])."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-| ]+\|$", lines[i + 1].strip()):
            head, rows = cells(lines[i]), []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            out.append((head, rows))
        else:
            i += 1
    return out


def find(secs, prefix):
    hits = [k for k in secs if k.startswith(prefix)]
    if len(hits) != 1:
        die("readout: %d sections start with %r" % (len(hits), prefix))
    return secs[hits[0]]


def tex(s):
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write(name, caption, label, align, header, rows, out_dir, src):
    os.makedirs(out_dir, exist_ok=True)
    L = ["% written by eval/error_analysis_tests/efbpt/d14_tables.py from " + src, "\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{%s}" % caption, "\\label{%s}" % label, "\\begin{tabular}{%s}" % align, "\\toprule",
         " & ".join(tex(h) for h in header) + " \\\\", "\\midrule"]
    L += [" & ".join(tex(c) for c in r) + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(out_dir, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    with open(os.path.join(out_dir, name + ".csv"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    if "|" in align:
        die("%s: a vertical line in the column spec" % name)


def build(readout, out_dir):
    text = open(readout, encoding="utf-8").read()
    secs = sections(text)

    # per arm (section 2, one subsection per task)
    arms, acc = [], {}
    for task in TASKS:
        head, rows = tables(find(secs, "### %s (n = " % task))[0]
        if head[:4] != ["arm", "acc % (correct)", "unparsed (%)", "truncated (%)"]:
            die("readout: per-arm table header of %s is %s" % (task, head))
        for r in rows:
            a, c = split2(r[1])
            u, up = split2(r[2])
            t, tp = split2(r[3])
            arms.append([task, r[0], MODEL[r[0][0]], a, c, up, tp, r[4]])
            acc[(task, r[0])] = a
    write("d14_arms", "D14: every arm of every task (accuracy in percent, all rows; unparsed and truncated in percent of all rows).", "tab:d14-arms",
          "lllrrrrr", ["Task", "Arm", "Model", "Acc %", "Correct", "Unparsed %", "Truncated %", "Reading failed"], arms, out_dir, readout)

    def family(prefix):
        head, rows = tables(find(secs, prefix))[0]
        if head != ["task", "test", "deciding line", "n", "acc first % (correct)", "acc second % (correct)", "delta pp", "b", "c", "raw p", "Holm p", "wording"]:
            die("readout: family table header under %r is %s" % (prefix, head))
        return rows

    def fam_rows(rows):
        out = []
        for r in rows:
            if r[2] == "missing":
                continue
            a, b = r[1].split(" vs ")
            out.append([r[0], MODEL[a[0]], b, acc[(r[0], b)], a, acc[(r[0], a)], r[2], r[3], split2(r[4])[0], split2(r[5])[0], r[6], r[9], r[10], r[11]])
        return out

    fam_head = ["Task", "Model", "Second arm", "Acc % (all rows)", "First arm", "Acc % (all rows)", "Deciding line", "n", "First %", "Second %", "Delta pp", "Raw p", "Holm p", "Wording"]
    write("d14_reading_effect", "D14 family G1: each model with the English translation R1 (first arm) against the same model on the Urdu item (second arm). "
          "Delta, n and the two percentages beside it are on the deciding line; Holm over the 9 tests.", "tab:d14-reading-effect",
          "lllrlrlrrrrrrl", fam_head, fam_rows(family("## 3. Family G1")), out_dir, readout)
    write("d14_method", "D14 family G2: Qwen3-14B with teacher facts (first arm) against Qwen3-14B on the Urdu item alone (second arm). Holm over the 2 tests.",
          "tab:d14-method", "lllrlrlrrrrrrl", fam_head, fam_rows(family("## 4. Family G2")), out_dir, readout)

    # original English item against the R1 translation: the LA lines of section 5
    head, rows = tables(find(secs, "## 5. Descriptive comparisons"))[0]
    if head != ["task", "comparison", "line", "n", "acc first % (correct)", "acc second % (correct)", "delta pp", "b", "c", "raw p"]:
        die("readout: descriptive table header is %s" % head)
    orig = []
    for r in rows:
        if r[2] == "missing" or not r[2].startswith("LA"):
            continue
        a, b = r[1].split(" vs ")
        if a.endswith("_og") and (b.endswith("_r1") or b == "S_ro"):
            orig.append([r[0], MODEL[a[0]], a, split2(r[4])[0], b, split2(r[5])[0], r[6], r[7], r[8], r[9]])
    write("d14_original", "D14, descriptive: the original English item (first arm) against the R1 translation (second arm), all rows.", "tab:d14-original",
          "lllrlrrrrr", ["Task", "Model", "First arm", "Acc %", "Second arm", "Acc %", "Delta pp", "b", "c", "Raw p"], orig, out_dir, readout)

    head, rows = tables(find(secs, "## 9. Seconds and tokens"))[0]
    if head != ["task", "arm", "rows generated", "seconds total", "seconds per row", "prompt tokens per row", "new tokens per row"]:
        die("readout: cost table header is %s" % head)
    write("d14_cost", "D14: seconds and tokens per generated row.", "tab:d14-cost", "llrrrrr",
          ["Task", "Arm", "Rows generated", "Seconds total", "Seconds per row", "Prompt tokens per row", "New tokens per row"], rows, out_dir, readout)


def check(readout, out_dir):
    def norm(t):
        return t.replace("−", "-").lstrip("+-")
    src = {norm(t) for t in NUM.findall(open(readout, encoding="utf-8").read())}
    bad = n = 0
    for name in FILES:
        for ext in (".tex", ".csv"):
            p = os.path.join(out_dir, name + ext)
            if not os.path.exists(p):
                die("missing " + p)
            for i, line in enumerate(open(p, encoding="utf-8").read().split("\n"), 1):
                if line.startswith("%") or line.startswith("\\caption") or line.startswith("\\label") or line.startswith("\\begin{tabular}"):
                    continue
                body = line.replace("Llama-3.1-70B", "").replace("Qwen3-14B", "").replace("qwen3.8-max", "").replace("R1", "").replace("\\_r1", "").replace("_r1", "")
                for t in NUM.findall(body):
                    n += 1
                    if norm(t) not in src:
                        bad += 1
                        print("NOT IN THE READOUT: %s line %d: %s" % (p, i, t))
    print("[check] %d numbers in %d files; not in the readout: %d" % (n, 2 * len(FILES), bad))
    if bad:
        die("check failed")
    print("[check] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--readout", default=READOUT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    if not args.check:
        build(args.readout, args.out_dir)
    check(args.readout, args.out_dir)


if __name__ == "__main__":
    main()
