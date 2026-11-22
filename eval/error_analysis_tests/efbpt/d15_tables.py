#!/usr/bin/env python3
"""
d15_tables.py — thesis tables for D15, written from the committed readout only:
  docs/handoff/D15_READOUT.md
No sealed output is opened, nothing is scored and nothing is computed: every cell is a string
copied from a line of the readout. Output under docs/thesis_tables/: LaTeX booktabs tables (no
vertical lines) and the same data as CSV files:
  d15_arms          every arm: accuracy on set A and on all sampled rows, unparsed, truncated
  d15_tests         families H1, H2, H3 on set A: deciding line, delta, b, c, raw and Holm p, wording
  d15_groups        every H1 and H2 pair inside URDU, REGIONAL and OTHER (descriptive)
  d15_translations  the three translations: valid, failed, flags
  d15_cost          seconds and tokens per generated row

  (no flag)  write the files, then run the check
  --check    check only: every number in every d15 .tex and .csv occurs as a number in the
             readout (signs ignored). Fatal if one does not.
  --readout PATH --out-dir DIR   testing hook: read another readout file and write under DIR
The table parser and writer of d14_tables are reused (imported).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_tables as TB  # noqa: E402  (cells, split2, sections, tables, find, write, NUM; it chdirs to the repo root)

READOUT = "docs/handoff/D15_READOUT.md"
OUT_DIR = "docs/thesis_tables"
FILES = ("d15_arms", "d15_tests", "d15_groups", "d15_translations", "d15_cost")
die = TB.die


def write(name, caption, label, align, header, rows, out_dir, src):
    """d14_tables.write, then its first comment line names this script."""
    TB.write(name, caption, label, align, header, rows, out_dir, src)
    p = os.path.join(out_dir, name + ".tex")
    lines = open(p, encoding="utf-8").read().split("\n")
    if not lines[0].startswith("% written by eval/error_analysis_tests/efbpt/d14_tables.py from "):
        die("%s: unexpected first line" % p)
    lines[0] = "% written by eval/error_analysis_tests/efbpt/d15_tables.py from " + src
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build(readout, out_dir):
    text = open(readout, encoding="utf-8").read()
    secs = TB.sections(text)

    per = {}
    for key in ("### set A (n = ", "### all sampled rows (n = "):
        head, rows = TB.tables(TB.find(secs, key))[0]
        if head != ["arm", "model", "acc % (correct)", "unparsed (%)", "truncated (%)", "reading_failed", "refused", "too_long"]:
            die("readout: per-arm table header under %r is %s" % (key, head))
        per[key] = {r[0]: r for r in rows}
    arms = []
    for arm, r in per["### set A (n = "].items():
        a, c = TB.split2(r[2])
        u, up = TB.split2(r[3])
        t, tp = TB.split2(r[4])
        aa, ac = TB.split2(per["### all sampled rows (n = "][arm][2])
        arms.append([arm, r[1], a, c, up, tp, aa, ac])
    write("d15_arms", "D15 (UrduMMLU, 2,600 native Urdu questions, 100 per subdomain): every arm. Accuracy in percent on set A and on all sampled rows; "
             "unparsed and truncated in percent of set A.", "tab:d15-arms", "llrrrrrr",
             ["Arm", "Model", "Acc % (set A)", "Correct (set A)", "Unparsed %", "Truncated %", "Acc % (all)", "Correct (all)"], arms, out_dir, readout)

    tests = []
    for fam, key in (("H1", "## 4. Family H1"), ("H2", "## 5. Family H2"), ("H3", "## 6. Family H3")):
        head, rows = TB.tables(TB.find(secs, key))[0]
        if head != ["test", "deciding line (set A)", "n", "acc first % (correct)", "acc second % (correct)", "delta pp", "b", "c", "raw p", "Holm p", "wording"]:
            die("readout: family table header under %r is %s" % (key, head))
        for r in rows:
            if r[1] == "missing":
                tests.append([fam, r[0], "missing"] + ["-"] * 9)
                continue
            tests.append([fam, r[0], r[1], r[2], TB.split2(r[3])[0], TB.split2(r[4])[0], r[5], r[6], r[7], r[8], r[9], r[10]])
    write("d15_tests", "D15 families on set A: H1 reading effect (the R1 English translation against the Urdu item), H2 self-translation, H3 teacher facts. "
             "Delta, n and the two percentages are on the deciding line; Holm inside each family.", "tab:d15-tests", "lllrrrrrrrrl",
             ["Family", "Test", "Line", "n", "First %", "Second %", "Delta pp", "b", "c", "Raw p", "Holm p", "Wording"], tests, out_dir, readout)

    head, rows = TB.tables(TB.find(secs, "## 9. Group and domain splits"))[0]
    if head != ["split", "test", "deciding line", "n", "acc first % (correct)", "acc second % (correct)", "delta pp", "b", "c", "raw p"]:
        die("readout: split table header is %s" % head)
    grp = []
    for r in rows:
        if not r[0].startswith("group ") or r[2] == "missing":
            continue
        grp.append([r[0][len("group "):], r[1], r[2], r[3], TB.split2(r[4])[0], TB.split2(r[5])[0], r[6], r[7], r[8], r[9]])
    write("d15_groups", "D15, descriptive: every H1 and H2 pair inside the subject groups (set A; raw p only).", "tab:d15-groups", "lllrrrrrrr",
             ["Group", "Test", "Line", "n", "First %", "Second %", "Delta pp", "b", "c", "Raw p"], grp, out_dir, readout)

    head, rows = TB.tables(TB.find(secs, "## 12. Translations and teacher facts"))[0]
    if head[:6] != ["translation", "model", "rows", "valid", "reading_failed", "too_long"]:
        die("readout: translation table header is %s" % head)
    write("d15_translations", "D15 translations of the Urdu item: R1 (qwen3.8-max), TL (Llama-3.1-70B-AWQ), SL (Qwen3-14B); flags are reported, not a check.",
             "tab:d15-translations", "llrrrrrrrrrr",
             ["Translation", "Model", "Rows", "Valid", "Failed", "Too long", "Requests", "Hit max tokens", "\"correct\"", "\"answer is\"", "Marked option", "Any flag"],
             rows, out_dir, readout)

    head, rows = TB.tables(TB.find(secs, "## 13. Seconds and tokens"))[0]
    if head != ["unit", "rows generated", "seconds total", "seconds per row", "prompt tokens per row", "new tokens per row"]:
        die("readout: cost table header is %s" % head)
    write("d15_cost", "D15: seconds and tokens per generated row.", "tab:d15-cost", "lrrrrr",
             ["Unit", "Rows generated", "Seconds total", "Seconds per row", "Prompt tokens per row", "New tokens per row"], rows, out_dir, readout)


def check(readout, out_dir):
    def norm(t):
        return t.replace("−", "-").lstrip("+-")
    src = {norm(t) for t in TB.NUM.findall(open(readout, encoding="utf-8").read())}
    bad = n = 0
    for name in FILES:
        for ext in (".tex", ".csv"):
            p = os.path.join(out_dir, name + ext)
            if not os.path.exists(p):
                die("missing " + p)
            for i, line in enumerate(open(p, encoding="utf-8").read().split("\n"), 1):
                if line.startswith("%") or line.startswith("\\caption") or line.startswith("\\label") or line.startswith("\\begin{tabular}"):
                    continue
                body = line
                for w in ("Llama-3.1-70B-AWQ", "Llama-3.1-70B", "Qwen3-14B", "qwen3.8-max", "R1", "H1", "H2", "H3", "\\_r1", "_r1"):
                    body = body.replace(w, "")
                for t in TB.NUM.findall(body):
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
