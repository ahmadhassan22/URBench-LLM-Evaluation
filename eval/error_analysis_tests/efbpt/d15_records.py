#!/usr/bin/env python3
"""
d15_records.py — the D15 records in README.md and experiments.md, written from the committed
readout (docs/handoff/D15_READOUT.md) and analysis pack (docs/handoff/D15_ANALYSIS.md) only.

Every number in the text is a string copied from a cell or a line of the readout; nothing is
scored or computed here. No historical text is deleted: the script only inserts
  README.md      the section "Reading Effect and Teacher Facts on Native Urdu Questions (D15)"
                 before "## Repository Structure"; Key Findings 32-35 after the last finding;
                 a D15 paragraph in "Current State" (after the D14 paragraph) and the date of its
                 heading; D15 bullets in "### Limits" before its "Sources:" line, and the D15
                 readout added to that line
  experiments.md the record "EFBPT D15: ..." at the end

  --write      insert the text (refuses if a D15 block is already there)
  --check      number check: every number in the D15 blocks of README.md and experiments.md and in
               docs/thesis_tables/d15_* occurs as a number in the readout, the analysis pack or the
               freeze (signs ignored). Fatal if one does not.
  --print      print the blocks only
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_tables as TB  # noqa: E402  (cells, split2, sections, tables, find, NUM; chdir to the repo root)

READOUT = "docs/handoff/D15_READOUT.md"
ANALYSIS = "docs/handoff/D15_ANALYSIS.md"
FREEZE = "docs/EFBPT_D15_NATIVE_URDU_FREEZE.md"
README = "README.md"
EXPERIMENTS = "experiments.md"
die = TB.die

DESC = {
    "S_ur": "Qwen3-14B, Urdu prompt (the CSQA fair prompt for four options)",
    "S_ro": "Qwen3-14B, Urdu prompt + the R1 English translation",
    "S_self": "Qwen3-14B, Urdu prompt + its own English translation (SL)",
    "S_tr": "Qwen3-14B, Urdu prompt + teacher facts made from the R1 translation",
    "T_ur": "Llama-3.1-70B, Urdu prompt",
    "T_ue": "Llama-3.1-70B, English instruction + the Urdu item",
    "T_r1": "Llama-3.1-70B, English prompt with the R1 translation",
    "T_self": "Llama-3.1-70B, English prompt with its own translation (TL)",
    "M_ur": "qwen3.8-max, Urdu prompt",
    "M_ue": "qwen3.8-max, English instruction + the Urdu item",
    "M_r1": "qwen3.8-max, English prompt with the R1 translation (its own translation)",
}
MODEL = {"S": "Qwen3-14B", "T": "Llama-3.1-70B", "M": "qwen3.8-max"}
WHAT = {"H1": "the English translation", "H2": "its own English translation", "H3": "the teacher facts"}


def parse():
    text = open(READOUT, encoding="utf-8").read()
    secs = TB.sections(text)
    R = {"text": text, "secs": secs}
    m = re.search(r"n\(sampled\) = (\d+), n\(set A\) = (\d+), dropped = (\d+)", text)
    if not m:
        die("readout: no set-A line")
    R["n"], R["nA"], R["drop"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"one run, (\d{4}-\d{2}-\d{2})", text)
    R["date"] = m.group(1)
    R["drop_table"] = TB.tables(TB.find(secs, "## 2. Sample, set A"))[0]
    R["armA"] = {r[0]: r for r in TB.tables(TB.find(secs, "### set A (n = "))[0][1]}
    R["armAll"] = {r[0]: r for r in TB.tables(TB.find(secs, "### all sampled rows (n = "))[0][1]}
    for fam, key in (("H1", "## 4. Family H1"), ("H2", "## 5. Family H2"), ("H3", "## 6. Family H3")):
        tabs = TB.tables(TB.find(secs, key))
        R[fam] = tabs[0][1]
        R[fam + "_beside"] = tabs[1][1]
        R[fam + "_lines"] = tabs[2][1]
    R["n9"] = [l for l in TB.find(secs, "## 7. Reading of the outcome") if l.startswith("- ")]
    R["desc"] = TB.tables(TB.find(secs, "## 8. Descriptive comparisons"))[0][1]
    R["splits"] = TB.tables(TB.find(secs, "## 9. Group and domain splits"))[0][1]
    R["pred"] = TB.tables(TB.find(secs, "## 11. Predictions"))[0][1]
    m = re.search(r"Predictions hit: (\d+) of (\d+)\.", text)
    R["pred_hits"] = (m.group(1), m.group(2))
    tabs = TB.tables(TB.find(secs, "## 12. Translations and teacher facts"))
    R["trans"], R["facts"] = tabs[0], tabs[1]
    R["ledger"] = next(l for l in TB.find(secs, "## 13. Seconds and tokens") if l.startswith("API ledger"))
    R["used"] = next((l for l in TB.find(secs, "## 13. Seconds and tokens") if l.startswith("Amount used")), "")
    return R


def fam_table(R, fam):
    beside = {}
    for r in R[fam + "_beside"]:
        beside.setdefault(r[0], []).append(r)
    out = ["| Test | Deciding line (set A) | n | First % | Second % | Δ pp | b | c | Raw p | Holm p | All sampled rows: line, n, Δ pp, b / c, raw p | Readout wording |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in R[fam]:
        if r[1] == "missing":
            out.append("| %s | missing | | | | | | | | | | missing |" % r[0])
            continue
        al = next((b for b in beside.get(r[0], []) if b[1] == "all sampled rows"), None)
        alls = "%s, %s, %s, %s / %s, %s" % (al[2], al[3], al[6], al[7], al[8], al[9]) if al else "-"
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                   % (r[0], r[1], r[2], TB.split2(r[3])[0], TB.split2(r[4])[0], r[5], r[6], r[7], r[8], r[9], alls, r[10]))
    extra = []
    for r in R[fam + "_beside"]:
        if r[1].startswith("set A rows with a valid"):
            extra.append("- %s, %s: line %s, n %s: %s → %s (%s pp, b %s / c %s, raw p %s); printed beside the test, no wording."
                         % (r[0], r[1], r[2], r[3], TB.split2(r[5])[0], TB.split2(r[4])[0], r[6], r[7], r[8], r[9]))
    return out, extra


def wording_bullets(R, fam):
    out = []
    beside = {}
    for r in R[fam + "_beside"]:
        beside.setdefault(r[0], []).append(r)
    for r in R[fam]:
        if r[1] == "missing":
            out.append("- %s: missing (an arm was not run)." % r[0])
            continue
        a, b = r[0].split(" vs ")
        la = next(x for x in R[fam + "_lines"] if x[0] == "set A" and x[1] == r[0] and x[2].startswith("LA"))
        s = ("- %s (%s): %s → %s on set A (%s pp, b %s / c %s, raw p %s)"
             % (MODEL[a[0]], r[0], TB.split2(la[5])[0], TB.split2(la[4])[0], la[6], la[7], la[8], la[9]))
        if r[1] != "LA":
            s += "; deciding line %s, n %s: %s pp, b %s / c %s, raw p %s" % (r[1], r[2], r[5], r[6], r[7], r[8])
        s += "; Holm p %s: %s: **%s**." % (r[9], WHAT[fam], r[10])
        out.append(s)
    return out


def arm_table(R):
    out = ["| Arm | What it is | Acc % set A (correct) | Unparsed (%) | Truncated (%) | Acc % all sampled rows (correct) |", "|---|---|---:|---:|---:|---:|"]
    for arm, r in R["armA"].items():
        out.append("| %s | %s | %s | %s | %s | %s |" % (arm, DESC[arm], r[2], r[3], r[4], R["armAll"][arm][2]))
    return out


def desc_table(R):
    out = ["| Comparison (descriptive, set A) | Line | n | First % | Second % | Δ pp | b / c | Raw p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in R["desc"]:
        if r[2] == "missing":
            out.append("| %s | missing | | | | | | |" % r[1])
            continue
        if r[2].startswith("LA") or "(deciding)" in r[2]:
            out.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s |" % (r[1], r[2], r[3], TB.split2(r[4])[0], TB.split2(r[5])[0], r[6], r[7], r[8], r[9]))
    return out


def group_table(R):
    out = ["| Group | Test | Line | n | First % | Second % | Δ pp | b / c | Raw p |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in R["splits"]:
        if r[0].startswith("group ") and r[2] != "missing":
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s / %s | %s |" % (r[0][6:], r[1], r[2], r[3], TB.split2(r[4])[0], TB.split2(r[5])[0], r[6], r[7], r[8], r[9]))
    return out


def pred_table(R):
    out = ["| Prediction (freeze N8, set A) | First % | Second % | Observed Δ pp | |", "|---|---:|---:|---:|---|"]
    for r in R["pred"]:
        out.append("| " + " | ".join(r) + " |")
    return out


def trans_lines(R):
    head, rows = R["trans"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]
    fh, fr = R["facts"]
    f = fr[0]
    out += ["", "Teacher facts (from the R1 translation): %s rows, replies parsed %s, facts kept %s, lines removed by the D10 verdict filter %s, by the D14 filter %s; rows without facts %s."
            % (f[1], f[2], f[3], f[4], f[5], f[7])]
    return out


def drop_lines(R):
    head, rows = R["drop_table"]
    return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(r) + " |" for r in rows if not r[0].startswith("domain ")]


def readme_section(R):
    D = ["## Reading Effect and Teacher Facts on Native Urdu Questions (D15)", ""]
    D.append("Under a new freeze (`docs/EFBPT_D15_NATIVE_URDU_FREEZE.md`) the D14 questions were asked on **native** Urdu questions: UrduMMLU (MBZUAI/UrduMMLU, CC BY 4.0), "
             "questions written in Urdu and never translated from English. The sample is 100 rows from each of the 26 subdomains (%s rows); set A = the sampled rows with a valid R1 "
             "translation and an answer from qwen3.8-max in all three of its arms (%s rows; %s dropped). Every test is read on set A. Families: H1 the reading effect (3 tests), "
             "H2 self-translation with no outside model (2 tests), H3 the teacher-facts method (1 test), Holm inside each family. The readout script was committed and pushed before it "
             "read any arm and was run once; the wording was fixed before the run: \"helps\" / \"hurts\" only with Holm p < 0.05, otherwise \"no clear effect\"."
             % (R["n"], R["nA"], R["drop"]))
    D.append("")
    D.append("**The sample is equal per subject, so every accuracy below is the mean over the 26 subjects, not the dataset-weighted UrduMMLU accuracy.**")
    D.append("")
    D.append("### Arms")
    D.append("")
    D.append("S = Qwen3-14B, T = Llama-3.1-70B-AWQ, M = qwen3.8-max. Unparsed counts as wrong. Truncated = the row reached the token limit. R1 = the English translation by qwen3.8-max; "
             "SL / TL = the model's own translation. A row without a valid translation counts as wrong in the arm that needs it.")
    D.append("")
    D += arm_table(R)
    D.append("")
    for fam, title, what in (("H1", "### Family H1: the reading effect on native Urdu", "First arm = the model with the R1 English translation; second arm = the same model on the Urdu item (S: Urdu prompt; T and M: English instruction with the Urdu item). 3 tests, Holm at 0.05."),
                             ("H2", "### Family H2: self-translation, no outside model", "First arm = the model with its own English translation; second arm = the same model on the Urdu item. 2 tests, Holm at 0.05."),
                             ("H3", "### Family H3: the teacher-facts method", "First arm = S_tr (the student with teacher facts made from the R1 translation); second arm = S_ur. 1 test.")):
        D.append(title)
        D.append("")
        D.append(what)
        D.append("")
        t, extra = fam_table(R, fam)
        D += t
        D.append("")
        D.append("In the readout wording:")
        D.append("")
        D += wording_bullets(R, fam)
        D += extra
        D.append("")
    D.append("### Reading of the outcome (freeze N9, fixed before the run)")
    D.append("")
    D += R["n9"]
    D.append("")
    D.append("### Descriptive comparisons (set A)")
    D.append("")
    D.append("Instruction language (`_ue` vs `_ur`), the D14 form (`_r1` vs `_ur`), and the translation sources. The all-rows line (LA) and the line the unparsed / truncation rule picks are shown.")
    D.append("")
    D += desc_table(R)
    D.append("")
    D.append("### Subject groups (set A, descriptive)")
    D.append("")
    D.append("Groups from the subdomain names only: URDU = urdu literature, urdu language, urdu grammar; REGIONAL = pakistan studies, islamic studies; OTHER = the other 21 subdomains. Domain splits and a per-subdomain table are in the readout.")
    D.append("")
    D += group_table(R)
    D.append("")
    D.append("### Predictions")
    D.append("")
    D.append("Predictions (freeze N8, read on LA of set A) hit: %s of %s." % R["pred_hits"])
    D.append("")
    D += pred_table(R)
    D.append("")
    D.append("### Translations and dropped rows")
    D.append("")
    D += trans_lines(R)
    D.append("")
    D += drop_lines(R)
    D.append("")
    D.append(R["ledger"])
    if R["used"]:
        D.append("")
        D.append(R["used"])
    D.append("")
    D.append("Sources: `docs/handoff/D15_READOUT.md` (every number in this section), `docs/EFBPT_D15_NATIVE_URDU_FREEZE.md`; analysis pack `docs/handoff/D15_ANALYSIS.md`; "
             "tables `docs/thesis_tables/d15_*`; record in `experiments.md`.")
    D.append("")
    D.append("---")
    D.append("")
    return D


def n9_short(R, fam):
    l = next(x for x in R["n9"] if x.startswith("- %s:" % fam))
    return l[2:]


def key_findings(R, start):
    D = []
    D.append("**%d. The reading effect on native Urdu questions (D15).**" % start)
    D.append("UrduMMLU, %s native Urdu questions (100 per subdomain), set A n = %s. %s" % (R["n"], R["nA"], n9_short(R, "H1")))
    D.append("")
    D += wording_bullets(R, "H1")
    D.append("")
    D.append("**%d. Self-translation with no outside model (D15).**" % (start + 1))
    D.append(n9_short(R, "H2"))
    D.append("")
    D += wording_bullets(R, "H2")
    D += fam_table(R, "H2")[1]
    D.append("")
    D.append("**%d. The teacher-facts method on native Urdu questions (D15).**" % (start + 2))
    D.append("")
    D += wording_bullets(R, "H3")
    D.append("")
    D.append("**%d. D15 against its predictions and by subject group.**" % (start + 3))
    D.append("Predictions (freeze N8, set A) hit: %s of %s. Missed:" % R["pred_hits"])
    D.append("")
    missed = [r for r in R["pred"] if r[-1] == "miss"]
    D += ["- %s: observed %s pp." % (r[0], r[3]) for r in missed] or ["- none."]
    D.append("")
    D.append("The H1 pairs inside the subject groups (set A, descriptive, raw p):")
    D.append("")
    for r in R["splits"]:
        if r[0].startswith("group ") and r[2] != "missing" and r[1] in [x[0] for x in R["H1"]]:
            D.append("- %s, %s: %s → %s (%s pp, b %s / c %s, raw p %s; line %s, n %s)."
                     % (r[0][6:], r[1], TB.split2(r[5])[0], TB.split2(r[4])[0], r[6], r[7], r[8], r[9], r[2], r[3]))
    D.append("")
    return D


def current_state(R):
    parts = []
    for fam in ("H1", "H2", "H3"):
        for r in R[fam]:
            parts.append("%s %s" % (r[0], r[10] if r[1] != "missing" else "missing"))
    return ("**Measured on native Urdu questions (D15, %s).** Reading effect, self-translation and teacher facts on UrduMMLU (%s rows, 100 per subdomain; set A n = %s), "
            "wording fixed before the run: %s. N9 reading: %s %s See *Reading Effect and Teacher Facts on Native Urdu Questions (D15)*."
            % (R["date"], R["n"], R["nA"], "; ".join(parts), n9_short(R, "H1"), n9_short(R, "H2")))


def limits(R):
    head, rows = R["trans"]
    t = {r[0]: r for r in rows}
    out = ["- **D15 sample.** 100 rows from each of the 26 UrduMMLU subdomains (%s rows): the accuracies are means over subjects, not the dataset-weighted accuracy; "
           "a subject group of 300 rows can show only differences of about 6 pp (freeze N7). One run of each arm." % R["n"],
           "- **D15 set A.** %s of %s sampled rows are in set A; %s are dropped (no valid R1 translation, or refused by qwen3.8-max in M_ur, M_ue or M_r1)."
           % (R["nA"], R["n"], R["drop"]),
           "- **D15 translations.** R1 is by qwen3.8-max, the same model as M, so M_r1 is self-translation for M. Rows without a valid translation: R1 %s, TL %s, SL %s of %s rows "
           "(reading_failed or too_long); such a row counts as wrong in the arm that needs it, and the deciding-line rule can move a test to the rows parsed in both arms."
           % (int(t["R1"][4]) + int(t["R1"][5]), int(t["TL"][4]) + int(t["TL"][5]), int(t["SL"][4]) + int(t["SL"][5]), R["n"])]
    return out


def experiments_record(R):
    D = ["", "## EFBPT D15: READING EFFECT AND TEACHER FACTS ON NATIVE URDU QUESTIONS (UrduMMLU) (recorded %s)" % R["date"], ""]
    D.append("Sources: `docs/handoff/D15_READOUT.md` (every number, unless another file is named), the freeze `docs/EFBPT_D15_NATIVE_URDU_FREEZE.md`. Analysis pack: `docs/handoff/D15_ANALYSIS.md`.")
    D.append("")
    D.append("**Purpose (freeze):** test the reading effect on native Urdu questions (never translated from English), with controls for the translator, the instruction language and "
             "the subject type; and test the teacher-facts method on the same set. D15 selects nothing and tunes nothing; D11, D13 and D14 stand as they are.")
    D.append("")
    D.append("**Setup:** UrduMMLU (MBZUAI/UrduMMLU, CC BY 4.0, revision aae060a8564c261393de206c94a4557c850e7f2f); valid rows (four options, key A-D, no exact duplicate); "
             "100 rows per subdomain with random.Random(1515) (%s rows); set A = rows with a valid R1 translation and an answer in M_ur, M_ue and M_r1 (n = %s). "
             "All 11 arm files, 3 translation files and the facts file were sealed before scoring; no job computed accuracy; the readout script was committed and pushed before it read "
             "any D15 arm and was run once. Scoring: the CSQA fair extract_answer with the class [A-D]; unparsed = wrong. Lines and wording as D14 F6." % (R["n"], R["nA"]))
    D.append("")
    D.append("**The sample is equal per subject: overall accuracy is the mean over subjects, not the dataset-weighted accuracy.**")
    D.append("")
    D += arm_table(R)
    D.append("")
    for fam, title in (("H1", "**Family H1** (reading effect; 3 tests, Holm):"), ("H2", "**Family H2** (self-translation; 2 tests, Holm):"), ("H3", "**Family H3** (method; 1 test):")):
        D.append(title)
        D.append("")
        t, extra = fam_table(R, fam)
        D += t
        D.append("")
        D += wording_bullets(R, fam) + extra
        D.append("")
    D.append("**Reading (freeze N9):**")
    D.append("")
    D += R["n9"]
    D.append("")
    D.append("**Descriptive (set A):**")
    D.append("")
    D += desc_table(R)
    D.append("")
    D += group_table(R)
    D.append("")
    D.append("**Predictions** (freeze N8, set A) hit: %s of %s." % R["pred_hits"])
    D.append("")
    D += pred_table(R)
    D.append("")
    D.append("**Translations, dropped rows, cost:**")
    D.append("")
    D += trans_lines(R)
    D.append("")
    D += drop_lines(R)
    D.append("")
    D.append(R["ledger"])
    if R["used"]:
        D.append("")
        D.append(R["used"])
    D.append("")
    return D


def write(R):
    rd = open(README, encoding="utf-8").read()
    if "(D15)" in rd or "D15, " in rd:
        die("README.md already holds a D15 block")
    # 1 section before Repository Structure
    anchor = "## Repository Structure"
    if rd.count(anchor) != 1:
        die("README: anchor %r not found once" % anchor)
    rd = rd.replace(anchor, "\n".join(readme_section(R)) + "\n" + anchor, 1)
    # 2 key findings after the last numbered finding (before the '---' that precedes Current State)
    nums = [int(x) for x in re.findall(r"^\*\*(\d+)\. ", rd, flags=re.M)]
    start = max(nums) + 1
    cs = re.search(r"\n---\n\n## Current State \((\d{4}-\d{2}-\d{2})\)", rd)
    if not cs:
        die("README: Current State heading not found")
    rd = rd[:cs.start()] + "\n" + "\n".join(key_findings(R, start)).rstrip("\n") + "\n" + rd[cs.start():]
    # 3 current state: heading date and a paragraph after the D14 paragraph
    old_head = "## Current State (%s)" % cs.group(1)
    rd = rd.replace(old_head, "## Current State (%s)" % R["date"], 1)
    d14p = re.search(r"\*\*Measured on three more tasks \(D14, [^\n]*\n", rd)
    if not d14p:
        die("README: the D14 Current State paragraph was not found")
    rd = rd[:d14p.end()] + "\n" + current_state(R) + "\n" + rd[d14p.end():]
    # 4 limits: bullets before the Sources line of '### Limits'
    li = rd.find("### Limits\n")
    src = rd.find("\nSources: ", li)
    if li < 0 or src < 0:
        die("README: '### Limits' or its Sources line not found")
    end = rd.find("\n", src + 1)
    srcline = rd[src + 1:end]
    newsrc = srcline.replace("`docs/handoff/D14_READOUT.md` (D14 limits)", "`docs/handoff/D14_READOUT.md` (D14 limits), `docs/handoff/D15_READOUT.md` (D15 limits)")
    if newsrc == srcline:
        die("README: the Limits Sources line does not name the D14 readout as expected")
    rd = rd[:src] + "\n" + "\n".join(limits(R)) + "\n\n" + newsrc + rd[end:]
    with open(README, "w", encoding="utf-8") as f:
        f.write(rd)
    ex = open(EXPERIMENTS, encoding="utf-8").read()
    if "EFBPT D15" in ex:
        die("experiments.md already holds a D15 record")
    with open(EXPERIMENTS, "w", encoding="utf-8") as f:
        f.write(ex.rstrip("\n") + "\n" + "\n".join(experiments_record(R)))
    print("[write] README.md: D15 section, Key Findings %d-%d, Current State paragraph and date, Limits bullets; experiments.md: D15 record" % (start, start + 3))


def d15_blocks():
    """The D15 text in README.md and experiments.md, located by the headings and labels this script writes."""
    rd = open(README, encoding="utf-8").read()
    out = []
    a = rd.find("## Reading Effect and Teacher Facts on Native Urdu Questions (D15)")
    b = rd.find("## Repository Structure", a)
    out.append(("README D15 section", rd[a:b]))
    m = re.search(r"\*\*(\d+)\. The reading effect on native Urdu questions \(D15\)\.\*\*", rd)
    e = rd.find("\n---\n\n## Current State", m.start())
    out.append(("README Key Findings", rd[m.start():e]))
    p = rd.find("**Measured on native Urdu questions (D15, ")
    out.append(("README Current State", rd[p:rd.find("\n", p)]))
    out.append(("README Limits", "\n".join(l for l in rd.split("\n") if l.startswith("- **D15 "))))
    ex = open(EXPERIMENTS, encoding="utf-8").read()
    out.append(("experiments.md D15 record", ex[ex.find("## EFBPT D15: "):]))
    for name, txt in out:
        if not txt.strip():
            die("%s not found" % name)
    return out


def check():
    def norm(t):
        return t.replace("−", "-").lstrip("+-")
    src = set()
    for p in (READOUT, ANALYSIS, FREEZE):
        src |= {norm(t) for t in TB.NUM.findall(open(p, encoding="utf-8").read())}
    blocks = d15_blocks()
    for name in ("d15_arms", "d15_tests", "d15_groups", "d15_translations", "d15_cost"):
        for ext in (".tex", ".csv"):
            p = os.path.join("docs/thesis_tables", name + ext)
            lines = [l for l in open(p, encoding="utf-8").read().split("\n")
                     if not (l.startswith("%") or l.startswith("\\caption") or l.startswith("\\label") or l.startswith("\\begin{tabular}"))]
            blocks.append((p, "\n".join(lines)))
    n = bad = 0
    for name, txt in blocks:
        for w in ("Llama-3.1-70B-AWQ", "Llama-3.1-70B", "Qwen3-14B", "qwen3.8-max", "CC BY 4.0", "aae060a8564c261393de206c94a4557c850e7f2f"):
            txt = txt.replace(w, "")
        txt = re.sub(r"`[^`]*`", "", txt)
        for t in TB.NUM.findall(txt):
            n += 1
            if norm(t) not in src:
                bad += 1
                print("NOT IN THE READOUT, ANALYSIS OR FREEZE: %s: %s" % (name, t))
    print("[check] %d numbers in %d D15 blocks; not in the sources: %d" % (n, len(blocks), bad))
    if bad:
        die("number check failed")
    print("[check] PASS")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--print", action="store_true")
    ap.add_argument("--readout", default=None, help="testing hook for --print: read another readout file")
    args = ap.parse_args()
    if args.readout:
        if not args.print:
            die("--readout is for --print only")
        global READOUT
        READOUT = args.readout
    if args.check:
        check()
        return
    R = parse()
    if args.print:
        print("\n".join(readme_section(R)))
        print("\n".join(key_findings(R, 32)))
        print(current_state(R))
        print("\n".join(limits(R)))
        print("\n".join(experiments_record(R)))
        return
    write(R)
    check()


if __name__ == "__main__":
    main()
