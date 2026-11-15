#!/usr/bin/env python3
"""
d13_analysis.py — D13 analysis pack. Descriptive only: it changes no verdict, no wording and no
number of docs/handoff/D13_READOUT.md, and it is not part of any test family.

CPU only; no model, no API; the key file is never opened. It reads the sealed arm files, the
sealed readings, teacher facts and retrieval files, and the gold labels, through the loaders of
d13_readout.py (imported; seals and MD5s are verified there). It writes exactly one file:
docs/handoff/D13_ANALYSIS.md. An existing file is never overwritten with different content.

Sections
  A  Reading effect, paired on eval458, lines L1 (label) and L2 (E3), with b, c and exact p.
  B  Facts effect: TR vs RO_eval split by gold yes / gold no, and by whether the R1 reading
     recovers every gold proper noun (row-M1 = 1, d11r_score.m1_row) or not.
  C  Overlap tables (both right / only first / only second / both wrong).
  D  Small students: unparsed rows sorted into surface classes (defined below, before any row
     was looked at), then 15 random unparsed rows per arm (seed 1010) for a hand check.
  E  Truncated rows per arm: count, and the E0 and E3 label split on them.
  F  Hand-check rows, 20 random each (seed 1010), in the format of the D11 readout §8.
  G  Cost: tokens and seconds per row per arm, copied from the readout; reader and teacher
     tokens of the TR pipeline from the sealed D11 files, where they are recorded.

PASS CHECK (fatal, before anything is written): the L1 accuracy and correct count of every arm
loaded here equal the per-arm rows printed in docs/handoff/D13_READOUT.md (MD5 pinned).

Surface classes of an unparsed generation (section D). A row is unparsed when the frozen
extractor E0 finds no JSON answer, no ہاں / نہیں string and no English yes / no word. The classes
are tested in this order and the first that fits is taken:
  EMPTY               no character other than white space
  VARIANT_SPELLING    a whole word that equals ہاں or نہیں after code-point normalisation
                      (Arabic heh / heh doachashmee -> ہ, Arabic yeh / alef maksura -> ی,
                      noon ghunna -> ن, diacritics, tatweel and zero-width joiners removed)
  DEVANAGARI          at least one Devanagari letter (U+0900-U+097F)
  OTHER_VERDICT_WORD  a whole word from: جی, نہ, نا, درست, غلط, صحیح, سچ, جھوٹ, true, false,
                      correct, incorrect (compared after the same normalisation, lower case)
  LOOP                some 30-character window occurs at 5 or more positions
  TRUNCATED_NO_LOOP   the row reached the token limit and is none of the above
  LATIN_ONLY          no Arabic-script letter and at least one Latin letter
  URDU_NO_VERDICT     Arabic-script text with none of the above
  OTHER               anything else

  --synthetic-root DIR  testing hook: read the synthetic arm files that
                        `d13_readout.py --synthetic-root DIR` wrote, skip the PASS CHECK and write
                        DIR/D13_ANALYSIS_SYNTHETIC.md. No D13 arm is opened.
"""
import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d13_common as C  # noqa: E402
from d13_common import die, md5_file, read_jsonl, load_set_rows  # noqa: E402
import d13_readout as RD  # noqa: E402  loaders, scorers, lines (nothing is run on import)
from d11r_score import m1_row  # noqa: E402

READOUT = ("docs/handoff/D13_READOUT.md", "8ff667881213f108871a09229a881653")
OUT_PATH = "docs/handoff/D13_ANALYSIS.md"
SEED = 1010
N_UNPARSED_SHOW, N_HAND, GEN_CHARS, UNP_CHARS = 15, 20, 110, 200
SET = "eval458"
REF_ARMS = ("A_r2_eval", "TR", "TR_L1_eval")
SMALL = ("Llama8B_A", "Llama8B_TR", "Gemma9B_A", "Gemma9B_TR", "Qwen8B_A", "Qwen8B_TR")
READING_PAIRS = [("API_tr_R1", "API_direct_ur_eval"), ("T_tr_R1", "T_direct_F_eval"), ("T_tr_L1", "T_direct_F_eval"), ("RO_eval", "A_r2_eval")]
OVERLAP_PAIRS = [("TR", "T_tr_R1"), ("TR", "API_tr_R1"), ("TR", "RO_eval"), ("TR", "API_direct_ur_eval")]
HAND_SETS = [("TR", "T_tr_R1"), ("T_tr_R1", "TR"), ("TR", "RAG_en"), ("API_tr_R1", "TR")]      # first right, second wrong
T_FACTS = "eval458/t_facts_R1"

# ---- surface classes (section D) ----
_MAP = {0x0647: 0x06C1, 0x06BE: 0x06C1, 0x06C0: 0x06C1, 0x06C2: 0x06C1, 0x064A: 0x06CC, 0x0649: 0x06CC, 0x06BA: 0x0646, 0x0643: 0x06A9}
_DROP = set(range(0x064B, 0x0653)) | {0x0670, 0x0640, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF}
HAAN_N = "".join(chr(c) for c in (0x06C1, 0x0627, 0x0646))             # ہاں after normalisation
NAHIN_N = "".join(chr(c) for c in (0x0646, 0x06C1, 0x06CC, 0x0646))    # نہیں after normalisation
CLASSES = ("EMPTY", "VARIANT_SPELLING", "DEVANAGARI", "OTHER_VERDICT_WORD", "LOOP", "TRUNCATED_NO_LOOP", "LATIN_ONLY", "URDU_NO_VERDICT", "OTHER")


def norm_ur(s):
    return "".join(chr(_MAP.get(ord(ch), ord(ch))) for ch in s if ord(ch) not in _DROP).lower()


VERDICT_WORDS = {norm_ur(w) for w in ("جی", "نہ", "نا", "درست", "غلط", "صحیح", "سچ", "جھوٹ", "true", "false", "correct", "incorrect")}


def words(s):
    out, cur = [], []
    for ch in s:
        if unicodedata.category(ch)[0] in ("L", "M"):
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def is_loop(text):
    if len(text) < 30:
        return False
    return max(Counter(text[i:i + 30] for i in range(len(text) - 29)).values()) >= 5


def surface_class(text, truncated):
    if not text.strip():
        return "EMPTY"
    ws = set(words(norm_ur(text)))
    if HAAN_N in ws or NAHIN_N in ws:
        return "VARIANT_SPELLING"
    if any(0x0900 <= ord(ch) <= 0x097F and unicodedata.category(ch)[0] == "L" for ch in text):
        return "DEVANAGARI"
    if ws & VERDICT_WORDS:
        return "OTHER_VERDICT_WORD"
    if is_loop(text):
        return "LOOP"
    if truncated:
        return "TRUNCATED_NO_LOOP"
    arabic = any(0x0600 <= ord(ch) <= 0x06FF and unicodedata.category(ch)[0] == "L" for ch in text)
    latin = any("a" <= ch.lower() <= "z" for ch in text)
    if not arabic and latin:
        return "LATIN_ONLY"
    if arabic:
        return "URDU_NO_VERDICT"
    return "OTHER"


def classes_selftest():
    cases = [("", False, "EMPTY"), ("  \n", True, "EMPTY"),
             ("جواب: " + "".join(chr(c) for c in (0x0647, 0x0627, 0x06BA)), False, "VARIANT_SPELLING"),          # Arabic heh
             ("".join(chr(c) for c in (0x0646, 0x06C1, 0x064A, 0x06BA)) + " ممکن", False, "VARIANT_SPELLING"),     # Arabic yeh
             ("इसका उत्तर", False, "DEVANAGARI"), ("یہ بیان درست ہے", False, "OTHER_VERDICT_WORD"), ("This is True.", False, "OTHER_VERDICT_WORD"),
             ("abc def ghi jkl mno pqr stu vwx " * 6, True, "LOOP"), ("ایک طویل جواب جو ختم", True, "TRUNCATED_NO_LOOP"),
             ("It depends on the context.", False, "LATIN_ONLY"), ("یہ ایک پیچیدہ سوال ہے", False, "URDU_NO_VERDICT"), ("12345 ...", False, "OTHER")]
    for text, tr, want in cases:
        got = surface_class(text, tr)
        if got != want:
            die("surface-class self-test: %r -> %s, expected %s" % (text, got, want))


# ---- helpers ----
def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def one(s, n):
    return cell(" ".join(str(s).split())[:n])


def lab(v):
    return v if v is not None else "None (unparsed)"


def readout_arm_rows():
    """(set, arm) -> 'acc (correct)' of the label row of the per-arm tables of the readout."""
    if md5_file(READOUT[0]) != READOUT[1]:
        die("%s MD5 %s != pinned %s" % (READOUT[0], md5_file(READOUT[0]), READOUT[1]))
    out, cost, sec, sub = {}, {}, None, None
    for l in open(READOUT[0], encoding="utf-8").read().split("\n"):
        if l.startswith("## "):
            sec = l[3:].split(".")[0]
        elif l.startswith("### "):
            sub = "eval458" if l[4:].startswith("eval458") else "dev200" if l[4:].startswith("DEV200") else None
        elif sec == "2" and sub and l.startswith("| ") and "(label)" in l:
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            out[(sub, c[0])] = c[2]
        elif sec == "10" and l.startswith(("| eval458", "| dev200")):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            cost[(c[0], c[1])] = c[2:]
    return out, cost


def raw_generations(arm, set_name):
    if arm in RD.REFS[set_name]:
        recs = read_jsonl(C.SEALED[RD.REFS[set_name][arm][0]][0])
        return [r["generation"] if "generation" in r else r["raw"] for r in recs]
    return [r["generation"] for r in C.read_rows(C.arm_paths(arm, set_name, "FULL")["rows"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-root", default=None)
    args = ap.parse_args()
    synthetic = bool(args.synthetic_root)
    out_path = OUT_PATH
    if synthetic:
        root = os.path.abspath(args.synthetic_root)
        if root == C.ROOT or root.startswith(C.ROOT + os.sep):
            die("--synthetic-root must be outside the repository")
        C.OUT_ROOT = os.path.join(root, "outputs_d13")
        out_path = os.path.join(root, "D13_ANALYSIS_SYNTHETIC.md")
    else:
        C.require_committed()
    classes_selftest()
    C.verify_contracts()

    rows = {s: load_set_rows(s) for s in C.SETS}
    arms, gens = {}, {}
    for s in C.SETS:
        for a in RD.REFS[s]:
            arms[(s, a)], _ = RD.load_ref(s, a, rows[s])
        for a, spec in C.ARMS.items():
            if s in spec["sets"]:
                arms[(s, a)] = RD.load_d13(a, s, rows[s])[0]
    for (s, a) in arms:
        gens[(s, a)] = raw_generations(a, s)
        if len(gens[(s, a)]) != len(arms[(s, a)]):
            die("%s %s: generations and scored rows differ in length" % (s, a))

    # ---- PASS CHECK ----
    printed, cost = readout_arm_rows()
    n_checked = 0
    for (s, a), recs in arms.items():
        st = RD.arm_stats(recs, "main")
        got = "%.2f (%d)" % (st["acc"], st["correct"])
        if (s, a) not in printed:
            die("PASS CHECK: %s %s is not in the per-arm tables of the readout" % (s, a))
        if got != printed[(s, a)] and not synthetic:
            die("PASS CHECK FAILED: %s %s L1 accuracy here %s, readout %s" % (s, a, got, printed[(s, a)]))
        n_checked += 1
    print("[pass-check] %d arms: L1 accuracy and correct count equal the readout%s" % (n_checked, " (SKIPPED: synthetic)" if synthetic else ""), flush=True)

    ev = {a: arms[(SET, a)] for (s, a) in arms if s == SET}
    R = rows[SET]
    ctx = C.make_ctx(SET, "FULL")
    reading = ctx["readings"]["R1"]
    C.verify_sealed((T_FACTS,))
    tfacts = {r["qid"]: r for r in read_jsonl(C.SEALED[T_FACTS][0])}
    m1 = {}
    for r in R:
        sc, found, n = m1_row(r["question_en"], reading[r["qid"]])
        m1[r["qid"]] = (sc, found, n)
    gold = {r["qid"]: r["gold"] for r in R}

    L = []
    w = L.append
    w("# D13 analysis pack (descriptive)%s" % (" — SYNTHETIC DATA, A TEST OF THE SCRIPT" if synthetic else ""))
    w("")
    w("Written by `eval/error_analysis_tests/efbpt/d13_analysis.py` (MD5 `%s`). Descriptive only: it changes no verdict and no wording of "
      "`%s` (MD5 `%s`), and nothing here belongs to a test family; p-values are raw, exact two-sided McNemar, with no correction. "
      "Set: eval458 (n = %d) unless stated. \"label\" = E0 (E0fa for the two T_tr arms, identical to E0 on every row); E3 = the validated scorer; unparsed = wrong. "
      "b = the first-named arm right and the second wrong; c = the reverse."
      % (md5_file(os.path.abspath(__file__)), READOUT[0], READOUT[1], len(R)))
    w("")
    w("Check before writing: the L1 accuracy and correct count of all %d arm files loaded here (eval458 and DEV200) equal the per-arm rows printed in the readout." % n_checked)
    w("")

    def line_row(name, ln):
        return "| %s | %s |" % (name, " | ".join(RD.line_cells(ln)))

    # ---- A ----
    w("## A. Reading effect: the same model with the English reading against the Urdu question")
    w("")
    w("The first arm gets the English reading (or the reading added as one line, for RO_eval); the second gets the Urdu question alone.")
    w("")
    w("| comparison | line | n | acc first | acc second | Δ pp | b | c | exact two-sided p |")
    w("|---|---|---|---|---|---|---|---|---|")
    for x, y in READING_PAIRS:
        ls = RD.all_lines(ev[x], ev[y])
        for k, nm in (("L1", "L1 label"), ("L2", "L2 E3")):
            w("| %s vs %s | %s | %s |" % (x, y, nm, " | ".join(RD.line_cells(ls[k]))))
    w("")

    # ---- B ----
    w("## B. Facts effect: TR vs RO_eval, split")
    w("")
    w("TR = reading -> teacher facts -> student. RO_eval = the student with the reading as one line and no facts. "
      "row-M1 = the share of the gold proper nouns of the row that the R1 reading recovers (`d11r_score.m1_row`); rows without a gold proper noun have no row-M1.")
    w("")
    groups = OrderedDict([
        ("all rows", lambda q: True), ("gold yes", lambda q: gold[q] == "yes"), ("gold no", lambda q: gold[q] == "no"),
        ("row-M1 = 1 (every gold proper noun recovered)", lambda q: m1[q][0] is not None and m1[q][1] == m1[q][2]),
        ("row-M1 < 1 (at least one gold proper noun missed)", lambda q: m1[q][0] is not None and m1[q][1] < m1[q][2]),
        ("no gold proper noun", lambda q: m1[q][0] is None),
    ])
    w("| rows | scorer | n | acc TR | acc RO_eval | Δ pp | b | c | exact two-sided p |")
    w("|---|---|---|---|---|---|---|---|---|")
    for name, f in groups.items():
        for key, nm in (("main", "label"), ("e3", "E3")):
            ln = RD.line(ev["TR"], ev["RO_eval"], key, lambda a, b, f=f: f(a["qid"]))
            w("| %s | %s | %s |" % (name, nm, " | ".join(RD.line_cells(ln)) if ln["n"] else "0 | — | — | — | 0 | 0 | — "))
    w("")
    w("For comparison, the same split for RO_eval vs A_r2_eval (translation line against the student alone):")
    w("")
    w("| rows | scorer | n | acc RO_eval | acc A_r2_eval | Δ pp | b | c | exact two-sided p |")
    w("|---|---|---|---|---|---|---|---|---|")
    for name, f in groups.items():
        for key, nm in (("main", "label"), ("e3", "E3")):
            ln = RD.line(ev["RO_eval"], ev["A_r2_eval"], key, lambda a, b, f=f: f(a["qid"]))
            w("| %s | %s | %s |" % (name, nm, " | ".join(RD.line_cells(ln)) if ln["n"] else "0 | — | — | — | 0 | 0 | — "))
    w("")

    # ---- C ----
    w("## C. Overlap tables")
    w("")
    w("| first vs second | scorer | both right | only first right | only second right | both wrong | n |")
    w("|---|---|---|---|---|---|---|")
    for x, y in OVERLAP_PAIRS:
        for key, nm in (("main", "label"), ("e3", "E3")):
            c = Counter((a[key] == a["gold"], b[key] == b["gold"]) for a, b in zip(ev[x], ev[y]))
            w("| %s vs %s | %s | %d | %d | %d | %d | %d |" % (x, y, nm, c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)], sum(c.values())))
    w("")
    w("By gold label, under the label scorer:")
    w("")
    w("| first vs second | gold | both right | only first right | only second right | both wrong | n |")
    w("|---|---|---|---|---|---|---|")
    for x, y in OVERLAP_PAIRS:
        for g in ("yes", "no"):
            c = Counter((a["main"] == a["gold"], b["main"] == b["gold"]) for a, b in zip(ev[x], ev[y]) if a["gold"] == g)
            w("| %s vs %s | %s | %d | %d | %d | %d | %d |" % (x, y, g, c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)], sum(c.values())))
    w("")

    # ---- D ----
    w("## D. Small students: the unparsed rows")
    w("")
    w("A row is unparsed when the frozen extractor E0 finds no JSON answer, no ہاں / نہیں string and no English yes / no word. Surface classes, written into the script "
      "before any row was looked at; they are tested in this order and the first that fits is taken:")
    w("")
    w("| class | definition |")
    w("|---|---|")
    for nm, d in (("EMPTY", "no character other than white space"),
                  ("VARIANT_SPELLING", "a whole word that equals ہاں or نہیں after code-point normalisation (Arabic heh and heh doachashmee to ہ, Arabic yeh and alef maksura to ی, noon ghunna to ن; diacritics, tatweel and zero-width joiners removed)"),
                  ("DEVANAGARI", "at least one Devanagari letter"),
                  ("OTHER_VERDICT_WORD", "a whole word from: جی, نہ, نا, درست, غلط, صحیح, سچ, جھوٹ, true, false, correct, incorrect"),
                  ("LOOP", "some 30-character window occurs at 5 or more positions"),
                  ("TRUNCATED_NO_LOOP", "the row reached the token limit and is none of the above"),
                  ("LATIN_ONLY", "no Arabic-script letter and at least one Latin letter"),
                  ("URDU_NO_VERDICT", "Arabic-script text with none of the above"),
                  ("OTHER", "anything else")):
        w("| %s | %s |" % (nm, d))
    w("")
    w("| arm | unparsed (E0) | " + " | ".join(CLASSES) + " | of them truncated |")
    w("|---|---|" + "---|" * (len(CLASSES) + 1))
    unp = OrderedDict()
    for a in SMALL:
        idx = [i for i, r in enumerate(ev[a]) if r["main"] is None]
        cl = [surface_class(gens[(SET, a)][i], ev[a][i]["trunc"]) for i in idx]
        unp[a] = (idx, cl)
        cnt = Counter(cl)
        w("| %s | %d | %s | %d |" % (a, len(idx), " | ".join(str(cnt[c]) for c in CLASSES), sum(1 for i in idx if ev[a][i]["trunc"])))
    w("")
    w("The same rows under E3: " + "; ".join("%s %d of %d also unparsed under E3" % (a, sum(1 for i in unp[a][0] if ev[a][i]["e3"] is None), len(unp[a][0])) for a in SMALL) + ".")
    w("")
    for a in SMALL:
        idx, cl = unp[a]
        pick = sorted(random.Random(SEED).sample(range(len(idx)), min(N_UNPARSED_SHOW, len(idx))))
        w("### D. %s — %d random unparsed rows of %d (seed %d), first %d characters, white space collapsed" % (a, len(pick), len(idx), SEED, UNP_CHARS))
        w("")
        w("| # | qid | gold | class | truncated | E3 label | generation |")
        w("|---|---|---|---|---|---|---|")
        for n, j in enumerate(pick, 1):
            i = idx[j]
            r = ev[a][i]
            w("| %d | `%s` | %s | %s | %s | %s | %s |" % (n, r["qid"], r["gold"], cl[j], "yes" if r["trunc"] else "no", lab(r["e3"]), one(gens[(SET, a)][i], UNP_CHARS)))
        w("")

    # ---- E ----
    w("## E. Truncated rows per arm")
    w("")
    w("Truncated = the row reached the token limit. Label split on the truncated rows only.")
    w("")
    w("| set | arm | truncated | E0: yes | E0: no | E0: unparsed | label right | E3: yes | E3: no | E3: unparsed | E3 right |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s, order in (("eval458", RD.EVAL_ORDER), ("dev200", RD.DEV_ORDER)):
        for a in order:
            tr = [r for r in arms[(s, a)] if r["trunc"]]
            c0, c3 = Counter(r["e0"] for r in tr), Counter(r["e3"] for r in tr)
            w("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (s, a, len(tr), c0["yes"], c0["no"], c0[None], sum(1 for r in tr if r["main"] == r["gold"]),
                                                                    c3["yes"], c3["no"], c3[None], sum(1 for r in tr if r["e3"] == r["gold"])))
    w("")

    # ---- F ----
    w("## F. Hand-check rows")
    w("")
    w("Format of the D11 readout §8: qid, gold, Urdu question, gold question_en, the R1 reading (row-M1 in parentheses; \"-\" = no gold proper noun), the first two teacher "
      "facts of TR, then the two answers: the label in bold and the first %d characters of the generation (white space collapsed). %d rows drawn at random (seed %d) from "
      "the rows where the first arm is right and the second wrong under the label; shown in eval458 file order. Right and wrong are as scored, so a bold label can differ "
      "from the opening word of a generation." % (GEN_CHARS, N_HAND, SEED))
    w("")
    retr = None
    for x, y in HAND_SETS:
        pool = [i for i, (a, b) in enumerate(zip(ev[x], ev[y])) if a["main"] == a["gold"] and b["main"] != b["gold"]]
        pick = sorted(random.Random(SEED).sample(pool, min(N_HAND, len(pool))))
        with_titles = "RAG_en" in (x, y)
        if with_titles and retr is None:
            retr, _ = C.load_retrieval("RAG_en", ctx)
        w("### F. %s right and %s wrong — %d of %d rows" % (x, y, len(pick), len(pool)))
        w("")
        w("| # | qid | gold | Urdu question | gold question_en | R1 reading (row-M1) | fact 1 | fact 2 | %s%s | %s |" % ("the 5 retrieved titles (RAG_en) | " if with_titles else "", x, y))
        w("|---|---|---|---|---|---|---|---|---|---|" + ("---|" if with_titles else ""))
        for n, i in enumerate(pick, 1):
            row, q = R[i], R[i]["qid"]
            facts = list(tfacts[q].get("facts") or [])
            f1, f2 = (facts + ["", ""])[:2]
            mm = "-" if m1[q][0] is None else "%.2f" % m1[q][0]
            titles = (" ; ".join(cell(h["title"]) for h in retr[i]["hits"]) + " | ") if with_titles else ""
            if with_titles and retr[i]["qid"] != q:
                die("retrieval rows are not aligned")
            w("| %d | `%s` | %s | %s | %s | %s (%s) | %s | %s | %s**%s** — %s | **%s** — %s |"
              % (n, q, row["gold"], cell(row["question_ur"]), cell(row["question_en"]), cell(reading[q]), mm, cell(f1), cell(f2), titles,
                 lab(ev[x][i]["main"]), one(gens[(SET, x)][i], GEN_CHARS), lab(ev[y][i]["main"]), one(gens[(SET, y)][i], GEN_CHARS)))
        w("")

    # ---- G ----
    w("## G. Cost")
    w("")
    w("Per arm, copied from §10 of the readout (`%s`): rows, truncated rows (%%), seconds per row, generated tokens per row, seconds total, generated tokens total." % READOUT[0])
    w("")
    w("| set | arm | rows | truncated rows (%) | seconds per row (mean) | generated tokens per row (mean) | seconds total | generated tokens total |")
    w("|---|---|---|---|---|---|---|---|")
    for s, order in (("eval458", RD.EVAL_ORDER), ("dev200", RD.DEV_ORDER)):
        for a in order:
            w("| %s | %s | %s |" % (s, a, " | ".join(cost[(s, a)])))
    w("")
    w("### The whole TR pipeline on eval458 (reader R1 -> teacher -> student)")
    w("")
    rd_recs = read_jsonl(ctx["reading_info"]["R1"]["file"])
    if [r["qid"] for r in rd_recs] != [r["qid"] for r in R]:
        die("reading file is not eval458 in order")

    def col(recs, k):
        v = [r[k] for r in recs if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        return v

    def stat(v):
        return ("%.2f" % (sum(v) / len(v)), "%d" % sum(v) if all(float(x).is_integer() for x in v) else "%.0f" % sum(v), len(v)) if v else ("not recorded", "not recorded", 0)

    tf = [tfacts[r["qid"]] for r in R]
    tr_tok = [r["tokens"] for r in ev["TR"] if isinstance(r["tokens"], (int, float))]
    w("| stage | model | field in the sealed file | rows with the field | mean per row | total |")
    w("|---|---|---|---|---|---|")
    for stage, model, recs, k in (("reader R1", "qwen3.8-max (API)", rd_recs, "prompt_tokens"), ("reader R1", "qwen3.8-max (API)", rd_recs, "output_tokens"),
                                  ("reader R1", "qwen3.8-max (API)", rd_recs, "latency_s"), ("teacher", "Llama-3.1-70B-AWQ", tf, "n_gen_tokens"),
                                  ("teacher", "Llama-3.1-70B-AWQ", tf, "prompt_tokens"), ("teacher", "Llama-3.1-70B-AWQ", tf, "seconds")):
        m, t, n = stat(col(recs, k))
        w("| %s | %s | `%s` | %d | %s | %s |" % (stage, model, k, n, m, t))
    m, t, n = stat(tr_tok)
    w("| student (arm TR) | Qwen3-14B | `n_gen_tokens` | %d | %s | %s |" % (n, m, t))
    w("| student (arm TR) | Qwen3-14B | `seconds` | 0 | not recorded | not recorded |")
    w("")
    seal_p, start_p = (os.path.join(os.path.dirname(C.SEALED[T_FACTS][0]), n_) for n_ in ("T_SEAL.json", "T_START.json"))
    tw = ""
    if os.path.exists(seal_p) and os.path.exists(start_p):
        sl, stt = json.load(open(seal_p, encoding="utf-8")), json.load(open(start_p, encoding="utf-8"))
        tw = (" The teacher job %s ran from %s to %s (wall clock of the whole job, model loading included)." % (sl.get("slurm_job_id"), stt.get("start_time"), sl.get("end_time")))
    w("What the sealed D11 files record: the reader file has prompt tokens, output tokens and request latency per row; the teacher file has generated tokens per row "
      "but no prompt tokens and no per-row time; the sealed student file of TR has generated tokens per row but no per-row time, so no seconds-per-row figure exists for "
      "the TR pipeline." + tw + " Reader file: `%s`; teacher file: `%s`." % (ctx["reading_info"]["R1"]["file"], C.SEALED[T_FACTS][0]))
    w("")
    text = "\n".join(L) + "\n"
    if os.path.exists(out_path):
        if open(out_path, encoding="utf-8").read() != text:
            die("%s exists with different content; not overwritten" % out_path)
        print("[unchanged] %s" % out_path)
    else:
        with open(out_path, "x", encoding="utf-8") as f:
            f.write(text)
        print("[written] %s (%d lines)" % (out_path, len(L)))


if __name__ == "__main__":
    main()
