#!/usr/bin/env python3
"""
d11_scorer_audit.py — D11 RTG Amendment 4: scorer audit (CPU only, DEV200 only).

Contract: docs/EFBPT_D11_RTG_AMENDMENT_4.md (MD5 pinned below). Reads SEALED
DEV200 files only (MD5 asserted); nothing under any eval458 directory is opened
(every path is checked). Nothing frozen is edited: E0 is the frozen extractor,
imported unchanged; E1-E3 (A4.3) are defined here, before any label exists.

BUILD (default, runs once, refuses to overwrite):
  items (A4.2): for arms A_r2 and TR, every row where the frozen label (E0) and
  the first-mention label (E1) differ, plus 20 rows per arm drawn from the rows
  where they agree: rng = random.Random(4242); rng.sample(agree rows of A_r2 in
  DEV200 order, 20), then rng.sample(agree rows of TR, 20) with the same rng.
  The items are then shuffled with a fresh random.Random(4242).
  docs/handoff/D11_SCORER_AUDIT_ITEMS.md  item number, Urdu question, generation
      (first 300 and last 250 characters; the whole text if it has <= 550)
  outputs/efbpt/d11/_audit/scorer_audit_key.json   arm, qid, gold, stratum and
      every extractor label per item. NEVER printed by this script.
  Only counts are printed.

SCORE (--score LABELS [--sensitivity]; run only after the adjudication):
  LABELS = a JSON object {"1": "yes", ...} or lines "<item> <label>" (space, tab,
  comma or colon separated); label in yes / no / none (none = no verdict, or
  both). Prints A4.3: for E0-E3 the agreement with the adjudicated labels in both
  strata and the stratified estimate per arm,
      (D * agreement_on_the_D_differ_rows + (200 - D) * agreement_on_the_20_agree_rows) / 200,
  and the VALIDATED scorer = highest mean of the two arms' stratified agreement
  (ties -> E2, then E3, then E1). Then A4.4 for DEV200: every sealed arm
  re-scored with the validated scorer (descriptive).
  The extractor labels of every item are RECOMPUTED from the sealed generations
  at score time (the key gives arm, qid and stratum); E0-E2 must equal the
  labels stored in the key. --sensitivity (Amendment 4 note N3: the ALT label
  file) prints the A4.3 table only: it selects nothing and runs no A4.4.

Candidate extractors (A4.3), all on the frozen strings ہاں / نہیں / marker:
  E0  frozen (efbpt_eval_dev200.extract_answer).
  E1  first-mention: E0 with the FIRST ہاں/نہیں substring of the answer segment
      instead of the last (= S1 of the DEV200 supplement; asserted equal to it).
  E2  E1 with whole-word matching: a mention counts only if the character
      before it and the character after it are not letters or combining marks
      (Unicode category L* / M*, or ZWNJ / ZWJ); یہاں, وہاں, جہاں, کہاں, انہیں,
      جنہیں and other longer words do not count. Rule 1 (JSON) and Rule 3
      (English yes/no, last) stay as in E0 / E1.
  E3  opening verdict, replacing Rule 2 ONLY (docs/EFBPT_D11_RTG_AMENDMENT_4_NOTE.md
      N1): Rule 1 (JSON) first, as frozen; then, in the answer segment (the text
      after the last marker; the whole text without a marker), the first
      whole-word ہاں / نہیں that starts within the first 12 whitespace-separated
      words; else the LAST whole-word mention of the segment; else Rule 3
      (English yes/no, last), as frozen; else unparsed. (The key written at
      build time holds E3 labels of the earlier reading without Rules 1 and 3;
      they are not used: see SCORE.)
"""
import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import OrderedDict
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from d11_common import load_set_rows, md5_file, die  # noqa: E402  (chdir to the repo root)
from efbpt_eval_dev200 import extract_answer, HAAN, NAHIN, MARKER, RE_JSON_ANSWER, RE_EN_WORD  # noqa: E402  frozen
from d5_balanced_extract import mcnemar  # noqa: E402
import d11_dev200_supplement as SUPP  # noqa: E402  sealed DEV200 arm list (MD5-pinned), sealed() loader, S1 variant

AMEND4_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_4.md"
AMEND4_MD5 = "605c56e95c05fcb05c0c248cbe7ccac1"
AMEND4_NOTE_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_4_NOTE.md"     # N1: E3 replaces Rule 2 only
ITEMS_PATH = "docs/handoff/D11_SCORER_AUDIT_ITEMS.md"
KEY_PATH = "outputs/efbpt/d11/_audit/scorer_audit_key.json"
AUDIT_ARMS = ("A_r2", "TR")
SEED = 4242
N_AGREE = 20
HEAD, TAIL = 300, 250
OPENING_WORDS = 12
LABELS = ("yes", "no", "none")
TIE_ORDER = ("E2", "E3", "E1", "E0")              # A4.3: ties -> E2, then E3, then E1
JOINERS = ("‌", "‍")


# ----------------------------------------------------------------------------
# A4.3 candidate extractors
# ----------------------------------------------------------------------------
def answer_segment(text):
    """The frozen Rule 2 segment: the text after the LAST marker; the whole text without one."""
    if MARKER in text:
        return text[text.rfind(MARKER) + len(MARKER):]
    return text


def _json_rule(text):
    m = None
    for m in RE_JSON_ANSWER.finditer(text):
        pass
    return m.group(1).lower() if m is not None else None


def _english_rule(text):
    m = None
    for m in RE_EN_WORD.finditer(text):
        pass
    return m.group(1).lower() if m is not None else None


def _letterlike(ch):
    return unicodedata.category(ch)[0] in ("L", "M") or ch in JOINERS


def whole_word_mentions(seg):
    """[(start index, 'yes' | 'no')] of ہاں / نہیں standing as separate words, in text order."""
    out = []
    for word, label in ((HAAN, "yes"), (NAHIN, "no")):
        i = seg.find(word)
        while i != -1:
            j = i + len(word)
            if not (i > 0 and _letterlike(seg[i - 1])) and not (j < len(seg) and _letterlike(seg[j])):
                out.append((i, label))
            i = seg.find(word, i + 1)
    return sorted(out)


def e0(text):
    return extract_answer(text)


def e1(text):
    """First-mention (S1): E0 with the first ہاں/نہیں substring of the segment instead of the last."""
    r = _json_rule(text)
    if r is not None:
        return r
    seg = answer_segment(text)
    i_haan, i_nahin = seg.find(HAAN), seg.find(NAHIN)
    if i_haan != -1 or i_nahin != -1:
        if i_nahin == -1:
            return "yes"
        if i_haan == -1:
            return "no"
        return "yes" if i_haan < i_nahin else "no"
    return _english_rule(text)


def e2(text):
    """E1 with whole-word matching."""
    r = _json_rule(text)
    if r is not None:
        return r
    ms = whole_word_mentions(answer_segment(text))
    if ms:
        return ms[0][1]
    return _english_rule(text)


def e3(text):
    """Opening verdict in place of Rule 2 only (note N1): Rule 1 (JSON) as frozen; then the first whole-word mention
    starting within the first 12 words of the answer segment, else the last whole-word mention; then Rule 3 (English)
    as frozen; else unparsed."""
    r = _json_rule(text)
    if r is not None:
        return r
    seg = answer_segment(text)
    ms = whole_word_mentions(seg)
    if not ms:
        return _english_rule(text)
    toks = list(re.finditer(r"\S+", seg))
    end = toks[min(OPENING_WORDS, len(toks)) - 1].end()
    opening = [m for m in ms if m[0] < end]
    return opening[0][1] if opening else ms[-1][1]


EXTRACTORS = OrderedDict([("E0", e0), ("E1", e1), ("E2", e2), ("E3", e3)])


def selftest():
    """Hand-made strings only (no data): the definitions above do what A4.3 says."""
    yahan, unhen = "ی" + HAAN, "ا" + NAHIN          # یہاں , انہیں
    fill = " ".join(["لفظ"] * 13)             # 13 filler words
    cases = [
        (NAHIN + "، x " + HAAN, ("yes", "no", "no", "no")),                 # opens no, later yes
        (HAAN + "، x " + NAHIN + " x", ("no", "yes", "yes", "yes")),         # opens yes, later no
        ("x " + yahan + " x", ("yes", "yes", None, None)),                        # یہاں is not a mention for E2/E3
        ("x " + unhen + " x", ("no", "no", None, None)),                          # انہیں is not a mention for E2/E3
        (yahan + " " + NAHIN, ("no", "yes", "no", "no")),
        (fill + " " + HAAN + " x " + NAHIN, ("no", "yes", "yes", "no")),          # no mention in the first 12 words -> E3 takes the last
        ("**" + NAHIN + "**۔ " + fill + " " + HAAN, ("yes", "no", "no", "no")),
        ("Yes, it is.", ("yes", "yes", "yes", "yes")),                            # Rule 3 (English) kept in every extractor
        ('{"answer": "no"} ' + HAAN, ("no", "no", "no", "no")),                   # Rule 1 (JSON) kept in every extractor
        ("x " + yahan + " x No.", ("yes", "yes", "no", "no")),                    # no whole-word mention -> Rule 3 for E2 / E3
        (NAHIN + " x " + MARKER + ": " + HAAN, ("yes", "yes", "yes", "yes")),     # marker cut
        ("x y z", (None, None, None, None)),
    ]
    for text, want in cases:
        got = tuple(f(text) for f in EXTRACTORS.values())
        if got != want:
            die("selftest: %r -> %s, expected %s" % (text, got, want))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def check_contract():
    if not os.path.exists(AMEND4_PATH):
        die("contract missing: " + AMEND4_PATH)
    got = md5_file(AMEND4_PATH)
    if got != AMEND4_MD5:
        die("%s MD5 %s != pinned %s" % (AMEND4_PATH, got, AMEND4_MD5))
    for p in (ITEMS_PATH, KEY_PATH):
        if "eval458" in p:
            die("refusing an eval458 path: %s" % p)
    return got


def load_arm(arm):
    path, want = SUPP.ARMS[arm]
    recs = SUPP.sealed(path, want)                           # MD5 asserted; refuses eval458 paths
    for r in recs:
        if e0(r["generation"]) != r["pred"]:
            die("%s qid %s: E0 does not reproduce the sealed pred" % (arm, r["qid"]))
        if e1(r["generation"]) != SUPP.extract(r["generation"], True):
            die("%s qid %s: E1 differs from the S1 first-mention variant" % (arm, r["qid"]))
    return recs


def label_str(x):
    return "none" if x is None else x


def fence_for(text):
    n = 3
    while "`" * n in text:
        n += 1
    return "`" * n


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def build():
    for p in (ITEMS_PATH, KEY_PATH):
        if os.path.exists(p):
            die("%s exists; the audit items are built once and never overwritten" % p)
    rows = load_set_rows("dev200")
    qids = [r["qid"] for r in rows]
    by_q = {r["qid"]: r for r in rows}
    rng = random.Random(SEED)
    items, strata = [], OrderedDict()
    for arm in AUDIT_ARMS:
        recs = load_arm(arm)
        if [r["qid"] for r in recs] != qids:
            die("%s: qids are not DEV200 in order" % arm)
        differ = [r for r in recs if e0(r["generation"]) != e1(r["generation"])]
        agree = [r for r in recs if e0(r["generation"]) == e1(r["generation"])]
        drawn = rng.sample(agree, N_AGREE)
        strata[arm] = OrderedDict([("n_rows", len(recs)), ("differ_total", len(differ)), ("agree_total", len(agree)),
                                   ("differ_in_audit", len(differ)), ("agree_in_audit", len(drawn))])
        for stratum, sel in (("differ", differ), ("agree", drawn)):
            for r in sel:
                items.append(OrderedDict([
                    ("arm", arm), ("stratum", stratum), ("qid", r["qid"]), ("gold", r["gold"]),
                    ("labels", OrderedDict((k, label_str(f(r["generation"]))) for k, f in EXTRACTORS.items())),
                    ("truncated", r["truncated"]), ("generation_chars", len(r["generation"])),
                    ("_question_ur", by_q[r["qid"]]["question_ur"]), ("_generation", r["generation"])]))
    random.Random(SEED).shuffle(items)

    L = ["# D11 scorer audit — items (Amendment 4 A4.2)", ""]
    for n, it in enumerate(items, 1):
        it["item"] = n
        g = it.pop("_generation")
        q = it.pop("_question_ur")
        L += ["## Item %d" % n, "", "Question: %s" % q, ""]
        if len(g) <= HEAD + TAIL:
            fz = fence_for(g)
            L += ["Generation (whole text):", fz, g, fz, ""]
        else:
            h, t = g[:HEAD], g[-TAIL:]
            fz = fence_for(h + t)
            L += ["Generation, first %d characters:" % HEAD, fz, h, fz, "", "Generation, last %d characters:" % TAIL, fz, t, fz, ""]
    text = "\n".join(L)
    for hidden in [it["qid"] for it in items]:
        if hidden in text:
            die("a qid would appear in the items file")
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    with open(ITEMS_PATH, "x", encoding="utf-8") as f:
        f.write(text)
    key = OrderedDict([
        ("what", "D11 RTG Amendment 4 A4.2 scorer audit key — hidden from the adjudicator; do not print"),
        ("amendment4_md5", AMEND4_MD5), ("script_md5", md5_file(os.path.abspath(__file__))),
        ("seed", SEED), ("draw", "rng = random.Random(4242); rng.sample(agree rows of A_r2, 20) then rng.sample(agree rows of TR, 20); items shuffled with a fresh random.Random(4242)"),
        ("python", sys.version.split()[0]),
        ("sealed_arms", OrderedDict((a, OrderedDict([("path", SUPP.ARMS[a][0]), ("md5", SUPP.ARMS[a][1])])) for a in AUDIT_ARMS)),
        ("items_file", OrderedDict([("path", ITEMS_PATH), ("md5", md5_file(ITEMS_PATH))])),
        ("strata", strata),
        ("items", [OrderedDict((k, it[k]) for k in ("item", "arm", "stratum", "qid", "gold", "labels", "truncated", "generation_chars")) for it in items])])
    with open(KEY_PATH, "x", encoding="utf-8") as f:
        json.dump(key, f, indent=1, ensure_ascii=False)
    print("[written] %s  (%d items; md5 %s)" % (ITEMS_PATH, len(items), key["items_file"]["md5"]))
    print("[written] %s  (key; not printed)" % KEY_PATH)
    print("\nitem counts (A4.2):")
    print("%-6s %14s %14s %16s %8s" % ("arm", "differ (all)", "agree (drawn)", "agree (of total)", "items"))
    for arm, s in strata.items():
        print("%-6s %14d %14d %16s %8d" % (arm, s["differ_in_audit"], s["agree_in_audit"], "%d of %d" % (s["agree_in_audit"], s["agree_total"]), s["differ_in_audit"] + s["agree_in_audit"]))
    print("%-6s %14d %14d %16s %8d" % ("total", sum(s["differ_in_audit"] for s in strata.values()), sum(s["agree_in_audit"] for s in strata.values()), "", len(items)))
    print("generation shown whole (<= %d characters): %d items; head %d + tail %d: %d items"
          % (HEAD + TAIL, sum(1 for it in items if it["generation_chars"] <= HEAD + TAIL), HEAD, TAIL, sum(1 for it in items if it["generation_chars"] > HEAD + TAIL)))


# ----------------------------------------------------------------------------
# score (A4.3) and DEV200 re-scoring (A4.4)
# ----------------------------------------------------------------------------
def read_labels(path, n_items):
    raw = open(path, encoding="utf-8").read()
    labels = {}
    try:
        obj = json.loads(raw)
        pairs = list(obj.items())
    except ValueError:
        pairs = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in re.split(r"[\s,:]+", line) if p]
            if len(parts) != 2:
                die("labels line %r is not '<item> <label>'" % line)
            pairs.append((parts[0], parts[1]))
    for k, v in pairs:
        k, v = int(k), str(v).strip().lower()
        if v not in LABELS:
            die("item %d: label %r is not one of %s" % (k, v, "/".join(LABELS)))
        if k in labels:
            die("item %d is labelled twice" % k)
        labels[k] = v
    if sorted(labels) != list(range(1, n_items + 1)):
        die("labels must cover items 1..%d exactly once (%d given)" % (n_items, len(labels)))
    return labels


def current_items(key):
    """The key's items with every extractor label recomputed NOW from the sealed generation of (arm, qid).
    E0-E2 must equal the labels stored in the key at build time; E3 is the note-N1 definition (the key's
    E3 labels are of the earlier reading and are only counted against)."""
    gens = {arm: {r["qid"]: r["generation"] for r in load_arm(arm)} for arm in AUDIT_ARMS}
    items, e3_changed = [], 0
    for it in key["items"]:
        lab = OrderedDict((e, label_str(f(gens[it["arm"]][it["qid"]]))) for e, f in EXTRACTORS.items())
        for e in ("E0", "E1", "E2"):
            if lab[e] != it["labels"][e]:
                die("item %d: %s recomputed differs from the key" % (it["item"], e))
        if (lab["E0"] != lab["E1"]) != (it["stratum"] == "differ"):
            die("item %d: stratum in the key does not match E0 vs E1" % it["item"])
        e3_changed += int(lab["E3"] != it["labels"]["E3"])
        items.append(OrderedDict([("item", it["item"]), ("arm", it["arm"]), ("stratum", it["stratum"]), ("labels", lab)]))
    return items, e3_changed


def agreement_table(strata, items, labels):
    """A4.3. Returns (per-arm table, mean stratified agreement per extractor as Fractions, validated extractor name)."""
    table, strat = OrderedDict(), OrderedDict((e, []) for e in EXTRACTORS)
    for arm, s in strata.items():
        table[arm] = OrderedDict()
        for e in EXTRACTORS:
            cell = OrderedDict()
            for stratum in ("differ", "agree"):
                its = [it for it in items if it["arm"] == arm and it["stratum"] == stratum]
                ok = sum(1 for it in its if it["labels"][e] == labels[it["item"]])
                cell[stratum] = (ok, len(its))
            (od, nd), (oa, na) = cell["differ"], cell["agree"]
            if nd != s["differ_total"] or na != N_AGREE:
                die("%s: stratum sizes in the key are inconsistent" % arm)
            est = (Fraction(od) + Fraction(s["agree_total"]) * Fraction(oa, na)) / Fraction(s["n_rows"])
            cell["stratified"] = est
            table[arm][e] = cell
            strat[e].append(est)
    mean = OrderedDict((e, sum(v) / len(v)) for e, v in strat.items())
    best = max(mean.values())
    validated = next(e for e in TIE_ORDER if mean[e] == best)
    return table, mean, validated


def rescore_dev200(name):
    """A4.4 (DEV200, descriptive): every sealed arm under extractor `name`, beside the frozen E0."""
    f = EXTRACTORS[name]
    out = OrderedDict()
    for arm in SUPP.ARMS:
        recs = load_arm(arm) if arm in AUDIT_ARMS else SUPP.sealed(*SUPP.ARMS[arm])
        out[arm] = [dict(qid=r["qid"], gold=r["gold"], pred=f(r["generation"]), pred_e0=r["pred"]) for r in recs]
    print("\nA4.4 — DEV200, every sealed arm re-scored with %s (descriptive; the frozen E0 stays the scorer of every gate)" % name)
    print("%-9s %10s %10s %12s %12s   %s" % ("arm", "E0 acc%", name + " acc%", "E0 unparsed", name + " unparsed", "vs A_r2 under %s: delta pp, b / c, exact two-sided p" % name))
    for arm, recs in out.items():
        n = len(recs)
        a0 = 100.0 * sum(1 for r in recs if r["pred_e0"] == r["gold"]) / n
        a1 = 100.0 * sum(1 for r in recs if r["pred"] == r["gold"]) / n
        line = "%-9s %10.2f %10.2f %12d %12d" % (arm, a0, a1, sum(1 for r in recs if r["pred_e0"] is None), sum(1 for r in recs if r["pred"] is None))
        if arm != "A_r2":
            b, c, p = mcnemar(recs, out["A_r2"], arm, "A_r2")
            line += "   %+.1f, %d / %d, p = %.4g" % (100.0 * (b - c) / n, b, c, p)
        print(line)
    return out


def score_mode(labels_path, sensitivity):
    if not os.path.exists(KEY_PATH) or not os.path.exists(ITEMS_PATH):
        die("build the items first")
    if "eval458" in labels_path:
        die("refusing an eval458 path")
    if not os.path.exists(AMEND4_NOTE_PATH):
        die("%s missing (N1 fixes the E3 rule scope used here)" % AMEND4_NOTE_PATH)
    key = json.load(open(KEY_PATH, encoding="utf-8"), object_pairs_hook=OrderedDict)
    if key["amendment4_md5"] != AMEND4_MD5 or md5_file(ITEMS_PATH) != key["items_file"]["md5"]:
        die("the items file or the amendment changed since the key was written")
    labels = read_labels(labels_path, len(key["items"]))
    items, e3_changed = current_items(key)
    table, mean, validated = agreement_table(key["strata"], items, labels)
    print("A4.3 — agreement of each candidate extractor with the adjudicated labels%s" % ("   [SENSITIVITY label set: selects nothing]" if sensitivity else ""))
    print("labels %s (md5 %s; %d items: yes %d / no %d / none %d); key md5 %s; items md5 %s; script md5 %s; note md5 %s"
          % (labels_path, md5_file(labels_path), len(labels), sum(1 for v in labels.values() if v == "yes"), sum(1 for v in labels.values() if v == "no"),
             sum(1 for v in labels.values() if v == "none"), md5_file(KEY_PATH), md5_file(ITEMS_PATH), md5_file(os.path.abspath(__file__)), md5_file(AMEND4_NOTE_PATH)))
    print("extractor labels recomputed from the sealed generations: E0-E2 equal the key on all %d items; E3 (note N1 scope) differs from the key's earlier-reading E3 on %d item(s)"
          % (len(items), e3_changed))
    for arm in table:
        s = key["strata"][arm]
        its = [it for it in items if it["arm"] == arm]
        print("\n%s: differ stratum %d rows (all audited), agree stratum %d of %d rows audited; adjudicated yes / no / none: differ %s, agree %s"
              % (arm, s["differ_total"], N_AGREE, s["agree_total"],
                 " / ".join(str(sum(1 for it in its if it["stratum"] == "differ" and labels[it["item"]] == l)) for l in LABELS),
                 " / ".join(str(sum(1 for it in its if it["stratum"] == "agree" and labels[it["item"]] == l)) for l in LABELS)))
        print("  %-4s %22s %22s %24s" % ("", "agreement, differ rows", "agreement, agree rows", "stratified estimate (200)"))
        for e, cell in table[arm].items():
            (od, nd), (oa, na) = cell["differ"], cell["agree"]
            print("  %-4s %15d / %-4d %15d / %-4d %23.2f%%" % (e, od, nd, oa, na, 100.0 * float(cell["stratified"])))
    print("\nitems right, both arms together:")
    print("  %-4s %22s %22s %16s" % ("", "of the 65 differ items", "of the 40 agree items", "of all 105"))
    for e in EXTRACTORS:
        d = sum(table[arm][e]["differ"][0] for arm in table)
        a = sum(table[arm][e]["agree"][0] for arm in table)
        print("  %-4s %22d %22d %16d" % (e, d, a, d + a))
    print("\nmean of the two arms' stratified agreement: " + "; ".join("%s %.2f%%" % (e, 100.0 * float(v)) for e, v in mean.items()))
    if sensitivity:
        print("highest mean under this SENSITIVITY label set (ties -> E2, E3, E1): %s — it selects nothing (note N3); no A4.4 table is run from it" % validated)
        return
    print("VALIDATED scorer (A4.3: highest mean; ties -> E2, then E3, then E1): %s" % validated)
    rescore_dev200(validated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", metavar="LABELS", default=None, help="A4.3 / A4.4 from an adjudicated labels file; run only after the adjudication")
    ap.add_argument("--sensitivity", action="store_true", help="with --score: a sensitivity label set (note N3): A4.3 table only, selects nothing")
    args = ap.parse_args()
    check_contract()
    selftest()
    if args.sensitivity and not args.score:
        ap.error("--sensitivity needs --score")
    if args.score:
        score_mode(args.score, args.sensitivity)
    else:
        build()


if __name__ == "__main__":
    main()
