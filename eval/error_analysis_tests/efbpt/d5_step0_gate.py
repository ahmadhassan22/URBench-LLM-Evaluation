#!/usr/bin/env python3
"""
D5 STEP 0 — existence gate.  CPU only, no model, no cluster scan.

Question: across all 71 D4 rows, how often does a SUPPLIED title receive
ZERO extracted facts?

Declared gate (docs/EFBPT_PLAN_A_FREEZE.md, DIAGNOSTIC D5):
  >= 30% of multi-title rows have at least one zero-fact title -> D5 RUNS
  <  30%                                                       -> D5 CANCELLED

READ-ONLY. Writes no files. All output to stdout.

ATTRIBUTION RULE (declared: "normalized name or clear head-noun appears")
  A fact is attributed to a title if EITHER
    (a) FULL   — the title's token sequence appears contiguously in the
                 fact (stem-insensitive), or
    (b) HEAD   — the title's head noun (last token, or the token before
                 " of " for "X of Y" titles) appears anywhere in the fact.
  Rule (b) is deliberately LOOSE. A loose rule attributes MORE facts,
  which produces FEWER zero-fact titles, which makes the gate HARDER to
  pass. The bias therefore runs against the hypothesis under test.

  Sensitivity: the strict rule (a) alone is also reported. If the two
  disagree about the gate, BOTH are reported and the declared rule
  (a OR b) governs.

Known limitation, reported not hidden: facts that refer to an entity
without naming it ("The mission lasted...") are attributed to nothing.
The unattributed-fact rate is printed so this is visible.
"""
import sys, os, json, re, string

REPO = os.environ.get("URBENCH_REPO", "/mnt/home/user41/URBench")
EXTRACTIONS = os.path.join(REPO, "outputs/efbpt/d4/d4_extractions.jsonl")
ARMX1 = os.path.join(REPO, "outputs/efbpt/d4/d4_armX1.jsonl")
GATE = 0.30

PUNCT = str.maketrans({c: " " for c in string.punctuation})
PAREN = re.compile(r"\s*\([^)]*\)\s*")
LEAD_ART = re.compile(r"^(the|a|an)\s+")
# tokens too generic to serve as a head noun on their own
WEAK_HEAD = {"series", "season", "film", "movie", "novel", "book", "show",
             "war", "type", "list", "company", "group", "band", "album"}


def toks(s):
    return [w for w in str(s).lower().translate(PUNCT).split() if w]


def stems(w):
    """Candidate stem set. Matching = non-empty intersection."""
    out = {w}
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        out.add(w[:-1])
    if len(w) > 4 and w.endswith("es"):
        out.add(w[:-2])
    if len(w) > 4 and w.endswith("ies"):
        out.add(w[:-3] + "y")
    return out


def eq(a, b):
    return bool(stems(a) & stems(b))


def title_tokens(title):
    t = PAREN.sub(" ", str(title).replace("_", " "))
    t = LEAD_ART.sub("", t.strip().lower())
    return toks(t)


def head_noun(tt):
    """Head of an English noun phrase: token before ' of ', else last token.
    Returns None when the only candidate is too generic to be safe."""
    if not tt:
        return None
    if "of" in tt:
        i = tt.index("of")
        if i > 0:
            cand = tt[i - 1]
            return cand if cand not in WEAK_HEAD else None
    cand = tt[-1]
    if cand in WEAK_HEAD and len(tt) > 1:
        cand = tt[-2]
    return cand if cand not in WEAK_HEAD else None


def contains_seq(ftoks, seq):
    n = len(seq)
    if n == 0 or n > len(ftoks):
        return False
    for i in range(len(ftoks) - n + 1):
        if all(eq(ftoks[i + j], seq[j]) for j in range(n)):
            return True
    return False


def attribute(fact, titles):
    """-> (set of titles matched by FULL, set matched by FULL-or-HEAD)"""
    ft = toks(fact)
    full, decl = set(), set()
    for t in titles:
        tt = title_tokens(t)
        if contains_seq(ft, tt):
            full.add(t)
            decl.add(t)
            continue
        h = head_noun(tt)
        if h and any(eq(w, h) for w in ft):
            decl.add(t)
    return full, decl


def run(rows, x1, verbose=True):
    multi = [r for r in rows if len(r["titles"]) >= 2]
    stats = {"full": 0, "decl": 0}
    zero_slots = {"full": 0, "decl": 0}
    total_slots = sum(len(r["titles"]) for r in multi)
    unattributed = 0
    total_facts = 0
    detail = []

    for r in rows:
        titles = r["titles"]
        cnt = {mode: {t: 0 for t in titles} for mode in ("full", "decl")}
        for fact in r["facts"]:
            total_facts += 1
            full, decl = attribute(fact, titles)
            if not decl:
                unattributed += 1
            for t in full:
                cnt["full"][t] += 1
            for t in decl:
                cnt["decl"][t] += 1
        if len(titles) >= 2:
            for mode in ("full", "decl"):
                z = [t for t in titles if cnt[mode][t] == 0]
                zero_slots[mode] += len(z)
                if z:
                    stats[mode] += 1
            detail.append((r["qid"], titles,
                           [cnt["decl"][t] for t in titles],
                           [t for t in titles if cnt["decl"][t] == 0],
                           x1.get(r["qid"], {}).get("pred", "?"),
                           x1.get(r["qid"], {}).get("gold", "?")))

    if verbose:
        print("rows total                    : %d" % len(rows))
        print("rows with >= 2 titles         : %d  (denominator)" % len(multi))
        print("title-slots in those rows     : %d" % total_slots)
        print("facts total                   : %d" % total_facts)
        print("facts attributed to NO title  : %d (%.1f%%)"
              % (unattributed, 100.0 * unattributed / max(total_facts, 1)))
        print()
        print("%-46s %6s %6s" % ("", "DECL", "STRICT"))
        print("%-46s %6s %6s" % ("(rule)", "a OR b", "a only"))
        for label, key in (("rows with >=1 zero-fact title", None),):
            pass
        d = 100.0 * stats["decl"] / max(len(multi), 1)
        f = 100.0 * stats["full"] / max(len(multi), 1)
        print("%-46s %5d  %5d" % ("rows with >=1 zero-fact title",
                                  stats["decl"], stats["full"]))
        print("%-46s %5.1f%% %5.1f%%" % ("  as % of multi-title rows", d, f))
        print("%-46s %5d  %5d" % ("zero-fact title-slots",
                                  zero_slots["decl"], zero_slots["full"]))
        print("%-46s %5.1f%% %5.1f%%"
              % ("  as % of title-slots",
                 100.0 * zero_slots["decl"] / max(total_slots, 1),
                 100.0 * zero_slots["full"] / max(total_slots, 1)))
        print()
        print("=" * 78)
        print("PER-ROW DETAIL (declared rule; * = row X1 answered wrong)")
        print("=" * 78)
        for qid, titles, counts, zeros, pred, gold in detail:
            mark = "*" if pred != gold else " "
            print("%s %s  %s" % (mark, qid,
                                 " | ".join("%s=%d" % (t, c)
                                            for t, c in zip(titles, counts))))
        print()
        print("=" * 78)
        gd = "RUNS" if d >= GATE * 100 else "CANCELLED"
        gf = "RUNS" if f >= GATE * 100 else "CANCELLED"
        print("DECLARED GATE  (rule a OR b, threshold %.0f%%): %.1f%% -> D5 %s"
              % (GATE * 100, d, gd))
        print("SENSITIVITY    (rule a only,  threshold %.0f%%): %.1f%% -> D5 %s"
              % (GATE * 100, f, gf))
        if gd != gf:
            print("NOTE: rules DISAGREE. Declared rule governs. Report both.")
        print("=" * 78)
    return stats, len(multi)


def selftest():
    """Validate the attribution rule against rows I read by hand in job 62521.
    Expected values were determined by human reading BEFORE this script ran."""
    cases = [
        # (titles, facts, titles expected to end with ZERO facts)
        (["Apollo 15", "Unicycle"],
         ["The Apollo 15 crew included Commander David Scott.",
          "Apollo 15 was the ninth crewed mission.",
          "The mission lasted from July 26 to August 7, 1971."],
         ["Unicycle"]),
        (["Fairy", "Fairyland", "Valkyrie"],
         ["Fairies are often described as metaphysical beings.",
          "Valkyries are female figures in Norse mythology.",
          "Valkyries are associated with the god Odin."],
         ["Fairyland"]),
        (["Abyssal plain", "Goat"],
         ["Abyssal plains are found on the deep ocean floor.",
          "Goats are domesticated species kept as livestock."],
         []),
        (["Spinal cord", "The Home Depot"],
         ["The spinal cord is a long, thin, tubular structure.",
          "The Home Depot is a home improvement retail corporation."],
         []),
        (["LendingTree", "Payday loan", "Retail"],
         ["SnapCap was acquired by LendingTree in 2017.",
          "LendingTree is an online lending marketplace."],
         ["Payday loan", "Retail"]),
        (["Audi R8", "Audi S8", "Sound barrier"],
         ["The Audi R8 is a mid-engine, 2-seater sports car.",
          "The Audi R8 uses quattro all-wheel drive."],
         ["Audi S8", "Sound barrier"]),
        (["The Godfather", "USB flash drive"],
         ["The Godfather is a 1972 American epic crime film.",
          "USB flash drives were first introduced in late 2000."],
         []),
    ]
    ok = True
    for titles, facts, expect_zero in cases:
        cnt = {t: 0 for t in titles}
        for f in facts:
            _, decl = attribute(f, titles)
            for t in decl:
                cnt[t] += 1
        got = sorted(t for t in titles if cnt[t] == 0)
        exp = sorted(expect_zero)
        flag = "OK " if got == exp else "FAIL"
        if got != exp:
            ok = False
        print("%s %-52s got=%s expected=%s"
              % (flag, "/".join(titles)[:52], got, exp))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    for p in (EXTRACTIONS, ARMX1):
        if not os.path.exists(p):
            sys.exit("MISSING: %s" % p)
    rows = [json.loads(l) for l in open(EXTRACTIONS, encoding="utf-8")]
    x1 = {}
    for l in open(ARMX1, encoding="utf-8"):
        r = json.loads(l)
        x1[r["qid"]] = r
    print("D5 STEP 0 — existence gate")
    print("source: %s (%d rows)" % (EXTRACTIONS, len(rows)))
    print()
    print("--- selftest against hand-read rows ---")
    if not selftest():
        sys.exit("SELFTEST FAILED — attribution rule is unreliable, stopping.")
    print()
    run(rows, x1)
    print("\n=== done. no files written. ===")


if __name__ == "__main__":
    main()
