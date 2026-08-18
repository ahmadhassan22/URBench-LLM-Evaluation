#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l0_part_a_census.py  —  L0 PART A: span census.

Declared in docs/EFBPT_PLAN_A_FREEZE.md -> EXPERIMENT L0, section D.
CPU only. READ-ONLY: writes no files, prints to stdout only.

Classifies every gold (urdu_span, canonical_title) pair into exactly one of:
  COREF     span refers to an entity named elsewhere ("his grandfather").
            Rule: translit similarity < 0.50 AND the span contains a
            possessive/demonstrative marker token. EXCLUDED from the gate.
  TRANSLIT  span is a phonetic rendering of the title.
            Rule: translit similarity >= 0.70.
  SEMANTIC  everything else.
LINKABLE = TRANSLIT + SEMANTIC.

Known limitation, declared before running: on a 23-pair hand-check the 0.70
threshold gave 100% precision and 77% recall. Missed transliterations
(Hades 0.67, CenturyLink 0.62, Broadway theatre 0.50) fall into SEMANTIC.
That is the conservative direction. The threshold is NOT tuned on the
evaluation pairs.
"""
import difflib, json, os, re, sys, unicodedata
from collections import Counter

GOLD = [("data/strategyqa_official/efbpt/plan_a_gold_100.jsonl", "qid", "entities"),
        ("data/strategyqa_official/efbpt/blind30_gold.jsonl", "urbench_qid",
         "question_entities")]
TRANSLIT_MIN = 0.70      # freeze D
# L0 AMENDMENT 1: automatic COREF detection removed. String similarity
# cannot separate a translation from a reference. COREF is now an explicit
# hand-confirmed list, adjudicated by a native Urdu speaker.
COREF_PAIRS = {
    ("اس کے دادا", "Genghis Khan"),
    ("نسان کے سی ای او", "Carlos Ghosn"),
}

URDU2LAT = {
 'آ':'a','ا':'a','ب':'b','پ':'p','ت':'t','ٹ':'t','ث':'s','ج':'j','چ':'ch',
 'ح':'h','خ':'kh','د':'d','ڈ':'d','ذ':'z','ر':'r','ڑ':'r','ز':'z','ژ':'zh',
 'س':'s','ش':'sh','ص':'s','ض':'z','ط':'t','ظ':'z','ع':'a','غ':'g','ف':'f',
 'ق':'k','ک':'k','گ':'g','ل':'l','م':'m','ن':'n','ں':'n','و':'o','ہ':'h',
 'ھ':'h','ء':'','ی':'i','ي':'i','ے':'e','ئ':'i','أ':'a','ۃ':'h','ة':'h',
}
DIAC = re.compile(r'[\u064B-\u0652\u0670\u0640]')
VOWELS = set('aeiou')


def translit(s):
    return ''.join(URDU2LAT.get(c, ' ' if c.isspace() else '')
                   for c in DIAC.sub('', s))


def latin_norm(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'\([^)]*\)', ' ', s)
    return re.sub(r'[^a-z ]', ' ', s.lower()).strip()


def skel(s):
    return ''.join(c for c in s if c not in VOWELS)


def sim(span, title):
    a, b = translit(span).replace(' ', ''), latin_norm(title).replace(' ', '')
    if not a or not b:
        return 0.0
    return max(difflib.SequenceMatcher(None, a, b).ratio(),
               difflib.SequenceMatcher(None, skel(a), skel(b)).ratio())


def classify(span, title):
    s = sim(span, title)
    if (span.strip(), title.strip()) in COREF_PAIRS:
        return "COREF", s
    if s >= TRANSLIT_MIN:
        return "TRANSLIT", s
    return "SEMANTIC", s


def main():
    pairs, seen = [], set()
    for path, idf, entf in GOLD:
        if not os.path.exists(path):
            sys.exit("MISSING: " + path)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for e in row.get(entf, []):
                sp, ti = e.get("urdu_span", ""), e.get("canonical_title", "")
                key = (row[idf], sp)
                if not ti.strip() or key in seen:
                    continue
                seen.add(key)
                b, s = classify(sp, ti)
                pairs.append({"qid": row[idf], "span": sp, "title": ti,
                              "bucket": b, "sim": s, "src": os.path.basename(path)})

    ct = Counter(p["bucket"] for p in pairs)
    n = len(pairs)
    link = ct["TRANSLIT"] + ct["SEMANTIC"]
    print("=" * 78)
    print("L0 PART A — SPAN CENSUS  (freeze section D)")
    print("=" * 78)
    print("  rows read            : %d" % len({p["qid"] for p in pairs}))
    print("  pairs after dedupe   : %d" % n)
    print("  distinct titles      : %d" % len({p["title"] for p in pairs}))
    print()
    for b in ("TRANSLIT", "SEMANTIC", "COREF"):
        print("  %-9s %4d  (%5.1f%%)" % (b, ct[b], 100.0 * ct[b] / max(n, 1)))
    print("  %-9s %4d  (%5.1f%%)  <- denominator for the gate"
          % ("LINKABLE", link, 100.0 * link / max(n, 1)))
    print()
    print("  POWER at n=%d (freeze G): smallest detectable gain" % link)
    for cand, gain in ((289, 3), (215, 4), (180, 5), (130, 7), (65, 14)):
        if link >= cand:
            print("    n=%d >= %d  ->  about %+dpp detectable" % (link, cand, gain))
            break
    else:
        print("    n=%d is BELOW 65. Power is insufficient; report this before"
              " any further step." % link)

    print("\n" + "-" * 78)
    print("FIRST 30 CLASSIFICATIONS — human check required (freeze D)")
    print("-" * 78)
    print("%-9s %6s  %-28s %-30s" % ("bucket", "sim", "urdu span (translit)", "gold title"))
    for p in pairs[:30]:
        print("%-9s %6.2f  %-28s %-30s"
              % (p["bucket"], p["sim"], translit(p["span"])[:28], p["title"][:30]))

    cor = [p for p in pairs if p["bucket"] == "COREF"]
    print("\n" + "-" * 78)
    print("ALL %d COREF CANDIDATES — confirm each is genuinely a reference"
          % len(cor))
    print("-" * 78)
    for p in cor:
        print("  %-24s -> %-28s  (sim %.2f, %s)"
              % (p["span"][:24], p["title"][:28], p["sim"], p["qid"][:8]))
    if not cor:
        print("  none detected")

    print("\n" + "-" * 78)
    print("BORDERLINE: 10 pairs closest to the %.2f TRANSLIT threshold"
          % TRANSLIT_MIN)
    print("-" * 78)
    for p in sorted(pairs, key=lambda x: abs(x["sim"] - TRANSLIT_MIN))[:10]:
        print("  %-9s %.2f  %-26s -> %s"
              % (p["bucket"], p["sim"], translit(p["span"])[:26], p["title"][:28]))

    print("\n=== done. no files written. ===")


if __name__ == "__main__":
    main()
