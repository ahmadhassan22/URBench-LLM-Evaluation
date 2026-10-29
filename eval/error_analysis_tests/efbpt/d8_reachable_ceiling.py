#!/usr/bin/env python3
"""
d8_reachable_ceiling.py

D8 — Reachable-page ceiling and English self-knowledge.
Contract: docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md (FROZEN 2026-10-27) as
amended by docs/EFBPT_D8_AMENDMENT_1.md (pre-run, 2026-10-27) and
docs/EFBPT_D8_AMENDMENT_2.md (pre-result, 2026-10-29: a verbatim gold-fact
substring in a pass-2 prompt of Xp/GEN/X1u is counted and reported as V6,
not fatal; V2b compares A_r with the archived A_r of failed job 107156).
Implements the freeze exactly; nothing here is tunable after numbers exist.

Arms (freeze §5, Amendment 1 A1), all paired with A_r on the same qids:
  A_r   200   no facts; D1 arm A user layout, re-run in this job (control)
  Xp    200   facts extracted from the gold pages that are traced EXACT
              AND present under the §3 title rule (PARTIAL dropped, A1);
              a qid with 0 such pages gets no pass 1 and the A_r layout
  GEN   200   facts the model writes from its own knowledge (English)
  X1u    68   facts extracted from ALL gold pages of D4-68 (question_en
              removed from the extraction prompt; the honest D4 number)

Everything frozen is IMPORTED, never re-implemented:
  prompt constants, extractor, Urdu instruction  <- efbpt_eval_dev200.py
  user-message layouts (arm A / facts block)     <- d1_eval_arms.py
  Devanagari secondary scorer                    <- d1_score_dual.py
  norm(), gold titles, score(), corpus path      <- d3_oracle_retrieval.py
  extraction prompt, fact cleaning, generation   <- d4_extract_facts.py
  paired exact McNemar                           <- d5_balanced_extract.py

Modes:
  --preflight-only      CPU, no model. Title map + every count assertion of
                        freeze §3 and Amendment 1 A2. Writes NOTHING.
  --limit N --dry-run   TEST (repo rule: --limit requires --dry-run). Runs
                        the pipeline on the first N rows and writes ONLY
                        *_TEST files under outputs/efbpt/d8/. Never a result.
  (no flags)            FULL run, canonical files.

Outputs (freeze §9), only under outputs/efbpt/d8/:
  d8_START.json, title_map.jsonl, d8_extractions_{Xp,GEN,X1u}.jsonl,
  d8_arm{A_r,Xp,GEN,X1u}.jsonl, d8_summary.json, d8_SEAL.json

Usage (from ~/URBench):
  python eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py --preflight-only
  python eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py --limit 5 --dry-run
  python eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py
"""

import argparse
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
import unicodedata
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from efbpt_eval_dev200 import (          # noqa: E402
    MODEL_PATH, DEV_PATH, SYSTEM_MESSAGE, MAX_NEW_TOKENS,
    INSTRUCTION_PATH, INSTRUCTION_MD5,
    extract_answer, load_instruction, die,
)
from d1_eval_arms import build_user       # noqa: E402  arm A / arm B layouts
from d1_score_dual import extract_answer_secondary   # noqa: E402
from d3_oracle_retrieval import (         # noqa: E402
    norm, required_titles, score, META_PATH, TITLE_RE,
)
from d4_extract_facts import (            # noqa: E402
    load_rows, clean_fact_lines, generate_all,
    EXTRACT_SYSTEM, EXTRACT_INSTRUCTION, CHUNKS_PER_TITLE, EXTRACT_MAX_NEW,
)
from d5_balanced_extract import mcnemar   # noqa: E402  paired exact, two-sided

# ----------------------------------------------------------------------------
# Frozen constants (freeze §1, §2, §5, §6, §7, §9; Amendment 1 A1-A3)
# ----------------------------------------------------------------------------

FREEZE_PATH = "docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md"
AMENDMENT_PATH = ["docs/EFBPT_D8_AMENDMENT_1.md",     # pre-run, 2026-10-27
                  "docs/EFBPT_D8_AMENDMENT_2.md"]     # pre-result, 2026-10-29
ARCHIVED_AR_PATH = "outputs/efbpt/d8_job107156_failed/d8_armA_r.jsonl"  # read-only (A2.3)
TRACE_PATH = "data/strategyqa_official/efbpt/stage0/lexical_trace_instances.jsonl"
TRACE_MD5 = "5b361d60b33a64ab5c8935b40a3550e1"
TRACE_COUNTS = OrderedDict([("total", 636), ("EXACT", 256),
                            ("PARTIAL", 46), ("ABSENT", 334)])
KNOWN_CLASSES = ("EXACT", "PARTIAL", "ABSENT")
TRACED_CLASSES = ("EXACT",)            # Amendment 1 A1: PARTIAL dropped from Xp

D4_X1_PATH = "outputs/efbpt/d4/d4_armX1.jsonl"          # read-only
D1_ARMA_PATH = "outputs/efbpt/d1/arms/d1_armA.jsonl"    # read-only
D4_EXCLUDED = ("1d923b8a2778448a9716", "8eecf2ff459ef5a9b98a",
               "f7a32d14efd589015fe1")                  # freeze §1
D4_71 = 71
D4_68 = 68

OUT_DIR = "outputs/efbpt/d8"
FORBIDDEN_WRITE_DIRS = ("outputs/efbpt/d1", "outputs/efbpt/d4", "outputs/efbpt/d6")

# pre-run counts, Amendment 1 A2 (EXACT only; asserted in every mode)
EXPECTED_TRACED_INSTANCES = 256
EXPECTED_TRACED_PRESENT = 156
EXPECTED_QIDS_WITH_PAGE = 124
EXPECTED_PAGES_HIST = {0: 76, 1: 96, 2: 24, 3: 4}

# GEN pass-1 system message, Amendment 1 A3 (Xp/X1u keep EXTRACT_SYSTEM)
GEN_SYSTEM = (
    "You are a careful research assistant. You write accurate English facts "
    "from your own knowledge. You never state a fact you are not confident "
    "about."
)

# GEN pass-1 instruction, frozen verbatim in §5
GEN_INSTRUCTION = (
    "Write the facts needed to answer the question below.\n"
    "Rules:\n"
    "1. Output ONLY short English factual sentences, one per line.\n"
    "2. No numbering, no bullets, no explanation, no answer to the question "
    "itself. Facts only.\n"
    "3. At most 6 facts."
)
GEN_MAX_FACTS = 6
GEN_MAX_NEW = EXTRACT_MAX_NEW          # 512, §5

CONTEXT_WINDOW = 40960                 # same check as D4

# validity (§6) and readings (§7)
V1_SPREAD_LIMIT = 5.0
V2_ACC_TOL = 2.0
V2_AGREE_MIN = 95.0
V3_LATIN_MIN = 0.90
V5_SAMPLE = 30
V5_SEED = 8
R1_MIN_CORRECT = 49
GATE_PP = 8.0
COND_PP = 5.0
R3_HARM_PP = -3.0
ALPHA = 0.05

ARMS = ("A_r", "Xp", "GEN", "X1u")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_file(path):
    with open(path, "rb") as f:
        return md5_bytes(f.read())


def md5_text(s):
    return md5_bytes(s.encode("utf-8"))


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def guard_out_path(path):
    """Every write goes through here. Only outputs/efbpt/d8/ is writable;
    outputs/efbpt/d1, d4, d6 are never opened for writing (freeze §9)."""
    ap = os.path.abspath(path)
    root = os.path.abspath(OUT_DIR)
    for forb in FORBIDDEN_WRITE_DIRS:
        fr = os.path.abspath(forb)
        if ap == fr or ap.startswith(fr + os.sep):
            die("refusing to write under %s: %s" % (forb, path))
    if not ap.startswith(root + os.sep):
        die("refusing to write outside %s: %s" % (OUT_DIR, path))
    return ap


WRITTEN = []    # paths written by this run, sealed at the end


def write_jsonl(path, records, expect_n=None):
    ap = guard_out_path(path)
    if expect_n is not None and len(records) != expect_n:
        die("%s: have %d records, expected %d" % (path, len(records), expect_n))
    with open(ap, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    nd = sum(1 for _ in open(ap, "r", encoding="utf-8"))
    if nd != len(records):
        die("%s: wrote %d records, disk has %d lines" % (path, len(records), nd))
    print("  [verified] %d lines -> %s" % (nd, path), flush=True)
    WRITTEN.append(path)


def write_json(path, obj):
    ap = guard_out_path(path)
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    with open(ap, "r", encoding="utf-8") as f:
        json.load(f)          # must read back as valid JSON
    print("  [verified] wrote %s" % path, flush=True)
    WRITTEN.append(path)


def log_full_prompt(label, prompt):
    """Requirement 10: the FULL rendered prompt, not a prefix."""
    print("\n----- BEGIN FULL PROMPT [%s] (%d chars) -----" % (label, len(prompt)))
    print(prompt)
    print("----- END FULL PROMPT [%s] -----" % label)
    print("  tail repr (thinking tag): %r" % prompt[-80:], flush=True)


def pkg_versions():
    import importlib.metadata as im
    v = OrderedDict()
    v["python"] = sys.version.split()[0]
    for name in ("torch", "transformers", "bitsandbytes", "accelerate",
                 "scipy", "peft", "tokenizers"):
        try:
            v[name] = im.version(name)
        except Exception:
            v[name] = None
    return v


# ----------------------------------------------------------------------------
# Corpus streaming (title lines only; never a full load)
# ----------------------------------------------------------------------------

def _decode_title(group):
    """Exact title string of a corpus line. Fast path when no escape is
    present; otherwise the JSON decoder, so every escape is exact."""
    if b"\\" in group:
        return json.loads(b'"' + group + b'"')
    return group.decode("utf-8")


def iter_corpus_titles(progress_every=4_000_000):
    """Yields (raw_line_bytes, exact_title) for every corpus line."""
    n = 0
    t0 = time.time()
    with open(META_PATH, "rb") as f:
        for raw in f:
            n += 1
            m = TITLE_RE.match(raw)
            if not m:
                continue
            yield raw, _decode_title(m.group(1))
            if n % progress_every == 0:
                print("[scan]   %d lines (%.0fs)" % (n, time.time() - t0),
                      flush=True)
    print("[scan] read %d lines in %.0fs" % (n, time.time() - t0), flush=True)


def build_title_map(gold_titles):
    """Freeze §3. One streaming pass. Returns {gold_title: decision}.
    present iff (i) a corpus title equals T exactly, else (ii) exactly one
    corpus title equals T after norm() and it is not all upper-case."""
    exact = set(gold_titles)
    keys = {}
    for T in gold_titles:
        keys.setdefault(norm(T), []).append(T)
    exact_found = set()
    cands = {k: set() for k in keys}
    print("[scan] title map: %d distinct gold titles, %d norm keys"
          % (len(exact), len(keys)), flush=True)
    for _raw, t in iter_corpus_titles():
        if t in exact:
            exact_found.add(t)
        nt = norm(t)
        if nt in cands:
            cands[nt].add(t)
    decisions = OrderedDict()
    for T in gold_titles:
        c = sorted(cands[norm(T)])
        if T in exact_found:
            corpus, rule, present = T, "exact", True
        elif len(c) == 1 and not c[0].isupper():
            corpus, rule, present = c[0], "norm_unique", True
        elif len(c) == 1:
            corpus, rule, present = None, "absent_norm_unique_all_caps", False
        elif len(c) == 0:
            corpus, rule, present = None, "absent_no_match", False
        else:
            corpus, rule, present = None, "absent_norm_ambiguous", False
        decisions[T] = OrderedDict([
            ("gold_title", T), ("corpus_title", corpus), ("present", present),
            ("rule", rule), ("norm_candidates", c),
        ])
    return decisions


def fetch_chunks_exact(titles, max_per_title):
    """One streaming pass. Chunks keyed by the EXACT corpus title (not
    norm()), file order preserved, at most max_per_title each."""
    got = {t: [] for t in titles}
    remaining = set(titles)
    print("[scan] fetching up to %d chunks for %d exact titles"
          % (max_per_title, len(titles)), flush=True)
    for raw, t in iter_corpus_titles():
        if t in remaining:
            got[t].append(json.loads(raw.decode("utf-8"))["text"])
            if len(got[t]) >= max_per_title:
                remaining.discard(t)
                if not remaining:
                    break
    return got


# ----------------------------------------------------------------------------
# Leak guard (requirement 6, freeze §5 "forbidden in any prompt")
# ----------------------------------------------------------------------------

def load_leak_needles():
    rows = read_jsonl(DEV_PATH)
    q_en = sorted({r["question_en"] for r in rows})
    facts = sorted({f.strip() for r in rows for f in r.get("urbench_facts", [])
                    if f and f.strip()})
    ev_ids = sorted({p for r in rows for p in r.get("evidence_paragraph_ids", [])
                     if isinstance(p, str) and p})
    # an evidence id is "<Title>-<n>"; matched as a standalone token so that
    # "7-1" does not fire inside a year range such as 1997-1998
    ev_re = re.compile(r"(?<![0-9A-Za-z])(?:%s)(?![0-9])"
                       % "|".join(re.escape(e) for e in ev_ids))
    return {"question_en": q_en, "facts": facts, "evidence_ids": ev_ids,
            "evidence_re": ev_re}


def leak_guard(label, prompts, needles, fatal_gold_facts=True):
    """question_en and evidence-id matches are always fatal. A gold-fact
    substring is fatal when fatal_gold_facts is True (every pass-1 prompt,
    A_r pass 2); otherwise (Amendment 2 A2.2: Xp/GEN/X1u pass 2, where the
    model's own pass-1 extraction may restate a page fact verbatim) it is
    collected and returned as a list of (prompt index, matched fact)."""
    hits = []
    for i, p in enumerate(prompts):
        for s in needles["question_en"]:
            if s in p:
                die("LEAK in %s prompt %d: contains question_en %r" % (label, i, s))
        for s in needles["facts"]:
            if s in p:
                if fatal_gold_facts:
                    die("LEAK in %s prompt %d: contains gold fact %r" % (label, i, s))
                hits.append((i, s))
        m = needles["evidence_re"].search(p)
        if m:
            die("LEAK in %s prompt %d: contains evidence id %r" % (label, i, m.group(0)))
    if hits:
        print("[leak-guard] %-9s %4d prompts: question_en and evidence ids clean; "
              "%d verbatim gold-fact substring(s) in %d prompt(s), reported as V6 "
              "(Amendment 2 A2.2, not fatal)"
              % (label, len(prompts), len(hits), len({i for i, _ in hits})), flush=True)
    else:
        print("[leak-guard] %-9s %4d prompts clean (%d question_en, %d gold facts, "
              "%d evidence ids checked)"
              % (label, len(prompts), len(needles["question_en"]),
                 len(needles["facts"]), len(needles["evidence_ids"])), flush=True)
    return hits


# ----------------------------------------------------------------------------
# GEN language check (requirement 7, freeze §6 V3)
# ----------------------------------------------------------------------------

def latin_fraction(texts):
    letters = latin = 0
    for t in texts:
        for ch in t:
            if ch.isalpha():
                letters += 1
                if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or \
                        unicodedata.name(ch, "").startswith("LATIN"):
                    latin += 1
    return (latin / letters if letters else 0.0), letters, latin


# ----------------------------------------------------------------------------
# Preflight (CPU, no model): every assertion of freeze §3 / Amendment 1 A2
# ----------------------------------------------------------------------------

def preflight():
    """Returns (instruction, rows, d4_68 qids, trace instances, decisions,
    xp_pages, d1A rows, meta) after every count assertion passed."""
    print("=" * 78)
    print("PREFLIGHT (freeze §3, Amendment 1 A2)")
    print("=" * 78, flush=True)

    if not os.path.exists(FREEZE_PATH):
        die("freeze file missing: " + FREEZE_PATH)
    for ap in AMENDMENT_PATH:
        if not os.path.exists(ap):
            die("amendment file missing: " + ap)
    freeze_md5 = md5_file(FREEZE_PATH)
    amendment_md5s = [md5_file(ap) for ap in AMENDMENT_PATH]
    script_md5 = md5_file(os.path.abspath(__file__))
    print("[md5] freeze    %s  %s" % (freeze_md5, FREEZE_PATH))
    for ap, am in zip(AMENDMENT_PATH, amendment_md5s):
        print("[md5] amendment %s  %s" % (am, ap))
    print("[md5] script    %s  %s" % (script_md5, os.path.relpath(__file__)))

    instruction = load_instruction()             # MD5 asserted inside
    print("[ok] Urdu instruction MD5 %s verified" % INSTRUCTION_MD5)

    rows = load_rows()                           # asserts 200 rows
    qids = [r["qid"] for r in rows]
    if len(set(qids)) != 200:
        die("DEV200 qids are not unique")
    print("[ok] DEV200: 200 rows, 200 unique qids")

    # ---- trace file ----
    got = md5_file(TRACE_PATH)
    if got != TRACE_MD5:
        die("trace MD5 mismatch: expected %s, got %s" % (TRACE_MD5, got))
    trace = read_jsonl(TRACE_PATH)
    cls = Counter(t["lexical_trace_class"] for t in trace)
    tc = OrderedDict([("total", len(trace)), ("EXACT", cls.get("EXACT", 0)),
                      ("PARTIAL", cls.get("PARTIAL", 0)),
                      ("ABSENT", cls.get("ABSENT", 0))])
    if tc != TRACE_COUNTS:
        die("trace counts %s != expected %s" % (dict(tc), dict(TRACE_COUNTS)))
    if set(cls) - set(KNOWN_CLASSES):
        die("unknown trace classes: %s" % sorted(cls))
    dev_pairs = {(r["qid"], t) for r in rows for t in r["required"]}
    tr_pairs = {(t["urbench_qid"], t["gold_title"]) for t in trace}
    if len(tr_pairs) != len(trace) or tr_pairs != dev_pairs:
        die("trace (qid, gold_title) instances do not equal DEV200 gold titles")
    print("[ok] trace MD5 %s; instances %d (EXACT %d, PARTIAL %d, ABSENT %d); "
          "equal to DEV200 (qid, gold_title) pairs"
          % (TRACE_MD5, tc["total"], tc["EXACT"], tc["PARTIAL"], tc["ABSENT"]))

    # ---- D4-68 ----
    d4x1 = read_jsonl(D4_X1_PATH)
    d4_qids = [r["qid"] for r in d4x1]
    if len(d4_qids) != D4_71 or len(set(d4_qids)) != D4_71:
        die("D4 armX1 has %d rows, expected %d unique" % (len(d4_qids), D4_71))
    if not set(D4_EXCLUDED) <= set(d4_qids):
        die("excluded D4 qids not all in D4-71")
    d4_68 = set(d4_qids) - set(D4_EXCLUDED)
    if len(d4_68) != D4_68 or not d4_68 <= set(qids):
        die("D4-68 has %d qids, expected %d within DEV200" % (len(d4_68), D4_68))
    print("[ok] D4-68: %d qids (D4-71 minus %d, freeze §1)" % (len(d4_68), len(D4_EXCLUDED)))

    # ---- D1 arm A (reference for V2) ----
    d1a = read_jsonl(D1_ARMA_PATH)
    if len(d1a) != 200 or {r["qid"] for r in d1a} != set(qids):
        die("D1 arm A does not cover DEV200 exactly")
    for r in d1a:
        if r.get("arm") != "A":
            die("D1 arm A file contains arm %r" % r.get("arm"))
        if extract_answer(r["generation"]) != r["pred"]:
            die("D1 arm A qid %s: stored pred != frozen extractor" % r["qid"])
    print("[ok] D1 arm A: 200 rows, preds reproduce the frozen extractor")

    # ---- title map (freeze §3), one streaming pass ----
    gold_titles = sorted({t for r in rows for t in r["required"]})
    decisions = build_title_map(gold_titles)
    rules = Counter(d["rule"] for d in decisions.values())
    print("[title-map] distinct gold titles %d: %s"
          % (len(decisions), dict(sorted(rules.items()))))
    n_present = sum(1 for d in decisions.values() if d["present"])
    print("[title-map] present %d / %d distinct titles" % (n_present, len(decisions)))
    for d in decisions.values():
        if d["rule"] == "norm_unique":
            print("    norm_unique: %r -> %r" % (d["gold_title"], d["corpus_title"]))
        elif d["rule"] in ("absent_norm_unique_all_caps", "absent_norm_ambiguous"):
            print("    %s: %r candidates %s" % (d["rule"], d["gold_title"],
                                               d["norm_candidates"]))

    # ---- EXACT-and-present counts (Amendment 1 A2) ----
    traced = [t for t in trace if t["lexical_trace_class"] in TRACED_CLASSES]
    if len(traced) != EXPECTED_TRACED_INSTANCES:
        die("EXACT instances %d != %d" % (len(traced), EXPECTED_TRACED_INSTANCES))
    tp = [t for t in traced if decisions[t["gold_title"]]["present"]]
    if len(tp) != EXPECTED_TRACED_PRESENT:
        die("EXACT-and-present %d != %d" % (len(tp), EXPECTED_TRACED_PRESENT))
    xp_pages = OrderedDict()
    tp_by_qid = {}
    for t in tp:
        tp_by_qid.setdefault(t["urbench_qid"], set()).add(t["gold_title"])
    for r in rows:
        # row["required"] is sorted, so page order is deterministic
        xp_pages[r["qid"]] = [t for t in r["required"]
                              if t in tp_by_qid.get(r["qid"], set())]
    hist = dict(Counter(len(v) for v in xp_pages.values()))
    n_with = sum(1 for v in xp_pages.values() if v)
    print("[title-map] EXACT instances %d, EXACT-and-present %d, qids with "
          ">=1 page %d, pages-per-qid histogram %s"
          % (len(traced), len(tp), n_with, dict(sorted(hist.items()))))
    if n_with != EXPECTED_QIDS_WITH_PAGE:
        die("qids with >=1 page %d != %d" % (n_with, EXPECTED_QIDS_WITH_PAGE))
    if hist != EXPECTED_PAGES_HIST:
        die("pages-per-qid histogram %s != %s" % (hist, EXPECTED_PAGES_HIST))

    # ---- all gold titles of D4-68 present ----
    bad = [(r["qid"], t) for r in rows if r["qid"] in d4_68
           for t in r["required"] if not decisions[t]["present"]]
    if bad:
        die("D4-68 gold titles absent under §3: %s" % bad[:5])
    print("[ok] every gold title of the D4-68 is present under §3")

    meta = OrderedDict([
        ("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", freeze_md5)])),
        ("amendments", [OrderedDict([("path", ap), ("md5", am)])
                        for ap, am in zip(AMENDMENT_PATH, amendment_md5s)]),
        ("script", OrderedDict([("path", os.path.relpath(__file__)),
                                ("md5", script_md5)])),
        ("pass1_instructions", OrderedDict([
            ("extract_system_md5", md5_text(EXTRACT_SYSTEM)),
            ("extract_instruction_md5", md5_text(EXTRACT_INSTRUCTION)),
            ("gen_system_md5", md5_text(GEN_SYSTEM)),
            ("gen_instruction_md5", md5_text(GEN_INSTRUCTION)),
        ])),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH),
                                          ("md5", INSTRUCTION_MD5)])),
        ("trace", OrderedDict([("path", TRACE_PATH), ("md5", TRACE_MD5),
                               ("counts", tc)])),
        ("title_map", OrderedDict([
            ("distinct_gold_titles", len(decisions)),
            ("present_distinct", n_present),
            ("rules", OrderedDict(sorted(rules.items()))),
            ("exact_instances", len(traced)),
            ("exact_and_present", len(tp)),
            ("qids_with_page", n_with),
            ("pages_hist", OrderedDict((str(k), v) for k, v in sorted(hist.items()))),
        ])),
    ])
    print("[preflight] PASS", flush=True)
    return instruction, rows, d4_68, trace, decisions, xp_pages, d1a, meta


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true",
                    help="CPU: title map + count assertions, write nothing, exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N DEV200 rows (TEST; requires --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="TEST: write only *_TEST files, never canonical ones")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.limit is not None and not args.dry_run:
        die("--limit requires --dry-run (a partial run never writes canonical files)")
    if args.limit is not None and args.limit <= 0:
        die("--limit must be positive")
    if args.preflight_only and (args.limit is not None or args.dry_run):
        die("--preflight-only takes no --limit/--dry-run")

    instruction, rows, d4_68, trace, decisions, xp_pages, d1a_all, meta = preflight()

    if args.preflight_only:
        print("\n[preflight-only] done. No files written.")
        return

    # ---------------- mode, subset, START ----------------
    suffix = "_TEST" if args.dry_run else ""
    is_test = args.dry_run
    sub = rows[:args.limit] if args.limit else rows
    sub_qids = [r["qid"] for r in sub]
    sub_set = set(sub_qids)
    x1u_rows = [r for r in sub if r["qid"] in d4_68]
    print("\n[mode] %s  rows=%d  X1u rows=%d  suffix=%r"
          % ("TEST — NOT A RESULT" if is_test else "FULL", len(sub),
             len(x1u_rows), suffix))
    if not is_test and (len(sub) != 200 or len(x1u_rows) != D4_68):
        die("FULL run must cover 200 rows and 68 X1u rows")

    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    start = OrderedDict([
        ("mode", "TEST" if is_test else "FULL"),
        ("args", vars(args)),
        ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("hostname", socket.gethostname()),
        ("slurm_job_id", job_id),
        ("n_rows", len(sub)), ("n_x1u_rows", len(x1u_rows)),
        ("model_path", MODEL_PATH),
        ("decoding", OrderedDict([
            ("batch_size", args.batch_size), ("do_sample", False),
            ("enable_thinking", False), ("quant", "nf4 double-quant bf16 compute"),
            ("extract_max_new", EXTRACT_MAX_NEW), ("gen_max_new", GEN_MAX_NEW),
            ("answer_max_new", MAX_NEW_TOKENS),
            ("chunks_per_title", CHUNKS_PER_TITLE), ("gen_max_facts", GEN_MAX_FACTS),
        ])),
        ("versions", pkg_versions()),
    ])
    start.update(meta)
    print("\n[START]")
    write_json(os.path.join(OUT_DIR, "d8_START%s.json" % suffix), start)

    # title map: one line per trace instance (636), decision attached
    tm_rows = []
    for t in trace:
        d = decisions[t["gold_title"]]
        tm_rows.append(OrderedDict([
            ("urbench_qid", t["urbench_qid"]), ("gold_title", t["gold_title"]),
            ("lexical_trace_class", t["lexical_trace_class"]),
            ("present", d["present"]), ("corpus_title", d["corpus_title"]),
            ("rule", d["rule"]), ("norm_candidates", d["norm_candidates"]),
            ("used_by_Xp", t["lexical_trace_class"] in TRACED_CLASSES and d["present"]),
            ("in_D4_68", t["urbench_qid"] in d4_68),
        ]))
    write_jsonl(os.path.join(OUT_DIR, "title_map%s.jsonl" % suffix), tm_rows,
                expect_n=TRACE_COUNTS["total"])

    # ---------------- pages and chunks (exact corpus titles) ----------------
    def corpus_titles(gold_list):
        return [decisions[t]["corpus_title"] for t in gold_list]

    xp_ct = OrderedDict((r["qid"], corpus_titles(xp_pages[r["qid"]])) for r in sub)
    x1u_ct = OrderedDict((r["qid"], corpus_titles(r["required"])) for r in x1u_rows)
    needed = sorted({t for v in xp_ct.values() for t in v}
                    | {t for v in x1u_ct.values() for t in v})
    if any(t is None for t in needed):
        die("an absent title reached the chunk fetch")
    chunks = fetch_chunks_exact(needed, CHUNKS_PER_TITLE)
    missing = [t for t in needed if not chunks[t]]
    if missing:
        die("present titles returned no chunk: %s" % missing[:5])
    lens = sorted(len(chunks[t]) for t in needed)
    print("[scan] %d titles fetched; chunks per title min=%d median=%d max=%d"
          % (len(needed), lens[0], lens[len(lens) // 2], lens[-1]))

    # ---------------- tokenizer, prompts, leak guard ----------------
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if im is not None and im >= 0:
        eos_ids.add(im)

    def chat(system, user):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    def extract_prompt(row, titles):
        # D4 pass-1 layout with question_ur in place of question_en (§5)
        passages = ["[%s] %s" % (ct, c) for ct in titles
                    for c in chunks[ct][:CHUNKS_PER_TITLE]]
        user = (EXTRACT_INSTRUCTION + "\n\nQuestion: " + row["question_ur"]
                + "\n\nPassages:\n" + "\n\n".join(passages))
        return chat(EXTRACT_SYSTEM, user)

    def gen_prompt(row):
        # Amendment 1 A3: GEN has no passages, so it gets GEN_SYSTEM
        return chat(GEN_SYSTEM, GEN_INSTRUCTION + "\n\nQuestion: " + row["question_ur"])

    def answer_prompt(row, facts):
        # facts -> D1 arm B layout (= D4 pass 2); none -> D1 arm A layout
        if facts:
            user = build_user(instruction, dict(row, facts=list(facts)), None, "B")
        else:
            user = build_user(instruction, row, None, "A")
        return chat(SYSTEM_MESSAGE, user)

    needles = load_leak_needles()

    xp1_rows = [r for r in sub if xp_ct[r["qid"]]]
    p_ar = [answer_prompt(r, []) for r in sub]
    p_xp1 = [extract_prompt(r, xp_ct[r["qid"]]) for r in xp1_rows]
    p_gen1 = [gen_prompt(r) for r in sub]
    p_x1u1 = [extract_prompt(r, x1u_ct[r["qid"]]) for r in x1u_rows]

    for label, ps, mx in (("A_r/pass2", p_ar, MAX_NEW_TOKENS),
                          ("Xp/pass1", p_xp1, EXTRACT_MAX_NEW),
                          ("GEN/pass1", p_gen1, GEN_MAX_NEW),
                          ("X1u/pass1", p_x1u1, EXTRACT_MAX_NEW)):
        if not ps:
            print("[len] %-10s no prompts" % label)
            continue
        tl = sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in ps)
        print("[len] %-10s n=%d tokens min=%d median=%d max=%d"
              % (label, len(ps), tl[0], tl[len(tl) // 2], tl[-1]))
        if tl[-1] + mx > CONTEXT_WINDOW:
            die("%s prompts would exceed the context window" % label)

    # requirement 6: before ANY generation
    leak_guard("A_r/pass2", p_ar, needles)
    leak_guard("Xp/pass1", p_xp1, needles)
    leak_guard("GEN/pass1", p_gen1, needles)
    leak_guard("X1u/pass1", p_x1u1, needles)

    # ---------------- model (exactly as D4) ----------------
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("\n[load] base model, 4-bit nf4, no adapters ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    # ---------------- PASS 1 ----------------
    extractions = OrderedDict()

    def run_pass1(arm, prs, prompts, titles_by_qid, max_new, keep):
        print("\n" + "=" * 70)
        print("PASS 1 — %s  (n=%d, max_new=%d, thinking off, greedy)"
              % (arm, len(prompts), max_new))
        print("=" * 70, flush=True)
        if not prompts:
            print("  (no rows)")
            extractions[arm] = []
            return
        log_full_prompt("%s pass 1, row 0, qid %s" % (arm, prs[0]["qid"]), prompts[0])
        gens = generate_all(model, tok, prompts, args.batch_size, eos_ids, max_new)
        out = []
        n_empty = 0
        for r, (text, n_new, trunc) in zip(prs, gens):
            all_facts = clean_fact_lines(text)
            facts = all_facts[:keep] if keep else all_facts
            if not facts:
                n_empty += 1
            rec = OrderedDict([
                ("qid", r["qid"]), ("arm", arm),
                ("titles_gold", [t for t in r["required"]
                                 if decisions[t]["corpus_title"] in titles_by_qid[r["qid"]]]
                 if titles_by_qid is not None else []),
                ("titles_corpus", titles_by_qid[r["qid"]] if titles_by_qid is not None else []),
                ("n_pages", len(titles_by_qid[r["qid"]]) if titles_by_qid is not None else 0),
                ("raw_extraction", text), ("facts", facts),
                ("n_facts", len(facts)), ("n_facts_raw", len(all_facts)),
                ("truncated", trunc), ("n_gen_tokens", n_new),
            ])
            out.append(rec)
        nf = sorted(e["n_facts"] for e in out)
        print("  facts per row: min=%d median=%d max=%d  empty=%d  truncated=%d"
              % (nf[0], nf[len(nf) // 2], nf[-1], n_empty,
                 sum(1 for e in out if e["truncated"])))
        print("  sample facts (row 0):")
        for f in out[0]["facts"][:6]:
            print("    - %s" % f)
        extractions[arm] = out
        write_jsonl(os.path.join(OUT_DIR, "d8_extractions_%s%s.jsonl" % (arm, suffix)),
                    out, expect_n=len(prs))

    run_pass1("Xp", xp1_rows, p_xp1, xp_ct, EXTRACT_MAX_NEW, None)
    run_pass1("GEN", sub, p_gen1, None, GEN_MAX_NEW, GEN_MAX_FACTS)
    run_pass1("X1u", x1u_rows, p_x1u1, x1u_ct, EXTRACT_MAX_NEW, None)

    # V3 computed as soon as GEN facts exist, logged here and again in order
    gen_kept = [f for e in extractions["GEN"] for f in e["facts"]]
    gen_raw = [e["raw_extraction"] for e in extractions["GEN"]]
    v3_frac, v3_letters, v3_latin = latin_fraction(gen_kept)
    v3_raw_frac = latin_fraction(gen_raw)[0]
    print("\n[V3] GEN kept facts: Latin letters %d / %d = %.4f (raw output %.4f)%s"
          % (v3_latin, v3_letters, v3_frac, v3_raw_frac,
             "   *** GEN VOID (< %.2f)" % V3_LATIN_MIN if v3_frac < V3_LATIN_MIN else ""),
          flush=True)

    # ---------------- PASS 2 ----------------
    facts_for = {
        "A_r": {r["qid"]: [] for r in sub},
        "Xp": {r["qid"]: [] for r in sub},
        "GEN": {r["qid"]: [] for r in sub},
        "X1u": {r["qid"]: [] for r in x1u_rows},
    }
    for arm in ("Xp", "GEN", "X1u"):
        for e in extractions[arm]:
            facts_for[arm][e["qid"]] = e["facts"]
    had_pass1 = {e["qid"] for e in extractions["Xp"]}
    results = OrderedDict()
    gold_fact_hits = OrderedDict()      # Amendment 2 A2.2: arm -> qid -> matched facts

    def run_pass2(arm, prs):
        print("\n" + "=" * 70)
        print("PASS 2 — arm %s  (n=%d, max_new=%d, frozen Urdu prompt)"
              % (arm, len(prs), MAX_NEW_TOKENS))
        print("=" * 70, flush=True)
        if not prs:
            print("  (no rows)")
            results[arm] = []
            gold_fact_hits[arm] = OrderedDict()
            return
        prompts = [answer_prompt(r, facts_for[arm][r["qid"]]) for r in prs]
        hits = leak_guard("%s/pass2" % arm, prompts, needles,
                          fatal_gold_facts=(arm == "A_r"))
        gold_fact_hits[arm] = OrderedDict()
        for i, s in hits:
            gold_fact_hits[arm].setdefault(prs[i]["qid"], []).append(s)
        log_full_prompt("%s pass 2, row 0, qid %s" % (arm, prs[0]["qid"]), prompts[0])
        gens = generate_all(model, tok, prompts, args.batch_size, eos_ids, MAX_NEW_TOKENS)
        recs = []
        for r, (text, n_new, trunc) in zip(prs, gens):
            fl = facts_for[arm][r["qid"]]
            rec = OrderedDict([
                ("qid", r["qid"]), ("arm", arm), ("gold", r["gold"]),
                ("pred", extract_answer(text)),
                ("pred_secondary", extract_answer_secondary(text)),
                ("truncated", trunc), ("n_gen_tokens", n_new),
                ("n_facts_given", len(fl)),
                ("n_pages", len(xp_ct[r["qid"]]) if arm == "Xp"
                 else len(x1u_ct[r["qid"]]) if arm == "X1u" else 0),
                ("had_pass1", (r["qid"] in had_pass1) if arm == "Xp"
                 else arm in ("GEN", "X1u")),
                ("generation", text),
            ])
            recs.append(rec)
        results[arm] = recs
        write_jsonl(os.path.join(OUT_DIR, "d8_arm%s%s.jsonl" % (arm, suffix)),
                    recs, expect_n=len(prs))
        s = score(recs, "pred")
        print("  accuracy %.2f%%  unparsed %.2f%%  (%d/%d correct)"
              % (s["accuracy"], s["unparsed_rate"], s["correct"], s["n"]))
        print("  sample generation (row 0, first 300 chars): %r"
              % recs[0]["generation"][:300], flush=True)

    run_pass2("A_r", sub)
    run_pass2("Xp", sub)
    run_pass2("GEN", sub)
    run_pass2("X1u", x1u_rows)

    # ---------------- references (read-only) ----------------
    d1a = [r for r in d1a_all if r["qid"] in sub_set]
    for r in d1a:
        r["pred_secondary"] = extract_answer_secondary(r["generation"])
    d4x1 = [r for r in read_jsonl(D4_X1_PATH) if r["qid"] in d4_68 and r["qid"] in sub_set]
    for r in d4x1:
        if "pred_secondary" not in r:
            r["pred_secondary"] = extract_answer_secondary(r["generation"])

    # ---------------- validity and readings, in the frozen order ----------------
    def sub_recs(recs, qset):
        return [r for r in recs if r["qid"] in qset]

    def acc(recs):
        return score(recs, "pred")["accuracy"] if recs else None

    def pair(x, y, lx, ly):
        if not x or not y:
            return OrderedDict([("n", 0), ("b", 0), ("c", 0), ("p", None)])
        b, c, p = mcnemar(x, y, lx, ly)
        return OrderedDict([("n", len(x)), ("acc_%s" % lx, acc(x)),
                            ("acc_%s" % ly, acc(y)),
                            ("delta_pp", round(acc(x) - acc(y), 2)),
                            ("wins_%s" % lx, b), ("losses_%s" % lx, c),
                            ("discordant", b + c), ("p_mcnemar_exact_two_sided", p)])

    summary = OrderedDict()
    summary["mode"] = "TEST — NOT A RESULT" if is_test else "FULL"
    summary["n"] = len(sub)
    summary["slurm_job_id"] = job_id
    hdr = "=" * 84
    print("\n" + hdr)
    print("D8 VALIDITY (freeze §6, fixed order)%s" % ("   (TEST — NOT A RESULT)" if is_test else ""))
    print(hdr)

    # V1
    unp = OrderedDict((a, score(results[a], "pred")["unparsed_rate"]) for a in ("A_r", "Xp", "GEN"))
    unp_sec = OrderedDict((a, score(results[a], "pred_secondary")["unparsed_rate"])
                          for a in ("A_r", "Xp", "GEN"))
    v1_spread = max(unp.values()) - min(unp.values())
    v1_ok = v1_spread <= V1_SPREAD_LIMIT
    summary["V1_unparsed_spread"] = OrderedDict([
        ("unparsed_rate_primary", unp), ("spread_pp", round(v1_spread, 2)),
        ("unparsed_rate_secondary", unp_sec),
        ("spread_secondary_pp", round(max(unp_sec.values()) - min(unp_sec.values()), 2)),
        ("limit_pp", V1_SPREAD_LIMIT), ("pass", v1_ok),
        ("verdict", "PASS" if v1_ok else "VOID (AMENDMENT 5D)")])
    print("V1 unparsed spread A_r/Xp/GEN: %s -> %.2f pp -> %s"
          % (dict(unp), v1_spread, summary["V1_unparsed_spread"]["verdict"]))

    # V2
    ar_by = {r["qid"]: r for r in results["A_r"]}
    agree = sum(1 for r in d1a if ar_by[r["qid"]]["pred"] == r["pred"])
    agree_pct = 100.0 * agree / len(d1a) if d1a else 0.0
    v2_diff = abs(acc(results["A_r"]) - acc(d1a))
    v2_ok = v2_diff <= V2_ACC_TOL and agree_pct >= V2_AGREE_MIN
    summary["V2_Ar_vs_D1_armA"] = OrderedDict([
        ("acc_A_r", acc(results["A_r"])), ("acc_D1_A", acc(d1a)),
        ("acc_diff_pp", round(v2_diff, 2)), ("rows_agree", agree), ("n", len(d1a)),
        ("row_agreement_pct", round(agree_pct, 2)),
        ("tolerance_pp", V2_ACC_TOL), ("agreement_min_pct", V2_AGREE_MIN),
        ("pass", v2_ok),
        ("nondeterminism_flag", not v2_ok),
        ("note", "readings pair with A_r only" if not v2_ok else "")])
    print("V2 A_r %.2f%% vs D1 arm A %.2f%% (diff %.2f pp); row agreement %d/%d = %.2f%% -> %s"
          % (acc(results["A_r"]), acc(d1a), v2_diff, agree, len(d1a), agree_pct,
             "PASS" if v2_ok else "NONDETERMINISM FLAG (readings use A_r only)"))

    # V2b (Amendment 2 A2.3, reported, not a gate): this A_r vs the archived
    # A_r of failed job 107156 (same stack, same batch size)
    if os.path.exists(ARCHIVED_AR_PATH):
        arch = [r for r in read_jsonl(ARCHIVED_AR_PATH) if r["qid"] in ar_by]
        agree_b = sum(1 for r in arch if ar_by[r["qid"]]["pred"] == r["pred"])
        agree_b_pct = 100.0 * agree_b / len(arch) if arch else 0.0
        v2b_diff = (acc(results["A_r"]) - acc(arch)) if arch else None
        summary["V2b_Ar_vs_archived_Ar"] = OrderedDict([
            ("archived_path", ARCHIVED_AR_PATH), ("gate", False),
            ("acc_A_r", acc(results["A_r"])), ("acc_archived_A_r", acc(arch)),
            ("acc_diff_pp", round(v2b_diff, 2) if v2b_diff is not None else None),
            ("rows_agree", agree_b), ("n", len(arch)),
            ("row_agreement_pct", round(agree_b_pct, 2))])
        print("V2b A_r %.2f%% vs archived A_r (job 107156) %.2f%% (diff %.2f pp); "
              "row agreement %d/%d = %.2f%% (reported, not a gate)"
              % (acc(results["A_r"]), acc(arch) if arch else 0.0,
                 v2b_diff if v2b_diff is not None else 0.0,
                 agree_b, len(arch), agree_b_pct))
    else:
        summary["V2b_Ar_vs_archived_Ar"] = OrderedDict([
            ("archived_path", ARCHIVED_AR_PATH), ("gate", False),
            ("note", "skipped: archived A_r file absent")])
        print("V2b skipped: archived A_r file absent (%s)" % ARCHIVED_AR_PATH)

    # V3
    v3_ok = v3_frac >= V3_LATIN_MIN
    summary["V3_GEN_latin_fraction"] = OrderedDict([
        ("latin_letters", v3_latin), ("letters", v3_letters),
        ("fraction_kept_facts", round(v3_frac, 4)),
        ("fraction_raw_output", round(v3_raw_frac, 4)),
        ("min", V3_LATIN_MIN), ("pass", v3_ok),
        ("verdict", "PASS" if v3_ok else "GEN VOID")])
    print("V3 GEN Latin fraction %.4f (raw %.4f) -> %s"
          % (v3_frac, v3_raw_frac, "PASS" if v3_ok else "GEN VOID"))

    # V4
    v4_hist = dict(Counter(len(xp_ct[q]) for q in sub_qids))
    v4_with = sum(1 for q in sub_qids if xp_ct[q])
    v4_ok = (v4_with == EXPECTED_QIDS_WITH_PAGE and v4_hist == EXPECTED_PAGES_HIST)
    summary["V4_Xp_pages"] = OrderedDict([
        ("qids_with_page", v4_with),
        ("pages_hist", OrderedDict((str(k), v) for k, v in sorted(v4_hist.items()))),
        ("expected_qids_with_page", EXPECTED_QIDS_WITH_PAGE),
        ("expected_hist", OrderedDict((str(k), v) for k, v in sorted(EXPECTED_PAGES_HIST.items()))),
        ("pass", v4_ok if not is_test else None),
        ("note", "subset of a TEST run; equality not required" if is_test else "")])
    print("V4 Xp qids with >=1 page %d, histogram %s -> %s"
          % (v4_with, dict(sorted(v4_hist.items())),
             "n/a (TEST subset)" if is_test else ("PASS" if v4_ok else "FAIL")))
    if not is_test and not v4_ok:
        die("V4: Xp page counts differ from the pre-run count in Amendment 1 A2")

    # V5 sample (reported, not a gate)
    xp_facts = [(e["qid"], e["titles_corpus"], f) for e in extractions["Xp"] for f in e["facts"]]
    rng = random.Random(V5_SEED)
    v5 = rng.sample(xp_facts, min(V5_SAMPLE, len(xp_facts))) if xp_facts else []
    summary["V5_Xp_hand_check_sample"] = OrderedDict([
        ("seed", V5_SEED), ("n", len(v5)), ("status", "PENDING hand check"),
        ("facts", [OrderedDict([("qid", q), ("titles_corpus", t), ("fact", f)]) for q, t, f in v5])])
    print("V5 %d random Xp facts listed in the summary for hand checking (seed %d)"
          % (len(v5), V5_SEED))

    # V6 (Amendment 2 A2.2, reported, not a gate): verbatim gold-fact
    # substrings in the pass-2 prompts of Xp / GEN / X1u, per arm
    v6_arms = OrderedDict()
    for arm in ("Xp", "GEN", "X1u"):
        hits_by_qid = gold_fact_hits.get(arm, OrderedDict())
        v6_arms[arm] = OrderedDict([
            ("count", sum(len(v) for v in hits_by_qid.values())),
            ("qids", len(hits_by_qid)),
            ("matches", [OrderedDict([("qid", q), ("fact", f)])
                         for q, fs in hits_by_qid.items() for f in fs])])
    summary["V6_gold_fact_verbatim_pass2"] = OrderedDict([
        ("gate", False),
        ("note", "model-restated page facts byte-identical to a gold fact; "
                 "pass-1 prompts and A_r pass 2 stay gated (Amendment 2 A2.2)"),
        ("arms", v6_arms)])
    print("V6 verbatim gold-fact substrings in pass-2 prompts (reported, not a gate): %s"
          % ", ".join("%s %d in %d qid(s)" % (a, v["count"], v["qids"])
                      for a, v in v6_arms.items()))

    # ---- readings ----
    print("\n" + hdr)
    print("D8 READINGS (freeze §7, paired exact McNemar two-sided vs A_r)%s"
          % ("   (TEST — NOT A RESULT)" if is_test else ""))
    print(hdr)
    if not v1_ok:
        print("*** V1 VOID: numbers below are for diagnosis only, not results.")

    # R1
    x1u = results["X1u"]
    ar_68 = sub_recs(results["A_r"], {r["qid"] for r in x1u})
    r1_correct = sum(1 for r in x1u if r["pred"] == r["gold"])
    r1_stands = r1_correct >= R1_MIN_CORRECT
    r1 = OrderedDict([
        ("n", len(x1u)), ("X1u_correct", r1_correct), ("threshold", R1_MIN_CORRECT),
        ("acc_X1u", acc(x1u)),
        ("D4_X1_reference_correct", sum(1 for r in d4x1 if r["pred"] == r["gold"])),
        ("D4_X1_reference_n", len(d4x1)),
        ("X1u_vs_A_r", pair(x1u, ar_68, "X1u", "A_r")),
        ("verdict", ("D4 STANDS: extraction works without question_en; X1u is the honest number"
                     if r1_stands else
                     "D4 OVERSTATED: X1u replaces X1 everywhere; declare and run D8b first")
         if not is_test else "n/a (TEST)")])
    summary["R1_question_en_confound"] = r1
    print("R1 X1u %d/%d correct (threshold %d/68) -> %s"
          % (r1_correct, len(x1u), R1_MIN_CORRECT, r1["verdict"]))

    # R2
    r2p = pair(results["Xp"], results["A_r"], "Xp", "A_r")
    d2, p2 = r2p.get("delta_pp"), r2p.get("p_mcnemar_exact_two_sided")
    if d2 is None or d2 < COND_PP:
        r2v = "NO-GO for the reachable-page route"
    elif d2 >= GATE_PP and p2 is not None and p2 < ALPHA:
        r2v = "GO: build the linker and run D9 end-to-end on DEV200"
    else:
        r2v = "CONDITIONAL: replacement corpus must raise traced-and-present; re-run Xp before D9"
    r2 = OrderedDict(r2p)
    r2["gate_pp"] = GATE_PP
    r2["conditional_pp"] = COND_PP
    r2["alpha"] = ALPHA
    r2["verdict"] = r2v if not is_test else "n/a (TEST)"
    summary["R2_reachable_ceiling"] = r2
    print("R2 Xp - A_r = %s pp, wins %s losses %s, p = %s -> %s"
          % (d2, r2p.get("wins_Xp"), r2p.get("losses_Xp"), p2, r2["verdict"]))

    # R3
    r3p = pair(results["GEN"], results["A_r"], "GEN", "A_r")
    d3, p3 = r3p.get("delta_pp"), r3p.get("p_mcnemar_exact_two_sided")
    if d3 is None:
        r3v = "n/a"
    elif d3 < R3_HARM_PP:
        r3v = "HARM: self-generated facts hurt; never used"
    elif d3 >= GATE_PP and p3 is not None and p3 < ALPHA:
        r3v = "VIABLE component: D9 combines page facts with GEN facts for unreached questions"
    else:
        r3v = "CLOSED: the model cannot supply the missing knowledge in English either"
    if not v3_ok:
        r3v = "GEN VOID (V3): " + r3v
    r3 = OrderedDict(r3p)
    r3["gate_pp"] = GATE_PP
    r3["harm_pp"] = R3_HARM_PP
    r3["alpha"] = ALPHA
    r3["verdict"] = r3v if not is_test else "n/a (TEST)"
    summary["R3_english_self_knowledge"] = r3
    print("R3 GEN - A_r = %s pp, wins %s losses %s, p = %s -> %s"
          % (d3, r3p.get("wins_GEN"), r3p.get("losses_GEN"), p3, r3["verdict"]))

    # secondary readings (reported without claim)
    with_page = {q for q in sub_qids if xp_ct[q]}
    sec = OrderedDict()
    sec["Xp_vs_A_r_on_qids_with_page"] = pair(sub_recs(results["Xp"], with_page),
                                              sub_recs(results["A_r"], with_page),
                                              "Xp", "A_r")
    x1u_q = {r["qid"] for r in x1u}
    sec["Xp_on_D4_68_vs_X1u"] = pair(sub_recs(results["Xp"], x1u_q), x1u, "Xp", "X1u")
    sec["Xp_on_D4_68_pages_hist"] = OrderedDict(
        (str(k), v) for k, v in sorted(Counter(len(xp_ct[q]) for q in x1u_q).items()))
    summary["secondary_readings"] = sec
    print("Secondary: Xp vs A_r on %d qids with >=1 page: %s"
          % (len(with_page), json.dumps(sec["Xp_vs_A_r_on_qids_with_page"])))
    print("Secondary: Xp on D4-68 vs X1u: %s" % json.dumps(sec["Xp_on_D4_68_vs_X1u"]))

    # ---- accuracy table, ONLY NOW ----
    n_yes = sum(1 for r in sub if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(sub) - n_yes) / len(sub)
    table = OrderedDict()
    print("\n" + hdr)
    print("ACCURACY TABLE%s   floor %.2f%%" % ("   (TEST — NOT A RESULT)" if is_test else "", floor))
    print(hdr)
    print("%-10s %-32s %5s %8s %9s %9s" % ("arm", "facts", "n", "acc%", "unpars%", "SEC acc%"))
    label = {"A_r": "none (re-run)", "Xp": "reachable gold pages, extracted",
             "GEN": "self-generated English facts", "X1u": "all gold pages, no question_en",
             "D1_A_ref": "none (D1, reused)", "D4_X1_ref": "all gold pages + question_en (D4)"}
    for arm, recs in list(results.items()) + [("D1_A_ref", d1a), ("D4_X1_ref", d4x1)]:
        if not recs:
            table[arm] = OrderedDict([("n", 0)])
            continue
        pri = score(recs, "pred")
        se = score(recs, "pred_secondary")
        table[arm] = OrderedDict([
            ("n", pri["n"]), ("accuracy", pri["accuracy"]), ("correct", pri["correct"]),
            ("unparsed_rate", pri["unparsed_rate"]),
            ("truncation_rate", round(100.0 * sum(1 for r in recs if r.get("truncated")) / pri["n"], 2)),
            ("predicted_yes_rate", pri["predicted_yes_rate"]),
            ("accuracy_secondary", se["accuracy"]), ("unparsed_rate_secondary", se["unparsed_rate"])])
        print("%-10s %-32s %5d %8.2f %9.2f %9.2f"
              % (arm, label[arm], pri["n"], pri["accuracy"], pri["unparsed_rate"], se["accuracy"]))
    summary["floor"] = round(floor, 2)
    summary["accuracy_table"] = table
    write_json(os.path.join(OUT_DIR, "d8_summary%s.json" % suffix), summary)

    # ---------------- SEAL ----------------
    files = OrderedDict()
    for p in WRITTEN:
        with open(p, "rb") as f:
            b = f.read()
        files[p] = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)),
                                ("lines", b.count(b"\n"))])
    seal = OrderedDict([
        ("mode", "TEST" if is_test else "FULL"), ("slurm_job_id", job_id),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("script_md5", meta["script"]["md5"]), ("freeze_md5", meta["freeze"]["md5"]),
        ("amendment_md5s", [a["md5"] for a in meta["amendments"]]),
        ("files", files)])
    write_json(os.path.join(OUT_DIR, "d8_SEAL%s.json" % suffix), seal)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD8 COMPLETE. Sealed %d files." % len(files))


if __name__ == "__main__":
    main()
