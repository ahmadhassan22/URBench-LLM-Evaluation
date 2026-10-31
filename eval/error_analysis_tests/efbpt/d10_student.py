#!/usr/bin/env python3
"""
d10_student.py  —  D10 Job S (student, HF transformers, nf4, batch size 1)

Contract: docs/EFBPT_D10_TEACHER_FREEZE.md (FROZEN, MD5
b3a822d8199034c7fe58d76cf5cc428f), sections 2, 4, 5, 6, 7, 8.
Implements the freeze exactly; nothing here is tunable after numbers exist.

Arms (freeze §4), n=200 each, same job, same row order, batch size 1:
  A_r2      Urdu question, no facts                     arm "A" layout (floor)
  TF        Urdu question + teacher facts of own qid    arm "B" layout
  TF_wrong  Urdu question + facts of derangement(qid)   arm "B" layout (control)
  TF_noq    no question  + teacher facts of own qid     arm "G" layout (leak probe)
TF / TF_wrong with an empty fact list -> arm "A" prompt; TF_noq with an empty
fact list -> arm "G" layout with an empty block. All kept, scored, counted.

Everything frozen is IMPORTED, never re-implemented:
  model path, SYSTEM_MESSAGE, instruction (+MD5), extractor  <- efbpt_eval_dev200
  user layouts arm "A"/"B"/"G", facts header                 <- d1_eval_arms
  secondary (Devanagari) extractor                           <- d1_score_dual
  norm(), required_titles(), score()                         <- d3_oracle_retrieval
  DEV200 loader, generate_all                                <- d4_extract_facts
  paired exact McNemar                                       <- d5_balanced_extract
  leak guard, needles, Latin fraction, md5/json helpers      <- d8_reachable_ceiling

Modes:
  --preflight-only   CPU, no model: freeze/script/instruction MD5s, V7 (T_SEAL
                     verification), teacher outputs, derangement, all four
                     prompt sets, leak guard (V6), V8, V9, V4, V5, R4.
                     Writes NOTHING.
  --limit N          TEST: first N rows; reads the TEST teacher outputs under
                     outputs/efbpt/d10/_test/teacher/ and writes ONLY under
                     outputs/efbpt/d10/_test/student/. Never a result.
  (no flags)         FULL: outputs/efbpt/d10/student/.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d10_student.py --preflight-only
  python eval/error_analysis_tests/efbpt/d10_student.py --limit 5
  python eval/error_analysis_tests/efbpt/d10_student.py
"""

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from efbpt_eval_dev200 import (          # noqa: E402
    MODEL_PATH, SYSTEM_MESSAGE, MAX_NEW_TOKENS, INSTRUCTION_PATH, INSTRUCTION_MD5,
    extract_answer, load_instruction, die,
)
from d1_eval_arms import build_user, FACTS_HEADER   # noqa: E402
from d1_score_dual import extract_answer_secondary  # noqa: E402
from d3_oracle_retrieval import norm, score         # noqa: E402
from d4_extract_facts import load_rows, generate_all  # noqa: E402
from d5_balanced_extract import mcnemar             # noqa: E402
from d8_reachable_ceiling import (                  # noqa: E402  same guard as D8
    leak_guard, load_leak_needles, latin_fraction,
    md5_bytes, md5_file, read_jsonl, pkg_versions,
)

# ----------------------------------------------------------------------------
# Frozen constants (freeze §2, §4, §5, §6, §7)
# ----------------------------------------------------------------------------

FREEZE_PATH = "docs/EFBPT_D10_TEACHER_FREEZE.md"
FREEZE_MD5 = "b3a822d8199034c7fe58d76cf5cc428f"

T_DIR_FULL = "outputs/efbpt/d10/teacher"
T_DIR_TEST = "outputs/efbpt/d10/_test/teacher"
OUT_DIR_FULL = "outputs/efbpt/d10/student"
OUT_DIR_TEST = "outputs/efbpt/d10/_test/student"
FORBIDDEN_WRITE_DIRS = ("outputs/efbpt/d1", "outputs/efbpt/d4", "outputs/efbpt/d6",
                        "outputs/efbpt/d8", "outputs/efbpt/d10/teacher",
                        "outputs/efbpt/d10/_test/teacher")

# sealed D6 outputs (read-only): the 71 qids and the G4 reference
D6_E1_PATH = "outputs/efbpt/d6/d6_armE1.jsonl"
D6_SUMMARY_PATH = "outputs/efbpt/d6/d6_summary.json"
D6_N = 71
D6_G4_INTERSECT = 42
D6_G4_GOLD_TOTAL = 200

BATCH_SIZE = 1                     # §2, §4: BATCH SIZE 1
CONTEXT_WINDOW = 40960             # same check as D4/D8

# §5
V1_SPREAD_LIMIT = 5.0
V2_PROBE_N_FULL = 20
V2_PROBE_N_TEST = 5
V4_PARSE_MIN = 190                 # of 200 (95%)
V4_PARSE_MIN_PCT = 95.0
V5_REMOVED_MAX = 100
# §6
GATE_PP = 8.0
COND_PP = 5.0
HARM_PP = -3.0
ALPHA = 0.05
R2_TDIRECT_UNPARSED_MAX = 10.0     # T3: above this, R2 also on parsed rows
R3_FORMAT_PP = 4.0
R4_MECHANISM_MIN_PCT = 30.0
D1_ARM_G_OVERALL = 42.0            # reference, D1 arm G
D1_ARM_G_PARSED_ONLY = 63.2
D8_PRED_YES_REF = OrderedDict([("A_r", 47.0), ("GEN", 26.0)])
SAMPLE_SEED = 8
SAMPLE_N = 30

ARMS = ("A_r2", "TF", "TF_wrong", "TF_noq")
LAYOUT = {"A_r2": "A", "TF": "B", "TF_wrong": "B", "TF_noq": "G"}

# ----------------------------------------------------------------------------
# Output helpers (same layout as D8; writable root depends on the mode)
# ----------------------------------------------------------------------------

OUT_DIR = None
WRITTEN = []


def guard_out_path(path):
    ap = os.path.abspath(path)
    root = os.path.abspath(OUT_DIR)
    for forb in FORBIDDEN_WRITE_DIRS:
        fr = os.path.abspath(forb)
        if ap == fr or ap.startswith(fr + os.sep):
            die("refusing to write under %s: %s" % (forb, path))
    if not ap.startswith(root + os.sep):
        die("refusing to write outside %s: %s" % (OUT_DIR, path))
    return ap


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
        json.load(f)
    print("  [verified] wrote %s" % path, flush=True)
    WRITTEN.append(path)


def log_full_prompt(label, prompt):
    print("\n----- BEGIN FULL PROMPT [%s] (%d chars) -----" % (label, len(prompt)))
    print(prompt)
    print("----- END FULL PROMPT [%s] -----" % label)
    print("  tail repr (thinking tag): %r" % prompt[-80:], flush=True)


def git_state():
    def run(*a):
        try:
            return subprocess.check_output(["git"] + list(a), cwd=_HERE,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    dirty = run("status", "--porcelain")
    return OrderedDict([("head", run("rev-parse", "HEAD")),
                        ("dirty", bool(dirty) if dirty is not None else None)])


# ----------------------------------------------------------------------------
# V7: the teacher seal
# ----------------------------------------------------------------------------

def verify_teacher_seal(t_dir):
    seal_path = os.path.join(t_dir, "T_SEAL.json")
    if not os.path.exists(seal_path):
        die("T_SEAL.json missing: %s (run Job T first)" % seal_path)
    seal = json.load(open(seal_path, encoding="utf-8"))
    if seal.get("freeze_md5") != FREEZE_MD5:
        die("T_SEAL freeze_md5 %s != %s" % (seal.get("freeze_md5"), FREEZE_MD5))
    checked = OrderedDict()
    for p, want in seal["files"].items():
        if not os.path.exists(p):
            die("V7: sealed file missing: %s" % p)
        b = open(p, "rb").read()
        got = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)), ("lines", b.count(b"\n"))])
        if got["md5"] != want["md5"] or got["bytes"] != want["bytes"] or got["lines"] != want["lines"]:
            die("V7: %s differs from T_SEAL: sealed %s, on disk %s" % (p, dict(want), dict(got)))
        checked[p] = got["md5"]
    need = {"t_facts.jsonl", "t_direct.jsonl", "derangement.json", "T_START.json"}
    have = {os.path.basename(p) for p in checked}
    if not need <= have:
        die("V7: T_SEAL does not cover %s" % sorted(need - have))
    print("[V7] T_SEAL %s: %d files verified (md5, bytes, lines) -> PASS"
          % (seal_path, len(checked)))
    return seal, checked


# ----------------------------------------------------------------------------
# V8: the fact lines inside a rendered prompt
# ----------------------------------------------------------------------------

def facts_in_prompt(prompt):
    """Returns (n_header, fact lines). The header is the frozen Urdu facts
    header followed by ':' and a newline; the lines are the consecutive
    '- ' lines after it (d1_eval_arms.facts_block)."""
    key = FACTS_HEADER + ":\n"
    n = prompt.count(key)
    if n == 0:
        return 0, []
    rest = prompt[prompt.index(key) + len(key):].split("\n")
    out = []
    for line in rest:
        if line.startswith("- "):
            out.append(line[2:])
        else:
            break
    return n, out


# ----------------------------------------------------------------------------
# Preflight (CPU, no model)
# ----------------------------------------------------------------------------

def preflight(limit, t_dir):
    print("=" * 78)
    print("D10 JOB S PREFLIGHT (freeze §4, §5)")
    print("=" * 78, flush=True)

    if not os.path.exists(FREEZE_PATH):
        die("freeze file missing: " + FREEZE_PATH)
    freeze_md5 = md5_file(FREEZE_PATH)
    if freeze_md5 != FREEZE_MD5:
        die("freeze MD5 mismatch: expected %s, got %s" % (FREEZE_MD5, freeze_md5))
    script_md5 = md5_file(os.path.abspath(__file__))
    print("[md5] freeze  %s  %s" % (freeze_md5, FREEZE_PATH))
    print("[md5] script  %s  %s" % (script_md5, os.path.relpath(__file__)))

    instruction = load_instruction()              # MD5 asserted inside
    print("[ok] Urdu instruction MD5 %s verified" % INSTRUCTION_MD5)

    rows = load_rows()
    qids = [r["qid"] for r in rows]
    if len(set(qids)) != 200:
        die("DEV200 qids are not unique")
    for r in rows:
        if "facts" in r:
            die("loader row carries a 'facts' key; gold facts must never reach build_user")
    print("[ok] DEV200: 200 rows, 200 unique qids, no 'facts' key in any row")
    sub = rows[:limit] if limit else rows
    sub_qids = [r["qid"] for r in sub]
    sub_set = set(sub_qids)

    # ---- V7 and the teacher outputs ----
    seal, seal_files = verify_teacher_seal(t_dir)
    t_start = json.load(open(os.path.join(t_dir, "T_START.json"), encoding="utf-8"))
    if t_start.get("freeze", {}).get("md5") != FREEZE_MD5:
        die("T_START freeze md5 differs")
    if t_start.get("urdu_instruction", {}).get("md5") != INSTRUCTION_MD5:
        die("T_START instruction md5 differs")
    t_facts = read_jsonl(os.path.join(t_dir, "t_facts.jsonl"))
    t_direct = read_jsonl(os.path.join(t_dir, "t_direct.jsonl"))
    der = json.load(open(os.path.join(t_dir, "derangement.json"), encoding="utf-8"))
    dmap = OrderedDict(der["map"])
    tf_q = [e["qid"] for e in t_facts]
    if tf_q != sub_qids:
        die("t_facts.jsonl qids (%d) != this run's DEV200 rows (%d) in order"
            % (len(tf_q), len(sub_qids)))
    if [e["qid"] for e in t_direct] != sub_qids:
        die("t_direct.jsonl qids != this run's rows")
    if list(dmap.keys()) != sub_qids or sorted(dmap.values()) != sorted(sub_qids) \
            or any(k == v for k, v in dmap.items()):
        die("derangement.json is not a fixed-point-free permutation of these qids")
    facts_by = OrderedDict((e["qid"], list(e["facts"])) for e in t_facts)
    teacher_by = OrderedDict((e["qid"], e) for e in t_facts)
    for q, fl in facts_by.items():
        if len(fl) > 6 or any((not isinstance(f, str)) or f != f.strip() or not f for f in fl):
            die("t_facts qid %s: facts are not <= 6 clean non-empty strings" % q)
        if teacher_by[q]["parse_ok"] is False and fl:
            die("t_facts qid %s: parse_ok false but facts present" % q)
    print("[ok] teacher outputs: %d t_facts rows, %d t_direct rows, derangement of %d"
          % (len(t_facts), len(t_direct), len(dmap)))

    # ---- V4, V5 from the sealed teacher file ----
    n_parse_ok = sum(1 for e in t_facts if e["parse_ok"])
    parse_pct = 100.0 * n_parse_ok / len(t_facts)
    v5_removed = sum(len(e["removed"]) for e in t_facts)
    print("[V4] teacher parse_ok %d/%d = %.1f%% -> %s"
          % (n_parse_ok, len(t_facts), parse_pct,
             "ok" if parse_pct >= V4_PARSE_MIN_PCT else "FORMAT FLAG"))
    print("[V5] verdict-filter removed lines %d -> %s"
          % (v5_removed, "ok" if v5_removed <= V5_REMOVED_MAX else "TEACHER-ANSWERING FLAG"))

    # ---- prompts (tokenizer only) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def chat(system, user):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    def source_qid(arm, q):
        return dmap[q] if arm == "TF_wrong" else q

    def arm_prompt(arm, row):
        """Freeze §4 layouts, verbatim rule set."""
        if arm == "A_r2":
            return chat(SYSTEM_MESSAGE, build_user(instruction, row, None, "A")), "A", []
        facts = facts_by[source_qid(arm, row["qid"])]
        if arm in ("TF", "TF_wrong"):
            if facts:
                user = build_user(instruction, dict(row, facts=list(facts)), None, "B")
                return chat(SYSTEM_MESSAGE, user), "B", facts
            return chat(SYSTEM_MESSAGE, build_user(instruction, row, None, "A")), "A", []
        # TF_noq: arm "G", with an empty block when the list is empty
        user = build_user(instruction, dict(row, facts=list(facts)), None, "G")
        return chat(SYSTEM_MESSAGE, user), "G", facts

    prompts = OrderedDict()     # arm -> list of (prompt, layout, facts, source qid)
    for arm in ARMS:
        prompts[arm] = []
        for r in sub:
            p, lay, fl = arm_prompt(arm, r)
            prompts[arm].append((p, lay, fl, source_qid(arm, r["qid"])))

    for arm in ARMS:
        ps = [p for p, _, _, _ in prompts[arm]]
        tl = sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in ps)
        lays = Counter(l for _, l, _, _ in prompts[arm])
        n_empty = sum(1 for _, _, fl, _ in prompts[arm] if not fl)
        print("[len] %-9s n=%d tokens min=%d median=%d max=%d  layouts %s  empty-facts %d"
              % (arm, len(ps), tl[0], tl[len(tl) // 2], tl[-1], dict(lays), n_empty))
        if tl[-1] + MAX_NEW_TOKENS > CONTEXT_WINDOW:
            die("%s prompts would exceed the context window" % arm)

    # ---- leak guard (§4): question_en / evidence ids fatal everywhere;
    #      gold-fact substrings fatal in A_r2, reported (V6) in the facts arms ----
    needles = load_leak_needles()
    v6 = OrderedDict()
    for arm in ARMS:
        hits = leak_guard(arm, [p for p, _, _, _ in prompts[arm]], needles,
                          fatal_gold_facts=(arm == "A_r2"))
        by_q = OrderedDict()
        for i, s in hits:
            by_q.setdefault(sub[i]["qid"], []).append(s)
        v6[arm] = OrderedDict([
            ("count", len(hits)), ("qids", len(by_q)),
            ("matches", [OrderedDict([("qid", q), ("fact", f)]) for q, fs in by_q.items() for f in fs])])

    # ---- V8: fact lines in every facts prompt == sealed post-T2 facts ----
    v8_checked = 0
    for arm in ARMS:
        for r, (p, lay, fl, sq) in zip(sub, prompts[arm]):
            n_hdr, found = facts_in_prompt(p)
            expect = facts_by[sq] if arm != "A_r2" else []
            if arm == "A_r2" or lay == "A":
                if n_hdr != 0:
                    die("V8: %s qid %s: facts header present in an arm-A prompt" % (arm, r["qid"]))
                if expect and lay == "A":
                    die("V8: %s qid %s: non-empty facts but arm-A layout" % (arm, r["qid"]))
                continue
            if n_hdr != 1:
                die("V8: %s qid %s: facts header appears %d times" % (arm, r["qid"], n_hdr))
            if found != expect:
                die("V8: %s qid %s (source %s): prompt facts %r != sealed facts %r"
                    % (arm, r["qid"], sq, found, expect))
            v8_checked += 1
    print("[V8] prompt-content check: %d facts prompts equal their sealed post-T2 facts -> PASS"
          % v8_checked)

    # ---- V9 ----
    all_injected = [f for q in sub_qids for f in facts_by[q]]
    v9_frac, v9_letters, v9_latin = latin_fraction(all_injected)
    print("[V9] Latin fraction of injected teacher facts %.4f (%d/%d letters; D8 GEN 0.9972)"
          % (v9_frac, v9_latin, v9_letters))

    # ---- R4 (needs no model): title-level, the D6 G4 way ----
    d6_qids = [r["qid"] for r in read_jsonl(D6_E1_PATH)]
    if len(d6_qids) != D6_N or len(set(d6_qids)) != D6_N or not set(d6_qids) <= set(qids):
        die("D6 arm E1 does not hold %d unique DEV200 qids" % D6_N)
    d6s = json.load(open(D6_SUMMARY_PATH, encoding="utf-8"))["gold_overlap"]
    if d6s.get("intersect") != D6_G4_INTERSECT or d6s.get("gold_total") != D6_G4_GOLD_TOTAL:
        die("D6 summary gold_overlap %s != reference %d/%d" % (d6s, D6_G4_INTERSECT, D6_G4_GOLD_TOTAL))
    req_by = {r["qid"]: {norm(t) for t in r["required"]} for r in rows}
    if limit is None and sum(len(req_by[q]) for q in d6_qids) != D6_G4_GOLD_TOTAL:
        die("required titles over the 71 D6 qids != %d" % D6_G4_GOLD_TOTAL)

    def r4_on(qset):
        inter = gold = 0
        per = []
        for q in sub_qids:
            if q not in qset:
                continue
            ents = {norm(e) for e in teacher_by[q]["entities"]}
            hit = sorted(req_by[q] & ents)
            inter += len(hit)
            gold += len(req_by[q])
            per.append(OrderedDict([("qid", q), ("gold_titles", sorted(req_by[q])),
                                    ("entities", teacher_by[q]["entities"]), ("hits", hit)]))
        return OrderedDict([("n_qids", len(per)), ("intersect", inter), ("gold_total", gold),
                            ("pct", round(100.0 * inter / gold, 2) if gold else None)]), per

    r4a, r4a_rows = r4_on(set(d6_qids))
    r4b, r4b_rows = r4_on(sub_set)
    r4_flag = (r4a["pct"] is not None) and (r4a["pct"] <= R4_MECHANISM_MIN_PCT)
    print("[R4] (a) D6-71: %s/%s = %s%% (reference 14B %d/%d = 21.0%%) -> %s"
          % (r4a["intersect"], r4a["gold_total"], r4a["pct"], D6_G4_INTERSECT, D6_G4_GOLD_TOTAL,
             "MECHANISM FLAG" if r4_flag else "ok"))
    print("[R4] (b) all rows: %s/%s = %s%% (descriptive)" % (r4b["intersect"], r4b["gold_total"], r4b["pct"]))

    log_full_prompt("A_r2 row 0, qid %s" % sub[0]["qid"], prompts["A_r2"][0][0])
    for arm in ("TF", "TF_wrong", "TF_noq"):
        i = next((i for i, (_, lay, _, _) in enumerate(prompts[arm]) if lay != "A"), 0)
        log_full_prompt("%s row %d, qid %s, layout %s" % (arm, i, sub[i]["qid"], prompts[arm][i][1]),
                        prompts[arm][i][0])

    pre = OrderedDict([
        ("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", freeze_md5)])),
        ("script", OrderedDict([("path", os.path.relpath(__file__)), ("md5", script_md5)])),
        ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("teacher_dir", t_dir),
        ("teacher_seal", OrderedDict([("script_md5", seal.get("script_md5")),
                                      ("slurm_job_id", seal.get("slurm_job_id")),
                                      ("files", seal_files)])),
        ("V7_T_SEAL_verified", OrderedDict([("pass", True), ("n_files", len(seal_files))])),
        ("V4_teacher_parse", OrderedDict([
            ("parse_ok", n_parse_ok), ("n", len(t_facts)), ("pct", round(parse_pct, 2)),
            ("min_pct", V4_PARSE_MIN_PCT), ("min_count_of_200", V4_PARSE_MIN),
            ("format_flag", parse_pct < V4_PARSE_MIN_PCT)])),
        ("V5_verdict_filter", OrderedDict([
            ("removed_lines", v5_removed), ("max", V5_REMOVED_MAX),
            ("teacher_answering_flag", v5_removed > V5_REMOVED_MAX)])),
        ("V6_gold_fact_verbatim", OrderedDict([("gate", False), ("arms", v6)])),
        ("V8_prompt_content", OrderedDict([("pass", True), ("facts_prompts_checked", v8_checked)])),
        ("V9_latin_fraction", OrderedDict([
            ("fraction", round(v9_frac, 4)), ("letters", v9_letters), ("latin_letters", v9_latin),
            ("n_facts", len(all_injected)), ("reference_D8_GEN", 0.9972)])),
        ("R4_mechanism", OrderedDict([
            ("definition", "title-level, D6 G4 way: gold required titles found (after norm) "
                           "in the teacher's parsed entities / total required titles"),
            ("a_D6_71", r4a), ("reference_14B_D6_G4", OrderedDict([
                ("intersect", D6_G4_INTERSECT), ("gold_total", D6_G4_GOLD_TOTAL), ("pct", 21.0)])),
            ("b_all_rows", r4b), ("flag_at_pct", R4_MECHANISM_MIN_PCT),
            ("mechanism_flag", r4_flag)])),
        ("layouts", OrderedDict((arm, OrderedDict([
            ("counts", OrderedDict(sorted(Counter(l for _, l, _, _ in prompts[arm]).items()))),
            ("empty_facts", sum(1 for _, _, fl, _ in prompts[arm] if not fl))])) for arm in ARMS)),
    ])
    print("[preflight] PASS", flush=True)
    return (instruction, sub, prompts, facts_by, teacher_by, dmap, t_direct, tok,
            pre, r4a_rows, r4b_rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true",
                    help="CPU: checks and prompts, write nothing, exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="TEST: first N rows; reads %s, writes only under %s"
                         % (T_DIR_TEST, OUT_DIR_TEST))
    args = ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        die("--limit must be positive")
    if args.preflight_only and args.limit is not None:
        die("--preflight-only takes no --limit")

    is_test = args.limit is not None
    OUT_DIR = OUT_DIR_TEST if is_test else OUT_DIR_FULL
    t_dir = T_DIR_TEST if is_test else T_DIR_FULL

    (instruction, sub, prompts, facts_by, teacher_by, dmap, t_direct, tok,
     pre, r4a_rows, r4b_rows) = preflight(args.limit, t_dir)

    if args.preflight_only:
        print("\n[preflight-only] done. No files written.")
        return

    print("\n[mode] %s  rows=%d  out=%s"
          % ("TEST — NOT A RESULT" if is_test else "FULL", len(sub), OUT_DIR))
    if not is_test and len(sub) != 200:
        die("FULL run must cover 200 rows")
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")

    # ---------------- model (exactly as D8 / D4) ----------------
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if im is not None and im >= 0:
        eos_ids.add(im)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("\n[load] base model, 4-bit nf4, no adapters ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    # ---------------- V2 determinism probe (preflight, before START) ----------------
    n_probe = min(V2_PROBE_N_TEST if is_test else V2_PROBE_N_FULL, len(sub))
    probe_prompts = [p for p, _, _, _ in prompts["A_r2"][:n_probe]]
    print("\n" + "=" * 70)
    print("V2 — determinism probe: %d A_r2 prompts, twice, batch size %d" % (n_probe, BATCH_SIZE))
    print("=" * 70, flush=True)
    g_a = generate_all(model, tok, probe_prompts, BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
    g_b = generate_all(model, tok, probe_prompts, BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
    n_same = sum(1 for (ta, _, _), (tb, _, _) in zip(g_a, g_b) if ta == tb)
    v2_pass = n_same == n_probe
    v2 = OrderedDict([("n", n_probe), ("identical", n_same), ("pass", v2_pass),
                      ("verdict", "PASS" if v2_pass else "DETERMINISM FLAG (run continues; readings are paired)")])
    print("V2 %d/%d identical strings -> %s" % (n_same, n_probe, v2["verdict"]), flush=True)

    start = OrderedDict([
        ("job", "S"), ("mode", "TEST" if is_test else "FULL"),
        ("args", vars(args)),
        ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("hostname", socket.gethostname()), ("slurm_job_id", job_id),
        ("n_rows", len(sub)), ("qids", [r["qid"] for r in sub]),
        ("model_path", MODEL_PATH),
        ("decoding", OrderedDict([
            ("batch_size", BATCH_SIZE), ("do_sample", False),
            ("enable_thinking", False), ("quant", "nf4 double-quant bf16 compute"),
            ("answer_max_new", MAX_NEW_TOKENS)])),
        ("versions", pkg_versions()),
        ("V2_determinism_probe", v2),
    ])
    start.update(pre)
    print("\n[START]")
    write_json(os.path.join(OUT_DIR, "S_START.json"), start)

    # ---------------- arms, same order, batch size 1 ----------------
    results = OrderedDict()
    for arm in ARMS:
        print("\n" + "=" * 70)
        print("ARM %s  (n=%d, layout %s, max_new=%d, batch size %d, greedy)"
              % (arm, len(sub), LAYOUT[arm], MAX_NEW_TOKENS, BATCH_SIZE))
        print("=" * 70, flush=True)
        ps = [p for p, _, _, _ in prompts[arm]]
        # generate_all sorts by length internally and returns in input order
        gens = generate_all(model, tok, ps, BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
        if len(gens) != len(ps):
            die("%s: %d generations for %d prompts" % (arm, len(gens), len(ps)))
        recs = []
        for r, (p, lay, fl, sq), (text, n_new, trunc) in zip(sub, prompts[arm], gens):
            pred = extract_answer(text)
            recs.append(OrderedDict([
                ("qid", r["qid"]), ("arm", arm), ("layout", lay), ("gold", r["gold"]),
                ("pred", pred), ("pred_secondary", extract_answer_secondary(text)),
                ("correct", pred == r["gold"]), ("unparsed", pred is None),
                ("truncated", trunc), ("n_gen_tokens", n_new),
                ("n_facts_given", len(fl)), ("facts_source_qid", sq if arm != "A_r2" else None),
                ("source_parse_ok", teacher_by[sq]["parse_ok"] if arm != "A_r2" else None),
                ("prompt", p), ("generation", text),
            ]))
        results[arm] = recs
        write_jsonl(os.path.join(OUT_DIR, "d10_arm%s.jsonl" % arm), recs, expect_n=len(sub))
        s = score(recs, "pred")
        print("  accuracy %.2f%%  unparsed %.2f%%  (%d/%d correct)  pred-yes %.2f%%"
              % (s["accuracy"], s["unparsed_rate"], s["correct"], s["n"], s["predicted_yes_rate"]))
        print("  sample generation (row 0, first 300 chars): %r" % recs[0]["generation"][:300], flush=True)

    # ---------------- validity and readings, frozen order ----------------
    def acc(recs):
        return score(recs, "pred")["accuracy"] if recs else None

    def pair(x, y, lx, ly):
        if not x or not y:
            return OrderedDict([("n", 0), ("b", 0), ("c", 0), ("p", None)])
        b, c, p = mcnemar(x, y, lx, ly)
        return OrderedDict([("n", len(x)), ("acc_%s" % lx, acc(x)), ("acc_%s" % ly, acc(y)),
                            ("delta_pp", round(acc(x) - acc(y), 2)),
                            ("wins_%s" % lx, b), ("losses_%s" % lx, c),
                            ("discordant", b + c), ("p_mcnemar_exact_two_sided", p)])

    def parsed_only(recs):
        n = sum(1 for r in recs if r["pred"] is not None)
        c = sum(1 for r in recs if r["pred"] is not None and r["pred"] == r["gold"])
        return OrderedDict([("parsed", n), ("correct", c),
                            ("accuracy", round(100.0 * c / n, 2) if n else None)])

    na = "n/a (TEST)"
    summary = OrderedDict()
    summary["mode"] = "TEST — NOT A RESULT" if is_test else "FULL"
    summary["n"] = len(sub)
    summary["slurm_job_id"] = job_id
    summary["teacher_seal"] = pre["teacher_seal"]
    hdr = "=" * 84
    print("\n" + hdr)
    print("D10 VALIDITY (freeze §5, fixed order)%s" % ("   (TEST — NOT A RESULT)" if is_test else ""))
    print(hdr)

    # V1
    unp = OrderedDict((a, score(results[a], "pred")["unparsed_rate"]) for a in ("A_r2", "TF", "TF_wrong"))
    v1_spread = max(unp.values()) - min(unp.values())
    v1_ok = v1_spread <= V1_SPREAD_LIMIT
    summary["V1_unparsed_spread"] = OrderedDict([
        ("unparsed_rate_primary", unp), ("spread_pp", round(v1_spread, 2)),
        ("TF_noq_unparsed_rate_reported", score(results["TF_noq"], "pred")["unparsed_rate"]),
        ("limit_pp", V1_SPREAD_LIMIT), ("pass", v1_ok),
        ("verdict", "PASS" if v1_ok else "VOID (AMENDMENT 5D)")])
    print("V1 unparsed spread A_r2/TF/TF_wrong: %s -> %.2f pp -> %s   (TF_noq unparsed %.2f%%, reported)"
          % (dict(unp), v1_spread, summary["V1_unparsed_spread"]["verdict"],
             summary["V1_unparsed_spread"]["TF_noq_unparsed_rate_reported"]))

    # V2
    summary["V2_determinism_probe"] = v2
    print("V2 determinism probe %d/%d identical -> %s" % (v2["identical"], v2["n"], v2["verdict"]))

    # V3: reaching this point means no guard was fatal in S; T completed and sealed
    summary["V3_leak_guards"] = OrderedDict([
        ("teacher_job", OrderedDict([("fatal", False), ("prompts_checked",
                                                       json.load(open(os.path.join(t_dir, "T_START.json"),
                                                                      encoding="utf-8"))["leak_guard"]["prompts_checked"])])),
        ("student_job", OrderedDict([("fatal", False), ("prompts_checked", sum(len(prompts[a]) for a in ARMS))])),
        ("pass", True)])
    print("V3 leak guards: none fatal in either job -> PASS")

    # V4, V5, V6, V7, V8, V9 (computed in preflight)
    summary["V4_teacher_parse"] = pre["V4_teacher_parse"]
    print("V4 teacher parse %d/%d = %.1f%% -> %s"
          % (pre["V4_teacher_parse"]["parse_ok"], pre["V4_teacher_parse"]["n"], pre["V4_teacher_parse"]["pct"],
             "FORMAT FLAG" if pre["V4_teacher_parse"]["format_flag"] else "PASS"))
    summary["V5_verdict_filter"] = pre["V5_verdict_filter"]
    print("V5 verdict-filter removed %d -> %s"
          % (pre["V5_verdict_filter"]["removed_lines"],
             "TEACHER-ANSWERING FLAG" if pre["V5_verdict_filter"]["teacher_answering_flag"] else "ok"))
    summary["V6_gold_fact_verbatim"] = pre["V6_gold_fact_verbatim"]
    print("V6 verbatim gold-fact substrings (reported, not a gate): %s"
          % ", ".join("%s %d in %d qid(s)" % (a, v["count"], v["qids"])
                      for a, v in pre["V6_gold_fact_verbatim"]["arms"].items()))
    summary["V7_T_SEAL_verified"] = pre["V7_T_SEAL_verified"]
    print("V7 T_SEAL verified (%d files) -> PASS" % pre["V7_T_SEAL_verified"]["n_files"])
    summary["V8_prompt_content"] = pre["V8_prompt_content"]
    print("V8 prompt-content check: %d facts prompts -> PASS" % pre["V8_prompt_content"]["facts_prompts_checked"])
    summary["V9_latin_fraction"] = pre["V9_latin_fraction"]
    print("V9 Latin fraction of injected facts %.4f" % pre["V9_latin_fraction"]["fraction"])

    # ---- readings ----
    print("\n" + hdr)
    print("D10 READINGS (freeze §6, paired exact McNemar two-sided)%s"
          % ("   (TEST — NOT A RESULT)" if is_test else ""))
    print(hdr)
    if not v1_ok:
        print("*** V1 VOID: numbers below are for diagnosis only, not results.")

    # R1
    r1p = pair(results["TF"], results["A_r2"], "TF", "A_r2")
    d1, p1 = r1p.get("delta_pp"), r1p.get("p_mcnemar_exact_two_sided")
    if d1 is None:
        r1v = "n/a"
    elif d1 >= GATE_PP and p1 is not None and p1 < ALPHA:
        r1v = "GO"
    elif d1 >= COND_PP:
        r1v = "CONDITIONAL"
    elif d1 <= HARM_PP:
        r1v = "HARM"
    else:
        r1v = "NO-GO"
    r1 = OrderedDict(r1p)
    r1.update([("gate_pp", GATE_PP), ("conditional_pp", COND_PP), ("harm_pp", HARM_PP), ("alpha", ALPHA),
               ("verdict", r1v if not is_test else na)])
    summary["R1_TF_vs_A_r2"] = r1
    print("R1 TF - A_r2 = %s pp, wins %s losses %s, p = %s -> %s"
          % (d1, r1p.get("wins_TF"), r1p.get("losses_TF"), p1, r1["verdict"]))

    # R2 (T-direct from the sealed teacher file)
    td = [OrderedDict([("qid", r["qid"]), ("gold", r["gold"]), ("pred", r["pred"])]) for r in t_direct]
    r2p = pair(results["TF"], td, "TF", "T_direct")
    a_tf, a_td, a_ar = acc(results["TF"]), acc(td), acc(results["A_r2"])
    if a_tf >= a_td:
        r2o = "(a) TF >= T-direct: combination beats both parts"
    elif a_ar + GATE_PP <= a_tf < a_td:
        r2o = "(b) A_r2+8 <= TF < T-direct: injection works, teacher is the ceiling -> D11 distillation"
    else:
        r2o = "(c) TF below gate: see §7"
    r2 = OrderedDict(r2p)
    r2["T_direct_unparsed_rate"] = score(td, "pred")["unparsed_rate"]
    r2["outcome"] = r2o if not is_test else na
    if r2["T_direct_unparsed_rate"] > R2_TDIRECT_UNPARSED_MAX:
        pq = {r["qid"] for r in td if r["pred"] is not None}
        r2["parsed_only"] = pair([r for r in results["TF"] if r["qid"] in pq],
                                 [r for r in td if r["qid"] in pq], "TF", "T_direct")
        r2["parsed_only"]["note"] = "T-direct unparsed > %.0f%%: R2 also on rows T-direct parsed" % R2_TDIRECT_UNPARSED_MAX
    summary["R2_TF_vs_T_direct"] = r2
    print("R2 TF %s%% vs T-direct %s%% (delta %s pp), wins %s losses %s, p = %s -> %s"
          % (a_tf, a_td, r2p.get("delta_pp"), r2p.get("wins_TF"), r2p.get("losses_TF"),
             r2p.get("p_mcnemar_exact_two_sided"), r2["outcome"]))

    # R3 controls
    r3w = pair(results["TF_wrong"], results["A_r2"], "TF_wrong", "A_r2")
    fmt_flag = abs(r3w["delta_pp"]) > R3_FORMAT_PP
    r3n = pair(results["TF_noq"], results["A_r2"], "TF_noq", "A_r2")
    leak_flag = acc(results["TF_noq"]) >= a_ar
    r3 = OrderedDict([
        ("TF_wrong_vs_A_r2", r3w), ("format_effect_limit_pp", R3_FORMAT_PP),
        ("format_effect_flag", fmt_flag if not is_test else None),
        ("TF_minus_TF_wrong_content_effect", pair(results["TF"], results["TF_wrong"], "TF", "TF_wrong")),
        ("TF_noq_vs_A_r2", r3n), ("leak_flag", leak_flag if not is_test else None),
        ("TF_noq_parsed_only", parsed_only(results["TF_noq"])),
        ("reference_D1_arm_G", OrderedDict([("overall", D1_ARM_G_OVERALL), ("parsed_only", D1_ARM_G_PARSED_ONLY)]))])
    summary["R3_controls"] = r3
    print("R3 TF_wrong - A_r2 = %s pp -> %s" % (r3w["delta_pp"], "FORMAT-EFFECT FLAG" if fmt_flag else "ok (<= 4.0)"))
    print("R3 TF_noq %s%% vs A_r2 %s%% -> %s; TF_noq parsed-only %s%% (D1 arm G: %.1f overall, %.1f parsed-only)"
          % (acc(results["TF_noq"]), a_ar, "LEAK FLAG" if leak_flag else "ok (< A_r2)",
             r3["TF_noq_parsed_only"]["accuracy"], D1_ARM_G_OVERALL, D1_ARM_G_PARSED_ONLY))

    # R4 (preflight)
    summary["R4_mechanism"] = pre["R4_mechanism"]
    print("R4 teacher entity reading (a) D6-71 %s%% (ref 21.0%%) -> %s; (b) all %s%%"
          % (pre["R4_mechanism"]["a_D6_71"]["pct"],
             "MECHANISM FLAG" if pre["R4_mechanism"]["mechanism_flag"] else "ok",
             pre["R4_mechanism"]["b_all_rows"]["pct"]))

    # ---- descriptives (never gates) ----
    nf = sorted(len(facts_by[q]) for q in facts_by)
    ar_by = {r["qid"]: r for r in results["A_r2"]}
    split = OrderedDict()
    for ok in (True, False):
        sel = [r for r in results["TF"] if r["source_parse_ok"] is ok]
        w = sum(1 for r in sel if r["correct"] and not ar_by[r["qid"]]["correct"])
        l = sum(1 for r in sel if (not r["correct"]) and ar_by[r["qid"]]["correct"])
        split["parse_ok_%s" % ok] = OrderedDict([("n", len(sel)), ("TF_wins", w), ("TF_losses", l)])
    desc = OrderedDict([
        ("predicted_yes_rate", OrderedDict((a, score(results[a], "pred")["predicted_yes_rate"]) for a in ARMS)),
        ("predicted_yes_reference_D8", D8_PRED_YES_REF),
        ("facts_per_question", OrderedDict([("min", nf[0]), ("median", nf[len(nf) // 2]), ("max", nf[-1]),
                                            ("empty", sum(1 for x in nf if x == 0))])),
        ("teacher_parse_failures", pre["V4_teacher_parse"]["n"] - pre["V4_teacher_parse"]["parse_ok"]),
        ("V5_removed_lines", pre["V5_verdict_filter"]["removed_lines"]),
        ("TF_wins_losses_by_parse_ok", split),
        ("empty_facts_rows", OrderedDict((a, pre["layouts"][a]["empty_facts"]) for a in ARMS)),
        ("truncation_rate", OrderedDict((a, round(100.0 * sum(1 for r in results[a] if r["truncated"]) / len(sub), 2))
                                        for a in ARMS)),
    ])
    summary["descriptive"] = desc

    # seed-8 sample of 30 TF facts for the hand check
    tf_facts = [(q, f) for q in facts_by for f in facts_by[q]]
    rng = random.Random(SAMPLE_SEED)
    smp = rng.sample(tf_facts, min(SAMPLE_N, len(tf_facts))) if tf_facts else []
    q_by = {r["qid"]: r for r in sub}
    sample_rows = [OrderedDict([("qid", q), ("question_ur", q_by[q]["question_ur"]),
                                ("entities", teacher_by[q]["entities"]), ("fact", f),
                                ("truth", None), ("relevance", None)]) for q, f in smp]
    write_jsonl(os.path.join(OUT_DIR, "tf_facts_sample30.jsonl"), sample_rows)
    summary["hand_check_sample"] = OrderedDict([("seed", SAMPLE_SEED), ("n", len(sample_rows)),
                                                ("file", os.path.join(OUT_DIR, "tf_facts_sample30.jsonl")),
                                                ("status", "PENDING hand check")])

    # ---- §7 decision tree, pre-declared ----
    flags = OrderedDict([
        ("DETERMINISM", not v2_pass),
        ("FORMAT", pre["V4_teacher_parse"]["format_flag"]),
        ("TEACHER_ANSWERING", pre["V5_verdict_filter"]["teacher_answering_flag"]),
        ("FORMAT_EFFECT", fmt_flag), ("LEAK", leak_flag),
        ("MECHANISM", pre["R4_mechanism"]["mechanism_flag"]),
        ("V1_VOID", not v1_ok)])
    if is_test:
        branch = na
    elif r1v == "GO" and r2o.startswith("(a)"):
        branch = "R1 GO and R2(a): write the eval458 freeze; one run on eval458, everything frozen"
    elif r1v == "GO" and r2o.startswith("(b)"):
        branch = "R1 GO and R2(b): write the D11 freeze (teacher facts on the 1632-question training pool -> QLoRA); eval458 once, at the end"
    elif r1v == "GO":
        branch = "R1 GO with R2(c): not a pre-named combination; report R2 as computed"
    elif r1v == "CONDITIONAL":
        branch = "R1 CONDITIONAL: exactly ONE fix iteration, declared in an amendment before it runs; then one FULL re-run"
    else:
        branch = "R1 %s: fact injection is closed as a family (retrieval, self, teacher all tested)" % r1v
    notes = []
    if fmt_flag and not is_test:
        notes.append("FORMAT-EFFECT FLAG: TF - TF_wrong is reported as the content effect; the method claim is reduced to that difference")
    if leak_flag and not is_test:
        notes.append("LEAK FLAG: TF is reported as teacher-answer transfer, not fact injection, whatever R1 says")
    if pre["R4_mechanism"]["mechanism_flag"] and not is_test:
        notes.append("MECHANISM FLAG: H1's reason is not supported even if R1 passes")
    summary["flags"] = flags
    summary["decision_tree"] = OrderedDict([("branch", branch), ("overrides", notes)])
    print("\n§7 branch: %s" % branch)
    for n_ in notes:
        print("§7 override: %s" % n_)

    # ---- accuracy table, only now ----
    n_yes = sum(1 for r in sub if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(sub) - n_yes) / len(sub)
    table = OrderedDict()
    print("\n" + hdr)
    print("ACCURACY TABLE%s   floor %.2f%%" % ("   (TEST — NOT A RESULT)" if is_test else "", floor))
    print(hdr)
    print("%-10s %-36s %5s %8s %9s %9s %8s" % ("arm", "facts", "n", "acc%", "unpars%", "SEC acc%", "yes%"))
    label = {"A_r2": "none (re-run, floor)", "TF": "teacher facts, own qid",
             "TF_wrong": "teacher facts, derangement(qid)", "TF_noq": "teacher facts, no question",
             "T_direct": "teacher alone on the student prompt"}
    for arm, recs in list(results.items()) + [("T_direct", t_direct)]:
        pri = score(recs, "pred")
        se = score(recs, "pred_secondary")
        table[arm] = OrderedDict([
            ("n", pri["n"]), ("accuracy", pri["accuracy"]), ("correct", pri["correct"]),
            ("unparsed_rate", pri["unparsed_rate"]),
            ("truncation_rate", round(100.0 * sum(1 for r in recs if r.get("truncated")) / pri["n"], 2)),
            ("predicted_yes_rate", pri["predicted_yes_rate"]),
            ("accuracy_secondary", se["accuracy"]), ("unparsed_rate_secondary", se["unparsed_rate"])])
        print("%-10s %-36s %5d %8.2f %9.2f %9.2f %8.2f"
              % (arm, label[arm], pri["n"], pri["accuracy"], pri["unparsed_rate"], se["accuracy"],
                 pri["predicted_yes_rate"]))
    summary["floor"] = round(floor, 2)
    summary["accuracy_table"] = table
    summary["R4_rows"] = OrderedDict([("a_D6_71", r4a_rows), ("b_all_rows", r4b_rows)])
    write_json(os.path.join(OUT_DIR, "d10_summary.json"), summary)

    # ---------------- SEAL ----------------
    files = OrderedDict()
    for p in WRITTEN:
        with open(p, "rb") as f:
            b = f.read()
        files[p] = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)), ("lines", b.count(b"\n"))])
    seal = OrderedDict([
        ("job", "S"), ("mode", "TEST" if is_test else "FULL"), ("slurm_job_id", job_id),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("script_md5", pre["script"]["md5"]), ("freeze_md5", pre["freeze"]["md5"]),
        ("teacher_seal_files", pre["teacher_seal"]["files"]),
        ("files", files)])
    write_json(os.path.join(OUT_DIR, "S_SEAL.json"), seal)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD10 JOB S COMPLETE. Sealed %d files." % len(files))


if __name__ == "__main__":
    main()
