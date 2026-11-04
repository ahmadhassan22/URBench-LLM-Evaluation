#!/usr/bin/env python3
"""
d11o_student.py  —  D11-O Job S (student, HF transformers, nf4, batch size 1)

Contract: docs/EFBPT_D11O_ORACLE_READING_FREEZE.md (FROZEN, MD5
ebd638eb4b4e48cde00202c8488df124) §2, §4, §5, §6, §7, on top of the D10
teacher freeze docs/EFBPT_D10_TEACHER_FREEZE.md §2, §4 (student model,
decoding, layouts; unchanged). Copy of d10v2_student.py with ONLY the D11-O
changes:

  rows      d1_eval_arms.load_rows (gold urbench_facts PRESENT, for the GF
            ceiling arm) joined with d4_extract_facts.load_rows (required gold
            titles, for R4); same file, same order, same gold
  arms      GF   Urdu question + gold urbench_facts, layout B (= D1 arm B)
            TO1  Urdu question + facts of the v1 oracle-reading teacher
            TO2  Urdu question + facts of the v2 oracle-reading teacher
            A_r2 the SEALED D10 floor file, reused after V2' (freeze §2, §4);
                 re-generated in this job only if V2' fails
            TF_v1, TF_v2  sealed D10 / D10v2 TF files, references only
  V2'       20 A_r2 prompts generated once at batch size 1 must equal the 20
            sealed A_r2 generations byte for byte
  leak      question_en and evidence ids FATAL in every arm; gold facts FATAL
            in TO1/TO2 (all hits are printed first, then the job dies);
            PERMITTED in GF and recorded (the ceiling arm)
  V7        both teacher seals; V8 on GF, TO1, TO2
  readings  O1–O6 (freeze §5), descriptive, no GO gate; the §6 design branch
            is printed from max(O2)
  outputs   outputs/efbpt/d11o/student/ (d11o_arm{GF,TO1,TO2}.jsonl,
            d11o_armA_r2.jsonl only if re-generated, d11o_summary.json,
            S_START.json, S_SEAL.json); TEST under outputs/efbpt/d11o/_test/student/

Everything frozen is IMPORTED, never re-implemented:
  model path, SYSTEM_MESSAGE, instruction (+MD5), extractor  <- efbpt_eval_dev200
  user layouts arm "A"/"B", facts header, rows WITH facts    <- d1_eval_arms
  secondary (Devanagari) extractor                           <- d1_score_dual
  norm(), score()                                            <- d3_oracle_retrieval
  rows with required titles, generate_all                    <- d4_extract_facts
  paired exact McNemar                                       <- d5_balanced_extract
  leak guard, needles, Latin fraction, md5/json helpers      <- d8_reachable_ceiling

Modes:
  --preflight-only   CPU, no model. Writes NOTHING. Dies on a missing T_SEAL.
  --limit N          TEST: first N rows; reads outputs/efbpt/d11o/_test/teacher_v{1,2}/
                     and writes ONLY under outputs/efbpt/d11o/_test/student/.
  (no flags)         FULL.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d11o_student.py --preflight-only
  python eval/error_analysis_tests/efbpt/d11o_student.py --limit 5
  python eval/error_analysis_tests/efbpt/d11o_student.py
"""

import argparse
import json
import os
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
from d1_eval_arms import build_user, FACTS_HEADER, load_rows as load_rows_with_facts   # noqa: E402
from d1_score_dual import extract_answer_secondary  # noqa: E402
from d3_oracle_retrieval import norm, score         # noqa: E402
from d4_extract_facts import load_rows as load_rows_required, generate_all  # noqa: E402
from d5_balanced_extract import mcnemar             # noqa: E402
from d8_reachable_ceiling import (                  # noqa: E402  same guard as D8
    leak_guard, load_leak_needles, latin_fraction,
    md5_bytes, md5_file, md5_text, read_jsonl, pkg_versions,
)

# ----------------------------------------------------------------------------
# Frozen constants
# ----------------------------------------------------------------------------

D11O_FREEZE_PATH = "docs/EFBPT_D11O_ORACLE_READING_FREEZE.md"
D11O_FREEZE_MD5 = "ebd638eb4b4e48cde00202c8488df124"
FREEZE_PATH = "docs/EFBPT_D10_TEACHER_FREEZE.md"
FREEZE_MD5 = "b3a822d8199034c7fe58d76cf5cc428f"

T_DIRS_FULL = OrderedDict([("v1", "outputs/efbpt/d11o/teacher_v1"), ("v2", "outputs/efbpt/d11o/teacher_v2")])
T_DIRS_TEST = OrderedDict([("v1", "outputs/efbpt/d11o/_test/teacher_v1"), ("v2", "outputs/efbpt/d11o/_test/teacher_v2")])
OUT_DIR_FULL = "outputs/efbpt/d11o/student"
OUT_DIR_TEST = "outputs/efbpt/d11o/_test/student"
FORBIDDEN_WRITE_DIRS = ("outputs/efbpt/d1", "outputs/efbpt/d4", "outputs/efbpt/d6",
                        "outputs/efbpt/d8", "outputs/efbpt/d10", "outputs/efbpt/d10v2",
                        "outputs/efbpt/d11o/teacher_v1", "outputs/efbpt/d11o/teacher_v2",
                        "outputs/efbpt/d11o/_test/teacher_v1", "outputs/efbpt/d11o/_test/teacher_v2")

# freeze §2: sealed references, pinned MD5s; abort on any mismatch
SEALED = OrderedDict([
    ("A_r2", ("outputs/efbpt/d10/student/d10_armA_r2.jsonl", "7ccec942c8f0eee08156d258baaf806d")),
    ("TF_v1", ("outputs/efbpt/d10/student/d10_armTF.jsonl", "c54b7553baa5ffa77204bed1dab37250")),
    ("TF_v2", ("outputs/efbpt/d10v2/student/d10_armTF.jsonl", "1a3cdba1f26ccca4b010a00676197f0a")),
])

# sealed D6 outputs (read-only): the 71 qids and the G4 reference (R4 / O5)
D6_E1_PATH = "outputs/efbpt/d6/d6_armE1.jsonl"
D6_SUMMARY_PATH = "outputs/efbpt/d6/d6_summary.json"
D6_N = 71
D6_G4_INTERSECT = 42
D6_G4_GOLD_TOTAL = 200
R4_URDU_INPUT_REF = OrderedDict([("D10_v1_a_D6_71", 34.0), ("D10v2_a_D6_71", 36.0)])

BATCH_SIZE = 1                     # D10 §2, §4: BATCH SIZE 1
CONTEXT_WINDOW = 40960
V2_PROBE_N_FULL = 20               # freeze §4 V2'
V2_PROBE_N_TEST = 5
V4_PARSE_MIN_PCT = 95.0
V5_REMOVED_MAX = 100
# freeze §6
O2_READING_WALL_PP = 9.0
O2_KNOWLEDGE_WALL_PP = 6.0
# D1 references (batch size 8)
D1_ARM_B = 78.0
D1_ARM_A = 57.5

GEN_ARMS = ("GF", "TO1", "TO2")                   # generated in this job
TEACHER_OF = OrderedDict([("TO1", "v1"), ("TO2", "v2")])
TF_OF = OrderedDict([("TO1", "TF_v1"), ("TO2", "TF_v2")])

# ----------------------------------------------------------------------------
# Output helpers (same layout as D10v2; writable root depends on the mode)
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
# V7: a D11-O teacher seal
# ----------------------------------------------------------------------------

def verify_teacher_seal(t_dir, version):
    seal_path = os.path.join(t_dir, "T_SEAL.json")
    if not os.path.exists(seal_path):
        die("T_SEAL.json missing: %s (run Job T %s first)" % (seal_path, version))
    seal = json.load(open(seal_path, encoding="utf-8"))
    if seal.get("d11o_freeze_md5") != D11O_FREEZE_MD5:
        die("T_SEAL d11o_freeze_md5 %s != %s" % (seal.get("d11o_freeze_md5"), D11O_FREEZE_MD5))
    if seal.get("freeze_md5") != FREEZE_MD5:
        die("T_SEAL freeze_md5 %s != %s" % (seal.get("freeze_md5"), FREEZE_MD5))
    if seal.get("version") != version:
        die("T_SEAL in %s is version %r, expected %r" % (t_dir, seal.get("version"), version))
    checked = OrderedDict()
    for p, want in seal["files"].items():
        if not os.path.exists(p):
            die("V7: sealed file missing: %s" % p)
        b = open(p, "rb").read()
        got = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)), ("lines", b.count(b"\n"))])
        if got["md5"] != want["md5"] or got["bytes"] != want["bytes"] or got["lines"] != want["lines"]:
            die("V7: %s differs from T_SEAL: sealed %s, on disk %s" % (p, dict(want), dict(got)))
        checked[p] = got["md5"]
    need = {"t_facts.jsonl", "T_START.json"}
    have = {os.path.basename(p) for p in checked}
    if not need <= have:
        die("V7: T_SEAL does not cover %s" % sorted(need - have))
    print("[V7] T_SEAL %s: %d files verified (md5, bytes, lines) -> PASS" % (seal_path, len(checked)))
    return seal, checked


def verify_sealed_refs():
    """freeze §2: the sealed D10 / D10v2 student files, by pinned MD5."""
    out = OrderedDict()
    for name, (path, want) in SEALED.items():
        if not os.path.exists(path):
            die("sealed reference missing: %s" % path)
        got = md5_file(path)
        if got != want:
            die("sealed %s MD5 %s != pinned %s" % (name, got, want))
        out[name] = OrderedDict([("path", path), ("md5", got)])
    print("[sealed] %s verified by pinned MD5" % ", ".join(SEALED))
    return out


# ----------------------------------------------------------------------------
# V8: the fact lines inside a rendered prompt (A1.6 cut at <|im_end|>)
# ----------------------------------------------------------------------------

def facts_in_prompt(prompt):
    key = FACTS_HEADER + ":\n"
    n = prompt.count(key)
    if n == 0:
        return 0, []
    rest = prompt[prompt.index(key) + len(key):].split("<|im_end|>")[0].split("\n")
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

def preflight(limit, t_dirs):
    print("=" * 78)
    print("D11-O JOB S PREFLIGHT (freeze §2, §4; D10 freeze §4)")
    print("=" * 78, flush=True)

    for label, path, want in (("d11o", D11O_FREEZE_PATH, D11O_FREEZE_MD5), ("d10", FREEZE_PATH, FREEZE_MD5)):
        if not os.path.exists(path):
            die("freeze file missing: " + path)
        got = md5_file(path)
        if got != want:
            die("%s freeze MD5 mismatch: expected %s, got %s" % (label, want, got))
        print("[md5] %-7s %s  %s" % (label, got, path))
    script_md5 = md5_file(os.path.abspath(__file__))
    print("[md5] script  %s  %s" % (script_md5, os.path.relpath(__file__)))

    instruction = load_instruction()              # MD5 asserted inside
    print("[ok] Urdu instruction MD5 %s verified" % INSTRUCTION_MD5)

    # ---- rows: d1 loader (facts) joined with d4 loader (required titles) ----
    rows_f = load_rows_with_facts()               # asserts 200, non-empty urbench_facts
    rows_r = load_rows_required()                 # asserts 200
    if [r["qid"] for r in rows_f] != [r["qid"] for r in rows_r]:
        die("d1 and d4 loaders disagree on the DEV200 qid order")
    rows = []
    for a, b in zip(rows_f, rows_r):
        if a["gold"] != b["gold"] or a["question_ur"] != b["question_ur"] or a["question_en"] != b["question_en"]:
            die("qid %s: d1 and d4 loaders disagree on gold / question" % a["qid"])
        rows.append(OrderedDict([("qid", a["qid"]), ("question_ur", a["question_ur"]),
                                 ("question_en", a["question_en"]), ("gold", a["gold"]),
                                 ("facts", list(a["facts"])), ("required", list(b["required"]))]))
    qids = [r["qid"] for r in rows]
    if len(set(qids)) != 200:
        die("DEV200 qids are not unique")
    print("[ok] DEV200: 200 rows, 200 unique qids, gold urbench_facts on every row (GF ceiling arm), "
          "required titles for R4; facts per row min=%d median=%d max=%d"
          % (min(len(r["facts"]) for r in rows), sorted(len(r["facts"]) for r in rows)[100],
             max(len(r["facts"]) for r in rows)))
    sub = rows[:limit] if limit else rows
    sub_qids = [r["qid"] for r in sub]
    sub_set = set(sub_qids)

    # ---- sealed references (freeze §2) ----
    sealed_md5 = verify_sealed_refs()
    sealed = OrderedDict()
    for name, (path, _) in SEALED.items():
        recs = [e for e in read_jsonl(path) if e["qid"] in sub_set]
        if [e["qid"] for e in recs] != sub_qids:
            die("sealed %s qids != this run's rows in order" % name)
        for e, r in zip(recs, sub):
            if e["gold"] != r["gold"]:
                die("sealed %s qid %s: gold %r != loader gold %r" % (name, e["qid"], e["gold"], r["gold"]))
        sealed[name] = recs
    print("[sealed] %d rows each: A_r2 %.2f%%, TF_v1 %.2f%%, TF_v2 %.2f%% (on these rows)"
          % (len(sub), score(sealed["A_r2"], "pred")["accuracy"], score(sealed["TF_v1"], "pred")["accuracy"],
             score(sealed["TF_v2"], "pred")["accuracy"]))

    # ---- prompts (tokenizer only): A_r2 (must equal the sealed prompts) and GF ----
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

    def no_facts(row):
        return OrderedDict((k, v) for k, v in row.items() if k != "facts")

    prompts = OrderedDict()     # arm -> list of (prompt, layout, facts, source)
    prompts["A_r2"] = [(chat(SYSTEM_MESSAGE, build_user(instruction, no_facts(r), None, "A")), "A", [], None) for r in sub]
    for (p, _, _, _), e in zip(prompts["A_r2"], sealed["A_r2"]):
        if p != e["prompt"]:
            die("A_r2 prompt rebuilt here differs from the sealed D10 prompt for qid %s" % e["qid"])
    print("[ok] A_r2 prompts rebuilt here are byte-identical to the %d sealed D10 prompts" % len(sub))
    gf_users = [build_user(instruction, r, None, "B") for r in sub]      # = D1 arm B user message
    prompts["GF"] = [(chat(SYSTEM_MESSAGE, u), "B", [f.strip() for f in r["facts"]], "gold") for u, r in zip(gf_users, sub)]
    print("[GF] user message row 0 (D1 arm B layout) md5 %s; %d gold facts" % (md5_text(gf_users[0]), len(sub[0]["facts"])))

    # ---- V7 and the two teacher outputs (dies here if a teacher has not run) ----
    seals, seal_files, t_starts, facts_by, teacher_by = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for version, t_dir in t_dirs.items():
        seal, files = verify_teacher_seal(t_dir, version)
        seals[version], seal_files[version] = seal, files
        t_start = json.load(open(os.path.join(t_dir, "T_START.json"), encoding="utf-8"))
        if t_start.get("d11o_freeze", {}).get("md5") != D11O_FREEZE_MD5:
            die("T_START %s: d11o_freeze md5 differs" % version)
        if t_start.get("freeze", {}).get("md5") != FREEZE_MD5:
            die("T_START %s: freeze md5 differs" % version)
        if t_start.get("urdu_instruction", {}).get("md5") != INSTRUCTION_MD5:
            die("T_START %s: instruction md5 differs" % version)
        if t_start.get("prompts", {}).get("version") != version:
            die("T_START %s: prompts.version is %r" % (version, t_start.get("prompts", {}).get("version")))
        t_starts[version] = t_start
        t_facts = read_jsonl(os.path.join(t_dir, "t_facts.jsonl"))
        if [e["qid"] for e in t_facts] != sub_qids:
            die("%s t_facts.jsonl qids (%d) != this run's rows (%d) in order" % (version, len(t_facts), len(sub_qids)))
        for e in t_facts:
            if e.get("teacher_version") != version:
                die("%s t_facts qid %s: teacher_version %r" % (version, e["qid"], e.get("teacher_version")))
            fl = e["facts"]
            if len(fl) > 6 or any((not isinstance(f, str)) or f != f.strip() or not f for f in fl):
                die("%s t_facts qid %s: facts are not <= 6 clean non-empty strings" % (version, e["qid"]))
            if e["parse_ok"] is False and fl:
                die("%s t_facts qid %s: parse_ok false but facts present" % (version, e["qid"]))
            for k in ("question_en", "unknown_terms", "sub_questions", "entities", "repaired"):
                if k not in e:
                    die("%s t_facts qid %s: row lacks key %r" % (version, e["qid"], k))
        facts_by[version] = OrderedDict((e["qid"], list(e["facts"])) for e in t_facts)
        teacher_by[version] = OrderedDict((e["qid"], e) for e in t_facts)
        n_ok = sum(1 for e in t_facts if e["parse_ok"])
        v5 = sum(len(e["removed"]) for e in t_facts)
        print("[ok] teacher %s: %d t_facts rows; parse_ok %d/%d = %.1f%% -> %s; repaired %d; V5 removed %d -> %s"
              % (version, len(t_facts), n_ok, len(t_facts), 100.0 * n_ok / len(t_facts),
                 "ok" if 100.0 * n_ok / len(t_facts) >= V4_PARSE_MIN_PCT else "FORMAT FLAG",
                 sum(1 for e in t_facts if e["repaired"]), v5,
                 "ok" if v5 <= V5_REMOVED_MAX else "TEACHER-ANSWERING FLAG"))

    for arm, version in TEACHER_OF.items():
        prompts[arm] = []
        for r in sub:
            facts = facts_by[version][r["qid"]]
            if facts:
                prompts[arm].append((chat(SYSTEM_MESSAGE, build_user(instruction, dict(no_facts(r), facts=list(facts)), None, "B")),
                                     "B", facts, version))
            else:
                prompts[arm].append((chat(SYSTEM_MESSAGE, build_user(instruction, no_facts(r), None, "A")), "A", [], version))

    for arm in ("A_r2",) + GEN_ARMS:
        ps = [p for p, _, _, _ in prompts[arm]]
        tl = sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in ps)
        lays = Counter(l for _, l, _, _ in prompts[arm])
        n_empty = sum(1 for _, _, fl, _ in prompts[arm] if not fl)
        print("[len] %-5s n=%d tokens min=%d median=%d max=%d  layouts %s  empty-facts %d"
              % (arm, len(ps), tl[0], tl[len(tl) // 2], tl[-1], dict(lays), n_empty))
        if tl[-1] + MAX_NEW_TOKENS > CONTEXT_WINDOW:
            die("%s prompts would exceed the context window" % arm)

    # ---- leak guard (freeze §4) ----
    needles = load_leak_needles()
    leak = OrderedDict()
    # A_r2 (sealed prompts) and GF: question_en / evidence ids fatal; gold facts
    # fatal in A_r2, PERMITTED in GF (the ceiling arm) and recorded
    leak_guard("A_r2", [p for p, _, _, _ in prompts["A_r2"]], needles, fatal_gold_facts=True)
    gf_hits = leak_guard("GF", [p for p, _, _, _ in prompts["GF"]], needles, fatal_gold_facts=False)
    leak["GF"] = OrderedDict([("gold_facts", "PERMITTED (ceiling arm, freeze §4)"),
                              ("gold_fact_substrings", len(gf_hits)),
                              ("prompts_with_gold_facts", len({i for i, _ in gf_hits})),
                              ("n_prompts", len(sub))])
    print("[leak-guard] GF: %d gold-fact substrings in %d/%d prompts — PERMITTED, recorded (ceiling arm)"
          % (len(gf_hits), len({i for i, _ in gf_hits}), len(sub)))
    # TO1 / TO2: gold facts FATAL (the teacher never saw them). All hits are
    # collected and printed first (O5 V6 report), then the job dies on any hit.
    for arm in TEACHER_OF:
        hits = leak_guard(arm, [p for p, _, _, _ in prompts[arm]], needles, fatal_gold_facts=False)
        by_q = OrderedDict()
        for i, s in hits:
            by_q.setdefault(sub[i]["qid"], []).append(s)
        leak[arm] = OrderedDict([("gold_facts", "FATAL (freeze §4)"), ("count", len(hits)), ("qids", len(by_q)),
                                 ("matches", [OrderedDict([("qid", q), ("fact", f)]) for q, fs in by_q.items() for f in fs])])
        print("[V6] %s verbatim gold-fact substrings: %d in %d qid(s)" % (arm, len(hits), len(by_q)))
        for q, fs in by_q.items():
            for f in fs:
                print("      %s  %r" % (q, f))
        if hits:
            die("LEAK (freeze §4): %s prompts contain %d verbatim gold urbench_facts (%d qids, listed above); "
                "the teacher never saw them, so a verbatim match is fatal in TO1/TO2" % (arm, len(hits), len(by_q)))

    # ---- V8: fact lines in every facts prompt == their source list ----
    v8_checked = 0
    for arm in GEN_ARMS:
        for r, (p, lay, fl, src) in zip(sub, prompts[arm]):
            n_hdr, found = facts_in_prompt(p)
            if lay == "A":
                if n_hdr != 0 or fl:
                    die("V8: %s qid %s: arm-A layout with a facts header or facts" % (arm, r["qid"]))
                continue
            if n_hdr != 1:
                die("V8: %s qid %s: facts header appears %d times" % (arm, r["qid"], n_hdr))
            if found != fl:
                die("V8: %s qid %s (source %s): prompt facts %r != source facts %r" % (arm, r["qid"], src, found, fl))
            v8_checked += 1
    print("[V8] prompt-content check: %d facts prompts equal their source facts (GF gold, TO1/TO2 sealed post-T2) -> PASS" % v8_checked)

    # ---- the teachers' own question_en / sub_questions never enter a student prompt ----
    t_texts = sorted({t for v in teacher_by for e in teacher_by[v].values()
                      for t in [e["question_en"]] + list(e["sub_questions"]) if t})
    for arm in ("A_r2",) + GEN_ARMS:
        for r, (p, _, _, _) in zip(sub, prompts[arm]):
            for t in t_texts:
                if t in p:
                    die("LEAK: %s qid %s prompt contains teacher text %r" % (arm, r["qid"], t))
    print("[ok] teacher question_en / sub_questions (%d strings) absent from all %d student prompts"
          % (len(t_texts), sum(len(prompts[a]) for a in ("A_r2",) + GEN_ARMS)))

    # ---- V9 ----
    v9 = OrderedDict()
    for arm in GEN_ARMS:
        fr, letters, latin = latin_fraction([f for _, _, fl, _ in prompts[arm] for f in fl])
        v9[arm] = OrderedDict([("fraction", round(fr, 4)), ("letters", letters), ("latin_letters", latin),
                               ("n_facts", sum(len(fl) for _, _, fl, _ in prompts[arm]))])
        print("[V9] %s Latin fraction of injected facts %.4f (%d/%d letters, %d facts)"
              % (arm, fr, latin, letters, v9[arm]["n_facts"]))

    # ---- O5: R4 entity reading for TO1 / TO2 (title-level, D6 G4 way) ----
    d6_qids = [r["qid"] for r in read_jsonl(D6_E1_PATH)]
    if len(d6_qids) != D6_N or len(set(d6_qids)) != D6_N or not set(d6_qids) <= set(qids):
        die("D6 arm E1 does not hold %d unique DEV200 qids" % D6_N)
    d6s = json.load(open(D6_SUMMARY_PATH, encoding="utf-8"))["gold_overlap"]
    if d6s.get("intersect") != D6_G4_INTERSECT or d6s.get("gold_total") != D6_G4_GOLD_TOTAL:
        die("D6 summary gold_overlap %s != reference %d/%d" % (d6s, D6_G4_INTERSECT, D6_G4_GOLD_TOTAL))
    req_by = {r["qid"]: {norm(t) for t in r["required"]} for r in rows}
    if limit is None and sum(len(req_by[q]) for q in d6_qids) != D6_G4_GOLD_TOTAL:
        die("required titles over the 71 D6 qids != %d" % D6_G4_GOLD_TOTAL)

    def r4_on(version, qset, field):
        inter = gold = 0
        per = []
        for q in sub_qids:
            if q not in qset:
                continue
            e = teacher_by[version][q]
            if field == "entities":
                hit = sorted(req_by[q] & {norm(x) for x in e["entities"]})
            else:
                qen = norm(e["question_en"])
                hit = sorted(t for t in req_by[q] if t and t in qen)
            inter += len(hit)
            gold += len(req_by[q])
            per.append(OrderedDict([("qid", q), ("gold_titles", sorted(req_by[q])),
                                    (field, e[field]), ("hits", hit)]))
        return OrderedDict([("n_qids", len(per)), ("intersect", inter), ("gold_total", gold),
                            ("pct", round(100.0 * inter / gold, 2) if gold else None)]), per

    r4 = OrderedDict()
    r4_rows = OrderedDict()
    for arm, version in TEACHER_OF.items():
        a, a_rows = r4_on(version, set(d6_qids), "entities")
        b, b_rows = r4_on(version, sub_set, "entities")
        r4[arm] = OrderedDict([("a_D6_71", a), ("b_all_rows", b)])
        r4_rows[arm] = OrderedDict([("a_D6_71", a_rows), ("b_all_rows", b_rows)])
        print("[R4/O5] %s (%s teacher, English input): (a) D6-71 %s/%s = %s%%; (b) all rows %s/%s = %s%%  "
              "(Urdu-input references: D10 v1 34.0%%, D10v2 36.0%%; 14B D6 G4 21.0%%)"
              % (arm, version, a["intersect"], a["gold_total"], a["pct"], b["intersect"], b["gold_total"], b["pct"]))
        if version == "v2":
            ra, _ = r4_on(version, set(d6_qids), "question_en")
            rb, _ = r4_on(version, sub_set, "question_en")
            r4[arm]["restatement_a_D6_71"] = ra
            r4[arm]["restatement_b_all_rows"] = rb
            print("[A3.3/O5] %s restatement-level reading (teacher question_en): (a) %s%%; (b) %s%%" % (arm, ra["pct"], rb["pct"]))

    log_full_prompt("A_r2 row 0, qid %s (sealed, reused)" % sub[0]["qid"], prompts["A_r2"][0][0])
    log_full_prompt("GF row 0, qid %s, layout B (= D1 arm B)" % sub[0]["qid"], prompts["GF"][0][0])
    for arm in TEACHER_OF:
        i = next((i for i, (_, lay, _, _) in enumerate(prompts[arm]) if lay != "A"), 0)
        log_full_prompt("%s row %d, qid %s, layout %s" % (arm, i, sub[i]["qid"], prompts[arm][i][1]), prompts[arm][i][0])

    pre = OrderedDict([
        ("d11o_freeze", OrderedDict([("path", D11O_FREEZE_PATH), ("md5", D11O_FREEZE_MD5)])),
        ("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", FREEZE_MD5)])),
        ("script", OrderedDict([("path", os.path.relpath(__file__)), ("md5", script_md5)])),
        ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("teacher_dirs", t_dirs),
        ("teacher_seals", OrderedDict((v, OrderedDict([("script_md5", seals[v].get("script_md5")),
                                                        ("slurm_job_id", seals[v].get("slurm_job_id")),
                                                        ("t1_parse_ok", seals[v].get("t1_parse_ok")),
                                                        ("t1_repaired_rows", seals[v].get("t1_repaired_rows")),
                                                        ("t2_removed_total", seals[v].get("t2_removed_total")),
                                                        ("files", seal_files[v])])) for v in t_dirs)),
        ("sealed_references", sealed_md5),
        ("GF_user_md5_row0", md5_text(gf_users[0])),
        ("V7_T_SEAL_verified", OrderedDict([("pass", True), ("n_seals", len(seals)),
                                            ("n_files", sum(len(f) for f in seal_files.values()))])),
        ("leak_guard", leak),
        ("V8_prompt_content", OrderedDict([("pass", True), ("facts_prompts_checked", v8_checked)])),
        ("V9_latin_fraction", v9),
        ("R4_O5", OrderedDict([
            ("definition", "title-level, D6 G4 way: gold required titles found (after norm) in the teacher's parsed entities / total required titles"),
            ("reference_14B_D6_G4_pct", 21.0), ("reference_urdu_input_pct", R4_URDU_INPUT_REF),
            ("arms", r4)])),
        ("layouts", OrderedDict((arm, OrderedDict([
            ("counts", OrderedDict(sorted(Counter(l for _, l, _, _ in prompts[arm]).items()))),
            ("empty_facts", sum(1 for _, _, fl, _ in prompts[arm] if not fl))])) for arm in ("A_r2",) + GEN_ARMS)),
    ])
    print("[preflight] PASS", flush=True)
    return instruction, sub, prompts, facts_by, teacher_by, sealed, tok, pre, r4_rows


# ----------------------------------------------------------------------------
# Readings (freeze §5, §6) — pure function of the records, testable offline
# ----------------------------------------------------------------------------

def compute_readings(sub, results, sealed, pre, is_test):
    """results: arm -> records (A_r2, GF, TO1, TO2); sealed: TF_v1, TF_v2 (and
    A_r2) records on the same qids. Returns the summary OrderedDict and prints
    the readings in the frozen order."""
    na = "n/a (TEST)"

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

    def by_gold(x, y, lx, ly):
        out = OrderedDict()
        for g in ("yes", "no"):
            xs = [r for r in x if r["gold"] == g]
            ys = [r for r in y if r["gold"] == g]
            out["gold_%s" % g] = pair(xs, ys, lx, ly)
            out["gold_%s" % g]["pred_yes_rate_%s" % lx] = score(xs, "pred")["predicted_yes_rate"] if xs else None
            out["gold_%s" % g]["pred_yes_rate_%s" % ly] = score(ys, "pred")["predicted_yes_rate"] if ys else None
        return out

    hdr = "=" * 84
    summary = OrderedDict()
    summary["mode"] = "TEST — NOT A RESULT" if is_test else "FULL"
    summary["n"] = len(sub)
    tags = "   (TEST — NOT A RESULT)" if is_test else ""

    print("\n" + hdr)
    print("D11-O VALIDITY (freeze §4; reported, no gate)%s" % tags)
    print(hdr)
    unp = OrderedDict((a, score(results[a], "pred")["unparsed_rate"]) for a in results)
    for name in ("TF_v1", "TF_v2"):
        unp[name] = score(sealed[name], "pred")["unparsed_rate"]
    summary["unparsed_rate_primary"] = unp
    print("V1 unparsed %% per arm: %s (reported; the floor and both TF references are sealed files)" % dict(unp))
    print("V2' determinism probe: %s" % pre["V2_prime"]["verdict"])
    print("V3 leak guards: none fatal in either job -> PASS; GF gold facts permitted: %d substrings in %d prompts"
          % (pre["leak_guard"]["GF"]["gold_fact_substrings"], pre["leak_guard"]["GF"]["prompts_with_gold_facts"]))
    for v, s in pre["teacher_seals"].items():
        print("V4/V5 teacher %s: parse_ok %s/%d, repaired %s, V5 removed %s" % (v, s["t1_parse_ok"], len(sub), s["t1_repaired_rows"], s["t2_removed_total"]))
    print("V6 verbatim gold-fact substrings in TO1/TO2: %d / %d (FATAL rule; the job ran, so none)"
          % (pre["leak_guard"]["TO1"]["count"], pre["leak_guard"]["TO2"]["count"]))
    print("V7 both T_SEALs verified (%d files) -> PASS" % pre["V7_T_SEAL_verified"]["n_files"])
    print("V8 prompt-content check: %d facts prompts -> PASS" % pre["V8_prompt_content"]["facts_prompts_checked"])
    print("V9 Latin fraction: %s" % ", ".join("%s %.4f" % (a, v["fraction"]) for a, v in pre["V9_latin_fraction"].items()))

    print("\n" + hdr)
    print("D11-O READINGS (freeze §5; paired exact McNemar two-sided; descriptive, no GO gate)%s" % tags)
    print(hdr)
    A = results["A_r2"]
    o1 = pair(results["GF"], A, "GF", "A_r2")
    summary["O1_GF_vs_A_r2"] = o1
    print("O1 GF - A_r2 = %s pp (knowledge ceiling at batch size 1), wins %s losses %s, p = %s   [D1 arm B - A at batch 8: +%.1f]"
          % (o1.get("delta_pp"), o1.get("wins_GF"), o1.get("losses_GF"), o1.get("p_mcnemar_exact_two_sided"), D1_ARM_B - D1_ARM_A))
    o2 = OrderedDict()
    for arm in TEACHER_OF:
        o2[arm] = pair(results[arm], A, arm, "A_r2")
        print("O2 %s - A_r2 = %s pp (perfect-reading teacher gain), wins %s losses %s, p = %s"
              % (arm, o2[arm].get("delta_pp"), o2[arm].get("wins_%s" % arm), o2[arm].get("losses_%s" % arm), o2[arm].get("p_mcnemar_exact_two_sided")))
    summary["O2_TO_vs_A_r2"] = o2
    ref = OrderedDict()
    for name in ("TF_v1", "TF_v2"):
        ref[name] = pair(sealed[name], A, name, "A_r2")
        print("   reference %s - A_r2 = %s pp (sealed, Urdu input), wins %s losses %s, p = %s"
              % (name, ref[name].get("delta_pp"), ref[name].get("wins_%s" % name), ref[name].get("losses_%s" % name), ref[name].get("p_mcnemar_exact_two_sided")))
    summary["reference_TF_vs_A_r2"] = ref
    o3 = OrderedDict()
    for arm, tf in TF_OF.items():
        o3[arm] = pair(results[arm], sealed[tf], arm, tf)
        print("O3 %s - %s = %s pp (reading effect: same prompt, English vs Urdu input), wins %s losses %s, p = %s"
              % (arm, tf, o3[arm].get("delta_pp"), o3[arm].get("wins_%s" % arm), o3[arm].get("losses_%s" % arm), o3[arm].get("p_mcnemar_exact_two_sided")))
    summary["O3_TO_vs_TF"] = o3
    o4 = OrderedDict()
    a_gf = acc(results["GF"])
    for arm, tf in TF_OF.items():
        a_to, a_tf = acc(results[arm]), acc(sealed[tf])
        den = a_gf - a_tf
        share = round((a_to - a_tf) / den, 3) if den else None
        o4[arm] = OrderedDict([("acc_TO", a_to), ("acc_TF", a_tf), ("acc_GF", a_gf),
                               ("TO_minus_TF_pp", round(a_to - a_tf, 2)), ("GF_minus_TF_pp", round(den, 2)),
                               ("reading_share", share)])
        print("O4 %s reading share = (TO - TF) / (GF - TF) = (%.2f - %.2f) / (%.2f - %.2f) = %s"
              % (arm, a_to, a_tf, a_gf, a_tf, share))
    summary["O4_reading_share"] = o4
    o5 = OrderedDict([("R4", pre["R4_O5"]), ("teacher_seals", pre["teacher_seals"]),
                      ("facts_per_row", OrderedDict()), ("V6_in_TO_prompts", OrderedDict((a, pre["leak_guard"][a]["count"]) for a in TEACHER_OF))])
    for arm in TEACHER_OF:
        nf = sorted(r["n_facts_given"] for r in results[arm])
        o5["facts_per_row"][arm] = OrderedDict([("min", nf[0]), ("median", nf[len(nf) // 2]), ("max", nf[-1]),
                                                ("empty", sum(1 for x in nf if x == 0)),
                                                ("distribution", OrderedDict(sorted(Counter(nf).items())))])
        print("O5 %s: R4 (a) D6-71 %s%%, (b) all %s%%; facts/row min %d median %d max %d empty %d; parse_ok %s repaired %s; V6 %d"
              % (arm, pre["R4_O5"]["arms"][arm]["a_D6_71"]["pct"], pre["R4_O5"]["arms"][arm]["b_all_rows"]["pct"],
                 nf[0], nf[len(nf) // 2], nf[-1], sum(1 for x in nf if x == 0),
                 pre["teacher_seals"][TEACHER_OF[arm]]["t1_parse_ok"], pre["teacher_seals"][TEACHER_OF[arm]]["t1_repaired_rows"],
                 pre["leak_guard"][arm]["count"]))
    summary["O5_teacher"] = o5
    o6 = OrderedDict()
    for arm in ("GF",) + tuple(TEACHER_OF):
        o6[arm] = by_gold(results[arm], A, arm, "A_r2")
        for g in ("yes", "no"):
            e = o6[arm]["gold_%s" % g]
            print("O6 %s vs A_r2, gold %-3s n=%s: Δ %s pp, wins %s losses %s, pred-yes %s%% vs %s%%"
                  % (arm, g, e.get("n"), e.get("delta_pp"), e.get("wins_%s" % arm), e.get("losses_%s" % arm),
                     e.get("pred_yes_rate_%s" % arm), e.get("pred_yes_rate_A_r2")))
    summary["O6_splits_by_gold"] = o6

    # freeze §6: pre-declared design decision, keyed on max(O2)
    deltas = [o2[a].get("delta_pp") for a in TEACHER_OF if o2[a].get("delta_pp") is not None]
    mx = max(deltas) if deltas else None
    if is_test or mx is None:
        branch = na
    elif mx >= O2_READING_WALL_PP:
        branch = ("max(O2) = %+.1f pp >= +%.1f: reading is the main wall -> D11 = reading-robust injection "
                  "(multi-reading agreement gate; title-grounded entity reading)" % (mx, O2_READING_WALL_PP))
    elif mx <= O2_KNOWLEDGE_WALL_PP:
        branch = ("max(O2) = %+.1f pp <= +%.1f: the teacher's knowledge is the main wall -> D11 targets fact relevance "
                  "(decomposition-guided retrieval for the teacher, or a second teacher for agreement); the reading route is not pursued"
                  % (mx, O2_KNOWLEDGE_WALL_PP))
    else:
        branch = "max(O2) = %+.1f pp, between +%.1f and +%.1f: both levers, agreement gate first" % (mx, O2_KNOWLEDGE_WALL_PP, O2_READING_WALL_PP)
    summary["design_decision_section_6"] = OrderedDict([("max_O2_pp", mx), ("reading_wall_pp", O2_READING_WALL_PP),
                                                        ("knowledge_wall_pp", O2_KNOWLEDGE_WALL_PP), ("branch", branch),
                                                        ("eval458", "reserved for the final D11 method, run once")])
    print("\n§6 design decision: %s" % branch)

    # accuracy table
    n_yes = sum(1 for r in sub if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(sub) - n_yes) / len(sub)
    table = OrderedDict()
    print("\n" + hdr)
    print("ACCURACY TABLE%s   floor %.2f%%" % (tags, floor))
    print(hdr)
    print("%-6s %-58s %5s %8s %9s %9s %8s" % ("arm", "facts", "n", "acc%", "unpars%", "SEC acc%", "yes%"))
    label = OrderedDict([("A_r2", "none (sealed D10 floor%s)" % ("" if pre["V2_prime"]["reused_sealed"] else ", RE-GENERATED here")),
                         ("GF", "gold urbench_facts (ceiling, = D1 arm B layout)"),
                         ("TO1", "v1 teacher facts, teacher read the gold English question"),
                         ("TO2", "v2 teacher facts, teacher read the gold English question"),
                         ("TF_v1", "v1 teacher facts, Urdu input (sealed D10)"),
                         ("TF_v2", "v2 teacher facts, Urdu input (sealed D10v2)")])
    for arm in label:
        recs = results[arm] if arm in results else sealed[arm]
        pri = score(recs, "pred")
        se = score(recs, "pred_secondary")
        table[arm] = OrderedDict([
            ("n", pri["n"]), ("accuracy", pri["accuracy"]), ("correct", pri["correct"]),
            ("unparsed_rate", pri["unparsed_rate"]),
            ("truncation_rate", round(100.0 * sum(1 for r in recs if r.get("truncated")) / pri["n"], 2)),
            ("predicted_yes_rate", pri["predicted_yes_rate"]),
            ("accuracy_secondary", se["accuracy"]), ("unparsed_rate_secondary", se["unparsed_rate"])])
        print("%-6s %-58s %5d %8.2f %9.2f %9.2f %8.2f"
              % (arm, label[arm], pri["n"], pri["accuracy"], pri["unparsed_rate"], se["accuracy"], pri["predicted_yes_rate"]))
    summary["floor"] = round(floor, 2)
    summary["accuracy_table"] = table
    return summary


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true",
                    help="CPU: checks and prompts, write nothing, exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="TEST: first N rows; reads the TEST teacher dirs, writes only under %s" % OUT_DIR_TEST)
    args = ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        die("--limit must be positive")
    if args.preflight_only and args.limit is not None:
        die("--preflight-only takes no --limit")

    is_test = args.limit is not None
    OUT_DIR = OUT_DIR_TEST if is_test else OUT_DIR_FULL
    t_dirs = T_DIRS_TEST if is_test else T_DIRS_FULL

    instruction, sub, prompts, facts_by, teacher_by, sealed, tok, pre, r4_rows = preflight(args.limit, t_dirs)

    if args.preflight_only:
        print("\n[preflight-only] done. No files written.")
        return

    print("\n[mode] %s  rows=%d  out=%s" % ("TEST — NOT A RESULT" if is_test else "FULL", len(sub), OUT_DIR))
    if not is_test and len(sub) != 200:
        die("FULL run must cover 200 rows")
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")

    # ---------------- model (exactly as D8 / D4 / D10) ----------------
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

    # ---------------- V2' (freeze §4): 20 A_r2 prompts once vs the SEALED generations ----------------
    n_probe = min(V2_PROBE_N_TEST if is_test else V2_PROBE_N_FULL, len(sub))
    probe_prompts = [p for p, _, _, _ in prompts["A_r2"][:n_probe]]
    print("\n" + "=" * 70)
    print("V2' — determinism probe vs the sealed A_r2: %d prompts once, batch size %d" % (n_probe, BATCH_SIZE))
    print("=" * 70, flush=True)
    g = generate_all(model, tok, probe_prompts, BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
    same = [ta == e["generation"] for (ta, _, _), e in zip(g, sealed["A_r2"][:n_probe])]
    n_same = sum(same)
    v2_pass = n_same == n_probe
    mism = [sealed["A_r2"][i]["qid"] for i, s in enumerate(same) if not s]
    v2p = OrderedDict([("n", n_probe), ("identical_to_sealed", n_same), ("pass", v2_pass),
                       ("mismatched_qids", mism), ("reused_sealed", v2_pass),
                       ("verdict", "PASS: %d/%d byte-identical to the sealed D10 A_r2 generations; sealed A_r2 reused as the floor" % (n_same, n_probe)
                        if v2_pass else
                        "FAIL: %d/%d identical to the sealed generations; A_r2 RE-GENERATED in this job (freeze §4)" % (n_same, n_probe))])
    print("V2' %s" % v2p["verdict"], flush=True)
    pre["V2_prime"] = v2p

    start = OrderedDict([
        ("job", "S"), ("diagnostic", "D11-O oracle reading"), ("mode", "TEST" if is_test else "FULL"),
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
    ])
    start.update(pre)
    print("\n[START]")
    write_json(os.path.join(OUT_DIR, "S_START.json"), start)

    # ---------------- arms ----------------
    def run_arm(arm):
        ps = [p for p, _, _, _ in prompts[arm]]
        print("\n" + "=" * 70)
        print("ARM %s  (n=%d, max_new=%d, batch size %d, greedy)" % (arm, len(sub), MAX_NEW_TOKENS, BATCH_SIZE))
        print("=" * 70, flush=True)
        gens = generate_all(model, tok, ps, BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
        if len(gens) != len(ps):
            die("%s: %d generations for %d prompts" % (arm, len(gens), len(ps)))
        recs = []
        for r, (p, lay, fl, src), (text, n_new, trunc) in zip(sub, prompts[arm], gens):
            pred = extract_answer(text)
            recs.append(OrderedDict([
                ("qid", r["qid"]), ("arm", arm), ("layout", lay), ("gold", r["gold"]),
                ("pred", pred), ("pred_secondary", extract_answer_secondary(text)),
                ("correct", pred == r["gold"]), ("unparsed", pred is None),
                ("truncated", trunc), ("n_gen_tokens", n_new),
                ("n_facts_given", len(fl)), ("facts_source", src),
                ("source_parse_ok", teacher_by[src][r["qid"]]["parse_ok"] if src in teacher_by else None),
                ("prompt", p), ("generation", text),
            ]))
        write_jsonl(os.path.join(OUT_DIR, "d11o_arm%s.jsonl" % arm), recs, expect_n=len(sub))
        s = score(recs, "pred")
        print("  accuracy %.2f%%  unparsed %.2f%%  (%d/%d correct)  pred-yes %.2f%%"
              % (s["accuracy"], s["unparsed_rate"], s["correct"], s["n"], s["predicted_yes_rate"]))
        print("  sample generation (row 0, first 300 chars): %r" % recs[0]["generation"][:300], flush=True)
        return recs

    results = OrderedDict()
    if v2_pass:
        # the sealed records, re-shaped to this job's record layout (prompt / generation kept verbatim)
        results["A_r2"] = [OrderedDict([
            ("qid", e["qid"]), ("arm", "A_r2"), ("layout", e["layout"]), ("gold", e["gold"]),
            ("pred", e["pred"]), ("pred_secondary", e["pred_secondary"]),
            ("correct", e["correct"]), ("unparsed", e["unparsed"]),
            ("truncated", e["truncated"]), ("n_gen_tokens", e["n_gen_tokens"]),
            ("n_facts_given", 0), ("facts_source", "sealed:%s" % SEALED["A_r2"][0]), ("source_parse_ok", None),
            ("prompt", e["prompt"]), ("generation", e["generation"])]) for e in sealed["A_r2"]]
        print("\n[A_r2] sealed D10 floor reused (%d rows, MD5 %s); not re-generated" % (len(sub), SEALED["A_r2"][1]))
    else:
        results["A_r2"] = run_arm("A_r2")
    for arm in GEN_ARMS:
        results[arm] = run_arm(arm)

    # ---------------- readings ----------------
    summary = compute_readings(sub, results, sealed, pre, is_test)
    summary["slurm_job_id"] = job_id
    summary["teacher_seals"] = pre["teacher_seals"]
    summary["sealed_references"] = pre["sealed_references"]
    summary["V2_prime"] = v2p
    summary["R4_rows"] = r4_rows
    write_json(os.path.join(OUT_DIR, "d11o_summary.json"), summary)

    # ---------------- SEAL ----------------
    files = OrderedDict()
    for p in WRITTEN:
        with open(p, "rb") as f:
            b = f.read()
        files[p] = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)), ("lines", b.count(b"\n"))])
    seal = OrderedDict([
        ("job", "S"), ("diagnostic", "D11-O oracle reading"), ("mode", "TEST" if is_test else "FULL"),
        ("slurm_job_id", job_id),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("script_md5", pre["script"]["md5"]),
        ("d11o_freeze_md5", pre["d11o_freeze"]["md5"]), ("freeze_md5", pre["freeze"]["md5"]),
        ("teacher_seal_files", OrderedDict((v, pre["teacher_seals"][v]["files"]) for v in pre["teacher_seals"])),
        ("sealed_references", pre["sealed_references"]),
        ("A_r2_reused_sealed", v2_pass),
        ("files", files)])
    write_json(os.path.join(OUT_DIR, "S_SEAL.json"), seal)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD11-O JOB S COMPLETE. Sealed %d files." % len(files))


if __name__ == "__main__":
    main()
