#!/usr/bin/env python3
"""
d10v2_teacher.py  —  D10 v2 Job T (teacher, vLLM): the one fix iteration

Contract: docs/EFBPT_D10_TEACHER_FREEZE.md (FROZEN, MD5
b3a822d8199034c7fe58d76cf5cc428f) §2, §3, §5, §8, as amended by
docs/EFBPT_D10_AMENDMENT_3.md (A3.1 T1-v2 decomposition prompt, A3.2 jobs,
A3.3 descriptives) and docs/EFBPT_D10_AMENDMENT_4.md (A4.1 JSON parse repair:
a reply with no balanced {...} block is retried once with "}" appended, then
once with "]}" appended; the row records repaired = "brace" /
"bracket_brace" / false; repaired rows are counted in the log, T_START and
T_SEAL) and docs/EFBPT_D10_AMENDMENT_5.md (A5.1: a balanced block that fails
json.loads is retried once with "]" inserted before its final "}"; repaired =
"bracket"). Copy of d10_teacher.py with ONLY the A3, A4 and A5 changes.

Job T v2 produces, for the DEV200 rows, with Meta-Llama-3.1-70B-Instruct-AWQ-INT4
under vLLM 0.13.0 on one L20:
  T1-v2 t_facts.jsonl  teacher question_en, unknown_terms, sub_questions,
                       entities + English facts per row (JSON reply, five keys,
                       max_tokens 768), parse_ok, raw text, prompt. Only the
                       post-T2 facts are ever injected; the other keys are
                       stored for audit (A3.1).
  T2  verdict filter   unchanged, applied to facts only; removed lines kept in
                       "removed" and counted (V5); first 6 kept
  T3 / T3F             NOT re-run (A3.2); the student v2 job reads the sealed
                       v1 t_direct.jsonl / t_direct_f.jsonl by pinned MD5
  T4  derangement.json qid -> donor qid, seed 1010, no fixed point; in FULL it
                       must be byte-identical to the sealed v1 file (A3.2)
  T5  leak guard       question_en / gold facts / evidence ids in ANY teacher
                       prompt: FATAL (d8_reachable_ceiling.leak_guard)
  T6  T_START.json, T_SEAL.json (both carry the freeze AND the Amendment 3, 4, 5 MD5s)

Everything frozen is IMPORTED, never re-implemented:
  Urdu instruction (+MD5), die                             <- efbpt_eval_dev200
  DEV200 loader (qid = urbench_qid, gold, required)        <- d4_extract_facts
  leak guard, needles, Latin fraction, md5/json helpers    <- d8_reachable_ceiling

Modes:
  --preflight-only   CPU, no model: md5 checks, rows, T1-v2 prompts
                     (tokenizer only), leak guard, derangement (+ v1 equality),
                     counts. Writes NOTHING.
  --limit N          TEST: first N rows, outputs ONLY under
                     outputs/efbpt/d10v2/_test/teacher/. Never a result.
  (no flags)         FULL: outputs/efbpt/d10v2/teacher/.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d10v2_teacher.py --preflight-only
  python eval/error_analysis_tests/efbpt/d10v2_teacher.py --limit 5
  python eval/error_analysis_tests/efbpt/d10v2_teacher.py
"""

import argparse
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from efbpt_eval_dev200 import (          # noqa: E402
    INSTRUCTION_PATH, INSTRUCTION_MD5, load_instruction, die,
)
from d4_extract_facts import load_rows    # noqa: E402  qid = urbench_qid
from d8_reachable_ceiling import (        # noqa: E402  same guard as D8
    leak_guard, load_leak_needles, latin_fraction,
    md5_bytes, md5_file, md5_text, read_jsonl, pkg_versions,
)

# ----------------------------------------------------------------------------
# Frozen constants (freeze §2, §3, §5, §8)
# ----------------------------------------------------------------------------

FREEZE_PATH = "docs/EFBPT_D10_TEACHER_FREEZE.md"
FREEZE_MD5 = "b3a822d8199034c7fe58d76cf5cc428f"
AMEND3_PATH = "docs/EFBPT_D10_AMENDMENT_3.md"
AMEND3_MD5 = "eb097415ba4fa254c96300e6e3d7755e"
AMEND4_PATH = "docs/EFBPT_D10_AMENDMENT_4.md"
AMEND4_MD5 = "28a39b3d2adea6eca309b40fc15a00e5"
AMEND5_PATH = "docs/EFBPT_D10_AMENDMENT_5.md"
AMEND5_MD5 = "53026e77e67a08124a4ee42aa38c556f"
# A4.1: repair attempts, in this order, only when no balanced block is found
REPAIR_ATTEMPTS = OrderedDict([("brace", "}"), ("bracket_brace", "]}")])
# A5.1 (Amendment 5): a balanced block that fails json.loads is retried once
# with "]" inserted before its final "}"; tag "bracket".
BLOCK_REPAIR = OrderedDict([("bracket", 'insert "]" before the final "}"')])

# A3.2: the v1 derangement (seed 1010) is sealed; the v2 file must equal it
V1_DERANGEMENT_PATH = "outputs/efbpt/d10/teacher/derangement.json"
V1_DERANGEMENT_MD5 = "8f00e2f9e54f27138964a0577a8eff52"

# §2 teacher
TEACHER_PATH = ("/mnt/home/user41/downloaded_models/LLM-Research/"
                "Meta-Llama-3___1-70B-Instruct-AWQ-INT4")
VLLM_VERSION = "0.13.0"
VLLM_ARGS = OrderedDict([
    ("tensor_parallel_size", 1),
    ("max_model_len", 4096),
    ("quantization", "awq"),
    ("gpu_memory_utilization", 0.90),
    ("max_num_seqs", 1),
    ("enforce_eager", True),
    ("trust_remote_code", True),
    ("disable_log_stats", True),
])
SAMPLING = OrderedDict([("temperature", 0.0), ("top_p", 1.0), ("seed", 0)])

# A3.1 T1-v2 — verbatim from docs/EFBPT_D10_AMENDMENT_3.md. The amendment wraps
# long lines at ~78 columns; a wrap inside a sentence is joined with one space,
# a line that ends a clause (":", ";", ".") keeps its newline, and the blank
# line is kept (same convention as the v1 script).
T_SYSTEM = (
    "You are a careful research assistant with broad world knowledge. You read a "
    "question written in Urdu. You do not answer it. You first write what the "
    "question asks in English, then the smaller questions whose answers would "
    "decide it, then short, true, self-contained English facts that answer those "
    "smaller questions. You only write facts you are confident are true, and only "
    "about terms you have identified with confidence; if you cannot identify a "
    "term, you list it as unknown and write nothing about it. You never state or "
    "hint at the answer to the question, you never compare the entities with each "
    "other, and you never use the words yes, no, therefore, thus, hence, or answer."
)
T_USER_TEMPLATE = (
    "Question (Urdu): {question_ur}\n"
    "\n"
    "Return only a JSON object with these keys, in this order:\n"
    "\"question_en\": the question in English, one sentence, your best reading;\n"
    "\"unknown_terms\": a list of terms in the question you could not identify with "
    "confidence (empty list if none);\n"
    "\"sub_questions\": a list of 2 to 4 short English questions whose answers "
    "decide the question;\n"
    "\"entities\": a list of the entities and concepts you identified, written as "
    "their English Wikipedia article titles;\n"
    "\"facts\": a list of at most 6 short English facts (one sentence each), each "
    "answering one of the sub-questions, only about entities you identified with "
    "confidence.\n"
    "No other text."
)
T1_MAX_TOKENS = 768                   # A3.1
MAX_FACTS = 6
# A3.1: the five keys, in order, with their required types
T1_KEYS = OrderedDict([("question_en", str), ("unknown_terms", list),
                       ("sub_questions", list), ("entities", list), ("facts", list)])

# §3 T2 — verdict filter, case-insensitive
VERDICT_RE = re.compile(
    r"\b(therefore|thus|hence|the answer|this question|the question|in conclusion)\b",
    re.IGNORECASE)
YESNO_RE = re.compile(r"^(yes|no)[,.:]", re.IGNORECASE)

# §3 T3 / A1.1 T3F: not re-run in v2 (A3.2); the sealed v1 files are reused
# by the student v2 job under pinned MD5s.

# §3 T4
DERANGE_SEED = 1010

# §5 V4, V5 (reported here, flagged in the student's summary too)
V4_PARSE_MIN_PCT = 95.0
V5_REMOVED_MAX = 100

OUT_DIR_FULL = "outputs/efbpt/d10v2/teacher"            # A3.2
OUT_DIR_TEST = "outputs/efbpt/d10v2/_test/teacher"
FORBIDDEN_WRITE_DIRS = ("outputs/efbpt/d1", "outputs/efbpt/d4", "outputs/efbpt/d6",
                        "outputs/efbpt/d8", "outputs/efbpt/d10",   # the whole sealed v1 tree
                        "outputs/efbpt/d10v2/student", "outputs/efbpt/d10v2/_test/student")

# ----------------------------------------------------------------------------
# Output helpers (same layout as D8; the writable root depends on the mode)
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
    print("----- END FULL PROMPT [%s] -----" % label, flush=True)


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
# T1 parsing and T2 filter
# ----------------------------------------------------------------------------

FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")


def first_json_object(text):
    """The first balanced {...} block of the text, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


_EMPTY = (False, "", [], [], [], [], False)


def _load_block(block):
    """A balanced {...} block, json-loaded, if it is a dict; else None."""
    try:
        obj = json.loads(block)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _load_object(text):
    """The first balanced {...} block of text, json-loaded, if it is a dict."""
    block = first_json_object(text)
    if block is None:
        return None
    return _load_block(block)


def parse_teacher_json(raw):
    """T1-v2 (A3.1): strip code fences, take the first {...} block, json.loads.
    All five keys must be present with the right types (question_en a string,
    the other four lists of strings); otherwise parse_ok=False, question_en ""
    and four empty lists.
    A4.1: if no balanced block is found, the text is retried with "}" appended
    (repaired = "brace"), then with "]}" appended (repaired = "bracket_brace");
    at most these two attempts, in that order.
    A5.1: if a balanced block is found but fails json.loads, it is retried once
    with "]" inserted before its final "}" (repaired = "bracket").
    The five-key rule is unchanged.
    Returns (ok, question_en, unknown_terms, sub_questions, entities, facts,
    repaired), repaired being False when no repair was used."""
    text = FENCE_RE.sub("", raw)
    repaired = False
    obj = None
    block = first_json_object(text)
    if block is None:
        for tag, suffix in REPAIR_ATTEMPTS.items():          # A4.1
            obj = _load_object(text + suffix)
            if obj is not None:
                repaired = tag
                break
    else:
        obj = _load_block(block)
        if obj is None:                                        # A5.1
            obj = _load_block(block[:-1] + "]}")
            if obj is not None:
                repaired = "bracket"
    if obj is None:
        return _EMPTY
    vals = []
    for key, typ in T1_KEYS.items():
        if key not in obj:
            return _EMPTY
        v = obj[key]
        if typ is str:
            if not isinstance(v, str):
                return _EMPTY
            vals.append(v.strip())
        else:
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                return _EMPTY
            vals.append([x.strip() for x in v if x.strip()])
    return (True,) + tuple(vals) + (repaired,)


def verdict_filter(facts):
    """T2: returns (kept[:MAX_FACTS], removed)."""
    kept, removed = [], []
    for f in facts:
        if VERDICT_RE.search(f) or YESNO_RE.match(f):
            removed.append(f)
        else:
            kept.append(f)
    return kept[:MAX_FACTS], removed


def derangement(qids, seed):
    """T4: a permutation of qids with no fixed point, reshuffled until none."""
    if len(qids) < 2:
        die("a derangement needs at least 2 qids (have %d)" % len(qids))
    rng = random.Random(seed)
    donors = list(qids)
    tries = 0
    while True:
        tries += 1
        rng.shuffle(donors)
        if all(a != b for a, b in zip(qids, donors)):
            break
    m = OrderedDict(zip(qids, donors))
    if sorted(m.values()) != sorted(qids) or any(k == v for k, v in m.items()):
        die("derangement check failed")
    return m, tries


def derangement_record(dmap):
    """The exact object written to derangement.json (same as the v1 script)."""
    return OrderedDict([("seed", DERANGE_SEED), ("n", len(dmap)),
                        ("note", "qid -> donor qid; TF_wrong gets the donor's post-T2 facts"),
                        ("map", dmap)])


# ----------------------------------------------------------------------------
# Preflight (CPU, no model)
# ----------------------------------------------------------------------------

def preflight(limit):
    print("=" * 78)
    print("D10 v2 JOB T PREFLIGHT (freeze §2, §3; Amendment 3 A3.1, A3.2)")
    print("=" * 78, flush=True)

    if not os.path.exists(FREEZE_PATH):
        die("freeze file missing: " + FREEZE_PATH)
    freeze_md5 = md5_file(FREEZE_PATH)
    if freeze_md5 != FREEZE_MD5:
        die("freeze MD5 mismatch: expected %s, got %s" % (FREEZE_MD5, freeze_md5))
    if not os.path.exists(AMEND3_PATH):
        die("Amendment 3 file missing: " + AMEND3_PATH)
    amend3_md5 = md5_file(AMEND3_PATH)
    if amend3_md5 != AMEND3_MD5:
        die("Amendment 3 MD5 mismatch: expected %s, got %s" % (AMEND3_MD5, amend3_md5))
    if not os.path.exists(AMEND4_PATH):
        die("Amendment 4 file missing: " + AMEND4_PATH)
    amend4_md5 = md5_file(AMEND4_PATH)
    if amend4_md5 != AMEND4_MD5:
        die("Amendment 4 MD5 mismatch: expected %s, got %s" % (AMEND4_MD5, amend4_md5))
    amend5_md5 = md5_file(AMEND5_PATH)
    if amend5_md5 != AMEND5_MD5:
        die("Amendment 5 MD5 mismatch: expected %s, got %s" % (AMEND5_MD5, amend5_md5))
    script_md5 = md5_file(os.path.abspath(__file__))
    print("[md5] freeze  %s  %s" % (freeze_md5, FREEZE_PATH))
    print("[md5] amend3  %s  %s" % (amend3_md5, AMEND3_PATH))
    print("[md5] amend4  %s  %s" % (amend4_md5, AMEND4_PATH))
    print("[md5] amend5  %s  %s" % (amend5_md5, AMEND5_PATH))
    print("[md5] script  %s  %s" % (script_md5, os.path.relpath(__file__)))

    instruction = load_instruction()              # MD5 asserted inside
    print("[ok] Urdu instruction MD5 %s verified" % INSTRUCTION_MD5)

    rows = load_rows()                            # asserts 200 rows
    qids = [r["qid"] for r in rows]
    if len(set(qids)) != 200:
        die("DEV200 qids are not unique")
    for r in rows:
        if "facts" in r:
            die("loader row carries a 'facts' key; gold facts must never reach build_user")
    print("[ok] DEV200: 200 rows, 200 unique qids, no 'facts' key in any row")

    sub = rows[:limit] if limit else rows
    sub_qids = [r["qid"] for r in sub]

    # ---- tokenizer only (the model is loaded later, in run mode) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)

    def render(system, user):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        # the text we store must be exactly what the model sees
        if tok(text, add_special_tokens=False)["input_ids"] != ids:
            die("chat template: stored prompt text does not re-tokenize to the generated ids")
        return text, ids

    p_facts = [render(T_SYSTEM, T_USER_TEMPLATE.format(question_ur=r["question_ur"]))
               for r in sub]
    # T3 / T3F: not rendered, not run (A3.2)

    for label, ps, mx in (("T1v2/facts", p_facts, T1_MAX_TOKENS),):
        tl = sorted(len(ids) for _, ids in ps)
        print("[len] %-10s n=%d tokens min=%d median=%d max=%d (+%d new <= %d)"
              % (label, len(ps), tl[0], tl[len(tl) // 2], tl[-1], mx,
                 VLLM_ARGS["max_model_len"]))
        if tl[-1] + mx > VLLM_ARGS["max_model_len"]:
            die("%s prompts would exceed max_model_len" % label)

    # ---- T5: leak guard, fatal on every needle, before anything else ----
    needles = load_leak_needles()
    leak_guard("T1v2/facts", [t for t, _ in p_facts], needles, fatal_gold_facts=True)

    # ---- T4 ----
    dmap, tries = derangement(sub_qids, DERANGE_SEED)
    print("[ok] derangement of %d qids, seed %d, %d shuffle(s), no fixed point"
          % (len(dmap), DERANGE_SEED, tries))
    # A3.2: the sealed v1 derangement must exist with its sealed MD5, and the
    # full 200-qid map must equal it (in TEST the map covers only N qids, so
    # only the sealed file itself is checked there).
    if not os.path.exists(V1_DERANGEMENT_PATH):
        die("sealed v1 derangement missing: " + V1_DERANGEMENT_PATH)
    v1_der_md5 = md5_file(V1_DERANGEMENT_PATH)
    if v1_der_md5 != V1_DERANGEMENT_MD5:
        die("sealed v1 derangement MD5 %s != %s" % (v1_der_md5, V1_DERANGEMENT_MD5))
    v1_der = json.load(open(V1_DERANGEMENT_PATH, encoding="utf-8"))
    if limit is None:
        if list(v1_der["map"].items()) != list(dmap.items()) \
                or v1_der.get("seed") != DERANGE_SEED or v1_der.get("n") != len(dmap):
            die("v2 derangement differs from the sealed v1 derangement (A3.2)")
        if md5_text(json.dumps(derangement_record(dmap), indent=2, ensure_ascii=False)) != V1_DERANGEMENT_MD5:
            die("v2 derangement.json would not be byte-identical to the sealed v1 file (A3.2)")
        print("[ok] derangement equals the sealed v1 derangement.json (MD5 %s)" % V1_DERANGEMENT_MD5)
    else:
        print("[ok] sealed v1 derangement.json present (MD5 %s); TEST map of %d qids not compared"
              % (V1_DERANGEMENT_MD5, len(dmap)))

    log_full_prompt("T1-v2 facts, row 0, qid %s" % sub[0]["qid"], p_facts[0][0])

    meta = OrderedDict([
        ("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", freeze_md5)])),
        ("amendment_3", OrderedDict([("path", AMEND3_PATH), ("md5", amend3_md5)])),
        ("amendment_4", OrderedDict([("path", AMEND4_PATH), ("md5", amend4_md5),
                                     ("parse_repair", OrderedDict((k, v) for k, v in REPAIR_ATTEMPTS.items())),
                                     ("t1_repaired_rows", "filled after T1")])),
        ("amendment_5", OrderedDict([("path", AMEND5_PATH), ("md5", amend5_md5),
                                     ("block_repair", OrderedDict((k, v) for k, v in BLOCK_REPAIR.items()))])),
        ("script", OrderedDict([("path", os.path.relpath(__file__)), ("md5", script_md5)])),
        ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("prompts", OrderedDict([
            ("t1_version", "T1-v2 (A3.1 decomposition)"),
            ("t_system_md5", md5_text(T_SYSTEM)),
            ("t_user_template_md5", md5_text(T_USER_TEMPLATE)),
            ("t1_keys", list(T1_KEYS.keys())),
            ("t1_max_tokens", T1_MAX_TOKENS),
            ("t3_t3f", "not re-run (A3.2); sealed v1 files reused by the student"),
        ])),
        ("verdict_filter", OrderedDict([("regex", VERDICT_RE.pattern),
                                        ("yes_no_regex", YESNO_RE.pattern),
                                        ("case_insensitive", True),
                                        ("max_facts", MAX_FACTS),
                                        ("applied_to", "facts only (A3.1)")])),
        ("derangement", OrderedDict([("seed", DERANGE_SEED), ("n", len(dmap)),
                                     ("shuffles", tries),
                                     ("v1_sealed_md5", V1_DERANGEMENT_MD5),
                                     ("equals_v1", limit is None)])),
        ("leak_guard", OrderedDict([("prompts_checked", len(p_facts)),
                                    ("fatal_hits", 0)])),
    ])
    print("[preflight] PASS", flush=True)
    return sub, p_facts, dmap, meta


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true",
                    help="CPU: checks and prompts, write nothing, exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="TEST: first N DEV200 rows; writes only under %s" % OUT_DIR_TEST)
    args = ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        die("--limit must be positive")
    if args.preflight_only and args.limit is not None:
        die("--preflight-only takes no --limit")

    is_test = args.limit is not None
    OUT_DIR = OUT_DIR_TEST if is_test else OUT_DIR_FULL

    sub, p_facts, dmap, meta = preflight(args.limit)

    if args.preflight_only:
        print("\n[preflight-only] done. No files written.")
        return

    print("\n[mode] %s  rows=%d  out=%s"
          % ("TEST — NOT A RESULT" if is_test else "FULL", len(sub), OUT_DIR))
    if not is_test and len(sub) != 200:
        die("FULL run must cover 200 rows")
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")

    # ---------------- vLLM ----------------
    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != VLLM_VERSION:
        die("vLLM %s found, freeze requires %s" % (vllm.__version__, VLLM_VERSION))

    versions = pkg_versions()
    versions["vllm"] = vllm.__version__

    start = OrderedDict([
        ("job", "T"), ("version", "v2 (Amendments 3, 4, 5)"), ("mode", "TEST" if is_test else "FULL"),
        ("args", vars(args)),
        ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("hostname", socket.gethostname()), ("slurm_job_id", job_id),
        ("n_rows", len(sub)), ("qids", [r["qid"] for r in sub]),
        ("teacher_path", TEACHER_PATH), ("vllm_args", VLLM_ARGS),
        ("sampling", SAMPLING), ("versions", versions),
    ])
    start.update(meta)
    print("\n[START]")
    write_json(os.path.join(OUT_DIR, "T_START.json"), start)

    print("\n[load] %s ..." % TEACHER_PATH, flush=True)
    t0 = time.time()
    llm = LLM(model=TEACHER_PATH, **VLLM_ARGS)
    print("[load] %.0fs" % (time.time() - t0), flush=True)

    def generate(prompt_ids, max_tokens, label):
        params = SamplingParams(temperature=SAMPLING["temperature"],
                                top_p=SAMPLING["top_p"], seed=SAMPLING["seed"],
                                max_tokens=max_tokens)
        print("\n" + "=" * 70)
        print("%s  (n=%d, max_tokens=%d, greedy, max_num_seqs=%d)"
              % (label, len(prompt_ids), max_tokens, VLLM_ARGS["max_num_seqs"]))
        print("=" * 70, flush=True)
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids],
                            params, use_tqdm=False)
        if len(outs) != len(prompt_ids):
            die("%s: vLLM returned %d outputs for %d prompts" % (label, len(outs), len(prompt_ids)))
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append((c.text, len(c.token_ids), c.finish_reason))
        el = time.time() - t0
        print("  %d rows in %.1fs (%.1fs/row)" % (len(res), el, el / max(len(res), 1)), flush=True)
        return res

    # ---------------- T1 + T2 ----------------
    g1 = generate([ids for _, ids in p_facts], T1_MAX_TOKENS, "T1-v2 — teacher decomposition + facts (A3.1)")
    t_facts = []
    n_parse_ok = n_removed_total = n_repaired = 0
    for r, (text, ids), (raw, n_new, fin) in zip(sub, p_facts, g1):
        ok, q_en, unknown, subqs, ents, facts_raw, repaired = parse_teacher_json(raw)
        kept, removed = verdict_filter(facts_raw) if ok else ([], [])
        n_parse_ok += int(ok)
        n_repaired += int(bool(repaired))
        n_removed_total += len(removed)
        t_facts.append(OrderedDict([
            ("qid", r["qid"]), ("arm", "T_facts_v2"),
            ("prompt", text), ("raw", raw),
            ("parse_ok", ok), ("repaired", repaired),          # A4.1
            # A3.1: stored for audit only; never enter any student prompt
            ("question_en", q_en), ("unknown_terms", unknown), ("sub_questions", subqs),
            ("entities", ents),
            ("facts_parsed", facts_raw), ("facts", kept), ("removed", removed),
            ("n_facts", len(kept)), ("n_facts_parsed", len(facts_raw)),
            ("n_removed", len(removed)), ("n_unknown_terms", len(unknown)),
            ("n_sub_questions", len(subqs)),
            ("truncated", fin == "length"), ("finish_reason", fin),
            ("n_gen_tokens", n_new),
        ]))
    nf = sorted(e["n_facts"] for e in t_facts)
    print("  parse_ok %d/%d (%.1f%%)   facts per row min=%d median=%d max=%d  "
          "empty=%d   removed (V5) %d   truncated %d"
          % (n_parse_ok, len(t_facts), 100.0 * n_parse_ok / len(t_facts),
             nf[0], nf[len(nf) // 2], nf[-1], sum(1 for x in nf if x == 0),
             n_removed_total, sum(1 for e in t_facts if e["truncated"])))
    print("  [A4.1/A5.1] repaired rows %d/%d %s"
          % (n_repaired, len(t_facts), dict(Counter(e["repaired"] for e in t_facts if e["repaired"]))))
    # A4.1: the repaired count also goes into T_START (re-written, then sealed)
    start["amendment_4"]["t1_repaired_rows"] = n_repaired
    write_json(os.path.join(OUT_DIR, "T_START.json"), start)
    # A3.3 descriptives (never gates)
    print("  [A3.3] facts-per-row distribution %s" % dict(sorted(Counter(nf).items())))
    print("  [A3.3] rows with empty facts %d; rows with >=1 unknown_term %d; unknown_terms per row %s"
          % (sum(1 for x in nf if x == 0),
             sum(1 for e in t_facts if e["n_unknown_terms"] > 0),
             dict(sorted(Counter(e["n_unknown_terms"] for e in t_facts).items()))))
    print("  [A3.3] sub_questions per row %s"
          % dict(sorted(Counter(e["n_sub_questions"] for e in t_facts).items())))
    print("  sample (row 0): question_en %r" % t_facts[0]["question_en"])
    print("  sample (row 0): unknown_terms %s" % t_facts[0]["unknown_terms"])
    print("  sample (row 0): sub_questions %s" % t_facts[0]["sub_questions"])
    print("  sample (row 0): entities %s" % t_facts[0]["entities"])
    for f in t_facts[0]["facts"]:
        print("    - %s" % f)
    frac, letters, latin = latin_fraction([f for e in t_facts for f in e["facts"]])
    print("  Latin fraction of kept facts %.4f (%d/%d letters)" % (frac, latin, letters))
    write_jsonl(os.path.join(OUT_DIR, "t_facts.jsonl"), t_facts, expect_n=len(sub))

    # ---------------- T3 and T3F: NOT re-run (A3.2) ----------------
    print("\n[A3.2] T3 / T3F not re-run; the student v2 job reads the sealed v1 "
          "t_direct.jsonl and t_direct_f.jsonl by pinned MD5", flush=True)

    # ---------------- T4 ----------------
    der_path = os.path.join(OUT_DIR, "derangement.json")
    write_json(der_path, derangement_record(dmap))
    if not is_test:
        got = md5_file(der_path)
        if got != V1_DERANGEMENT_MD5:
            die("written derangement.json MD5 %s != sealed v1 %s (A3.2)" % (got, V1_DERANGEMENT_MD5))
        print("  [ok] derangement.json byte-identical to the sealed v1 file (MD5 %s)" % got)

    # ---------------- report (no gate here; the student's summary flags) ----------------
    parse_pct = 100.0 * n_parse_ok / len(t_facts)
    print("\nV4 teacher JSON parse rate %d/%d = %.1f%% (min %.0f%%) -> %s"
          % (n_parse_ok, len(t_facts), parse_pct, V4_PARSE_MIN_PCT,
             "ok" if parse_pct >= V4_PARSE_MIN_PCT else "FORMAT FLAG"))
    print("V5 verdict-filter removed lines %d (max %d) -> %s"
          % (n_removed_total, V5_REMOVED_MAX,
             "ok" if n_removed_total <= V5_REMOVED_MAX else "TEACHER-ANSWERING FLAG"))
    fr = Counter(e["finish_reason"] for e in t_facts)
    print("finish reasons T1: %s" % dict(fr))

    # ---------------- SEAL ----------------
    files = OrderedDict()
    for p in OrderedDict.fromkeys(WRITTEN):      # T_START.json is written twice (A4.1)
        with open(p, "rb") as f:
            b = f.read()
        files[p] = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)),
                                ("lines", b.count(b"\n"))])
    seal = OrderedDict([
        ("job", "T"), ("version", "v2 (Amendments 3, 4, 5)"), ("mode", "TEST" if is_test else "FULL"),
        ("slurm_job_id", job_id),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("script_md5", meta["script"]["md5"]), ("freeze_md5", meta["freeze"]["md5"]),
        ("amendment3_md5", meta["amendment_3"]["md5"]),
        ("amendment4_md5", meta["amendment_4"]["md5"]),
        ("amendment5_md5", meta["amendment_5"]["md5"]),
        ("n_rows", len(sub)),
        ("t1_parse_ok", n_parse_ok), ("t1_repaired_rows", n_repaired),
        ("t2_removed_total", n_removed_total),
        ("files", files)])
    write_json(os.path.join(OUT_DIR, "T_SEAL.json"), seal)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD10 v2 JOB T COMPLETE. Sealed %d files." % len(files))


if __name__ == "__main__":
    main()
