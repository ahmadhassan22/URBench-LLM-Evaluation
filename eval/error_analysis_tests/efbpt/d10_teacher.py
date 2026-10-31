#!/usr/bin/env python3
"""
d10_teacher.py  —  D10 Job T (teacher, vLLM)

Contract: docs/EFBPT_D10_TEACHER_FREEZE.md (FROZEN, MD5
b3a822d8199034c7fe58d76cf5cc428f), sections 2, 3, 5 (V3, V4, V5), 8.
Implements the freeze exactly; nothing here is tunable after numbers exist.

Job T produces, for the DEV200 rows, with Meta-Llama-3.1-70B-Instruct-AWQ-INT4
under vLLM 0.13.0 on one L20:
  T1  t_facts.jsonl    teacher entities + English facts per row (JSON reply),
                       parse_ok, raw text, prompt
  T2  verdict filter   applied once to the parsed facts; removed lines kept in
                       "removed" and counted (V5); first 6 kept
  T3  t_direct.jsonl   teacher answers the student's exact no-facts prompt
                       (build_user arm "A" + SYSTEM_MESSAGE), scored with the
                       frozen primary and secondary extractors
  T4  derangement.json qid -> donor qid, seed 1010, no fixed point
  T5  leak guard       question_en / gold facts / evidence ids in ANY teacher
                       prompt: FATAL (d8_reachable_ceiling.leak_guard)
  T6  T_START.json, T_SEAL.json

Everything frozen is IMPORTED, never re-implemented:
  SYSTEM_MESSAGE, Urdu instruction (+MD5), extractor, die  <- efbpt_eval_dev200
  user layouts (arm "A")                                   <- d1_eval_arms
  secondary (Devanagari) extractor                         <- d1_score_dual
  DEV200 loader (qid = urbench_qid, gold, required)        <- d4_extract_facts
  leak guard, needles, Latin fraction, md5/json helpers    <- d8_reachable_ceiling

Modes:
  --preflight-only   CPU, no model: md5 checks, rows, both prompt sets
                     (tokenizer only), leak guard, derangement, counts.
                     Writes NOTHING.
  --limit N          TEST: first N rows, outputs ONLY under
                     outputs/efbpt/d10/_test/teacher/. Never a result.
  (no flags)         FULL: outputs/efbpt/d10/teacher/.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d10_teacher.py --preflight-only
  python eval/error_analysis_tests/efbpt/d10_teacher.py --limit 5
  python eval/error_analysis_tests/efbpt/d10_teacher.py
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
    SYSTEM_MESSAGE, MAX_NEW_TOKENS, INSTRUCTION_PATH, INSTRUCTION_MD5,
    extract_answer, load_instruction, die,
)
from d1_eval_arms import build_user       # noqa: E402  arm "A" layout
from d1_score_dual import extract_answer_secondary   # noqa: E402
from d3_oracle_retrieval import score     # noqa: E402
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

# §3 T1 — verbatim. The freeze wraps long lines at ~78 columns; a wrap inside
# a sentence is joined with one space, a line that ends a clause (":", ";",
# ".") keeps its newline, and the blank line is kept.
T_SYSTEM = (
    "You are a careful research assistant with broad world knowledge. You read a "
    "question written in Urdu. You do not answer it. You identify the real-world "
    "entities and concepts the question is about, and you write short, true, "
    "self-contained English facts about them that someone would need in order to "
    "decide the question. You only write facts you are confident are true. You "
    "never state or hint at the answer to the question, and you never use the "
    "words yes, no, therefore, thus, hence, or answer."
)
T_USER_TEMPLATE = (
    "Question (Urdu): {question_ur}\n"
    "\n"
    "Return only a JSON object with two keys:\n"
    "\"entities\": a list of the entities and concepts in the question, written as "
    "their English Wikipedia article titles;\n"
    "\"facts\": a list of at most 6 short English facts (one sentence each) about "
    "those entities that are needed to decide the question.\n"
    "No other text."
)
T1_MAX_TOKENS = 512
MAX_FACTS = 6

# §3 T2 — verdict filter, case-insensitive
VERDICT_RE = re.compile(
    r"\b(therefore|thus|hence|the answer|this question|the question|in conclusion)\b",
    re.IGNORECASE)
YESNO_RE = re.compile(r"^(yes|no)[,.:]", re.IGNORECASE)

# §3 T3
T3_MAX_TOKENS = MAX_NEW_TOKENS        # 1024, the student's frozen value

# §3 T4
DERANGE_SEED = 1010

# §5 V4, V5 (reported here, flagged in the student's summary too)
V4_PARSE_MIN_PCT = 95.0
V5_REMOVED_MAX = 100

OUT_DIR_FULL = "outputs/efbpt/d10/teacher"
OUT_DIR_TEST = "outputs/efbpt/d10/_test/teacher"
FORBIDDEN_WRITE_DIRS = ("outputs/efbpt/d1", "outputs/efbpt/d4", "outputs/efbpt/d6",
                        "outputs/efbpt/d8", "outputs/efbpt/d10/student")

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


def parse_teacher_json(raw):
    """T1: strip code fences, take the first {...} block, json.loads.
    entities and facts must both be lists of strings; otherwise
    (parse_ok=False, [], [])."""
    block = first_json_object(FENCE_RE.sub("", raw))
    if block is None:
        return False, [], []
    try:
        obj = json.loads(block)
    except Exception:
        return False, [], []
    if not isinstance(obj, dict):
        return False, [], []
    ents, facts = obj.get("entities"), obj.get("facts")
    for lst in (ents, facts):
        if not isinstance(lst, list) or not all(isinstance(x, str) for x in lst):
            return False, [], []
    ents = [e.strip() for e in ents if e.strip()]
    facts = [f.strip() for f in facts if f.strip()]
    return True, ents, facts


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


# ----------------------------------------------------------------------------
# Preflight (CPU, no model)
# ----------------------------------------------------------------------------

def preflight(limit):
    print("=" * 78)
    print("D10 JOB T PREFLIGHT (freeze §2, §3)")
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
    p_direct = [render(SYSTEM_MESSAGE, build_user(instruction, r, None, "A"))
                for r in sub]

    for label, ps, mx in (("T1/facts", p_facts, T1_MAX_TOKENS),
                          ("T3/direct", p_direct, T3_MAX_TOKENS)):
        tl = sorted(len(ids) for _, ids in ps)
        print("[len] %-10s n=%d tokens min=%d median=%d max=%d (+%d new <= %d)"
              % (label, len(ps), tl[0], tl[len(tl) // 2], tl[-1], mx,
                 VLLM_ARGS["max_model_len"]))
        if tl[-1] + mx > VLLM_ARGS["max_model_len"]:
            die("%s prompts would exceed max_model_len" % label)

    # ---- T5: leak guard, fatal on every needle, before anything else ----
    needles = load_leak_needles()
    leak_guard("T1/facts", [t for t, _ in p_facts], needles, fatal_gold_facts=True)
    leak_guard("T3/direct", [t for t, _ in p_direct], needles, fatal_gold_facts=True)

    # ---- T4 ----
    dmap, tries = derangement(sub_qids, DERANGE_SEED)
    print("[ok] derangement of %d qids, seed %d, %d shuffle(s), no fixed point"
          % (len(dmap), DERANGE_SEED, tries))

    log_full_prompt("T1 facts, row 0, qid %s" % sub[0]["qid"], p_facts[0][0])
    log_full_prompt("T3 direct, row 0, qid %s" % sub[0]["qid"], p_direct[0][0])

    meta = OrderedDict([
        ("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", freeze_md5)])),
        ("script", OrderedDict([("path", os.path.relpath(__file__)), ("md5", script_md5)])),
        ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("prompts", OrderedDict([
            ("t_system_md5", md5_text(T_SYSTEM)),
            ("t_user_template_md5", md5_text(T_USER_TEMPLATE)),
            ("student_system_message_md5", md5_text(SYSTEM_MESSAGE)),
            ("t1_max_tokens", T1_MAX_TOKENS), ("t3_max_tokens", T3_MAX_TOKENS),
        ])),
        ("verdict_filter", OrderedDict([("regex", VERDICT_RE.pattern),
                                        ("yes_no_regex", YESNO_RE.pattern),
                                        ("case_insensitive", True),
                                        ("max_facts", MAX_FACTS)])),
        ("derangement", OrderedDict([("seed", DERANGE_SEED), ("n", len(dmap)),
                                     ("shuffles", tries)])),
        ("leak_guard", OrderedDict([("prompts_checked", len(p_facts) + len(p_direct)),
                                    ("fatal_hits", 0)])),
    ])
    print("[preflight] PASS", flush=True)
    return sub, p_facts, p_direct, dmap, meta


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

    sub, p_facts, p_direct, dmap, meta = preflight(args.limit)

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
        ("job", "T"), ("mode", "TEST" if is_test else "FULL"),
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
    g1 = generate([ids for _, ids in p_facts], T1_MAX_TOKENS, "T1 — teacher facts")
    t_facts = []
    n_parse_ok = n_removed_total = 0
    for r, (text, ids), (raw, n_new, fin) in zip(sub, p_facts, g1):
        ok, ents, facts_raw = parse_teacher_json(raw)
        kept, removed = verdict_filter(facts_raw) if ok else ([], [])
        n_parse_ok += int(ok)
        n_removed_total += len(removed)
        t_facts.append(OrderedDict([
            ("qid", r["qid"]), ("arm", "T_facts"),
            ("prompt", text), ("raw", raw),
            ("parse_ok", ok), ("entities", ents),
            ("facts_parsed", facts_raw), ("facts", kept), ("removed", removed),
            ("n_facts", len(kept)), ("n_facts_parsed", len(facts_raw)),
            ("n_removed", len(removed)),
            ("truncated", fin == "length"), ("finish_reason", fin),
            ("n_gen_tokens", n_new),
        ]))
    nf = sorted(e["n_facts"] for e in t_facts)
    print("  parse_ok %d/%d (%.1f%%)   facts per row min=%d median=%d max=%d  "
          "empty=%d   removed (V5) %d   truncated %d"
          % (n_parse_ok, len(t_facts), 100.0 * n_parse_ok / len(t_facts),
             nf[0], nf[len(nf) // 2], nf[-1], sum(1 for x in nf if x == 0),
             n_removed_total, sum(1 for e in t_facts if e["truncated"])))
    print("  sample (row 0): entities %s" % t_facts[0]["entities"])
    for f in t_facts[0]["facts"]:
        print("    - %s" % f)
    frac, letters, latin = latin_fraction([f for e in t_facts for f in e["facts"]])
    print("  Latin fraction of kept facts %.4f (%d/%d letters)" % (frac, latin, letters))
    write_jsonl(os.path.join(OUT_DIR, "t_facts.jsonl"), t_facts, expect_n=len(sub))

    # ---------------- T3 ----------------
    g3 = generate([ids for _, ids in p_direct], T3_MAX_TOKENS, "T3 — teacher direct (student prompt)")
    t_direct = []
    for r, (text, ids), (raw, n_new, fin) in zip(sub, p_direct, g3):
        t_direct.append(OrderedDict([
            ("qid", r["qid"]), ("arm", "T_direct"), ("gold", r["gold"]),
            ("pred", extract_answer(raw)),
            ("pred_secondary", extract_answer_secondary(raw)),
            ("truncated", fin == "length"), ("finish_reason", fin),
            ("n_gen_tokens", n_new), ("n_facts_given", 0),
            ("prompt", text), ("generation", raw),
        ]))
    s = score(t_direct, "pred")
    print("  T-direct accuracy %.2f%%  unparsed %.2f%%  (%d/%d correct)  pred-yes %.2f%%"
          % (s["accuracy"], s["unparsed_rate"], s["correct"], s["n"], s["predicted_yes_rate"]))
    print("  sample generation (row 0, first 300 chars): %r" % t_direct[0]["generation"][:300])
    write_jsonl(os.path.join(OUT_DIR, "t_direct.jsonl"), t_direct, expect_n=len(sub))

    # ---------------- T4 ----------------
    write_json(os.path.join(OUT_DIR, "derangement.json"),
               OrderedDict([("seed", DERANGE_SEED), ("n", len(dmap)),
                            ("note", "qid -> donor qid; TF_wrong gets the donor's post-T2 facts"),
                            ("map", dmap)]))

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
    for p in WRITTEN:
        with open(p, "rb") as f:
            b = f.read()
        files[p] = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)),
                                ("lines", b.count(b"\n"))])
    seal = OrderedDict([
        ("job", "T"), ("mode", "TEST" if is_test else "FULL"), ("slurm_job_id", job_id),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("script_md5", meta["script"]["md5"]), ("freeze_md5", meta["freeze"]["md5"]),
        ("n_rows", len(sub)),
        ("t1_parse_ok", n_parse_ok), ("t2_removed_total", n_removed_total),
        ("files", files)])
    write_json(os.path.join(OUT_DIR, "T_SEAL.json"), seal)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD10 JOB T COMPLETE. Sealed %d files." % len(files))


if __name__ == "__main__":
    main()
