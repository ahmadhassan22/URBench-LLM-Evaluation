#!/usr/bin/env python3
"""
d11o_teacher.py  —  D11-O Job T (teacher, vLLM): oracle-reading facts

Contract: docs/EFBPT_D11O_ORACLE_READING_FREEZE.md (FROZEN, MD5
ebd638eb4b4e48cde00202c8488df124) §3, on top of the D10 teacher freeze
docs/EFBPT_D10_TEACHER_FREEZE.md §2 (model, vLLM args, sampling; unchanged).
Copy of d10v2_teacher.py with ONLY these changes:

  --version v1 | v2   selects the prompt pair.
      v1 = D10 T1 SYSTEM/USER (d10_teacher.py, two keys, max_tokens 512,
           strict parse, no repairs — as in D10)
      v2 = D10v2 T1-v2 SYSTEM/USER (d10v2_teacher.py, five keys, max_tokens
           768, A4.1/A5.1 parse repairs — as in D10v2)
      each with EXACTLY two edits (freeze §3):
        "a question written in Urdu"        -> "a question written in English"
        "Question (Urdu): {question_ur}"    -> "Question: {question_en}"
  teacher input = the gold DEV200 question_en (the oracle reading); the
      student still sees only the Urdu question.
  leak guard on teacher prompts: gold facts and evidence ids FATAL;
      question_en is the intended input and is NOT a needle here (freeze §3).
  no T-direct / T-direct-F, no derangement (no TF_wrong arm in D11-O).
  outputs by version: outputs/efbpt/d11o/teacher_v1/ or teacher_v2/
      (t_facts.jsonl, T_START.json, T_SEAL.json); TEST under
      outputs/efbpt/d11o/_test/teacher_v1|v2/.

Everything frozen is IMPORTED, never re-implemented:
  Urdu instruction (+MD5), die                             <- efbpt_eval_dev200
  DEV200 loader (qid = urbench_qid, question_en, gold)     <- d4_extract_facts
  leak guard, needles, Latin fraction, md5/json helpers    <- d8_reachable_ceiling
  v1 prompt pair, strict parser, verdict filter            <- d10_teacher
  v2 prompt pair, repairing parser                         <- d10v2_teacher

Modes:
  --preflight-only   CPU, no model: md5 checks, rows, prompts (tokenizer
                     only), leak guard. Writes NOTHING.
  --limit N          TEST: first N rows, outputs ONLY under
                     outputs/efbpt/d11o/_test/teacher_<version>/. Never a result.
  (no flags)         FULL: outputs/efbpt/d11o/teacher_<version>/.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d11o_teacher.py --version v1 --preflight-only
  python eval/error_analysis_tests/efbpt/d11o_teacher.py --version v2 --limit 5
  python eval/error_analysis_tests/efbpt/d11o_teacher.py --version v1
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
    INSTRUCTION_PATH, INSTRUCTION_MD5, load_instruction, die,
)
from d4_extract_facts import load_rows    # noqa: E402  qid = urbench_qid, question_en, gold
from d8_reachable_ceiling import (        # noqa: E402  same guard as D8
    leak_guard, load_leak_needles, latin_fraction,
    md5_bytes, md5_file, md5_text, pkg_versions,
)
import d10_teacher as T1V1                # noqa: E402  v1 prompt pair + strict parser
import d10v2_teacher as T1V2              # noqa: E402  v2 prompt pair + repairing parser

# ----------------------------------------------------------------------------
# Frozen constants
# ----------------------------------------------------------------------------

D11O_FREEZE_PATH = "docs/EFBPT_D11O_ORACLE_READING_FREEZE.md"
D11O_FREEZE_MD5 = "ebd638eb4b4e48cde00202c8488df124"
FREEZE_PATH = T1V1.FREEZE_PATH                      # D10 teacher freeze, §2 unchanged
FREEZE_MD5 = T1V1.FREEZE_MD5
AMEND_V2 = OrderedDict([                            # provenance of the v2 prompt / parser
    ("amendment_3", (T1V2.AMEND3_PATH, T1V2.AMEND3_MD5)),
    ("amendment_4", (T1V2.AMEND4_PATH, T1V2.AMEND4_MD5)),
    ("amendment_5", (T1V2.AMEND5_PATH, T1V2.AMEND5_MD5)),
])

# §2 teacher (identical constants in both D10 scripts; asserted below)
TEACHER_PATH = T1V2.TEACHER_PATH
VLLM_VERSION = T1V2.VLLM_VERSION
VLLM_ARGS = T1V2.VLLM_ARGS
SAMPLING = T1V2.SAMPLING
if (T1V1.TEACHER_PATH, T1V1.VLLM_VERSION, dict(T1V1.VLLM_ARGS), dict(T1V1.SAMPLING)) != \
        (TEACHER_PATH, VLLM_VERSION, dict(VLLM_ARGS), dict(SAMPLING)):
    die("D10 and D10v2 teacher constants differ; freeze §2 is not a single setting")

# freeze §3: the two edits, applied to each version's SYSTEM and USER template
EDIT_SYSTEM = ("a question written in Urdu", "a question written in English")
EDIT_USER = ("Question (Urdu): {question_ur}", "Question: {question_en}")


def oracle_prompts(system, user_template):
    if system.count(EDIT_SYSTEM[0]) != 1:
        die("SYSTEM prompt does not contain %r exactly once" % EDIT_SYSTEM[0])
    if user_template.count(EDIT_USER[0]) != 1:
        die("USER template does not contain %r exactly once" % EDIT_USER[0])
    return (system.replace(EDIT_SYSTEM[0], EDIT_SYSTEM[1]),
            user_template.replace(EDIT_USER[0], EDIT_USER[1]))


VERSIONS = OrderedDict([
    ("v1", OrderedDict([
        ("arm", "T_facts_O1"), ("source", "D10 T1 (d10_teacher.py)"),
        ("max_tokens", T1V1.T1_MAX_TOKENS),            # 512
        ("keys", ["entities", "facts"]),
        ("repairs", "none (as in D10)")])),
    ("v2", OrderedDict([
        ("arm", "T_facts_O2"), ("source", "D10v2 T1-v2 (d10v2_teacher.py, A3.1)"),
        ("max_tokens", T1V2.T1_MAX_TOKENS),            # 768
        ("keys", list(T1V2.T1_KEYS.keys())),
        ("repairs", "A4.1 brace / bracket_brace; A5.1 bracket (as in D10v2)")])),
])
MAX_FACTS = T1V2.MAX_FACTS
if T1V1.MAX_FACTS != MAX_FACTS:
    die("MAX_FACTS differs between D10 and D10v2")

# §5 V4, V5 (reported here, flagged in the student's summary too)
V4_PARSE_MIN_PCT = 95.0
V5_REMOVED_MAX = 100

OUT_ROOT_FULL = "outputs/efbpt/d11o"
OUT_ROOT_TEST = "outputs/efbpt/d11o/_test"
FORBIDDEN_WRITE_DIRS = ("outputs/efbpt/d1", "outputs/efbpt/d4", "outputs/efbpt/d6",
                        "outputs/efbpt/d8", "outputs/efbpt/d10", "outputs/efbpt/d10v2",
                        "outputs/efbpt/d11o/student", "outputs/efbpt/d11o/_test/student")

# ----------------------------------------------------------------------------
# Output helpers (same layout as D10v2; the writable root depends on the mode)
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
# Parsing: the version's own parser, normalised to one record shape
# ----------------------------------------------------------------------------

def parse(version, raw):
    """Returns (ok, question_en_teacher, unknown_terms, sub_questions, entities,
    facts_parsed, repaired). v1 rows carry "" / [] / [] / False for the keys the
    two-key prompt does not produce."""
    if version == "v1":
        ok, ents, facts = T1V1.parse_teacher_json(raw)
        return ok, "", [], [], ents, facts, False
    return T1V2.parse_teacher_json(raw)


# the T2 verdict filter is the same function in both scripts (same regexes)
if (T1V1.VERDICT_RE.pattern, T1V1.YESNO_RE.pattern) != (T1V2.VERDICT_RE.pattern, T1V2.YESNO_RE.pattern):
    die("verdict filter differs between D10 and D10v2")
verdict_filter = T1V2.verdict_filter


# ----------------------------------------------------------------------------
# Preflight (CPU, no model)
# ----------------------------------------------------------------------------

def preflight(version, limit):
    print("=" * 78)
    print("D11-O JOB T PREFLIGHT  (%s; freeze §3; D10 freeze §2)" % version)
    print("=" * 78, flush=True)

    for label, path, want in (("d11o", D11O_FREEZE_PATH, D11O_FREEZE_MD5),
                              ("d10", FREEZE_PATH, FREEZE_MD5)):
        if not os.path.exists(path):
            die("freeze file missing: " + path)
        got = md5_file(path)
        if got != want:
            die("%s freeze MD5 mismatch: expected %s, got %s" % (label, want, got))
        print("[md5] %-7s %s  %s" % (label, got, path))
    amend_md5 = OrderedDict()
    for key, (path, want) in AMEND_V2.items():
        got = md5_file(path)
        if got != want:
            die("%s MD5 mismatch: expected %s, got %s" % (key, want, got))
        amend_md5[key] = OrderedDict([("path", path), ("md5", got)])
        print("[md5] %-7s %s  %s" % (key.replace("amendment_", "amend"), got, path))
    for mod in (T1V1, T1V2):
        print("[md5] %-7s %s  %s" % ("src", md5_file(mod.__file__), os.path.relpath(mod.__file__)))
    script_md5 = md5_file(os.path.abspath(__file__))
    print("[md5] script  %s  %s" % (script_md5, os.path.relpath(__file__)))

    load_instruction()                            # MD5 asserted inside (not used by the teacher)
    print("[ok] Urdu instruction MD5 %s verified" % INSTRUCTION_MD5)

    rows = load_rows()                            # asserts 200 rows
    qids = [r["qid"] for r in rows]
    if len(set(qids)) != 200:
        die("DEV200 qids are not unique")
    for r in rows:
        if "facts" in r:
            die("loader row carries a 'facts' key; gold facts must never reach the teacher")
        if not r.get("question_en", "").strip():
            die("qid %s: empty question_en (the oracle input)" % r["qid"])
    print("[ok] DEV200: 200 rows, 200 unique qids, no 'facts' key, question_en present on every row")

    sub = rows[:limit] if limit else rows

    spec = VERSIONS[version]
    if version == "v1":
        system, user_t = oracle_prompts(T1V1.T_SYSTEM, T1V1.T_USER_TEMPLATE)
    else:
        system, user_t = oracle_prompts(T1V2.T_SYSTEM, T1V2.T_USER_TEMPLATE)
    print("[prompt] %s: %s; edits applied: %r -> %r; %r -> %r"
          % (version, spec["source"], EDIT_SYSTEM[0], EDIT_SYSTEM[1], EDIT_USER[0], EDIT_USER[1]))

    # ---- tokenizer only (the model is loaded later, in run mode) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)

    def render(system_, user_):
        msgs = [{"role": "system", "content": system_},
                {"role": "user", "content": user_}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        if tok(text, add_special_tokens=False)["input_ids"] != ids:
            die("chat template: stored prompt text does not re-tokenize to the generated ids")
        return text, ids

    p_facts = [render(system, user_t.format(question_en=r["question_en"])) for r in sub]
    mx = spec["max_tokens"]
    tl = sorted(len(ids) for _, ids in p_facts)
    print("[len] %-10s n=%d tokens min=%d median=%d max=%d (+%d new <= %d)"
          % ("TO%s/facts" % version[1], len(p_facts), tl[0], tl[len(tl) // 2], tl[-1], mx,
             VLLM_ARGS["max_model_len"]))
    if tl[-1] + mx > VLLM_ARGS["max_model_len"]:
        die("prompts would exceed max_model_len")

    # ---- leak guard (freeze §3): gold facts and evidence ids FATAL;
    #      question_en is the intended input and is not a needle here ----
    needles = load_leak_needles()
    n_qen = len(needles["question_en"])
    needles["question_en"] = []
    print("[leak-guard] question_en needles removed for the teacher prompts (%d strings; the oracle input); "
          "gold facts (%d) and evidence ids (%d) remain FATAL"
          % (n_qen, len(needles["facts"]), len(needles["evidence_ids"])))
    leak_guard("TO%s/facts" % version[1], [t for t, _ in p_facts], needles, fatal_gold_facts=True)
    # every prompt must contain exactly its own gold question_en and no other row's
    qen_all = [r["question_en"] for r in rows]
    for r, (t, _) in zip(sub, p_facts):
        if ("Question: %s\n" % r["question_en"]) not in t:
            die("qid %s: prompt lacks its own question_en" % r["qid"])
        others = [q for q in qen_all if q != r["question_en"] and q in t]
        if others:
            die("qid %s: prompt contains another row's question_en %r" % (r["qid"], others[0]))
    print("[ok] each prompt carries exactly its own gold question_en")

    log_full_prompt("TO%s facts, row 0, qid %s" % (version[1], sub[0]["qid"]), p_facts[0][0])

    meta = OrderedDict([
        ("d11o_freeze", OrderedDict([("path", D11O_FREEZE_PATH), ("md5", D11O_FREEZE_MD5)])),
        ("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", FREEZE_MD5)])),
        ("script", OrderedDict([("path", os.path.relpath(__file__)), ("md5", script_md5)])),
        ("source_scripts", OrderedDict((os.path.relpath(m.__file__), md5_file(m.__file__)) for m in (T1V1, T1V2))),
        ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("prompts", OrderedDict([
            ("version", version), ("arm", spec["arm"]), ("source", spec["source"]),
            ("teacher_input", "gold DEV200 question_en (oracle reading, freeze §1)"),
            ("edits", OrderedDict([("system", list(EDIT_SYSTEM)), ("user", list(EDIT_USER))])),
            ("t_system_md5", md5_text(system)),
            ("t_user_template_md5", md5_text(user_t)),
            ("keys", spec["keys"]), ("max_tokens", mx), ("repairs", spec["repairs"]),
            ("t_direct", "not run (D11-O has no teacher-direct arm)"),
            ("derangement", "not run (D11-O has no TF_wrong arm)"),
        ])),
        ("verdict_filter", OrderedDict([("regex", T1V2.VERDICT_RE.pattern),
                                        ("yes_no_regex", T1V2.YESNO_RE.pattern),
                                        ("case_insensitive", True),
                                        ("max_facts", MAX_FACTS),
                                        ("applied_to", "facts only")])),
        ("leak_guard", OrderedDict([("prompts_checked", len(p_facts)),
                                    ("needles", "gold facts + evidence ids (question_en is the input)"),
                                    ("fatal_hits", 0)])),
    ])
    for k, v in amend_md5.items():
        meta[k] = v
    print("[preflight] PASS", flush=True)
    return sub, p_facts, system, user_t, meta


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(VERSIONS),
                    help="v1 = D10 T1 prompt pair; v2 = D10v2 T1-v2 prompt pair (freeze §3)")
    ap.add_argument("--preflight-only", action="store_true",
                    help="CPU: checks and prompts, write nothing, exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="TEST: first N DEV200 rows; writes only under %s/teacher_<version>/" % OUT_ROOT_TEST)
    args = ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        die("--limit must be positive")
    if args.preflight_only and args.limit is not None:
        die("--preflight-only takes no --limit")

    version = args.version
    is_test = args.limit is not None
    OUT_DIR = os.path.join(OUT_ROOT_TEST if is_test else OUT_ROOT_FULL, "teacher_%s" % version)

    sub, p_facts, system, user_t, meta = preflight(version, args.limit)

    if args.preflight_only:
        print("\n[preflight-only] done. No files written.")
        return

    print("\n[mode] %s  rows=%d  out=%s"
          % ("TEST — NOT A RESULT" if is_test else "FULL", len(sub), OUT_DIR))
    if not is_test and len(sub) != 200:
        die("FULL run must cover 200 rows")
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    spec = VERSIONS[version]

    # ---------------- vLLM ----------------
    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != VLLM_VERSION:
        die("vLLM %s found, freeze requires %s" % (vllm.__version__, VLLM_VERSION))

    versions = pkg_versions()
    versions["vllm"] = vllm.__version__

    start = OrderedDict([
        ("job", "T"), ("diagnostic", "D11-O oracle reading"), ("version", version),
        ("mode", "TEST" if is_test else "FULL"),
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

    params = SamplingParams(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                            seed=SAMPLING["seed"], max_tokens=spec["max_tokens"])
    print("\n" + "=" * 70)
    print("TO%s — teacher facts from the gold English question  (n=%d, max_tokens=%d, greedy, max_num_seqs=%d)"
          % (version[1], len(p_facts), spec["max_tokens"], VLLM_ARGS["max_num_seqs"]))
    print("=" * 70, flush=True)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": ids} for _, ids in p_facts], params, use_tqdm=False)
    if len(outs) != len(p_facts):
        die("vLLM returned %d outputs for %d prompts" % (len(outs), len(p_facts)))
    el = time.time() - t0
    print("  %d rows in %.1fs (%.1fs/row)" % (len(outs), el, el / max(len(outs), 1)), flush=True)

    t_facts = []
    n_parse_ok = n_removed_total = n_repaired = 0
    for r, (text, _), o in zip(sub, p_facts, outs):
        c = o.outputs[0]
        raw, n_new, fin = c.text, len(c.token_ids), c.finish_reason
        ok, q_en, unknown, subqs, ents, facts_raw, repaired = parse(version, raw)
        kept, removed = verdict_filter(facts_raw) if ok else ([], [])
        n_parse_ok += int(ok)
        n_repaired += int(bool(repaired))
        n_removed_total += len(removed)
        t_facts.append(OrderedDict([
            ("qid", r["qid"]), ("arm", spec["arm"]), ("teacher_version", version),
            ("teacher_input_question_en", r["question_en"]),      # the oracle input (audit)
            ("prompt", text), ("raw", raw),
            ("parse_ok", ok), ("repaired", repaired),
            # stored for audit only; never enter any student prompt
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
    print("  repaired rows %d/%d %s"
          % (n_repaired, len(t_facts), dict(Counter(e["repaired"] for e in t_facts if e["repaired"]))))
    start["prompts"]["t1_repaired_rows"] = n_repaired
    write_json(os.path.join(OUT_DIR, "T_START.json"), start)
    print("  facts-per-row distribution %s" % dict(sorted(Counter(nf).items())))
    if version == "v2":
        print("  rows with >=1 unknown_term %d; sub_questions per row %s"
              % (sum(1 for e in t_facts if e["n_unknown_terms"] > 0),
                 dict(sorted(Counter(e["n_sub_questions"] for e in t_facts).items()))))
        print("  sample (row 0): question_en (teacher restatement) %r" % t_facts[0]["question_en"])
    print("  sample (row 0): entities %s" % t_facts[0]["entities"])
    for f in t_facts[0]["facts"]:
        print("    - %s" % f)
    frac, letters, latin = latin_fraction([f for e in t_facts for f in e["facts"]])
    print("  Latin fraction of kept facts %.4f (%d/%d letters)" % (frac, latin, letters))
    write_jsonl(os.path.join(OUT_DIR, "t_facts.jsonl"), t_facts, expect_n=len(sub))

    parse_pct = 100.0 * n_parse_ok / len(t_facts)
    print("\nV4 teacher JSON parse rate %d/%d = %.1f%% (min %.0f%%) -> %s"
          % (n_parse_ok, len(t_facts), parse_pct, V4_PARSE_MIN_PCT,
             "ok" if parse_pct >= V4_PARSE_MIN_PCT else "FORMAT FLAG"))
    print("V5 verdict-filter removed lines %d (max %d) -> %s"
          % (n_removed_total, V5_REMOVED_MAX,
             "ok" if n_removed_total <= V5_REMOVED_MAX else "TEACHER-ANSWERING FLAG"))
    print("finish reasons: %s" % dict(Counter(e["finish_reason"] for e in t_facts)))

    # ---------------- SEAL ----------------
    files = OrderedDict()
    for p in OrderedDict.fromkeys(WRITTEN):      # T_START.json is written twice
        with open(p, "rb") as f:
            b = f.read()
        files[p] = OrderedDict([("md5", md5_bytes(b)), ("bytes", len(b)),
                                ("lines", b.count(b"\n"))])
    seal = OrderedDict([
        ("job", "T"), ("diagnostic", "D11-O oracle reading"), ("version", version),
        ("mode", "TEST" if is_test else "FULL"),
        ("slurm_job_id", job_id),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("script_md5", meta["script"]["md5"]),
        ("d11o_freeze_md5", meta["d11o_freeze"]["md5"]), ("freeze_md5", meta["freeze"]["md5"]),
        ("n_rows", len(sub)),
        ("t1_parse_ok", n_parse_ok), ("t1_repaired_rows", n_repaired),
        ("t2_removed_total", n_removed_total),
        ("files", files)])
    write_json(os.path.join(OUT_DIR, "T_SEAL.json"), seal)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD11-O JOB T (%s) COMPLETE. Sealed %d files." % (version, len(files)))


if __name__ == "__main__":
    main()
