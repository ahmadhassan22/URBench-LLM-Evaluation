#!/usr/bin/env python3
"""
d13_common.py — shared constants, loaders, prompt builders, leak guards and
row-file helpers for D13 (baselines and transfer after the D11 held-out test).

Contract: docs/EFBPT_D13_BASELINES_TRANSFER_FREEZE.md (MD5 pinned below).
Everything frozen is IMPORTED and nothing in it is changed:
  item sets, readings, leak needles            <- d11_common
  student model path, system message, instruction, max_new   <- efbpt_eval_dev200
  user layouts "A" / "B", facts header         <- d1_eval_arms
  leak guard                                   <- d8_reachable_ceiling
  facts-in-prompt check                        <- d11o_student
  teacher path, vLLM arguments, sampling, the T-direct-F line  <- d10_teacher
  the RO line label                            <- d11_ablate_ro
  the Qwen3-14B A / TR prompts with the Amendment 6 guard      <- d11_student.preflight

Nothing here scores anything: no extractor is imported, no gold label is
written to any D13 row file (freeze §7.6). Nothing here reads the API key.
"""
import json
import os
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from d11_common import (  # noqa: E402
    load_set_rows, needles_for, load_readings, verify_contracts as d11_verify_contracts,
    die, md5_file, git_state, read_jsonl, file_entry, TEST_N,
)
from efbpt_eval_dev200 import (  # noqa: E402
    MODEL_PATH, SYSTEM_MESSAGE, MAX_NEW_TOKENS, INSTRUCTION_PATH, INSTRUCTION_MD5, load_instruction,
)
from d1_eval_arms import build_user, FACTS_HEADER  # noqa: E402
from d8_reachable_ceiling import leak_guard  # noqa: E402
from d11o_student import facts_in_prompt  # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

# ----------------------------------------------------------------------------
# Contracts (freeze §0, §1)
# ----------------------------------------------------------------------------
D13_FREEZE_PATH = "docs/EFBPT_D13_BASELINES_TRANSFER_FREEZE.md"
D13_FREEZE_MD5 = "2ee24716b0e0ca2db0cc5f8fb986ac08"
EXTRA_CONTRACTS = (
    ("d13_freeze", D13_FREEZE_PATH, D13_FREEZE_MD5),
    ("d11_amendment_3", "docs/EFBPT_D11_RTG_AMENDMENT_3.md", "39cbac213583c9d08ab3be7c07a97f3c"),
    ("d11_amendment_5", "docs/EFBPT_D11_RTG_AMENDMENT_5.md", "45c118de9804390d12853f231e7632a0"),
    ("d11_amendment_6", "docs/EFBPT_D11_RTG_AMENDMENT_6.md", "bbc3d856ff9e9e7a745a90342d5b0519"),
    ("d10_freeze", "docs/EFBPT_D10_TEACHER_FREEZE.md", "b3a822d8199034c7fe58d76cf5cc428f"),
    ("d10_amendment_1", "docs/EFBPT_D10_AMENDMENT_1.md", "ac828661e26d77ca9bf00695ce4b8b93"),
    ("d8_freeze", "docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md", "d95438fac2cb097d38cccdf5db6a3d35"),
)

# freeze §1: the sealed D11-R reading files
READING_MD5 = {
    ("R1", "dev200"): "1f5b2db3c1403f63b3525a1562edd176",
    ("R1", "eval458"): "3211b45ff4bf6b75f07d7ed60d4991eb",
    ("R3", "dev200"): "0d06ec1a926fe3c186c125ad723139cf",
    ("R3", "eval458"): "9833dc91c9a2ee42c667f4f70747dbec",
}
READER_KEYS = ("R1", "R3")

# freeze §1: sealed files, read only
SEALED = OrderedDict([
    ("eval458/A_r2_eval", ("outputs/efbpt/d11/student/eval458/d11_armA_r2_eval.jsonl", "f139ba9247cdd9fef191f00d404d8212")),
    ("eval458/TR", ("outputs/efbpt/d11/student/eval458/d11_armTR.jsonl", "e38d64114089a88b2deb202de0240ca9")),
    ("eval458/TR_L1_eval", ("outputs/efbpt/d11/student/eval458/d11_armTR_L1_eval.jsonl", "4a1b2118bbcaa9348713a600b9281733")),
    ("dev200/A_r2", ("outputs/efbpt/d10/student/d10_armA_r2.jsonl", "7ccec942c8f0eee08156d258baaf806d")),
    ("dev200/TR", ("outputs/efbpt/d11/student/dev200/d11_armTR.jsonl", "bab987cc492012f4e32dbcd348be9668")),
    ("dev200/TR_L1", ("outputs/efbpt/d11/student/dev200/d11_armTR_L1.jsonl", "b8db599f8c4b427ddcb2d772fd8f09aa")),
    ("dev200/T_direct_F", ("outputs/efbpt/d10/teacher/t_direct_f.jsonl", "40ebfe643e244c5a1a3fff729f23462c")),
    ("dev200/API_direct_ur", ("outputs/efbpt/d11/api_direct/dev200/api_direct_ur.jsonl", "31c82fa180a2677ce9df85db2ac3e0d7")),
    ("dev200/RO", ("outputs/efbpt/d11/ablate_ro/dev200/d11_armRO.jsonl", "d3a8850836389d89eac5369601de2f98")),
    ("dev200/t_facts_R1", ("outputs/efbpt/d11/teacher/dev200/R1/t_facts.jsonl", "160406e12205635e69f5c95140f9936f")),
    ("eval458/t_facts_R1", ("outputs/efbpt/d11/teacher/eval458/R1/t_facts.jsonl", "4e2c4f30fd9603711ded98df90fdc477")),
])
# freeze §3, L5: the 50 rows the old answer-free probes used
FIRST50_FILE = ("data/sdfr_splits/strategyqa_eval.jsonl", "8f26dae65cfe22678b46359ff9ad43d2")
FIRST50_N, L5_ROWS = 50, 408

SET_SHAPE = {"dev200": (200, 87, 113), "eval458": (458, 221, 237)}
SETS = ("dev200", "eval458")
MODES = ("TEST", "FULL")

# ----------------------------------------------------------------------------
# Arms (freeze §2)
# ----------------------------------------------------------------------------
BOTH = ("dev200", "eval458")
EVAL = ("eval458",)
ARMS = OrderedDict([
    # Group 1, large model alone
    ("T_direct_F_eval", OrderedDict([("group", "large_alone"), ("kind", "teacher"), ("sets", EVAL), ("input", "question_ur")])),
    ("T_tr_R1", OrderedDict([("group", "large_alone"), ("kind", "teacher"), ("sets", BOTH), ("input", "reading"), ("reader", "R1")])),
    ("T_tr_L1", OrderedDict([("group", "large_alone"), ("kind", "teacher"), ("sets", BOTH), ("input", "reading"), ("reader", "R3")])),
    ("API_direct_ur_eval", OrderedDict([("group", "large_alone"), ("kind", "api"), ("sets", EVAL), ("input", "question_ur")])),
    ("API_tr_R1", OrderedDict([("group", "large_alone"), ("kind", "api"), ("sets", BOTH), ("input", "reading"), ("reader", "R1")])),
    # Group 2, traditional RAG (student Qwen3-14B)
    ("RAG_ur", OrderedDict([("group", "rag"), ("kind", "qwen14"), ("sets", BOTH), ("input", "question_ur")])),
    ("RAG_en", OrderedDict([("group", "rag"), ("kind", "qwen14"), ("sets", BOTH), ("input", "reading"), ("reader", "R1")])),
    # Group 3, ablation
    ("RO_eval", OrderedDict([("group", "ablation"), ("kind", "qwen14"), ("sets", EVAL), ("input", "reading"), ("reader", "R1")])),
    # Group 4, transfer to other students
    ("Llama8B_A", OrderedDict([("group", "transfer"), ("kind", "other"), ("sets", EVAL), ("student", "llama8"), ("layout", "A")])),
    ("Llama8B_TR", OrderedDict([("group", "transfer"), ("kind", "other"), ("sets", EVAL), ("student", "llama8"), ("layout", "TR")])),
    ("Gemma9B_A", OrderedDict([("group", "transfer"), ("kind", "other"), ("sets", EVAL), ("student", "gemma9"), ("layout", "A")])),
    ("Gemma9B_TR", OrderedDict([("group", "transfer"), ("kind", "other"), ("sets", EVAL), ("student", "gemma9"), ("layout", "TR")])),
    ("Qwen8B_A", OrderedDict([("group", "transfer"), ("kind", "other"), ("sets", EVAL), ("student", "qwen8"), ("layout", "A")])),
    ("Qwen8B_TR", OrderedDict([("group", "transfer"), ("kind", "other"), ("sets", EVAL), ("student", "qwen8"), ("layout", "TR")])),
])
QWEN14_ARMS = tuple(a for a, s in ARMS.items() if s["kind"] == "qwen14")
TEACHER_ARMS = tuple(a for a, s in ARMS.items() if s["kind"] == "teacher")
API_ARMS = tuple(a for a, s in ARMS.items() if s["kind"] == "api")
RAG_ARMS = ("RAG_ur", "RAG_en")

_DM = "/mnt/home/user41/downloaded_models/"
OTHER_STUDENTS = OrderedDict([   # freeze §7.4
    ("llama8", OrderedDict([("name", "Llama8B"), ("label", "Llama-3.1-8B-Instruct"),
                            ("path", _DM + "LLM-Research/Meta-Llama-3___1-8B-Instruct"),
                            ("end_of_turn", "<|eot_id|>"), ("system_role", True), ("thinking_switch", False), ("context", 131072)])),
    ("gemma9", OrderedDict([("name", "Gemma9B"), ("label", "Gemma-2-9B-it"),
                            ("path", _DM + "AI-ModelScope/gemma-2-9b-it"),
                            ("end_of_turn", "<end_of_turn>"), ("system_role", False), ("thinking_switch", False), ("context", 8192)])),
    ("qwen8", OrderedDict([("name", "Qwen8B"), ("label", "Qwen3-8B"),
                           ("path", _DM + "Qwen/Qwen3-8B"),
                           ("end_of_turn", "<|im_end|>"), ("system_role", True), ("thinking_switch", True), ("context", 40960)])),
])
QWEN14_CONTEXT = 40960

# freeze §7.1: the T_tr user message = these three parts joined by blank lines
T_TR_INSTRUCTION = "Answer this question."
T_TR_FINAL_LINE = "At the end, write on a separate line: Final answer: yes or Final answer: no"
# freeze §2 / §7.2
API_TR_SYSTEM = "Answer only yes or no."
# freeze §2 / §7.3
RAG_TOP_K = 5
RAG_NTOTAL = 23963971
RETRIEVE_PY = ("rag/retrieve.py", "c1ecdf1df7ebf8738425cb836bf3d218")

SLOT = "@@D13_SLOT@@"                     # stands for the variable part of a prompt in the template checks

OUT_ROOT = "outputs/efbpt/d13"
TEST_ROOT = os.path.join(OUT_ROOT, "_test")
REGRESS_ROOT = os.path.join(OUT_ROOT, "_regress")

FROZEN_IMPORTS = ("d11_common.py", "d11_student.py", "d10_teacher.py", "d11_ablate_ro.py", "d11_api_direct.py",
                  "d1_eval_arms.py", "efbpt_eval_dev200.py", "d4_extract_facts.py", "d8_reachable_ceiling.py",
                  "d11o_student.py", "d3_oracle_retrieval.py", "d11r_common.py")
D13_SCRIPTS = ("d13_common.py", "d13_preflight.py", "d13_api.py", "d13_teacher.py", "d13_retrieve.py", "d13_student.py")


# ----------------------------------------------------------------------------
# Contracts, scripts, paths
# ----------------------------------------------------------------------------
def verify_contracts():
    """The D11 contracts (d11_common) plus the D13 freeze and the documents it cites, by pinned MD5."""
    out = d11_verify_contracts()
    for key, path, want in EXTRA_CONTRACTS:
        if not os.path.exists(path):
            die("contract missing: " + path)
        got = md5_file(path)
        if got != want:
            die("%s MD5 %s != pinned %s" % (path, got, want))
        out[key] = OrderedDict([("path", path), ("md5", got)])
        print("[md5] %-18s %s  %s" % (key, got, path))
    return out


def script_md5s():
    out = OrderedDict()
    for name in D13_SCRIPTS + FROZEN_IMPORTS:
        p = os.path.join(_HERE, name)
        out[os.path.relpath(p, ROOT)] = md5_file(p) if os.path.exists(p) else None
    out[RETRIEVE_PY[0]] = md5_file(os.path.join(ROOT, RETRIEVE_PY[0]))
    return out


def require_committed():
    """A FULL run needs the d13 scripts committed and unchanged, so that every sealed output is tied to committed code."""
    import subprocess
    pat = "eval/error_analysis_tests/efbpt/d13_*"
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", pat], cwd=ROOT).decode().strip()
        tracked = subprocess.check_output(["git", "ls-files", "--", pat], cwd=ROOT).decode().split()
    except Exception as e:
        die("git is not usable (%s); a FULL run needs the committed d13 scripts" % type(e).__name__)
    missing = [n for n in D13_SCRIPTS if ("eval/error_analysis_tests/efbpt/" + n) not in tracked]
    if dirty or missing:
        die("FULL run refused: the d13 scripts are not committed and clean (untracked/modified: %s; not tracked: %s)"
            % (dirty.replace("\n", "; ") or "-", ", ".join(missing) or "-"))
    print("[ok] d13 scripts are committed and unchanged (HEAD %s)" % git_state()["head"])


def check_arm_set(arm, set_name, mode):
    if arm not in ARMS:
        die("unknown arm %r" % arm)
    if mode not in MODES:
        die("unknown mode %r" % mode)
    if mode == "TEST":
        if set_name != "dev200":
            die("TEST-5 uses DEV200 rows only (freeze C4); got set %r" % set_name)
    elif set_name not in ARMS[arm]["sets"]:
        die("arm %s is not defined on %s (freeze §2: %s)" % (arm, set_name, ", ".join(ARMS[arm]["sets"])))


def arm_dir(arm, set_name, mode):
    check_arm_set(arm, set_name, mode)
    g = ARMS[arm]["group"]
    return os.path.join(TEST_ROOT, g, "dev200") if mode == "TEST" else os.path.join(OUT_ROOT, g, set_name)


def arm_paths(arm, set_name, mode):
    d = arm_dir(arm, set_name, mode)
    return OrderedDict([("dir", d), ("rows", os.path.join(d, "d13_%s.jsonl" % arm)),
                        ("start", os.path.join(d, "START_%s.json" % arm)), ("seal", os.path.join(d, "SEAL_%s.json" % arm))])


def retrieval_paths(arm, set_name, mode):
    d = os.path.join(TEST_ROOT if mode == "TEST" else OUT_ROOT, "rag", set_name, "retrieval")
    return OrderedDict([("dir", d), ("rows", os.path.join(d, "retrieval_%s.jsonl" % arm)),
                        ("seal", os.path.join(d, "RETRIEVAL_SEAL_%s.json" % arm)),
                        ("title_map", os.path.join(d, "title_map.jsonl")), ("title_seal", os.path.join(d, "TITLE_MAP_SEAL.json"))])


def guard_out(path):
    """Every D13 write goes under outputs/efbpt/d13/ and nowhere else."""
    ap = os.path.abspath(path)
    root = os.path.abspath(OUT_ROOT)
    if not ap.startswith(root + os.sep):
        die("refusing to write outside %s: %s" % (OUT_ROOT, path))
    return ap


def write_json(path, obj):
    ap = guard_out(path)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    tmp = ap + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ap)


def record_line(rec):
    return (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")


def read_rows(path, repair=False):
    """Completed rows of a row file. A row is complete when its line ends with a
    newline (each row is flushed and fsynced). repair=True (a starting job) cuts
    off a torn tail, which is the unfinished row of a crashed job."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        b = f.read()
    end = b.rfind(b"\n") + 1
    if end != len(b):
        if not repair:
            die("%s ends with a torn line (%d bytes)" % (path, len(b) - end))
        print("[resume] %s: torn tail of %d bytes (an unfinished row) cut off; completed rows untouched" % (path, len(b) - end), flush=True)
        with open(guard_out(path), "r+b") as f:
            f.truncate(end)
            f.flush()
            os.fsync(f.fileno())
        b = b[:end]
    return [json.loads(l.decode("utf-8"), object_pairs_hook=OrderedDict) for l in b.split(b"\n") if l]


def append_row(f, rec):
    f.write(record_line(rec))
    f.flush()
    os.fsync(f.fileno())


def open_arm(arm, set_name, mode, expected, start_obj, prompt_key="prompt"):
    """Start or resume one arm. expected = the rows of this run in order, each with
    'qid' and the prompt (prompt_key). Returns the number of finished rows, or None
    if the arm is already sealed (a sealed file is never rewritten). Finished rows
    are verified against the prompts of this run and are never generated again."""
    pth = arm_paths(arm, set_name, mode)
    os.makedirs(guard_out(pth["dir"]), exist_ok=True)
    if os.path.exists(pth["seal"]):
        seal = json.load(open(pth["seal"], encoding="utf-8"), object_pairs_hook=OrderedDict)
        ent = seal["files"].get(pth["rows"])
        if ent is None or file_entry(pth["rows"])["md5"] != ent["md5"]:
            die("%s is sealed but its row file does not match the seal" % arm)
        print("[sealed] %s %s is already sealed (%d rows); nothing is generated again" % (arm, set_name, seal["n_rows"]), flush=True)
        return None
    done = read_rows(pth["rows"], repair=True)
    if len(done) > len(expected):
        die("%s: %d rows on disk, %d expected" % (pth["rows"], len(done), len(expected)))
    for i, (d, e) in enumerate(zip(done, expected)):
        if d["qid"] != e["qid"] or d.get(prompt_key) != e[prompt_key]:
            die("%s row %d: the finished row does not match the prompt of this run (qid %s)" % (pth["rows"], i, e["qid"]))
    run = OrderedDict([("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("rows_already_done", len(done)),
                       ("slurm_job_id", os.environ.get("SLURM_JOB_ID")), ("scripts", start_obj.get("scripts")), ("git", start_obj.get("git"))])
    if os.path.exists(pth["start"]):
        cur = json.load(open(pth["start"], encoding="utf-8"), object_pairs_hook=OrderedDict)
        cur.setdefault("runs", []).append(run)
        start_obj = cur
    else:
        start_obj = OrderedDict(start_obj)
        start_obj["runs"] = [run]
    write_json(pth["start"], start_obj)
    if done:
        print("[resume] %s %s: %d finished rows kept; continuing at row %d" % (arm, set_name, len(done), len(done)), flush=True)
    return len(done)


def seal_arm(arm, set_name, mode, expected, extra=None):
    """Write SEAL_<ARM>.json when every row is present. Not scored: the seal holds counts, MD5s, seconds and tokens only."""
    pth = arm_paths(arm, set_name, mode)
    recs = read_rows(pth["rows"])
    if [r["qid"] for r in recs] != [e["qid"] for e in expected]:
        die("%s: %d rows on disk do not equal the %d rows of the run in order" % (pth["rows"], len(recs), len(expected)))
    secs = [r.get("seconds") for r in recs if isinstance(r.get("seconds"), (int, float))]
    toks = [r.get("n_gen_tokens") for r in recs if isinstance(r.get("n_gen_tokens"), (int, float))]
    s = OrderedDict([
        ("experiment", "D13"), ("arm", arm), ("group", ARMS[arm]["group"]), ("set", set_name), ("mode", mode),
        ("complete", True), ("n_rows", len(recs)), ("scored", False),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("slurm_job_id", os.environ.get("SLURM_JOB_ID")),
        ("d13_freeze_md5", D13_FREEZE_MD5), ("scripts", script_md5s()), ("git", git_state()),
        ("seconds_total", round(sum(secs), 3) if secs else None), ("gen_tokens_total", int(sum(toks)) if toks else None),
        ("rows_truncated", sum(1 for r in recs if r.get("truncated"))),
    ])
    if extra:
        s.update(extra)
    s["files"] = OrderedDict((p, file_entry(p)) for p in (pth["start"], pth["rows"]))
    write_json(pth["seal"], s)
    print("  [sealed] %s (%d rows; not scored) rows MD5 %s" % (pth["seal"], len(recs), s["files"][pth["rows"]]["md5"]), flush=True)
    return s


def no_facts(row):
    return OrderedDict((k, v) for k, v in row.items() if k != "facts")


# ----------------------------------------------------------------------------
# Context: rows, needles, readings of one set
# ----------------------------------------------------------------------------
def make_ctx(set_name, mode):
    """Rows (all, and the rows of this run), leak needles and the sealed R1 / R3
    readings of one set. TEST = DEV200 rows 0-4 (freeze C4)."""
    if mode == "TEST" and set_name != "dev200":
        die("TEST-5 uses DEV200 rows only (freeze C4)")
    rows_all = load_set_rows(set_name)
    n_yes = sum(1 for r in rows_all if r["gold"] == "yes")
    if (len(rows_all), n_yes, len(rows_all) - n_yes) != SET_SHAPE[set_name]:
        die("%s: %d rows, %d yes / %d no; expected %s" % (set_name, len(rows_all), n_yes, len(rows_all) - n_yes, SET_SHAPE[set_name]))
    sub = rows_all[:TEST_N] if mode == "TEST" else rows_all
    readings, rinfo = OrderedDict(), OrderedDict()
    for key in READER_KEYS:
        rd, info = load_readings(key, set_name, rows_all)
        if info["md5"] != READING_MD5[(key, set_name)]:
            die("%s readings of %s: MD5 %s != the D13 freeze §1 value %s" % (key, set_name, info["md5"], READING_MD5[(key, set_name)]))
        readings[key], rinfo[key] = rd, info
    print("[ok] %s: readings R1 %s, R3 %s = R_SEAL = D13 freeze §1" % (set_name, rinfo["R1"]["md5"], rinfo["R3"]["md5"]))
    return OrderedDict([
        ("set", set_name), ("mode", mode), ("rows_all", rows_all), ("sub", sub),
        ("needles", needles_for(rows_all, set_name)), ("readings", readings), ("reading_info", rinfo),
        ("reading_strings", sorted({s for k in readings for s in readings[k].values() if s.strip()})),
        ("instruction", load_instruction()),
    ])


def verify_sealed(names=None):
    out = OrderedDict()
    for name, (path, want) in SEALED.items():
        if names is not None and name not in names:
            continue
        if not os.path.exists(path):
            die("sealed file missing: " + path)
        got = md5_file(path)
        if got != want:
            die("sealed %s MD5 %s != the D13 freeze §1 value %s" % (path, got, want))
        out[name] = OrderedDict([("path", path), ("md5", got)])
    return out


# ----------------------------------------------------------------------------
# Leak guards (freeze §7.5)
# ----------------------------------------------------------------------------
def guard_no_reading(label, texts, reading_strings):
    for i, t in enumerate(texts):
        for rd in reading_strings:
            if rd in t:
                die("LEAK: %s prompt %d contains a reading %r" % (label, i, rd[:80]))


def guard_question_prompts(label, prompts, ctx):
    """Arms whose input is the Urdu question: no question_en, no gold fact, no
    evidence id and no reading string anywhere in the prompt. Fatal."""
    leak_guard(label, prompts, ctx["needles"], fatal_gold_facts=True)
    guard_no_reading(label, prompts, ctx["reading_strings"])


def guard_template(label, text, ctx):
    """D11 A1.1 (a): the prompt with its variable slot blanked has no question_en,
    no gold fact, no evidence id, no required title and no reading. Fatal."""
    leak_guard(label + "/template", [text], ctx["needles"], fatal_gold_facts=True)
    m = ctx["needles"]["title_re"].search(text)
    if m:
        die("LEAK: %s template contains a required title %r" % (label, m.group(0)))
    guard_no_reading(label + "/template", [text], ctx["reading_strings"])


def guard_readings(label, ctx, reader_key):
    """D11 A1.1 (b), (c): each reading used has no other row's question_en, no gold
    fact and no evidence id (fatal); (c) holds because the readings were loaded by
    d11_common.load_readings (MD5 = R_SEAL = freeze). Own-row equality is counted."""
    nd, rd = ctx["needles"], ctx["readings"][reader_key]
    eq = 0
    for r in ctx["sub"]:
        s = rd[r["qid"]]
        for f in nd["facts"]:
            if f in s:
                die("LEAK: %s qid %s: the reading contains gold fact %r" % (label, r["qid"], f))
        m = nd["evidence_re"].search(s)
        if m:
            die("LEAK: %s qid %s: the reading contains evidence id %r" % (label, r["qid"], m.group(0)))
        for q in nd["question_en"]:
            if q in s and q != r["question_en"]:
                die("LEAK: %s qid %s: the reading contains another row's question_en" % (label, r["qid"]))
        eq += int(s.strip() == r["question_en"].strip())
    print("[leak-guard] %-18s %d readings (%s): no gold fact, no evidence id, no other-row question_en; equal to own question_en %d (reported)"
          % (label, len(ctx["sub"]), reader_key, eq), flush=True)
    return OrderedDict([("reader", reader_key), ("readings_checked", len(ctx["sub"])), ("readings_equal_own_question_en", eq)])


# ----------------------------------------------------------------------------
# Prompt builders (CPU; tokenizers only)
# ----------------------------------------------------------------------------
def load_tok(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def chat14(tok, user):
    """The Qwen3-14B student prompt exactly as D10 / D11 render it."""
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def check_len(label, tok, prompts, max_new, context):
    tl = sorted(n_tokens(tok, p) for p in prompts)
    print("[len] %-18s n=%d tokens min=%d median=%d max=%d (+%d new <= %d)" % (label, len(tl), tl[0], tl[len(tl) // 2], tl[-1], max_new, context))
    if tl[-1] + max_new > context:
        die("%s prompts would exceed the context window" % label)
    return OrderedDict([("min", tl[0]), ("median", tl[len(tl) // 2]), ("max", tl[-1])])


def t_tr_user(reading):
    return T_TR_INSTRUCTION + "\n\n" + reading + "\n\n" + T_TR_FINAL_LINE


def build_teacher_prompts(arm, ctx, tok):
    """Teacher arms (freeze §7.1). Returns ([{qid, prompt, ids}], info)."""
    from d10_teacher import TDF_LINE, T3_MAX_TOKENS, VLLM_ARGS
    spec, sub = ARMS[arm], ctx["sub"]

    def render(system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        if tok(text, add_special_tokens=False)["input_ids"] != ids:
            die("teacher chat template: the stored prompt text does not re-tokenize to the generated ids")
        return text, ids

    info = OrderedDict([("arm", arm), ("input", spec["input"])])
    if arm == "T_direct_F_eval":
        ps = [render(SYSTEM_MESSAGE, build_user(ctx["instruction"], no_facts(r), None, "A") + "\n\n" + TDF_LINE) for r in sub]
        guard_question_prompts(arm, [t for t, _ in ps], ctx)
        if ctx["set"] == "dev200":
            path, _ = SEALED["dev200/T_direct_F"]
            verify_sealed(("dev200/T_direct_F",))
            sealed = {e["qid"]: e for e in read_jsonl(path)}
            for r, (t, _) in zip(sub, ps):
                if sealed[r["qid"]]["prompt"] != t:
                    die("T_direct_F prompt rebuilt here differs from the sealed D10 prompt for qid %s (freeze §7.1)" % r["qid"])
            print("[ok] T_direct_F_eval: %d DEV200 prompts are byte-identical to the sealed D10 T_direct_F prompts" % len(sub))
            info["prompts_identical_to_sealed_D10"] = len(sub)
    else:
        key = spec["reader"]
        rd = ctx["readings"][key]
        tmpl = render(SYSTEM_MESSAGE, t_tr_user(SLOT))[0]
        if tmpl.count(SLOT) != 1:
            die("%s: the template does not carry exactly one reading slot" % arm)
        guard_template(arm, tmpl.replace(SLOT, ""), ctx)
        ps = [render(SYSTEM_MESSAGE, t_tr_user(rd[r["qid"]])) for r in sub]
        for r, (t, _) in zip(sub, ps):
            if t != tmpl.replace(SLOT, rd[r["qid"]]):
                die("%s qid %s: the prompt is not the template with the reading in its slot" % (arm, r["qid"]))
        info.update(guard_readings(arm, ctx, key))
        info["user_template"] = t_tr_user("{reading}")
    info["prompt_tokens"] = check_len(arm, tok, [t for t, _ in ps], T3_MAX_TOKENS, VLLM_ARGS["max_model_len"])
    return [OrderedDict([("qid", r["qid"]), ("prompt", t), ("ids", ids)]) for r, (t, ids) in zip(sub, ps)], info


def build_api_messages(arm, ctx):
    """API arms (freeze §7.2). Returns ([{qid, system, user, prompt}], info); prompt = system + newline + user, for the row check."""
    from d11_api_direct import INSTRUCTION_UR
    spec, sub = ARMS[arm], ctx["sub"]
    info = OrderedDict([("arm", arm), ("input", spec["input"])])
    if arm == "API_direct_ur_eval":
        msgs = [(INSTRUCTION_UR, r["question_ur"]) for r in sub]
        guard_question_prompts(arm, [s + "\n" + u for s, u in msgs], ctx)
    else:
        rd = ctx["readings"][spec["reader"]]
        guard_template(arm, API_TR_SYSTEM + "\n", ctx)
        msgs = [(API_TR_SYSTEM, rd[r["qid"]]) for r in sub]
        info.update(guard_readings(arm, ctx, spec["reader"]))
    info["system_message"] = msgs[0][0]
    return [OrderedDict([("qid", r["qid"]), ("system", s), ("user", u), ("prompt", s + "\n" + u)]) for r, (s, u) in zip(sub, msgs)], info


def build_ro_prompts(ctx, tok):
    """RO_eval (freeze §7.4): layout A plus one line directly after the Urdu
    question, built as d11_ablate_ro builds arm RO. Returns ([{qid, prompt}], info)."""
    from d11_ablate_ro import RO_LABEL
    sub, rd, instruction = ctx["sub"], ctx["readings"]["R1"], ctx["instruction"]
    prompts, a_prompts = [], []
    for r in sub:
        s = rd[r["qid"]]
        for k, v in (("question_ur", r["question_ur"]), ("R1 reading", s)):
            if "\n" in v or v != v.strip() or not v:
                die("RO_eval qid %s: %s is empty, has a newline or outer white space" % (r["qid"], k))
        user_a = build_user(instruction, no_facts(r), None, "A")
        p_a = chat14(tok, user_a)
        line = RO_LABEL + s
        user = build_user(instruction, dict(no_facts(r), question_ur=r["question_ur"] + "\n" + line), None, "A")
        if user != user_a + "\n" + line or user.count("\n") != user_a.count("\n") + 1:
            die("RO_eval qid %s: the user message is not layout A + one line" % r["qid"])
        p = chat14(tok, user)
        if p.count(RO_LABEL) != 1 or p_a.replace(user_a, user) != p:
            die("RO_eval qid %s: the prompt is not the layout-A prompt with the one line added after the question" % r["qid"])
        if FACTS_HEADER in p:
            die("RO_eval qid %s: the prompt contains the facts header" % r["qid"])
        a_prompts.append(p_a)
        prompts.append(p)
    guard_question_prompts("RO_eval/A-part", a_prompts, ctx)
    info = OrderedDict([("arm", "RO_eval"), ("ro_label", RO_LABEL)])
    info.update(guard_readings("RO_eval", ctx, "R1"))
    if ctx["set"] == "dev200":
        path, _ = SEALED["dev200/RO"]
        verify_sealed(("dev200/RO",))
        sealed = {e["qid"]: e for e in read_jsonl(path)}
        for r, p in zip(sub, prompts):
            if sealed[r["qid"]]["prompt"] != p:
                die("RO prompt rebuilt here differs from the sealed DEV200 RO prompt for qid %s (freeze §7.4)" % r["qid"])
        print("[ok] RO_eval: %d DEV200 prompts are byte-identical to the sealed DEV200 RO prompts" % len(sub))
        info["prompts_identical_to_sealed_RO"] = len(sub)
    info["prompt_tokens"] = check_len("RO_eval", tok, prompts, MAX_NEW_TOKENS, QWEN14_CONTEXT)
    return [OrderedDict([("qid", r["qid"]), ("prompt", p)]) for r, p in zip(sub, prompts)], info


def one_line(s):
    return " ".join(str(s).split())


def rag_entry(hit):
    """freeze §7.3: one facts-block entry "Title: text"; white space runs become one space."""
    return one_line(hit["title"]) + ": " + one_line(hit["text"])


def rag_query(arm, row, ctx):
    return row["question_ur"] if arm == "RAG_ur" else ctx["readings"]["R1"][row["qid"]]


def guard_rag_queries(arm, ctx):
    """The retrieval query is exactly question_ur (RAG_ur) or exactly the R1 reading (RAG_en)."""
    if arm == "RAG_ur":
        guard_question_prompts("RAG_ur/query", [r["question_ur"] for r in ctx["sub"]], ctx)
        return OrderedDict([("query", "question_ur")])
    info = guard_readings("RAG_en/query", ctx, "R1")
    info["query"] = "R1 reading"
    return info


def load_retrieval(arm, ctx):
    """The sealed retrieval file of one RAG arm for the rows of this run."""
    pth = retrieval_paths(arm, ctx["set"], ctx["mode"])
    for p in (pth["rows"], pth["seal"]):
        if not os.path.exists(p):
            die("%s: retrieval file missing: %s (run d13_retrieve.py first)" % (arm, p))
    seal = json.load(open(pth["seal"], encoding="utf-8"), object_pairs_hook=OrderedDict)
    ent = seal["files"].get(pth["rows"])
    if ent is None or file_entry(pth["rows"])["md5"] != ent["md5"]:
        die("%s: the retrieval file does not match its seal" % arm)
    if seal.get("index_ntotal") != RAG_NTOTAL or seal.get("top_k") != RAG_TOP_K or seal.get("retrieve_py_md5") != RETRIEVE_PY[1]:
        die("%s: the retrieval seal does not carry the frozen retriever settings" % arm)
    recs = read_jsonl(pth["rows"])
    if [e["qid"] for e in recs] != [r["qid"] for r in ctx["sub"]]:
        die("%s: retrieval qids are not the rows of this run in order" % arm)
    for r, e in zip(ctx["sub"], recs):
        if e["query"] != rag_query(arm, r, ctx):
            die("%s qid %s: the stored query is not the frozen query string" % (arm, r["qid"]))
        if len(e["hits"]) != RAG_TOP_K or [h["rank"] for h in e["hits"]] != list(range(1, RAG_TOP_K + 1)):
            die("%s qid %s: not %d hits in rank order" % (arm, r["qid"], RAG_TOP_K))
        if e["entries"] != [rag_entry(h) for h in e["hits"]]:
            die("%s qid %s: stored entries differ from 'Title: text' of the stored hits" % (arm, r["qid"]))
    return recs, OrderedDict([("file", pth["rows"]), ("md5", ent["md5"]), ("seal", pth["seal"]), ("seal_md5", md5_file(pth["seal"]))])


def build_rag_prompts(arm, ctx, tok):
    """RAG student prompts (freeze §7.3, §7.5). Returns ([{qid, prompt}], info)."""
    sub, nd, instruction = ctx["sub"], ctx["needles"], ctx["instruction"]
    recs, rinfo = load_retrieval(arm, ctx)
    prompts, blanks, logged = [], [], []
    for r, e in zip(sub, recs):
        entries = list(e["entries"])
        p = chat14(tok, build_user(instruction, dict(no_facts(r), facts=entries), None, "B"))
        blank = chat14(tok, build_user(instruction, dict(no_facts(r), facts=[SLOT]), None, "B"))
        if blank.count("- " + SLOT) != 1 or p != blank.replace("- " + SLOT, "\n".join("- " + x for x in entries)):
            die("%s qid %s: the prompt is not layout B with the %d entries as its facts block" % (arm, r["qid"], RAG_TOP_K))
        n_hdr, found = facts_in_prompt(p)
        if n_hdr != 1 or found != entries:
            die("%s qid %s: the facts block in the prompt does not equal the sealed retrieval record" % (arm, r["qid"]))
        blanks.append(blank.replace(SLOT, ""))
        prompts.append(p)
        for rank, x in enumerate(entries, 1):          # inside the retrieved block
            for q in nd["question_en"]:
                if q in x:
                    die("LEAK: %s qid %s rank %d: retrieved text contains question_en %r" % (arm, r["qid"], rank, q))
            for s in ctx["reading_strings"]:
                if s in x:
                    die("LEAK: %s qid %s rank %d: retrieved text contains a reading %r" % (arm, r["qid"], rank, s[:80]))
            for m in nd["evidence_re"].finditer(x):
                logged.append(OrderedDict([("qid", r["qid"]), ("rank", rank), ("kind", "evidence_id"), ("needle", m.group(0)), ("title", e["hits"][rank - 1]["title"])]))
            for f in nd["facts"]:
                if f in x:
                    logged.append(OrderedDict([("qid", r["qid"]), ("rank", rank), ("kind", "gold_fact"), ("needle", f), ("title", e["hits"][rank - 1]["title"])]))
    guard_question_prompts(arm + "/outside-block", blanks, ctx)       # outside the retrieved block: every needle fatal
    n_ev = sum(1 for x in logged if x["kind"] == "evidence_id")
    print("[leak-guard] %-18s inside the retrieved block: no question_en, no reading; evidence-id strings %d, gold-fact strings %d (logged, not fatal; freeze §7.5)"
          % (arm, n_ev, len(logged) - n_ev), flush=True)
    info = OrderedDict([("arm", arm), ("retrieval", rinfo), ("logged_hits_in_retrieved_text", logged)])
    info["prompt_tokens"] = check_len(arm, tok, prompts, MAX_NEW_TOKENS, QWEN14_CONTEXT)
    return [OrderedDict([("qid", r["qid"]), ("prompt", p)]) for r, p in zip(sub, prompts)], info


def d11_prompts(set_name, mode):
    """The Qwen3-14B layout-A and TR prompts of the rows of this run, built and
    guarded by the frozen d11_student.preflight (Amendment 6 included), and
    checked against the sealed D11 files. Returns a dict with the user message of
    every row in both layouts."""
    import d11_student as S11
    limit = TEST_N if mode == "TEST" else None
    if set_name == "eval458":
        if mode == "TEST":
            die("TEST-5 uses DEV200 rows only (freeze C4)")
        names = ("A_r2_eval", "TR")
        res = S11.preflight("eval458", None, False, "TR", list(names))
        sealed_names = ("eval458/A_r2_eval", "eval458/TR")
    else:
        names = ("A_r2", "TR")
        res = S11.preflight("dev200", limit, False, None, list(names))
        sealed_names = ("dev200/A_r2", "dev200/TR")
    instruction, sub, _arms, prompts, _teacher_by, _sealed_a, _dmap, tok, pre = res
    verify_sealed(sealed_names + (set_name + "/t_facts_R1",))
    out = OrderedDict([("sub", sub), ("tok14", tok), ("d11_preflight", OrderedDict([("leak_guard", pre["leak_guard"]), ("layouts", pre["layouts"])]))])
    for lay, name, sname in (("A", names[0], sealed_names[0]), ("TR", names[1], sealed_names[1])):
        sealed = {e["qid"]: e for e in read_jsonl(SEALED[sname][0])}
        users = []
        for r, (p, layout, fl, _src) in zip(sub, prompts[name]):
            if sealed[r["qid"]]["prompt"] != p:
                die("%s qid %s: the prompt rebuilt by d11_student.preflight differs from the sealed %s prompt" % (name, r["qid"], sname))
            if layout == "A":
                u = build_user(instruction, no_facts(r), None, "A")
            elif layout == "B":
                u = build_user(instruction, dict(no_facts(r), facts=list(fl)), None, "B")
            else:
                die("%s qid %s: unexpected layout %r" % (name, r["qid"], layout))
            if chat14(tok, u) != p:
                die("%s qid %s: the user message rebuilt here does not give the sealed prompt" % (name, r["qid"]))
            users.append(OrderedDict([("qid", r["qid"]), ("user", u), ("layout", layout), ("n_facts", len(fl))]))
        out[lay] = users
        print("[ok] %s %s: %d prompts equal the sealed file %s; user messages recovered" % (set_name, name, len(users), sname))
    return out


def render_student(skey, tok, user):
    spec = OTHER_STUDENTS[skey]
    if spec["system_role"]:
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]
    else:                                   # Gemma has no system role: the system text opens the user turn (freeze §7.4)
        msgs = [{"role": "user", "content": SYSTEM_MESSAGE + "\n\n" + user}]
    kw = {"enable_thinking": False} if spec["thinking_switch"] else {}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw))
    if tok(text, add_special_tokens=False)["input_ids"] != ids:
        die("%s chat template: the stored prompt text does not re-tokenize to the generated ids" % skey)
    return text


def build_other_student_prompts(skey, d11p, ctx, tok):
    """Group 4 prompts (freeze §7.4, §7.5) of one student for the rows of this run.
    Returns ({"A": [{qid, prompt, layout, n_facts}], "TR": [...]}, info)."""
    spec = OTHER_STUDENTS[skey]
    out, info = OrderedDict(), OrderedDict([("student", skey), ("label", spec["label"]), ("path", spec["path"])])
    shell = render_student(skey, tok, SLOT)
    if shell.count(SLOT) != 1 or SYSTEM_MESSAGE not in shell:
        die("%s: the template shell does not carry the system text and exactly one user slot" % skey)
    guard_template(skey + "/shell", shell.replace(SLOT, ""), ctx)        # everything the template adds around the user message
    info["template_shell"] = shell
    for lay in ("A", "TR"):
        arm = spec["name"] + "_" + lay
        ps = []
        for e in d11p[lay]:
            p = render_student(skey, tok, e["user"])
            if p != shell.replace(SLOT, e["user"]):
                die("%s qid %s: the prompt is not the template shell with the Qwen3-14B user message in its slot" % (arm, e["qid"]))
            ps.append(OrderedDict([("qid", e["qid"]), ("prompt", p), ("layout", e["layout"]), ("n_facts", e["n_facts"])]))
        if lay == "A":
            guard_question_prompts(arm, [x["prompt"] for x in ps], ctx)
        else:      # the user message passed the frozen D11 guard (Amendment 6) inside d11_student.preflight; the shell is clean
            guard_no_reading(arm, [x["prompt"] for x in ps], ctx["reading_strings"])
            print("[leak-guard] %-18s %d prompts: user message = the sealed TR user message (D11 guard with Amendment 6), shell clean, no reading" % (arm, len(ps)))
        info[arm] = OrderedDict([("prompt_tokens", check_len(arm, tok, [x["prompt"] for x in ps], MAX_NEW_TOKENS, spec["context"])),
                                 ("layouts", dict((k, sum(1 for x in ps if x["layout"] == k)) for k in sorted({x["layout"] for x in ps})))])
        out[lay] = ps
    return out, info


def weights_present(path):
    """config.json plus every shard named in model.safetensors.index.json."""
    idx = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(os.path.join(path, "config.json")) or not os.path.exists(idx):
        return False, 0, 0
    shards = sorted(set(json.load(open(idx))["weight_map"].values()))
    have = [s for s in shards if os.path.exists(os.path.join(path, s))]
    return len(have) == len(shards) and len(shards) > 0, len(have), len(shards)


def log_full_prompt(label, prompt):
    print("\n----- FULL PROMPT: %s -----" % label)
    print(prompt)
    print("----- END PROMPT -----", flush=True)
