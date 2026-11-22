#!/usr/bin/env python3
"""
d15_common.py — shared code of D15 (docs/EFBPT_D15_NATIVE_URDU_FREEZE.md).

Reading effect and teacher facts on native Urdu questions (UrduMMLU sample). This module holds
the D15 pins, the rows and the sample, the prompt builders, the item blocks, the translation
parser and flags, the output / seal helpers and the D15 API ledger. It scores nothing: no
extractor is defined or imported here, and the builders never read a gold field.

The D14 code is reused by import (d14_common): the output guard, row files, chunked generation,
the API client, the budget ledger, the translation label, the facts block, the teacher prompts,
engine, parser and filters. Its output root, ledger path and budget caps are pointed at D15 when
this module is imported (freeze 11.9, 11.12); nothing else of d14_common is changed.

Frozen text is not retyped:
  UR prompt        the instr / q_block / raw statements of the pinned CSQA fair script with the
                   four replacements of freeze N2, each asserted to occur exactly once (11.4)
  translation label, facts block, teacher prompts, engine, parser and filters, API settings,
  translation system message
                   d14_common (and through it d11_ablate_ro, d1_eval_arms, d10_teacher,
                   d11o_teacher, d11_api_direct, d11r_common)
The EN instruction and trailer are typed here from the freeze and checked against the freeze text
by the preflight.
"""
import ast
import fcntl
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

import d14_common as C14  # noqa: E402
from d14_common import die, rp, md5_file, file_entry, record_line, append_row, read_rows, is_sealed, one_line  # noqa: E402,F401

ROOT = C14.ROOT
EFBPT = C14.EFBPT
FAIR_SCRIPT = C14.FAIR + "cot_csqa_baseline_fair.py"
FAIR_SEALED = "outputs/sdfr/cot_csqa_baseline_fair_qwen3_14b.jsonl"
FAIR_SEALED_CORRECT = 190

FREEZE_PATH = "docs/EFBPT_D15_NATIVE_URDU_FREEZE.md"
FREEZE_MD5 = "86cfd8ff6f994def4f801d68543a16ef"
DATA = "data/urdummlu/urdummlu.json"
DATA_REVISION = "aae060a8564c261393de206c94a4557c850e7f2f"
DATA_URL = "https://hf-mirror.com/datasets/MBZUAI/UrduMMLU/resolve/%s/urdummlu.json" % DATA_REVISION
SAMPLE_PATH = EFBPT + "d15_sample_ids.txt"
SAMPLE_MD5 = "b430f4ee354ddfa3cc5fd8df71f96c69"   # written by d15_sample.py after the freeze (11.3): 2,600 ids

N_ROWS = 26431
N_DOMAINS = 5
N_SUBDOMAINS = 26
SEED = 1515
PER_SUBDOMAIN = 100
LETTERS = ("A", "B", "C", "D")
LABELS = ("Question", "A", "B", "C", "D")

MODES = ("TEST", "FULL")
TEST_N = 5
CHUNK = C14.CHUNK

STUDENT_PATH = C14.STUDENT_PATH
STUDENT_MAX_MODEL_LEN = C14.STUDENT_MAX_MODEL_LEN
STUDENT_GPU_UTIL = C14.STUDENT_GPU_UTIL
STUDENT_STOP = C14.STUDENT_STOP
STUDENT_ARM_MAX_TOKENS = 1024          # the CSQA fair value (checked against its source by student_max_tokens)
LOCAL_TR_MAX_TOKENS = 512              # N3: TL and SL
TEACHER_ARM_MAX_TOKENS = 1024          # N4
API_ARM_MAX_TOKENS = 1024              # N4
R1_MAX_TOKENS = C14.R1_MAX_TOKENS      # 512 (D14 F3)
R1_CONCURRENCY = C14.R1_CONCURRENCY    # 4 (D14 F3)
R1_SYSTEM = C14.R1_SYSTEM              # N3: the D14 F3 system message, unchanged
VLLM_VERSION = C14.VLLM_VERSION

OUT_ROOT = "outputs/efbpt/d15"
TEST_ROOT = os.path.join(OUT_ROOT, "_test")
LEDGER = os.path.join(OUT_ROOT, "api_ledger.json")
BUDGET_CALLS = 11500                   # block limit / freeze 11.9
BUDGET_TOKENS = 5000000
TOKEN_STOP_AT = 4950000                # no request is sent from here on, so that requests in flight cannot pass the cap

# Point the reused d14_common output guard, ledger and caps at D15 (freeze 11.9, 11.12).
C14.OUT_ROOT = OUT_ROOT
C14.TEST_ROOT = TEST_ROOT
C14.LEDGER = LEDGER
C14.BUDGET_CALLS = BUDGET_CALLS
C14.BUDGET_TOKENS = BUDGET_TOKENS
C14.TOKEN_STOP_AT = TOKEN_STOP_AT

PINS = OrderedDict([   # freeze 11.1 (the sample file is added when it exists, 11.3)
    (FREEZE_PATH, FREEZE_MD5),
    (DATA, "a25bbf61360af54747d2b1b8dbd96041"),
    ("data/urdummlu/stats.json", "cb6b24df4c212684043791256d4d4ff0"),
    (FAIR_SCRIPT, "70491653824525aeaa58a3282d5d177f"),
    (FAIR_SEALED, "40e09c2208cce4217f86fb54c1cc034c"),
    (EFBPT + "d14_common.py", "116c2d5fef8ec4ba5b8df844a1410fd9"),
    (EFBPT + "d14_score.py", "fb1d2df15619067f85f7bc0b8edb591d"),
    (EFBPT + "d14_readout.py", "1734cee1b977395ef6cb175875321380"),
    (EFBPT + "d14_balance.py", "4098c33f54f0c76f38a7a970c500a408"),
    (EFBPT + "d14_translate.py", "a90e6375ae7f2e965d09ee74960321be"),
    (EFBPT + "d14_api.py", "dbf5d5d57c11ce64c71637c60487c252"),
    (EFBPT + "d14_teacher.py", "db30780a08f2f1fac6150a9f34387fc3"),
    (EFBPT + "d10_teacher.py", "60bca5ba3d60e28e93fac1c643bf6802"),
    (EFBPT + "d11o_teacher.py", "338df12c1c8b2f3c362b1f2076244c3b"),
    (EFBPT + "d1_eval_arms.py", "10917f65b7852b406aec709a95d0a45b"),
    (EFBPT + "d11_ablate_ro.py", "b3d42f4c7b6805d4f91f45a2e28dca5b"),
    (EFBPT + "d11_api_direct.py", "4af85505d859b481aaf1ab0caf26905f"),
    (EFBPT + "d11r_common.py", "4205923e6aca8294d9c1e10f860523a7"),
    ("docs/EFBPT_D14_MORE_TASKS_FREEZE.md", "31bc15881ad8102a2b35a676abf1c268"),
    ("docs/EFBPT_D14_AMENDMENT_1.md", "6153e4ead003bc4b183df9ec28be470b"),
    ("docs/EFBPT_D10_TEACHER_FREEZE.md", "b3a822d8199034c7fe58d76cf5cc428f"),
])
if SAMPLE_MD5:
    PINS[SAMPLE_PATH] = SAMPLE_MD5

D15_SCRIPTS = ("d15_common.py", "d15_sample.py", "d15_sample_ids.txt", "d15_score.py", "d15_preflight.py", "d15_translate.py",
               "d15_api.py", "d15_balance.py", "d15_teacher.py", "d15_teacher.sbatch", "d15_student.py", "d15_student.sbatch")

# N2: the EN prompt (typed from the freeze; the preflight checks it against the freeze text)
EN_INSTRUCTION = ("You are given a question and four options (A, B, C, D).\nFirst think briefly about which answer is correct.\n"
                  "At the end use only this form: Answer: X\nwhere X is only A, B, C or D.")
EN_TRAILER = "Step-by-step reasoning:\n\nAnswer:"

# N2: the four replacements on the source text of the fair script's instr and q_block statements
UR_REPLACEMENTS = (("پانچ", "چار"), ("(A، B، C، D، E)", "(A، B، C، D)"), ("A، B، C، D یا E", "A، B، C یا D"), ("\\nE: {E}", ""))

# N1 / freeze 11.2: groups from the subdomain names only
GROUPS = ("URDU", "REGIONAL", "OTHER")


def group_of(subdomain):
    s = subdomain.lower()
    if "urdu" in s:
        return "URDU"
    if "islam" in s or "pakistan" in s:
        return "REGIONAL"
    return "OTHER"


# N4: arms. reading = the translation the arm needs (R1, TL, SL) or None; facts = the arm needs the teacher facts.
ARMS = OrderedDict([
    ("S_ur", OrderedDict([("model", "S"), ("kind", "ur"), ("reading", None), ("facts", False)])),
    ("S_ro", OrderedDict([("model", "S"), ("kind", "ur+tr"), ("reading", "R1"), ("facts", False)])),
    ("S_self", OrderedDict([("model", "S"), ("kind", "ur+tr"), ("reading", "SL"), ("facts", False)])),
    ("S_tr", OrderedDict([("model", "S"), ("kind", "ur+facts"), ("reading", "R1"), ("facts", True)])),
    ("T_ur", OrderedDict([("model", "T"), ("kind", "ur"), ("reading", None), ("facts", False)])),
    ("T_ue", OrderedDict([("model", "T"), ("kind", "ue"), ("reading", None), ("facts", False)])),
    ("T_r1", OrderedDict([("model", "T"), ("kind", "en"), ("reading", "R1"), ("facts", False)])),
    ("T_self", OrderedDict([("model", "T"), ("kind", "en"), ("reading", "TL"), ("facts", False)])),
    ("M_ur", OrderedDict([("model", "M"), ("kind", "ur"), ("reading", None), ("facts", False)])),
    ("M_ue", OrderedDict([("model", "M"), ("kind", "ue"), ("reading", None), ("facts", False)])),
    ("M_r1", OrderedDict([("model", "M"), ("kind", "en"), ("reading", "R1"), ("facts", False)])),
])
MODEL_DIR = {"S": "student", "T": "teacher", "M": "api"}
TRANSLATIONS = OrderedDict([("R1", "qwen3.8-max (API)"), ("TL", "Llama-3.1-70B-AWQ, D10 engine"), ("SL", "Qwen3-14B, thinking off")])


def arms_of(model=None):
    return [a for a, s in ARMS.items() if model is None or s["model"] == model]


# ----------------------------------------------------------------------------
# Pins, git
# ----------------------------------------------------------------------------
def verify_pins(verbose=True, need_sample=True):
    if need_sample and not SAMPLE_MD5:
        die("the sample MD5 is not pinned yet (run d15_sample.py and pin its MD5 in d15_common.py)")
    out = OrderedDict()
    for path, want in PINS.items():
        if not os.path.exists(rp(path)):
            die("pinned file missing: " + path)
        got = md5_file(path)
        if got != want:
            die("%s MD5 %s != pinned %s" % (path, got, want))
        out[path] = got
    if verbose:
        print("[md5] %d pinned files verified (D15 freeze %s)" % (len(out), FREEZE_MD5), flush=True)
    return out


def script_md5s():
    return OrderedDict((EFBPT + n, md5_file(EFBPT + n) if os.path.exists(rp(EFBPT + n)) else None) for n in D15_SCRIPTS)


git_state = C14.git_state


def require_committed():
    """A FULL run needs the d15 scripts committed and unchanged (freeze 11.12)."""
    pat = EFBPT + "d15_*"
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", pat], cwd=ROOT).decode().strip()
        tracked = subprocess.check_output(["git", "ls-files", "--", pat], cwd=ROOT).decode().split()
    except Exception as e:
        die("git is not usable (%s); a FULL run needs the committed d15 scripts" % type(e).__name__)
    missing = [n for n in D15_SCRIPTS if (EFBPT + n) not in tracked]
    if dirty or missing:
        die("FULL run refused: the d15 scripts are not committed and clean (untracked/modified: %s; not tracked: %s)"
            % (dirty.replace("\n", "; ") or "-", ", ".join(missing) or "-"))
    print("[ok] d15 scripts are committed and unchanged (HEAD %s)" % git_state()["head"], flush=True)


# ----------------------------------------------------------------------------
# Source rows, validity, sample (freeze N1, 11.3)
# ----------------------------------------------------------------------------
_CACHE = {}


def source_rows():
    """The 26,431 rows of the pinned file, in file order."""
    if "src" not in _CACHE:
        got = md5_file(DATA)
        if got != PINS[DATA]:
            die("%s MD5 %s != pinned %s" % (DATA, got, PINS[DATA]))
        with open(rp(DATA), encoding="utf-8") as f:
            rows = json.load(f, object_pairs_hook=OrderedDict)
        if not isinstance(rows, list) or len(rows) != N_ROWS:
            die("%s: %s rows, expected %d" % (DATA, len(rows) if isinstance(rows, list) else type(rows).__name__, N_ROWS))
        _CACHE["src"] = rows
    return _CACHE["src"]


def invalid_reason(r):
    """None for a valid row (freeze 11.3), else the first failed condition."""
    if not isinstance(r.get("question"), str) or not r["question"].strip():
        return "question empty"
    o = r.get("options")
    if not isinstance(o, dict) or set(o) != set(LETTERS):
        return "option keys not exactly A-D"
    if not all(isinstance(o[k], str) and o[k].strip() for k in LETTERS):
        return "an option is empty"
    if r.get("correct_key") not in LETTERS:
        return "correct_key not in A-D"
    return None


def select_sample():
    """(ids in ascending order, report). Validity, then duplicates over the whole file (lowest id kept), then a new
    random.Random(1515) per subdomain (sorted names) sampling 100 of the kept rows sorted by id (all if <= 100)."""
    rows = source_rows()
    rep = OrderedDict([("rows", len(rows)), ("invalid", Counter()), ("duplicates_dropped", 0), ("per_subdomain", OrderedDict())])
    valid = []
    for r in rows:
        why = invalid_reason(r)
        if why:
            rep["invalid"][why] += 1
            continue
        valid.append(r)
    seen, kept = {}, []
    for r in sorted(valid, key=lambda x: x["id"]):
        key = (r["question"],) + tuple(r["options"][k] for k in LETTERS)
        if key in seen:
            rep["duplicates_dropped"] += 1
            continue
        seen[key] = r["id"]
        kept.append(r)
    by_sub = OrderedDict()
    for r in kept:
        by_sub.setdefault(r["subdomain"], []).append(r)
    total_by_sub = Counter(r["subdomain"] for r in rows)
    ids = []
    for sub in sorted(total_by_sub):
        L = sorted(by_sub.get(sub, []), key=lambda x: x["id"])
        rng = random.Random(SEED)
        pick = rng.sample(L, PER_SUBDOMAIN) if len(L) > PER_SUBDOMAIN else list(L)
        ids += [r["id"] for r in pick]
        rep["per_subdomain"][sub] = OrderedDict([("rows", total_by_sub[sub]), ("kept", len(L)), ("sampled", len(pick)), ("group", group_of(sub))])
    rep["invalid"] = OrderedDict(sorted(rep["invalid"].items()))
    rep["valid"] = len(valid)
    rep["kept"] = len(kept)
    ids = sorted(ids)
    if len(set(ids)) != len(ids):
        die("the sample holds an id twice")
    rep["sampled"] = len(ids)
    return ids, rep


def sample_ids():
    """The committed id list (freeze 11.3), MD5-checked."""
    if "ids" not in _CACHE:
        if not SAMPLE_MD5:
            die("the sample MD5 is not pinned yet")
        got = md5_file(SAMPLE_PATH)
        if got != SAMPLE_MD5:
            die("%s MD5 %s != pinned %s" % (SAMPLE_PATH, got, SAMPLE_MD5))
        with open(rp(SAMPLE_PATH), encoding="utf-8") as f:
            ids = [int(l) for l in f.read().split("\n") if l.strip()]
        if ids != sorted(set(ids)):
            die("%s is not in ascending order without repeats" % SAMPLE_PATH)
        _CACHE["ids"] = ids
    return _CACHE["ids"]


def rid_of(i):
    return "um_%05d" % i


def _text_only(r):
    """A copy of the row with its text fields only: the gold key never reaches a builder."""
    return OrderedDict([("question", r["question"]), ("options", OrderedDict((k, r["options"][k]) for k in LETTERS))])


def load_rows(mode="FULL"):
    """[{rid, id, item}] of the sample in run order (ascending id); item = {question, options} only. TEST = first 5."""
    if mode not in MODES:
        die("unknown mode %r" % mode)
    by_id = {r["id"]: r for r in source_rows()}
    rows = []
    for i in sample_ids():
        r = by_id[i]
        if invalid_reason(r):
            die("sampled id %d is not a valid row" % i)
        rows.append(OrderedDict([("rid", rid_of(i)), ("id", i), ("item", _text_only(r))]))
    return rows[:TEST_N] if mode == "TEST" else rows


def row_meta():
    """{rid: {subdomain, domain, group, level, length_tier}} of the sample (readout and preflight; no gold)."""
    by_id = {r["id"]: r for r in source_rows()}
    return OrderedDict((rid_of(i), OrderedDict([("subdomain", by_id[i]["subdomain"]), ("domain", by_id[i]["domain"]),
                                                ("group", group_of(by_id[i]["subdomain"])), ("level", by_id[i]["level"]),
                                                ("length_tier", by_id[i]["length_tier"])])) for i in sample_ids())


def load_gold():
    """{rid: correct_key} of the sample. Readout only (and the preflight's leak check)."""
    by_id = {r["id"]: r for r in source_rows()}
    return OrderedDict((rid_of(i), by_id[i]["correct_key"]) for i in sample_ids())


# ----------------------------------------------------------------------------
# Prompts (freeze N2, 11.4)
# ----------------------------------------------------------------------------
def _ur_code():
    """The instr and q_block statements of the fair script's prompt loop with the four N2 replacements made on their
    source text (each asserted to occur exactly once), followed by its unchanged raw statement; compiled."""
    if "ur_code" not in _CACHE:
        src = C14._source(FAIR_SCRIPT)
        tree = ast.parse(src)
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "item"]
        if len(loops) != 1:
            die("%s: expected one `for item in ...` prompt loop, found %d" % (FAIR_SCRIPT, len(loops)))
        seg = OrderedDict()
        for st in loops[0].body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                seg[st.targets[0].id] = ast.get_source_segment(src, st)
        for k in ("instr", "q_block", "raw"):
            if k not in seg:
                die("%s: no `%s = ...` statement in the prompt loop" % (FAIR_SCRIPT, k))
        text = seg["instr"] + "\n" + seg["q_block"]
        counts = OrderedDict()
        for old, new in UR_REPLACEMENTS:
            counts[old] = text.count(old)
            if counts[old] != 1:
                die("N2 replacement %r occurs %d times in the instr / q_block source, expected exactly once" % (old, counts[old]))
            text = text.replace(old, new)
        if text.count("E: {E}") or "{E}" in text:
            die("the E option is still in the changed q_block source")
        _CACHE["ur_code"] = compile(text + "\n" + seg["raw"] + "\n", FAIR_SCRIPT + " (D15 N2 replacements)", "exec")
        _CACHE["ur_counts"] = counts
        _CACHE["ur_source"] = text + "\n" + seg["raw"]
    return _CACHE["ur_code"]


def _run_ur(question, opts):
    ns = {"item": {"question": question}}
    for k in LETTERS:
        ns[k] = opts[k]
    exec(_ur_code(), ns)
    return ns["instr"], ns["q_block"], ns["raw"]


def ur_trailer():
    """The trailer of the UR prompt: the part of q_block after the item lines, taken from a probe row."""
    if "trailer" not in _CACHE:
        _, q, _ = _run_ur("Q", {"A": "a", "B": "b", "C": "c", "D": "d"})
        if q.count("D: d\n\n") != 1:
            die("cannot find the end of the item lines in the probe q_block")
        _CACHE["trailer"] = q.split("D: d\n\n", 1)[1]
    return _CACHE["trailer"]


def ur_parts(item):
    """(instruction, item lines, trailer) of the UR user message; joined by blank lines they are that message."""
    if set(item) != {"question", "options"}:
        die("a builder got a row with fields %s (only question and options are allowed)" % sorted(item))
    instr, q_block, raw = _run_ur(item["question"], item["options"])
    if raw != instr + "\n\n" + q_block:
        die("UR: raw is not instr + blank line + q_block")
    tr = ur_trailer()
    if not q_block.endswith("\n\n" + tr):
        die("UR: q_block does not end with the trailer")
    return instr, q_block[:-(len(tr) + 2)], tr


def ur_user(item, block=None):
    """The UR user message; with `block`, that block sits between the item lines and the trailer (D14 F4 layout)."""
    instr, items, trailer = ur_parts(item)
    return "\n\n".join([instr, items] + ([block] if block else []) + [trailer])


def item_block(texts):
    """D14 9.4: one line per label, label + ': ' + text; white-space runs inside a field are one space."""
    if len(texts) != len(LABELS):
        die("%d texts for %d labels" % (len(texts), len(LABELS)))
    return "\n".join("%s: %s" % (lab, one_line(tx)) for lab, tx in zip(LABELS, texts))


def ur_block(item):
    """The Urdu item block: the translation USER message and the item block of UE."""
    return item_block([item["question"]] + [item["options"][k] for k in LETTERS])


def parse_translation(text):
    """D14 9.5 with the D15 labels: the item block of a valid translation, or None."""
    lines = [l.strip() for l in (text or "").strip().split("\n")]
    lines = [l for l in lines if l]
    if len(lines) != len(LABELS):
        return None
    texts = []
    for lab, l in zip(LABELS, lines):
        if not l.startswith(lab + ":"):
            return None
        tx = l[len(lab) + 1:].strip()
        if not tx:
            return None
        texts.append(tx)
    return item_block(texts)


def en_user(block):
    return EN_INSTRUCTION + "\n\n" + block + "\n\n" + EN_TRAILER


def tr_label_line():
    return C14.tr_label_line()


def facts_block(facts):
    return C14.facts_block(facts)


def build_users(arm, rows, readings=None, facts=None):
    """[{rid, status, user}] for one arm. status 'ok' or 'reading_failed' (such a row is not generated).
    readings = {name: {rid: item block or None}}; facts = {rid: [fact, ...]} (S_tr)."""
    if arm not in ARMS:
        die("unknown arm %r" % arm)
    spec = ARMS[arm]
    rd = spec["reading"]
    if rd and (readings is None or rd not in readings):
        die("arm %s needs the sealed %s readings" % (arm, rd))
    if spec["facts"] and facts is None:
        die("arm %s needs the sealed teacher facts" % arm)
    out = []
    for row in rows:
        rid, item = row["rid"], row["item"]
        rec = OrderedDict([("rid", rid), ("status", "ok"), ("user", None)])
        if rd and readings[rd][rid] is None:
            rec["status"] = "reading_failed"
            out.append(rec)
            continue
        kind = spec["kind"]
        if kind == "ur":
            rec["user"] = ur_user(item)
        elif kind == "ur+tr":
            rec["user"] = ur_user(item, tr_label_line() + "\n" + readings[rd][rid])
        elif kind == "ur+facts":
            fl = list(facts[rid])
            rec["n_facts"] = len(fl)
            rec["user"] = ur_user(item, facts_block(fl) if fl else None)
        elif kind == "ue":
            rec["user"] = en_user(ur_block(item))
        elif kind == "en":
            rec["user"] = en_user(readings[rd][rid])
        else:
            die("unknown prompt kind %r" % kind)
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Translation flags (N3, freeze 11.5; reported, not a check)
# ----------------------------------------------------------------------------
MARKS = ("✓", "✔", "✅", "☑", "√", "←", "*", "<-", "(correct)", "(answer)")


def translation_flags(text):
    t = text or ""
    low = t.lower()
    opt_lines = [l.strip() for l in t.split("\n") if l.strip()[:2] in ("A:", "B:", "C:", "D:")]
    marked = sum(1 for l in opt_lines if any(m in l.lower() for m in MARKS))
    return OrderedDict([("correct", "correct" in low), ("answer_is", "answer is" in low), ("mark_on_one_option", marked == 1)])


# ----------------------------------------------------------------------------
# Output paths, start / resume / seal (freeze 11.12)
# ----------------------------------------------------------------------------
def base_dir(mode):
    if mode not in MODES:
        die("unknown mode %r" % mode)
    return TEST_ROOT if mode == "TEST" else OUT_ROOT


def _paths(d, stem):
    return OrderedDict([("dir", d), ("rows", os.path.join(d, "d15_%s.jsonl" % stem)), ("work", os.path.join(d, "d15_%s.work.jsonl" % stem)),
                        ("start", os.path.join(d, "START_%s.json" % stem)), ("seal", os.path.join(d, "SEAL_%s.json" % stem))])


def tr_paths(name, mode):
    if name not in TRANSLATIONS:
        die("unknown translation %r" % name)
    return _paths(os.path.join(base_dir(mode), "tr"), "tr_" + name)


def facts_paths(mode):
    return _paths(os.path.join(base_dir(mode), "facts"), "facts")


def arm_paths(arm, mode):
    return _paths(os.path.join(base_dir(mode), MODEL_DIR[ARMS[arm]["model"]]), arm)


guard_out = C14.guard_out
write_json = C14.write_json


def add_run(pth, start_obj, n_done):
    run = OrderedDict([("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("rows_already_done", n_done),
                       ("slurm_job_id", os.environ.get("SLURM_JOB_ID")), ("hostname", socket.gethostname()),
                       ("scripts", script_md5s()), ("git", git_state())])
    if os.path.exists(rp(pth["start"])):
        cur = json.load(open(rp(pth["start"]), encoding="utf-8"), object_pairs_hook=OrderedDict)
        cur.setdefault("runs", []).append(run)
        start_obj = cur
    else:
        start_obj = OrderedDict(start_obj)
        start_obj["runs"] = [run]
    write_json(pth["start"], start_obj)


def open_rows(pth, expected, start_obj, key="prompt", ok_alias=()):
    """Start or resume a row file written in run order (D14 open_rows with the D15 START record). Returns the number
    of finished rows, or None if already sealed. ok_alias: statuses a finished row may carry where the run expects
    'ok' (a local translation whose output was invalid is written as reading_failed)."""
    os.makedirs(guard_out(pth["dir"]), exist_ok=True)
    if is_sealed(pth):
        print("[sealed] %s is already sealed; nothing is generated again" % pth["rows"], flush=True)
        return None
    done = read_rows(pth["rows"], repair=True)
    if len(done) > len(expected):
        die("%s: %d rows on disk, %d expected" % (pth["rows"], len(done), len(expected)))
    for i, (d, e) in enumerate(zip(done, expected)):
        st_ok = d["status"] == e["status"] or (e["status"] == "ok" and d["status"] in ok_alias)
        if d["rid"] != e["rid"] or not st_ok or d.get(key) != e.get(key):
            die("%s row %d: the finished row does not match this run (rid %s)" % (pth["rows"], i, e["rid"]))
    add_run(pth, start_obj, len(done))
    if done:
        print("[resume] %s: %d finished rows kept; continuing at row %d" % (pth["rows"], len(done), len(done)), flush=True)
    return len(done)


def seal_rows(pth, expected_rids, head, extra=None):
    """Write the SEAL file when every row is present in order. Not scored: counts, MD5s, seconds and tokens only."""
    recs = read_rows(pth["rows"])
    if [r["rid"] for r in recs] != list(expected_rids):
        die("%s: %d rows on disk do not equal the %d rows of the run in order" % (pth["rows"], len(recs), len(expected_rids)))
    for r in recs:
        for k in ("pred", "correct", "gold", "correct_key"):
            if k in r:
                die("%s: a row carries the field %r (row files hold no score)" % (pth["rows"], k))
    secs = [r.get("seconds") for r in recs if isinstance(r.get("seconds"), (int, float))]
    gen = [r.get("n_gen_tokens") for r in recs if isinstance(r.get("n_gen_tokens"), (int, float))]
    prm = [r.get("n_prompt_tokens") for r in recs if isinstance(r.get("n_prompt_tokens"), (int, float))]
    status = OrderedDict()
    for r in recs:
        status[r["status"]] = status.get(r["status"], 0) + 1
    s = OrderedDict(head)
    s.update(OrderedDict([
        ("complete", True), ("n_rows", len(recs)), ("status_counts", status), ("scored", False),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("slurm_job_id", os.environ.get("SLURM_JOB_ID")),
        ("hostname", socket.gethostname()), ("d15_freeze_md5", FREEZE_MD5), ("sample_md5", SAMPLE_MD5),
        ("scripts", script_md5s()), ("git", git_state()),
        ("seconds_total", round(sum(secs), 3) if secs else None), ("prompt_tokens_total", int(sum(prm)) if prm else None),
        ("gen_tokens_total", int(sum(gen)) if gen else None), ("rows_truncated", sum(1 for r in recs if r.get("truncated"))),
    ]))
    if extra:
        s.update(extra)
    files = [pth["start"], pth["rows"]] + ([pth["work"]] if os.path.exists(rp(pth["work"])) else [])
    s["files"] = OrderedDict((p, file_entry(p)) for p in files)
    write_json(pth["seal"], s)
    print("  [sealed] %s (%d rows %s; not scored) rows MD5 %s" % (pth["seal"], len(recs), dict(status), s["files"][pth["rows"]]["md5"]), flush=True)
    return s


def load_reading(name, mode, rows):
    """{rid: item block, or None for a reading_failed / too_long row} from the sealed translation file."""
    pth = tr_paths(name, mode)
    if not is_sealed(pth):
        die("the %s readings (%s) are not sealed: %s" % (name, mode, pth["seal"]))
    recs = read_rows(pth["rows"])
    if [r["rid"] for r in recs] != [r["rid"] for r in rows]:
        die("%s: the sealed rows are not the rows of this run" % pth["rows"])
    out = OrderedDict()
    for r, row in zip(recs, rows):
        if r["user"] != ur_block(row["item"]):
            die("%s %s: the sealed record was made from a different Urdu item block" % (pth["rows"], r["rid"]))
        if r["status"] == "ok":
            if not r["block"] or parse_translation(r["block"]) != r["block"]:
                die("%s %s: the sealed block is not a valid item block" % (pth["rows"], r["rid"]))
            out[r["rid"]] = r["block"]
        elif r["status"] in ("reading_failed", "too_long"):
            out[r["rid"]] = None
        else:
            die("%s %s: unknown status %r" % (pth["rows"], r["rid"], r["status"]))
    return out, OrderedDict([("name", name), ("file", pth["rows"]), ("md5", md5_file(pth["rows"])),
                             ("not_valid", sum(1 for v in out.values() if v is None))])


def load_facts(mode, rows, r1):
    """{rid: [facts]} from the sealed teacher-facts file (a row without a valid R1 reading has an empty list)."""
    pth = facts_paths(mode)
    if not is_sealed(pth):
        die("the teacher facts (%s) are not sealed: %s" % (mode, pth["seal"]))
    recs = read_rows(pth["rows"])
    if [r["rid"] for r in recs] != [r["rid"] for r in rows]:
        die("%s: the sealed facts rows are not the rows of this run" % pth["rows"])
    out = OrderedDict()
    for r in recs:
        if (r["status"] == "reading_failed") != (r1[r["rid"]] is None):
            die("%s %s: status does not agree with the sealed R1 reading" % (pth["rows"], r["rid"]))
        if r["status"] == "ok" and r["r1_block"] != r1[r["rid"]]:
            die("%s %s: the facts were made from a different R1 block" % (pth["rows"], r["rid"]))
        fl = r["facts"]
        if not isinstance(fl, list) or len(fl) > 6 or not all(isinstance(x, str) and x.strip() and "\n" not in x for x in fl):
            die("%s %s: facts are not <= 6 clean one-line strings" % (pth["rows"], r["rid"]))
        out[r["rid"]] = list(fl)
    return out, OrderedDict([("file", pth["rows"]), ("md5", md5_file(pth["rows"])), ("rows_without_facts", sum(1 for v in out.values() if not v))])


# ----------------------------------------------------------------------------
# API ledger (freeze 11.9): the d14_common ledger code, pointed at the D15 path and caps above
# ----------------------------------------------------------------------------
Api = C14.Api
BudgetStop = C14.BudgetStop
api_settings = C14.api_settings
response_text = C14.response_text
log_full_prompt = C14.log_full_prompt
ledger_read = C14.ledger_read


def ledger_init():
    """Create the D15 ledger with its own header if it does not exist (before the first request)."""
    ap = guard_out(LEDGER)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    with open(ap + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not os.path.exists(ap):
                led = OrderedDict([("what", "D15 API budget ledger: every request sent counts as one call; tokens = prompt + completion from the usage fields"),
                                   ("cap_calls", BUDGET_CALLS), ("cap_tokens", BUDGET_TOKENS), ("token_stop_at", TOKEN_STOP_AT),
                                   ("calls", 0), ("prompt_tokens", 0), ("completion_tokens", 0), ("tokens", 0),
                                   ("answered_without_usage", 0), ("stopped", None), ("by_tag", OrderedDict()), ("balance", [])])
                tmp = ap + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(led, f, indent=1, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, ap)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    if C14.LEDGER != LEDGER or C14.BUDGET_CALLS != BUDGET_CALLS or C14.TOKEN_STOP_AT != TOKEN_STOP_AT:
        die("the reused ledger code is not pointed at the D15 ledger and caps")


# ----------------------------------------------------------------------------
# Chat templates, engines, chunked generation (reused)
# ----------------------------------------------------------------------------
load_tok = C14.load_tok
student_chat = C14.student_chat
teacher_render = C14.teacher_render
n_tokens = C14.n_tokens
teacher_frozen = C14.teacher_frozen
facts_user = C14.facts_user
d14_filter = C14.d14_filter
generate_rows = C14.generate_rows
vllm_outputs = C14.vllm_outputs


def student_tr_chat(tok, user):
    """SL (freeze 11.5): system = the D14 F3 message, user = the Urdu item block, thinking off."""
    return tok.apply_chat_template([{"role": "system", "content": R1_SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def student_max_tokens():
    """max_tokens and the engine line of the CSQA fair script, read from its source (checked against the freeze values)."""
    mt = C14.fair_max_tokens("csqa")
    if mt != STUDENT_ARM_MAX_TOKENS:
        die("the CSQA fair max_tokens %d differs from %d" % (mt, STUDENT_ARM_MAX_TOKENS))
    return mt


def after_think(text):
    """The text after the last </think> (the fair script's strip and split)."""
    return (text or "").strip().split("</think>")[-1]
