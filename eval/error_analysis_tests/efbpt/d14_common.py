#!/usr/bin/env python3
"""
d14_common.py — shared code of D14 (docs/EFBPT_D14_MORE_TASKS_FREEZE.md).

Reading effect and teacher facts on CSQA, PIQA and GSM8K. This module holds the
pins, the rows, the prompt builders, the item blocks, the output / seal helpers
and the API budget ledger. It scores nothing: no extractor is defined or
imported here, and the builders never read a gold field.

Frozen text is not retyped:
  UR prompt        the statements of the pinned fair script, run on the row (9.3)
  translation label, facts block
                   the RO_LABEL statement of d11_ablate_ro.py and the
                   FACTS_HEADER / facts_block statements of d1_eval_arms.py,
                   taken from the pinned sources (9.6)
  API settings     d11_api_direct / d11r_common, imported (9.5, 9.9)
  teacher prompts, engine, parser, filter
                   d10_teacher / d11o_teacher, imported (9.7, 9.8)
The EN prompt and the translation system message are typed here from the freeze
and checked against the freeze text by the preflight.
"""
import ast
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
EFBPT = "eval/error_analysis_tests/efbpt/"
FAIR = "eval/error_analysis_tests/"

FREEZE_PATH = "docs/EFBPT_D14_MORE_TASKS_FREEZE.md"
FREEZE_MD5 = "31bc15881ad8102a2b35a676abf1c268"

MODES = ("TEST", "FULL")
TEST_N = 5                      # F8: TEST-5 = rows 0-4 of each task
CHUNK = 64                      # 9.6, 9.8: prompts per generate call

STUDENT_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
STUDENT_MAX_MODEL_LEN = 8192    # the fair scripts' engine line, checked against their source by the preflight
STUDENT_GPU_UTIL = 0.85
STUDENT_STOP = ["<|im_end|>"]
TEACHER_MAX_NUM_SEQS = 16       # F4
TEACHER_ARM_MAX_TOKENS = 1024   # F4
API_ARM_MAX_TOKENS = 1024       # F4
R1_MAX_TOKENS = 512             # F3
R1_CONCURRENCY = 4              # F3
BUDGET_CALLS = 13000            # 9.9
BUDGET_TOKENS = 7000000         # 9.9
TOKEN_STOP_AT = 6950000         # no request is sent from here on, so that requests in flight cannot pass the cap
VLLM_VERSION = "0.13.0"

TASKS = OrderedDict([
    ("csqa", OrderedDict([
        ("n", 1500), ("ur", "data/csqa_raw/csqa_train_1500_ur.jsonl"), ("en", "data/csqa_raw/csqa_train_1500_en.jsonl"),
        ("fair", FAIR + "cot_csqa_baseline_fair.py"), ("sealed", "outputs/sdfr/cot_csqa_baseline_fair_qwen3_14b.jsonl"),
        ("sealed_correct", 190), ("eval_range", (1200, 1500)), ("max_tokens", 1024),
        ("labels", ("Question", "A", "B", "C", "D", "E")), ("id_field", "id"), ("id_mismatch_ok", (4,)), ("has_tr", True)])),
    ("piqa", OrderedDict([
        ("n", 750), ("ur", "data/piqa_raw/piqa_train_750_ur_fixed.jsonl"), ("en", "data/piqa_raw/piqa_train_750.jsonl"),
        ("fair", FAIR + "cot_piqa_baseline_fair.py"), ("sealed", "outputs/sdfr/cot_piqa_baseline_fair_qwen3_14b.jsonl"),
        ("sealed_correct", 108), ("eval_range", (600, 750)), ("max_tokens", 2048),
        ("labels", ("Goal", "Solution 1", "Solution 2")), ("id_field", None), ("id_mismatch_ok", ()), ("has_tr", True)])),
    ("gsm8k", OrderedDict([
        ("n", 700), ("ur", "data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl"), ("en", "data/gsm8k_raw/gsm8k_main_train_700.jsonl"),
        ("fair", FAIR + "cot_gsm8k_baseline_fair.py"), ("sealed", "outputs/sdfr/cot_gsm8k_baseline_fair_qwen3_14b.jsonl"),
        ("sealed_correct", 621), ("eval_range", (0, 700)), ("max_tokens", 2048),
        ("labels", ("Problem",)), ("id_field", None), ("id_mismatch_ok", ()), ("has_tr", False)])),
])

PINS = OrderedDict([   # freeze 9.1
    (FREEZE_PATH, FREEZE_MD5),
    ("data/csqa_raw/csqa_train_1500_ur.jsonl", "396601c0f6c1b26aaa4c29c3fc6cf018"),
    ("data/csqa_raw/csqa_train_1500_en.jsonl", "4a47619e60c4be2478757b8c8e6581bc"),
    ("data/piqa_raw/piqa_train_750_ur_fixed.jsonl", "01586d82da54b781aefb1d5427389515"),
    ("data/piqa_raw/piqa_train_750.jsonl", "9176d3b16cddebde5401ffe54dec49c2"),
    ("data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl", "ffb57c3df417799ad1d03f79a5aafc12"),
    ("data/gsm8k_raw/gsm8k_main_train_700.jsonl", "651a81016197dbaba7bc4b8a64d55c82"),
    (FAIR + "cot_csqa_baseline_fair.py", "70491653824525aeaa58a3282d5d177f"),
    (FAIR + "cot_piqa_baseline_fair.py", "cd6f074c4ea96820226b48768767ad37"),
    (FAIR + "cot_gsm8k_baseline_fair.py", "0ac419f37b59b5695434736f9abe9a4a"),
    ("outputs/sdfr/cot_csqa_baseline_fair_qwen3_14b.jsonl", "40e09c2208cce4217f86fb54c1cc034c"),
    ("outputs/sdfr/cot_piqa_baseline_fair_qwen3_14b.jsonl", "d969451aa8563c4524e89aef959553d5"),
    ("outputs/sdfr/cot_gsm8k_baseline_fair_qwen3_14b.jsonl", "d94ea3127817cf7357ca085f7a06e273"),
    (EFBPT + "d10_teacher.py", "60bca5ba3d60e28e93fac1c643bf6802"),
    (EFBPT + "d11o_teacher.py", "338df12c1c8b2f3c362b1f2076244c3b"),
    (EFBPT + "d1_eval_arms.py", "10917f65b7852b406aec709a95d0a45b"),
    (EFBPT + "d11_ablate_ro.py", "b3d42f4c7b6805d4f91f45a2e28dca5b"),
    (EFBPT + "d11_api_direct.py", "4af85505d859b481aaf1ab0caf26905f"),
    (EFBPT + "d11r_common.py", "4205923e6aca8294d9c1e10f860523a7"),
    ("docs/KNOWN_DATA_ISSUES.md", "4168fcd9a0de984445adc560cb6fff0b"),
    ("docs/handoff/D14_PREP.md", "bc2480c98ed4c98bab18be22e5f9db8c"),
    ("docs/EFBPT_D13_AMENDMENT_2.md", "12809c115ae8b6fba6bf6afe7fb6c51c"),
    ("docs/EFBPT_D14_AMENDMENT_1.md", "6153e4ead003bc4b183df9ec28be470b"),
])

D14_SCRIPTS = ("d14_common.py", "d14_score.py", "d14_preflight.py", "d14_translate.py", "d14_api.py", "d14_balance.py",
               "d14_teacher.py", "d14_teacher.sbatch", "d14_student.py", "d14_student.sbatch")

# F2: the EN prompt (typed from the freeze; the preflight checks each string against the freeze text)
EN = OrderedDict([
    ("csqa", OrderedDict([
        ("instruction", "You are given a question and five options (A, B, C, D, E).\nFirst think briefly about which answer is correct.\n"
                        "At the end use only this form: Answer: X\nwhere X is only A, B, C, D or E."),
        ("trailer", "Step-by-step reasoning:\n\nAnswer:")])),
    ("piqa", OrderedDict([
        ("instruction", "A goal and two possible solutions are given.\nFirst think step by step and in detail about which solution is more "
                        "sensible and agrees with common sense.\nThen at the end write only one digit: 0 or 1.\n"
                        "0 means Solution 1 is correct, 1 means Solution 2 is correct."),
        ("trailer", "Reasoning steps:\nFinal answer:")])),
    ("gsm8k", OrderedDict([
        ("instruction", "You are given a math word problem.\nFirst think step by step and solve the problem.\n"
                        "After that, write only one numeric answer after \"Final answer:\".\n"
                        "Do not include any unit, words or extra text in the final answer."),
        ("trailer", "Reasoning steps:\nFinal answer:")])),
])

# F3: the translation system message (typed from the freeze; checked against the freeze text by the preflight)
R1_SYSTEM = ("You are an expert Urdu-to-English translator. Translate the text exactly, preserving its meaning and its line structure: "
             "keep every label (such as 'A:', 'Goal:', 'Solution 1:') and every number unchanged, one output line per input line. "
             "Named entities are usually transliterations of English names; write each in its standard English spelling. "
             "Do not answer, do not solve and do not explain. Output only the English translation.")

# F4: arms. model S / T / M; prompt kind; needs_r1 = the arm needs the R1 reading of the row; scorer = which
# variant of the task's extractor the readout uses (9.11). Order = the order inside one job.
ARMS = OrderedDict([
    ("S_ur", OrderedDict([("model", "S"), ("kind", "ur"), ("needs_r1", False), ("scorer", "ur")])),
    ("S_ro", OrderedDict([("model", "S"), ("kind", "ur+r1"), ("needs_r1", True), ("scorer", "ur")])),
    ("S_og", OrderedDict([("model", "S"), ("kind", "ur+og"), ("needs_r1", False), ("scorer", "ur")])),
    ("S_tr", OrderedDict([("model", "S"), ("kind", "ur+facts"), ("needs_r1", True), ("scorer", "ur")])),
    ("T_ur", OrderedDict([("model", "T"), ("kind", "ur"), ("needs_r1", False), ("scorer", "ur")])),
    ("T_r1", OrderedDict([("model", "T"), ("kind", "en+r1"), ("needs_r1", True), ("scorer", "en")])),
    ("T_og", OrderedDict([("model", "T"), ("kind", "en+og"), ("needs_r1", False), ("scorer", "en")])),
    ("M_ur", OrderedDict([("model", "M"), ("kind", "ur"), ("needs_r1", False), ("scorer", "ur")])),
    ("M_r1", OrderedDict([("model", "M"), ("kind", "en+r1"), ("needs_r1", True), ("scorer", "en")])),
    ("M_og", OrderedDict([("model", "M"), ("kind", "en+og"), ("needs_r1", False), ("scorer", "en")])),
])
MODEL_DIR = {"S": "student", "T": "teacher", "M": "api"}

OUT_ROOT = "outputs/efbpt/d14"
TEST_ROOT = os.path.join(OUT_ROOT, "_test")
LEDGER = os.path.join(OUT_ROOT, "api_ledger.json")


def die(msg):
    print("[FATAL] " + msg, flush=True)
    sys.exit(1)


def arms_of(task, model=None):
    return [a for a, s in ARMS.items() if (model is None or s["model"] == model) and (a != "S_tr" or TASKS[task]["has_tr"])]


# ----------------------------------------------------------------------------
# Files, hashes, git
# ----------------------------------------------------------------------------
def rp(path):
    """Absolute path of a repo-relative path."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def md5_file(path):
    h = hashlib.md5()
    with open(rp(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with open(rp(path), encoding="utf-8") as f:
        return [json.loads(l, object_pairs_hook=OrderedDict) for l in f if l.strip()]


def file_entry(path):
    return OrderedDict([("md5", md5_file(path)), ("bytes", os.path.getsize(rp(path)))])


def verify_pins(verbose=True):
    out = OrderedDict()
    for path, want in PINS.items():
        if not os.path.exists(rp(path)):
            die("pinned file missing: " + path)
        got = md5_file(path)
        if got != want:
            die("%s MD5 %s != pinned %s" % (path, got, want))
        out[path] = got
    if verbose:
        print("[md5] %d pinned files verified (freeze %s)" % (len(out), FREEZE_MD5), flush=True)
    return out


def script_md5s():
    return OrderedDict((EFBPT + n, md5_file(EFBPT + n) if os.path.exists(rp(EFBPT + n)) else None) for n in D14_SCRIPTS)


def git_state():
    def run(*a):
        try:
            return subprocess.check_output(("git",) + a, cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    dirty = run("status", "--porcelain")
    return OrderedDict([("head", run("rev-parse", "HEAD")), ("dirty", bool(dirty) if dirty is not None else None)])


def require_committed():
    """A FULL run needs the d14 scripts committed and unchanged (freeze 9.12)."""
    pat = EFBPT + "d14_*"
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", pat], cwd=ROOT).decode().strip()
        tracked = subprocess.check_output(["git", "ls-files", "--", pat], cwd=ROOT).decode().split()
    except Exception as e:
        die("git is not usable (%s); a FULL run needs the committed d14 scripts" % type(e).__name__)
    missing = [n for n in D14_SCRIPTS if (EFBPT + n) not in tracked]
    if dirty or missing:
        die("FULL run refused: the d14 scripts are not committed and clean (untracked/modified: %s; not tracked: %s)"
            % (dirty.replace("\n", "; ") or "-", ", ".join(missing) or "-"))
    print("[ok] d14 scripts are committed and unchanged (HEAD %s)" % git_state()["head"], flush=True)


def guard_out(path):
    """Every D14 write goes under outputs/efbpt/d14/ and nowhere else."""
    ap = os.path.abspath(rp(path))
    root = os.path.abspath(rp(OUT_ROOT))
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


def append_row(f, rec):
    f.write(record_line(rec))
    f.flush()
    os.fsync(f.fileno())


def read_rows(path, repair=False):
    """Completed rows of a row file. A row is complete when its line ends with a newline. repair=True (a starting
    run) cuts off a torn tail, which is the unfinished row of a crashed run."""
    if not os.path.exists(rp(path)):
        return []
    with open(rp(path), "rb") as f:
        b = f.read()
    end = b.rfind(b"\n") + 1
    if end != len(b):
        if not repair:
            die("%s ends with a torn line (%d bytes)" % (path, len(b) - end))
        print("[resume] %s: torn tail of %d bytes cut off; completed rows untouched" % (path, len(b) - end), flush=True)
        with open(guard_out(path), "r+b") as f:
            f.truncate(end)
            f.flush()
            os.fsync(f.fileno())
        b = b[:end]
    return [json.loads(l.decode("utf-8"), object_pairs_hook=OrderedDict) for l in b.split(b"\n") if l]


# ----------------------------------------------------------------------------
# Statements taken from pinned sources (never retyped)
# ----------------------------------------------------------------------------
def _source(path):
    if path not in PINS:
        die("%s is not a pinned file" % path)
    got = md5_file(path)
    if got != PINS[path]:
        die("%s MD5 %s != pinned %s" % (path, got, PINS[path]))
    with open(rp(path), encoding="utf-8") as f:
        return f.read()


def _top_level(path, names):
    """Run the top-level assignments / function definitions with these names from a pinned source file."""
    tree = ast.parse(_source(path))
    ns = {"re": re, "OrderedDict": OrderedDict}
    found = set()
    for node in tree.body:
        name = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.FunctionDef):
            name = node.name
        if name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            found.add(name)
    if found != set(names):
        die("%s: names not found: %s" % (path, sorted(set(names) - found)))
    return ns


_CACHE = {}


def tr_label_line():
    """9.6: the d11_ablate_ro constant RO_LABEL without its two leading words and without the space after its colon."""
    if "tr_label" not in _CACHE:
        ro = _top_level(EFBPT + "d11_ablate_ro.py", ("RO_LABEL",))["RO_LABEL"]
        words = ro.split(" ")
        if len(words) != 5 or words[-1] != "" or not words[3].endswith(":"):
            die("RO_LABEL does not have the expected shape (4 words, a colon, one trailing space): %r" % ro)
        _CACHE["tr_label"] = " ".join(words[2:4])
    return _CACHE["tr_label"]


def facts_block(facts):
    """d1_eval_arms.facts_block, the frozen facts-block format."""
    if "facts_block" not in _CACHE:
        _CACHE["facts_block"] = _top_level(EFBPT + "d1_eval_arms.py", ("FACTS_HEADER", "facts_block"))["facts_block"]
    return _CACHE["facts_block"](facts)


def _fair_statements(task):
    """The statements of the fair script's prompt loop up to and including `raw = ...` (9.3), compiled."""
    key = "fair_" + task
    if key not in _CACHE:
        path = TASKS[task]["fair"]
        tree = ast.parse(_source(path))
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "item"]
        if len(loops) != 1:
            die("%s: expected one `for item in ...` prompt loop, found %d" % (path, len(loops)))
        body, seen_raw = [], False
        for st in loops[0].body:
            if not isinstance(st, ast.Assign):
                break
            tg = st.targets[0]
            body.append(st)
            if isinstance(tg, ast.Name) and tg.id == "raw":
                seen_raw = True
                break
        if not seen_raw:
            die("%s: no `raw = ...` statement in the prompt loop" % path)
        names = [ast.unparse(st.targets[0]) for st in body]
        if names[-3:] != ["instr", "q_block", "raw"]:
            die("%s: the prompt loop does not end with instr, q_block, raw (found %s)" % (path, names))
        _CACHE[key] = compile(ast.Module(body=body, type_ignores=[]), path, "exec")
    return _CACHE[key]


def fair_max_tokens(task):
    """max_tokens and the engine line of the fair script, read from its source (checked against the freeze values)."""
    src = _source(TASKS[task]["fair"])
    m = re.findall(r"SamplingParams\(temperature=0\.0, max_tokens=(\d+), stop=\[\"<\|im_end\|>\"\]\)", src)
    e = re.findall(r"LLM\(model=MODEL_PATH, dtype=\"bfloat16\", max_model_len=(\d+), gpu_memory_utilization=([0-9.]+)\)", src)
    if len(m) != 1 or len(e) != 1:
        die("%s: the SamplingParams / LLM lines are not as expected" % TASKS[task]["fair"])
    if int(m[0]) != TASKS[task]["max_tokens"] or int(e[0][0]) != STUDENT_MAX_MODEL_LEN or float(e[0][1]) != STUDENT_GPU_UTIL:
        die("%s: max_tokens / engine values differ from the freeze" % TASKS[task]["fair"])
    if 'MODEL_PATH = "%s"' % STUDENT_PATH not in src or "enable_thinking=True" not in src:
        die("%s: model path / thinking flag differ from the freeze" % TASKS[task]["fair"])
    return int(m[0])


# ----------------------------------------------------------------------------
# Rows and prompt parts. The builders read only the text fields of a row.
# ----------------------------------------------------------------------------
TEXT_FIELDS = {"csqa": ("question", "choices"), "piqa": ("goal", "sol1", "sol2"), "gsm8k": ("question",)}


def _text_only(task, row):
    """A copy of the row with its text fields only: the gold field never reaches a builder."""
    return OrderedDict((k, row[k]) for k in TEXT_FIELDS[task])


def load_rows(task, mode="FULL"):
    """[{rid, pos, ur, en}] in file order. ur / en carry the text fields only. TEST = rows 0-4."""
    t = TASKS[task]
    ur, en = read_jsonl(t["ur"]), read_jsonl(t["en"])
    if len(ur) != t["n"] or len(en) != t["n"]:
        die("%s: %d Urdu rows, %d English rows, expected %d" % (task, len(ur), len(en), t["n"]))
    rows = [OrderedDict([("rid", "%s_%04d" % (task, i)), ("pos", i), ("ur", _text_only(task, a)), ("en", _text_only(task, b))])
            for i, (a, b) in enumerate(zip(ur, en))]
    return rows[:TEST_N] if mode == "TEST" else rows


def field_texts(task, item):
    if task == "csqa":
        ch = item["choices"]
        return [item["question"]] + [ch["text"][ch["label"].index(x)] if x in ch["label"] else "" for x in "ABCDE"]
    if task == "piqa":
        return [item["goal"], item["sol1"], item["sol2"]]
    return [item["question"]]


def one_line(s):
    return " ".join(str(s).split())


def item_block(task, texts):
    """9.4: one line per label, label + ': ' + text; white-space runs inside a field are one space."""
    labels = TASKS[task]["labels"]
    if len(texts) != len(labels):
        die("%s: %d texts for %d labels" % (task, len(texts), len(labels)))
    return "\n".join("%s: %s" % (lab, one_line(tx)) for lab, tx in zip(labels, texts))


def ur_block(task, row):
    return item_block(task, field_texts(task, row["ur"]))


def og_block(task, row):
    return item_block(task, field_texts(task, row["en"]))


def parse_translation(task, text):
    """9.5: the R1 item block of a valid translation, or None."""
    lines = [l.strip() for l in (text or "").strip().split("\n")]
    lines = [l for l in lines if l]
    labels = TASKS[task]["labels"]
    if len(lines) != len(labels):
        return None
    texts = []
    for lab, l in zip(labels, lines):
        if not l.startswith(lab + ":"):
            return None
        tx = l[len(lab) + 1:].strip()
        if not tx:
            return None
        texts.append(tx)
    return item_block(task, texts)


def ur_parts(task, row):
    """(instruction, item lines, trailer) of the fair script's user message; the three joined by blank lines are that message."""
    ns = {"item": row["ur"]}
    exec(_fair_statements(task), ns)
    instr, q_block, raw = ns["instr"], ns["q_block"], ns["raw"]
    if raw != instr + "\n\n" + q_block:
        die("%s %s: raw is not instr + blank line + q_block" % (task, row["rid"]))
    cut = q_block.find("\n\n")
    if cut < 0:
        die("%s %s: q_block has no blank line before the trailer" % (task, row["rid"]))
    items, trailer = q_block[:cut], q_block[cut + 2:]
    if trailer != ur_trailer(task):
        die("%s %s: the trailer differs from the task's trailer (a field holds a blank line?)" % (task, row["rid"]))
    return instr, items, trailer


def ur_trailer(task):
    """The trailer of the fair prompt: the part of q_block after the item lines, taken from a probe row."""
    key = "trailer_" + task
    if key not in _CACHE:
        probe = {"csqa": {"question": "Q", "choices": {"label": list("ABCDE"), "text": ["a", "b", "c", "d", "e"]}},
                 "piqa": {"goal": "G", "sol1": "S1", "sol2": "S2"}, "gsm8k": {"question": "Q"}}[task]
        ns = {"item": probe}
        exec(_fair_statements(task), ns)
        q = ns["q_block"]
        last = {"csqa": "E: e", "piqa": "S2", "gsm8k": "Q"}[task]
        if q.count(last + "\n\n") != 1:
            die("%s: cannot find the end of the item lines in the probe q_block" % task)
        _CACHE[key] = q.split(last + "\n\n", 1)[1]
    return _CACHE[key]


def ur_user(task, row, block=None):
    """The UR user message; with `block`, that block sits between the last item line and the trailer (F4)."""
    instr, items, trailer = ur_parts(task, row)
    parts = [instr, items] + ([block] if block else []) + [trailer]
    return "\n\n".join(parts)


def en_user(task, block):
    return EN[task]["instruction"] + "\n\n" + block + "\n\n" + EN[task]["trailer"]


def build_users(task, arm, rows, r1=None, facts=None):
    """[{rid, status, user}] for one arm. status 'ok' or 'reading_failed' (9.5: such a row is not generated).
    r1 = {rid: R1 item block or None}; facts = {rid: [fact, ...]} (S_tr)."""
    if arm not in arms_of(task):
        die("arm %s is not defined on %s" % (arm, task))
    spec = ARMS[arm]
    kind = spec["kind"]
    if spec["needs_r1"] and r1 is None:
        die("arm %s needs the sealed R1 readings" % arm)
    if kind == "ur+facts" and facts is None:
        die("arm %s needs the sealed teacher facts" % arm)
    out = []
    for row in rows:
        rid = row["rid"]
        rec = OrderedDict([("rid", rid), ("status", "ok"), ("user", None)])
        if spec["needs_r1"] and r1[rid] is None:
            rec["status"] = "reading_failed"
            out.append(rec)
            continue
        if kind == "ur":
            rec["user"] = ur_user(task, row)
        elif kind == "ur+r1":
            rec["user"] = ur_user(task, row, tr_label_line() + "\n" + r1[rid])
        elif kind == "ur+og":
            rec["user"] = ur_user(task, row, tr_label_line() + "\n" + og_block(task, row))
        elif kind == "ur+facts":
            fl = list(facts[rid])
            rec["n_facts"] = len(fl)
            rec["user"] = ur_user(task, row, facts_block(fl) if fl else None)
        elif kind == "en+r1":
            rec["user"] = en_user(task, r1[rid])
        elif kind == "en+og":
            rec["user"] = en_user(task, og_block(task, row))
        else:
            die("unknown prompt kind %r" % kind)
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Output paths, start / resume / seal
# ----------------------------------------------------------------------------
def base_dir(mode):
    if mode not in MODES:
        die("unknown mode %r" % mode)
    return TEST_ROOT if mode == "TEST" else OUT_ROOT


def _paths(d, stem):
    return OrderedDict([("dir", d), ("rows", os.path.join(d, "d14_%s.jsonl" % stem)), ("work", os.path.join(d, "d14_%s.work.jsonl" % stem)),
                        ("start", os.path.join(d, "START_%s.json" % stem)), ("seal", os.path.join(d, "SEAL_%s.json" % stem))])


def r1_paths(task, mode):
    return _paths(os.path.join(base_dir(mode), "r1"), "r1_" + task)


def facts_paths(task, mode):
    return _paths(os.path.join(base_dir(mode), "facts"), "facts_" + task)


def arm_paths(task, arm, mode):
    return _paths(os.path.join(base_dir(mode), MODEL_DIR[ARMS[arm]["model"]], task), arm)


def is_sealed(pth):
    """True if the seal exists and the row file matches it; fatal on a mismatch."""
    if not os.path.exists(rp(pth["seal"])):
        return False
    seal = json.load(open(rp(pth["seal"]), encoding="utf-8"), object_pairs_hook=OrderedDict)
    ent = seal["files"].get(pth["rows"])
    if ent is None or md5_file(pth["rows"]) != ent["md5"]:
        die("%s exists but the row file does not match it" % pth["seal"])
    return True


def open_rows(pth, expected, start_obj, key="prompt"):
    """Start or resume a row file written in row order. expected = the rows of this run in order ({rid, status, <key>}).
    Returns the number of finished rows, or None if already sealed. Finished rows are checked against this run and
    never generated again."""
    os.makedirs(guard_out(pth["dir"]), exist_ok=True)
    if is_sealed(pth):
        print("[sealed] %s is already sealed; nothing is generated again" % pth["rows"], flush=True)
        return None
    done = read_rows(pth["rows"], repair=True)
    if len(done) > len(expected):
        die("%s: %d rows on disk, %d expected" % (pth["rows"], len(done), len(expected)))
    for i, (d, e) in enumerate(zip(done, expected)):
        if d["rid"] != e["rid"] or d["status"] != e["status"] or d.get(key) != e.get(key):
            die("%s row %d: the finished row does not match this run (rid %s)" % (pth["rows"], i, e["rid"]))
    add_run(pth, start_obj, len(done))
    if done:
        print("[resume] %s: %d finished rows kept; continuing at row %d" % (pth["rows"], len(done), len(done)), flush=True)
    return len(done)


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


def seal_rows(pth, expected_rids, head, extra=None):
    """Write the SEAL file when every row is present in order. Not scored: counts, MD5s, seconds and tokens only."""
    recs = read_rows(pth["rows"])
    if [r["rid"] for r in recs] != list(expected_rids):
        die("%s: %d rows on disk do not equal the %d rows of the run in order" % (pth["rows"], len(recs), len(expected_rids)))
    for r in recs:
        for k in ("pred", "correct", "gold"):
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
        ("hostname", socket.gethostname()), ("d14_freeze_md5", FREEZE_MD5), ("scripts", script_md5s()), ("git", git_state()),
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


def load_r1(task, mode, rows):
    """{rid: R1 item block, or None for a reading_failed row} from the sealed R1 file of the task."""
    pth = r1_paths(task, mode)
    if not is_sealed(pth):
        die("the R1 readings of %s (%s) are not sealed: %s" % (task, mode, pth["seal"]))
    recs = read_rows(pth["rows"])
    if [r["rid"] for r in recs] != [r["rid"] for r in rows]:
        die("%s: the sealed R1 rows are not the rows of this run" % pth["rows"])
    out = OrderedDict()
    for r, row in zip(recs, rows):
        if r["user"] != ur_block(task, row):
            die("%s %s: the sealed R1 record was made from a different Urdu item block" % (pth["rows"], r["rid"]))
        if r["status"] == "ok":
            if not r["r1_block"] or parse_translation(task, r["r1_block"]) != r["r1_block"]:
                die("%s %s: the sealed R1 block is not a valid item block" % (pth["rows"], r["rid"]))
            out[r["rid"]] = r["r1_block"]
        elif r["status"] == "reading_failed":
            out[r["rid"]] = None
        else:
            die("%s %s: unknown status %r" % (pth["rows"], r["rid"], r["status"]))
    return out, OrderedDict([("file", pth["rows"]), ("md5", md5_file(pth["rows"])), ("reading_failed", sum(1 for v in out.values() if v is None))])


def load_facts(task, mode, rows, r1):
    """{rid: [facts]} from the sealed teacher-facts file of the task (a reading_failed row has an empty list and is
    never generated in S_tr)."""
    pth = facts_paths(task, mode)
    if not is_sealed(pth):
        die("the teacher facts of %s (%s) are not sealed: %s" % (task, mode, pth["seal"]))
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
# API: budget ledger and client (9.9). The key never leaves the client object.
# ----------------------------------------------------------------------------
class BudgetStop(Exception):
    pass


def _ledger_update(fn):
    ap = guard_out(LEDGER)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    with open(ap + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if os.path.exists(ap):
                with open(ap, encoding="utf-8") as f:
                    led = json.load(f, object_pairs_hook=OrderedDict)
            else:
                led = OrderedDict([("what", "D14 API budget ledger: every request sent counts as one call; tokens = prompt + completion from the usage fields"),
                                   ("cap_calls", BUDGET_CALLS), ("cap_tokens", BUDGET_TOKENS), ("token_stop_at", TOKEN_STOP_AT),
                                   ("calls", 0), ("prompt_tokens", 0), ("completion_tokens", 0), ("tokens", 0),
                                   ("answered_without_usage", 0), ("stopped", None), ("by_tag", OrderedDict()), ("balance", [])])
            res = fn(led)
            tmp = ap + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(led, f, indent=1, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, ap)
            return res
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def ledger_read():
    return _ledger_update(lambda led: json.loads(json.dumps(led), object_pairs_hook=OrderedDict))


def _tag(led, tag):
    return led["by_tag"].setdefault(tag, OrderedDict([("calls", 0), ("prompt_tokens", 0), ("completion_tokens", 0)]))


def ledger_reserve(tag):
    """Count one call before it is sent. Raises BudgetStop if a cap is reached or API work was stopped."""
    def fn(led):
        if led["stopped"]:
            return "stopped: " + led["stopped"]
        if led["calls"] + 1 > BUDGET_CALLS:
            led["stopped"] = "call cap reached (%d calls)" % led["calls"]
            return led["stopped"]
        if led["tokens"] >= TOKEN_STOP_AT:
            led["stopped"] = "token cap reached (%d tokens)" % led["tokens"]
            return led["stopped"]
        led["calls"] += 1
        _tag(led, tag)["calls"] += 1
        return None
    why = _ledger_update(fn)
    if why:
        raise BudgetStop(why)


def ledger_tokens(tag, usage):
    def fn(led):
        p, c = (usage or {}).get("prompt_tokens"), (usage or {}).get("completion_tokens")
        if not isinstance(p, int) or not isinstance(c, int):
            led["answered_without_usage"] += 1
            return
        led["prompt_tokens"] += p
        led["completion_tokens"] += c
        led["tokens"] = led["prompt_tokens"] + led["completion_tokens"]
        t = _tag(led, tag)
        t["prompt_tokens"] += p
        t["completion_tokens"] += c
    _ledger_update(fn)


def ledger_stop(reason):
    def fn(led):
        if not led["stopped"]:
            led["stopped"] = reason
    _ledger_update(fn)


QUOTA_WORDS = ("insufficient", "arrearage", "balance", "额度", "余额", "欠费")


def is_quota_error(status, text):
    """A quota or balance error of the gateway or the provider (9.9: API work stops on the first one)."""
    low = (text or "").lower()
    if status == 402:
        return True
    if any(w in low for w in QUOTA_WORDS):
        return True
    return status in (401, 403) and "quota" in low


KEY_FILE = "~/.urbench_api_env"
NAME_BASE, NAME_KEY = "URB_API_BASE", "URB_API_KEY"


def _read_key_file():
    """Parse NAME=value lines of the key file. Returns (base, key). Prints nothing from the file."""
    path = os.path.expanduser(KEY_FILE)
    if not os.path.exists(path):
        die("key file %s does not exist" % KEY_FILE)
    vals = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export "):].strip()
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            vals[k.strip()] = v
    base, key = vals.get(NAME_BASE, "").strip(), vals.get(NAME_KEY, "").strip()
    if not base or not key:
        die("key file %s: %s present: %s, %s present: %s (values are never shown)" % (KEY_FILE, NAME_BASE, bool(base), NAME_KEY, bool(key)))
    return base, key


def api_settings():
    """The A1.6 settings (d11_api_direct) and the D11-R retry policy (d11r_common), imported."""
    if "api" not in _CACHE:
        for p in (EFBPT + "d11_api_direct.py", EFBPT + "d11r_common.py"):
            _source(p)                       # MD5 check before the import
        import d11_api_direct as A
        import d11r_common as R
        _CACHE["api"] = OrderedDict([
            ("model", A.MODEL), ("request_extra", OrderedDict(A.REQUEST_EXTRA)), ("temperature", A.TEMPERATURE),
            ("concurrency", A.CONCURRENCY), ("timeout_s", A.REQUEST_TIMEOUT_S), ("retry_status", sorted(A.RETRY_STATUS)),
            ("backoff_s", list(R.API_BACKOFF_S)), ("attempts", R.API_ATTEMPTS)])
        _CACHE["scrub"] = (R.scrub, R.scrub_obj)
    return _CACHE["api"]


class Api:
    """Request body and retry loop of d13_api.Api / d11_api_direct.Api (policy unchanged); the messages and
    max_tokens are arguments; every request is counted in the ledger before it is sent. The key is read from the
    key file inside this process, kept in one private attribute, and goes out only in the Authorization header."""

    def __init__(self):
        import requests
        from urllib.parse import urlparse
        self.requests = requests
        self.s = api_settings()
        self.scrub, self.scrub_obj = _CACHE["scrub"]
        base, key = _read_key_file()
        self._key = key
        self.base = base.rstrip("/")
        self.url = self.base + "/chat/completions"
        self.host = urlparse(base).netloc or "<unparsed>"
        self.secrets = (key,)
        self.session = requests.Session()

    def call(self, messages, max_tokens, tag):
        """One row request with the retry policy. Returns {status, attempts, latency_s, response, error, refused}.
        Raises BudgetStop when a cap is reached or on a quota / balance error."""
        s = self.s
        body = OrderedDict([("model", s["model"]), ("messages", messages), ("temperature", s["temperature"]), ("max_tokens", max_tokens)])
        body.update(s["request_extra"])
        attempts, err, status, resp_json, last_latency = 0, None, None, None, None
        while attempts < s["attempts"]:
            ledger_reserve(tag)
            attempts += 1
            t0 = time.time()
            try:
                r = self.session.post(self.url, headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
                                      data=json.dumps(body), timeout=s["timeout_s"])
                last_latency = time.time() - t0
                status = r.status_code
                try:
                    j = r.json()
                except Exception:
                    j = None
                if status == 200 and isinstance(j, dict) and j.get("choices"):
                    resp_json, err = j, None
                    ledger_tokens(tag, j.get("usage"))
                    break
                err = "HTTP %s: %s" % (status, self.scrub(r.text[:600], self.secrets))
                if is_quota_error(status, r.text):
                    ledger_stop("quota or balance error: " + err[:300])
                    raise BudgetStop("quota or balance error: " + err[:300])
                if status in s["retry_status"] and attempts < s["attempts"]:
                    time.sleep(s["backoff_s"][min(attempts - 1, len(s["backoff_s"]) - 1)])
                    continue
                break
            except self.requests.RequestException as e:
                last_latency = time.time() - t0
                status = None
                err = "network: %s" % self.scrub(str(e)[:300], self.secrets)
                if attempts < s["attempts"]:
                    time.sleep(s["backoff_s"][min(attempts - 1, len(s["backoff_s"]) - 1)])
                    continue
                break
        return OrderedDict([("status", status), ("attempts", attempts), ("latency_s", last_latency),
                            ("response", self.scrub_obj(resp_json, self.secrets)), ("error", err),
                            ("refused", resp_json is None and status == 400)])


def response_text(resp):
    """(text, finish_reason, usage) of an ok response."""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return text, ch.get("finish_reason"), resp.get("usage") or {}


def log_full_prompt(label, prompt):
    print("\n" + "-" * 30 + " FULL PROMPT: " + label + " " + "-" * 30)
    print(prompt)
    print("-" * 30 + " END PROMPT " + "-" * 30 + "\n", flush=True)


# ----------------------------------------------------------------------------
# Chat templates and context length (9.3, 9.10)
# ----------------------------------------------------------------------------
def load_tok(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def student_chat(tok, user):
    """The fair scripts' call: one user message, generation prompt added, thinking on."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=True)


def teacher_render(tok, msgs):
    """(text, ids) of the teacher's chat template with the generation prompt added (as d11_teacher / d13_common)."""
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
    if tok(text, add_special_tokens=False)["input_ids"] != ids:
        die("teacher chat template: the stored prompt text does not re-tokenize to the generated ids")
    return text, ids


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def teacher_frozen():
    """The frozen teacher code (d10_teacher, d11o_teacher), imported after an MD5 check."""
    if "teacher" not in _CACHE:
        for p in (EFBPT + "d10_teacher.py", EFBPT + "d11o_teacher.py"):
            _source(p)
        import d10_teacher as T
        import d11o_teacher as O
        system, user_t = O.oracle_prompts(T.T_SYSTEM, T.T_USER_TEMPLATE)
        if user_t.count("{question_en}") != 1:
            die("the D11 teacher user template does not hold exactly one {question_en} slot")
        args = OrderedDict(T.VLLM_ARGS)
        args["max_num_seqs"] = TEACHER_MAX_NUM_SEQS          # F4: the D10 engine with max_num_seqs 16
        # Amendment 1 A1.2: with max_num_seqs above 1 vLLM 0.13.0 sizes its start-up run with max_num_batched_tokens
        # 8192 and the engine does not start on an L20; the D10 engine (max_num_seqs 1) runs with 4096 = max_model_len.
        # That value is set explicitly. It changes no prompt, model or decoding.
        args["max_num_batched_tokens"] = args["max_model_len"]
        _CACHE["teacher"] = OrderedDict([
            ("path", T.TEACHER_PATH), ("vllm_version", T.VLLM_VERSION), ("vllm_args", args), ("sampling", OrderedDict(T.SAMPLING)),
            ("facts_system", system), ("facts_user_template", user_t), ("facts_max_tokens", T.T1_MAX_TOKENS),
            ("max_facts", T.MAX_FACTS), ("parse", T.parse_teacher_json), ("verdict_filter", T.verdict_filter),
            ("verdict_re", T.VERDICT_RE.pattern), ("yesno_re", T.YESNO_RE.pattern)])
    return _CACHE["teacher"]


# 9.7: the D14 filter on the lines kept by the D10 verdict filter
D14_WORD_RE = re.compile(r"option|choice|solution 1|solution 2", re.IGNORECASE)
D14_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])[A-E]:|\([A-E]\)")


def d14_filter(facts):
    """(kept, removed) under the D14 filter."""
    kept, removed = [], []
    for f in facts:
        (removed if (D14_WORD_RE.search(f) or D14_LABEL_RE.search(f)) else kept).append(f)
    return kept, removed


def facts_user(block):
    """The D11 teacher fact prompt with the R1 item block in its question slot."""
    return teacher_frozen()["facts_user_template"].replace("{question_en}", block)


# ----------------------------------------------------------------------------
# Chunked generation for the vLLM jobs (9.6, 9.8)
# ----------------------------------------------------------------------------
def generate_rows(pth, expected, k, gen_fn, make_rec, label):
    """Generate rows k.. of `expected` in row order, CHUNK rows per call. Rows whose status is not 'ok' are written
    in place without a generation. gen_fn(list of ok rows) -> one dict per row (text, finish_reason, n_gen_tokens,
    n_prompt_tokens). Every finished chunk is appended and synced. Seconds of a row = chunk seconds / generated rows."""
    t0 = time.time()
    with open(guard_out(pth["rows"]), "ab") as f:
        for lo in range(k, len(expected), CHUNK):
            block = expected[lo:lo + CHUNK]
            ok = [e for e in block if e["status"] == "ok"]
            t1 = time.time()
            outs = gen_fn(ok) if ok else []
            secs = time.time() - t1
            if len(outs) != len(ok):
                die("%s: the engine returned %d outputs for %d prompts" % (label, len(outs), len(ok)))
            it = iter(outs)
            for e in block:
                o = next(it) if e["status"] == "ok" else None
                rec = make_rec(e, o)
                rec["seconds"] = round(secs / len(ok), 3) if o is not None else None
                rec["chunk_seconds"] = round(secs, 3) if o is not None else None
                rec["chunk_rows"] = len(ok) if o is not None else None
                append_row(f, rec)
            trunc = sum(1 for o in outs if o["finish_reason"] == "length")
            print("  [chunk] %s rows %d..%d of %d: generated %d, new tokens %d, reached max_tokens %d, %.1fs  (%.0fs)"
                  % (label, lo, lo + len(block) - 1, len(expected), len(ok), sum(o["n_gen_tokens"] for o in outs), trunc, secs, time.time() - t0), flush=True)


def vllm_outputs(outs):
    """vLLM RequestOutput list -> the dicts generate_rows expects."""
    res = []
    for o in outs:
        c = o.outputs[0]
        res.append(OrderedDict([("text", c.text), ("finish_reason", c.finish_reason), ("n_gen_tokens", len(c.token_ids)),
                                ("n_prompt_tokens", len(o.prompt_token_ids) if o.prompt_token_ids is not None else None)]))
    return res
