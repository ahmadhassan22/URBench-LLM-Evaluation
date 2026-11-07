#!/usr/bin/env python3
"""
d11r_common.py — shared constants and helpers for the D11-R reader benchmark
(docs/EFBPT_D11R_READER_FREEZE.md, MD5 pinned below).

Item sets (§1), the reader instruction (§2), post-processing (§2), the reading
record schema, R_START / R_SEAL writers and small md5 / jsonl helpers. Nothing
here touches a model or the network. The API key is never read here.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

FREEZE_PATH = "docs/EFBPT_D11R_READER_FREEZE.md"
FREEZE_MD5 = "ed260e0ea2373337e8a3fb850bebe963"
# Amendment 1 (A1.1 thinking setting, A1.2 M1 denominator, A1.3 data force-add,
# A1.4 prices): recorded in every R_START / R_SEAL
AMEND1_PATH = "docs/EFBPT_D11R_AMENDMENT_1.md"
AMEND1_MD5 = "fb7bc3c922bb28d667241a7dd75b6938"

MAPPED_PATH = "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl"
MAPPED_MD5 = "bb982798279f5d519ddef01f5cb7ba8c"

# §1 item sets. eval458 = the is_eval rows of the mapped file, in file order.
ITEM_SETS = OrderedDict([
    ("rdev300", OrderedDict([("path", "data/strategyqa_official/efbpt/rdev300_seed2026.jsonl"),
                             ("md5", "e307ad857b2f646f423bcd09fd3060d7"), ("n", 300), ("filter", None)])),
    ("dev200", OrderedDict([("path", "data/strategyqa_official/dev200_seed4242.jsonl"),
                            ("md5", "4907f292175b543b17799e6e37b6724b"), ("n", 200), ("filter", None)])),
    ("eval458", OrderedDict([("path", MAPPED_PATH), ("md5", MAPPED_MD5), ("n", 458), ("filter", "is_eval")])),
])
SCORABLE_SETS = ("rdev300", "dev200")          # §1 / §5: eval458 readings are sealed unread

# §2 — the one instruction for every LLM reader (verbatim from the freeze; the
# freeze wraps at ~78 columns, wraps are joined with one space).
SYSTEM_PROMPT = (
    "You are an expert Urdu-to-English translator. Translate the question "
    "exactly, preserving its meaning. Named entities in Urdu questions are usually "
    "transliterations of English names (people, places, brands, titles, works); "
    "write each in its standard English spelling. Do not answer the question and "
    "do not explain. Output only the English question, on one line."
)
USER_TEMPLATE = "{question_ur}"

API_MODELS = ("qwen3.8-max", "deepseek-v4-pro", "deepseek-v4-flash", "qwen3.7-flash")
API_TEMPERATURE = 0
API_MAX_TOKENS = 256
API_CONCURRENCY = 4
API_BACKOFF_S = (2, 4, 8)
API_ATTEMPTS = 3
# §2 thinking switches, tried in this order by the PROBE; "default" = no switch
THINKING_VARIANTS = OrderedDict([
    ("enable_thinking_false", {"enable_thinking": False}),
    ("thinking_disabled", {"thinking": {"type": "disabled"}}),
    ("reasoning_effort_none", {"reasoning_effort": "none"}),
    ("default", {}),
])

LOCAL_READERS = OrderedDict([
    ("llama", "L1_llama70b_awq"),
    ("gemma", "L2_gemma2_27b_nf4"),
    ("qwen14b", "L3_qwen3_14b_nf4"),
    ("nllb", "L4_nllb200_3.3b"),
])
LOCAL_MAX_NEW = 128

OUT_ROOT = "outputs/efbpt/d11r"
PROBE_DIR = os.path.join(OUT_ROOT, "_probe")
TEST_ROOT = os.path.join(OUT_ROOT, "_test")
SCORES_DIR = os.path.join(OUT_ROOT, "_scores")


def die(msg):
    print("FATAL: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def api_reader_name(model):
    return "api_" + model


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_text(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def file_entry(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = sum(1 for l in f if l.strip())
    return OrderedDict([("md5", md5_file(path)), ("bytes", os.path.getsize(path)), ("lines", lines)])


def git_state():
    def run(*a):
        try:
            return subprocess.check_output(["git"] + list(a), cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    dirty = run("status", "--porcelain")
    return OrderedDict([("head", run("rev-parse", "HEAD")), ("dirty", bool(dirty) if dirty is not None else None)])


def verify_freeze():
    p = os.path.join(ROOT, FREEZE_PATH)
    if not os.path.exists(p):
        die("freeze missing: " + FREEZE_PATH)
    got = md5_file(p)
    if got != FREEZE_MD5:
        die("freeze MD5 %s != pinned %s" % (got, FREEZE_MD5))
    pa = os.path.join(ROOT, AMEND1_PATH)
    if not os.path.exists(pa):
        die("amendment missing: " + AMEND1_PATH)
    got_a = md5_file(pa)
    if got_a != AMEND1_MD5:
        die("amendment 1 MD5 %s != pinned %s" % (got_a, AMEND1_MD5))
    return got


def contracts():
    """freeze + amendment identity, written into every R_START / R_SEAL."""
    return OrderedDict([("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", FREEZE_MD5)])),
                        ("amendment_1", OrderedDict([("path", AMEND1_PATH), ("md5", AMEND1_MD5)]))])


def _is_true(v):
    return v is True or str(v) == "True"


def load_items(set_name, with_gold=False, limit=None):
    """Rows of an item set: qid, question_ur (+ question_en, answer,
    evidence_paragraph_ids when with_gold). Readers call with_gold=False so the
    gold never enters a reader process (§5). MD5 and n asserted."""
    if set_name not in ITEM_SETS:
        die("unknown item set %r (choose from %s)" % (set_name, ", ".join(ITEM_SETS)))
    spec = ITEM_SETS[set_name]
    p = os.path.join(ROOT, spec["path"])
    got = md5_file(p)
    if got != spec["md5"]:
        die("%s MD5 %s != pinned %s" % (spec["path"], got, spec["md5"]))
    rows = read_jsonl(p)
    if spec["filter"] == "is_eval":
        rows = [r for r in rows if _is_true(r.get("is_eval"))]
    if len(rows) != spec["n"]:
        die("%s has %d rows, expected %d" % (set_name, len(rows), spec["n"]))
    qids = [r["urbench_qid"] for r in rows]
    if len(set(qids)) != len(qids):
        die("%s: duplicate qids" % set_name)
    out = []
    for r in rows:
        q = str(r.get("question_ur", "")).strip()
        if not q:
            die("%s qid %s: empty question_ur" % (set_name, r["urbench_qid"]))
        item = OrderedDict([("qid", r["urbench_qid"]), ("question_ur", q)])
        if with_gold:
            item["question_en"] = str(r.get("question_en", "")).strip()
            item["answer"] = r.get("answer")
            ev = r.get("evidence_paragraph_ids")
            if isinstance(ev, str):
                try:
                    import ast
                    ev = ast.literal_eval(ev)
                except Exception:
                    ev = None
            item["evidence_paragraph_ids"] = list(ev) if ev else []
        out.append(item)
    if limit:
        out = out[:limit]
    return out


def item_set_manifest():
    m = OrderedDict()
    for k, spec in ITEM_SETS.items():
        m[k] = OrderedDict([("path", spec["path"]), ("md5", spec["md5"]), ("n", spec["n"])])
    return m


# ----------------------------------------------------------------------------
# §2 post-processing: first non-empty line, strip quotes / markdown
# ----------------------------------------------------------------------------
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.S)
QUOTES = "\"'“”‘’«»`"
MD_EDGE_RE = re.compile(r"^[\s\*_`#>\-•]+|[\s\*_`]+$")
LATIN_RE = re.compile(r"[A-Za-z]")


def postprocess(raw_text):
    """Returns (reading, flags). flags records what was stripped so the audit
    can see it: think_block, multi_line, quotes, markdown, empty."""
    flags = OrderedDict()
    text = raw_text or ""
    if "<think>" in text:
        flags["think_block"] = True
        text = THINK_RE.sub("", text)
        text = OPEN_THINK_RE.sub("", text)      # unterminated think block (truncated)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 1:
        flags["multi_line"] = len(lines)
    line = lines[0] if lines else ""
    before = line
    line = MD_EDGE_RE.sub("", line).strip()
    if line != before:
        flags["markdown"] = True
    before = line
    line = line.strip(QUOTES).strip()
    if line != before:
        flags["quotes"] = True
    # a code fence line ("```") leaves nothing: fall through to the next line
    if not line and len(lines) > 1:
        for cand in lines[1:]:
            cand = MD_EDGE_RE.sub("", cand).strip().strip(QUOTES).strip()
            if cand:
                line = cand
                flags["skipped_empty_first_line"] = True
                break
    if not line:
        flags["empty"] = True
    return line, flags


def latin_letters(s):
    return len(LATIN_RE.findall(s or ""))


def has_arabic_script(s):
    for ch in s or "":
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or \
                0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            return True
    return False


def scrub(text, secrets):
    """Defensive: remove any secret string from text before it is stored or
    printed. Secrets never appear in API responses, but nothing is stored
    without passing through here."""
    if not isinstance(text, str):
        return text
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "<REDACTED>")
    return text


def scrub_obj(obj, secrets):
    if isinstance(obj, dict):
        return OrderedDict((scrub(k, secrets), scrub_obj(v, secrets)) for k, v in obj.items())
    if isinstance(obj, list):
        return [scrub_obj(v, secrets) for v in obj]
    return scrub(obj, secrets)


# ----------------------------------------------------------------------------
# Reading record (one schema for API and local readers)
# ----------------------------------------------------------------------------
def make_record(qid, set_name, reader, question_ur, raw_text, finish_reason, prompt_tokens,
                output_tokens, reasoning_tokens, reasoning_evidence, latency_s, ok, error, extra=None):
    reading, flags = postprocess(raw_text if ok else "")
    rec = OrderedDict([
        ("qid", qid), ("set", set_name), ("reader", reader), ("question_ur", question_ur),
        ("reading", reading), ("raw_text", raw_text), ("finish_reason", finish_reason),
        ("prompt_tokens", prompt_tokens), ("output_tokens", output_tokens),
        ("reasoning_tokens", reasoning_tokens), ("reasoning_evidence", reasoning_evidence),
        ("latency_s", None if latency_s is None else round(latency_s, 3)),
        ("ok", bool(ok)), ("error", error), ("flags", flags),
        ("latin_letters", latin_letters(reading)), ("urdu_script", has_arabic_script(reading)),
        ("ts", time.strftime("%Y-%m-%d %H:%M:%S")),
    ])
    if extra:
        rec.update(extra)
    return rec


def reader_dir(reader, test=False):
    return os.path.join(TEST_ROOT if test else OUT_ROOT, reader)


def readings_path(reader, set_name, test=False):
    return os.path.join(reader_dir(reader, test), "readings_%s.jsonl" % set_name)


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def write_start(rdir, info):
    """R_START.json once per reader directory; a later run appends to
    info["runs"] instead of rewriting the start record."""
    p = os.path.join(rdir, "R_START.json")
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(p):
        cur = json.load(open(p, encoding="utf-8"), object_pairs_hook=OrderedDict)
        cur.setdefault("runs", []).append(OrderedDict([("start_time", info.get("start_time")),
                                                       ("hostname", info.get("hostname")),
                                                       ("slurm_job_id", info.get("slurm_job_id")),
                                                       ("sets", info.get("sets")),
                                                       ("script_md5", info.get("script_md5")),
                                                       ("git", info.get("git")),
                                                       ("contracts", contracts())]))
        write_json(p, cur)
        print("  [verified] updated %s (run %d)" % (p, len(cur["runs"]) + 1), flush=True)
        return cur
    info = OrderedDict(info)
    info.update(contracts())
    info["item_sets"] = item_set_manifest()
    write_json(p, info)
    print("  [verified] wrote %s" % p, flush=True)
    return info


def update_seal(rdir, reader, set_name, path, n_expected, extra=None):
    """R_SEAL.json per reader: one entry per completed item set, rewritten each
    time a set completes. Verifies the line count against the item set."""
    p = os.path.join(rdir, "R_SEAL.json")
    if os.path.exists(p):
        seal = json.load(open(p, encoding="utf-8"), object_pairs_hook=OrderedDict)
    else:
        seal = OrderedDict([("reader", reader)])
        seal.update(contracts())
        seal.update([("item_sets", item_set_manifest()), ("files", OrderedDict()), ("sets_complete", [])])
    fe = file_entry(path)
    if fe["lines"] < n_expected:
        die("%s: %d lines < %d items; not sealing" % (path, fe["lines"], n_expected))
    fe["n_items"] = n_expected
    fe["sealed_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if extra:
        fe.update(extra)
    seal["files"][os.path.relpath(path, ROOT) if os.path.isabs(path) else path] = fe
    if set_name not in seal["sets_complete"]:
        seal["sets_complete"].append(set_name)
    seal["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(p, seal)
    print("  [sealed] %s  %s md5 %s lines %d" % (set_name, path, fe["md5"], fe["lines"]), flush=True)
    return seal


def done_qids(path):
    """qids with an ok record in an existing readings file (resume support)."""
    if not os.path.exists(path):
        return set(), 0
    recs = read_jsonl(path)
    return {r["qid"] for r in recs if r.get("ok")}, len(recs)


def compact_jsonl(path, items_order):
    """Rewrite a readings file so that it holds exactly one ok record per qid,
    in item-set order (drops superseded error records). Used before sealing."""
    recs = read_jsonl(path)
    best = OrderedDict()
    for r in recs:
        if r.get("ok") or r["qid"] not in best:
            best[r["qid"]] = r
    ordered = [best[q] for q in items_order if q in best]
    missing = [q for q in items_order if q not in best]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(ordered), missing
