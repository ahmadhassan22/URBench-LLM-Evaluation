#!/usr/bin/env python3
"""
d11_common.py — shared constants and loaders for D11 RTG (read-then-ground).

Contract: docs/EFBPT_D11_RTG_FREEZE.md (MD5 pinned below) as amended by
docs/EFBPT_D11_RTG_AMENDMENT_1.md (A1.1 leak-guard reading rule, A1.2 V1 arm
set, A1.3 jobs, A1.4 derangement, A1.5 eval458 mechanics, A1.6 API_direct_ur)
and docs/EFBPT_D11_RTG_AMENDMENT_2.md (eval458: A2.2 arms, A2.3 per-arm jobs
and the DEV200 regression, A2.4 CPU preflight, A2.5 crash rule).

Set-aware item loading (freeze §8: --set dev200|eval458) is built on the
pinned D11-R item sets (d11r_common.load_items: file MD5 and n asserted) and,
for DEV200, cross-checked row by row against the D1 loader (rows WITH gold
urbench_facts) and the D4 loader (required titles) that every earlier student
used. Nothing here touches a model or the network; the API key is never read.
"""
import json
import os
import re
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from d11r_common import (  # noqa: E402
    ROOT, ITEM_SETS, load_items, read_jsonl, md5_file, die, git_state,
    FREEZE_PATH as D11R_FREEZE_PATH, FREEZE_MD5 as D11R_FREEZE_MD5,
    AMEND1_PATH as D11R_AMEND1_PATH, AMEND1_MD5 as D11R_AMEND1_MD5,
)
from d3_oracle_retrieval import required_titles  # noqa: E402  (the D4 loader's title rule)

os.chdir(ROOT)

D11_FREEZE_PATH = "docs/EFBPT_D11_RTG_FREEZE.md"
D11_FREEZE_MD5 = "ce7d68801c302fa5a8d173bd3ee93eb7"
D11_AMEND1_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_1.md"
D11_AMEND1_MD5 = "0b7202af45eac170ec2d8fe281269e9e"
D11_AMEND2_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_2.md"
D11_AMEND2_MD5 = "4ea8b0e6406a8a7ea05e33b212215f96"
D11R_AMEND2_PATH = "docs/EFBPT_D11R_AMENDMENT_2.md"
D11R_AMEND2_MD5 = "6c466ab2d31bffd3bfe96f43c24678f3"
GO_EVAL458_PATH = "docs/D11_GO_EVAL458.md"

SETS = ("dev200", "eval458")
TEST_N = 5                                   # freeze §9 TEST-5

# freeze §2: the three readers; DEV200 reading-file MD5s pinned there
READERS = OrderedDict([
    ("R1", OrderedDict([("reader", "api_qwen3.8-max"), ("role", "SELECTED"),
                        ("dev200_md5", "1f5b2db3c1403f63b3525a1562edd176")])),
    ("R2", OrderedDict([("reader", "api_deepseek-v4-pro"), ("role", "RUNNER-UP"),
                        ("dev200_md5", "cd77944e45a98292777382b9d9f04aef")])),
    ("R3", OrderedDict([("reader", "L1_llama70b_awq"), ("role", "best local; API-free variant"),
                        ("dev200_md5", "0d06ec1a926fe3c186c125ad723139cf")])),
])
D11R_OUT = "outputs/efbpt/d11r"

# sealed D10 references (freeze §4)
D10_A_R2 = ("outputs/efbpt/d10/student/d10_armA_r2.jsonl", "7ccec942c8f0eee08156d258baaf806d")
D10_DERANGEMENT = ("outputs/efbpt/d10/teacher/derangement.json", "8f00e2f9e54f27138964a0577a8eff52")
DERANGE_SEED = 1010

OUT_ROOT = "outputs/efbpt/d11"
TEST_ROOT = os.path.join(OUT_ROOT, "_test")
REGRESS_ROOT = os.path.join(OUT_ROOT, "_regress")   # Amendment 2 A2.3: DEV200 regression, never a result

# Amendment 2 A2.2: the eval458 readers (R1 primary + control, R3 secondary); no other arm
EVAL458_READERS = ("R1", "R3")
# Amendment 2 A2.4: the eval458 item set as declared in freeze §8
EVAL458_N, EVAL458_YES, EVAL458_NO = 458, 221, 237
# Amendment 2 A2.3: the sealed DEV200 TR file the regression is compared with
DEV200_TR_SEALED = ("outputs/efbpt/d11/student/dev200/d11_armTR.jsonl", "bab987cc492012f4e32dbcd348be9668")
REGRESS_N = 5                                # rows 0-4
FORBIDDEN_WRITE_DIRS = ("outputs/efbpt/d1", "outputs/efbpt/d4", "outputs/efbpt/d6", "outputs/efbpt/d8",
                        "outputs/efbpt/d10", "outputs/efbpt/d10v2", "outputs/efbpt/d11o", "outputs/efbpt/d11r")

# V4 (freeze §5): readings that already carry an answer word, reported
YESNO_EN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
HAAN = "".join(chr(c) for c in [0x06C1, 0x0627, 0x06BA])
NAHIN = "".join(chr(c) for c in [0x0646, 0x06C1, 0x06CC, 0x06BA])


def out_dir(job, set_name, test, sub=None):
    p = os.path.join(TEST_ROOT if test else OUT_ROOT, job, set_name)
    return os.path.join(p, sub) if sub else p


def verify_contracts():
    """D11 freeze + Amendments 1 and 2 (and the D11-R contracts the reading
    files were produced under), by pinned MD5; returned for every START / SEAL."""
    out = OrderedDict()
    for key, path, want in (("d11_freeze", D11_FREEZE_PATH, D11_FREEZE_MD5),
                            ("d11_amendment_1", D11_AMEND1_PATH, D11_AMEND1_MD5),
                            ("d11_amendment_2", D11_AMEND2_PATH, D11_AMEND2_MD5),
                            ("d11r_freeze", D11R_FREEZE_PATH, D11R_FREEZE_MD5),
                            ("d11r_amendment_1", D11R_AMEND1_PATH, D11R_AMEND1_MD5),
                            ("d11r_amendment_2", D11R_AMEND2_PATH, D11R_AMEND2_MD5)):
        if not os.path.exists(path):
            die("contract missing: " + path)
        got = md5_file(path)
        if got != want:
            die("%s MD5 %s != pinned %s" % (path, got, want))
        out[key] = OrderedDict([("path", path), ("md5", got)])
        print("[md5] %-18s %s  %s" % (key, got, path))
    return out


def require_eval458_gate(set_name):
    """freeze §8: eval458 runs only if docs/D11_GO_EVAL458.md exists."""
    if set_name == "eval458" and not os.path.exists(GO_EVAL458_PATH):
        die("refusing --set eval458: %s does not exist (freeze §8)" % GO_EVAL458_PATH)


def load_set_rows(set_name, limit=None):
    """Rows of a set: qid, question_ur, question_en, gold ('yes'/'no'), facts
    (gold urbench_facts, stripped; needed only for leak needles), required
    (gold evidence titles), evidence_ids. Pinned file MD5 and n asserted by
    d11r_common.load_items; DEV200 cross-checked against the D1 and D4 loaders."""
    if set_name not in SETS:
        die("unknown set %r" % set_name)
    items = load_items(set_name, with_gold=True)
    raw = {r["urbench_qid"]: r for r in read_jsonl(ITEM_SETS[set_name]["path"])}
    rows = []
    for it in items:
        r = raw[it["qid"]]
        a = it["answer"]
        if not isinstance(a, bool):
            die("qid %s: answer is %r, expected bool" % (it["qid"], a))
        facts = [f.strip() for f in (r.get("urbench_facts") or []) if isinstance(f, str) and f.strip()]
        rows.append(OrderedDict([("qid", it["qid"]), ("question_ur", it["question_ur"]),
                                 ("question_en", it["question_en"]), ("gold", "yes" if a else "no"),
                                 ("facts", facts), ("required", required_titles(it["evidence_paragraph_ids"])),
                                 ("evidence_ids", list(it["evidence_paragraph_ids"]))]))
    if len({r["qid"] for r in rows}) != len(rows):
        die("%s: duplicate qids" % set_name)
    if set_name == "dev200":
        from d1_eval_arms import load_rows as d1_rows
        from d4_extract_facts import load_rows as d4_rows
        a_rows, b_rows = d1_rows(), d4_rows()
        if [r["qid"] for r in a_rows] != [r["qid"] for r in rows] or [r["qid"] for r in b_rows] != [r["qid"] for r in rows]:
            die("DEV200: the D1 / D4 loaders and the pinned item set disagree on the qid order")
        for x, a, b in zip(rows, a_rows, b_rows):
            if (a["gold"], a["question_ur"], a["question_en"]) != (x["gold"], x["question_ur"], x["question_en"]):
                die("DEV200 qid %s: D1 loader disagrees on gold / question" % x["qid"])
            if [f.strip() for f in a["facts"]] != x["facts"]:
                die("DEV200 qid %s: D1 loader disagrees on urbench_facts" % x["qid"])
            if b["gold"] != x["gold"] or b["required"] != x["required"]:
                die("DEV200 qid %s: D4 loader disagrees on gold / required titles" % x["qid"])
        print("[ok] DEV200: 200 rows equal the D1 (facts) and D4 (required titles) loaders row by row")
    n_yes = sum(1 for r in rows if r["gold"] == "yes")
    print("[ok] %s: %d rows, gold yes %d / no %d, floor %.2f%%"
          % (set_name, len(rows), n_yes, len(rows) - n_yes, 100.0 * max(n_yes, len(rows) - n_yes) / len(rows)))
    if set_name == "eval458":
        # Amendment 2 A2.4 (freeze §8): 458 rows, 221 yes / 237 no, overlap 0 with DEV200 and RDEV300
        if (len(rows), n_yes, len(rows) - n_yes) != (EVAL458_N, EVAL458_YES, EVAL458_NO):
            die("eval458: %d rows, %d yes / %d no; expected %d, %d / %d (A2.4)"
                % (len(rows), n_yes, len(rows) - n_yes, EVAL458_N, EVAL458_YES, EVAL458_NO))
        mine = {r["qid"] for r in rows}
        mine_ur = {r["question_ur"] for r in rows}
        for other in ("dev200", "rdev300"):
            o_items = load_items(other, with_gold=False)
            ov = sorted(mine & {it["qid"] for it in o_items})
            ov_ur = sorted(mine_ur & {it["question_ur"] for it in o_items})
            if ov or ov_ur:
                die("eval458 overlaps %s: %d qid(s) %s, %d question_ur string(s) (A2.4: overlap must be 0)"
                    % (other, len(ov), ov[:5], len(ov_ur)))
            print("[ok] eval458 overlap with %s (%d rows): 0 qids, 0 question_ur strings" % (other, len(o_items)))
    return rows[:limit] if limit else rows


def needles_for(rows, set_name):
    """Leak needles in the d8_reachable_ceiling shape (question_en, facts,
    evidence_ids, evidence_re) plus required titles (A1.1). For DEV200 the
    three d8 lists must equal d8.load_leak_needles() exactly."""
    q_en = sorted({r["question_en"] for r in rows})
    facts = sorted({f for r in rows for f in r["facts"]})
    ev_ids = sorted({p for r in rows for p in r["evidence_ids"] if isinstance(p, str) and p})
    ev_re = re.compile(r"(?<![0-9A-Za-z])(?:%s)(?![0-9])" % "|".join(re.escape(e) for e in ev_ids)) if ev_ids else re.compile(r"(?!x)x")
    titles = sorted({t for r in rows for t in r["required"] if t})
    title_re = re.compile(r"(?<![0-9A-Za-z])(?:%s)(?![0-9A-Za-z])" % "|".join(re.escape(t) for t in titles)) if titles else re.compile(r"(?!x)x")
    if set_name == "dev200":
        from d8_reachable_ceiling import load_leak_needles
        ref = load_leak_needles()
        if (ref["question_en"], ref["facts"], ref["evidence_ids"]) != (q_en, facts, ev_ids):
            die("DEV200 leak needles differ from d8_reachable_ceiling.load_leak_needles()")
        print("[ok] DEV200 leak needles equal d8_reachable_ceiling.load_leak_needles() (%d question_en, %d facts, %d evidence ids)"
              % (len(q_en), len(facts), len(ev_ids)))
    return {"question_en": q_en, "facts": facts, "evidence_ids": ev_ids, "evidence_re": ev_re,
            "titles": titles, "title_re": title_re}


def load_readings(reader_key, set_name, rows):
    """freeze §2 / §5 V4 / §8: the sealed D11-R reading file of one reader for
    one set. Verifies the file MD5 against the reader's R_SEAL.json entry and
    (DEV200) against the MD5 pinned in the freeze; one record per qid in item
    order, every record ok with a non-empty reading. Returns (OrderedDict qid ->
    reading, info)."""
    spec = READERS[reader_key]
    rdir = os.path.join(D11R_OUT, spec["reader"])
    path = os.path.join(rdir, "readings_%s.jsonl" % set_name)
    seal_path = os.path.join(rdir, "R_SEAL.json")
    for p in (path, seal_path):
        if not os.path.exists(p):
            die("reader file missing: %s" % p)
    seal = json.load(open(seal_path, encoding="utf-8"), object_pairs_hook=OrderedDict)
    if set_name not in seal.get("sets_complete", []):
        die("%s: set %s is not in R_SEAL sets_complete %s" % (spec["reader"], set_name, seal.get("sets_complete")))
    entry = seal["files"].get(path)
    if entry is None:
        die("%s: R_SEAL has no entry for %s" % (spec["reader"], path))
    got = md5_file(path)
    if got != entry["md5"]:
        die("%s: MD5 %s != R_SEAL %s" % (path, got, entry["md5"]))
    if set_name == "dev200" and got != spec["dev200_md5"]:
        die("%s: MD5 %s != freeze §2 pinned %s" % (path, got, spec["dev200_md5"]))
    recs = read_jsonl(path)
    qids = [r["qid"] for r in rows]
    if len(recs) != len(rows) or entry.get("lines") != len(rows):
        die("%s: %d records / %s sealed lines for %d items (one line per qid required)" % (path, len(recs), entry.get("lines"), len(rows)))
    if [r["qid"] for r in recs] != qids:
        die("%s: qids are not the item set in order" % path)
    readings = OrderedDict()
    v4_flag, exact = [], []
    by_q = {r["qid"]: r for r in rows}
    for r in recs:
        if not r.get("ok") or not (r.get("reading") or "").strip():
            die("%s: qid %s has no ok, non-empty reading" % (path, r["qid"]))
        rd = r["reading"]
        readings[r["qid"]] = rd
        if YESNO_EN_RE.search(rd) or HAAN in rd or NAHIN in rd:
            v4_flag.append(r["qid"])
        if rd.strip() == by_q[r["qid"]]["question_en"].strip():
            exact.append(r["qid"])
    info = OrderedDict([("key", reader_key), ("reader", spec["reader"]), ("role", spec["role"]),
                        ("file", path), ("md5", got), ("lines", len(recs)),
                        ("r_seal", OrderedDict([("path", seal_path), ("md5", md5_file(seal_path)), ("entry", entry)])),
                        ("pinned_in_freeze", spec["dev200_md5"] if set_name == "dev200" else None),
                        ("V4_reading_contains_yes_no", OrderedDict([("n", len(v4_flag)), ("qids", v4_flag)])),
                        ("readings_equal_gold_question_en", OrderedDict([("n", len(exact)), ("qids", exact)]))])
    print("[V4] %s %s: %s md5 %s = R_SEAL%s; %d/%d ok readings; contain yes/no/%s/%s: %d; equal to gold question_en: %d"
          % (reader_key, spec["reader"], set_name, got, " = freeze §2" if set_name == "dev200" else "",
             len(recs), len(rows), HAAN, NAHIN, len(v4_flag), len(exact)))
    return readings, info


def guard_out_path(out_root, path):
    ap = os.path.abspath(path)
    root = os.path.abspath(out_root)
    for forb in FORBIDDEN_WRITE_DIRS:
        fr = os.path.abspath(forb)
        if ap == fr or ap.startswith(fr + os.sep):
            die("refusing to write under %s: %s" % (forb, path))
    if not ap.startswith(root + os.sep):
        die("refusing to write outside %s: %s" % (out_root, path))
    return ap


def file_entry(path):
    with open(path, "rb") as f:
        b = f.read()
    import hashlib
    return OrderedDict([("md5", hashlib.md5(b).hexdigest()), ("bytes", len(b)), ("lines", b.count(b"\n"))])
