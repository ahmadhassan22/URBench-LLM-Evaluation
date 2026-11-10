#!/usr/bin/env python3
"""
d11_eval458_provenance.py — D11 RTG Amendment 6 note N2: the provenance listing of the eval458 run.

Contract: docs/EFBPT_D11_RTG_AMENDMENT_6.md A6.3 / A6.4 and docs/EFBPT_D11_RTG_AMENDMENT_6_NOTE.md N2.
The A5.2 readout script (d11_eval458_readout.py) stays byte-identical; this separate script lists what
A6.3 / A6.4 ask the readout to list:
  1 contract MD5s (freeze, amendments, notes) on disk and as recorded by every arm at its start
  2 the scripts on disk (the readout script MD5 is checked against the one named in N2)
  3 every A2.3 regression record (the one of script a733caed and the one of the A6 guard change)
  4 per student arm: start / seal, script MD5s, the regression record it ran under, resumes
  5 teacher parse counts (T_SEAL and a recount from t_facts.jsonl)
  6 empty-facts rows per arm
  7 A6.3 non-specific evidence-id hits (prompt row, needle, owner rows, fact sentence, needle in the reading)
  8 V6 hits (gold fact strings contained in a facts prompt; reported)

It computes NO statistic on answers. CPU only, no model, no gold labels loaded. The student generation
files (d11_arm<ARM>.jsonl) are never parsed: they are read as bytes only, to compare md5 / bytes / lines
with their seal. Everything listed comes from S_START / S_SEAL / S_RESUME, T_SEAL / t_facts.jsonl
(teacher facts, no answers), REGRESS.json and the contract files. Edits no frozen file.

  python eval/error_analysis_tests/efbpt/d11_eval458_provenance.py --partial
        list what exists now (unsealed or missing arms are marked); print only, write nothing.
  python eval/error_analysis_tests/efbpt/d11_eval458_provenance.py
        the listing of the run. Refuses unless A_r2_eval, TR and TR_wrong_eval are sealed and no check
        fails. Writes docs/handoff/D11_EVAL458_PROVENANCE.md (TR_L1_eval sealed too) or
        docs/handoff/D11_EVAL458_PROVENANCE_PRIMARY.md (before TR_L1_eval is sealed). An existing file
        is never overwritten with different content.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from d11_common import md5_file, die, file_entry, read_jsonl, OUT_ROOT, REGRESS_ROOT, READERS  # noqa: E402  (chdir to the repo root)

ARMS = ("A_r2_eval", "TR", "TR_wrong_eval", "TR_L1_eval")
PRIMARY_ARMS = ("A_r2_eval", "TR", "TR_wrong_eval")
READER_OF = OrderedDict([("TR", "R1"), ("TR_wrong_eval", "R1"), ("TR_L1_eval", "R3")])   # A2.2
S_DIR = os.path.join(OUT_ROOT, "student", "eval458")
T_DIR = os.path.join(OUT_ROOT, "teacher", "eval458")
DERANGEMENT = os.path.join(S_DIR, "derangement_eval458.json")
REGRESS_GLOB = os.path.join(REGRESS_ROOT, "student", "dev200*", "REGRESS.json")
D = "eval/error_analysis_tests/efbpt/"
STUDENT, COMMON, READOUT = D + "d11_student.py", D + "d11_common.py", D + "d11_eval458_readout.py"
READOUT_MD5 = "8fb00396fc676a1066d924dfdb3ed874"          # note N2: the A5.2 readout script stays byte-identical
SCRIPTS = (STUDENT, COMMON, D + "d11o_student.py", D + "d11_teacher.py", READOUT, D + "d11_scorer_audit.py")
# (document, path, key in the `contracts` of S_START / REGRESS.json or None)
DOCS = (
    ("D11 freeze", "docs/EFBPT_D11_RTG_FREEZE.md", "d11_freeze"),
    ("Amendment 1", "docs/EFBPT_D11_RTG_AMENDMENT_1.md", "d11_amendment_1"),
    ("Amendment 2", "docs/EFBPT_D11_RTG_AMENDMENT_2.md", "d11_amendment_2"),
    ("Amendment 3", "docs/EFBPT_D11_RTG_AMENDMENT_3.md", None),
    ("Amendment 4", "docs/EFBPT_D11_RTG_AMENDMENT_4.md", None),
    ("Amendment 4 note", "docs/EFBPT_D11_RTG_AMENDMENT_4_NOTE.md", None),
    ("Amendment 5", "docs/EFBPT_D11_RTG_AMENDMENT_5.md", None),
    ("Amendment 6", "docs/EFBPT_D11_RTG_AMENDMENT_6.md", "d11_amendment_6"),
    ("Amendment 6 note", "docs/EFBPT_D11_RTG_AMENDMENT_6_NOTE.md", None),
    ("GO eval458", "docs/D11_GO_EVAL458.md", None),
    ("D11-R reader freeze", "docs/EFBPT_D11R_READER_FREEZE.md", "d11r_freeze"),
    ("D11-R Amendment 1", "docs/EFBPT_D11R_AMENDMENT_1.md", "d11r_amendment_1"),
    ("D11-R Amendment 2", "docs/EFBPT_D11R_AMENDMENT_2.md", "d11r_amendment_2"),
    ("D10 teacher freeze", "docs/EFBPT_D10_TEACHER_FREEZE.md", "d10_freeze"),
)
OUT_FULL = "docs/handoff/D11_EVAL458_PROVENANCE.md"
OUT_PRIMARY = "docs/handoff/D11_EVAL458_PROVENANCE_PRIMARY.md"


def jload(path):
    return json.load(open(path, encoding="utf-8"), object_pairs_hook=OrderedDict)


def git(*a):
    try:
        return subprocess.check_output(["git"] + list(a), stderr=subprocess.DEVNULL)
    except Exception:
        return None


def last_commit(path):
    out = git("log", "-1", "--format=%H", "--", path)
    h = out.decode().strip()[:7] if out else ""
    st = git("status", "--porcelain", "--", path)
    if not h:
        return "not committed"
    return h + (" + uncommitted change" if st and st.strip() else "")


def blob_md5(head, path):
    b = git("show", "%s:%s" % (head, path)) if head else None
    return hashlib.md5(b).hexdigest() if b is not None else None


def seal_mismatches(seal):
    """md5 / bytes / lines of every sealed file against the file on disk (bytes only; nothing is parsed)."""
    bad = []
    for p, want in seal["files"].items():
        if not os.path.exists(p):
            bad.append("%s missing" % p)
            continue
        got = file_entry(p)
        if (got["md5"], got["bytes"], got["lines"]) != (want["md5"], want["bytes"], want["lines"]):
            bad.append("%s differs from its seal" % p)
    return bad


def code(s):
    return "`%s`" % str(s).replace("`", "'").replace("|", "\\|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="list what exists now; print only, write nothing")
    args = ap.parse_args()
    problems = []
    L = []
    w = L.append

    # ---- what exists -------------------------------------------------------------------------
    starts, seals, status = OrderedDict(), OrderedDict(), OrderedDict()
    for arm in ARMS:
        sp, zp = os.path.join(S_DIR, "S_START_%s.json" % arm), os.path.join(S_DIR, "S_SEAL_%s.json" % arm)
        if os.path.exists(sp):
            starts[arm] = jload(sp)
        if os.path.exists(zp):
            if arm not in starts:
                die("%s exists without %s" % (zp, sp))
            seals[arm] = jload(zp)
        status[arm] = "sealed" if arm in seals else ("started, not sealed" if arm in starts else "not started")
    if not args.partial:
        missing = [a for a in PRIMARY_ARMS if status[a] != "sealed"]
        if missing:
            die("not sealed: %s. The listing of the run needs A_r2_eval, TR and TR_wrong_eval sealed (use --partial to list what exists now)" % ", ".join(missing))
    full = status["TR_L1_eval"] == "sealed"
    regress = OrderedDict((p, jload(p)) for p in sorted(glob.glob(REGRESS_GLOB)))
    reg_md5 = OrderedDict((p, md5_file(p)) for p in regress)

    w("# D11 eval458 — provenance listing (Amendment 6 A6.3 / A6.4; note N2)")
    w("")
    w("Written by `%s` (MD5 `%s`). It computes no statistic on answers: no gold label is loaded and the student generation files are never parsed "
      "(bytes only, to compare md5 / bytes / lines with their seal). Sources: S_START / S_SEAL / S_RESUME of every arm, T_SEAL and t_facts.jsonl of the teacher, "
      "the A2.3 regression records, the contract files." % (D + "d11_eval458_provenance.py", md5_file(os.path.abspath(__file__))))
    w("")
    if args.partial:
        w("**PARTIAL listing (--partial): state of the files now; nothing is written.**")
    else:
        w("Arms listed: %s." % (", ".join(a for a in ARMS if status[a] == "sealed") + ("" if full else " (TR_L1_eval not sealed yet: PRIMARY listing)")))
    w("")
    w("| arm | state |")
    w("|---|---|")
    for arm in ARMS:
        w("| %s | %s |" % (arm, status[arm]))
    w("")

    # ---- 1 contracts -------------------------------------------------------------------------
    w("## 1. Contract MD5s")
    w("")
    w("| document | file | MD5 on disk | last commit of the file | recorded at the start of |")
    w("|---|---|---|---|---|")
    for name, path, key in DOCS:
        if not os.path.exists(path):
            problems.append("contract file missing: %s" % path)
            w("| %s | `%s` | MISSING | — | — |" % (name, path))
            continue
        got = md5_file(path)
        rec = []
        for arm, st in starts.items():
            c = (st.get("contracts") or {}).get(key) if key else None
            if c is None:
                continue
            if c["md5"] == got:
                rec.append("%s =" % arm)
            else:
                rec.append("%s ≠ (`%s`)" % (arm, c["md5"]))
                problems.append("%s: %s recorded MD5 %s at its start, the file on disk is %s" % (path, arm, c["md5"], got))
        w("| %s | `%s` | `%s` | %s | %s |" % (name, path, got, last_commit(path), "; ".join(rec) if rec else "—"))
    w("")
    w("\"=\": the MD5 in the arm's S_START equals the file on disk. An arm that started before Amendment 6 (script a733caed) does not record it (A6.4).")
    w("")

    # ---- 2 scripts on disk -------------------------------------------------------------------
    w("## 2. Scripts on disk")
    w("")
    w("| script | MD5 on disk | last commit of the file |")
    w("|---|---|---|")
    for p in SCRIPTS:
        w("| `%s` | `%s` | %s |" % (p, md5_file(p), last_commit(p)))
    w("")
    if md5_file(READOUT) != READOUT_MD5:
        problems.append("%s MD5 %s != %s (note N2: the readout script stays byte-identical)" % (READOUT, md5_file(READOUT), READOUT_MD5))
    w("Readout script (A5.2) MD5 = `%s` named in the Amendment 6 note N2: **%s**." % (READOUT_MD5, "yes" if md5_file(READOUT) == READOUT_MD5 else "NO"))
    w("")

    # ---- 3 regression records ----------------------------------------------------------------
    w("## 3. A2.3 regression records (DEV200, arm TR, rows 0-4, byte compare with the sealed file)")
    w("")
    w("| record | MD5 of the record | verdict | identical rows | SLURM job | node | end | git head | `d11_student.py` | `d11_common.py` | `d11o_student.py` | Amendment 6 in its contracts | sealed DEV200 TR file |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, r in regress.items():
        sc = r.get("scripts") or {}
        if r.get("verdict") != "PASS":
            problems.append("regression record %s: verdict %r" % (p, r.get("verdict")))
        w("| `%s` | `%s` | %s | %s / %s | %s | %s | %s | `%s` | `%s` | `%s` | `%s` | %s | `%s` |"
          % (p, reg_md5[p], r.get("verdict"), r.get("n_identical"), r.get("n_rows"), r.get("slurm_job_id"), r.get("hostname"), r.get("end_time"),
             (r.get("git") or {}).get("head", "?")[:7], sc.get(STUDENT), sc.get(COMMON), sc.get(D + "d11o_student.py"),
             "yes" if "d11_amendment_6" in (r.get("contracts") or {}) else "no", (r.get("sealed_file") or {}).get("md5")))
    if not regress:
        problems.append("no regression record under %s" % REGRESS_GLOB)
        w("| (none found) | | | | | | | | | | | | |")
    w("")

    # ---- 4 student arms ----------------------------------------------------------------------
    w("## 4. Student arms: start, scripts, regression record, seal")
    w("")
    w("| arm | state | SLURM job (start) | node | start | git head at start (dirty) | `d11_student.py` | = that file at the git head | `d11_common.py` | regression record it ran under | same scripts as that record |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        if arm not in starts:
            w("| %s | not started | — | — | — | — | — | — | — | — | — |" % arm)
            continue
        st = starts[arm]
        sc = st.get("scripts") or {}
        head = (st.get("git") or {}).get("head")
        at_head = blob_md5(head, STUDENT)
        same_blob = "yes" if at_head == sc.get(STUDENT) else "NO (`%s`)" % at_head
        if at_head != sc.get(STUDENT):
            problems.append("%s: d11_student.py MD5 %s in S_START, %s at its git head %s" % (arm, sc.get(STUDENT), at_head, head))
        reg = st.get("regression") or {}
        match = [p for p in regress if reg_md5[p] == reg.get("md5")]
        if match:
            r = regress[match[0]]
            reg_txt = "job %s, `%s`" % (r.get("slurm_job_id"), match[0])
            same = dict(r.get("scripts") or {}) == dict(sc) and r.get("verdict") == "PASS"
            if not same:
                problems.append("%s: its regression record %s is not a PASS with the same scripts" % (arm, match[0]))
        else:
            reg_txt, same = "NOT FOUND (record MD5 `%s`, job %s)" % (reg.get("md5"), reg.get("slurm_job_id")), False
            problems.append("%s: no regression record on disk has the MD5 %s recorded in its S_START" % (arm, reg.get("md5")))
        w("| %s | %s | %s | %s | %s | %s (%s) | `%s` | %s | `%s` | %s | %s |"
          % (arm, status[arm], st.get("slurm_job_id"), st.get("hostname"), st.get("start_time"), (head or "?")[:7], (st.get("git") or {}).get("dirty"),
             sc.get(STUDENT), same_blob, sc.get(COMMON), reg_txt, "yes (PASS)" if same else "NO"))
    w("")
    by_script = OrderedDict()
    for arm, st in starts.items():
        by_script.setdefault((st.get("scripts") or {}).get(STUDENT), []).append(arm)
    w("`d11_student.py` versions used (A6.4): " + "; ".join("`%s` — %s" % (m, ", ".join(a)) for m, a in by_script.items()) + ".")
    w("")
    w("| arm | seal | SLURM job (seal) | end | rows | `script_md5` in the seal | sealed files = disk (md5, bytes, lines) | `scored` | resumes (time, job, rows completed before) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        if arm not in starts:
            w("| %s | — | — | — | — | — | — | — | — |" % arm)
            continue
        rp = os.path.join(S_DIR, "S_RESUME_%s.jsonl" % arm)
        res = read_jsonl(rp) if os.path.exists(rp) else []
        res_txt = "; ".join("%s, job %s, %s rows" % (e.get("time"), e.get("slurm_job_id"), e.get("rows_completed")) for e in res) if res else "none"
        if arm not in seals:
            ap_ = os.path.join(S_DIR, "d11_arm%s.jsonl" % arm)
            n_now = file_entry(ap_)["lines"] if os.path.exists(ap_) else 0
            w("| %s | not sealed | — | — | %d of %s on disk now | — | — | — | %s |" % (arm, n_now, starts[arm].get("n_rows"), res_txt))
            continue
        z, st = seals[arm], starts[arm]
        bad = seal_mismatches(z)
        problems.extend("%s seal: %s" % (arm, b) for b in bad)
        if dict(z.get("scripts") or {}) != dict(st.get("scripts") or {}):
            problems.append("%s: the scripts in S_SEAL differ from S_START" % arm)
        if z.get("n_rows") != st.get("n_rows") or z.get("complete") is not True:
            problems.append("%s: seal n_rows %s / complete %s, S_START n_rows %s" % (arm, z.get("n_rows"), z.get("complete"), st.get("n_rows")))
        w("| %s | `%s` | %s | %s | %s | `%s` | %s | %s | %s |"
          % (arm, os.path.join(S_DIR, "S_SEAL_%s.json" % arm), z.get("slurm_job_id"), z.get("end_time"), z.get("n_rows"), z.get("script_md5"),
             "yes (%d files)" % len(z["files"]) if not bad else "NO", z.get("scored"), res_txt))
    w("")

    # ---- 5 teacher ---------------------------------------------------------------------------
    w("## 5. Teacher parse counts")
    w("")
    w("| reader | used by | SLURM job | end | teacher script MD5 | rows | parse_ok (T_SEAL / recount) | repaired rows (T_SEAL / recount) | facts removed by the verdict filter T2 (T_SEAL / recount) | rows with no facts | sealed files = disk | reading file MD5 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    no_facts = OrderedDict()                          # reader -> [(row, qid, parse_ok, n_facts_parsed, n_removed)]
    t_seals = OrderedDict()
    for key in ("R1", "R3"):
        used = ", ".join(a for a, k in READER_OF.items() if k == key)
        zp = os.path.join(T_DIR, key, "T_SEAL.json")
        if not os.path.exists(zp):
            w("| %s (%s) | %s | not sealed | — | — | — | — | — | — | — | — | — |" % (key, READERS[key]["reader"], used))
            if any(a in starts for a, k in READER_OF.items() if k == key):
                problems.append("teacher %s: an arm using it has started but %s is missing" % (key, zp))
            continue
        z = jload(zp)
        t_seals[key] = z
        bad = seal_mismatches(z)
        problems.extend("teacher %s seal: %s" % (key, b) for b in bad)
        recs = read_jsonl(os.path.join(T_DIR, key, "t_facts.jsonl"))
        n_ok, n_rep, n_rem = sum(1 for r in recs if r.get("parse_ok")), sum(1 for r in recs if r.get("repaired")), sum(r.get("n_removed") or 0 for r in recs)
        no_facts[key] = [(i, r["qid"], r.get("parse_ok"), r.get("n_facts_parsed"), r.get("n_removed")) for i, r in enumerate(recs) if not r.get("facts")]
        if (len(recs), n_ok, n_rep, n_rem) != (z.get("n_rows"), z.get("t1_parse_ok"), z.get("t1_repaired_rows"), z.get("t2_removed_total")):
            problems.append("teacher %s: recount (rows %d, parse_ok %d, repaired %d, T2 removed %d) differs from T_SEAL" % (key, len(recs), n_ok, n_rep, n_rem))
        for arm, k in READER_OF.items():
            if k == key and arm in starts:
                ts = ((starts[arm].get("teacher_seals") or {}).get(key) or {}).get("files") or {}
                if dict(ts) != {p: e["md5"] for p, e in z["files"].items()}:
                    problems.append("%s: the teacher %s files recorded in its S_START are not the sealed teacher files" % (arm, key))
        w("| %s (%s) | %s | %s | %s | `%s` | %s | %s / %d | %s / %d | %s / %d | %d | %s | `%s` |"
          % (key, z.get("reader"), used, z.get("slurm_job_id"), z.get("end_time"), z.get("script_md5"), z.get("n_rows"), z.get("t1_parse_ok"), n_ok,
             z.get("t1_repaired_rows"), n_rep, z.get("t2_removed_total"), n_rem, len(no_facts[key]), "yes (%d files)" % len(z["files"]) if not bad else "NO",
             (z.get("reading_file") or {}).get("md5")))
    w("")
    for key, rows_ in no_facts.items():
        for i, qid, ok, n_parsed, n_removed in rows_:
            w("- %s: row %d (qid %s) has no facts — parse_ok %s, facts parsed %s, removed by T2 %s." % (key, i, qid, ok, n_parsed, n_removed))
    if any(no_facts.values()):
        w("")

    # ---- 6 empty-facts rows per arm ----------------------------------------------------------
    w("## 6. Empty-facts rows per arm")
    w("")
    w("| arm | layouts (S_START) | empty-facts prompts (S_START) | prompt rows without facts |")
    w("|---|---|---|---|")
    dmap = jload(DERANGEMENT)["map"] if os.path.exists(DERANGEMENT) else None
    for arm in ARMS:
        if arm not in starts:
            w("| %s | not started | — | — |" % arm)
            continue
        st = starts[arm]
        lay = (st.get("layouts") or {}).get(arm) or {}
        counts = ", ".join("%s %s" % (k, v) for k, v in sorted((lay.get("counts") or {}).items()))
        key = READER_OF.get(arm)
        if key is None:
            rows_txt = "all rows (no-facts arm, layout A)"
        elif key not in no_facts:
            rows_txt = "teacher %s not sealed" % key
        else:
            qids = st["qids"]
            empty_q = {q for _, q, _, _, _ in no_facts[key]}
            if arm == "TR_wrong_eval":
                if dmap is None:
                    rows_ = None
                    rows_txt = "%s not written yet" % DERANGEMENT
                else:
                    rows_ = [(i, q, dmap[q]) for i, q in enumerate(qids) if dmap[q] in empty_q]
                    rows_txt = "; ".join("row %d (qid %s) <- facts of row %d (qid %s)" % (i, q, qids.index(d), d) for i, q, d in rows_) or "none"
            else:
                rows_ = [(i, q) for i, q in enumerate(qids) if q in empty_q]
                rows_txt = "; ".join("row %d (qid %s)" % (i, q) for i, q in rows_) or "none"
            if rows_ is not None and len(rows_) != lay.get("empty_facts"):
                problems.append("%s: %d prompt rows without teacher facts, S_START says %s" % (arm, len(rows_), lay.get("empty_facts")))
        w("| %s | %s | %s | %s |" % (arm, counts, lay.get("empty_facts"), rows_txt))
    w("")

    # ---- 7 A6.3 ------------------------------------------------------------------------------
    w("## 7. A6.3 non-specific evidence-id hits (logged, not fatal)")
    w("")
    w("| arm | prompt row (qid) | facts of row (qid) | needle (letters) | rows owning the evidence id | needle in the reading | fact sentence |")
    w("|---|---|---|---|---|---|---|")
    n_hits = 0
    for arm in READER_OF:
        if arm not in starts:
            w("| %s | not started | — | — | — | — | — |" % arm)
            continue
        g = (starts[arm].get("leak_guard") or {}).get(arm) or {}
        if "A6_3_nonspecific_evidence_hits" not in g:
            problems.append("%s: its S_START has no A6_3_nonspecific_evidence_hits entry (a facts arm started without the A6 guard)" % arm)
            w("| %s | NOT RECORDED in S_START | — | — | — | — | — |" % arm)
            continue
        hits = g["A6_3_nonspecific_evidence_hits"]
        if not hits:
            w("| %s | none | — | — | — | — | — |" % arm)
        for h in hits:
            n_hits += 1
            w("| %s | %d (%s) | %d (%s) | %s (%d) | %s | %s | %s |"
              % (arm, h["prompt_row"], h["qid"], h["facts_source_row"], h["facts_source_qid"], code(h["needle"]), h["needle_letters"],
                 ", ".join("%d (%s)" % (o["row"], o["qid"]) for o in h["owner_rows"]), "yes" if h["needle_in_reading"] else "no", code(h["fact"])))
    w("")
    w("A_r2_eval has no facts; the rule does not apply to it. Evidence ids with 3 or more letters, question_en and the readings stay fatal: an arm with such a hit does not start.")
    w("")

    # ---- 8 V6 --------------------------------------------------------------------------------
    w("## 8. V6 hits: gold fact strings contained in a facts prompt (reported, not fatal)")
    w("")
    w("| arm | hits (S_START) | prompt row (qid) | gold fact string |")
    w("|---|---|---|---|")
    for arm in ARMS:
        if arm not in starts:
            w("| %s | not started | — | — |" % arm)
            continue
        st = starts[arm]
        g = (st.get("leak_guard") or {}).get(arm) or {}
        ms = g.get("matches") or []
        if not ms:
            w("| %s | %s%s | — | — |" % (arm, g.get("count"), " (gold facts are fatal for this arm: no facts arm)" if arm not in READER_OF else ""))
        for m in ms:
            w("| %s | %s in %s prompt(s) | %s (%s) | %s |" % (arm, g.get("count"), g.get("qids"), st["qids"].index(m["qid"]) if m["qid"] in st["qids"] else "?", m["qid"], code(m["fact"])))
    w("")

    # ---- checks ------------------------------------------------------------------------------
    w("## Checks")
    w("")
    if problems:
        for p in problems:
            w("- FAILED: %s" % p)
    else:
        w("All checks of this listing hold: seals equal the files on disk, every started arm ran under a PASSED regression record with the same scripts, "
          "its `d11_student.py` is the file at its git head, the teacher recounts equal the T_SEAL counts, the recorded contract MD5s equal the files on disk.")
    w("")
    text = "\n".join(L)
    print(text)
    if args.partial:
        print("[partial] nothing written; %d failed check(s)" % len(problems))
        return
    if problems:
        die("%d failed check(s); nothing written" % len(problems))
    out = OUT_FULL if full else OUT_PRIMARY
    if os.path.exists(out):
        if open(out, encoding="utf-8").read() != text:
            die("%s exists with different content; not overwritten" % out)
        print("[unchanged] %s" % out)
    else:
        with open(out, "x", encoding="utf-8") as f:
            f.write(text)
        print("[written] %s" % out)


if __name__ == "__main__":
    main()
