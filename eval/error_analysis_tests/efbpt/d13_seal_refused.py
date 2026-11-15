#!/usr/bin/env python3
"""
d13_seal_refused.py — D13 Amendment 3: seal API_tr_R1 on eval458 with the two rows
that the provider's input inspection refuses.

Contract: docs/EFBPT_D13_AMENDMENT_3.md (MD5 pinned below), freeze §7.2 and §7.6.
  A3.1  the two refused qids (named below).
  A3.2  a row refused by the provider's input inspection on two separate attempts is
        recorded as refused, with the error text, and is not requested again. The arm
        is sealed with 456 ok records and 2 refused records (458 rows present).
  A3.4  the seal is written by this new script after all GPU jobs are sealed. No
        existing d13 script is edited: d13_common is imported and used as it is.

What it does. It reads the work file of the arm (the ok and error records that
d13_api.py wrote), checks it, writes the row file d13_API_tr_R1.jsonl in item order
(456 ok records exactly as stored, 2 refused records) and the seal SEAL_API_tr_R1.json
through d13_common.seal_arm. After the seal, d13_api.py stops at "[sealed]" for this
arm, so no row is requested again.

It makes no API call and never opens the key file. It scores nothing: no extractor is
imported, the rows carry no pred / correct / gold field, the seal has scored: false.

  --check   every check, nothing written (works before the script is committed)
  (no flag) needs the d13 scripts, this one included, committed and unchanged
"""
import argparse
import json
import os
import sys
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d13_common as C  # noqa: E402  (chdir to the repo root)
from d13_common import die, md5_file, read_jsonl  # noqa: E402

AMEND3_PATH = "docs/EFBPT_D13_AMENDMENT_3.md"
AMEND3_MD5 = "344590e857e9f3ac075519120892e5d6"
ARM, SET, MODE = "API_tr_R1", "eval458", "FULL"
REFUSED_QIDS = ("90538ec7aa1100a34829", "78f2f99d04b9acd8bada")      # A3.1
N_ROWS, N_OK, N_REFUSED = 458, 456, 2                                # A3.2
REFUSAL_STATUS, REFUSAL_TYPE = 400, "data_inspection_failed"         # A3.1
MIN_ATTEMPTS = 2                                                     # A3.2: two separate attempts


def all_other_arms_sealed():
    """A3.4: every other arm file of the freeze (18) is sealed before this seal is written."""
    out = []
    for arm, spec in C.ARMS.items():
        for s in spec["sets"]:
            if (arm, s) == (ARM, SET):
                continue
            pth = C.arm_paths(arm, s, "FULL")
            if not os.path.exists(pth["seal"]):
                die("A3.4: %s on %s is not sealed (%s missing)" % (arm, s, pth["seal"]))
            seal = json.load(open(pth["seal"], encoding="utf-8"))
            if seal.get("complete") is not True or seal.get("scored") is not False or seal.get("arm") != arm or seal.get("set") != s:
                die("A3.4: %s is not a complete, unscored seal of %s on %s" % (pth["seal"], arm, s))
            ent = seal["files"].get(pth["rows"])
            if ent is None or C.file_entry(pth["rows"])["md5"] != ent["md5"]:
                die("A3.4: the row file of %s on %s does not match its seal" % (arm, s))
            out.append(arm + "/" + s)
    return out


def run_of(ts, run_starts):
    """Index of the d13_api.py run a record belongs to (the last run that started at or before ts)."""
    k = -1
    for i, st in enumerate(run_starts):
        if st <= ts:
            k = i
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="every check, nothing written")
    args = ap.parse_args()
    print("=" * 78)
    print("D13 Amendment 3: seal %s on %s with %d refused rows%s" % (ARM, SET, N_REFUSED, "   --check: NOTHING IS WRITTEN" if args.check else ""))
    print("=" * 78, flush=True)
    contracts = C.verify_contracts()
    if md5_file(AMEND3_PATH) != AMEND3_MD5:
        die("%s MD5 %s != pinned %s" % (AMEND3_PATH, md5_file(AMEND3_PATH), AMEND3_MD5))
    print("[md5] %-18s %s  %s" % ("d13_amendment_3", AMEND3_MD5, AMEND3_PATH))
    if not args.check:
        C.require_committed()

    pth = C.arm_paths(ARM, SET, MODE)
    work = os.path.join(pth["dir"], "d13_%s.work.jsonl" % ARM)
    if os.path.exists(pth["seal"]):
        seal = json.load(open(pth["seal"], encoding="utf-8"))
        if C.file_entry(pth["rows"])["md5"] != seal["files"][pth["rows"]]["md5"]:
            die("%s is sealed but its row file does not match the seal" % ARM)
        print("[sealed] %s %s is already sealed (%d rows; seal MD5 %s); a sealed file is never rewritten" % (ARM, SET, seal["n_rows"], md5_file(pth["seal"])))
        return
    if os.path.exists(pth["rows"]):
        die("%s exists without a seal; not overwritten" % pth["rows"])
    for p in (work, pth["start"]):
        if not os.path.exists(p):
            die("missing: " + p)

    sealed = all_other_arms_sealed()
    print("[ok] A3.4: the other %d arm files are sealed, complete and not scored; row files match their seals" % len(sealed))

    # the messages of the arm, rebuilt and guarded exactly as d13_api.py builds them (no key, no call)
    ctx = C.make_ctx(SET, MODE)
    msgs, _info = C.build_api_messages(ARM, ctx)
    if len(msgs) != N_ROWS:
        die("%d messages, expected %d" % (len(msgs), N_ROWS))
    by_qid = OrderedDict((m["qid"], m) for m in msgs)

    start = json.load(open(pth["start"], encoding="utf-8"), object_pairs_hook=OrderedDict)
    if (start.get("arm"), start.get("set"), start.get("mode"), start.get("n_rows")) != (ARM, SET, MODE, N_ROWS) or start.get("qids") != list(by_qid):
        die("%s is not the START file of %s %s %s with the %d qids in order" % (pth["start"], ARM, SET, MODE, N_ROWS))
    run_starts = [r["start_time"] for r in start["runs"]]

    recs = read_jsonl(work)
    ok, err = OrderedDict(), OrderedDict()
    for r in recs:
        m = by_qid.get(r["qid"])
        if m is None or r.get("arm") != ARM or r.get("set") != SET:
            die("%s: a record of qid %s does not belong to %s %s" % (work, r.get("qid"), ARM, SET))
        if r["system"] != m["system"] or r["user"] != m["user"] or r["prompt"] != m["prompt"]:
            die("%s: a record of qid %s does not match the messages of the arm" % (work, r["qid"]))
        if any(k in r for k in ("pred", "correct", "gold")):
            die("%s: a record of qid %s carries a scoring field" % (work, r["qid"]))
        if r.get("ok"):
            ok.setdefault(r["qid"], r)               # the first ok record, as d13_api.py keeps it
        else:
            err.setdefault(r["qid"], []).append(r)
    missing = [q for q in by_qid if q not in ok]
    if len(ok) != N_OK or sorted(missing) != sorted(REFUSED_QIDS):
        die("%d qids have an ok record and %d have none (%s); Amendment 3 names %d ok and the two qids %s"
            % (len(ok), len(missing), ", ".join(missing[:5]), N_OK, ", ".join(REFUSED_QIDS)))
    if any(q in ok for q in err):
        die("a qid has both an ok record and an error record; Amendment 3 does not cover this")

    refused = OrderedDict()
    for q in missing:
        rs = err.get(q, [])
        for r in rs:
            if r.get("http_status") != REFUSAL_STATUS or REFUSAL_TYPE not in (r.get("error") or ""):
                die("qid %s: an error record is not HTTP %d %s (A3.1); it is %r" % (q, REFUSAL_STATUS, REFUSAL_TYPE, (r.get("error") or "")[:120]))
            if r.get("generation") or r.get("raw_response") is not None:
                die("qid %s: a refused record carries an answer" % q)
        runs = sorted({run_of(r["ts"], run_starts) for r in rs})
        if len(rs) < MIN_ATTEMPTS or len(runs) < MIN_ATTEMPTS or -1 in runs:
            die("qid %s: %d refusal records in %d separate runs; A3.2 needs two separate attempts" % (q, len(rs), len(runs)))
        last = rs[-1]
        rec = OrderedDict(last)                      # same fields as every other row; ok false, no answer
        rec["refused"] = True
        rec["refused_rule"] = "D13 Amendment 3 A3.2"
        rec["refused_attempts"] = [OrderedDict([("ts", r["ts"]), ("run", run_of(r["ts"], run_starts) + 1), ("http_status", r["http_status"]),
                                                ("attempts", r["attempts"]), ("error", r["error"])]) for r in rs]
        refused[q] = rec
        print("[refused] qid %s: %d refusals in runs %s (%s); error text: %s"
              % (q, len(rs), ", ".join(str(x + 1) for x in runs), ", ".join(r["ts"] for r in rs), last["error"]))

    rows = [ok[q] if q in ok else refused[q] for q in by_qid]
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_ref = sum(1 for r in rows if r.get("refused"))
    if (len(rows), n_ok, n_ref) != (N_ROWS, N_OK, N_REFUSED) or [r["qid"] for r in rows] != list(by_qid):
        die("row file would hold %d rows, %d ok, %d refused; A3.2 says %d, %d, %d" % (len(rows), n_ok, n_ref, N_ROWS, N_OK, N_REFUSED))
    fin = Counter(str(r["finish_reason"]) for r in rows if r.get("ok"))
    print("[ok] %d rows in item order: %d ok (first ok record of each qid, unchanged), %d refused; finish reasons of the ok rows: %s; work file MD5 %s"
          % (len(rows), n_ok, n_ref, dict(fin), md5_file(work)))
    if args.check:
        print("[check] all checks passed. Nothing was written.")
        return

    tmp = C.guard_out(pth["rows"]) + ".tmp"
    with open(tmp, "wb") as f:
        for r in rows:
            f.write(C.record_line(r))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, pth["rows"])
    me = os.path.relpath(os.path.abspath(__file__), C.ROOT)
    extra = OrderedDict([
        ("model", start["model"]), ("api_host", start["api_host"]),
        ("finish_reasons", dict(Counter(r["finish_reason"] for r in rows if r.get("ok")))),
        ("work_file", OrderedDict([("path", work), ("md5", md5_file(work))])),
        ("n_ok", n_ok), ("n_refused", n_ref), ("refused_qids", [q for q in by_qid if q in refused]),
        ("refused_rule", "D13 Amendment 3 A3.2: refused by the provider's input inspection on two separate attempts; recorded as refused with the error text; not requested again"),
        ("d13_amendment_3", OrderedDict([("path", AMEND3_PATH), ("md5", AMEND3_MD5)])),
        ("sealed_by", OrderedDict([("script", me), ("md5", md5_file(os.path.abspath(__file__)))])),
        ("contracts", contracts),
        ("note", "seconds_total and gen_tokens_total cover the 458 stored records; the 2 refused records hold the latency of the refused call and no tokens"),
    ])
    C.seal_arm(ARM, SET, MODE, msgs, extra)
    print("[done] %s %s: %d rows sealed (%d ok, %d refused), not scored. Seal MD5 %s" % (ARM, SET, len(rows), n_ok, n_ref, md5_file(pth["seal"])))


if __name__ == "__main__":
    main()
