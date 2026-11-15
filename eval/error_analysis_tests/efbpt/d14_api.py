#!/usr/bin/env python3
"""
d14_api.py — D14 commercial-model arms M_ur, M_r1, M_og (freeze F4, 9.9), run on the login node.

qwen3.8-max with the D11 Amendment 1 A1.6 settings (d11_api_direct constants,
imported): enable_thinking false, temperature 0, 2 concurrent requests, the
D11-R retry policy; max_tokens 1024. One user message, no system message:
  M_ur  the UR prompt (the fair script's user message)
  M_r1  the EN prompt with the sealed R1 item block
  M_og  the EN prompt with the original English item block
A row whose reading failed is not requested in M_r1 (status reading_failed). A
row refused by the provider on two separate requests is recorded as refused and
not requested again.

The arms of one task run one after another in this process, in the order given.
Resumable: every request is appended to a work file; ok rows are never requested
again. An arm is sealed only when every row is present. Nothing is scored: no
extractor is imported, rows carry no pred / correct / gold.

KEY HANDLING (9.9): as d14_translate.py.

  --task csqa|piqa|gsm8k  --arms M_ur,M_r1,M_og  --mode TEST|FULL  [--preflight-only]
Exit codes: 0 every arm sealed; 3 rows still open (run again); 4 API budget / quota stop.
"""
import argparse
import os
import sys
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_common as C  # noqa: E402
from d14_common import die  # noqa: E402

MAX_REFUSED = 2      # 9.9 / D13 Amendment 3: refused on two separate requests = refused


def decide(reqs):
    """('ok', record) | ('refused', None) | ('open', None) from the requests of one row."""
    for r in reqs:
        if r["answered"]:
            return "ok", r
    if sum(1 for r in reqs if r["refused"]) >= MAX_REFUSED:
        return "refused", None
    return "open", None


def run_arm(task, arm, mode, rows, r1, r1_info, api, S):
    pth = C.arm_paths(task, arm, mode)
    os.makedirs(C.guard_out(pth["dir"]), exist_ok=True)
    if C.is_sealed(pth):
        print("[sealed] %s %s (%s) is already sealed; no call is made" % (task, arm, mode), flush=True)
        return 0
    users = C.build_users(task, arm, rows, r1)
    by = OrderedDict((u["rid"], u) for u in users)
    tag = "%s/%s/%s" % (arm, task, mode)
    reqs = OrderedDict((u["rid"], []) for u in users)
    for r in C.read_rows(pth["work"], repair=True):
        u = by.get(r["rid"])
        if u is None or u["status"] != "ok" or r["prompt"] != u["user"]:
            die("%s: a request record of %s does not match the message of this run" % (pth["work"], r["rid"]))
        reqs[r["rid"]].append(r)
    todo = [u["rid"] for u in users if u["status"] == "ok" and decide(reqs[u["rid"]])[0] == "open"]
    n_fail = sum(1 for u in users if u["status"] != "ok")
    print("\n[arm] %s %s %s: rows %d  reading_failed %d  decided %d  to do %d"
          % (task, arm, mode, len(users), n_fail, len(users) - n_fail - len(todo), len(todo)), flush=True)
    if users and users[0]["status"] == "ok":
        C.log_full_prompt("%s %s row 0 (the one user message)" % (task, arm), users[0]["user"])
    start = OrderedDict([
        ("experiment", "D14"), ("arm", arm), ("task", task), ("mode", mode), ("model", S["model"]),
        ("request_extra", S["request_extra"]), ("temperature", S["temperature"]), ("max_tokens", C.API_ARM_MAX_TOKENS),
        ("concurrency", S["concurrency"]), ("retry", "429/5xx and connection errors, backoff %s, %d attempts" % (S["backoff_s"], S["attempts"])),
        ("messages", "one user message, no system message"), ("prompt_kind", C.ARMS[arm]["kind"]),
        ("api_host", api.host), ("n_rows", len(users)), ("rids", [u["rid"] for u in users]),
        ("r1", r1_info), ("pins", C.verify_pins(verbose=False)), ("scored", False)])
    C.add_run(pth, start, len(users) - len(todo))

    lock = threading.Lock()
    counts = Counter()
    stop = {"why": None}
    t0 = time.time()

    def work_one(rid, fw):
        while stop["why"] is None and decide(reqs[rid])[0] == "open":
            try:
                res = api.call([{"role": "user", "content": by[rid]["user"]}], C.API_ARM_MAX_TOKENS, tag)
            except C.BudgetStop as e:
                stop["why"] = str(e)
                return
            answered = res["response"] is not None
            text, fin, usage = C.response_text(res["response"]) if answered else ("", None, {})
            rec = OrderedDict([
                ("rid", rid), ("arm", arm), ("task", task), ("prompt", by[rid]["user"]),
                ("answered", answered), ("refused", bool(res["refused"])), ("error", res["error"]), ("http_status", res["status"]),
                ("attempts", res["attempts"]), ("seconds", res["latency_s"]), ("generation", text), ("finish_reason", fin),
                ("n_prompt_tokens", usage.get("prompt_tokens")), ("n_gen_tokens", usage.get("completion_tokens")),
                ("ts", time.strftime("%Y-%m-%d %H:%M:%S")), ("raw_response", res["response"])])
            with lock:
                C.append_row(fw, rec)
                reqs[rid].append(rec)
                counts["ok" if answered else "refused" if rec["refused"] else "transport"] += 1
            if not answered and not rec["refused"]:
                return

    if todo:
        with open(C.guard_out(pth["work"]), "ab") as fw, ThreadPoolExecutor(max_workers=S["concurrency"]) as ex:
            futs = [ex.submit(work_one, rid, fw) for rid in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if i % 50 == 0 or i == len(todo):
                    print("  %s %s %d/%d rows  requests: %s  (%.0fs)" % (task, arm, i, len(todo), dict(counts), time.time() - t0), flush=True)

    state = OrderedDict((u["rid"], decide(reqs[u["rid"]]) if u["status"] == "ok" else (u["status"], None)) for u in users)
    still = [rid for rid, (st, _) in state.items() if st == "open"]
    if stop["why"]:
        print("  API STOP: %s. %s %s: %d rows are still open; the arm is not sealed." % (stop["why"], task, arm, len(still)), flush=True)
        return 4
    if still:
        print("  INCOMPLETE: %s %s: %d rows without an answer (run the same command again to resume): %s" % (task, arm, len(still), ", ".join(still[:5])), flush=True)
        return 3

    tmp = C.guard_out(pth["rows"]) + ".tmp"
    with open(tmp, "wb") as f:
        for u in users:
            st, ok = state[u["rid"]]
            rs = reqs[u["rid"]]
            rec = OrderedDict([
                ("rid", u["rid"]), ("arm", arm), ("task", task), ("status", st), ("prompt", u["user"]),
                ("generation", ok["generation"] if ok else ""), ("finish_reason", ok["finish_reason"] if ok else None),
                ("truncated", bool(ok and ok["finish_reason"] == "length")),
                ("seconds", ok["seconds"] if ok else None), ("n_prompt_tokens", ok["n_prompt_tokens"] if ok else None),
                ("n_gen_tokens", ok["n_gen_tokens"] if ok else None), ("requests", len(rs)),
                ("error", None if st != "refused" else "; ".join(str(r["error"]) for r in rs if r["refused"]))])
            f.write(C.record_line(rec))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, C.rp(pth["rows"]))
    head = OrderedDict([("experiment", "D14"), ("arm", arm), ("task", task), ("mode", mode)])
    C.seal_rows(pth, [u["rid"] for u in users], head, OrderedDict([
        ("model", S["model"]), ("api_host", api.host), ("max_tokens", C.API_ARM_MAX_TOKENS),
        ("finish_reasons", dict(Counter(str(state[u["rid"]][1]["finish_reason"]) for u in users if state[u["rid"]][0] == "ok"))),
        ("requests", sum(len(v) for v in reqs.values()))]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--arms", default="M_ur,M_r1,M_og")
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true", help="build the messages; no key read, no call, nothing written")
    args = ap.parse_args()
    task, mode = args.task, args.mode
    arms = args.arms.split(",")
    for a in arms:
        if a not in C.arms_of(task, "M"):
            die("%s is not an API arm of %s" % (a, task))
    print("=" * 78)
    print("D14 API arms %s   task %s   %s" % (arms, task, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    C.verify_pins()
    if mode == "FULL" and not args.preflight_only:
        C.require_committed()
    rows = C.load_rows(task, mode)
    r1 = info = None
    if any(C.ARMS[a]["needs_r1"] for a in arms):
        r1, info = C.load_r1(task, mode, rows)
        print("[r1] sealed %s MD5 %s, reading_failed %d" % (info["file"], info["md5"], info["reading_failed"]))
    if args.preflight_only:
        for a in arms:
            us = C.build_users(task, a, rows, r1)
            print("[preflight-only] %s %s: %d messages built (%d not requested: reading_failed). No key read, no call, nothing written."
                  % (task, a, len(us), sum(1 for u in us if u["status"] != "ok")))
        return
    S = C.api_settings()
    api = C.Api()
    print("[api] host %s   model %s   %s   temperature %s   max_tokens %d   concurrency %d   key: read from the key file inside this process, never stored"
          % (api.host, S["model"], dict(S["request_extra"]), S["temperature"], C.API_ARM_MAX_TOKENS, S["concurrency"]), flush=True)
    for a in arms:
        rc = run_arm(task, a, mode, rows, r1, info, api, S)
        if rc:
            sys.exit(rc)
    print("[done] %s %s: arms %s sealed, not scored" % (task, mode, arms), flush=True)


if __name__ == "__main__":
    main()
