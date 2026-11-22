#!/usr/bin/env python3
"""
d15_api.py — D15 commercial-model arms M_ur, M_ue, M_r1 (freeze N4, 11.7, 11.9), login node.

qwen3.8-max with the D14 settings (d11_api_direct constants through d14_common): enable_thinking
false, temperature 0, 2 concurrent requests per arm process, the D11-R retry policy; max_tokens
1024. One user message, no system message:
  M_ur  the UR prompt;  M_ue  the EN instruction with the Urdu item block;  M_r1  the EN prompt with R1
A row whose R1 reading failed is not requested in M_r1 (status reading_failed). A row refused by
the provider on two separate requests is recorded as refused and not requested again
(d14_api.decide, imported).

Freeze 11.9: one process per arm, side by side. Resumable: every request is appended to a work
file; ok rows are never requested again. An arm is sealed only when every row is present.
Nothing is scored: no extractor is imported, rows carry no pred / correct / gold.

KEY HANDLING: as d15_translate.py.

  --arm M_ur|M_ue|M_r1  --mode TEST|FULL  [--preflight-only]
Exit codes: 0 sealed; 3 rows still open (run again); 4 API budget / quota stop.
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

import d15_common as C  # noqa: E402
from d15_common import die  # noqa: E402
from d14_api import decide  # noqa: E402  ('ok' | 'refused' after two refusals | 'open'), D14 9.9


def run_arm(arm, mode, rows, readings, rinfo, api, S):
    pth = C.arm_paths(arm, mode)
    os.makedirs(C.guard_out(pth["dir"]), exist_ok=True)
    if C.is_sealed(pth):
        print("[sealed] %s (%s) is already sealed; no call is made" % (arm, mode), flush=True)
        return 0
    users = C.build_users(arm, rows, readings)
    by = OrderedDict((u["rid"], u) for u in users)
    tag = "%s/%s" % (arm, mode)
    reqs = OrderedDict((u["rid"], []) for u in users)
    for r in C.read_rows(pth["work"], repair=True):
        u = by.get(r["rid"])
        if u is None or u["status"] != "ok" or r["prompt"] != u["user"]:
            die("%s: a request record of %s does not match the message of this run" % (pth["work"], r["rid"]))
        reqs[r["rid"]].append(r)
    todo = [u["rid"] for u in users if u["status"] == "ok" and decide(reqs[u["rid"]])[0] == "open"]
    n_fail = sum(1 for u in users if u["status"] != "ok")
    print("\n[arm] %s %s: rows %d  reading_failed %d  decided %d  to do %d"
          % (arm, mode, len(users), n_fail, len(users) - n_fail - len(todo), len(todo)), flush=True)
    first = next((u for u in users if u["status"] == "ok"), None)
    if first:
        C.log_full_prompt("%s %s (the one user message)" % (arm, first["rid"]), first["user"])
    start = OrderedDict([
        ("experiment", "D15"), ("arm", arm), ("mode", mode), ("model", S["model"]),
        ("request_extra", S["request_extra"]), ("temperature", S["temperature"]), ("max_tokens", C.API_ARM_MAX_TOKENS),
        ("concurrency", S["concurrency"]), ("retry", "429/5xx and connection errors, backoff %s, %d attempts" % (S["backoff_s"], S["attempts"])),
        ("messages", "one user message, no system message"), ("prompt_kind", C.ARMS[arm]["kind"]),
        ("api_host", api.host), ("n_rows", len(users)), ("rids", [u["rid"] for u in users]),
        ("readings", rinfo), ("pins", C.verify_pins(verbose=False)), ("scored", False)])
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
                ("rid", rid), ("arm", arm), ("prompt", by[rid]["user"]),
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
                if i % 100 == 0 or i == len(todo):
                    print("  %s %d/%d rows  requests: %s  (%.0fs)" % (arm, i, len(todo), dict(counts), time.time() - t0), flush=True)

    state = OrderedDict((u["rid"], decide(reqs[u["rid"]]) if u["status"] == "ok" else (u["status"], None)) for u in users)
    still = [rid for rid, (st, _) in state.items() if st == "open"]
    if stop["why"]:
        print("  API STOP: %s. %s: %d rows are still open; the arm is not sealed." % (stop["why"], arm, len(still)), flush=True)
        return 4
    if still:
        print("  INCOMPLETE: %s: %d rows without an answer (run the same command again to resume): %s" % (arm, len(still), ", ".join(still[:5])), flush=True)
        return 3

    tmp = C.guard_out(pth["rows"]) + ".tmp"
    with open(tmp, "wb") as f:
        for u in users:
            st, ok = state[u["rid"]]
            rs = reqs[u["rid"]]
            rec = OrderedDict([
                ("rid", u["rid"]), ("arm", arm), ("status", st), ("prompt", u["user"]),
                ("generation", ok["generation"] if ok else ""), ("finish_reason", ok["finish_reason"] if ok else None),
                ("truncated", bool(ok and ok["finish_reason"] == "length")),
                ("seconds", ok["seconds"] if ok else None), ("n_prompt_tokens", ok["n_prompt_tokens"] if ok else None),
                ("n_gen_tokens", ok["n_gen_tokens"] if ok else None), ("requests", len(rs)),
                ("error", None if st != "refused" else "; ".join(str(r["error"]) for r in rs if r["refused"]))])
            f.write(C.record_line(rec))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, C.rp(pth["rows"]))
    head = OrderedDict([("experiment", "D15"), ("arm", arm), ("mode", mode)])
    C.seal_rows(pth, [u["rid"] for u in users], head, OrderedDict([
        ("model", S["model"]), ("api_host", api.host), ("max_tokens", C.API_ARM_MAX_TOKENS),
        ("finish_reasons", dict(Counter(str(state[u["rid"]][1]["finish_reason"]) for u in users if state[u["rid"]][0] == "ok"))),
        ("requests", sum(len(v) for v in reqs.values()))]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=C.arms_of("M"))
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true", help="build the messages; no key read, no call, nothing written")
    args = ap.parse_args()
    arm, mode = args.arm, args.mode
    print("=" * 78)
    print("D15 API arm %s   %s" % (arm, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    C.verify_pins()
    if mode == "FULL" and not args.preflight_only:
        C.require_committed()
    rows = C.load_rows(mode)
    readings, rinfo = OrderedDict(), None
    rd = C.ARMS[arm]["reading"]
    if rd:
        readings[rd], rinfo = C.load_reading(rd, mode, rows)
        print("[%s] sealed %s MD5 %s, not valid %d" % (rd, rinfo["file"], rinfo["md5"], rinfo["not_valid"]))
    if args.preflight_only:
        us = C.build_users(arm, rows, readings)
        print("[preflight-only] %s: %d messages built (%d not requested: reading_failed). No key read, no call, nothing written."
              % (arm, len(us), sum(1 for u in us if u["status"] != "ok")))
        return
    C.ledger_init()
    S = C.api_settings()
    api = C.Api()
    print("[api] host %s   model %s   %s   temperature %s   max_tokens %d   concurrency %d   key: read from the key file inside this process, never stored"
          % (api.host, S["model"], dict(S["request_extra"]), S["temperature"], C.API_ARM_MAX_TOKENS, S["concurrency"]), flush=True)
    rc = run_arm(arm, mode, rows, readings, rinfo, api, S)
    if rc:
        sys.exit(rc)
    print("[done] %s %s sealed, not scored" % (arm, mode), flush=True)


if __name__ == "__main__":
    main()
