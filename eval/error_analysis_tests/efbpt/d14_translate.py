#!/usr/bin/env python3
"""
d14_translate.py — D14 translation R1 (freeze F3, 9.5), run on the login node.

One request per row: qwen3.8-max, thinking off, temperature 0, max_tokens 512,
4 concurrent, the D11-R retry policy. SYSTEM = the F3 translator message; USER =
the item block with the Urdu text. The gold answer is never sent: the rows this
script sees carry the text fields only.

A translation is valid if it has the same labels in the same order (9.5). An
answered request with an invalid output is requested once more; a row with two
decided requests (answered-invalid or refused by the provider's input
inspection, HTTP 400) and no valid output is reading_failed. A row with only
network / 429 / 5xx failures stays open and is requested again on a resume.

Resumable: every request is appended to a work file; decided rows are never
requested again. When every row is decided, one row file is written in row
order and sealed. Nothing is scored.

KEY HANDLING (9.9): the key file is read inside this process only (d14_common.Api);
the key is never printed, logged or stored; every stored string is scrubbed.

  --task csqa|piqa|gsm8k  --mode TEST|FULL
Exit codes: 0 sealed; 3 rows still open (run again to resume); 4 API budget / quota stop.
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

MAX_DECIDED = 2      # 9.5: one request, and one more if the first is invalid or refused


def decide(task, reqs):
    """('ok', block) | ('reading_failed', None) | ('open', None) from the requests of one row."""
    for r in reqs:
        if r["answered"] and r["r1_block"] is not None:
            return "ok", r["r1_block"]
    decided = [r for r in reqs if r["answered"] or r["refused"]]
    if len(decided) >= MAX_DECIDED:
        return "reading_failed", None
    return "open", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--mode", required=True, choices=C.MODES)
    args = ap.parse_args()
    task, mode = args.task, args.mode
    print("=" * 78)
    print("D14 translation R1   task %s   %s" % (task, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    C.verify_pins()
    if mode == "FULL":
        C.require_committed()
    rows = C.load_rows(task, mode)
    users = OrderedDict((r["rid"], C.ur_block(task, r)) for r in rows)
    pth = C.r1_paths(task, mode)
    os.makedirs(C.guard_out(pth["dir"]), exist_ok=True)
    if C.is_sealed(pth):
        print("[sealed] the R1 readings of %s (%s) are already sealed; no call is made" % (task, mode))
        return

    S = C.api_settings()
    tag = "r1/%s/%s" % (task, mode)
    reqs = OrderedDict((rid, []) for rid in users)
    for r in C.read_rows(pth["work"], repair=True):
        if r["rid"] not in users or r["user"] != users[r["rid"]] or r["system"] != C.R1_SYSTEM:
            die("%s: a request record of %s does not match the messages of this run" % (pth["work"], r["rid"]))
        reqs[r["rid"]].append(r)
    todo = [rid for rid in users if decide(task, reqs[rid])[0] == "open"]
    print("  rows %d  decided %d  to do %d" % (len(users), len(users) - len(todo), len(todo)), flush=True)

    api = C.Api() if todo else None
    start = OrderedDict([
        ("experiment", "D14"), ("what", "translation R1"), ("task", task), ("mode", mode), ("model", S["model"]),
        ("request_extra", S["request_extra"]), ("temperature", S["temperature"]), ("max_tokens", C.R1_MAX_TOKENS),
        ("concurrency", C.R1_CONCURRENCY), ("retry", "429/5xx and connection errors, backoff %s, %d attempts" % (S["backoff_s"], S["attempts"])),
        ("system_message", C.R1_SYSTEM), ("user_message", "the item block with the Urdu text (freeze 9.4)"),
        ("api_host", api.host if api else None), ("n_rows", len(users)), ("rids", list(users)),
        ("pins", C.verify_pins(verbose=False)), ("scored", False)])
    C.add_run(pth, start, len(users) - len(todo))
    if api:
        print("[api] host %s   model %s   %s   temperature %s   max_tokens %d   concurrency %d   key: read from the key file inside this process, never stored"
              % (api.host, S["model"], dict(S["request_extra"]), S["temperature"], C.R1_MAX_TOKENS, C.R1_CONCURRENCY), flush=True)

    lock = threading.Lock()
    counts = Counter()
    stop = {"why": None}
    t0 = time.time()

    def work_one(rid, fw):
        """Request a row until it is decided, a transport failure leaves it open, or the budget stops."""
        while stop["why"] is None and decide(task, reqs[rid])[0] == "open":
            msgs = [{"role": "system", "content": C.R1_SYSTEM}, {"role": "user", "content": users[rid]}]
            try:
                res = api.call(msgs, C.R1_MAX_TOKENS, tag)
            except C.BudgetStop as e:
                stop["why"] = str(e)
                return
            answered = res["response"] is not None
            text, fin, usage = C.response_text(res["response"]) if answered else ("", None, {})
            rec = OrderedDict([
                ("rid", rid), ("task", task), ("system", C.R1_SYSTEM), ("user", users[rid]),
                ("answered", answered), ("refused", bool(res["refused"])), ("error", res["error"]), ("http_status", res["status"]),
                ("attempts", res["attempts"]), ("seconds", res["latency_s"]), ("text", text), ("finish_reason", fin),
                ("r1_block", C.parse_translation(task, text) if answered else None),
                ("n_prompt_tokens", usage.get("prompt_tokens")), ("n_gen_tokens", usage.get("completion_tokens")),
                ("ts", time.strftime("%Y-%m-%d %H:%M:%S")), ("raw_response", res["response"])])
            with lock:
                C.append_row(fw, rec)
                reqs[rid].append(rec)
                counts["valid" if rec["r1_block"] is not None else "invalid" if answered else "refused" if rec["refused"] else "transport"] += 1
            if not answered and not rec["refused"]:
                return          # transport failure after the retry policy: the row stays open for a resume

    if todo:
        with open(C.guard_out(pth["work"]), "ab") as fw, ThreadPoolExecutor(max_workers=C.R1_CONCURRENCY) as ex:
            futs = [ex.submit(work_one, rid, fw) for rid in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if i % 50 == 0 or i == len(todo):
                    print("  %d/%d rows  requests: %s  (%.0fs)" % (i, len(todo), dict(counts), time.time() - t0), flush=True)

    state = OrderedDict((rid, decide(task, reqs[rid])) for rid in users)
    open_rows = [rid for rid, (st, _) in state.items() if st == "open"]
    if stop["why"]:
        print("  API STOP: %s. %d rows are still open; nothing is sealed." % (stop["why"], len(open_rows)))
        sys.exit(4)
    if open_rows:
        print("  INCOMPLETE: %d rows without a decided request (run the same command again to resume): %s" % (len(open_rows), ", ".join(open_rows[:5])))
        sys.exit(3)

    tmp = C.guard_out(pth["rows"]) + ".tmp"
    n_req = n_cut = 0
    with open(tmp, "wb") as f:
        for rid in users:
            st, block = state[rid]
            rs = reqs[rid]
            n_req += len(rs)
            secs = [r["seconds"] for r in rs if isinstance(r["seconds"], (int, float))]
            rec = OrderedDict([
                ("rid", rid), ("task", task), ("status", st), ("system", C.R1_SYSTEM), ("user", users[rid]), ("r1_block", block),
                ("accepted_finish_reason", next((r["finish_reason"] for r in rs if r["answered"] and r["r1_block"] is not None), None)),
                ("failure", None if st == "ok" else "; ".join("refused" if r["refused"] else "invalid output" if r["answered"] else "transport" for r in rs)),
                ("requests", [OrderedDict((k, r[k]) for k in ("answered", "refused", "error", "http_status", "text", "finish_reason", "seconds",
                                                               "n_prompt_tokens", "n_gen_tokens", "ts")) for r in rs]),
                ("seconds", round(sum(secs), 3) if secs else None),
                ("n_prompt_tokens", sum(r["n_prompt_tokens"] or 0 for r in rs)), ("n_gen_tokens", sum(r["n_gen_tokens"] or 0 for r in rs))])
            n_cut += int(rec["accepted_finish_reason"] == "length")
            f.write(C.record_line(rec))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, C.rp(pth["rows"]))
    nfail = sum(1 for st, _ in state.values() if st == "reading_failed")
    head = OrderedDict([("experiment", "D14"), ("what", "translation R1"), ("task", task), ("mode", mode)])
    C.seal_rows(pth, list(users), head, OrderedDict([
        ("model", S["model"]), ("api_host", api.host if api else None), ("requests", n_req),
        ("accepted_readings_that_reached_max_tokens", n_cut),
        ("reading_failed", nfail), ("reading_failed_rids", [rid for rid, (st, _) in state.items() if st == "reading_failed"])]))
    print("[done] R1 %s %s: %d rows sealed, reading_failed %d, requests %d; not scored" % (task, mode, len(users), nfail, n_req), flush=True)


if __name__ == "__main__":
    main()
