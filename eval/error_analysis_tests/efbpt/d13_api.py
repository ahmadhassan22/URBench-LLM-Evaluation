#!/usr/bin/env python3
"""
d13_api.py — D13 Group 1 API arms (freeze §2, §7.2), run on the login node.

  API_direct_ur_eval  D11 Amendment 1 A1.6 unchanged: system "صرف ہاں یا نہیں لکھیں"
                      (the d11_api_direct constant), user = question_ur. Set eval458.
  API_tr_R1           system "Answer only yes or no.", user = the sealed R1 reading.
                      Sets dev200, eval458.
Model and settings are the d11_api_direct constants (qwen3.8-max,
enable_thinking=false, temperature 0, max_tokens 256, 2 concurrent, D11-R retry
policy). Resumable: ok records are kept and never requested again.

KEY HANDLING (freeze §7.2): the key file ~/.urbench_api_env is read here, inside
this Python process, and nowhere else. The key is never put into the process
environment, never printed, never logged, never stored: it lives in one private
attribute and goes out only in the Authorization header. Every stored string is
scrubbed. START and SEAL files record the host name only.

Nothing is scored: no extractor is imported, rows carry no pred / correct / gold.

  --arm API_direct_ur_eval|API_tr_R1  --set dev200|eval458  --mode TEST|FULL
TEST = DEV200 rows 0-4 under outputs/efbpt/d13/_test/ (never a result). For
API_direct_ur_eval the five TEST answers are compared with the sealed DEV200
file (reported, not fatal).
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d13_common as C  # noqa: E402
from d13_common import die, git_state, read_jsonl  # noqa: E402
from d11_api_direct import (  # noqa: E402  the A1.6 settings, imported
    MODEL, REQUEST_EXTRA, TEMPERATURE, MAX_TOKENS, CONCURRENCY, REQUEST_TIMEOUT_S, RETRY_STATUS,
)
from d11r_common import scrub, scrub_obj, API_BACKOFF_S, API_ATTEMPTS  # noqa: E402

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


class Api:
    """Same request body and retry loop as d11_api_direct.Api (policy unchanged);
    the two messages are arguments, and the key comes from the key file, not from
    the environment."""

    def __init__(self):
        base, key = _read_key_file()
        self._key = key
        self.url = base.rstrip("/") + "/chat/completions"
        self.host = urlparse(base).netloc or "<unparsed>"
        self.secrets = (key,)
        self.session = requests.Session()

    def call(self, system, user):
        body = OrderedDict([
            ("model", MODEL),
            ("messages", [{"role": "system", "content": system}, {"role": "user", "content": user}]),
            ("temperature", TEMPERATURE), ("max_tokens", MAX_TOKENS)])
        body.update(REQUEST_EXTRA)
        attempts, err, status, resp_json, last_latency = 0, None, None, None, None
        while attempts < API_ATTEMPTS:
            attempts += 1
            t0 = time.time()
            try:
                r = self.session.post(self.url, headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
                                      data=json.dumps(body), timeout=REQUEST_TIMEOUT_S)
                last_latency = time.time() - t0
                status = r.status_code
                try:
                    j = r.json()
                except Exception:
                    j = None
                if status == 200 and isinstance(j, dict) and j.get("choices"):
                    resp_json, err = j, None
                    break
                err = "HTTP %s: %s" % (status, scrub(r.text[:600], self.secrets))
                if status in RETRY_STATUS and attempts < API_ATTEMPTS:
                    time.sleep(API_BACKOFF_S[min(attempts - 1, len(API_BACKOFF_S) - 1)])
                    continue
                break
            except requests.RequestException as e:
                last_latency = time.time() - t0
                err = "network: %s" % scrub(str(e)[:300], self.secrets)
                if attempts < API_ATTEMPTS:
                    time.sleep(API_BACKOFF_S[min(attempts - 1, len(API_BACKOFF_S) - 1)])
                    continue
                break
        return OrderedDict([("status", status), ("attempts", attempts), ("latency_s", last_latency),
                            ("response", scrub_obj(resp_json, self.secrets)), ("error", err)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=C.API_ARMS)
    ap.add_argument("--set", default="dev200", choices=C.SETS)
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true", help="build and guard the messages; no key read, no call")
    args = ap.parse_args()
    arm, mode = args.arm, args.mode
    set_name = "dev200" if mode == "TEST" else args.set
    C.check_arm_set(arm, set_name, mode)
    print("=" * 78)
    print("D13 API arm %s   set %s   %s" % (arm, set_name, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    contracts = C.verify_contracts()
    if mode == "FULL":
        C.require_committed()
    ctx = C.make_ctx(set_name, mode)
    msgs, info = C.build_api_messages(arm, ctx)
    if args.preflight_only:
        print("[preflight-only] %d messages built and guarded. No key read, no call, nothing written." % len(msgs))
        return

    pth = C.arm_paths(arm, set_name, mode)
    work = os.path.join(pth["dir"], "d13_%s.work.jsonl" % arm)       # ok and error records in arrival order
    os.makedirs(C.guard_out(pth["dir"]), exist_ok=True)
    if os.path.exists(pth["seal"]):
        seal = json.load(open(pth["seal"], encoding="utf-8"))
        if C.file_entry(pth["rows"])["md5"] != seal["files"][pth["rows"]]["md5"]:
            die("%s is sealed but its row file does not match the seal" % arm)
        print("[sealed] %s %s is already sealed (%d rows); no call is made" % (arm, set_name, seal["n_rows"]))
        return

    api = Api()
    print("[api] host %s   model %s   %s   temperature %s   max_tokens %d   concurrency %d   key: read from the key file inside this process, never stored"
          % (api.host, MODEL, dict(REQUEST_EXTRA), TEMPERATURE, MAX_TOKENS, CONCURRENCY), flush=True)
    by_qid = OrderedDict((m["qid"], m) for m in msgs)
    done = OrderedDict()
    if os.path.exists(work):
        for r in read_jsonl(work):
            if r.get("ok"):
                m = by_qid.get(r["qid"])
                if m is None or r["system"] != m["system"] or r["user"] != m["user"]:
                    die("%s: an ok record of qid %s does not match the messages of this run" % (work, r["qid"]))
                done.setdefault(r["qid"], r)
    todo = [m for m in msgs if m["qid"] not in done]
    print("  items %d  already ok %d  to do %d" % (len(msgs), len(done), len(todo)), flush=True)

    start = OrderedDict([
        ("experiment", "D13"), ("arm", arm), ("group", C.ARMS[arm]["group"]), ("set", set_name), ("mode", mode),
        ("model", MODEL), ("request_extra", REQUEST_EXTRA), ("temperature", TEMPERATURE), ("max_tokens", MAX_TOKENS),
        ("concurrency", CONCURRENCY), ("retry", "429/5xx and connection errors, backoff %s, %d attempts" % (API_BACKOFF_S, API_ATTEMPTS)),
        ("system_message", info["system_message"]), ("user_message", "question_ur" if arm == "API_direct_ur_eval" else "the sealed R1 reading"),
        ("api_host", api.host), ("hostname", socket.gethostname()), ("n_rows", len(msgs)), ("qids", [m["qid"] for m in msgs]),
        ("contracts", contracts), ("scripts", C.script_md5s()), ("git", git_state()), ("guards", info),
        ("readings", OrderedDict((k, OrderedDict([("file", v["file"]), ("md5", v["md5"])])) for k, v in ctx["reading_info"].items())),
        ("scored", False),
    ])
    run = OrderedDict([("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("rows_already_done", len(done)), ("git", start["git"])])
    if os.path.exists(pth["start"]):
        cur = json.load(open(pth["start"], encoding="utf-8"), object_pairs_hook=OrderedDict)
        cur.setdefault("runs", []).append(run)
        start = cur
    else:
        start["runs"] = [run]
    C.write_json(pth["start"], start)

    lock = threading.Lock()
    counts = Counter()
    t0 = time.time()

    def work_one(m):
        return m, api.call(m["system"], m["user"])

    with open(C.guard_out(work), "ab") as fw, ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(work_one, m) for m in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            m, res = fut.result()
            ok = res["response"] is not None
            text, fin, usage = "", None, {}
            if ok:
                ch = (res["response"].get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                text = msg.get("content") or ""
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                fin = ch.get("finish_reason")
                usage = res["response"].get("usage") or {}
            rec = OrderedDict([
                ("qid", m["qid"]), ("arm", arm), ("set", set_name), ("model", MODEL),
                ("system", m["system"]), ("user", m["user"]), ("prompt", m["prompt"]),
                ("ok", ok), ("error", res["error"]), ("http_status", res["status"]), ("attempts", res["attempts"]),
                ("seconds", res["latency_s"]), ("generation", text), ("finish_reason", fin), ("truncated", fin == "length"),
                ("prompt_tokens", usage.get("prompt_tokens")), ("n_gen_tokens", usage.get("completion_tokens")),
                ("ts", time.strftime("%Y-%m-%d %H:%M:%S")), ("raw_response", res["response"])])
            with lock:
                C.append_row(fw, rec)
                if ok:
                    done.setdefault(m["qid"], rec)
                counts["ok" if ok else "err"] += 1
                if i % 20 == 0 or i == len(todo):
                    print("  %d/%d  ok %d  err %d  (%.0fs)" % (i, len(todo), counts["ok"], counts["err"], time.time() - t0), flush=True)

    missing = [m["qid"] for m in msgs if m["qid"] not in done]
    if missing:
        print("  INCOMPLETE: %d qids without an ok record (run the same command again to resume): %s" % (len(missing), ", ".join(missing[:5])))
        sys.exit(3)
    tmp = C.guard_out(pth["rows"]) + ".tmp"
    with open(tmp, "wb") as f:
        for m in msgs:
            f.write(C.record_line(done[m["qid"]]))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, pth["rows"])
    extra = OrderedDict([("model", MODEL), ("api_host", api.host),
                         ("finish_reasons", dict(Counter(done[m["qid"]]["finish_reason"] for m in msgs))),
                         ("work_file", OrderedDict([("path", work), ("md5", C.md5_file(work))]))])
    C.seal_arm(arm, set_name, mode, msgs, extra)

    if mode == "TEST" and arm == "API_direct_ur_eval":
        path, _ = C.SEALED["dev200/API_direct_ur"]
        C.verify_sealed(("dev200/API_direct_ur",))
        sealed = {e["qid"]: e for e in read_jsonl(path)}
        rows = [OrderedDict([("qid", m["qid"]), ("answer_identical_to_sealed", done[m["qid"]]["generation"] == sealed[m["qid"]]["raw"]),
                             ("question_identical", m["user"] == sealed[m["qid"]]["question_ur"])]) for m in msgs]
        n = sum(1 for x in rows if x["answer_identical_to_sealed"])
        rep = OrderedDict([("what", "TEST-5: API_direct_ur on DEV200 rows 0-4 vs the sealed DEV200 file (reported, not fatal; freeze §7.2)"),
                           ("sealed_file", OrderedDict([("path", path), ("md5", C.SEALED["dev200/API_direct_ur"][1])])),
                           ("n_rows", len(rows)), ("n_answers_identical", n), ("rows", rows)])
        C.write_json(os.path.join(pth["dir"], "TEST_REPORT_%s.json" % arm), rep)
        print("  [TEST-5] API_direct_ur: %d/%d answers are byte-identical to the sealed DEV200 answers; questions identical %d/%d (reported)"
              % (n, len(rows), sum(1 for x in rows if x["question_identical"]), len(rows)))
    print("[done] %s %s %s: %d rows sealed, not scored" % (arm, set_name, mode, len(msgs)), flush=True)


if __name__ == "__main__":
    main()
