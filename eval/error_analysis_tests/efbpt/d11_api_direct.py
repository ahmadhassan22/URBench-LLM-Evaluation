#!/usr/bin/env python3
"""
d11_api_direct.py — D11 RTG §4 descriptive reference API_direct_ur (not gated).

qwen3.8-max, enable_thinking=false, temperature 0, the Urdu question alone:
system message "صرف ہاں یا نہیں لکھیں" (write only yes or no), user message =
question_ur. 200 DEV200 calls; accuracy = the reader model answering alone.
Amendment 1 A1.6: max_tokens 256, 2 concurrent, D11-R retry policy (429/5xx
and connection errors, backoff 2/4/8 s, 3 attempts), resumable (ok records
kept), scored with the frozen extractor (efbpt_eval_dev200.extract_answer,
Rule 2) and the secondary scorer, sealed. Refuses any set but dev200.

KEY HANDLING as d11r_api_reader.py: URB_API_BASE / URB_API_KEY from the
environment only (source ~/.urbench_api_env), Authorization header only,
every stored string scrubbed, never printed.

Outputs: outputs/efbpt/d11/api_direct/dev200/ api_direct_ur.jsonl, A_START.json, A_SEAL.json
         (--limit N: outputs/efbpt/d11/_test/api_direct/dev200/, never a result)
Usage (from ~/URBench, env urbench_eval, after `set -a; source ~/.urbench_api_env; set +a`):
  python eval/error_analysis_tests/efbpt/d11_api_direct.py --set dev200
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
from d11_common import (  # noqa: E402
    out_dir, verify_contracts, load_set_rows, needles_for, guard_out_path, file_entry, die, md5_file, git_state,
)
from d11r_common import scrub, scrub_obj, append_jsonl, read_jsonl, API_BACKOFF_S, API_ATTEMPTS  # noqa: E402
from efbpt_eval_dev200 import extract_answer  # noqa: E402  frozen extractor (Rule 2)
from d1_score_dual import extract_answer_secondary  # noqa: E402
from d3_oracle_retrieval import score  # noqa: E402
from d8_reachable_ceiling import leak_guard  # noqa: E402

MODEL = "qwen3.8-max"
REQUEST_EXTRA = OrderedDict([("enable_thinking", False)])
TEMPERATURE = 0
MAX_TOKENS = 256
CONCURRENCY = 2
REQUEST_TIMEOUT_S = 180
RETRY_STATUS = {429, 500, 502, 503, 504}
# freeze §4, verbatim: "صرف ہاں یا نہیں لکھیں" — built from codepoints so a lookalike cannot slip in
INSTRUCTION_UR = "".join(chr(c) for c in [
    0x0635, 0x0631, 0x0641, 0x0020, 0x06C1, 0x0627, 0x06BA, 0x0020, 0x06CC, 0x0627, 0x0020,
    0x0646, 0x06C1, 0x06CC, 0x06BA, 0x0020, 0x0644, 0x06A9, 0x06BE, 0x06CC, 0x06BA])
assert INSTRUCTION_UR == "صرف ہاں یا نہیں لکھیں"


class Api:
    """Same credential handling and retry loop as d11r_api_reader.Api (that
    class hard-wires the translator prompt, so the loop is repeated here with
    the §4 messages; policy unchanged)."""

    def __init__(self):
        base = os.environ.get("URB_API_BASE", "").strip()
        key = os.environ.get("URB_API_KEY", "").strip()
        if not base or not key:
            die("URB_API_BASE / URB_API_KEY not in the environment (source ~/.urbench_api_env first)")
        self._key = key
        self.url = base.rstrip("/") + "/chat/completions"
        self.host = urlparse(base).netloc or "<unparsed>"
        self.secrets = (key,)
        self.session = requests.Session()

    def call(self, question_ur):
        body = OrderedDict([
            ("model", MODEL),
            ("messages", [{"role": "system", "content": INSTRUCTION_UR}, {"role": "user", "content": question_ur}]),
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
                            ("response", scrub_obj(resp_json, self.secrets)), ("error", err),
                            ("request", scrub_obj(body, self.secrets))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev200", choices=("dev200",), help="API_direct_ur is defined on DEV200 only (freeze §4)")
    ap.add_argument("--limit", type=int, default=None, help="TEST: first N rows under _test/ (never a result)")
    args = ap.parse_args()
    is_test = args.limit is not None
    out = out_dir("api_direct", args.set, is_test)
    path = os.path.join(out, "api_direct_ur.jsonl")
    print("=" * 78)
    print("D11 RTG API_direct_ur  (freeze §4 descriptive reference; A1.6)%s" % ("  TEST — NOT A RESULT" if is_test else ""))
    print("=" * 78, flush=True)
    contracts = verify_contracts()
    rows = load_set_rows(args.set, limit=args.limit)
    needles = needles_for(load_set_rows(args.set), args.set)
    api = Api()
    print("[api] host %s   model %s   %s   temperature %s   max_tokens %d   concurrency %d   key: from environment, never stored"
          % (api.host, MODEL, dict(REQUEST_EXTRA), TEMPERATURE, MAX_TOKENS, CONCURRENCY))
    # leak guard: the prompt is the instruction + the Urdu question and nothing else
    leak_guard("API_direct_ur", [INSTRUCTION_UR + "\n" + r["question_ur"] for r in rows], needles, fatal_gold_facts=True)
    os.makedirs(out, exist_ok=True)
    guard_out_path(out, path)
    done = {r["qid"] for r in read_jsonl(path) if r.get("ok")} if os.path.exists(path) else set()
    todo = [r for r in rows if r["qid"] not in done]
    print("  items %d  already ok %d  to do %d" % (len(rows), len(done), len(todo)), flush=True)
    start = OrderedDict([
        ("job", "API_direct_ur"), ("experiment", "D11 RTG"), ("set", args.set), ("mode", "TEST" if is_test else "FULL"),
        ("model", MODEL), ("request_extra", REQUEST_EXTRA), ("temperature", TEMPERATURE), ("max_tokens", MAX_TOKENS),
        ("concurrency", CONCURRENCY), ("retry", "429/5xx and connection errors, backoff %s, %d attempts" % (API_BACKOFF_S, API_ATTEMPTS)),
        ("system_message", INSTRUCTION_UR), ("user_template", "{question_ur}"),
        ("api_host", api.host), ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("hostname", socket.gethostname()),
        ("n_rows", len(rows)), ("contracts", contracts),
        ("script_md5", md5_file(os.path.abspath(__file__))), ("git", git_state()),
        ("extractor", "efbpt_eval_dev200.extract_answer (Rule 2) + d1_score_dual.extract_answer_secondary"),
    ])
    sp = os.path.join(out, "A_START.json")
    if os.path.exists(sp):
        cur = json.load(open(sp, encoding="utf-8"), object_pairs_hook=OrderedDict)
        cur.setdefault("runs", []).append(OrderedDict([("start_time", start["start_time"]), ("script_md5", start["script_md5"]), ("git", start["git"])]))
        start = cur
    json.dump(start, open(sp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    lock = threading.Lock()
    counts = Counter()
    t0 = time.time()

    def work(r):
        return r, api.call(r["question_ur"])

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r, res = fut.result()
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
            pred = extract_answer(text) if ok else None
            rec = OrderedDict([
                ("qid", r["qid"]), ("arm", "API_direct_ur"), ("set", args.set), ("model", MODEL), ("gold", r["gold"]),
                ("question_ur", r["question_ur"]), ("ok", ok), ("error", res["error"]), ("http_status", res["status"]),
                ("attempts", res["attempts"]), ("latency_s", res["latency_s"]),
                ("raw", text), ("pred", pred), ("pred_secondary", extract_answer_secondary(text) if ok else None),
                ("correct", ok and pred == r["gold"]), ("unparsed", ok and pred is None),
                ("finish_reason", fin), ("prompt_tokens", usage.get("prompt_tokens")), ("completion_tokens", usage.get("completion_tokens")),
                ("ts", time.strftime("%Y-%m-%d %H:%M:%S")), ("raw_response", res["response"])])
            with lock:
                append_jsonl(path, rec)
                counts["ok" if ok else "err"] += 1
                if i % 20 == 0 or i == len(todo):
                    print("  %d/%d  ok %d  err %d  (%.0fs)" % (i, len(todo), counts["ok"], counts["err"], time.time() - t0), flush=True)
    recs = read_jsonl(path)
    best = OrderedDict()
    for r in recs:
        if r.get("ok") or r["qid"] not in best:
            best[r["qid"]] = r
    missing = [r["qid"] for r in rows if not best.get(r["qid"], {}).get("ok")]
    if missing:
        print("  INCOMPLETE: %d qids without an ok record (re-run the same command to resume): %s" % (len(missing), ", ".join(missing[:5])))
        sys.exit(3)
    ordered = [best[r["qid"]] for r in rows]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    pri, sec = score(ordered, "pred"), score(ordered, "pred_secondary")
    n_yes = sum(1 for r in rows if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(rows) - n_yes) / len(rows)
    summary = OrderedDict([
        ("n", pri["n"]), ("accuracy", pri["accuracy"]), ("correct", pri["correct"]), ("unparsed_rate", pri["unparsed_rate"]),
        ("predicted_yes_rate", pri["predicted_yes_rate"]), ("accuracy_secondary", sec["accuracy"]), ("unparsed_rate_secondary", sec["unparsed_rate"]),
        ("floor", round(floor, 2)), ("finish_reasons", dict(Counter(r["finish_reason"] for r in ordered))),
        ("mean_completion_tokens", round(sum(r["completion_tokens"] or 0 for r in ordered) / len(ordered), 1)),
        ("mean_latency_s", round(sum(r["latency_s"] or 0 for r in ordered) / len(ordered), 3)),
        ("sample_raw_row0", ordered[0]["raw"][:200])])
    line = ("API_direct_ur %s %s%s: accuracy %.2f%% (%d/%d correct)  unparsed %.2f%%  pred-yes %.2f%%  secondary %.2f%%  floor %.2f%%  "
            "(descriptive reference, not gated; A_r2 60.0 sealed D10)"
            % (MODEL, args.set, " TEST" if is_test else "", pri["accuracy"], pri["correct"], pri["n"], pri["unparsed_rate"],
               pri["predicted_yes_rate"], sec["accuracy"], floor))
    print("\n" + line, flush=True)
    files = OrderedDict((p, file_entry(p)) for p in (sp, path))
    seal = OrderedDict([("job", "API_direct_ur"), ("experiment", "D11 RTG"), ("set", args.set), ("mode", "TEST" if is_test else "FULL"),
                        ("model", MODEL), ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("script_md5", md5_file(os.path.abspath(__file__))),
                        ("d11_freeze_md5", contracts["d11_freeze"]["md5"]), ("d11_amendment1_md5", contracts["d11_amendment_1"]["md5"]),
                        ("summary", summary), ("summary_line", line), ("files", files)])
    json.dump(seal, open(os.path.join(out, "A_SEAL.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("  [sealed] %s" % os.path.join(out, "A_SEAL.json"))


if __name__ == "__main__":
    main()
