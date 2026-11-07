#!/usr/bin/env python3
"""
d11r_api_reader.py — D11-R API readers (docs/EFBPT_D11R_READER_FREEZE.md §2, §4).

OpenAI-compatible chat completions at $URB_API_BASE/chat/completions with
temperature 0, max_tokens 256, 4 concurrent requests, retry on 429/5xx (and
connection errors) with 2/4/8 s backoff, 3 attempts. Models: qwen3.8-max,
deepseek-v4-pro, deepseek-v4-flash, qwen3.7-flash.

KEY HANDLING. The key is read from the environment variable URB_API_KEY (and
the base URL from URB_API_BASE), typically via `source ~/.urbench_api_env`.
It is placed in the Authorization header only. It is never printed, logged or
written: every stored or printed string passes through scrub(), which would
replace the key by <REDACTED> if it ever appeared (it does not appear in API
responses). R_START records only the host name of the base URL.

Modes
  --probe  [--model M|all]    §4 PROBE: the first 3 RDEV300 rows × the thinking
                              variants (enable_thinking=false; thinking=
                              {"type":"disabled"}; reasoning_effort="none";
                              default). Records only, no metric. Writes
                              outputs/efbpt/d11r/_probe/probe_<model>.jsonl and
                              updates _probe/probe_summary.json with the chosen
                              variant per model = first variant (in that order)
                              accepted on all probe rows with reasoning_tokens
                              == 0, else "default". Prints a table.
  --full --model M --set S    FULL run for one model and one item set, with the
                              variant chosen by the probe (R_START records it).
                              Resumable: qids with an ok record in the readings
                              file are skipped (the key is time-limited). When
                              every qid of the set is done, the file is
                              compacted to one record per qid in item order and
                              sealed into R_SEAL.json.
  --limit N                   with --full: TEST on the first N rows, written
                              under outputs/efbpt/d11r/_test/<reader>/.

Usage (from ~/URBench, env urbench_eval, after `source ~/.urbench_api_env`):
  python eval/error_analysis_tests/efbpt/d11r_api_reader.py --probe --model all
  nohup python eval/error_analysis_tests/efbpt/d11r_api_reader.py --full --model qwen3.8-max --set rdev300 \
        > logs/d11r_api_qwen3.8-max_rdev300.log 2>&1 &
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from d11r_common import (  # noqa: E402
    ROOT, FREEZE_MD5, API_MODELS, API_TEMPERATURE, API_MAX_TOKENS, API_CONCURRENCY, API_BACKOFF_S,
    API_ATTEMPTS, THINKING_VARIANTS, SYSTEM_PROMPT, USER_TEMPLATE, PROBE_DIR, ITEM_SETS,
    die, verify_freeze, load_items, api_reader_name, md5_file, scrub, scrub_obj, make_record,
    reader_dir, readings_path, append_jsonl, write_json, write_start, update_seal, done_qids,
    compact_jsonl, git_state,
)

os.chdir(ROOT)
PROBE_ROWS = 3
REQUEST_TIMEOUT_S = 180
RETRY_STATUS = {429, 500, 502, 503, 504}
PROBE_SUMMARY = os.path.join(PROBE_DIR, "probe_summary.json")


# ----------------------------------------------------------------------------
# Credentials: environment only
# ----------------------------------------------------------------------------
class Api:
    def __init__(self):
        base = os.environ.get("URB_API_BASE", "").strip()
        key = os.environ.get("URB_API_KEY", "").strip()
        if not base or not key:
            die("URB_API_BASE / URB_API_KEY not in the environment (source ~/.urbench_api_env first); "
                "nothing is read from files")
        self._key = key
        self.url = base.rstrip("/") + "/chat/completions"
        self.host = urlparse(base).netloc or "<unparsed>"
        self.secrets = (key,)
        self.session = requests.Session()

    def headers(self):
        return {"Authorization": "Bearer " + self._key, "Content-Type": "application/json"}

    def call(self, model, question_ur, extra):
        """One chat completion with the frozen retry policy. Returns an
        OrderedDict: status, attempts, latency_s, response (dict|None),
        error (str|None), request (body without credentials)."""
        body = OrderedDict([
            ("model", model),
            ("messages", [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": USER_TEMPLATE.format(question_ur=question_ur)}]),
            ("temperature", API_TEMPERATURE),
            ("max_tokens", API_MAX_TOKENS),
        ])
        body.update(extra)
        attempts = 0
        err = None
        status = None
        resp_json = None
        t_all = time.time()
        last_latency = None
        while attempts < API_ATTEMPTS:
            attempts += 1
            t0 = time.time()
            try:
                r = self.session.post(self.url, headers=self.headers(), data=json.dumps(body),
                                      timeout=REQUEST_TIMEOUT_S)
                last_latency = time.time() - t0
                status = r.status_code
                text = r.text
                try:
                    j = r.json()
                except Exception:
                    j = None
                if status == 200 and isinstance(j, dict) and j.get("choices"):
                    resp_json = j
                    err = None
                    break
                err = "HTTP %s: %s" % (status, scrub(text[:600], self.secrets))
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
                            ("total_s", time.time() - t_all), ("response", scrub_obj(resp_json, self.secrets)),
                            ("error", err), ("request", scrub_obj(body, self.secrets))])


def extract(resp):
    """content, reasoning_content, finish_reason, prompt_tokens, completion_tokens,
    reasoning_tokens (reported or None), reasoning_evidence."""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):                       # some gateways return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    content = content or ""
    reasoning_content = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = resp.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    rt = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if rt is None:
        rt = usage.get("reasoning_tokens")
    if rt is not None:
        evidence = "usage"
    elif reasoning_content:
        evidence = "reasoning_content_present_no_usage_count"
    elif "<think>" in content:
        evidence = "think_tag_in_content_no_usage_count"
    else:
        evidence = "not_reported_no_reasoning_content"
    return OrderedDict([
        ("content", content), ("reasoning_content", reasoning_content), ("finish_reason", ch.get("finish_reason")),
        ("prompt_tokens", usage.get("prompt_tokens")), ("completion_tokens", usage.get("completion_tokens")),
        ("reasoning_tokens", rt), ("reasoning_evidence", evidence),
    ])


def effective_reasoning_zero(ex):
    """The §2 rule 'reasoning_tokens == 0': a reported 0, or nothing reported
    AND no reasoning content / think tag in the reply."""
    if ex["reasoning_tokens"] is not None:
        return ex["reasoning_tokens"] == 0
    return ex["reasoning_evidence"] == "not_reported_no_reasoning_content"


def script_identity():
    return OrderedDict([("path", os.path.relpath(os.path.abspath(__file__), ROOT)),
                        ("md5", md5_file(os.path.abspath(__file__)))])


def probe_record(it, res, model, vname, extra, mode):
    ok = res["response"] is not None
    exd = extract(res["response"]) if ok else None
    return make_record(it["qid"], "rdev300", api_reader_name(model), it["question_ur"],
                       exd["content"] if ok else "", exd["finish_reason"] if ok else None,
                       exd["prompt_tokens"] if ok else None, exd["completion_tokens"] if ok else None,
                       exd["reasoning_tokens"] if ok else None, exd["reasoning_evidence"] if ok else None,
                       res["latency_s"], ok, res["error"],
                       extra=OrderedDict([("mode", mode), ("model", model), ("thinking_variant", vname),
                                          ("request_extra", extra), ("http_status", res["status"]),
                                          ("attempts", res["attempts"]),
                                          ("reasoning_content", exd["reasoning_content"] if ok else None),
                                          ("raw_response", res["response"])])), exd


# ----------------------------------------------------------------------------
# PROBE
# ----------------------------------------------------------------------------
def run_probe(api, models):
    verify_freeze()
    os.makedirs(PROBE_DIR, exist_ok=True)
    items = load_items("rdev300", with_gold=False, limit=PROBE_ROWS)
    order = [it["qid"] for it in items]
    print("D11-R PROBE  %d RDEV300 rows × %d models × %d variants  (records only, no metric)"
          % (len(items), len(models), len(THINKING_VARIANTS)), flush=True)
    if os.path.exists(PROBE_SUMMARY):
        summary = json.load(open(PROBE_SUMMARY, encoding="utf-8"), object_pairs_hook=OrderedDict)
    else:
        summary = OrderedDict([("freeze_md5", FREEZE_MD5), ("probe_rows", order), ("models", OrderedDict())])
    table = []
    for model in models:
        out_path = os.path.join(PROBE_DIR, "probe_%s.jsonl" % model)
        if os.path.exists(out_path):
            os.remove(out_path)
        per_variant = OrderedDict()
        for vname, extra in THINKING_VARIANTS.items():
            results = []
            with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as ex:
                futs = {ex.submit(api.call, model, it["question_ur"], extra): it for it in items}
                for fut in as_completed(futs):
                    results.append((futs[fut], fut.result()))
            results.sort(key=lambda x: order.index(x[0]["qid"]))
            rts, cts, lats, evid, errors, row0 = [], [], [], [], [], None
            accepted = 0
            zero_all = True
            for it, res in results:
                rec, exd = probe_record(it, res, model, vname, extra, "probe")
                append_jsonl(out_path, rec)
                lats.append(None if res["latency_s"] is None else round(res["latency_s"], 2))
                if exd is not None:
                    accepted += 1
                    rts.append(exd["reasoning_tokens"])
                    cts.append(exd["completion_tokens"])
                    evid.append(exd["reasoning_evidence"])
                    zero_all = zero_all and effective_reasoning_zero(exd)
                else:
                    errors.append(res["error"])
                    zero_all = False
                if it["qid"] == order[0]:
                    row0 = rec
            all_ok = accepted == len(items)
            per_variant[vname] = OrderedDict([
                ("accepted", accepted), ("n", len(items)), ("all_accepted", all_ok),
                ("reasoning_tokens", rts), ("reasoning_evidence", evid),
                ("reasoning_zero_on_all_rows", bool(all_ok and zero_all)), ("completion_tokens", cts),
                ("latency_s", lats), ("errors", errors[:3]),
                ("row0_reading", row0["reading"] if row0 else None),
                ("row0_flags", row0["flags"] if row0 else None),
            ])
            table.append((model, vname, per_variant[vname]))
        chosen = "default"
        for vname in THINKING_VARIANTS:
            if vname == "default":
                break
            if per_variant[vname]["reasoning_zero_on_all_rows"]:
                chosen = vname
                break
        summary["models"][model] = OrderedDict([
            ("probe_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("variants", per_variant),
            ("chosen_variant", chosen), ("chosen_request_extra", THINKING_VARIANTS[chosen]),
            ("rule", "first variant in freeze order accepted on all probe rows with reasoning_tokens == 0, else default"),
            ("records", os.path.relpath(out_path, ROOT)), ("api_host", api.host)])
        write_json(PROBE_SUMMARY, summary)
        print("  %s -> chosen variant: %s" % (model, chosen), flush=True)

    print("\n%-18s %-24s %-10s %-18s %-16s %-20s  %s" % ("model", "variant", "accepted", "reasoning_tok",
                                                         "completion_tok", "latency_s", "reading of row 0"))
    for model, vname, pv in table:
        if pv["all_accepted"]:
            acc = "yes"
        else:
            acc = "%d/%d %s" % (pv["accepted"], pv["n"], (pv["errors"][0][:70] if pv["errors"] else ""))
        print("%-18s %-24s %-10s %-18s %-16s %-20s  %s" % (
            model, vname, acc, str(pv["reasoning_tokens"]), str(pv["completion_tokens"]),
            str(pv["latency_s"]), (pv["row0_reading"] or "")[:100]))
    print("\nprobe summary -> %s" % PROBE_SUMMARY, flush=True)


# ----------------------------------------------------------------------------
# FULL (one model, one set, resumable)
# ----------------------------------------------------------------------------
def run_full(api, model, set_name, limit):
    verify_freeze()
    if not os.path.exists(PROBE_SUMMARY):
        die("probe summary missing (%s); run --probe first (§2: the variant is chosen by the probe)" % PROBE_SUMMARY)
    summary = json.load(open(PROBE_SUMMARY, encoding="utf-8"), object_pairs_hook=OrderedDict)
    if model not in summary.get("models", {}):
        die("model %s has no probe entry; run --probe --model %s first" % (model, model))
    variant = summary["models"][model]["chosen_variant"]
    extra = THINKING_VARIANTS[variant]
    reader = api_reader_name(model)
    test = bool(limit)
    rdir = reader_dir(reader, test)
    items = load_items(set_name, with_gold=False, limit=limit)
    order = [it["qid"] for it in items]
    path = readings_path(reader, set_name, test)
    os.makedirs(rdir, exist_ok=True)
    done, n_existing = done_qids(path)
    todo = [it for it in items if it["qid"] not in done]
    print("D11-R FULL  reader %s  set %s  variant %s%s" % (reader, set_name, variant,
                                                           "  (TEST limit %d)" % limit if test else ""))
    print("  items %d  already ok %d  to do %d  (existing records %d)" % (len(items), len(done), len(todo), n_existing),
          flush=True)
    write_start(rdir, OrderedDict([
        ("job", "R-api"), ("reader", reader), ("model", model), ("mode", "TEST" if test else "FULL"),
        ("thinking_variant", variant), ("request_extra", extra),
        ("probe_summary", OrderedDict([("path", os.path.relpath(PROBE_SUMMARY, ROOT)), ("md5", md5_file(PROBE_SUMMARY))])),
        ("api", OrderedDict([("host", api.host), ("endpoint", "chat/completions"), ("temperature", API_TEMPERATURE),
                             ("max_tokens", API_MAX_TOKENS), ("concurrency", API_CONCURRENCY),
                             ("retry", "429/5xx, backoff %s, %d attempts" % (API_BACKOFF_S, API_ATTEMPTS))])),
        ("system_prompt", SYSTEM_PROMPT), ("user_template", USER_TEMPLATE),
        ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("hostname", socket.gethostname()),
        ("slurm_job_id", os.environ.get("SLURM_JOB_ID")), ("sets", [set_name]),
        ("script_md5", script_identity()["md5"]), ("git", git_state()),
    ]))
    lock = threading.Lock()
    counts = {"ok": 0, "err": 0}
    lat = []
    t0 = time.time()

    def work(it):
        return it, api.call(model, it["question_ur"], extra)

    with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            it, res = fut.result()
            rec, exd = probe_record(it, res, model, variant, extra, "test" if test else "full")
            rec["set"] = set_name
            del rec["request_extra"]
            with lock:
                append_jsonl(path, rec)
                if exd is not None:
                    counts["ok"] += 1
                    if res["latency_s"] is not None:
                        lat.append(res["latency_s"])
                else:
                    counts["err"] += 1
                if i % 20 == 0 or i == len(todo):
                    el = time.time() - t0
                    print("  %d/%d  ok %d  err %d  (%.0fs, mean latency %.2fs)"
                          % (i, len(todo), counts["ok"], counts["err"], el, sum(lat) / len(lat) if lat else 0.0), flush=True)
    done, _ = done_qids(path)
    missing = [q for q in order if q not in done]
    if missing:
        print("  INCOMPLETE: %d qids without an ok record (re-run the same command to resume): %s%s"
              % (len(missing), ", ".join(missing[:5]), " ..." if len(missing) > 5 else ""), flush=True)
        sys.exit(3)
    n_final, _ = compact_jsonl(path, order)
    if n_final != len(order):
        die("compaction left %d records for %d items" % (n_final, len(order)))
    update_seal(rdir, reader, set_name, path, len(order),
                extra=OrderedDict([("thinking_variant", variant), ("mode", "TEST" if test else "FULL")]))
    print("D11-R FULL %s %s COMPLETE: %d readings sealed" % (reader, set_name, n_final), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", help="one of %s, or all (probe only)" % ", ".join(API_MODELS))
    ap.add_argument("--set", choices=list(ITEM_SETS), help="item set (full only)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="TEST: first N rows, written under _test/")
    args = ap.parse_args()

    if args.model != "all" and args.model not in API_MODELS:
        die("unknown model %r; choose from %s" % (args.model, ", ".join(API_MODELS)))
    api = Api()
    print("[api] host %s   models %s   key: from environment, never stored" % (api.host, ", ".join(API_MODELS)), flush=True)
    if args.probe:
        if args.set:
            die("--probe reads the first %d RDEV300 rows; --set is not accepted" % PROBE_ROWS)
        run_probe(api, list(API_MODELS) if args.model == "all" else [args.model])
    else:
        if args.model == "all" or not args.set:
            die("--full needs --model <one model> and --set <rdev300|dev200|eval458>")
        run_full(api, args.model, args.set, args.limit)


if __name__ == "__main__":
    main()
