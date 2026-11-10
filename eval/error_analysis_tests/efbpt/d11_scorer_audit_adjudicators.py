#!/usr/bin/env python3
"""
d11_scorer_audit_adjudicators.py — D11 RTG Amendment 5 A5.6: two further adjudicators
(API models) on the same 105 scorer-audit items, same rubric, blind. Agreement only:
they select nothing (the validated scorer stays the one chosen by the PRIMARY labels).

Login node, CPU. DEV200 audit items only; nothing under an eval458 directory is opened.
Edits no frozen file and does not edit d11_scorer_audit.py (its MD5 is pinned by the
readout script); A4.3 is recomputed by importing its functions.

Adjudicators: qwen3.8-max and deepseek-v4-pro. temperature 0, default thinking (no
thinking switch in the request), max_tokens 4000; ONE retry with enable_thinking=false
if the reply is empty. Transport errors (429 / 5xx / network) follow the D11-R retry
policy. 2 concurrent calls. One item per call:
  system = the rubric of docs/EFBPT_D11_RTG_AMENDMENT_4_NOTE.md N2 (R1-R4) in English,
           WITHOUT the item numbers the human adjudicator listed there;
  user   = the Urdu question + the text shown in docs/handoff/D11_SCORER_AUDIT_ITEMS.md
           (whole text, or first 300 / last 250 characters), exactly as the human saw it.
The model never receives arm, gold, qid, any extractor label or any other label.
Reply = one word: yes / no / none. Parsing: think tags removed; the reply stripped of
non-letters must be yes / no / none ("strict"); else the first alphabetic word if it is
one of the three ("lenient", counted); else "invalid" (counted; a category of its own
in the agreement statistics; never re-asked).

KEY HANDLING: URB_API_BASE / URB_API_KEY from the environment, else parsed from
~/.urbench_api_env; Authorization header only; every stored string scrubbed; never printed.

Outputs
  outputs/efbpt/d11/_audit/adjudicators/<model>/ replies.jsonl (raw replies), START.json, SEAL.json
  docs/handoff/D11_SCORER_AUDIT_LABELS_QWEN.json, _DEEPSEEK.json, _MAJORITY.json
  docs/handoff/D11_SCORER_AUDIT_ADJUDICATORS.md   pairwise agreement and Cohen's kappa for the three
      pairs (PRIMARY, qwen, deepseek), Fleiss' kappa, the items where either model disagrees with
      PRIMARY (item number and the three labels only), A4.3 recomputed with the majority label
      (sensitivity). Majority = the label given by at least two of the three; no two agree -> none.

  python eval/error_analysis_tests/efbpt/d11_scorer_audit_adjudicators.py --dry      # print the messages of item 1; no call
  python eval/error_analysis_tests/efbpt/d11_scorer_audit_adjudicators.py            # call (resumable), seal, report
  python eval/error_analysis_tests/efbpt/d11_scorer_audit_adjudicators.py --report   # report from the sealed replies only
"""
import argparse
import json
import os
import re
import socket
import sys
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from d11_common import md5_file, die, file_entry, git_state  # noqa: E402  (chdir to the repo root)
from d11r_common import scrub, scrub_obj, read_jsonl, API_BACKOFF_S, API_ATTEMPTS, md5_text  # noqa: E402

ITEMS_PATH = "docs/handoff/D11_SCORER_AUDIT_ITEMS.md"
ITEMS_MD5 = "7448bae2e400790f2b93c00359b26960"
PRIMARY_PATH = "docs/handoff/D11_SCORER_AUDIT_LABELS_PRIMARY.json"
PRIMARY_MD5 = "fb47ed80e34aac4c782bfc90d0f89fcb"
NOTE_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_4_NOTE.md"
NOTE_MD5 = "f9d690a7396de16612f40a879004c0b4"
AMEND5_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_5.md"
AMEND5_MD5 = "45c118de9804390d12853f231e7632a0"
OUT_ROOT = "outputs/efbpt/d11/_audit/adjudicators"
REPORT_PATH = "docs/handoff/D11_SCORER_AUDIT_ADJUDICATORS.md"
MODELS = OrderedDict([("qwen", "qwen3.8-max"), ("deepseek", "deepseek-v4-pro")])
LABEL_FILES = OrderedDict([("qwen", "docs/handoff/D11_SCORER_AUDIT_LABELS_QWEN.json"), ("deepseek", "docs/handoff/D11_SCORER_AUDIT_LABELS_DEEPSEEK.json"),
                           ("majority", "docs/handoff/D11_SCORER_AUDIT_LABELS_MAJORITY.json")])
N_ITEMS = 105
TEMPERATURE, MAX_TOKENS, CONCURRENCY, REQUEST_TIMEOUT_S = 0, 4000, 2, 600
RETRY_STATUS = {429, 500, 502, 503, 504}
LABELS = ("yes", "no", "none")
ENV_FILE = os.path.expanduser("~/.urbench_api_env")
HAAN = "".join(chr(c) for c in [0x06C1, 0x0627, 0x06BA])
NAHIN = "".join(chr(c) for c in [0x0646, 0x06C1, 0x06CC, 0x06BA])

# The rubric N2 R1-R4 of the Amendment 4 note, in English, without the item numbers listed there.
SYSTEM = (
    "You are adjudicating which verdict an answer text states. You are given a yes/no question written in Urdu and an answer "
    "written in Urdu by another system (the whole answer, or only its first 300 and last 250 characters). Decide which verdict "
    "the answer text itself states for the question: yes, no, or none. Do not answer the question yourself and do not judge "
    "whether the answer is correct; use only the text shown.\n"
    "Rubric:\n"
    "R1. If the text's statements about the question agree, give that verdict. An opening %(haan)s whose own sentence denies the "
    "proposition is read by the sentence.\n"
    "R2. If the same text contains an explicit yes-verdict and an explicit no-verdict, answer none.\n"
    "R3. If the text gives no verdict, a hedge, or a request for clarification, answer none.\n"
    "R4. If there is no %(haan)s / %(nahin)s word but the proposition is plainly affirmed or denied, give that verdict.\n"
    "Answer with exactly one word: yes, no, or none."
) % {"haan": HAAN, "nahin": NAHIN}


def check_inputs():
    for p, want in ((ITEMS_PATH, ITEMS_MD5), (PRIMARY_PATH, PRIMARY_MD5), (NOTE_PATH, NOTE_MD5), (AMEND5_PATH, AMEND5_MD5)):
        if "eval458" in p:
            die("refusing an eval458 path")
        got = md5_file(p)
        if got != want:
            die("%s MD5 %s != pinned %s" % (p, got, want))


def parse_items():
    """The 105 items exactly as written in the items file: (item number, question, [(caption, text), ...])."""
    lines = open(ITEMS_PATH, encoding="utf-8").read().split("\n")
    items, i = [], 0
    while i < len(lines):
        m = re.match(r"^## Item (\d+)$", lines[i])
        if not m:
            i += 1
            continue
        n = int(m.group(1))
        if not lines[i + 2].startswith("Question: "):
            die("item %d: no Question line" % n)
        q = lines[i + 2][len("Question: "):]
        i += 4
        blocks = []
        while i < len(lines) and lines[i].startswith("Generation"):
            cap, fence = lines[i], lines[i + 1]
            if not re.match(r"^`{3,}$", fence):
                die("item %d: no fence after %r" % (n, cap))
            j = i + 2
            while lines[j] != fence:
                j += 1
            blocks.append((cap, "\n".join(lines[i + 2:j])))
            i = j + 2
        items.append((n, q, blocks))
    if [n for n, _, _ in items] != list(range(1, N_ITEMS + 1)):
        die("items file: expected items 1..%d" % N_ITEMS)
    # round trip: the parsed pieces rebuild the file byte for byte
    out = ["# D11 scorer audit — items (Amendment 4 A4.2)", ""]
    for n, q, blocks in items:
        out += ["## Item %d" % n, "", "Question: %s" % q, ""]
        for k, (cap, text) in enumerate(blocks):
            fz = "```"
            while fz in "".join(t for _, t in blocks):
                fz += "`"
            out += [cap, fz, text, fz, ""]
    if "\n".join(out) != "\n".join(lines):
        die("items file: the parsed items do not rebuild the file")
    return items


def user_message(q, blocks):
    caps = [c for c, _ in blocks]
    if caps == ["Generation (whole text):"]:
        return "Question (Urdu): %s\n\nAnswer text (whole text):\n%s" % (q, blocks[0][1])
    if caps == ["Generation, first 300 characters:", "Generation, last 250 characters:"]:
        return ("Question (Urdu): %s\n\nAnswer text, first 300 characters:\n%s\n\n[... the middle of the answer is not shown ...]\n\n"
                "Answer text, last 250 characters:\n%s" % (q, blocks[0][1], blocks[1][1]))
    die("unexpected item layout %r" % caps)


def parse_reply(text):
    """(label, how): label in yes / no / none / invalid; how in strict / lenient / invalid / empty."""
    t = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S | re.I)
    if not t.strip():
        return "invalid", "empty"
    core = re.sub(r"[^a-z]", "", t.strip().lower())
    if core in LABELS:
        return core, "strict"
    m = re.search(r"[A-Za-z]+", t)
    if m and m.group(0).lower() in LABELS:
        return m.group(0).lower(), "lenient"
    return "invalid", "invalid"


def load_key_env():
    base, key = os.environ.get("URB_API_BASE", "").strip(), os.environ.get("URB_API_KEY", "").strip()
    if (not base or not key) and os.path.exists(ENV_FILE):
        for ln in open(ENV_FILE, encoding="utf-8"):
            m = re.match(r"^\s*(?:export\s+)?(URB_API_BASE|URB_API_KEY)\s*=\s*(.*?)\s*$", ln)
            if m:
                v = m.group(2).strip().strip("'\"")
                if m.group(1) == "URB_API_BASE" and not base:
                    base = v
                if m.group(1) == "URB_API_KEY" and not key:
                    key = v
    if not base or not key:
        die("URB_API_BASE / URB_API_KEY not found (environment or ~/.urbench_api_env)")
    return base, key


class Api:
    def __init__(self):
        import requests
        self.requests = requests
        base, key = load_key_env()
        self._key = key
        self.url = base.rstrip("/") + "/chat/completions"
        self.host = urlparse(base).netloc or "<unparsed>"
        self.secrets = (key,)
        self.local = threading.local()

    def _post(self, body):
        if not hasattr(self.local, "s"):
            self.local.s = self.requests.Session()
        attempts, err, status, resp = 0, None, None, None
        while attempts < API_ATTEMPTS:
            attempts += 1
            try:
                r = self.local.s.post(self.url, headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
                                      data=json.dumps(body), timeout=REQUEST_TIMEOUT_S)
                status = r.status_code
                try:
                    j = r.json()
                except Exception:
                    j = None
                if status == 200 and isinstance(j, dict) and j.get("choices"):
                    resp, err = j, None
                    break
                err = "HTTP %s: %s" % (status, scrub(r.text[:600], self.secrets))
                if status in RETRY_STATUS and attempts < API_ATTEMPTS:
                    time.sleep(API_BACKOFF_S[min(attempts - 1, len(API_BACKOFF_S) - 1)])
                    continue
                break
            except self.requests.RequestException as e:
                err = "network: %s" % scrub(str(e)[:300], self.secrets)
                if attempts < API_ATTEMPTS:
                    time.sleep(API_BACKOFF_S[min(attempts - 1, len(API_BACKOFF_S) - 1)])
                    continue
                break
        return resp, err, status, attempts

    def adjudicate(self, model, n, user):
        body = OrderedDict([("model", model), ("messages", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]),
                            ("temperature", TEMPERATURE), ("max_tokens", MAX_TOKENS)])
        t0 = time.time()
        calls = []
        resp, err, status, attempts = self._post(body)
        content = ((resp or {}).get("choices") or [{}])[0].get("message", {}).get("content") or "" if resp else ""
        calls.append(OrderedDict([("request_extra", {}), ("status", status), ("attempts", attempts), ("error", err), ("response", scrub_obj(resp, self.secrets))]))
        retried = False
        if resp is not None and parse_reply(content)[1] == "empty":
            # the one retry of the protocol: empty reply -> enable_thinking=false
            retried = True
            body2 = OrderedDict(body)
            body2["enable_thinking"] = False
            resp, err, status, attempts = self._post(body2)
            content = ((resp or {}).get("choices") or [{}])[0].get("message", {}).get("content") or "" if resp else ""
            calls.append(OrderedDict([("request_extra", {"enable_thinking": False}), ("status", status), ("attempts", attempts), ("error", err),
                                      ("response", scrub_obj(resp, self.secrets))]))
        label, how = parse_reply(content) if resp is not None else (None, None)
        usage = (resp or {}).get("usage") or {}
        return OrderedDict([("item", n), ("model", model), ("ok", resp is not None), ("label", label), ("parse", how), ("reply", scrub(content, self.secrets)),
                            ("retried_enable_thinking_false", retried), ("finish_reason", ((resp or {}).get("choices") or [{}])[0].get("finish_reason") if resp else None),
                            ("completion_tokens", usage.get("completion_tokens")), ("prompt_tokens", usage.get("prompt_tokens")),
                            ("latency_s", round(time.time() - t0, 2)), ("error", err), ("user_message_md5", md5_text(user)), ("calls", calls),
                            ("ts", time.strftime("%Y-%m-%d %H:%M:%S"))])


def model_dir(short):
    d = os.path.join(OUT_ROOT, MODELS[short])
    if "eval458" in d:
        die("refusing an eval458 path")
    return d


def ok_records(short):
    p = os.path.join(model_dir(short), "replies.jsonl")
    recs = OrderedDict()
    if os.path.exists(p):
        for r in read_jsonl(p):
            if r.get("ok"):
                recs[r["item"]] = r                     # the last ok record of an item
    return recs


def run_model(api, short, items):
    model = MODELS[short]
    d = model_dir(short)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "replies.jsonl")
    done = ok_records(short)
    todo = [(n, q, b) for n, q, b in items if n not in done]
    print("[%s] items %d, already ok %d, to do %d" % (model, len(items), len(done), len(todo)), flush=True)
    start_p = os.path.join(d, "START.json")
    if not os.path.exists(start_p):
        with open(start_p, "x", encoding="utf-8") as f:
            json.dump(OrderedDict([
                ("what", "D11 RTG A5.6 scorer-audit adjudicator (agreement only; selects nothing)"), ("model", model), ("host", api.host),
                ("temperature", TEMPERATURE), ("max_tokens", MAX_TOKENS), ("thinking", "default (no switch); one retry with enable_thinking=false on an empty reply"),
                ("concurrency", CONCURRENCY), ("system_message", SYSTEM), ("system_message_md5", md5_text(SYSTEM)),
                ("items_file", OrderedDict([("path", ITEMS_PATH), ("md5", ITEMS_MD5)])), ("note_md5", NOTE_MD5), ("amendment5_md5", AMEND5_MD5),
                ("script_md5", md5_file(os.path.abspath(__file__))), ("git", git_state()), ("hostname", socket.gethostname()),
                ("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))]), f, indent=2, ensure_ascii=False)
    lock = threading.Lock()
    n_done = [0]

    def work(it):
        n, q, blocks = it
        rec = api.adjudicate(model, n, user_message(q, blocks))
        with lock:
            with open(path, "ab") as f:
                f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            n_done[0] += 1
            if n_done[0] % 15 == 0 or not rec["ok"]:
                print("  [%s] %d/%d%s" % (model, n_done[0], len(todo), "" if rec["ok"] else "   item %d FAILED: %s" % (n, rec["error"])), flush=True)
        return rec["ok"]

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(as_completed([ex.submit(work, it) for it in todo]))
    done = ok_records(short)
    missing = [n for n, _, _ in items if n not in done]
    if missing:
        print("[%s] %d items without an ok reply; rerun to retry them" % (model, len(missing)), flush=True)
        return False
    seal = OrderedDict([("model", model), ("complete", True), ("n_items", len(done)), ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
                        ("script_md5", md5_file(os.path.abspath(__file__))), ("system_message_md5", md5_text(SYSTEM)),
                        ("files", OrderedDict((p, file_entry(p)) for p in (start_p, path)))])
    with open(os.path.join(d, "SEAL.json"), "w", encoding="utf-8") as f:
        json.dump(seal, f, indent=2, ensure_ascii=False)
    print("[%s] sealed: %s (%d replies)" % (model, os.path.join(d, "SEAL.json"), len(done)), flush=True)
    return True


# ----------------------------------------------------------------------------
# agreement statistics
# ----------------------------------------------------------------------------
def cohen(a, b):
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss(raters):
    n_r, n_i = len(raters), len(raters[0])
    cats = sorted(set(x for r in raters for x in r))
    counts = [[sum(1 for r in raters if r[i] == k) for k in cats] for i in range(n_i)]
    p_i = [(sum(c * c for c in row) - n_r) / (n_r * (n_r - 1)) for row in counts]
    p_bar = sum(p_i) / n_i
    p_k = [sum(row[j] for row in counts) / (n_i * n_r) for j in range(len(cats))]
    p_e = sum(p * p for p in p_k)
    return p_bar, (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def majority(labels3):
    c = Counter(x for x in labels3 if x in LABELS)
    top = c.most_common(1)
    return top[0][0] if top and top[0][1] >= 2 else "none"


def write_once(path, text):
    if os.path.exists(path):
        if open(path, encoding="utf-8").read() != text:
            die("%s exists with different content; not overwritten" % path)
        return "unchanged"
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    return "written"


def report():
    seals, recs = OrderedDict(), OrderedDict()
    for short in MODELS:
        d = model_dir(short)
        sp = os.path.join(d, "SEAL.json")
        if not os.path.exists(sp):
            die("%s missing: the %s replies are not sealed" % (sp, MODELS[short]))
        seal = json.load(open(sp, encoding="utf-8"))
        for p, want in seal["files"].items():
            got = file_entry(p)
            if (got["md5"], got["bytes"], got["lines"]) != (want["md5"], want["bytes"], want["lines"]):
                die("%s differs from its seal" % p)
        seals[short] = seal
        recs[short] = ok_records(short)
        if sorted(recs[short]) != list(range(1, N_ITEMS + 1)):
            die("%s: replies do not cover items 1..%d" % (MODELS[short], N_ITEMS))
    primary = {int(k): v for k, v in json.load(open(PRIMARY_PATH, encoding="utf-8")).items()}
    lab = OrderedDict([("PRIMARY", [primary[i] for i in range(1, N_ITEMS + 1)])])
    for short in MODELS:
        lab[short] = [recs[short][i]["label"] for i in range(1, N_ITEMS + 1)]
    maj = [majority([lab[k][i] for k in lab]) for i in range(N_ITEMS)]

    L = []
    w = L.append
    w("# D11 scorer audit — two further adjudicators (Amendment 5 A5.6)")
    w("")
    w("Agreement only: these labels select nothing; the validated scorer is the one chosen by the PRIMARY labels (E3). "
      "Written by `eval/error_analysis_tests/efbpt/d11_scorer_audit_adjudicators.py` (MD5 `%s`). Items `%s` (MD5 `%s`), %d items; "
      "PRIMARY labels MD5 `%s`. Each model got the rubric N2 R1–R4 in English as the system message (MD5 `%s`; the item numbers of the note are not in it) "
      "and one item per call (the Urdu question and the text shown to the human adjudicator); it did not receive arm, gold, qid or any label. "
      "temperature %d, default thinking, max_tokens %d, one retry with enable_thinking=false on an empty reply."
      % (md5_file(os.path.abspath(__file__)), ITEMS_PATH, ITEMS_MD5, N_ITEMS, PRIMARY_MD5, md5_text(SYSTEM), TEMPERATURE, MAX_TOKENS))
    w("")
    w("| adjudicator | yes | no | none | invalid | strict one-word replies | lenient | retried with enable_thinking=false | raw replies (sealed) |")
    w("|---|---|---|---|---|---|---|---|---|")
    c = Counter(lab["PRIMARY"])
    w("| PRIMARY (adjudicator A) | %d | %d | %d | %d | — | — | — | `%s` |" % (c["yes"], c["no"], c["none"], c["invalid"], PRIMARY_PATH))
    for short, model in MODELS.items():
        c = Counter(lab[short])
        rs = list(recs[short].values())
        w("| %s | %d | %d | %d | %d | %d | %d | %d | `%s` (MD5 `%s`) |"
          % (model, c["yes"], c["no"], c["none"], c["invalid"], sum(1 for r in rs if r["parse"] == "strict"), sum(1 for r in rs if r["parse"] == "lenient"),
             sum(1 for r in rs if r["retried_enable_thinking_false"]), os.path.join(model_dir(short), "replies.jsonl"),
             seals[short]["files"][os.path.join(model_dir(short), "replies.jsonl")]["md5"]))
    c = Counter(maj)
    w("| majority of the three | %d | %d | %d | %d | — | — | — | — |" % (c["yes"], c["no"], c["none"], c["invalid"]))
    w("")
    w("## Pairwise agreement and Cohen's kappa")
    w("")
    w("| pair | items agreeing | agreement % | Cohen's kappa |")
    w("|---|---|---|---|")
    names = list(lab)
    disp = {"PRIMARY": "PRIMARY", "qwen": MODELS["qwen"], "deepseek": MODELS["deepseek"]}
    for i in range(3):
        for j in range(i + 1, 3):
            po, k = cohen(lab[names[i]], lab[names[j]])
            w("| %s vs %s | %d / %d | %.2f | %.3f |" % (disp[names[i]], disp[names[j]], round(po * N_ITEMS), N_ITEMS, 100.0 * po, k))
    pb, fk = fleiss([lab[k] for k in names])
    w("")
    w("Fleiss' kappa (three adjudicators, %d items): **%.3f** (mean pairwise agreement per item %.2f %%). All three agree on %d items."
      % (N_ITEMS, fk, 100.0 * pb, sum(1 for i in range(N_ITEMS) if len({lab[k][i] for k in names}) == 1)))
    w("")
    w("## Items where either model disagrees with PRIMARY")
    w("")
    w("| item | PRIMARY | %s | %s |" % (MODELS["qwen"], MODELS["deepseek"]))
    w("|---|---|---|---|")
    nd = 0
    for i in range(N_ITEMS):
        if lab["qwen"][i] != lab["PRIMARY"][i] or lab["deepseek"][i] != lab["PRIMARY"][i]:
            nd += 1
            w("| %d | %s | %s | %s |" % (i + 1, lab["PRIMARY"][i], lab["qwen"][i], lab["deepseek"][i]))
    w("")
    w("%d items; on %d of them both models give the same label, against PRIMARY (the majority label then differs from PRIMARY)."
      % (nd, sum(1 for i in range(N_ITEMS) if maj[i] != lab["PRIMARY"][i])))
    w("")

    # ---- A4.3 with the majority label (sensitivity) ----
    import d11_scorer_audit as A
    key = json.load(open(A.KEY_PATH, encoding="utf-8"), object_pairs_hook=OrderedDict)
    items, _ = A.current_items(key)
    w("## A4.3 recomputed with the majority label (sensitivity; selects nothing)")
    w("")
    w("Majority = the label given by at least two of PRIMARY, %s, %s; no two agree → none. For comparison, the PRIMARY column repeats the table of the audit."
      % (MODELS["qwen"], MODELS["deepseek"]))
    w("")
    tabs = OrderedDict()
    for name, labels in (("PRIMARY", {i + 1: lab["PRIMARY"][i] for i in range(N_ITEMS)}), ("majority", {i + 1: maj[i] for i in range(N_ITEMS)})):
        tabs[name] = A.agreement_table(key["strata"], items, labels)
    w("| arm | extractor | differ rows right (PRIMARY) | differ rows right (majority) | agree rows right (PRIMARY) | agree rows right (majority) | stratified % (PRIMARY) | stratified % (majority) |")
    w("|---|---|---|---|---|---|---|---|")
    for arm in tabs["PRIMARY"][0]:
        for e in A.EXTRACTORS:
            p, m = tabs["PRIMARY"][0][arm][e], tabs["majority"][0][arm][e]
            w("| %s | %s | %d / %d | %d / %d | %d / %d | %d / %d | %.2f | %.2f |" % (arm, e, p["differ"][0], p["differ"][1], m["differ"][0], m["differ"][1],
                                                                         p["agree"][0], p["agree"][1], m["agree"][0], m["agree"][1],
                                                                         100.0 * float(p["stratified"]), 100.0 * float(m["stratified"])))
    w("")
    w("| extractor | items right of 65 differ (PRIMARY / majority) | items right of 40 agree (PRIMARY / majority) | mean stratified agreement % (PRIMARY / majority) |")
    w("|---|---|---|---|")
    for e in A.EXTRACTORS:
        row = []
        for name in ("PRIMARY", "majority"):
            t, mean, _ = tabs[name]
            row.append((sum(t[a][e]["differ"][0] for a in t), sum(t[a][e]["agree"][0] for a in t), 100.0 * float(mean[e])))
        w("| %s | %d / %d | %d / %d | %.2f / %.2f |" % (e, row[0][0], row[1][0], row[0][1], row[1][1], row[0][2], row[1][2]))
    w("")
    w("Highest mean under the majority label (ties → E2, E3, E1): %s. Under PRIMARY: %s (the validated scorer; unchanged by this table)." % (tabs["majority"][2], tabs["PRIMARY"][2]))
    w("")
    text = "\n".join(L)
    for short in MODELS:
        st = write_once(LABEL_FILES[short], json.dumps(OrderedDict((str(i + 1), lab[short][i]) for i in range(N_ITEMS)), ensure_ascii=False) + "\n")
        print("[%s] %s" % (st, LABEL_FILES[short]))
    st = write_once(LABEL_FILES["majority"], json.dumps(OrderedDict((str(i + 1), maj[i]) for i in range(N_ITEMS)), ensure_ascii=False) + "\n")
    print("[%s] %s" % (st, LABEL_FILES["majority"]))
    print("\n" + text)
    print("[%s] %s" % (write_once(REPORT_PATH, text), REPORT_PATH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print the system message and the user message of item 1; no API call")
    ap.add_argument("--report", action="store_true", help="report from the sealed replies only; no API call")
    args = ap.parse_args()
    check_inputs()
    items = parse_items()
    for text, want in (("yes", "yes"), ("No.", "no"), (" **None** ", "none"), ("<think>x yes</think>\nno", "no"), ("Yes, because", "yes"), ("", "invalid"), ("maybe", "invalid")):
        if parse_reply(text)[0] != want:
            die("parse_reply selftest: %r" % text)
    if args.dry:
        print("items parsed: %d (whole text %d, head + tail %d); system message md5 %s" % (len(items), sum(1 for _, _, b in items if len(b) == 1),
                                                                                         sum(1 for _, _, b in items if len(b) == 2), md5_text(SYSTEM)))
        print("----- SYSTEM -----\n%s\n----- USER (item 1) -----\n%s\n-----" % (SYSTEM, user_message(items[0][1], items[0][2])))
        return
    if not args.report:
        api = Api()
        print("[api] host %s; models %s; temperature %s; max_tokens %d; default thinking; concurrency %d; key: never stored or printed"
              % (api.host, ", ".join(MODELS.values()), TEMPERATURE, MAX_TOKENS, CONCURRENCY), flush=True)
        ok = [run_model(api, short, items) for short in MODELS]
        if not all(ok):
            die("not every item has a reply; rerun")
    report()


if __name__ == "__main__":
    main()
