#!/usr/bin/env python3
"""
d11_teacher.py — D11 RTG Job T: v1 teacher facts from the sealed D11-R reading.

Contract: docs/EFBPT_D11_RTG_FREEZE.md §3 (MD5 pinned in d11_common) as amended
by docs/EFBPT_D11_RTG_AMENDMENT_1.md A1.1 / A1.3 / A1.5, on the D11-O v1 path
(d11o_teacher.py: D10 T1 prompt pair with the two English-input edits, strict
parser, T2 verdict filter, max_tokens 512, D10 §2 engine). Everything frozen is
IMPORTED from d11o_teacher / d10_teacher, never re-implemented; the ONLY change
is the English input: {question_en} = the sealed D11-R reading of that qid
(freeze §3), nothing else from the row.

  --reader R1|R2|R3      api_qwen3.8-max | api_deepseek-v4-pro | L1_llama70b_awq (freeze §2)
  --set dev200|eval458   eval458 only if docs/D11_GO_EVAL458.md exists (§8)
  --mode TEST|FULL       TEST = first 5 rows, outputs under outputs/efbpt/d11/_test/teacher/<set>/<reader>/
  --preflight-only       CPU, no model, writes nothing

Leak guards (A1.1): (a) the prompt with the reading slot blanked carries no
question_en, gold fact, evidence id or required title (FATAL); (b) the reading
carries no OTHER row's question_en, no gold fact, no evidence id (FATAL);
reading == own gold question_en counted; required titles in the reading
reported. V4: reading file MD5 = R_SEAL (= freeze §2 for DEV200), one ok
record per qid.

Outputs: outputs/efbpt/d11/teacher/<set>/<R1|R2|R3>/ t_facts.jsonl, T_START.json, T_SEAL.json.
Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d11_teacher.py --reader R1 --set dev200 --mode TEST --preflight-only
  python eval/error_analysis_tests/efbpt/d11_teacher.py --reader R1 --set dev200 --mode FULL
"""
import argparse
import json
import os
import socket
import sys
import time
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from d11_common import (  # noqa: E402
    READERS, SETS, TEST_N, out_dir, verify_contracts, require_eval458_gate, load_set_rows,
    needles_for, load_readings, guard_out_path, file_entry, die, md5_file, git_state,
)
import d11o_teacher as O                  # noqa: E402  the frozen v1 oracle-reading path
from efbpt_eval_dev200 import INSTRUCTION_PATH, INSTRUCTION_MD5, load_instruction  # noqa: E402
from d8_reachable_ceiling import leak_guard, latin_fraction, md5_text, pkg_versions  # noqa: E402

VERSION = "v1"                            # freeze §3: the v1 path only
SPEC = O.VERSIONS[VERSION]
WRITTEN = []
OUT_DIR = None


def write_jsonl(path, records, expect_n):
    ap = guard_out_path(OUT_DIR, path)
    if len(records) != expect_n:
        die("%s: have %d records, expected %d" % (path, len(records), expect_n))
    with open(ap, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    nd = sum(1 for _ in open(ap, "r", encoding="utf-8"))
    if nd != len(records):
        die("%s: wrote %d records, disk has %d lines" % (path, len(records), nd))
    print("  [verified] %d lines -> %s" % (nd, path), flush=True)
    WRITTEN.append(path)


def write_json(path, obj):
    ap = guard_out_path(OUT_DIR, path)
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    json.load(open(ap, encoding="utf-8"))
    print("  [verified] wrote %s" % path, flush=True)
    WRITTEN.append(path)


def preflight(reader_key, set_name, limit):
    print("=" * 78)
    print("D11 RTG JOB T PREFLIGHT  reader %s  set %s  (freeze §3, A1.1; D11-O v1 path; D10 freeze §2)" % (reader_key, set_name))
    print("=" * 78, flush=True)
    contracts = verify_contracts()
    for label, path, want in (("d11o_freeze", O.D11O_FREEZE_PATH, O.D11O_FREEZE_MD5),
                              ("d10_freeze", O.FREEZE_PATH, O.FREEZE_MD5)):
        got = md5_file(path)
        if got != want:
            die("%s MD5 %s != pinned %s" % (path, got, want))
        contracts[label] = OrderedDict([("path", path), ("md5", got)])
        print("[md5] %-18s %s  %s" % (label, got, path))
    for key, (path, want) in O.AMEND_V2.items():
        got = md5_file(path)
        if got != want:
            die("%s MD5 %s != pinned %s" % (path, got, want))
        contracts["d10_" + key] = OrderedDict([("path", path), ("md5", got)])
    scripts = OrderedDict((os.path.relpath(m.__file__), md5_file(m.__file__)) for m in (O, O.T1V1, O.T1V2))
    scripts[os.path.relpath(__file__)] = md5_file(os.path.abspath(__file__))
    scripts[os.path.relpath(os.path.join(_HERE, "d11_common.py"))] = md5_file(os.path.join(_HERE, "d11_common.py"))
    for p, m in scripts.items():
        print("[md5] %-18s %s  %s" % ("src", m, p))
    load_instruction()
    print("[ok] Urdu instruction MD5 %s verified (not used by the teacher)" % INSTRUCTION_MD5)

    rows = load_set_rows(set_name)
    for r in rows:
        if not r["question_en"].strip():
            die("qid %s: empty gold question_en (needed as a leak needle)" % r["qid"])
    readings, rinfo = load_readings(reader_key, set_name, rows)
    sub = rows[:limit] if limit else rows

    system, user_t = O.oracle_prompts(O.T1V1.T_SYSTEM, O.T1V1.T_USER_TEMPLATE)
    print("[prompt] v1: %s; edits applied: %r -> %r; %r -> %r; {question_en} := the sealed D11-R reading"
          % (SPEC["source"], O.EDIT_SYSTEM[0], O.EDIT_SYSTEM[1], O.EDIT_USER[0], O.EDIT_USER[1]))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(O.TEACHER_PATH, trust_remote_code=True)

    def render(system_, user_):
        msgs = [{"role": "system", "content": system_}, {"role": "user", "content": user_}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        if tok(text, add_special_tokens=False)["input_ids"] != ids:
            die("chat template: stored prompt text does not re-tokenize to the generated ids")
        return text, ids

    p_facts = [render(system, user_t.format(question_en=readings[r["qid"]])) for r in sub]
    mx = SPEC["max_tokens"]
    tl = sorted(len(ids) for _, ids in p_facts)
    print("[len] TR/facts n=%d tokens min=%d median=%d max=%d (+%d new <= %d)"
          % (len(p_facts), tl[0], tl[len(tl) // 2], tl[-1], mx, O.VLLM_ARGS["max_model_len"]))
    if tl[-1] + mx > O.VLLM_ARGS["max_model_len"]:
        die("prompts would exceed max_model_len")

    # ---- leak guards (freeze §3 as read by A1.1) ----
    needles = needles_for(rows, set_name)
    blank, _ = render(system, user_t.format(question_en="<READING>"))
    leak_guard("T/template", [blank], needles, fatal_gold_facts=True)
    if needles["title_re"].search(blank):
        die("LEAK: the teacher prompt template contains a required title %r" % needles["title_re"].search(blank).group(0))
    print("[leak-guard] (a) template with the reading slot blanked: question_en, gold facts, evidence ids, %d titles clean" % len(needles["titles"]))
    exact, title_hits = [], OrderedDict()
    for r in sub:
        rd = readings[r["qid"]]
        for q in needles["question_en"]:
            if q in rd and q != r["question_en"]:
                die("LEAK: qid %s reading contains another row's question_en %r" % (r["qid"], q))
        for s in needles["facts"]:
            if s in rd:
                die("LEAK: qid %s reading contains gold fact %r" % (r["qid"], s))
        m = needles["evidence_re"].search(rd)
        if m:
            die("LEAK: qid %s reading contains evidence id %r" % (r["qid"], m.group(0)))
        if rd.strip() == r["question_en"].strip():
            exact.append(r["qid"])
        th = sorted(set(needles["title_re"].findall(rd)))
        if th:
            title_hits[r["qid"]] = th
    print("[leak-guard] (b) %d readings: no other-row question_en, no gold fact, no evidence id -> PASS; "
          "equal to own gold question_en %d (reported); readings containing a required title %d (reported, reading quality)"
          % (len(sub), len(exact), len(title_hits)))
    # the prompt is system + template + reading and nothing else from the row
    for r, (t, _) in zip(sub, p_facts):
        if t.count("Question: %s\n" % readings[r["qid"]]) != 1:
            die("qid %s: prompt lacks its reading exactly once" % r["qid"])
        if t != render(system, user_t.format(question_en=readings[r["qid"]]))[0]:
            die("qid %s: prompt is not a pure function of system + template + reading" % r["qid"])
    print("[ok] each prompt = system + user template + its own sealed reading; nothing else from the row")
    O.log_full_prompt("TR facts, reader %s, row 0, qid %s" % (reader_key, sub[0]["qid"]), p_facts[0][0])

    meta = OrderedDict([
        ("contracts", contracts), ("scripts", scripts), ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("set", set_name), ("reader", rinfo),
        ("prompts", OrderedDict([
            ("version", VERSION), ("arm", "T_facts_%s" % reader_key), ("source", SPEC["source"]),
            ("teacher_input", "the sealed D11-R %s reading of each qid (freeze §3); nothing else from the row" % reader_key),
            ("edits", OrderedDict([("system", list(O.EDIT_SYSTEM)), ("user", list(O.EDIT_USER))])),
            ("t_system_md5", md5_text(system)), ("t_user_template_md5", md5_text(user_t)),
            ("keys", SPEC["keys"]), ("max_tokens", mx), ("repairs", SPEC["repairs"])])),
        ("verdict_filter", OrderedDict([("regex", O.T1V2.VERDICT_RE.pattern), ("yes_no_regex", O.T1V2.YESNO_RE.pattern),
                                        ("case_insensitive", True), ("max_facts", O.MAX_FACTS), ("applied_to", "facts only")])),
        ("leak_guard", OrderedDict([
            ("rule", "A1.1: (a) template blanked: question_en / gold facts / evidence ids / titles FATAL; "
                     "(b) reading: other-row question_en / gold facts / evidence ids FATAL; own equality counted; titles reported"),
            ("prompts_checked", len(p_facts)), ("fatal_hits", 0),
            ("readings_equal_own_question_en", OrderedDict([("n", len(exact)), ("qids", exact)])),
            ("readings_with_required_title", OrderedDict([("n", len(title_hits)), ("hits", title_hits)])),
            ("needles", OrderedDict([("question_en", len(needles["question_en"])), ("facts", len(needles["facts"])),
                                     ("evidence_ids", len(needles["evidence_ids"])), ("titles", len(needles["titles"]))]))])),
    ])
    print("[preflight] PASS", flush=True)
    return sub, p_facts, meta, readings


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=list(READERS))
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--mode", required=True, choices=("TEST", "FULL"))
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    is_test = args.mode == "TEST"
    limit = TEST_N if is_test else None
    require_eval458_gate(args.set)
    OUT_DIR = out_dir("teacher", args.set, is_test, args.reader)

    sub, p_facts, meta, readings = preflight(args.reader, args.set, limit)
    if args.preflight_only:
        print("\n[preflight-only] done. No files written.")
        return
    print("\n[mode] %s  rows=%d  out=%s" % ("TEST — NOT A RESULT" if is_test else "FULL", len(sub), OUT_DIR))
    if not is_test and len(sub) != len(readings):
        die("FULL run must cover every row of the set")
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")

    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != O.VLLM_VERSION:
        die("vLLM %s found, freeze requires %s" % (vllm.__version__, O.VLLM_VERSION))
    versions = pkg_versions()
    versions["vllm"] = vllm.__version__
    start = OrderedDict([
        ("job", "T"), ("experiment", "D11 RTG"), ("reader_key", args.reader), ("version", VERSION),
        ("mode", args.mode), ("args", vars(args)), ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("hostname", socket.gethostname()), ("slurm_job_id", job_id),
        ("n_rows", len(sub)), ("qids", [r["qid"] for r in sub]),
        ("teacher_path", O.TEACHER_PATH), ("vllm_args", O.VLLM_ARGS), ("sampling", O.SAMPLING), ("versions", versions),
    ])
    start.update(meta)
    print("\n[START]")
    write_json(os.path.join(OUT_DIR, "T_START.json"), start)

    print("\n[load] %s ..." % O.TEACHER_PATH, flush=True)
    t0 = time.time()
    llm = LLM(model=O.TEACHER_PATH, **O.VLLM_ARGS)
    print("[load] %.0fs" % (time.time() - t0), flush=True)
    params = SamplingParams(temperature=O.SAMPLING["temperature"], top_p=O.SAMPLING["top_p"],
                            seed=O.SAMPLING["seed"], max_tokens=SPEC["max_tokens"])
    print("\n" + "=" * 70)
    print("TR — v1 teacher facts from the %s reading  (n=%d, max_tokens=%d, greedy, max_num_seqs=%d)"
          % (args.reader, len(p_facts), SPEC["max_tokens"], O.VLLM_ARGS["max_num_seqs"]))
    print("=" * 70, flush=True)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": ids} for _, ids in p_facts], params, use_tqdm=False)
    if len(outs) != len(p_facts):
        die("vLLM returned %d outputs for %d prompts" % (len(outs), len(p_facts)))
    el = time.time() - t0
    print("  %d rows in %.1fs (%.1fs/row)" % (len(outs), el, el / max(len(outs), 1)), flush=True)

    t_facts = []
    n_parse_ok = n_removed_total = n_repaired = 0
    for r, (text, _), o in zip(sub, p_facts, outs):
        c = o.outputs[0]
        raw, n_new, fin = c.text, len(c.token_ids), c.finish_reason
        ok, q_en, unknown, subqs, ents, facts_raw, repaired = O.parse(VERSION, raw)
        kept, removed = O.verdict_filter(facts_raw) if ok else ([], [])
        n_parse_ok += int(ok)
        n_repaired += int(bool(repaired))
        n_removed_total += len(removed)
        t_facts.append(OrderedDict([
            ("qid", r["qid"]), ("arm", "T_facts_%s" % args.reader), ("teacher_version", VERSION),
            ("reader", args.reader), ("set", args.set),
            ("teacher_input_reading", readings[r["qid"]]),          # the sealed reading (audit)
            ("prompt", text), ("raw", raw), ("parse_ok", ok), ("repaired", repaired),
            ("question_en", q_en), ("unknown_terms", unknown), ("sub_questions", subqs), ("entities", ents),
            ("facts_parsed", facts_raw), ("facts", kept), ("removed", removed),
            ("n_facts", len(kept)), ("n_facts_parsed", len(facts_raw)), ("n_removed", len(removed)),
            ("n_unknown_terms", len(unknown)), ("n_sub_questions", len(subqs)),
            ("truncated", fin == "length"), ("finish_reason", fin), ("n_gen_tokens", n_new),
        ]))
    nf = sorted(e["n_facts"] for e in t_facts)
    print("  parse_ok %d/%d (%.1f%%)   facts per row min=%d median=%d max=%d  empty=%d   removed (V5) %d   truncated %d"
          % (n_parse_ok, len(t_facts), 100.0 * n_parse_ok / len(t_facts), nf[0], nf[len(nf) // 2], nf[-1],
             sum(1 for x in nf if x == 0), n_removed_total, sum(1 for e in t_facts if e["truncated"])))
    print("  facts-per-row distribution %s" % dict(sorted(Counter(nf).items())))
    print("  sample (row 0): reading %r" % t_facts[0]["teacher_input_reading"])
    print("  sample (row 0): entities %s" % t_facts[0]["entities"])
    for f in t_facts[0]["facts"]:
        print("    - %s" % f)
    frac, letters, latin = latin_fraction([f for e in t_facts for f in e["facts"]])
    print("  Latin fraction of kept facts %.4f (%d/%d letters)" % (frac, latin, letters))
    write_jsonl(os.path.join(OUT_DIR, "t_facts.jsonl"), t_facts, expect_n=len(sub))
    parse_pct = 100.0 * n_parse_ok / len(t_facts)
    print("\nV4(D10) teacher JSON parse rate %d/%d = %.1f%% (min %.0f%%) -> %s"
          % (n_parse_ok, len(t_facts), parse_pct, O.V4_PARSE_MIN_PCT, "ok" if parse_pct >= O.V4_PARSE_MIN_PCT else "FORMAT FLAG"))
    print("V5(D10) verdict-filter removed lines %d (max %d) -> %s"
          % (n_removed_total, O.V5_REMOVED_MAX, "ok" if n_removed_total <= O.V5_REMOVED_MAX else "TEACHER-ANSWERING FLAG"))
    print("finish reasons: %s" % dict(Counter(e["finish_reason"] for e in t_facts)))

    files = OrderedDict((p, file_entry(p)) for p in OrderedDict.fromkeys(WRITTEN))
    seal = OrderedDict([
        ("job", "T"), ("experiment", "D11 RTG"), ("reader_key", args.reader), ("reader", READERS[args.reader]["reader"]),
        ("set", args.set), ("version", VERSION), ("mode", args.mode), ("slurm_job_id", job_id),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("script_md5", meta["scripts"][os.path.relpath(__file__)]),
        ("d11_freeze_md5", meta["contracts"]["d11_freeze"]["md5"]), ("d11_amendment1_md5", meta["contracts"]["d11_amendment_1"]["md5"]),
        ("d11o_freeze_md5", meta["contracts"]["d11o_freeze"]["md5"]), ("d10_freeze_md5", meta["contracts"]["d10_freeze"]["md5"]),
        ("reading_file", OrderedDict([("path", meta["reader"]["file"]), ("md5", meta["reader"]["md5"])])),
        ("n_rows", len(sub)), ("t1_parse_ok", n_parse_ok), ("t1_repaired_rows", n_repaired), ("t2_removed_total", n_removed_total),
        ("files", files)])
    write_json(os.path.join(OUT_DIR, "T_SEAL.json"), seal)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD11 RTG JOB T (%s, %s) COMPLETE. Sealed %d files." % (args.reader, args.set, len(files)))


if __name__ == "__main__":
    main()
