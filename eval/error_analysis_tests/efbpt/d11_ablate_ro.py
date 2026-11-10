#!/usr/bin/env python3
"""
d11_ablate_ro.py — D11 RTG Amendment 3 A3.6: DEV200 baseline ablation RO / RO_gold.

Descriptive, not gated, DEV200 only (no eval458 use: the set is fixed to dev200
and every path is checked). Contract: docs/EFBPT_D11_RTG_AMENDMENT_3.md A3.6
(MD5 pinned below).

  RO      = frozen student, layout A, plus ONE line directly after the Urdu
            question:  سوال کا انگریزی ترجمہ: {R1 reading}
  RO_gold = the same with the gold question_en (oracle translate-test ceiling)
No facts in either arm. V2' probe (5 sealed A_r2 rows regenerated, byte-identical)
is FATAL. Exact two-sided McNemar against the sealed arms: RO vs A_r2, TR vs RO,
RO_gold vs A_r2, TO1 vs RO_gold. Predictions (A3.6): RO - A_r2 in [0, +6];
RO_gold - A_r2 in [+1, +8]; TR - RO >= +2; TO1 - RO_gold >= +5. Unparsed,
truncation and Devanagari-answer counts reported.

Nothing frozen is edited; everything frozen is IMPORTED:
  DEV200 rows, R1 readings (MD5 = freeze §2), leak needles      <- d11_common
  model path, SYSTEM_MESSAGE, instruction (+MD5), extractor     <- efbpt_eval_dev200
  user layout "A"                                               <- d1_eval_arms.build_user
  secondary extractor, Devanagari strings                       <- d1_score_dual
  score()                                                       <- d3_oracle_retrieval
  generate_all (batch size 1)                                   <- d4_extract_facts
  paired exact McNemar                                          <- d5_balanced_extract
  leak guard, package versions                                  <- d8_reachable_ceiling
  student loader (nf4, as every D11 student job)                <- d11_student.load_student_model
The RO user message is build_user(instruction, row', None, "A") with
row'["question_ur"] = question_ur + "\\n" + the line; it is asserted to equal the
layout-A user message + "\\n" + the line, and the layout-A prompt is asserted
byte-identical to the sealed D10 A_r2 prompt of the row.

One row per generate call (batch size 1); each finished row is appended and
fsynced; a restarted job resumes from its last completed row. Per-arm seal.
Outputs: outputs/efbpt/d11/ablate_ro/dev200/ d11_armRO.jsonl, d11_armRO_gold.jsonl,
S_START.json, S_SEAL_RO.json, S_SEAL_RO_gold.json, d11_ablate_ro_summary.json, S_SEAL.json.

  python eval/error_analysis_tests/efbpt/d11_ablate_ro.py --preflight-only   # CPU: checks + one full RO prompt
  python eval/error_analysis_tests/efbpt/d11_ablate_ro.py                    # GPU job
"""
import argparse
import json
import os
import socket
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from d11_common import (  # noqa: E402
    verify_contracts, load_set_rows, needles_for, load_readings, file_entry, die, md5_file, git_state, read_jsonl,
    D10_A_R2, OUT_ROOT, FORBIDDEN_WRITE_DIRS,
)
from efbpt_eval_dev200 import (  # noqa: E402
    MODEL_PATH, SYSTEM_MESSAGE, MAX_NEW_TOKENS, INSTRUCTION_PATH, INSTRUCTION_MD5, extract_answer, load_instruction,
)
from d1_eval_arms import build_user  # noqa: E402
from d1_score_dual import extract_answer_secondary, DEV_YES, DEV_NO  # noqa: E402
from d3_oracle_retrieval import score  # noqa: E402
from d4_extract_facts import generate_all  # noqa: E402
from d5_balanced_extract import mcnemar  # noqa: E402
from d8_reachable_ceiling import leak_guard, pkg_versions  # noqa: E402
from d11_student import load_student_model  # noqa: E402  the D11 student loader, unchanged

SET = "dev200"                                   # A3.6: DEV200 only
AMEND3_PATH = "docs/EFBPT_D11_RTG_AMENDMENT_3.md"
AMEND3_MD5 = "39cbac213583c9d08ab3be7c07a97f3c"
# A3.6: the one added line. Built from codepoints (as the frozen extractor's strings) and asserted against the amendment text.
RO_LABEL = "".join(chr(c) for c in [0x0633, 0x0648, 0x0627, 0x0644, 0x0020, 0x06A9, 0x0627, 0x0020,
                                    0x0627, 0x0646, 0x06AF, 0x0631, 0x06CC, 0x0632, 0x06CC, 0x0020,
                                    0x062A, 0x0631, 0x062C, 0x0645, 0x06C1, 0x003A, 0x0020])
ARMS = ("RO", "RO_gold")                         # run in this order
OUT_DIR = os.path.join(OUT_ROOT, "ablate_ro", SET)
BATCH_SIZE = 1
CONTEXT_WINDOW = 40960
V2_PROBE_N = 5
# sealed DEV200 arms the ablation is paired with (MD5s as in docs/handoff/D11_FULL_READOUT.md §1)
SEALED = OrderedDict([
    ("A_r2", D10_A_R2),
    ("TR", ("outputs/efbpt/d11/student/dev200/d11_armTR.jsonl", "bab987cc492012f4e32dbcd348be9668")),
    ("TO1", ("outputs/efbpt/d11o/student/d11o_armTO1.jsonl", "6548b408a68a8e94bb65b9ca6d023fdd")),
])
PAIRS = (("RO", "A_r2"), ("TR", "RO"), ("RO_gold", "A_r2"), ("TO1", "RO_gold"))
PREDICTIONS = OrderedDict([                      # A3.6, pp
    ("RO - A_r2 in [0, +6]", ("RO", "A_r2", 0.0, 6.0)),
    ("RO_gold - A_r2 in [+1, +8]", ("RO_gold", "A_r2", 1.0, 8.0)),
    ("TR - RO >= +2", ("TR", "RO", 2.0, None)),
    ("TO1 - RO_gold >= +5", ("TO1", "RO_gold", 5.0, None)),
])


def out_path(name):
    """A path under OUT_DIR; never an eval458 path, never a sealed directory."""
    p = os.path.join(OUT_DIR, name)
    ap = os.path.abspath(p)
    if "eval458" in ap or not ap.startswith(os.path.abspath(OUT_DIR) + os.sep):
        die("refusing to write %s" % p)
    for forb in FORBIDDEN_WRITE_DIRS:
        if ap.startswith(os.path.abspath(forb) + os.sep):
            die("refusing to write under %s" % forb)
    return p


def write_json(name, obj):
    p = out_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    json.load(open(p, encoding="utf-8"))
    print("  [verified] wrote %s" % p, flush=True)
    return p


def record_line(rec):
    return (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")


def read_rows(path, repair=False):
    """Completed rows (each a newline-terminated JSON line). repair=True cuts a torn tail left by a crashed job."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        b = f.read()
    end = b.rfind(b"\n") + 1
    if end != len(b):
        if not repair:
            die("%s ends with a torn line" % path)
        print("[resume] %s: torn tail of %d bytes cut off" % (path, len(b) - end), flush=True)
        with open(path, "r+b") as f:
            f.truncate(end)
            f.flush()
            os.fsync(f.fileno())
        b = b[:end]
    return [json.loads(l.decode("utf-8"), object_pairs_hook=OrderedDict) for l in b.split(b"\n") if l]


def has_devanagari_answer(text):
    return any(s in text for s in DEV_YES + DEV_NO)


def preflight():
    print("=" * 78)
    print("D11 RTG A3.6 — DEV200 ablation RO / RO_gold: PREFLIGHT (CPU, no model)")
    print("=" * 78, flush=True)
    contracts = verify_contracts()
    got = md5_file(AMEND3_PATH)
    if got != AMEND3_MD5:
        die("%s MD5 %s != pinned %s" % (AMEND3_PATH, got, AMEND3_MD5))
    contracts["d11_amendment_3"] = OrderedDict([("path", AMEND3_PATH), ("md5", got)])
    print("[md5] %-18s %s  %s" % ("d11_amendment_3", got, AMEND3_PATH))
    a3 = " ".join(open(AMEND3_PATH, encoding="utf-8").read().split())
    if ('"%s{R1 reading}"' % RO_LABEL) not in a3:
        die("the RO line label is not the one written in Amendment 3 A3.6")
    print("[ok] RO line = %r + reading: the string of Amendment 3 A3.6" % RO_LABEL)
    scripts = OrderedDict((os.path.relpath(p), md5_file(p)) for p in
                          (os.path.abspath(__file__), os.path.join(_HERE, "d11_common.py"), os.path.join(_HERE, "d11_student.py"),
                           os.path.join(_HERE, "d1_eval_arms.py"), os.path.join(_HERE, "efbpt_eval_dev200.py"), os.path.join(_HERE, "d4_extract_facts.py")))
    instruction = load_instruction()
    print("[ok] Urdu instruction MD5 %s verified" % INSTRUCTION_MD5)

    rows = load_set_rows(SET)
    qids = [r["qid"] for r in rows]
    readings, rinfo = load_readings("R1", SET, rows)          # MD5 = R_SEAL = freeze §2
    for r in rows:
        for k in ("question_ur", "question_en"):
            if "\n" in r[k] or r[k] != r[k].strip() or not r[k]:
                die("qid %s: %s is empty, has a newline or outer whitespace" % (r["qid"], k))
        if "\n" in readings[r["qid"]] or readings[r["qid"]] != readings[r["qid"]].strip():
            die("qid %s: the R1 reading has a newline or outer whitespace" % r["qid"])

    sealed = OrderedDict()
    for a, (path, want) in SEALED.items():
        if "eval458" in path:
            die("refusing an eval458 path")
        g = md5_file(path)
        if g != want:
            die("sealed %s MD5 %s != pinned %s" % (path, g, want))
        recs = read_jsonl(path)
        if [e["qid"] for e in recs] != qids or any(e["gold"] != r["gold"] for e, r in zip(recs, rows)):
            die("sealed %s rows differ from DEV200" % path)
        sealed[a] = recs
        print("[sealed] %-5s %s verified (%s); %.2f%%" % (a, path, g, score(recs, "pred")["accuracy"]))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def chat(user):
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def no_facts(row):
        return OrderedDict((k, v) for k, v in row.items() if k != "facts")

    line_of = {"RO": lambda r: RO_LABEL + readings[r["qid"]], "RO_gold": lambda r: RO_LABEL + r["question_en"]}
    prompts, users = OrderedDict((a, []) for a in ARMS), OrderedDict((a, []) for a in ARMS)
    a_prompts = []
    for r, e in zip(rows, sealed["A_r2"]):
        user_a = build_user(instruction, no_facts(r), None, "A")
        p_a = chat(user_a)
        if p_a != e["prompt"]:
            die("layout-A prompt rebuilt here differs from the sealed D10 A_r2 prompt for qid %s" % r["qid"])
        a_prompts.append(p_a)
        for a in ARMS:
            line = line_of[a](r)
            user = build_user(instruction, dict(no_facts(r), question_ur=r["question_ur"] + "\n" + line), None, "A")
            if user != user_a + "\n" + line or user.count("\n") != user_a.count("\n") + 1:
                die("%s qid %s: the user message is not layout A + one line" % (a, r["qid"]))
            p = chat(user)
            if p.count(RO_LABEL) != 1 or (p_a.replace(user_a, user) != p):
                die("%s qid %s: the prompt is not the layout-A prompt with the one line added after the question" % (a, r["qid"]))
            users[a].append(user)
            prompts[a].append(p)
    print("[ok] layout-A prompts rebuilt here are byte-identical to the %d sealed D10 A_r2 prompts" % len(rows))
    print("[ok] RO / RO_gold prompt = that layout-A prompt + exactly one line directly after the Urdu question; nothing else changes")
    for a in ARMS:
        tl = sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts[a])
        print("[len] %-8s n=%d tokens min=%d median=%d max=%d" % (a, len(tl), tl[0], tl[len(tl) // 2], tl[-1]))
        if tl[-1] + MAX_NEW_TOKENS > CONTEXT_WINDOW:
            die("%s prompts would exceed the context window" % a)

    # ---- leak guards: the layout-A part is guarded as every A arm (question_en, gold facts, evidence ids FATAL);
    #      the added line is a reading (RO) or the gold question_en (RO_gold) BY DESIGN and is checked on its own:
    #      no gold fact, no evidence id, no OTHER row's question_en in an RO line. No facts, no facts header.
    needles = needles_for(rows, SET)
    leak_guard("A-part", a_prompts, needles, fatal_gold_facts=True)
    info = OrderedDict()
    for a in ARMS:
        eq = 0
        for r in rows:
            line = line_of[a](r)[len(RO_LABEL):]
            for s in needles["facts"]:
                if s in line:
                    die("LEAK: %s qid %s line contains gold fact %r" % (a, r["qid"], s))
            m = needles["evidence_re"].search(line)
            if m:
                die("LEAK: %s qid %s line contains evidence id %r" % (a, r["qid"], m.group(0)))
            if a == "RO":
                for q in needles["question_en"]:
                    if q in line and q != r["question_en"]:
                        die("LEAK: RO qid %s reading contains another row's question_en" % r["qid"])
                eq += int(line.strip() == r["question_en"].strip())
        info[a] = OrderedDict([("line", "label + R1 reading" if a == "RO" else "label + gold question_en (oracle, by design)"),
                               ("lines_equal_gold_question_en", eq if a == "RO" else len(rows))])
        print("[leak-guard] %-8s %d added lines: no gold fact, no evidence id%s"
              % (a, len(rows), "; no other-row question_en; equal to own gold question_en %d (reported)" % eq if a == "RO" else "; the line IS the gold question_en (oracle arm)"))
    from d1_eval_arms import FACTS_HEADER
    if any(FACTS_HEADER in p for a in ARMS for p in prompts[a]):
        die("a RO / RO_gold prompt contains the facts header")
    print("[ok] no facts header in any RO / RO_gold prompt (no facts in either arm)")

    pre = OrderedDict([
        ("contracts", contracts), ("scripts", scripts), ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("set", SET), ("arms", list(ARMS)), ("ro_label", RO_LABEL), ("reader", rinfo),
        ("sealed_arms", OrderedDict((a, OrderedDict([("path", p), ("md5", m)])) for a, (p, m) in SEALED.items())),
        ("leak_guard", info)])
    print("[preflight] PASS", flush=True)
    return rows, prompts, a_prompts, sealed, tok, pre


def print_full_prompt(label, prompt):
    print("\n----- BEGIN FULL PROMPT [%s] (%d chars) -----" % (label, len(prompt)))
    print(prompt)
    print("----- END FULL PROMPT [%s] -----" % label)
    print("  repr: %r" % prompt, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true", help="CPU: all checks and one full RO prompt; no model, nothing written")
    args = ap.parse_args()
    rows, prompts, a_prompts, sealed, tok, pre = preflight()
    print_full_prompt("RO row 0, qid %s" % rows[0]["qid"], prompts["RO"][0])
    print_full_prompt("RO_gold row 0, qid %s" % rows[0]["qid"], prompts["RO_gold"][0])
    if args.preflight_only:
        print("\n[preflight-only] done. No model loaded. No files written.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    qids = [r["qid"] for r in rows]

    # ---- resume state (completed rows are verified and never regenerated) ----
    done = OrderedDict()
    for a in ARMS:
        recs = read_rows(out_path("d11_arm%s.jsonl" % a), repair=True)
        for i, rec in enumerate(recs):
            if i >= len(rows) or rec["qid"] != qids[i] or rec["arm"] != a or rec["prompt"] != prompts[a][i]:
                die("d11_arm%s.jsonl row %d does not match the row / prompt built now" % (a, i))
        done[a] = recs
        if recs:
            print("[resume] arm %s: %d/%d rows completed and verified" % (a, len(recs), len(rows)))
    start_p = out_path("S_START.json")
    if os.path.exists(start_p):
        st = json.load(open(start_p, encoding="utf-8"))
        if json.dumps(st.get("scripts"), sort_keys=True) != json.dumps(pre["scripts"], sort_keys=True):
            die("resume: the scripts differ from those in %s" % start_p)
        with open(out_path("S_RESUME.jsonl"), "ab") as f:
            f.write(record_line(OrderedDict([("time", time.strftime("%Y-%m-%d %H:%M:%S")), ("slurm_job_id", job_id), ("hostname", socket.gethostname()),
                                             ("rows_completed", OrderedDict((a, len(done[a])) for a in ARMS))])))
    elif any(done.values()):
        die("arm rows exist but %s is missing" % start_p)

    model, eos_ids = load_student_model(tok)

    # ---- V2' (A3.6: fatal): 5 sealed A_r2 prompts regenerated, byte-identical ----
    print("\n" + "=" * 70)
    print("V2' — %d sealed A_r2 prompts regenerated once at batch size %d; must be byte-identical (A3.6: fatal)" % (V2_PROBE_N, BATCH_SIZE))
    print("=" * 70, flush=True)
    g = [generate_all(model, tok, [p], BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)[0] for p in a_prompts[:V2_PROBE_N]]
    mism = [e["qid"] for (text, _, _), e in zip(g, sealed["A_r2"][:V2_PROBE_N]) if text != e["generation"]]
    v2 = OrderedDict([("n", V2_PROBE_N), ("identical_to_sealed", V2_PROBE_N - len(mism)), ("pass", not mism), ("mismatched_qids", mism),
                      ("slurm_job_id", job_id), ("hostname", socket.gethostname())])
    print("V2' %s: %d/%d byte-identical to the sealed D10 A_r2 generations" % ("PASS" if not mism else "FAIL", V2_PROBE_N - len(mism), V2_PROBE_N), flush=True)
    if mism:
        die("V2' failed (A3.6: fatal); mismatched qids %s" % mism)

    if not os.path.exists(start_p):
        start = OrderedDict([
            ("job", "ablate_ro"), ("experiment", "D11 RTG A3.6 (DEV200 ablation, descriptive)"), ("set", SET),
            ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("hostname", socket.gethostname()), ("slurm_job_id", job_id),
            ("n_rows", len(rows)), ("qids", qids), ("model_path", MODEL_PATH),
            ("decoding", OrderedDict([("batch_size", BATCH_SIZE), ("do_sample", False), ("enable_thinking", False),
                                      ("quant", "nf4 double-quant bf16 compute"), ("answer_max_new", MAX_NEW_TOKENS)])),
            ("versions", pkg_versions()), ("V2_prime", v2)])
        start.update(pre)
        print("\n[START]")
        write_json("S_START.json", start)

    results = OrderedDict()
    for a in ARMS:
        path = out_path("d11_arm%s.jsonl" % a)
        seal_p = out_path("S_SEAL_%s.json" % a)
        k = len(done[a])
        if os.path.exists(seal_p):
            s = json.load(open(seal_p, encoding="utf-8"))
            if k != len(rows) or dict(s["files"][path]) != dict(file_entry(path)):
                die("%s does not match %s" % (seal_p, path))
            print("\n[done] arm %s is sealed (%d rows); nothing is regenerated" % (a, k))
            results[a] = done[a]
            continue
        print("\n" + "=" * 70)
        print("ARM %s  rows %d..%d of %d  (layout A + one line, max_new=%d, batch size %d, greedy)" % (a, k, len(rows) - 1, len(rows), MAX_NEW_TOKENS, BATCH_SIZE))
        print("=" * 70, flush=True)
        with open(path, "ab") as f:
            for i in range(k, len(rows)):
                text, n_new, trunc = generate_all(model, tok, [prompts[a][i]], BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)[0]
                pred = extract_answer(text)
                rec = OrderedDict([
                    ("qid", qids[i]), ("arm", a), ("layout", "A+line"), ("gold", rows[i]["gold"]), ("pred", pred),
                    ("pred_secondary", extract_answer_secondary(text)), ("correct", pred == rows[i]["gold"]), ("unparsed", pred is None),
                    ("truncated", trunc), ("n_gen_tokens", n_new), ("n_facts_given", 0),
                    ("added_line_source", "R1 reading" if a == "RO" else "gold question_en"),
                    ("prompt", prompts[a][i]), ("generation", text)])
                f.write(record_line(rec))
                f.flush()
                os.fsync(f.fileno())
                done[a].append(rec)
        recs = read_rows(path)
        if [r["qid"] for r in recs] != qids:
            die("%s: rows on disk are not DEV200 in order" % path)
        results[a] = recs
        s = score(recs, "pred")
        print("  %s accuracy %.2f%%  unparsed %.2f%%  (%d/%d correct)  pred-yes %.2f%%" % (a, s["accuracy"], s["unparsed_rate"], s["correct"], s["n"], s["predicted_yes_rate"]))
        write_json("S_SEAL_%s.json" % a, OrderedDict([
            ("job", "ablate_ro"), ("arm", a), ("set", SET), ("slurm_job_id", job_id), ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
            ("complete", True), ("n_rows", len(recs)), ("d11_amendment3_md5", AMEND3_MD5), ("scripts", pre["scripts"]),
            ("files", OrderedDict([(path, file_entry(path))]))]))

    # ---- summary (A3.6) ----
    allr = OrderedDict(list(sealed.items()) + list(results.items()))
    hdr = "=" * 84
    print("\n" + hdr)
    print("D11 RTG A3.6 — DEV200 ablation RO / RO_gold (descriptive, not gated)")
    print(hdr)
    table = OrderedDict()
    print("%-8s %5s %8s %8s %9s %9s %8s %8s %12s %12s" % ("arm", "n", "acc%", "correct", "unparsed", "truncated", "SEC acc%", "yes%", "deva-any", "deva-answer"))
    for a in ("A_r2", "RO", "RO_gold", "TR", "TO1"):
        recs = allr[a]
        pri, se = score(recs, "pred"), score(recs, "pred_secondary")
        table[a] = OrderedDict([
            ("n", pri["n"]), ("accuracy", pri["accuracy"]), ("correct", pri["correct"]),
            ("unparsed", sum(1 for r in recs if r["pred"] is None)), ("truncated", sum(1 for r in recs if r["truncated"])),
            ("accuracy_secondary", se["accuracy"]), ("predicted_yes_rate", pri["predicted_yes_rate"]),
            ("devanagari_answer_string_in_generation", sum(1 for r in recs if has_devanagari_answer(r["generation"]))),
            ("devanagari_answer_rows_primary_ne_secondary", sum(1 for r in recs if r["pred"] != r["pred_secondary"])),
            ("source", "sealed" if a in sealed else "this job")])
        t = table[a]
        print("%-8s %5d %8.2f %8d %9d %9d %8.2f %8.2f %12d %12d" % (a, t["n"], t["accuracy"], t["correct"], t["unparsed"], t["truncated"], t["accuracy_secondary"],
                                                                 t["predicted_yes_rate"], t["devanagari_answer_string_in_generation"], t["devanagari_answer_rows_primary_ne_secondary"]))
    pairs = OrderedDict()
    for x, y in PAIRS:
        b, c, p = mcnemar(allr[x], allr[y], x, y)
        pairs["%s_vs_%s" % (x, y)] = OrderedDict([("acc_" + x, table[x]["accuracy"]), ("acc_" + y, table[y]["accuracy"]),
                                                  ("delta_pp", round(table[x]["accuracy"] - table[y]["accuracy"], 2)), ("b", b), ("c", c), ("p_mcnemar_exact_two_sided", p)])
        e = pairs["%s_vs_%s" % (x, y)]
        print("%-8s - %-8s = %+6.2f pp (%.2f vs %.2f), b %d c %d, p = %s" % (x, y, e["delta_pp"], table[x]["accuracy"], table[y]["accuracy"], b, c, p))
    preds = OrderedDict()
    for label, (x, y, lo, hi) in PREDICTIONS.items():
        d = round(table[x]["accuracy"] - table[y]["accuracy"], 2)
        preds[label] = OrderedDict([("observed", d), ("hit", d >= lo and (hi is None or d <= hi))])
    print("A3.6 predictions: " + "; ".join("%s: %+.2f -> %s" % (k, v["observed"], "hit" if v["hit"] else "miss") for k, v in preds.items()))
    summary = OrderedDict([("experiment", "D11 RTG A3.6: DEV200 ablation RO / RO_gold; descriptive, not gated, no eval458 use"), ("set", SET), ("n", len(rows)),
                           ("slurm_job_id", job_id), ("V2_prime", v2), ("table", table), ("pairs", pairs), ("predictions_A3_6", preds),
                           ("contracts", pre["contracts"]), ("scripts", pre["scripts"]), ("reader", pre["reader"]), ("sealed_arms", pre["sealed_arms"])])
    write_json("d11_ablate_ro_summary.json", summary)
    names = ["S_START.json"] + ["d11_arm%s.jsonl" % a for a in ARMS] + ["S_SEAL_%s.json" % a for a in ARMS] + ["d11_ablate_ro_summary.json"]
    if os.path.exists(out_path("S_RESUME.jsonl")):
        names.append("S_RESUME.jsonl")
    write_json("S_SEAL.json", OrderedDict([
        ("job", "ablate_ro"), ("experiment", "D11 RTG A3.6"), ("set", SET), ("slurm_job_id", job_id), ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("complete", True), ("d11_amendment3_md5", AMEND3_MD5), ("script_md5", pre["scripts"][os.path.relpath(os.path.abspath(__file__))]),
        ("files", OrderedDict((out_path(n), file_entry(out_path(n))) for n in names))]))
    print("\nD11 RTG A3.6 ablation COMPLETE. Descriptive only.")


if __name__ == "__main__":
    main()
