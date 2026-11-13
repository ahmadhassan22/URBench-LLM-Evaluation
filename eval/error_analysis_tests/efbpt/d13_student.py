#!/usr/bin/env python3
"""
d13_student.py — D13 student arms (freeze §2, §6, §7.3, §7.4).

  --model qwen14   the D10 §2 student (Qwen3-14B nf4, greedy, thinking off,
                   max_new 1024, batch size 1), loaded by the frozen
                   d11_student.load_student_model. Arms: RAG_ur, RAG_en (layout B,
                   the sealed retrieval entries in the facts block), RO_eval
                   (layout A + the translation line, D11 A3.6).
                   --mode REGRESS = freeze C1: the D11 A2.3 regression (DEV200, arm
                   TR, rows 0-4) through this script's generation path; generations
                   and prompts must be byte-identical to the sealed file. Every other
                   qwen14 run refuses to start without a PASS made by these scripts.
  --model llama8|gemma9|qwen8   Group 4: S_A (layout A) and S_TR (layout B with the
                   sealed R1 teacher facts of TR). User message = the Qwen3-14B user
                   message; the model's own chat template; nf4 double-quant; greedy;
                   max_new 1024; batch size 1.
                   TEST = freeze C2: DEV200 rows 0-4 in both layouts generated twice,
                   every pair byte-identical (fatal otherwise). A FULL run needs that
                   PASS and first repeats the five S_TR rows (reported).

Generation is d4_extract_facts.generate_all with one prompt per call, as in D11.
Every finished row is appended and fsynced; a crashed arm resumes and finished
rows are never generated again. Nothing is scored: no extractor is imported and
rows carry no pred / correct / gold.

  --mode TEST|FULL|REGRESS  --set dev200|eval458  --arms A,B  [--preflight-only]
TEST = DEV200 rows 0-4 under outputs/efbpt/d13/_test/ (never a result).
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

import d13_common as C  # noqa: E402
from d13_common import die, git_state, read_jsonl, md5_file, MAX_NEW_TOKENS, MODEL_PATH  # noqa: E402
from d8_reachable_ceiling import pkg_versions  # noqa: E402

BATCH_SIZE = 1
REGRESS_N = 5
REGRESS_SPLIT = 3                      # rows 0-2, then a resume for rows 3-4
GATE_SCRIPTS = ("eval/error_analysis_tests/efbpt/d13_student.py", "eval/error_analysis_tests/efbpt/d13_common.py",
                "eval/error_analysis_tests/efbpt/d11_student.py", "eval/error_analysis_tests/efbpt/d4_extract_facts.py")
REGRESS_DIR = os.path.join(C.REGRESS_ROOT, "qwen14")
REGRESS_JSON = os.path.join(REGRESS_DIR, "REGRESS.json")
DECODING = OrderedDict([("do_sample", False), ("max_new_tokens", MAX_NEW_TOKENS), ("batch_size", BATCH_SIZE),
                        ("quantization", "4-bit nf4, double quantization, bf16 compute"), ("attention", "sdpa")])


def c2_json(skey):
    return os.path.join(C.TEST_ROOT, "transfer", "dev200", "C2_%s.json" % skey)


def gate_md5s():
    s = C.script_md5s()
    return OrderedDict((k, s[k]) for k in GATE_SCRIPTS)


def require_pass(path, what):
    if not os.path.exists(path):
        die("%s: no record at %s; run it first" % (what, path))
    rec = json.load(open(path, encoding="utf-8"))
    if rec.get("verdict") != "PASS":
        die("%s: verdict %r at %s" % (what, rec.get("verdict"), path))
    if rec.get("gate_scripts") != dict(gate_md5s()):
        die("%s: the PASS at %s was made with other script versions; run it again" % (what, path))
    print("[ok] %s: PASS on record (%s, job %s), same scripts" % (what, path, rec.get("slurm_job_id")))
    return OrderedDict([("path", path), ("md5", md5_file(path)), ("verdict", "PASS"), ("slurm_job_id", rec.get("slurm_job_id")),
                        ("hostname", rec.get("hostname"))])


def load_other_model(skey, tok):
    """Group 4 student, loaded as the D10 §2 student is (nf4 double-quant, bf16
    compute, sdpa). Generation stops at the tokenizer's end-of-sequence token, the
    template's end-of-turn token, or a stop token of the model's own generation
    config (freeze §7.4)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    spec = C.OTHER_STUDENTS[skey]
    eot = tok.convert_tokens_to_ids(spec["end_of_turn"])
    if not isinstance(eot, int) or eot < 0 or eot == tok.unk_token_id:
        die("%s: the end-of-turn token %r is not in the tokenizer" % (skey, spec["end_of_turn"]))
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    print("\n[load] %s, 4-bit nf4 ..." % spec["path"], flush=True)
    model = AutoModelForCausalLM.from_pretrained(spec["path"], quantization_config=bnb, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    stop = {eot}
    if tok.eos_token_id is not None:
        stop.add(tok.eos_token_id)
    own = model.generation_config.eos_token_id
    for t in (own if isinstance(own, (list, tuple)) else [own]):
        if t is not None:
            stop.add(int(t))
    model.generation_config.eos_token_id = sorted(stop)
    model.config.use_cache = True
    model.eval()
    print("[load] stop token ids %s (end of turn %r = %d; tokenizer eos %s; model default %s)" % (sorted(stop), spec["end_of_turn"], eot, tok.eos_token_id, own), flush=True)
    return model, stop


def gen_one(model, tok, prompt, eos_ids):
    from d4_extract_facts import generate_all
    t0 = time.time()
    gens = generate_all(model, tok, [prompt], BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
    if len(gens) != 1:
        die("%d generations for 1 prompt" % len(gens))
    return gens[0], time.time() - t0


def run_rows(path, arm, set_name, prompts, k, n_target, model, tok, eos_ids):
    """Rows k..n_target-1, one prompt per call, each row appended and fsynced. Prints no answer content."""
    t0 = time.time()
    with open(C.guard_out(path), "ab") as f:
        for i in range(k, n_target):
            p = prompts[i]
            (text, n_new, trunc), secs = gen_one(model, tok, p["prompt"], eos_ids)
            rec = OrderedDict([("qid", p["qid"]), ("arm", arm), ("set", set_name)])
            for key in ("layout", "n_facts"):
                if key in p:
                    rec[key] = p[key]
            rec.update([("truncated", trunc), ("n_gen_tokens", n_new), ("seconds", round(secs, 3)), ("prompt", p["prompt"]), ("generation", text)])
            C.append_row(f, rec)
            print("  [row] %s %d/%d qid %s: %d new tokens%s  %.1fs  (%.0fs)"
                  % (arm, i + 1, len(prompts), p["qid"], n_new, " TRUNCATED" if trunc else "", secs, time.time() - t0), flush=True)


def start_obj(arm, set_name, mode, prompts, contracts, info, student, extra):
    s = OrderedDict([
        ("experiment", "D13"), ("arm", arm), ("group", C.ARMS[arm]["group"]), ("set", set_name), ("mode", mode),
        ("student", student), ("decoding", DECODING), ("generation", "d4_extract_facts.generate_all, one prompt per call"),
        ("hostname", socket.gethostname()), ("n_rows", len(prompts)), ("qids", [p["qid"] for p in prompts]),
        ("contracts", contracts), ("scripts", C.script_md5s()), ("git", git_state()), ("guards", info),
        ("urdu_instruction", OrderedDict([("path", C.INSTRUCTION_PATH), ("md5", C.INSTRUCTION_MD5)])),
        ("versions", pkg_versions()), ("scored", False)])
    s.update(extra)
    return s


def run_arm(arm, set_name, mode, prompts, contracts, info, student, extra, model, tok, eos_ids):
    k = C.open_arm(arm, set_name, mode, prompts, start_obj(arm, set_name, mode, prompts, contracts, info, student, extra))
    if k is None:
        return
    print("\n" + "=" * 70)
    print("ARM %s  set %s  rows %d..%d of %d  (max_new=%d, batch size %d, greedy)" % (arm, set_name, k, len(prompts) - 1, len(prompts), MAX_NEW_TOKENS, BATCH_SIZE))
    print("=" * 70, flush=True)
    run_rows(C.arm_paths(arm, set_name, mode)["rows"], arm, set_name, prompts, k, len(prompts), model, tok, eos_ids)
    C.seal_arm(arm, set_name, mode, prompts, OrderedDict([("student", student), ("hostname", socket.gethostname())]))


# ----------------------------------------------------------------------------
# Qwen3-14B: regression (C1), RAG and RO arms
# ----------------------------------------------------------------------------
def qwen14_regress(contracts):
    """Freeze C1 = D11 Amendment 2 A2.3 through this script's path."""
    import d11_student as S11
    d11p = C.d11_prompts("dev200", "TEST")
    tok = d11p["tok14"]
    prompts = [OrderedDict([("qid", e["qid"]), ("prompt", C.chat14(tok, e["user"])), ("layout", e["layout"]), ("n_facts", e["n_facts"])]) for e in d11p["TR"]]
    path, want = C.SEALED["dev200/TR"]
    C.verify_sealed(("dev200/TR",))
    with open(path, "rb") as f:
        sealed = [json.loads(l.decode("utf-8")) for l in f.read().split(b"\n")[:REGRESS_N]]
    if len(prompts) != REGRESS_N or [s["qid"] for s in sealed] != [p["qid"] for p in prompts]:
        die("regression: the first %d sealed rows are not DEV200 rows 0-%d" % (REGRESS_N, REGRESS_N - 1))
    rows_path = os.path.join(REGRESS_DIR, "d13_regress_TR.jsonl")
    os.makedirs(C.guard_out(REGRESS_DIR), exist_ok=True)
    for p in (rows_path, REGRESS_JSON):          # a regression is never a result; each run starts clean
        if os.path.exists(p):
            os.remove(C.guard_out(p))
    model, eos_ids = S11.load_student_model(tok)
    run_rows(rows_path, "TR", "dev200", prompts, 0, REGRESS_SPLIT, model, tok, eos_ids)
    done = C.read_rows(rows_path, repair=True)   # the resume path: re-read from disk, then continue
    if [d["qid"] for d in done] != [p["qid"] for p in prompts[:REGRESS_SPLIT]]:
        die("regression: the rows on disk are not rows 0-%d" % (REGRESS_SPLIT - 1))
    run_rows(rows_path, "TR", "dev200", prompts, len(done), REGRESS_N, model, tok, eos_ids)
    recs = C.read_rows(rows_path)
    print("\n" + "=" * 78)
    print("C1 REGRESSION — DEV200, arm TR, rows 0-%d, the D13 student path vs sealed %s (MD5 %s)" % (REGRESS_N - 1, path, want))
    print("=" * 78)
    out_rows = []
    for i, (rec, se) in enumerate(zip(recs, sealed)):
        same_gen = rec["generation"].encode("utf-8") == se["generation"].encode("utf-8")
        same_prompt = rec["prompt"].encode("utf-8") == se["prompt"].encode("utf-8")
        same_n = rec["n_gen_tokens"] == se["n_gen_tokens"] and rec["truncated"] == se["truncated"]
        out_rows.append(OrderedDict([("row", i), ("qid", rec["qid"]), ("generation_identical", same_gen), ("prompt_identical", same_prompt),
                                     ("tokens_and_truncation_identical", same_n), ("n_gen_tokens", rec["n_gen_tokens"]), ("seconds", rec["seconds"])]))
        print("  row %d qid %s: generation %s (%d bytes), prompt %s, token count and truncation flag %s"
              % (i, rec["qid"], "IDENTICAL" if same_gen else "DIFFERENT", len(rec["generation"].encode("utf-8")),
                 "identical" if same_prompt else "DIFFERENT", "identical" if same_n else "DIFFERENT"))
    ok = len(out_rows) == REGRESS_N and all(x["generation_identical"] and x["prompt_identical"] and x["tokens_and_truncation_identical"] for x in out_rows)
    n_same = sum(1 for x in out_rows if x["generation_identical"])
    C.write_json(REGRESS_JSON, OrderedDict([
        ("experiment", "D13"), ("what", "freeze C1: D11 A2.3 regression, DEV200 arm TR rows 0-%d through the D13 student path" % (REGRESS_N - 1)),
        ("verdict", "PASS" if ok else "FAIL"), ("n_rows", REGRESS_N), ("n_identical", n_same),
        ("resume_exercised", "rows 0-%d generated, the row file re-read from disk, rows %d-%d generated" % (REGRESS_SPLIT - 1, REGRESS_SPLIT, REGRESS_N - 1)),
        ("sealed_file", OrderedDict([("path", path), ("md5", want)])), ("rows", out_rows),
        ("gate_scripts", gate_md5s()), ("scripts", C.script_md5s()), ("contracts", contracts), ("git", git_state()),
        ("slurm_job_id", os.environ.get("SLURM_JOB_ID")), ("hostname", socket.gethostname()),
        ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("versions", pkg_versions())]))
    print("C1 REGRESSION %s: %d/%d generations byte-identical to the sealed file" % ("PASS" if ok else "FAIL", n_same, REGRESS_N), flush=True)
    return ok


def qwen14_arms(args, contracts):
    import d11_student as S11
    mode = args.mode
    set_name = "dev200" if mode == "TEST" else args.set
    arms = args.arms.split(",") if args.arms else [a for a in C.QWEN14_ARMS if mode == "TEST" or set_name in C.ARMS[a]["sets"]]
    for a in arms:
        if a not in C.QWEN14_ARMS:
            die("%s is not a Qwen3-14B arm" % a)
        C.check_arm_set(a, set_name, mode)
    ctx = C.make_ctx(set_name, mode)
    tok = C.load_tok(MODEL_PATH)
    built = OrderedDict()
    for a in arms:
        built[a] = C.build_ro_prompts(ctx, tok) if a == "RO_eval" else C.build_rag_prompts(a, ctx, tok)
        C.log_full_prompt("%s row 0, qid %s" % (a, built[a][0][0]["qid"]), built[a][0][0]["prompt"])
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No model loaded, nothing written.")
        return
    reg = require_pass(REGRESS_JSON, "C1 regression")
    student = OrderedDict([("label", "Qwen3-14B"), ("path", MODEL_PATH), ("loader", "d11_student.load_student_model"), ("thinking", "off")])
    pending = [a for a in arms if not os.path.exists(C.arm_paths(a, set_name, mode)["seal"])]
    if not pending:
        print("[done] every requested arm is already sealed; no model loaded")
        return
    model, eos_ids = S11.load_student_model(tok)
    for a in arms:
        prompts, info = built[a]
        run_arm(a, set_name, mode, prompts, contracts, info, student, OrderedDict([("regression", reg)]), model, tok, eos_ids)
        if mode == "TEST" and a == "RO_eval":
            path, _ = C.SEALED["dev200/RO"]
            sealed = {e["qid"]: e for e in read_jsonl(path)}
            pth = C.arm_paths(a, set_name, mode)
            rows = [OrderedDict([("qid", r["qid"]), ("prompt_identical", r["prompt"] == sealed[r["qid"]]["prompt"]),
                                 ("generation_identical", r["generation"] == sealed[r["qid"]]["generation"])]) for r in C.read_rows(pth["rows"])]
            n = sum(1 for x in rows if x["generation_identical"])
            C.write_json(os.path.join(pth["dir"], "TEST_REPORT_%s.json" % a), OrderedDict([
                ("what", "TEST-5: RO on DEV200 rows 0-4 vs the sealed DEV200 RO file (prompt identity fatal on CPU; generation identity reported; freeze §7.4)"),
                ("sealed_file", OrderedDict([("path", path), ("md5", C.SEALED["dev200/RO"][1])])), ("n_rows", len(rows)),
                ("n_prompts_identical", sum(1 for x in rows if x["prompt_identical"])), ("n_generations_identical", n), ("rows", rows)]))
            print("  [TEST-5] RO: %d/%d generations are byte-identical to the sealed DEV200 RO generations (reported)" % (n, len(rows)), flush=True)
    print("\n[done] qwen14 %s %s: arms %s sealed, not scored" % (set_name, mode, arms), flush=True)


# ----------------------------------------------------------------------------
# Group 4: other students
# ----------------------------------------------------------------------------
def other_student(args, contracts):
    skey, mode = args.model, args.mode
    spec = C.OTHER_STUDENTS[skey]
    ok, have, n = C.weights_present(spec["path"])
    if not ok:
        die("weights of %s are not complete on the cluster (%d/%d shards): STOP and report, no download" % (spec["label"], have, n))
    tok = C.load_tok(spec["path"])
    student = OrderedDict([("key", skey), ("label", spec["label"]), ("path", spec["path"]), ("end_of_turn", spec["end_of_turn"]),
                           ("system_role", spec["system_role"]), ("thinking", "off (enable_thinking=False)" if spec["thinking_switch"] else "no switch")])
    ctx_test = C.make_ctx("dev200", "TEST")
    test_prompts, test_info = C.build_other_student_prompts(skey, C.d11_prompts("dev200", "TEST"), ctx_test, tok)
    names = OrderedDict((lay, spec["name"] + "_" + lay) for lay in ("A", "TR"))
    if mode == "FULL":
        set_name = args.set
        arms = args.arms.split(",") if args.arms else list(names.values())
        for a in arms:
            if a not in names.values():
                die("%s is not an arm of %s" % (a, spec["label"]))
            C.check_arm_set(a, set_name, mode)
        ctx = C.make_ctx(set_name, mode)
        full_prompts, full_info = C.build_other_student_prompts(skey, C.d11_prompts(set_name, "FULL"), ctx, tok)
    for lay, a in names.items():
        C.log_full_prompt("%s TEST-5 row 0, qid %s" % (a, test_prompts[lay][0]["qid"]), test_prompts[lay][0]["prompt"])
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No model loaded, nothing written.")
        return

    if mode == "TEST":
        model, eos_ids = load_other_model(skey, tok)
        student["stop_token_ids"] = sorted(eos_ids)
        rows_out, all_ok = [], True
        for lay, a in names.items():
            prompts = test_prompts[lay]
            run_arm(a, "dev200", "TEST", prompts, contracts, test_info, student, OrderedDict(), model, tok, eos_ids)   # pass 1
            first = C.read_rows(C.arm_paths(a, "dev200", "TEST")["rows"])
            print("\n[C2] %s: second pass over the same %d DEV200 rows" % (a, len(prompts)), flush=True)
            for i, (p, r1) in enumerate(zip(prompts, first)):                                                         # pass 2
                (text, n_new, trunc), secs = gen_one(model, tok, p["prompt"], eos_ids)
                same = text.encode("utf-8") == r1["generation"].encode("utf-8") and n_new == r1["n_gen_tokens"] and trunc == r1["truncated"]
                all_ok = all_ok and same
                rows_out.append(OrderedDict([("arm", a), ("row", i), ("qid", p["qid"]), ("identical", same), ("n_gen_tokens", [r1["n_gen_tokens"], n_new]),
                                             ("seconds", [r1["seconds"], round(secs, 3)])]))
                print("  [C2] %s row %d qid %s: second generation %s (%d / %d new tokens)" % (a, i, p["qid"], "IDENTICAL" if same else "DIFFERENT", r1["n_gen_tokens"], n_new), flush=True)
        C.write_json(c2_json(skey), OrderedDict([
            ("experiment", "D13"), ("what", "freeze C2: %s, DEV200 rows 0-4 in both layouts generated twice in one job" % spec["label"]),
            ("verdict", "PASS" if all_ok else "FAIL"), ("n_pairs", len(rows_out)), ("n_identical", sum(1 for x in rows_out if x["identical"])),
            ("rows", rows_out), ("student", student), ("gate_scripts", gate_md5s()), ("scripts", C.script_md5s()), ("contracts", contracts),
            ("git", git_state()), ("slurm_job_id", os.environ.get("SLURM_JOB_ID")), ("hostname", socket.gethostname()),
            ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("versions", pkg_versions())]))
        print("C2 %s %s: %d/%d pairs byte-identical" % (spec["label"], "PASS" if all_ok else "FAIL", sum(1 for x in rows_out if x["identical"]), len(rows_out)), flush=True)
        if not all_ok:
            sys.exit(1)
        return

    c2 = require_pass(c2_json(skey), "C2 (%s)" % spec["label"])
    pending = [a for a in arms if not os.path.exists(C.arm_paths(a, set_name, mode)["seal"])]
    if not pending:
        print("[done] every requested arm is already sealed; no model loaded")
        return
    model, eos_ids = load_other_model(skey, tok)
    student["stop_token_ids"] = sorted(eos_ids)
    # freeze §7.4: repeat the five S_TR DEV200 rows and compare with the TEST-5 file (reported, not fatal)
    t5 = C.read_rows(C.arm_paths(names["TR"], "dev200", "TEST")["rows"])
    rep = []
    for p, r1 in zip(test_prompts["TR"], t5):
        (text, n_new, _trunc), _secs = gen_one(model, tok, p["prompt"], eos_ids)
        rep.append(OrderedDict([("qid", p["qid"]), ("identical_to_test5", r1["qid"] == p["qid"] and text == r1["generation"])]))
    n_rep = sum(1 for x in rep if x["identical_to_test5"])
    print("[C2 repeat] %s: %d/%d S_TR DEV200 rows byte-identical to the TEST-5 file (reported)" % (spec["label"], n_rep, len(rep)), flush=True)
    extra = OrderedDict([("c2", c2), ("c2_repeat_vs_test5", OrderedDict([("n_rows", len(rep)), ("n_identical", n_rep), ("rows", rep)]))])
    lay_of = {v: k for k, v in names.items()}
    for a in arms:
        run_arm(a, set_name, mode, full_prompts[lay_of[a]], contracts, full_info, student, extra, model, tok, eos_ids)
    print("\n[done] %s %s %s: arms %s sealed, not scored" % (skey, set_name, mode, arms), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("qwen14",) + tuple(C.OTHER_STUDENTS))
    ap.add_argument("--mode", required=True, choices=("TEST", "FULL", "REGRESS"))
    ap.add_argument("--set", default="dev200", choices=C.SETS)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    print("=" * 78)
    print("D13 student   model %s   mode %s%s" % (args.model, args.mode, "" if args.mode != "TEST" else "  (TEST-5 — NOT A RESULT)"))
    print("=" * 78, flush=True)
    contracts = C.verify_contracts()
    if args.mode == "FULL":
        C.require_committed()
    if args.model == "qwen14":
        if args.mode == "REGRESS":
            sys.exit(0 if qwen14_regress(contracts) else 1)
        qwen14_arms(args, contracts)
    else:
        if args.mode == "REGRESS":
            die("REGRESS is the Qwen3-14B regression (freeze C1)")
        other_student(args, contracts)


if __name__ == "__main__":
    main()
