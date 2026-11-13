#!/usr/bin/env python3
"""
d13_teacher.py — D13 Group 1 teacher arms (freeze §2, §7.1): the large model
alone, no student.

  T_direct_F_eval  the D10 Amendment 1 A1.1 prompt unchanged (Urdu question +
                   final-answer line). Set eval458.
  T_tr_R1          input = the sealed R1 English reading only; English instruction
                   ending with the final-answer line. Sets dev200, eval458.
  T_tr_L1          as T_tr_R1 with the sealed R3 (L1_llama70b_awq) reading.
Engine exactly D10 §2 (d10_teacher constants, imported): Llama-3.1-70B-Instruct-
AWQ-INT4, vLLM 0.13.0, the frozen engine arguments, temperature 0, top_p 1,
seed 0, max_tokens 1024. One prompt per call; every finished row is appended
and fsynced; a crashed arm resumes and finished rows are never generated again.

Nothing is scored: no extractor is imported, rows carry no pred / correct / gold.

  --arms A,B  --set dev200|eval458  --mode TEST|FULL  [--preflight-only]
TEST = DEV200 rows 0-4 of all three arms under outputs/efbpt/d13/_test/ (never a
result). T_direct_F_eval TEST prompts must equal the sealed D10 prompts (fatal,
checked on CPU); whether the five generations equal the sealed ones is reported.
"""
import argparse
import os
import socket
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d13_common as C  # noqa: E402
from d13_common import die, git_state, read_jsonl  # noqa: E402
from d10_teacher import TEACHER_PATH, VLLM_VERSION, VLLM_ARGS, SAMPLING, T3_MAX_TOKENS  # noqa: E402
from d8_reachable_ceiling import pkg_versions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None, help="comma list; default: every teacher arm defined for the set (TEST: all three)")
    ap.add_argument("--set", default="dev200", choices=C.SETS)
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    mode = args.mode
    set_name = "dev200" if mode == "TEST" else args.set
    arms = args.arms.split(",") if args.arms else [a for a in C.TEACHER_ARMS if mode == "TEST" or set_name in C.ARMS[a]["sets"]]
    for a in arms:
        if a not in C.TEACHER_ARMS:
            die("%s is not a teacher arm" % a)
        C.check_arm_set(a, set_name, mode)
    print("=" * 78)
    print("D13 teacher arms %s   set %s   %s" % (arms, set_name, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    contracts = C.verify_contracts()
    if mode == "FULL":
        C.require_committed()
    ctx = C.make_ctx(set_name, mode)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    built = OrderedDict()
    for a in arms:
        built[a] = C.build_teacher_prompts(a, ctx, tok)
        C.log_full_prompt("%s row 0, qid %s" % (a, built[a][0][0]["qid"]), built[a][0][0]["prompt"])
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No model loaded, nothing written.")
        return

    scripts, git = C.script_md5s(), git_state()
    todo = OrderedDict()
    for a in arms:
        prompts, info = built[a]
        start = OrderedDict([
            ("experiment", "D13"), ("arm", a), ("group", C.ARMS[a]["group"]), ("set", set_name), ("mode", mode),
            ("teacher_path", TEACHER_PATH), ("vllm_version", VLLM_VERSION), ("vllm_args", VLLM_ARGS), ("sampling", SAMPLING),
            ("max_tokens", T3_MAX_TOKENS), ("one_prompt_per_call", True),
            ("hostname", socket.gethostname()), ("n_rows", len(prompts)), ("qids", [p["qid"] for p in prompts]),
            ("contracts", contracts), ("scripts", scripts), ("git", git), ("guards", info),
            ("readings", OrderedDict((k, OrderedDict([("file", v["file"]), ("md5", v["md5"])])) for k, v in ctx["reading_info"].items())),
            ("scored", False),
        ])
        k = C.open_arm(a, set_name, mode, prompts, start)
        if k is not None:
            todo[a] = k
    if not todo:
        print("[done] every requested arm is already sealed; no model loaded")
        return

    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != VLLM_VERSION:
        die("vLLM %s found, the D10 freeze requires %s" % (vllm.__version__, VLLM_VERSION))
    versions = pkg_versions()
    versions["vllm"] = vllm.__version__
    print("\n[load] %s ..." % TEACHER_PATH, flush=True)
    t0 = time.time()
    llm = LLM(model=TEACHER_PATH, **VLLM_ARGS)
    print("[load] %.0fs" % (time.time() - t0), flush=True)
    params = SamplingParams(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], seed=SAMPLING["seed"], max_tokens=T3_MAX_TOKENS)

    for a, k in todo.items():
        prompts, _info = built[a]
        pth = C.arm_paths(a, set_name, mode)
        print("\n" + "=" * 70)
        print("ARM %s  rows %d..%d of %d  (max_tokens=%d, greedy, max_num_seqs=%d, one prompt per call)"
              % (a, k, len(prompts) - 1, len(prompts), T3_MAX_TOKENS, VLLM_ARGS["max_num_seqs"]))
        print("=" * 70, flush=True)
        ta = time.time()
        with open(C.guard_out(pth["rows"]), "ab") as f:
            for i in range(k, len(prompts)):
                p = prompts[i]
                t1 = time.time()
                outs = llm.generate([{"prompt_token_ids": p["ids"]}], params, use_tqdm=False)
                secs = time.time() - t1
                if len(outs) != 1:
                    die("%s row %d: vLLM returned %d outputs for 1 prompt" % (a, i, len(outs)))
                c = outs[0].outputs[0]
                rec = OrderedDict([
                    ("qid", p["qid"]), ("arm", a), ("set", set_name), ("truncated", c.finish_reason == "length"),
                    ("finish_reason", c.finish_reason), ("n_gen_tokens", len(c.token_ids)), ("seconds", round(secs, 3)),
                    ("prompt", p["prompt"]), ("generation", c.text)])
                C.append_row(f, rec)
                print("  [row] %s %d/%d qid %s: %d new tokens%s  %.1fs  (%.0fs)"
                      % (a, i + 1, len(prompts), p["qid"], rec["n_gen_tokens"], " TRUNCATED" if rec["truncated"] else "", secs, time.time() - ta), flush=True)
        C.seal_arm(a, set_name, mode, prompts, OrderedDict([("versions", versions), ("hostname", socket.gethostname())]))

        if mode == "TEST" and a == "T_direct_F_eval":
            path, _ = C.SEALED["dev200/T_direct_F"]
            sealed = {e["qid"]: e for e in read_jsonl(path)}
            recs = C.read_rows(pth["rows"])
            rows = [OrderedDict([("qid", r["qid"]), ("prompt_identical", r["prompt"] == sealed[r["qid"]]["prompt"]),
                                 ("generation_identical", r["generation"] == sealed[r["qid"]]["generation"]),
                                 ("n_gen_tokens", r["n_gen_tokens"]), ("sealed_n_gen_tokens", sealed[r["qid"]]["n_gen_tokens"])]) for r in recs]
            n = sum(1 for x in rows if x["generation_identical"])
            rep = OrderedDict([("what", "TEST-5: T_direct_F on DEV200 rows 0-4 vs the sealed D10 file (prompt identity fatal on CPU; generation identity reported; freeze §7.1)"),
                               ("sealed_file", OrderedDict([("path", path), ("md5", C.SEALED["dev200/T_direct_F"][1])])),
                               ("n_rows", len(rows)), ("n_prompts_identical", sum(1 for x in rows if x["prompt_identical"])),
                               ("n_generations_identical", n), ("rows", rows)])
            C.write_json(os.path.join(pth["dir"], "TEST_REPORT_%s.json" % a), rep)
            print("  [TEST-5] T_direct_F: %d/%d generations are byte-identical to the sealed D10 generations (reported)" % (n, len(rows)), flush=True)
    print("\n[done] %s %s: arms %s sealed, not scored" % (set_name, mode, list(todo)), flush=True)


if __name__ == "__main__":
    main()
