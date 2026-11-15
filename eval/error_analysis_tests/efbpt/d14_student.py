#!/usr/bin/env python3
"""
d14_student.py — D14 student job (freeze F4, 9.6): Qwen3-14B under vLLM.

Engine and sampling of the task's fair script: LLM(model, dtype="bfloat16",
max_model_len=8192, gpu_memory_utilization=0.85), SamplingParams(temperature=0.0,
max_tokens=N, stop=["<|im_end|>"]) with N read from the fair script's source;
one user message, the tokenizer's chat template with the generation prompt and
enable_thinking=True, as in the fair script.

  S_ur  the UR prompt (the fair script's user message)
  S_ro  UR prompt + one block between the last item line and the trailer:
        the translation label line, then the sealed R1 item block
  S_og  as S_ro with the original English item block
  S_tr  (CSQA, PIQA) UR prompt + the facts block (d1_eval_arms.facts_block) at the
        same place, facts = the sealed D14 teacher facts; a row without facts has
        no block
A row whose reading failed is not generated in S_ro and S_tr. The arms of one
task run one after another with one loaded engine. Chunks of 64 prompts per
call in row order; every finished chunk is appended and synced; a crashed arm
resumes and finished rows are never generated again.

Nothing is scored: no extractor is imported, rows carry no pred / correct / gold.

  --tasks csqa,piqa,gsm8k  [--arms S_ur,...]  --mode TEST|FULL  [--preflight-only]
TEST = rows 0-4 under outputs/efbpt/d14/_test/ (never a result).
"""
import argparse
import os
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_common as C  # noqa: E402
from d14_common import die  # noqa: E402


def build_unit(task, arm, mode, tok, cache):
    if task not in cache:
        rows = C.load_rows(task, mode)
        r1, r1_info = C.load_r1(task, mode, rows)
        facts = finfo = None
        if C.TASKS[task]["has_tr"]:
            facts, finfo = C.load_facts(task, mode, rows, r1)
        cache[task] = (rows, r1, r1_info, facts, finfo)
    rows, r1, r1_info, facts, finfo = cache[task]
    mx = C.fair_max_tokens(task)
    exp = []
    for u in C.build_users(task, arm, rows, r1, facts):
        e = OrderedDict([("rid", u["rid"]), ("status", u["status"]), ("prompt", None), ("n_prompt_tokens", None), ("n_facts", u.get("n_facts"))])
        if u["status"] == "ok":
            e["prompt"] = C.student_chat(tok, u["user"])
            if e["prompt"].count(u["user"]) != 1:
                die("%s %s: the chat prompt does not hold the user message exactly once" % (arm, u["rid"]))
            e["n_prompt_tokens"] = C.n_tokens(tok, e["prompt"])
            if e["n_prompt_tokens"] + mx > C.STUDENT_MAX_MODEL_LEN:
                e["status"] = "too_long"
        exp.append(e)
    return OrderedDict([("task", task), ("arm", arm), ("label", "%s %s" % (task, arm)), ("pth", C.arm_paths(task, arm, mode)),
                        ("expected", exp), ("max_tokens", mx), ("r1", r1_info), ("facts", finfo if arm == "S_tr" else None)])


def describe(unit):
    exp = unit["expected"]
    n = sorted(e["n_prompt_tokens"] for e in exp if e["n_prompt_tokens"] is not None)
    st = OrderedDict()
    for e in exp:
        st[e["status"]] = st.get(e["status"], 0) + 1
    extra = ""
    if unit["arm"] == "S_tr":
        extra = "  rows without facts (prompt = the S_ur prompt) %d" % sum(1 for e in exp if e["status"] == "ok" and not e["n_facts"])
    print("[unit] %-12s rows %d  status %s  prompt tokens min %s median %s max %s  max_tokens %d%s"
          % (unit["label"], len(exp), dict(st), n[0] if n else "-", n[len(n) // 2] if n else "-", n[-1] if n else "-", unit["max_tokens"], extra), flush=True)
    first = next((e for e in exp if e["status"] == "ok"), None)
    if first:
        C.log_full_prompt("%s, %s" % (unit["label"], first["rid"]), first["prompt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="csqa,piqa,gsm8k")
    ap.add_argument("--arms", default=None, help="comma list; default: every student arm of the task")
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    mode, tasks = args.mode, args.tasks.split(",")
    for t in tasks:
        if t not in C.TASKS:
            die("unknown task %r" % t)
    print("=" * 78)
    print("D14 student job   tasks %s   %s" % (tasks, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    pins = C.verify_pins()
    if mode == "FULL" and not args.preflight_only:
        C.require_committed()
    tok = C.load_tok(C.STUDENT_PATH)
    cache, units = {}, []
    for t in tasks:
        arms = [a for a in C.arms_of(t, "S") if args.arms is None or a in args.arms.split(",")]
        units += [build_unit(t, a, mode, tok, cache) for a in arms]
    if not units:
        die("nothing to do")
    for u in units:
        describe(u)
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No model loaded, nothing written.")
        return

    engine = OrderedDict([("model", C.STUDENT_PATH), ("dtype", "bfloat16"), ("max_model_len", C.STUDENT_MAX_MODEL_LEN),
                          ("gpu_memory_utilization", C.STUDENT_GPU_UTIL)])
    todo = []
    for u in units:
        start = OrderedDict([
            ("experiment", "D14"), ("arm", u["arm"]), ("task", u["task"]), ("mode", mode), ("engine", engine),
            ("sampling", OrderedDict([("temperature", 0.0), ("max_tokens", u["max_tokens"]), ("stop", C.STUDENT_STOP)])),
            ("chat_template", "one user message, add_generation_prompt=True, enable_thinking=True"), ("prompt_kind", C.ARMS[u["arm"]]["kind"]),
            ("chunk", C.CHUNK), ("n_rows", len(u["expected"])), ("rids", [e["rid"] for e in u["expected"]]),
            ("r1", u["r1"]), ("facts", u["facts"]), ("pins", pins), ("scored", False)])
        k = C.open_rows(u["pth"], u["expected"], start)
        if k is not None:
            todo.append((u, k))
    if not todo:
        print("[done] every requested arm is already sealed; no model loaded")
        return

    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != C.VLLM_VERSION:
        die("vLLM %s found, expected %s" % (vllm.__version__, C.VLLM_VERSION))
    print("\n[load] %s ..." % engine, flush=True)
    t0 = time.time()
    llm = LLM(model=C.STUDENT_PATH, dtype="bfloat16", max_model_len=C.STUDENT_MAX_MODEL_LEN, gpu_memory_utilization=C.STUDENT_GPU_UTIL)
    print("[load] %.0fs" % (time.time() - t0), flush=True)

    for u, k in todo:
        sp = SamplingParams(temperature=0.0, max_tokens=u["max_tokens"], stop=C.STUDENT_STOP)
        print("\n" + "=" * 70)
        print("ARM %s  rows %d..%d of %d  (max_tokens=%d, temperature 0, %d prompts per call)"
              % (u["label"], k, len(u["expected"]) - 1, len(u["expected"]), u["max_tokens"], C.CHUNK))
        print("=" * 70, flush=True)

        def gen(ok, sp=sp):
            return C.vllm_outputs(llm.generate([e["prompt"] for e in ok], sp, use_tqdm=False))

        def rec(e, o, u=u):
            return OrderedDict([("rid", e["rid"]), ("arm", u["arm"]), ("task", u["task"]), ("status", e["status"]), ("prompt", e["prompt"]),
                                ("generation", o["text"] if o else ""), ("finish_reason", o["finish_reason"] if o else None),
                                ("truncated", bool(o and o["finish_reason"] == "length")),
                                ("n_prompt_tokens", o["n_prompt_tokens"] if o else e["n_prompt_tokens"]),
                                ("n_gen_tokens", o["n_gen_tokens"] if o else None), ("n_facts", e["n_facts"])])

        C.generate_rows(u["pth"], u["expected"], k, gen, rec, u["label"])
        head = OrderedDict([("experiment", "D14"), ("arm", u["arm"]), ("task", u["task"]), ("mode", mode)])
        C.seal_rows(u["pth"], [e["rid"] for e in u["expected"]], head, OrderedDict([("vllm", vllm.__version__), ("engine", engine)]))
    print("\n[done] %s: %d arms sealed, not scored" % (mode, len(todo)), flush=True)


if __name__ == "__main__":
    main()
