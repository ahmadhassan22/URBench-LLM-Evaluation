#!/usr/bin/env python3
"""
d14_teacher.py — D14 teacher job (freeze F4, 9.7, 9.8): Llama-3.1-70B-Instruct-AWQ-INT4 under vLLM.

  facts  teacher facts for S_tr (CSQA and PIQA): the D11 teacher fact prompt unchanged
         (d11o_teacher.oracle_prompts on the d10_teacher prompt pair, imported), its question
         slot filled with the sealed R1 item block; max_tokens 512; d10_teacher.parse_teacher_json;
         the D10 verdict filter T2 unchanged; then the D14 filter. Removed lines are stored.
  arms   T_ur (UR prompt), T_r1 (EN prompt with the R1 item block), T_og (EN prompt with the
         original item block); one user message through the model's chat template; max_tokens 1024.
Engine: the D10 engine arguments (d10_teacher.VLLM_ARGS, imported) with max_num_seqs 16;
temperature 0, top_p 1, seed 0. Chunks of 64 prompts per call in row order; every finished
chunk is appended and synced; a crashed unit resumes and finished rows are never generated again.

Nothing is scored: no extractor is imported, rows carry no pred / correct / gold.

  --what facts|arms|all  --tasks csqa,piqa,gsm8k  --mode TEST|FULL  [--preflight-only]
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


def build_facts_unit(task, mode, tok, T):
    rows = C.load_rows(task, mode)
    r1, info = C.load_r1(task, mode, rows)
    exp = []
    for row in rows:
        block = r1[row["rid"]]
        e = OrderedDict([("rid", row["rid"]), ("status", "ok"), ("prompt", None), ("ids", None), ("r1_block", block)])
        if block is None:
            e["status"] = "reading_failed"
        else:
            e["prompt"], e["ids"] = C.teacher_render(tok, [{"role": "system", "content": T["facts_system"]},
                                                           {"role": "user", "content": C.facts_user(block)}])
            if e["prompt"].count(block) != 1:
                die("facts prompt of %s does not hold its R1 block exactly once" % row["rid"])
            if len(e["ids"]) + T["facts_max_tokens"] > T["vllm_args"]["max_model_len"]:
                e["status"] = "too_long"
        exp.append(e)
    return OrderedDict([("kind", "facts"), ("task", task), ("label", "facts %s" % task), ("pth", C.facts_paths(task, mode)),
                        ("expected", exp), ("max_tokens", T["facts_max_tokens"]), ("r1", info)])


def build_arm_unit(task, arm, mode, tok, T):
    rows = C.load_rows(task, mode)
    r1 = info = None
    if C.ARMS[arm]["needs_r1"]:
        r1, info = C.load_r1(task, mode, rows)
    exp = []
    for u in C.build_users(task, arm, rows, r1):
        e = OrderedDict([("rid", u["rid"]), ("status", u["status"]), ("prompt", None), ("ids", None)])
        if u["status"] == "ok":
            e["prompt"], e["ids"] = C.teacher_render(tok, [{"role": "user", "content": u["user"]}])
            if e["prompt"].count(u["user"]) != 1:
                die("%s %s: the chat prompt does not hold the user message exactly once" % (arm, u["rid"]))
            if len(e["ids"]) + C.TEACHER_ARM_MAX_TOKENS > T["vllm_args"]["max_model_len"]:
                e["status"] = "too_long"
        exp.append(e)
    return OrderedDict([("kind", "arm"), ("task", task), ("arm", arm), ("label", "%s %s" % (task, arm)), ("pth", C.arm_paths(task, arm, mode)),
                        ("expected", exp), ("max_tokens", C.TEACHER_ARM_MAX_TOKENS), ("r1", info)])


def describe(unit):
    exp = unit["expected"]
    n = sorted(len(e["ids"]) for e in exp if e["ids"] is not None)
    st = OrderedDict()
    for e in exp:
        st[e["status"]] = st.get(e["status"], 0) + 1
    print("[unit] %-12s rows %d  status %s  prompt tokens min %s median %s max %s  max_tokens %d"
          % (unit["label"], len(exp), dict(st), n[0] if n else "-", n[len(n) // 2] if n else "-", n[-1] if n else "-", unit["max_tokens"]), flush=True)
    first = next((e for e in exp if e["status"] == "ok"), None)
    if first:
        C.log_full_prompt("%s, %s" % (unit["label"], first["rid"]), first["prompt"])


def facts_record(e, o, T):
    rec = OrderedDict([("rid", e["rid"]), ("status", e["status"]), ("r1_block", e["r1_block"]), ("prompt", e["prompt"]),
                       ("generation", o["text"] if o else ""), ("finish_reason", o["finish_reason"] if o else None),
                       ("truncated", bool(o and o["finish_reason"] == "length")),
                       ("n_prompt_tokens", len(e["ids"]) if e["ids"] is not None else None), ("n_gen_tokens", o["n_gen_tokens"] if o else None)])
    ok, ents, raw = T["parse"](o["text"]) if o else (False, [], [])
    kept, removed_verdict = T["verdict_filter"](raw) if ok else ([], [])
    kept2, removed_d14 = C.d14_filter(kept)
    facts = [C.one_line(f) for f in kept2]
    facts = [f for f in facts if f]
    rec.update(OrderedDict([("parse_ok", bool(ok)), ("entities", ents), ("facts_raw", raw), ("removed_verdict", removed_verdict),
                            ("removed_d14", removed_d14), ("facts", facts)]))
    return rec


def arm_record(e, o, arm, task):
    return OrderedDict([("rid", e["rid"]), ("arm", arm), ("task", task), ("status", e["status"]), ("prompt", e["prompt"]),
                        ("generation", o["text"] if o else ""), ("finish_reason", o["finish_reason"] if o else None),
                        ("truncated", bool(o and o["finish_reason"] == "length")),
                        ("n_prompt_tokens", len(e["ids"]) if e["ids"] is not None else None), ("n_gen_tokens", o["n_gen_tokens"] if o else None)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=("facts", "arms", "all"))
    ap.add_argument("--tasks", default="csqa,piqa,gsm8k")
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    mode, tasks = args.mode, args.tasks.split(",")
    for t in tasks:
        if t not in C.TASKS:
            die("unknown task %r" % t)
    print("=" * 78)
    print("D14 teacher job   what %s   tasks %s   %s" % (args.what, tasks, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    pins = C.verify_pins()
    if mode == "FULL" and not args.preflight_only:
        C.require_committed()
    T = C.teacher_frozen()
    tok = C.load_tok(T["path"])

    plan = []        # (kind, task, arm): facts units first; an arm unit is built when its turn comes
    if args.what in ("facts", "all"):
        plan += [("facts", t, None) for t in tasks if C.TASKS[t]["has_tr"]]
    if args.what in ("arms", "all"):
        plan += [("arm", t, a) for t in tasks for a in C.arms_of(t, "T")]
    if not plan:
        die("nothing to do for --what %s --tasks %s" % (args.what, args.tasks))

    def build(kind, task, arm):
        return build_facts_unit(task, mode, tok, T) if kind == "facts" else build_arm_unit(task, arm, mode, tok, T)

    units = [build(*p) for p in plan]
    for u in units:
        describe(u)
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No model loaded, nothing written.")
        return

    todo = []
    for u in units:
        start = OrderedDict([
            ("experiment", "D14"), ("what", u["kind"]), ("task", u["task"]), ("arm", u.get("arm")), ("mode", mode),
            ("teacher_path", T["path"]), ("vllm_version", T["vllm_version"]), ("vllm_args", T["vllm_args"]), ("sampling", T["sampling"]),
            ("max_tokens", u["max_tokens"]), ("chunk", C.CHUNK), ("n_rows", len(u["expected"])), ("rids", [e["rid"] for e in u["expected"]]),
            ("r1", u["r1"]), ("pins", pins), ("scored", False)])
        if u["kind"] == "facts":
            start.update(OrderedDict([("system_message", T["facts_system"]), ("user_template", T["facts_user_template"]),
                                      ("verdict_filter", [T["verdict_re"], T["yesno_re"]]),
                                      ("d14_filter", [C.D14_WORD_RE.pattern + " (case-insensitive)", C.D14_LABEL_RE.pattern + " (case-sensitive)"])]))
        k = C.open_rows(u["pth"], u["expected"], start)
        if k is not None:
            todo.append((u, k))
    if not todo:
        print("[done] every requested unit is already sealed; no model loaded")
        return

    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != T["vllm_version"]:
        die("vLLM %s found, the D10 freeze requires %s" % (vllm.__version__, T["vllm_version"]))
    print("\n[load] %s  args %s ..." % (T["path"], dict(T["vllm_args"])), flush=True)
    t0 = time.time()
    llm = LLM(model=T["path"], **T["vllm_args"])
    print("[load] %.0fs" % (time.time() - t0), flush=True)

    for u, k in todo:
        params = SamplingParams(temperature=T["sampling"]["temperature"], top_p=T["sampling"]["top_p"], seed=T["sampling"]["seed"], max_tokens=u["max_tokens"])
        print("\n" + "=" * 70)
        print("UNIT %s  rows %d..%d of %d  (max_tokens=%d, greedy, max_num_seqs=%d, %d prompts per call)"
              % (u["label"], k, len(u["expected"]) - 1, len(u["expected"]), u["max_tokens"], T["vllm_args"]["max_num_seqs"], C.CHUNK))
        print("=" * 70, flush=True)

        def gen(ok, params=params):
            return C.vllm_outputs(llm.generate([{"prompt_token_ids": e["ids"]} for e in ok], params, use_tqdm=False))

        if u["kind"] == "facts":
            C.generate_rows(u["pth"], u["expected"], k, gen, lambda e, o: facts_record(e, o, T), u["label"])
            recs = C.read_rows(u["pth"]["rows"])
            extra = OrderedDict([
                ("parse_ok", sum(1 for r in recs if r["parse_ok"])), ("rows_without_facts", sum(1 for r in recs if not r["facts"])),
                ("facts_total", sum(len(r["facts"]) for r in recs)),
                ("removed_verdict_lines", sum(len(r["removed_verdict"]) for r in recs)), ("removed_d14_lines", sum(len(r["removed_d14"]) for r in recs)),
                ("rows_with_a_d14_removed_line", sum(1 for r in recs if r["removed_d14"])), ("vllm", vllm.__version__)])
        else:
            C.generate_rows(u["pth"], u["expected"], k, gen, lambda e, o, u=u: arm_record(e, o, u["arm"], u["task"]), u["label"])
            extra = OrderedDict([("vllm", vllm.__version__)])
        head = OrderedDict([("experiment", "D14"), ("what", u["kind"]), ("task", u["task"]), ("arm", u.get("arm")), ("mode", mode)])
        C.seal_rows(u["pth"], [e["rid"] for e in u["expected"]], head, extra)
    print("\n[done] %s: %d units sealed, not scored" % (mode, len(todo)), flush=True)


if __name__ == "__main__":
    main()
