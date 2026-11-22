#!/usr/bin/env python3
"""
d15_teacher.py — D15 teacher job (freeze N3, N4, 11.5-11.8): Llama-3.1-70B-Instruct-AWQ-INT4, vLLM.

  tl     the teacher translating by itself: system = the D14 F3 message, user = the Urdu item
         block; the D10 engine (d10_teacher.VLLM_ARGS unchanged, max_num_seqs 1), temperature 0,
         top_p 1, seed 0, max_tokens 512. Valid = labels Question, A, B, C, D in order; an invalid
         output is not repeated (reading_failed for T_self).
  facts  teacher facts for S_tr: the D14 facts unit unchanged (d14_teacher.facts_record, imported)
         with the sealed R1 item block in the question slot; max_tokens 512; D14 engine.
  arms   T_ur (UR), T_ue (EN instruction + Urdu item block), T_r1 (EN with R1), T_self (EN with TL);
         one user message through the chat template; max_tokens 1024; the D14 engine =
         d10_teacher.VLLM_ARGS with max_num_seqs 16 and max_num_batched_tokens 4096 (D14 A1.2).
`tl` runs alone in its job (its engine differs); facts and arms share one engine.
Chunks of 64 prompts per call in run order; every finished chunk is appended and synced; a
crashed unit resumes and finished rows are never generated again.

Nothing is scored: no extractor is imported, rows carry no pred / correct / gold.

  --what tl|facts|arms|facts+arms  [--arms T_ur,...]  --mode TEST|FULL  [--preflight-only]
TEST = the first 5 sampled rows under outputs/efbpt/d15/_test/ (never a result).
"""
import argparse
import os
import sys
import time
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d15_common as C  # noqa: E402
from d15_common import die  # noqa: E402
import d14_teacher as T14  # noqa: E402  (facts_record, arm_record: the D14 record makers, unchanged)


def build_tl_unit(mode, tok, T):
    exp = []
    for row in C.load_rows(mode):
        user = C.ur_block(row["item"])
        e = OrderedDict([("rid", row["rid"]), ("status", "ok"), ("user", user)])
        e["prompt"], e["ids"] = C.teacher_render(tok, [{"role": "system", "content": C.R1_SYSTEM}, {"role": "user", "content": user}])
        if e["prompt"].count(user) != 1:
            die("TL prompt of %s does not hold its Urdu item block exactly once" % row["rid"])
        if len(e["ids"]) + C.LOCAL_TR_MAX_TOKENS > T["d10_args"]["max_model_len"]:
            e["status"] = "too_long"
        exp.append(e)
    return OrderedDict([("kind", "tl"), ("label", "TL"), ("pth", C.tr_paths("TL", mode)), ("expected", exp),
                        ("max_tokens", C.LOCAL_TR_MAX_TOKENS), ("readings", None)])


def build_facts_unit(mode, tok, T):
    rows = C.load_rows(mode)
    r1, info = C.load_reading("R1", mode, rows)
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
    return OrderedDict([("kind", "facts"), ("label", "facts"), ("pth", C.facts_paths(mode)), ("expected", exp),
                        ("max_tokens", T["facts_max_tokens"]), ("readings", [info])])


def build_arm_unit(arm, mode, tok, T):
    rows = C.load_rows(mode)
    readings, infos = OrderedDict(), []
    rd = C.ARMS[arm]["reading"]
    if rd:
        readings[rd], inf = C.load_reading(rd, mode, rows)
        infos.append(inf)
    exp = []
    for u in C.build_users(arm, rows, readings):
        e = OrderedDict([("rid", u["rid"]), ("status", u["status"]), ("prompt", None), ("ids", None)])
        if u["status"] == "ok":
            e["prompt"], e["ids"] = C.teacher_render(tok, [{"role": "user", "content": u["user"]}])
            if e["prompt"].count(u["user"]) != 1:
                die("%s %s: the chat prompt does not hold the user message exactly once" % (arm, u["rid"]))
            if len(e["ids"]) + C.TEACHER_ARM_MAX_TOKENS > T["vllm_args"]["max_model_len"]:
                e["status"] = "too_long"
        exp.append(e)
    return OrderedDict([("kind", "arm"), ("arm", arm), ("label", arm), ("pth", C.arm_paths(arm, mode)), ("expected", exp),
                        ("max_tokens", C.TEACHER_ARM_MAX_TOKENS), ("readings", infos)])


def describe(unit):
    exp = unit["expected"]
    n = sorted(len(e["ids"]) for e in exp if e.get("ids") is not None)
    st = Counter(e["status"] for e in exp)
    print("[unit] %-8s rows %d  status %s  prompt tokens min %s median %s max %s  max_tokens %d"
          % (unit["label"], len(exp), dict(st), n[0] if n else "-", n[len(n) // 2] if n else "-", n[-1] if n else "-", unit["max_tokens"]), flush=True)
    first = next((e for e in exp if e["status"] == "ok"), None)
    if first:
        C.log_full_prompt("%s, %s" % (unit["label"], first["rid"]), first["prompt"])


def tl_record(e, o):
    text = o["text"] if o else ""
    block = C.parse_translation(text) if o else None
    status = e["status"] if e["status"] != "ok" else ("ok" if block is not None else "reading_failed")
    return OrderedDict([("rid", e["rid"]), ("status", status), ("system", C.R1_SYSTEM), ("user", e["user"]), ("prompt", e["prompt"]),
                        ("generation", text), ("text", text), ("block", block), ("finish_reason", o["finish_reason"] if o else None),
                        ("truncated", bool(o and o["finish_reason"] == "length")), ("flags", C.translation_flags(text)),
                        ("n_prompt_tokens", len(e["ids"]) if e["ids"] is not None else None), ("n_gen_tokens", o["n_gen_tokens"] if o else None)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=("tl", "facts", "arms", "facts+arms"))
    ap.add_argument("--arms", default=None, help="comma list of teacher arms; default: all four")
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    mode = args.mode
    arms = C.arms_of("T") if not args.arms else args.arms.split(",")
    for a in arms:
        if a not in C.arms_of("T"):
            die("%s is not a teacher arm" % a)
    print("=" * 78)
    print("D15 teacher job   what %s   %s" % (args.what, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    pins = C.verify_pins()
    if mode == "FULL" and not args.preflight_only:
        C.require_committed()
    T = C.teacher_frozen()
    import d10_teacher
    T["d10_args"] = OrderedDict(d10_teacher.VLLM_ARGS)          # the D10 engine, unchanged (TL)
    tok = C.load_tok(T["path"])

    plan = []
    if args.what == "tl":
        plan = [("tl", None)]
    if args.what in ("facts", "facts+arms"):
        plan.append(("facts", None))
    if args.what in ("arms", "facts+arms"):
        plan += [("arm", a) for a in arms]

    def build(kind, arm):
        if kind == "tl":
            return build_tl_unit(mode, tok, T)
        if kind == "facts":
            return build_facts_unit(mode, tok, T)
        return build_arm_unit(arm, mode, tok, T)

    units = [build(*p) for p in plan]
    for u in units:
        describe(u)
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No model loaded, nothing written.")
        return
    pending = [u for u in units if not C.is_sealed(u["pth"])]
    if not pending:
        print("[done] every requested unit is already sealed; no model loaded")
        return

    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != T["vllm_version"]:
        die("vLLM %s found, the D10 freeze requires %s" % (vllm.__version__, T["vllm_version"]))
    vargs = T["d10_args"] if args.what == "tl" else T["vllm_args"]
    engine_name = "D10 engine (d10_teacher.VLLM_ARGS unchanged)" if args.what == "tl" else "D14 engine (D10 arguments, max_num_seqs 16, max_num_batched_tokens 4096; D14 A1.2)"
    print("\n[load] %s  %s  args %s ..." % (T["path"], engine_name, dict(vargs)), flush=True)
    t0 = time.time()
    llm = LLM(model=T["path"], **vargs)
    print("[load] %.0fs" % (time.time() - t0), flush=True)

    todo = []
    for u in units:
        start = OrderedDict([
            ("experiment", "D15"), ("what", u["kind"]), ("arm", u.get("arm")), ("mode", mode),
            ("teacher_path", T["path"]), ("vllm_version", T["vllm_version"]), ("engine", engine_name), ("vllm_args", vargs),
            ("sampling", T["sampling"]), ("max_tokens", u["max_tokens"]), ("chunk", C.CHUNK), ("n_rows", len(u["expected"])),
            ("rids", [e["rid"] for e in u["expected"]]), ("readings", u["readings"]), ("pins", pins), ("scored", False)])
        if u["kind"] == "tl":
            start.update(OrderedDict([("system_message", C.R1_SYSTEM), ("user_message", "the Urdu item block (freeze 11.4)")]))
        if u["kind"] == "facts":
            start.update(OrderedDict([("system_message", T["facts_system"]), ("user_template", T["facts_user_template"]),
                                      ("verdict_filter", [T["verdict_re"], T["yesno_re"]]),
                                      ("d14_filter", [C.C14.D14_WORD_RE.pattern + " (case-insensitive)", C.C14.D14_LABEL_RE.pattern + " (case-sensitive)"])]))
        k = C.open_rows(u["pth"], u["expected"], start, ok_alias=("reading_failed",) if u["kind"] == "tl" else ())
        if k is not None:
            todo.append((u, k))

    for u, k in todo:
        params = SamplingParams(temperature=T["sampling"]["temperature"], top_p=T["sampling"]["top_p"], seed=T["sampling"]["seed"], max_tokens=u["max_tokens"])
        print("\n" + "=" * 70)
        print("UNIT %s  rows %d..%d of %d  (max_tokens=%d, greedy, max_num_seqs=%d, %d prompts per call)"
              % (u["label"], k, len(u["expected"]) - 1, len(u["expected"]), u["max_tokens"], vargs["max_num_seqs"], C.CHUNK))
        print("=" * 70, flush=True)

        def gen(ok, params=params):
            return C.vllm_outputs(llm.generate([{"prompt_token_ids": e["ids"]} for e in ok], params, use_tqdm=False))

        head = OrderedDict([("experiment", "D15"), ("what", u["kind"]), ("arm", u.get("arm")), ("mode", mode)])
        if u["kind"] == "tl":
            C.generate_rows(u["pth"], u["expected"], k, gen, tl_record, u["label"])
            recs = C.read_rows(u["pth"]["rows"])
            fl = Counter()
            for r in recs:
                for kk, v in r["flags"].items():
                    fl[kk] += int(v)
                fl["any"] += int(any(r["flags"].values()))
            st = Counter(r["status"] for r in recs)
            extra = OrderedDict([("reading_failed", st.get("reading_failed", 0)), ("too_long", st.get("too_long", 0)),
                                 ("accepted_readings_that_reached_max_tokens", sum(1 for r in recs if r["status"] == "ok" and r["truncated"])),
                                 ("flags", OrderedDict(fl)), ("vllm", vllm.__version__), ("vllm_args", vargs)])
        elif u["kind"] == "facts":
            C.generate_rows(u["pth"], u["expected"], k, gen, lambda e, o: T14.facts_record(e, o, T), u["label"])
            recs = C.read_rows(u["pth"]["rows"])
            extra = OrderedDict([
                ("parse_ok", sum(1 for r in recs if r["parse_ok"])), ("rows_without_facts", sum(1 for r in recs if not r["facts"])),
                ("facts_total", sum(len(r["facts"]) for r in recs)),
                ("removed_verdict_lines", sum(len(r["removed_verdict"]) for r in recs)), ("removed_d14_lines", sum(len(r["removed_d14"]) for r in recs)),
                ("rows_with_a_d14_removed_line", sum(1 for r in recs if r["removed_d14"])), ("vllm", vllm.__version__), ("vllm_args", vargs)])
        else:
            C.generate_rows(u["pth"], u["expected"], k, gen, lambda e, o, u=u: T14.arm_record(e, o, u["arm"], "urdummlu"), u["label"])
            extra = OrderedDict([("vllm", vllm.__version__), ("vllm_args", vargs)])
        C.seal_rows(u["pth"], [e["rid"] for e in u["expected"]], head, extra)
    print("\n[done] %s: %d units sealed, not scored" % (mode, len(todo)), flush=True)


if __name__ == "__main__":
    main()
