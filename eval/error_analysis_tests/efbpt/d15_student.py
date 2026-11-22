#!/usr/bin/env python3
"""
d15_student.py — D15 student job (freeze N3, N4, 11.5, 11.7): Qwen3-14B under vLLM.

Engine of the CSQA fair script: LLM(model, dtype="bfloat16", max_model_len=8192,
gpu_memory_utilization=0.85).
  sl    Qwen3-14B translating by itself: system = the D14 F3 message, user = the Urdu item block,
        chat template with enable_thinking=False; SamplingParams(temperature=0.0, max_tokens=512,
        stop=["<|im_end|>"]); the text after the last </think> if any. Valid = labels Question,
        A, B, C, D in order; an invalid output is not repeated (reading_failed for S_self).
  arms  SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|im_end|>"]) (CSQA fair values);
        one user message, chat template with enable_thinking=True:
        S_ur   UR prompt
        S_ro   UR + the translation label line + the R1 item block
        S_self UR + the translation label line + the SL item block
        S_tr   UR + the facts block of the sealed teacher facts (no block for a row without facts)
A row without a valid reading is not generated in the arm that needs it. Units run one after
another with one loaded engine; an arm unit is built only when its turn comes (so `sl` can be
sealed earlier in the same job). Chunks of 64 prompts per call in run order; every finished chunk
is appended and synced; a crashed unit resumes and finished rows are never generated again.

Nothing is scored: no extractor is imported, rows carry no pred / correct / gold.

  --what sl|arms|sl+arms  [--arms S_ur,...]  --mode TEST|FULL  [--preflight-only]
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

ENGINE = OrderedDict([("model", C.STUDENT_PATH), ("dtype", "bfloat16"), ("max_model_len", C.STUDENT_MAX_MODEL_LEN),
                      ("gpu_memory_utilization", C.STUDENT_GPU_UTIL)])


def build_sl_unit(mode, tok):
    exp = []
    for row in C.load_rows(mode):
        user = C.ur_block(row["item"])
        e = OrderedDict([("rid", row["rid"]), ("status", "ok"), ("user", user)])
        e["prompt"] = C.student_tr_chat(tok, user)
        if e["prompt"].count(user) != 1:
            die("SL prompt of %s does not hold its Urdu item block exactly once" % row["rid"])
        e["n_prompt_tokens"] = C.n_tokens(tok, e["prompt"])
        if e["n_prompt_tokens"] + C.LOCAL_TR_MAX_TOKENS > C.STUDENT_MAX_MODEL_LEN:
            e["status"] = "too_long"
        exp.append(e)
    return OrderedDict([("kind", "sl"), ("label", "SL"), ("pth", C.tr_paths("SL", mode)), ("expected", exp),
                        ("max_tokens", C.LOCAL_TR_MAX_TOKENS), ("readings", None), ("facts", None)])


def build_arm_unit(arm, mode, tok):
    rows = C.load_rows(mode)
    spec = C.ARMS[arm]
    readings, infos, facts, finfo = OrderedDict(), [], None, None
    if spec["reading"]:
        readings[spec["reading"]], inf = C.load_reading(spec["reading"], mode, rows)
        infos.append(inf)
    if spec["facts"]:
        facts, finfo = C.load_facts(mode, rows, readings["R1"])
    mx = C.student_max_tokens()
    exp = []
    for u in C.build_users(arm, rows, readings, facts):
        e = OrderedDict([("rid", u["rid"]), ("status", u["status"]), ("prompt", None), ("n_prompt_tokens", None), ("n_facts", u.get("n_facts"))])
        if u["status"] == "ok":
            e["prompt"] = C.student_chat(tok, u["user"])
            if e["prompt"].count(u["user"]) != 1:
                die("%s %s: the chat prompt does not hold the user message exactly once" % (arm, u["rid"]))
            e["n_prompt_tokens"] = C.n_tokens(tok, e["prompt"])
            if e["n_prompt_tokens"] + mx > C.STUDENT_MAX_MODEL_LEN:
                e["status"] = "too_long"
        exp.append(e)
    return OrderedDict([("kind", "arm"), ("arm", arm), ("label", arm), ("pth", C.arm_paths(arm, mode)), ("expected", exp),
                        ("max_tokens", mx), ("readings", infos), ("facts", finfo)])


def describe(unit):
    exp = unit["expected"]
    n = sorted(e["n_prompt_tokens"] for e in exp if e.get("n_prompt_tokens") is not None)
    st = Counter(e["status"] for e in exp)
    extra = ""
    if unit.get("arm") == "S_tr":
        extra = "  rows without facts (prompt = the S_ur prompt) %d" % sum(1 for e in exp if e["status"] == "ok" and not e["n_facts"])
    print("[unit] %-8s rows %d  status %s  prompt tokens min %s median %s max %s  max_tokens %d%s"
          % (unit["label"], len(exp), dict(st), n[0] if n else "-", n[len(n) // 2] if n else "-", n[-1] if n else "-", unit["max_tokens"], extra), flush=True)
    first = next((e for e in exp if e["status"] == "ok"), None)
    if first:
        C.log_full_prompt("%s, %s" % (unit["label"], first["rid"]), first["prompt"])


def sl_record(e, o):
    gen = o["text"] if o else ""
    text = C.after_think(gen) if o else ""
    block = C.parse_translation(text) if o else None
    status = e["status"] if e["status"] != "ok" else ("ok" if block is not None else "reading_failed")
    return OrderedDict([("rid", e["rid"]), ("status", status), ("system", C.R1_SYSTEM), ("user", e["user"]), ("prompt", e["prompt"]),
                        ("generation", gen), ("text", text), ("block", block), ("finish_reason", o["finish_reason"] if o else None),
                        ("truncated", bool(o and o["finish_reason"] == "length")), ("flags", C.translation_flags(text)),
                        ("n_prompt_tokens", o["n_prompt_tokens"] if o else e["n_prompt_tokens"]), ("n_gen_tokens", o["n_gen_tokens"] if o else None)])


def arm_record(e, o, arm):
    return OrderedDict([("rid", e["rid"]), ("arm", arm), ("status", e["status"]), ("prompt", e["prompt"]),
                        ("generation", o["text"] if o else ""), ("finish_reason", o["finish_reason"] if o else None),
                        ("truncated", bool(o and o["finish_reason"] == "length")),
                        ("n_prompt_tokens", o["n_prompt_tokens"] if o else e["n_prompt_tokens"]),
                        ("n_gen_tokens", o["n_gen_tokens"] if o else None), ("n_facts", e["n_facts"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=("sl", "arms", "sl+arms"))
    ap.add_argument("--arms", default=None, help="comma list of student arms; default: all four")
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    mode = args.mode
    arms = C.arms_of("S") if not args.arms else args.arms.split(",")
    for a in arms:
        if a not in C.arms_of("S"):
            die("%s is not a student arm" % a)
    print("=" * 78)
    print("D15 student job   what %s   %s" % (args.what, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    pins = C.verify_pins()
    if mode == "FULL" and not args.preflight_only:
        C.require_committed()
    tok = C.load_tok(C.STUDENT_PATH)
    plan = ([("sl", None)] if args.what in ("sl", "sl+arms") else []) + ([("arm", a) for a in arms] if args.what in ("arms", "sl+arms") else [])

    def build(kind, arm):
        return build_sl_unit(mode, tok) if kind == "sl" else build_arm_unit(arm, mode, tok)

    # units that can be built now are described; an arm that needs SL waits for SL when SL runs in this job
    first = []
    for kind, arm in plan:
        if kind == "arm" and C.ARMS[arm]["reading"] == "SL" and any(k == "sl" for k, _ in plan) and not C.is_sealed(C.tr_paths("SL", mode)):
            print("[unit] %-8s built after SL is sealed in this job" % arm)
            continue
        u = build(kind, arm)
        describe(u)
        first.append((kind, arm))
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No model loaded, nothing written.")
        return

    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != C.VLLM_VERSION:
        die("vLLM %s found, expected %s" % (vllm.__version__, C.VLLM_VERSION))
    llm = None
    for kind, arm in plan:
        u = build(kind, arm)
        if (kind, arm) not in first:
            describe(u)
        start = OrderedDict([
            ("experiment", "D15"), ("what", kind), ("arm", arm), ("mode", mode), ("engine", ENGINE),
            ("sampling", OrderedDict([("temperature", 0.0), ("max_tokens", u["max_tokens"]), ("stop", C.STUDENT_STOP)])),
            ("chat_template", "system + user, add_generation_prompt=True, enable_thinking=False" if kind == "sl"
             else "one user message, add_generation_prompt=True, enable_thinking=True"),
            ("prompt_kind", C.ARMS[arm]["kind"] if arm else "translation"), ("chunk", C.CHUNK), ("n_rows", len(u["expected"])),
            ("rids", [e["rid"] for e in u["expected"]]), ("readings", u["readings"]), ("facts", u["facts"]), ("pins", pins), ("scored", False)])
        k = C.open_rows(u["pth"], u["expected"], start, ok_alias=("reading_failed",) if kind == "sl" else ())
        if k is None:
            continue
        if llm is None:
            print("\n[load] %s ..." % dict(ENGINE), flush=True)
            t0 = time.time()
            llm = LLM(model=C.STUDENT_PATH, dtype="bfloat16", max_model_len=C.STUDENT_MAX_MODEL_LEN, gpu_memory_utilization=C.STUDENT_GPU_UTIL)
            print("[load] %.0fs" % (time.time() - t0), flush=True)
        sp = SamplingParams(temperature=0.0, max_tokens=u["max_tokens"], stop=C.STUDENT_STOP)
        print("\n" + "=" * 70)
        print("UNIT %s  rows %d..%d of %d  (max_tokens=%d, temperature 0, %d prompts per call)"
              % (u["label"], k, len(u["expected"]) - 1, len(u["expected"]), u["max_tokens"], C.CHUNK))
        print("=" * 70, flush=True)

        def gen(ok, sp=sp):
            return C.vllm_outputs(llm.generate([e["prompt"] for e in ok], sp, use_tqdm=False))

        head = OrderedDict([("experiment", "D15"), ("what", kind), ("arm", arm), ("mode", mode)])
        if kind == "sl":
            C.generate_rows(u["pth"], u["expected"], k, gen, sl_record, u["label"])
            recs = C.read_rows(u["pth"]["rows"])
            fl = Counter()
            for r in recs:
                for kk, v in r["flags"].items():
                    fl[kk] += int(v)
                fl["any"] += int(any(r["flags"].values()))
            st = Counter(r["status"] for r in recs)
            extra = OrderedDict([("reading_failed", st.get("reading_failed", 0)), ("too_long", st.get("too_long", 0)),
                                 ("accepted_readings_that_reached_max_tokens", sum(1 for r in recs if r["status"] == "ok" and r["truncated"])),
                                 ("flags", OrderedDict(fl)), ("vllm", vllm.__version__), ("engine", ENGINE)])
        else:
            C.generate_rows(u["pth"], u["expected"], k, gen, lambda e, o, arm=arm: arm_record(e, o, arm), u["label"])
            extra = OrderedDict([("vllm", vllm.__version__), ("engine", ENGINE)])
        C.seal_rows(u["pth"], [e["rid"] for e in u["expected"]], head, extra)
    print("\n[done] %s: units sealed or already sealed: %s; not scored" % (mode, [a or k for k, a in plan]), flush=True)


if __name__ == "__main__":
    main()
