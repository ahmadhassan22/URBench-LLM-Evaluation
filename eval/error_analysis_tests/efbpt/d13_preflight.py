#!/usr/bin/env python3
"""
d13_preflight.py — D13 CPU preflight (freeze C7). No model, no generation, no
network, no API key, no file written.

Checks, all fatal:
  - contracts by pinned MD5 (D13 freeze, D11 freeze and amendments, D10, D8);
  - sealed reference files by MD5 (freeze §1);
  - row counts: DEV200 200 (87 / 113), eval458 458 (221 / 237), overlap 0;
  - reading MD5s R1 / R3 on both sets = R_SEAL = freeze §1;
  - the first-50 rule of L5: 408 rows remain;
  - the weights of the three other students are on the cluster, every shard;
  - every prompt of every arm that can be built without a model run, on both
    sets, with its leak guard (freeze §7.5), plus the TEST-5 shape (DEV200 rows
    0-4) of every arm with the prompt-identity checks of §7.1 / §7.4.
RAG student prompts need the sealed retrieval files; they are built here when
those files exist and otherwise in the student job, with the same code.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d13_preflight.py
"""
import os
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d13_common as C  # noqa: E402
from d13_common import die, md5_file, read_jsonl  # noqa: E402


def first50_check(ctx_eval):
    path, want = C.FIRST50_FILE
    got = md5_file(path)
    if got != want:
        die("%s MD5 %s != the D13 freeze §3 value %s" % (path, got, want))
    ev = read_jsonl(path)
    ids = [r["qid"] for r in ev]
    mine = [r["qid"] for r in ctx_eval["rows_all"]]
    if ids != mine:
        die("%s does not hold the eval458 qids in the loader's order" % path)
    drop = set(ids[:C.FIRST50_N])
    left = [r for r in ctx_eval["rows_all"] if r["qid"] not in drop]
    if len(drop) != C.FIRST50_N or len(left) != C.L5_ROWS:
        die("L5: %d rows remain after dropping the first %d, expected %d" % (len(left), C.FIRST50_N, C.L5_ROWS))
    print("[ok] L5: %s (MD5 %s) = the eval458 qids in loader order; first %d dropped, %d rows remain"
          % (path, got, C.FIRST50_N, len(left)))


def main():
    t0 = time.time()
    print("=" * 78)
    print("D13 CPU PREFLIGHT (freeze C7)  —  no model, no generation, nothing written")
    print("=" * 78, flush=True)
    C.verify_contracts()
    sealed = C.verify_sealed()
    print("[ok] %d sealed reference files equal the MD5s of freeze §1" % len(sealed))
    for key, spec in C.OTHER_STUDENTS.items():
        ok, have, n = C.weights_present(spec["path"])
        if not ok:
            die("weights of %s are not complete on the cluster (%d/%d shards at %s): STOP and report, no download" % (spec["label"], have, n, spec["path"]))
        print("[ok] weights %-22s %d/%d shards  %s" % (spec["label"], have, n, spec["path"]))

    summary = []
    ctxs = OrderedDict()
    for set_name in C.SETS:
        print("\n" + "-" * 78 + "\nSET %s (all rows)\n" % set_name + "-" * 78, flush=True)
        ctxs[set_name] = C.make_ctx(set_name, "FULL")
    print("\n" + "-" * 78 + "\nTEST-5 shape: DEV200 rows 0-4\n" + "-" * 78, flush=True)
    ctx_test = C.make_ctx("dev200", "TEST")
    first50_check(ctxs["eval458"])

    def runs_of(arm):
        """(label, ctx) of every run of an arm: its FULL sets, and TEST-5."""
        return [(s, ctxs[s]) for s in C.ARMS[arm]["sets"]] + [("TEST-5", ctx_test)]

    # ---- Group 1: teacher arms ----
    print("\n" + "=" * 78 + "\nGROUP 1  teacher arms (Llama-3.1-70B-AWQ tokenizer)\n" + "=" * 78, flush=True)
    from transformers import AutoTokenizer
    from d10_teacher import TEACHER_PATH
    tok_t = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    for arm in C.TEACHER_ARMS:
        for label, ctx in runs_of(arm):
            ps, _info = C.build_teacher_prompts(arm, ctx, tok_t)
            summary.append((arm, label, len(ps)))
        if arm != "T_direct_F_eval":
            C.log_full_prompt("%s, TEST-5 row 0" % arm, ps[0]["prompt"])

    # ---- Group 1: API arms ----
    print("\n" + "=" * 78 + "\nGROUP 1  API arms (messages only; no call, no key)\n" + "=" * 78, flush=True)
    for arm in C.API_ARMS:
        for label, ctx in runs_of(arm):
            ms, _info = C.build_api_messages(arm, ctx)
            summary.append((arm, label, len(ms)))

    # ---- Groups 2 and 3: Qwen3-14B student ----
    print("\n" + "=" * 78 + "\nGROUPS 2 and 3  Qwen3-14B student arms\n" + "=" * 78, flush=True)
    tok14 = C.load_tok(C.MODEL_PATH)
    for label, ctx in runs_of("RO_eval"):
        ps, _info = C.build_ro_prompts(ctx, tok14)
        summary.append(("RO_eval", label, len(ps)))
    for arm in C.RAG_ARMS:
        for label, ctx in runs_of(arm):
            C.guard_rag_queries(arm, ctx)
            pth = C.retrieval_paths(arm, ctx["set"], ctx["mode"])
            if os.path.exists(pth["seal"]):
                ps, _info = C.build_rag_prompts(arm, ctx, tok14)
                summary.append((arm, label, len(ps)))
            else:
                print("[note] %s %s: no sealed retrieval file yet; queries guarded here, student prompts are built and guarded in the student job" % (arm, label))
                summary.append((arm + " (queries)", label, len(ctx["sub"])))

    # ---- Group 4: other students ----
    print("\n" + "=" * 78 + "\nGROUP 4  other students (user message = the sealed Qwen3-14B A / TR user message)\n" + "=" * 78, flush=True)
    d11_runs = [("eval458", ctxs["eval458"], C.d11_prompts("eval458", "FULL")), ("TEST-5", ctx_test, C.d11_prompts("dev200", "TEST"))]
    for skey, spec in C.OTHER_STUDENTS.items():
        tok_s = C.load_tok(spec["path"])
        for label, ctx, d11p in d11_runs:
            out, _info = C.build_other_student_prompts(skey, d11p, ctx, tok_s)
            for lay in ("A", "TR"):
                summary.append((spec["name"] + "_" + lay, label, len(out[lay])))
        C.log_full_prompt("%s_TR, TEST-5 row 0" % spec["name"], out["TR"][0]["prompt"])

    print("\n" + "=" * 78 + "\nD13 PREFLIGHT SUMMARY (prompts built and guarded)\n" + "=" * 78)
    for arm, label, n in summary:
        print("  %-22s %-8s %4d prompts" % (arm, label, n))
    print("\n[D13 preflight] PASS  (%.0fs)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
