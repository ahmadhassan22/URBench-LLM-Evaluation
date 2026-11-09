#!/usr/bin/env python3
"""
d11_student.py — D11 RTG Job S: student arms with reader-grounded teacher facts.

Contract: docs/EFBPT_D11_RTG_FREEZE.md §1, §4–§8 (MD5 pinned in d11_common) as
amended by docs/EFBPT_D11_RTG_AMENDMENT_1.md (A1.1 leak guards, A1.2 V1 arm
set, A1.4 derangement, A1.5 eval458). Student exactly D10 §2 (Qwen3-14B nf4,
greedy, thinking off, max_new 1024, batch size 1, frozen instruction, build_user
layouts, d1_score_dual scorers, exact two-sided McNemar). Copy of the D11-O /
D10 student arm logic; everything frozen is IMPORTED:
  model path, SYSTEM_MESSAGE, instruction (+MD5), extractor  <- efbpt_eval_dev200
  user layouts arm "A"/"B"/"G", facts header                 <- d1_eval_arms
  secondary extractor                                        <- d1_score_dual
  score()                                                    <- d3_oracle_retrieval
  generate_all (batch size 1)                                <- d4_extract_facts
  paired exact McNemar                                       <- d5_balanced_extract
  leak guard, Latin fraction, md5 helpers                    <- d8_reachable_ceiling
  V8 facts_in_prompt with the <|im_end|> cut (D10 A1.6)      <- d11o_student
  derangement(qids, seed)                                    <- d10_teacher

Arms (freeze §4, DEV200): A_r2 (sealed D10 floor, reused after V2': 5 rows
byte-identical, FATAL otherwise), TR (R1 facts, B), TR2 (R2, B), TR_L1 (R3, B),
TR_wrong (R1 facts of derangement(qid), seed 1010 = the sealed D10 map, B),
TR_noq (R1 facts, G). eval458 (§8, A1.5): A_r2_eval (A, generated), the chosen
TR arm (--eval-arm = dev200_best of the sealed FULL DEV200 summary), TR_wrong_eval.

  --set dev200|eval458  --mode TEST|FULL  [--preflight-only]  [--eval-arm TR|TR2|TR_L1]
Outputs: outputs/efbpt/d11/student/<set>/ d11_arm<ARM>.jsonl, d11_summary.json,
S_START.json, S_SEAL.json (rewritten after every arm), derangement_eval458.json.
TEST (5 rows) under outputs/efbpt/d11/_test/student/<set>/, never a result.
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
    READERS, SETS, TEST_N, out_dir, verify_contracts, require_eval458_gate, load_set_rows, needles_for,
    load_readings, guard_out_path, file_entry, die, md5_file, git_state, read_jsonl,
    D10_A_R2, D10_DERANGEMENT, DERANGE_SEED, OUT_ROOT,
)
from efbpt_eval_dev200 import (  # noqa: E402
    MODEL_PATH, SYSTEM_MESSAGE, MAX_NEW_TOKENS, INSTRUCTION_PATH, INSTRUCTION_MD5, extract_answer, load_instruction,
)
from d1_eval_arms import build_user  # noqa: E402
from d1_score_dual import extract_answer_secondary  # noqa: E402
from d3_oracle_retrieval import score  # noqa: E402
from d4_extract_facts import generate_all  # noqa: E402
from d5_balanced_extract import mcnemar  # noqa: E402
from d8_reachable_ceiling import leak_guard, latin_fraction, md5_text, pkg_versions  # noqa: E402
from d11o_student import facts_in_prompt  # noqa: E402  V8 checker with the <|im_end|> cut (D10 A1.6)
from d10_teacher import derangement  # noqa: E402  T4, seed 1010
import d11o_student as S11O  # noqa: E402  (provenance: D10 freeze MD5 pinned there)

D10_FREEZE_PATH, D10_FREEZE_MD5 = S11O.FREEZE_PATH, S11O.FREEZE_MD5
BATCH_SIZE = 1
CONTEXT_WINDOW = 40960
V2_PROBE_N = 5                           # freeze §4 V2': 5 rows
V1_SPREAD_LIMIT = 5.0                    # §5 V1
V5_TR_WRONG_EXPECTED_PP = -3.0           # §5 V5 / §6: TR_wrong <= A_r2 - 3 expected
V4_PARSE_MIN_PCT, V5_REMOVED_MAX = 95.0, 100
GO_PP, COND_PP, ALPHA = 8.0, 5.0, 0.05   # §7
EVAL_GATE_PP = 5.0                       # §8
PREDICTIONS = OrderedDict([              # §6, pp vs A_r2 unless stated
    ("TR_vs_A_r2", (9.0, 13.0)), ("TR2_minus_TR", (-2.0, 2.0)), ("TR_L1_vs_A_r2", (7.0, 11.0)),
    ("TR_L1_ge_TR_minus_2", "mechanism = explicit reading stage, not the API model"),
    ("TR_noq_le_TR", True), ("TR_wrong_le_A_r2_minus_3", True)])
REFS = OrderedDict([("A_r2_D10", 60.0), ("TF_v1_D10", 65.0), ("GF_D11O", 76.0), ("TO1_D11O", 74.0)])

DEV_ARMS = ("A_r2", "TR", "TR2", "TR_L1", "TR_wrong", "TR_noq")
READER_OF = OrderedDict([("TR", "R1"), ("TR2", "R2"), ("TR_L1", "R3"), ("TR_wrong", "R1"), ("TR_noq", "R1")])
LAYOUT = {"A_r2": "A", "A_r2_eval": "A", "TR": "B", "TR2": "B", "TR_L1": "B", "TR_wrong": "B", "TR_noq": "G", "TR_wrong_eval": "B"}
V1_ARMS_DEV = ("A_r2", "TR", "TR2", "TR_L1", "TR_wrong")          # A1.2: TR_noq excluded, reported
TR_ARMS = ("TR", "TR2", "TR_L1")

OUT_DIR = None
WRITTEN = []


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
    if path not in WRITTEN:
        WRITTEN.append(path)


def write_json(path, obj):
    ap = guard_out_path(OUT_DIR, path)
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    json.load(open(ap, encoding="utf-8"))
    print("  [verified] wrote %s" % path, flush=True)
    if path not in WRITTEN:
        WRITTEN.append(path)


def log_full_prompt(label, prompt):
    print("\n----- BEGIN FULL PROMPT [%s] (%d chars) -----" % (label, len(prompt)))
    print(prompt)
    print("----- END FULL PROMPT [%s] -----" % label)
    print("  tail repr (thinking tag): %r" % prompt[-80:], flush=True)


def verify_teacher_seal(t_dir, reader_key, set_name, contracts):
    """V7-style: the D11 T_SEAL of one reader (md5, bytes, lines of every file)."""
    seal_path = os.path.join(t_dir, "T_SEAL.json")
    if not os.path.exists(seal_path):
        die("T_SEAL.json missing: %s (run Job T %s %s first)" % (seal_path, reader_key, set_name))
    seal = json.load(open(seal_path, encoding="utf-8"), object_pairs_hook=OrderedDict)
    for k, want in (("d11_freeze_md5", contracts["d11_freeze"]["md5"]), ("d11_amendment1_md5", contracts["d11_amendment_1"]["md5"]),
                    ("reader_key", reader_key), ("set", set_name), ("version", "v1")):
        if seal.get(k) != want:
            die("T_SEAL %s: %s is %r, expected %r" % (seal_path, k, seal.get(k), want))
    checked = OrderedDict()
    for p, want in seal["files"].items():
        if not os.path.exists(p):
            die("V7: sealed file missing: %s" % p)
        got = file_entry(p)
        if (got["md5"], got["bytes"], got["lines"]) != (want["md5"], want["bytes"], want["lines"]):
            die("V7: %s differs from T_SEAL: sealed %s, on disk %s" % (p, dict(want), dict(got)))
        checked[p] = got["md5"]
    if not {"t_facts.jsonl", "T_START.json"} <= {os.path.basename(p) for p in checked}:
        die("V7: T_SEAL does not cover t_facts.jsonl and T_START.json")
    print("[V7] T_SEAL %s: %d files verified (md5, bytes, lines) -> PASS" % (seal_path, len(checked)))
    return seal, checked


def dev200_best_from_summary(summary):
    """§7: DEV200-best of {TR, TR2, TR_L1} by accuracy; tie -> TR (then TR2)."""
    accs = [(summary["accuracy_table"][a]["accuracy"], -i, a) for i, a in enumerate(TR_ARMS)]
    return max(accs)[2]


def preflight(set_name, limit, is_test, eval_arm):
    print("=" * 78)
    print("D11 RTG JOB S PREFLIGHT  set %s  (freeze §1, §4, §5; A1.1, A1.2, A1.4, A1.5; D10 freeze §2, §4)" % set_name)
    print("=" * 78, flush=True)
    contracts = verify_contracts()
    got = md5_file(D10_FREEZE_PATH)
    if got != D10_FREEZE_MD5:
        die("D10 freeze MD5 %s != pinned %s" % (got, D10_FREEZE_MD5))
    contracts["d10_freeze"] = OrderedDict([("path", D10_FREEZE_PATH), ("md5", got)])
    print("[md5] %-18s %s  %s" % ("d10_freeze", got, D10_FREEZE_PATH))
    scripts = OrderedDict([(os.path.relpath(__file__), md5_file(os.path.abspath(__file__))),
                           (os.path.relpath(os.path.join(_HERE, "d11_common.py")), md5_file(os.path.join(_HERE, "d11_common.py"))),
                           (os.path.relpath(S11O.__file__), md5_file(S11O.__file__))])
    instruction = load_instruction()
    print("[ok] Urdu instruction MD5 %s verified" % INSTRUCTION_MD5)

    rows = load_set_rows(set_name)
    sub = rows[:limit] if limit else rows
    sub_qids = [r["qid"] for r in sub]
    sub_set = set(sub_qids)
    n_yes = sum(1 for r in sub if r["gold"] == "yes")
    floor = 100.0 * max(n_yes, len(sub) - n_yes) / len(sub)

    # ---- which arms, which readers ----
    if set_name == "dev200":
        arms = list(DEV_ARMS)
        reader_of = OrderedDict(READER_OF)
        chosen = None
    else:
        # A1.5: --eval-arm must equal dev200_best of the sealed FULL DEV200 summary
        if eval_arm not in TR_ARMS:
            die("--set eval458 needs --eval-arm TR|TR2|TR_L1 (freeze §7/§8)")
        dev_dir = out_dir("student", "dev200", False)
        dev_seal = os.path.join(dev_dir, "S_SEAL.json")
        if not os.path.exists(dev_seal):
            die("eval458 needs the sealed FULL DEV200 student run: %s missing" % dev_seal)
        dsum = json.load(open(os.path.join(dev_dir, "d11_summary.json"), encoding="utf-8"), object_pairs_hook=OrderedDict)
        best = dsum.get("decision", {}).get("dev200_best") or dev200_best_from_summary(dsum)
        if best != eval_arm:
            die("--eval-arm %s != dev200_best %s in the sealed DEV200 summary (freeze §7)" % (eval_arm, best))
        if dsum.get("decision", {}).get("verdict") not in ("GO", "CONDITIONAL"):
            die("DEV200 decision is %r: eval458 does not run (freeze §7)" % dsum.get("decision", {}).get("verdict"))
        chosen = eval_arm
        arms = ["A_r2_eval", chosen, "TR_wrong_eval"]
        reader_of = OrderedDict([(chosen, READER_OF[chosen]), ("TR_wrong_eval", READER_OF[chosen])])
        print("[ok] eval458 arm = %s (dev200_best in the sealed DEV200 summary; verdict %s)" % (chosen, dsum["decision"]["verdict"]))
    readers_needed = sorted(set(reader_of.values()))

    # ---- V4: the sealed readings (needed only to assert they never enter a student prompt) ----
    readings_all, rinfo = OrderedDict(), OrderedDict()
    for key in readers_needed:
        rd, info = load_readings(key, set_name, rows)
        readings_all[key], rinfo[key] = rd, info

    # ---- V7: teacher seals and facts ----
    seals, seal_files, facts_by, teacher_by = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for key in readers_needed:
        t_dir = out_dir("teacher", set_name, is_test, key)
        seal, files = verify_teacher_seal(t_dir, key, set_name, contracts)
        seals[key], seal_files[key] = seal, files
        if seal.get("reading_file", {}).get("md5") != rinfo[key]["md5"]:
            die("teacher %s was run on reading file md5 %s, this job sees %s" % (key, seal.get("reading_file", {}).get("md5"), rinfo[key]["md5"]))
        t_facts = read_jsonl(os.path.join(t_dir, "t_facts.jsonl"))
        if [e["qid"] for e in t_facts] != sub_qids:
            die("%s t_facts.jsonl qids (%d) != this run's rows (%d) in order" % (key, len(t_facts), len(sub_qids)))
        for e in t_facts:
            if e.get("teacher_version") != "v1" or e.get("reader") != key:
                die("%s t_facts qid %s: version/reader mismatch" % (key, e["qid"]))
            fl = e["facts"]
            if len(fl) > 6 or any((not isinstance(f, str)) or f != f.strip() or not f for f in fl):
                die("%s t_facts qid %s: facts are not <= 6 clean non-empty strings" % (key, e["qid"]))
            if e["parse_ok"] is False and fl:
                die("%s t_facts qid %s: parse_ok false but facts present" % (key, e["qid"]))
            if e.get("teacher_input_reading") != readings_all[key][e["qid"]]:
                die("%s t_facts qid %s: teacher input differs from the sealed reading" % (key, e["qid"]))
        facts_by[key] = OrderedDict((e["qid"], list(e["facts"])) for e in t_facts)
        teacher_by[key] = OrderedDict((e["qid"], e) for e in t_facts)
        n_ok = sum(1 for e in t_facts if e["parse_ok"])
        v5 = sum(len(e["removed"]) for e in t_facts)
        print("[ok] teacher %s (%s): %d rows; parse_ok %d/%d = %.1f%% -> %s; repaired %d; verdict-filter removed %d -> %s; teacher input == sealed reading on every row"
              % (key, READERS[key]["reader"], len(t_facts), n_ok, len(t_facts), 100.0 * n_ok / len(t_facts),
                 "ok" if 100.0 * n_ok / len(t_facts) >= V4_PARSE_MIN_PCT else "FORMAT FLAG",
                 sum(1 for e in t_facts if e["repaired"]), v5, "ok" if v5 <= V5_REMOVED_MAX else "TEACHER-ANSWERING FLAG"))

    # ---- derangement (freeze §4, A1.4) ----
    der_info = OrderedDict([("seed", DERANGE_SEED)])
    if set_name == "dev200" and not is_test:
        got = md5_file(D10_DERANGEMENT[0])
        if got != D10_DERANGEMENT[1]:
            die("sealed D10 derangement MD5 %s != pinned %s" % (got, D10_DERANGEMENT[1]))
        dmap = OrderedDict(json.load(open(D10_DERANGEMENT[0], encoding="utf-8"), object_pairs_hook=OrderedDict)["map"])
        if list(dmap) != sub_qids or sorted(dmap.values()) != sorted(sub_qids) or any(k == v for k, v in dmap.items()):
            die("sealed D10 derangement is not a fixed-point-free permutation of these qids")
        recomputed, tries = derangement(sub_qids, DERANGE_SEED)
        if recomputed != dmap:
            die("d10_teacher.derangement(qids, %d) does not reproduce the sealed D10 map" % DERANGE_SEED)
        der_info.update([("source", D10_DERANGEMENT[0]), ("md5", got), ("recomputed_equal", True), ("shuffles", tries)])
        print("[ok] TR_wrong derangement = the sealed D10 map (%s), reproduced by d10_teacher.derangement seed %d" % (got, DERANGE_SEED))
    else:
        dmap, tries = derangement(sub_qids, DERANGE_SEED)
        der_info.update([("source", "d10_teacher.derangement over these %d qids (A1.4)" % len(sub_qids)), ("shuffles", tries)])
        print("[ok] derangement of %d qids, seed %d, %d shuffle(s), no fixed point (A1.4: %s)"
              % (len(dmap), DERANGE_SEED, tries, "TEST" if is_test else set_name))

    # ---- sealed A_r2 (dev200) ----
    sealed_a = None
    if set_name == "dev200":
        got = md5_file(D10_A_R2[0])
        if got != D10_A_R2[1]:
            die("sealed A_r2 MD5 %s != pinned %s" % (got, D10_A_R2[1]))
        sealed_a = [e for e in read_jsonl(D10_A_R2[0]) if e["qid"] in sub_set]
        if [e["qid"] for e in sealed_a] != sub_qids or any(e["gold"] != r["gold"] for e, r in zip(sealed_a, sub)):
            die("sealed A_r2 rows differ from this run's rows")
        print("[sealed] A_r2 %s verified (%s); %d rows, %.2f%% on these rows" % (D10_A_R2[0], got, len(sealed_a), score(sealed_a, "pred")["accuracy"]))

    # ---- prompts (tokenizer only) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def chat(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def no_facts(row):
        return OrderedDict((k, v) for k, v in row.items() if k != "facts")

    def arm_prompt(arm, row):
        """D10 freeze §4 layouts, verbatim rule set (D10/D11-O student)."""
        if LAYOUT[arm] == "A":
            return chat(SYSTEM_MESSAGE, build_user(instruction, no_facts(row), None, "A")), "A", [], None
        key = reader_of[arm]
        src_q = dmap[row["qid"]] if arm in ("TR_wrong", "TR_wrong_eval") else row["qid"]
        facts = facts_by[key][src_q]
        if LAYOUT[arm] == "B":
            if facts:
                return chat(SYSTEM_MESSAGE, build_user(instruction, dict(no_facts(row), facts=list(facts)), None, "B")), "B", facts, src_q
            return chat(SYSTEM_MESSAGE, build_user(instruction, no_facts(row), None, "A")), "A", [], src_q
        return chat(SYSTEM_MESSAGE, build_user(instruction, dict(no_facts(row), facts=list(facts)), None, "G")), "G", facts, src_q

    prompts = OrderedDict((arm, [arm_prompt(arm, r) for r in sub]) for arm in arms)
    if sealed_a is not None:
        for (p, _, _, _), e in zip(prompts["A_r2"], sealed_a):
            if p != e["prompt"]:
                die("A_r2 prompt rebuilt here differs from the sealed D10 prompt for qid %s" % e["qid"])
        print("[ok] A_r2 prompts rebuilt here are byte-identical to the %d sealed D10 prompts" % len(sub))
    for arm in arms:
        ps = [p for p, _, _, _ in prompts[arm]]
        tl = sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in ps)
        print("[len] %-13s n=%d tokens min=%d median=%d max=%d  layouts %s  empty-facts %d"
              % (arm, len(ps), tl[0], tl[len(tl) // 2], tl[-1], dict(Counter(l for _, l, _, _ in prompts[arm])),
                 sum(1 for _, _, fl, _ in prompts[arm] if not fl)))
        if tl[-1] + MAX_NEW_TOKENS > CONTEXT_WINDOW:
            die("%s prompts would exceed the context window" % arm)

    # ---- V3 leak guards (A1.1): question_en, evidence ids, readings FATAL; gold facts V6 ----
    needles = needles_for(rows, set_name)
    leak = OrderedDict()
    all_readings = sorted({rd for key in readings_all for rd in readings_all[key].values() if rd.strip()})
    for arm in arms:
        ps = [p for p, _, _, _ in prompts[arm]]
        if LAYOUT[arm] == "A":
            leak_guard(arm, ps, needles, fatal_gold_facts=True)
            leak[arm] = OrderedDict([("gold_facts", "FATAL (no facts arm)"), ("count", 0), ("qids", 0), ("matches", [])])
        else:
            hits = leak_guard(arm, ps, needles, fatal_gold_facts=False)
            by_q = OrderedDict()
            for i, s in hits:
                by_q.setdefault(sub[i]["qid"], []).append(s)
            leak[arm] = OrderedDict([("gold_facts", "REPORTED as V6, not fatal (A1.1; D10 A2.2)"), ("count", len(hits)), ("qids", len(by_q)),
                                     ("matches", [OrderedDict([("qid", q), ("fact", f)]) for q, fs in by_q.items() for f in fs])])
            print("[V6] %s verbatim gold-fact substrings: %d in %d qid(s) (reported)" % (arm, len(hits), len(by_q)))
        for r, p in zip(sub, ps):
            for rd in all_readings:
                if rd in p:
                    die("LEAK: %s qid %s prompt contains a reading %r" % (arm, r["qid"], rd[:80]))
    print("[ok] no sealed reading (%d strings) appears in any of the %d student prompts" % (len(all_readings), sum(len(prompts[a]) for a in arms)))

    # ---- V8 (A1.6 cut) ----
    v8 = 0
    for arm in arms:
        for r, (p, lay, fl, src) in zip(sub, prompts[arm]):
            n_hdr, found = facts_in_prompt(p)
            if lay == "A":
                if n_hdr != 0 or fl:
                    die("V8: %s qid %s: arm-A layout with a facts header or facts" % (arm, r["qid"]))
                continue
            if n_hdr != 1:
                die("V8: %s qid %s: facts header appears %d times" % (arm, r["qid"], n_hdr))
            if found != fl:
                die("V8: %s qid %s (source qid %s): prompt facts %r != source facts %r" % (arm, r["qid"], src, found, fl))
            v8 += 1
    print("[V8] prompt-content check with the <|im_end|> cut: %d facts prompts equal their sealed post-T2 source facts -> PASS" % v8)
    v9 = OrderedDict()
    for arm in arms:
        if LAYOUT[arm] != "A":
            fr, letters, latin = latin_fraction([f for _, _, fl, _ in prompts[arm] for f in fl])
            v9[arm] = OrderedDict([("fraction", round(fr, 4)), ("letters", letters), ("latin_letters", latin), ("n_facts", sum(len(fl) for _, _, fl, _ in prompts[arm]))])
            print("[V9] %s Latin fraction of injected facts %.4f (%d facts)" % (arm, fr, v9[arm]["n_facts"]))
    for arm in arms:
        i = next((i for i, (_, lay, _, _) in enumerate(prompts[arm]) if lay != "A"), 0) if LAYOUT[arm] != "A" else 0
        log_full_prompt("%s row %d, qid %s, layout %s" % (arm, i, sub[i]["qid"], prompts[arm][i][1]), prompts[arm][i][0])

    pre = OrderedDict([
        ("contracts", contracts), ("scripts", scripts), ("git", git_state()),
        ("urdu_instruction", OrderedDict([("path", INSTRUCTION_PATH), ("md5", INSTRUCTION_MD5)])),
        ("set", set_name), ("arms", arms), ("reader_of", reader_of), ("eval_arm", chosen), ("floor", round(floor, 2)),
        ("readers", rinfo), ("teacher_dirs", OrderedDict((k, out_dir("teacher", set_name, is_test, k)) for k in readers_needed)),
        ("teacher_seals", OrderedDict((k, OrderedDict([("script_md5", seals[k].get("script_md5")), ("slurm_job_id", seals[k].get("slurm_job_id")),
                                                       ("t1_parse_ok", seals[k].get("t1_parse_ok")), ("t1_repaired_rows", seals[k].get("t1_repaired_rows")),
                                                       ("t2_removed_total", seals[k].get("t2_removed_total")), ("files", seal_files[k])])) for k in readers_needed)),
        ("derangement", der_info), ("sealed_A_r2", OrderedDict([("path", D10_A_R2[0]), ("md5", D10_A_R2[1])]) if sealed_a is not None else None),
        ("V7_T_SEAL_verified", OrderedDict([("pass", True), ("n_seals", len(seals)), ("n_files", sum(len(f) for f in seal_files.values()))])),
        ("leak_guard", leak), ("V8_prompt_content", OrderedDict([("pass", True), ("facts_prompts_checked", v8)])), ("V9_latin_fraction", v9),
        ("layouts", OrderedDict((arm, OrderedDict([("counts", OrderedDict(sorted(Counter(l for _, l, _, _ in prompts[arm]).items()))),
                                                   ("empty_facts", sum(1 for _, _, fl, _ in prompts[arm] if not fl))])) for arm in arms)),
    ])
    print("[preflight] PASS", flush=True)
    return instruction, sub, arms, prompts, teacher_by, sealed_a, dmap, tok, pre


def pair(x, y, lx, ly):
    def acc(recs):
        return score(recs, "pred")["accuracy"]
    b, c, p = mcnemar(x, y, lx, ly)
    return OrderedDict([("n", len(x)), ("acc_%s" % lx, acc(x)), ("acc_%s" % ly, acc(y)), ("delta_pp", round(acc(x) - acc(y), 2)),
                        ("wins_%s" % lx, b), ("losses_%s" % lx, c), ("discordant", b + c), ("p_mcnemar_exact_two_sided", p)])


def compute_readings(set_name, sub, arms, results, pre, is_test):
    hdr = "=" * 84
    tags = "   (TEST — NOT A RESULT)" if is_test else ""
    summary = OrderedDict([("mode", "TEST — NOT A RESULT" if is_test else "FULL"), ("set", set_name), ("n", len(sub)), ("floor", pre["floor"])])
    floor_arm = "A_r2" if set_name == "dev200" else "A_r2_eval"
    A = results[floor_arm]
    print("\n" + hdr)
    print("D11 RTG VALIDITY (freeze §5; A1.1, A1.2)%s" % tags)
    print(hdr)
    v1_arms = [a for a in (V1_ARMS_DEV if set_name == "dev200" else arms) if a in results]
    unp = OrderedDict((a, score(results[a], "pred")["unparsed_rate"]) for a in results)
    spread = max(unp[a] for a in v1_arms) - min(unp[a] for a in v1_arms)
    v1_ok = spread <= V1_SPREAD_LIMIT
    summary["V1_unparsed_spread"] = OrderedDict([("unparsed_rate_primary", unp), ("arms", v1_arms), ("spread_pp", round(spread, 2)),
                                                 ("limit_pp", V1_SPREAD_LIMIT), ("pass", v1_ok),
                                                 ("verdict", "reported only (TEST)" if is_test else ("PASS" if v1_ok else "VOID (spread > 5.0 pp; A1.2)"))])
    print("V1 unparsed %% %s -> spread %.2f pp over %s -> %s%s" % (dict(unp), spread, v1_arms, summary["V1_unparsed_spread"]["verdict"],
                                                                  "   (TR_noq %.2f%% reported, excluded; A1.2)" % unp["TR_noq"] if "TR_noq" in unp else ""))
    print("V2' %s" % pre.get("V2_prime", {}).get("verdict", "n/a (eval458, A1.5)"))
    print("V3 leak guards: none fatal (teacher A1.1 (a)(b); student question_en / evidence ids / readings) -> PASS; V6 gold-fact restatements: %s"
          % {a: pre["leak_guard"][a]["count"] for a in arms if LAYOUT[a] != "A"})
    print("V4 readings = sealed files: %s" % {k: v["md5"] for k, v in pre["readers"].items()})
    for k, v in pre["readers"].items():
        print("   %s readings containing yes/no/ہاں/نہیں: %d (reported); equal to gold question_en: %d" % (k, v["V4_reading_contains_yes_no"]["n"], v["readings_equal_gold_question_en"]["n"]))
    print("V7 teacher seals verified (%d files) -> PASS; V8 %d facts prompts -> PASS; V9 %s"
          % (pre["V7_T_SEAL_verified"]["n_files"], pre["V8_prompt_content"]["facts_prompts_checked"], {a: v["fraction"] for a, v in pre["V9_latin_fraction"].items()}))

    print("\n" + hdr)
    print("D11 RTG READINGS (paired exact McNemar, two-sided, same qids)%s" % tags)
    print(hdr)
    pairs = OrderedDict()
    for arm in arms:
        if arm != floor_arm:
            pairs["%s_vs_%s" % (arm, floor_arm)] = pair(results[arm], A, arm, floor_arm)
            e = pairs["%s_vs_%s" % (arm, floor_arm)]
            print("%-13s - %s = %+6.2f pp (%.2f vs %.2f), wins %d losses %d, p = %s" % (arm, floor_arm, e["delta_pp"], e["acc_%s" % arm], e["acc_%s" % floor_arm], e["wins_%s" % arm], e["losses_%s" % arm], e["p_mcnemar_exact_two_sided"]))
    if set_name == "dev200":
        for a, b in (("TR2", "TR"), ("TR_L1", "TR"), ("TR_noq", "TR"), ("TR", "TR_wrong")):
            pairs["%s_vs_%s" % (a, b)] = pair(results[a], results[b], a, b)
            e = pairs["%s_vs_%s" % (a, b)]
            print("%-13s - %-8s = %+6.2f pp, wins %d losses %d, p = %s" % (a, b, e["delta_pp"], e["wins_%s" % a], e["losses_%s" % a], e["p_mcnemar_exact_two_sided"]))
    summary["pairs"] = pairs
    wrong_arm = "TR_wrong" if set_name == "dev200" else "TR_wrong_eval"
    d_wrong = pairs["%s_vs_%s" % (wrong_arm, floor_arm)]["delta_pp"]
    v5 = OrderedDict([("TR_wrong_minus_floor_pp", d_wrong), ("expected", "<= %.1f" % V5_TR_WRONG_EXPECTED_PP),
                      ("as_expected", d_wrong <= V5_TR_WRONG_EXPECTED_PP),
                      ("A2_1_sign_reading", "WRONG-FACTS HARM (expected direction)" if d_wrong < -4.0 else ("FORMAT-EFFECT FLAG (> +4.0)" if d_wrong > 4.0 else "within ±4.0: neither flag"))])
    summary["V5_TR_wrong"] = v5
    print("V5 %s - %s = %+.2f pp -> expected <= %.1f: %s; A2.1 sign reading: %s" % (wrong_arm, floor_arm, d_wrong, V5_TR_WRONG_EXPECTED_PP, "as expected" if v5["as_expected"] else "NOT as expected (reported)", v5["A2_1_sign_reading"]))

    if set_name == "dev200":
        tr = pairs["TR_vs_A_r2"]
        acc = lambda a: score(results[a], "pred")["accuracy"]  # noqa: E731
        preds = OrderedDict([
            ("TR_vs_A_r2 in [+9,+13]", OrderedDict([("observed", tr["delta_pp"]), ("hit", 9.0 <= tr["delta_pp"] <= 13.0)])),
            ("TR2 within ±2 of TR", OrderedDict([("observed", round(acc("TR2") - acc("TR"), 2)), ("hit", abs(acc("TR2") - acc("TR")) <= 2.0)])),
            ("TR_L1_vs_A_r2 in [+7,+11]", OrderedDict([("observed", pairs["TR_L1_vs_A_r2"]["delta_pp"]), ("hit", 7.0 <= pairs["TR_L1_vs_A_r2"]["delta_pp"] <= 11.0)])),
            ("TR_L1 >= TR - 2 (mechanism = reading stage)", OrderedDict([("observed", round(acc("TR_L1") - acc("TR"), 2)), ("hit", acc("TR_L1") >= acc("TR") - 2.0)])),
            ("TR_noq <= TR", OrderedDict([("observed", round(acc("TR_noq") - acc("TR"), 2)), ("hit", acc("TR_noq") <= acc("TR"))])),
            ("TR_wrong <= A_r2 - 3", OrderedDict([("observed", d_wrong), ("hit", d_wrong <= -3.0)]))])
        summary["predictions_section_6"] = preds
        print("\n§6 predictions: " + "; ".join("%s: %s -> %s" % (k, v["observed"], "hit" if v["hit"] else "miss") for k, v in preds.items()))
        if is_test:
            verdict, note = "n/a (TEST)", "TEST — NOT A RESULT"
        elif tr["delta_pp"] >= GO_PP and tr["p_mcnemar_exact_two_sided"] < ALPHA:
            verdict, note = "GO", "TR - A_r2 >= +8.0 and p < 0.05 -> eval458 once (freeze §7, §8)"
        elif COND_PP <= tr["delta_pp"] < GO_PP:
            verdict, note = "CONDITIONAL", "+5.0 <= TR - A_r2 < +8.0 -> eval458 once with the DEV200-best of {TR, TR2, TR_L1}"
        elif tr["delta_pp"] >= GO_PP:
            verdict, note = "CONDITIONAL", "TR - A_r2 >= +8.0 but p >= 0.05 -> treated as CONDITIONAL (+5.0 to +7.9 band rule by magnitude; p not met for GO)"
        else:
            verdict, note = "CLOSED", "TR - A_r2 < +5.0 -> RTG closed; fallback D12 = agreement-gated reading"
        best = dev200_best_from_summary(OrderedDict([("accuracy_table", OrderedDict((a, OrderedDict([("accuracy", acc(a))])) for a in TR_ARMS))]))
        summary["decision"] = OrderedDict([("rule", "freeze §7: TR vs A_r2, exact McNemar; GO >= +8.0 & p < 0.05; CONDITIONAL +5.0..+7.9; else closed; eval458 arm = DEV200-best of {TR, TR2, TR_L1}, tie -> TR"),
                                           ("TR_minus_A_r2_pp", tr["delta_pp"]), ("p", tr["p_mcnemar_exact_two_sided"]), ("verdict", verdict), ("note", note),
                                           ("dev200_best", best), ("to_record_in", "docs/D11_DECISION.md (readout from sealed files)")])
        print("\n§7 DECISION: %s — %s; DEV200-best of {TR, TR2, TR_L1} = %s%s" % (verdict, note, best, tags))
    else:
        e = pairs["%s_vs_%s" % (pre["eval_arm"], floor_arm)]
        gate = e["delta_pp"] >= EVAL_GATE_PP and e["p_mcnemar_exact_two_sided"] < ALPHA
        summary["decision"] = OrderedDict([("rule", "freeze §8: one run; gate +5.0 and p < 0.05"), ("arm", pre["eval_arm"]), ("delta_pp", e["delta_pp"]),
                                           ("p", e["p_mcnemar_exact_two_sided"]), ("verdict", "n/a (TEST)" if is_test else ("PASS" if gate else "FAIL"))])
        print("\n§8 eval458 gate: %s - %s = %+.2f pp, p = %s -> %s%s" % (pre["eval_arm"], floor_arm, e["delta_pp"], e["p_mcnemar_exact_two_sided"], summary["decision"]["verdict"], tags))

    table = OrderedDict()
    print("\n" + hdr)
    print("ACCURACY TABLE%s   floor %.2f%%   references: %s" % (tags, pre["floor"], dict(REFS)))
    print(hdr)
    print("%-13s %-10s %5s %8s %9s %9s %8s %8s" % ("arm", "layout", "n", "acc%", "unpars%", "SEC acc%", "yes%", "trunc%"))
    for arm in arms:
        recs = results[arm]
        pri, se = score(recs, "pred"), score(recs, "pred_secondary")
        table[arm] = OrderedDict([("n", pri["n"]), ("accuracy", pri["accuracy"]), ("correct", pri["correct"]), ("unparsed_rate", pri["unparsed_rate"]),
                                  ("truncation_rate", round(100.0 * sum(1 for r in recs if r.get("truncated")) / pri["n"], 2)),
                                  ("predicted_yes_rate", pri["predicted_yes_rate"]), ("accuracy_secondary", se["accuracy"]), ("unparsed_rate_secondary", se["unparsed_rate"]),
                                  ("layouts", pre["layouts"][arm]["counts"]), ("empty_facts_rows", pre["layouts"][arm]["empty_facts"])])
        print("%-13s %-10s %5d %8.2f %9.2f %9.2f %8.2f %8.2f" % (arm, LAYOUT[arm], pri["n"], pri["accuracy"], pri["unparsed_rate"], se["accuracy"], pri["predicted_yes_rate"], table[arm]["truncation_rate"]))
    summary["accuracy_table"] = table
    summary["references_sealed"] = REFS
    return summary


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--mode", required=True, choices=("TEST", "FULL"))
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--eval-arm", choices=TR_ARMS, default=None, help="eval458 only: must equal dev200_best of the sealed DEV200 summary")
    args = ap.parse_args()
    is_test = args.mode == "TEST"
    limit = TEST_N if is_test else None
    require_eval458_gate(args.set)
    OUT_DIR = out_dir("student", args.set, is_test)

    instruction, sub, arms, prompts, teacher_by, sealed_a, dmap, tok, pre = preflight(args.set, limit, is_test, args.eval_arm)
    if args.preflight_only:
        print("\n[preflight-only] done. No files written.")
        return
    print("\n[mode] %s  rows=%d  arms=%s  out=%s" % ("TEST — NOT A RESULT" if is_test else "FULL", len(sub), arms, OUT_DIR))
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    if args.set == "eval458":
        write_json(os.path.join(OUT_DIR, "derangement_eval458.json"),
                   OrderedDict([("seed", DERANGE_SEED), ("n", len(dmap)), ("note", "qid -> donor qid; TR_wrong_eval gets the donor's post-T2 facts (A1.4)"), ("map", dmap)]))

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if im is not None and im >= 0:
        eos_ids.add(im)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    print("\n[load] base model, 4-bit nf4, no adapters ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    # ---- V2' (freeze §4, §5): 5 A_r2 prompts once vs the sealed generations; FATAL on any difference ----
    if sealed_a is not None:
        n_probe = min(V2_PROBE_N, len(sub))
        print("\n" + "=" * 70)
        print("V2' — %d sealed A_r2 prompts regenerated once at batch size %d; must be byte-identical" % (n_probe, BATCH_SIZE))
        print("=" * 70, flush=True)
        g = generate_all(model, tok, [p for p, _, _, _ in prompts["A_r2"][:n_probe]], BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
        same = [ta == e["generation"] for (ta, _, _), e in zip(g, sealed_a[:n_probe])]
        mism = [sealed_a[i]["qid"] for i, s in enumerate(same) if not s]
        pre["V2_prime"] = OrderedDict([("n", n_probe), ("identical_to_sealed", sum(same)), ("pass", not mism), ("mismatched_qids", mism),
                                       ("verdict", "PASS: %d/%d byte-identical to the sealed D10 A_r2 generations; sealed A_r2 reused" % (sum(same), n_probe) if not mism
                                        else "FAIL: %d/%d identical" % (sum(same), n_probe))])
        print("V2' %s" % pre["V2_prime"]["verdict"], flush=True)
        if mism:
            die("V2' failed (freeze §5: fatal); mismatched qids %s" % mism)

    start = OrderedDict([
        ("job", "S"), ("experiment", "D11 RTG"), ("set", args.set), ("mode", args.mode), ("args", vars(args)),
        ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("hostname", socket.gethostname()), ("slurm_job_id", job_id),
        ("n_rows", len(sub)), ("qids", [r["qid"] for r in sub]), ("model_path", MODEL_PATH),
        ("decoding", OrderedDict([("batch_size", BATCH_SIZE), ("do_sample", False), ("enable_thinking", False),
                                  ("quant", "nf4 double-quant bf16 compute"), ("answer_max_new", MAX_NEW_TOKENS)])),
        ("versions", pkg_versions())])
    start.update(pre)
    print("\n[START]")
    write_json(os.path.join(OUT_DIR, "S_START.json"), start)

    def seal(final_summary=None):
        files = OrderedDict((p, file_entry(p)) for p in WRITTEN)
        s = OrderedDict([("job", "S"), ("experiment", "D11 RTG"), ("set", args.set), ("mode", args.mode), ("slurm_job_id", job_id),
                         ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("complete", final_summary is not None),
                         ("script_md5", pre["scripts"][os.path.relpath(__file__)]),
                         ("d11_freeze_md5", pre["contracts"]["d11_freeze"]["md5"]), ("d11_amendment1_md5", pre["contracts"]["d11_amendment_1"]["md5"]),
                         ("d10_freeze_md5", pre["contracts"]["d10_freeze"]["md5"]),
                         ("teacher_seal_files", OrderedDict((k, pre["teacher_seals"][k]["files"]) for k in pre["teacher_seals"])),
                         ("readers", OrderedDict((k, OrderedDict([("file", v["file"]), ("md5", v["md5"])])) for k, v in pre["readers"].items())),
                         ("sealed_A_r2", pre["sealed_A_r2"]), ("derangement", pre["derangement"]), ("files", files)])
        if final_summary is not None:
            s["decision"] = final_summary.get("decision")
        path = os.path.join(OUT_DIR, "S_SEAL.json")
        ap_ = guard_out_path(OUT_DIR, path)
        with open(ap_, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        print("  [sealed] %s (%d files%s)" % (path, len(files), ", complete" if final_summary is not None else ", partial"), flush=True)

    def run_arm(arm):
        ps = [p for p, _, _, _ in prompts[arm]]
        print("\n" + "=" * 70)
        print("ARM %s  (n=%d, layout %s, max_new=%d, batch size %d, greedy)" % (arm, len(sub), LAYOUT[arm], MAX_NEW_TOKENS, BATCH_SIZE))
        print("=" * 70, flush=True)
        gens = generate_all(model, tok, ps, BATCH_SIZE, eos_ids, MAX_NEW_TOKENS)
        if len(gens) != len(ps):
            die("%s: %d generations for %d prompts" % (arm, len(gens), len(ps)))
        recs = []
        for r, (p, lay, fl, src), (text, n_new, trunc) in zip(sub, prompts[arm], gens):
            pred = extract_answer(text)
            key = pre["reader_of"].get(arm)
            recs.append(OrderedDict([
                ("qid", r["qid"]), ("arm", arm), ("layout", lay), ("gold", r["gold"]), ("pred", pred),
                ("pred_secondary", extract_answer_secondary(text)), ("correct", pred == r["gold"]), ("unparsed", pred is None),
                ("truncated", trunc), ("n_gen_tokens", n_new), ("n_facts_given", len(fl)), ("facts_source_qid", src),
                ("facts_source_reader", key), ("source_parse_ok", teacher_by[key][src]["parse_ok"] if key and src else None),
                ("prompt", p), ("generation", text)]))
        write_jsonl(os.path.join(OUT_DIR, "d11_arm%s.jsonl" % arm), recs, expect_n=len(sub))
        s = score(recs, "pred")
        print("  accuracy %.2f%%  unparsed %.2f%%  (%d/%d correct)  pred-yes %.2f%%" % (s["accuracy"], s["unparsed_rate"], s["correct"], s["n"], s["predicted_yes_rate"]))
        print("  sample generation (row 0, first 300 chars): %r" % recs[0]["generation"][:300], flush=True)
        seal()
        return recs

    results = OrderedDict()
    for arm in arms:
        if arm == "A_r2" and sealed_a is not None:
            results[arm] = [OrderedDict([
                ("qid", e["qid"]), ("arm", "A_r2"), ("layout", e["layout"]), ("gold", e["gold"]), ("pred", e["pred"]),
                ("pred_secondary", e["pred_secondary"]), ("correct", e["correct"]), ("unparsed", e["unparsed"]), ("truncated", e["truncated"]),
                ("n_gen_tokens", e["n_gen_tokens"]), ("n_facts_given", 0), ("facts_source_qid", None),
                ("facts_source_reader", "sealed:%s" % D10_A_R2[0]), ("source_parse_ok", None), ("prompt", e["prompt"]), ("generation", e["generation"])]) for e in sealed_a]
            print("\n[A_r2] sealed D10 floor reused (%d rows, MD5 %s); not re-generated" % (len(sub), D10_A_R2[1]))
        else:
            results[arm] = run_arm(arm)

    summary = compute_readings(args.set, sub, arms, results, pre, is_test)
    summary["slurm_job_id"] = job_id
    summary["V2_prime"] = pre.get("V2_prime")
    summary["teacher_seals"] = pre["teacher_seals"]
    summary["readers"] = pre["readers"]
    write_json(os.path.join(OUT_DIR, "d11_summary.json"), summary)
    seal(summary)
    if is_test:
        print("\nTEST COMPLETE. Nothing above is a result.")
    else:
        print("\nD11 RTG JOB S (%s) COMPLETE. Sealed %d files." % (args.set, len(WRITTEN)))
        if not summary["V1_unparsed_spread"]["pass"]:
            print("*** V1 VOID (A1.2): readings are for diagnosis only; exit 3")
            sys.exit(3)


if __name__ == "__main__":
    main()
