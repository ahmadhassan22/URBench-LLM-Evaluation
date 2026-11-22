#!/usr/bin/env python3
"""
d15_preflight.py — D15 CPU preflight (freeze N10, 11.1-11.5, 11.8, 11.10, 11.12). Writes nothing.

  1 pinned MD5s (sample included); the EN instruction and trailer and the N2 replacement strings
    are in the freeze text; the translation system message is the D14 F3 one
  2 row counts (26,431 rows, 5 domains, 26 subdomains); the sample drawn again equals the
    committed id list (2,600 ids, MD5 pinned); groups
  3 the UR prompt replacements (each exactly once) and the UR prompt of every sampled row: its
    three parts, a block between the item lines and the trailer; the UE prompt and item blocks
  4 the scorer on the sealed CSQA fair output with the unrestricted class [A-E]: 190 / 300 (FATAL);
    hand-made cases for [A-D]
  5 no gold in any prompt: the builders get question and options only; no prompt holds
    "correct_key"
  6 context length per arm and translation (tokenizers only)
With --mode TEST|FULL --with-readings the arms that need sealed readings / facts are built too.

No D15 arm file is opened for its answers. No model is loaded.

  python eval/error_analysis_tests/efbpt/d15_preflight.py [--no-tokenizers]
  python eval/error_analysis_tests/efbpt/d15_preflight.py --mode FULL --with-readings
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d15_common as C  # noqa: E402
import d15_score as SC  # noqa: E402
from d15_common import die  # noqa: E402


def check_freeze_text():
    txt = open(C.rp(C.FREEZE_PATH), encoding="utf-8").read()
    flat = re.sub(r"\s*\n\s*", " ", txt)          # the freeze wraps long string literals over lines
    n = 0
    for name, s in (("EN instruction", C.EN_INSTRUCTION), ("EN trailer", "\n\n" + C.EN_TRAILER),
                    ("EN item block", C.item_block(["{q}", "{A}", "{B}", "{C}", "{D}"]))):
        lit = '"%s"' % s.replace("\n", "\\n")
        if lit not in flat:
            die("%s is not in the freeze text as written: %s" % (name, lit))
        n += 1
    for old, new in C.UR_REPLACEMENTS[:3]:
        if old not in txt or new not in txt:
            die("N2 replacement %r -> %r is not in the freeze text" % (old, new))
        n += 1
    if '"E: {E}" line removed' not in flat:
        die("the E line removal is not in the freeze text")
    n += 1
    if C.R1_SYSTEM != C.C14.R1_SYSTEM:
        die("the translation system message is not the D14 F3 message")
    print("[ok] freeze text: EN instruction, EN trailer, EN item block layout and the 4 N2 replacements are in it (%d checks); "
          "the translation system message is d14_common.R1_SYSTEM" % n)


def check_rows():
    src = C.source_rows()
    dom = Counter(r["domain"] for r in src)
    sub = Counter(r["subdomain"] for r in src)
    if len(src) != C.N_ROWS or len(dom) != C.N_DOMAINS or len(sub) != C.N_SUBDOMAINS:
        die("rows %d, domains %d, subdomains %d; expected %d, %d, %d" % (len(src), len(dom), len(sub), C.N_ROWS, C.N_DOMAINS, C.N_SUBDOMAINS))
    print("[rows] %d rows, %d domains, %d subdomains" % (len(src), len(dom), len(sub)))
    ids, rep = C.select_sample()
    if ids != C.sample_ids():
        die("the sample drawn again differs from the committed id list")
    print("[sample] drawn again = %s (MD5 %s): %d ids; valid %d, invalid %s, duplicates dropped %d"
          % (C.SAMPLE_PATH, C.SAMPLE_MD5, len(ids), rep["valid"], dict(rep["invalid"]), rep["duplicates_dropped"]))
    meta = C.row_meta()
    g = Counter(m["group"] for m in meta.values())
    d = Counter(m["domain"] for m in meta.values())
    s = Counter(m["subdomain"] for m in meta.values())
    print("  groups %s; domains %s; subdomains %d (min %d, max %d rows each)" % (dict(g), dict(d), len(s), min(s.values()), max(s.values())))
    gsub = OrderedDict()
    for sname in sorted(s):
        gsub.setdefault(C.group_of(sname), []).append(sname)
    for k in C.GROUPS:
        print("  %-8s %s" % (k, ", ".join(gsub.get(k, []))))
    rows = C.load_rows()
    print("  TEST-5 rows: %s" % ", ".join(r["rid"] for r in rows[:C.TEST_N]))
    return rows


def check_prompts(rows):
    C._ur_code()
    print("[UR] replacements on the fair script's instr / q_block source, occurrences before replacement: %s"
          % ", ".join("%r: %d" % kv for kv in C._CACHE["ur_counts"].items()))
    print("  trailer %r; translation label %r" % (C.ur_trailer(), C.tr_label_line()))
    nl = 0
    for r in rows:
        it = r["item"]
        if set(it) != {"question", "options"} or set(it["options"]) != set(C.LETTERS):
            die("%s: a builder row copy has other fields" % r["rid"])
        instr, items, tr = C.ur_parts(it)
        u = C.ur_user(it)
        if u != instr + "\n\n" + items + "\n\n" + tr:
            die("%s: the UR prompt is not instruction, item lines, trailer" % r["rid"])
        if C.ur_user(it, "X") != instr + "\n\n" + items + "\n\nX\n\n" + tr:
            die("%s: a block does not land between the item lines and the trailer" % r["rid"])
        if "پانچ" in instr or "E)" in instr or "یا E" in instr or "\nE: " in items:
            die("%s: an E option is left in the UR prompt" % r["rid"])
        nl += int(any("\n" in x for x in [it["question"]] + list(it["options"].values())))
        ub = C.ur_block(it)
        if C.parse_translation(ub) != ub or len(ub.split("\n")) != 5:
            die("%s: the Urdu item block does not have 5 label lines" % r["rid"])
        ue = C.build_users("T_ue", [r])[0]["user"]
        if ue != C.EN_INSTRUCTION + "\n\n" + ub + "\n\n" + C.EN_TRAILER:
            die("%s: UE layout" % r["rid"])
    print("[prompt] %d UR, UE and item blocks built; rows with a line break inside a field (kept in UR, one space in item blocks): %d"
          % (len(rows), nl))
    C.log_full_prompt("UR, %s" % rows[0]["rid"], C.ur_user(rows[0]["item"]))
    C.log_full_prompt("UE, %s" % rows[0]["rid"], C.build_users("T_ue", rows[:1])[0]["user"])


def check_scorer():
    n = SC.self_check()
    c, k, d = SC.score_fair()
    if (c, k, d) != (C.FAIR_SEALED_CORRECT, 300, 0):
        die("the scorer with %s gives %d / %d on the sealed CSQA fair output (pred differs on %d rows); expected %d / 300"
            % (SC.CLASS_FAIR, c, k, d, C.FAIR_SEALED_CORRECT))
    print("[scorer] sealed CSQA fair output with %s: %d / %d, the recomputed prediction equals the stored one on every row; "
          "%d hand-made cases for %s pass" % (SC.CLASS_FAIR, c, k, n, SC.CLASS_D15))


def stats(ns):
    ns = sorted(ns)
    return "min %s median %s max %s" % (ns[0], ns[len(ns) // 2], ns[-1]) if ns else "-"


def check_lengths(rows, mode, with_readings):
    readings, facts, info = OrderedDict(), None, []
    if with_readings:
        rws = C.load_rows(mode)
        for name in C.TRANSLATIONS:
            if C.is_sealed(C.tr_paths(name, mode)):
                readings[name], inf = C.load_reading(name, mode, rws)
                info.append("%s not valid %d" % (name, inf["not_valid"]))
        if "R1" in readings and C.is_sealed(C.facts_paths(mode)):
            facts, finf = C.load_facts(mode, rws, readings["R1"])
            info.append("facts: rows without facts %d" % finf["rows_without_facts"])
        rows = rws
        print("[readings] %s: %s" % (mode, "; ".join(info) or "none sealed"))
    s_tok = C.load_tok(C.STUDENT_PATH)
    T = C.teacher_frozen()
    t_tok = C.load_tok(T["path"])
    mx = C.student_max_tokens()
    # translations
    sl = [C.n_tokens(s_tok, C.student_tr_chat(s_tok, C.ur_block(r["item"]))) for r in rows]
    tl = [len(C.teacher_render(t_tok, [{"role": "system", "content": C.R1_SYSTEM}, {"role": "user", "content": C.ur_block(r["item"])}])[1]) for r in rows]
    print("[length] SL prompt tokens %s; too_long (+%d > %d): %d" % (stats(sl), C.LOCAL_TR_MAX_TOKENS, C.STUDENT_MAX_MODEL_LEN,
                                                                     sum(1 for x in sl if x + C.LOCAL_TR_MAX_TOKENS > C.STUDENT_MAX_MODEL_LEN)))
    print("[length] TL prompt tokens %s; too_long (+%d > %d): %d" % (stats(tl), C.LOCAL_TR_MAX_TOKENS, T["vllm_args"]["max_model_len"],
                                                                     sum(1 for x in tl if x + C.LOCAL_TR_MAX_TOKENS > T["vllm_args"]["max_model_len"])))
    for arm, spec in C.ARMS.items():
        if spec["reading"] and spec["reading"] not in readings:
            print("[length] %-6s needs the %s readings (not sealed in %s): skipped" % (arm, spec["reading"], mode))
            continue
        if spec["facts"] and facts is None:
            print("[length] %-6s needs the teacher facts (not sealed in %s): skipped" % (arm, mode))
            continue
        us = C.build_users(arm, rows, readings, facts)
        for u in us:
            if u["user"] is not None and "correct_key" in u["user"]:
                die("%s %s: the prompt holds 'correct_key'" % (arm, u["rid"]))
        ok = [u for u in us if u["status"] == "ok"]
        if spec["model"] == "S":
            n = [C.n_tokens(s_tok, C.student_chat(s_tok, u["user"])) for u in ok]
            lim, mt = C.STUDENT_MAX_MODEL_LEN, mx
        elif spec["model"] == "T":
            n = [len(C.teacher_render(t_tok, [{"role": "user", "content": u["user"]}])[1]) for u in ok]
            lim, mt = T["vllm_args"]["max_model_len"], C.TEACHER_ARM_MAX_TOKENS
        else:
            n = [C.n_tokens(s_tok, u["user"]) for u in ok]
            lim, mt = None, C.API_ARM_MAX_TOKENS
        tl_n = sum(1 for x in n if lim and x + mt > lim)
        print("[length] %-6s rows %d  generated %d  reading_failed %d  prompt tokens %s  too_long %s"
              % (arm, len(us), len(ok), len(us) - len(ok), stats(n), tl_n if lim else "n/a (API; Qwen3 tokenizer count shown)"))
    if facts is not None:
        fl = [facts[r["rid"]] for r in rows]
        print("[facts] rows %d  rows without facts %d  facts total %d" % (len(fl), sum(1 for f in fl if not f), sum(len(f) for f in fl)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="FULL", choices=C.MODES)
    ap.add_argument("--with-readings", action="store_true")
    ap.add_argument("--no-tokenizers", action="store_true", help="skip the context-length check (no tokenizer is loaded)")
    args = ap.parse_args()
    print("=" * 78)
    print("D15 PREFLIGHT (CPU; writes nothing; no model; no arm answer is read)")
    print("=" * 78, flush=True)
    C.verify_pins()
    check_freeze_text()
    rows = check_rows()
    check_prompts(rows)
    check_scorer()
    if not args.no_tokenizers:
        check_lengths(rows if args.mode == "FULL" else rows[:C.TEST_N], args.mode, args.with_readings)
    print("\n[preflight] PASS")


if __name__ == "__main__":
    main()
