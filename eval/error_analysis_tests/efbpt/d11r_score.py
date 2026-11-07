#!/usr/bin/env python3
"""
d11r_score.py — D11-R metrics (docs/EFBPT_D11R_READER_FREEZE.md §3) for any set
of reader files, and the pre-declared selection rule (§4) on RDEV300.

Refuses --set eval458 (§1, §5: those readings are sealed unread).

Metrics per reader, readings joined to the gold question_en by qid
(a missing or failed reading scores as an empty string and is counted):
  M1  proper-noun recall (PRIMARY). Gold proper nouns = tokens of the gold
      question that are not the first token, with all Unicode punctuation
      removed, of >= 2 characters, starting with an upper-case letter (all-caps
      kept; stoplist: I), plus digit-only tokens. Found if the lower-cased
      token equals a punctuation-stripped reading token OR is a substring of
      the punctuation-stripped lower-cased reading. Macro over rows with >= 1
      proper noun; rows without are counted and excluded.
  M2  chrF: sacrebleu corpus_chrf with default settings (char order 6, word
      order 0, beta 2); the mean sentence chrF is reported alongside.
  M3  exact match after norm() (d3_oracle_retrieval.norm: "_"->" ", strip,
      lower, collapse whitespace).
  M4  title presence: over rows with evidence ids, required titles
      (evidence id with the trailing "-<n>" removed) found as norm substring
      in the reading / found in the gold question (ratio to the gold ceiling);
      the share of all required titles is reported too.
  Also: latency (mean, median, p90), output tokens, reasoning tokens, cost at
  list price (configs/d11r_api_prices.json; n/a until filled), garbled
  readings (< 3 Latin letters), Urdu-script leakage, missing/failed rows.

Selection (--set rdev300, >= 2 readers): best M1; readers within 1.0 pp of the
best M1 are decided by higher M2; remaining ties: local over API, then cheaper
over dearer (cost unknown -> tie left open and reported). Winner and runner-up
are printed and written to outputs/efbpt/d11r/_scores/scores_<set>.json.

Usage (from ~/URBench, env urbench_eval):
  python eval/error_analysis_tests/efbpt/d11r_score.py --set rdev300            # every reader dir with that file
  python eval/error_analysis_tests/efbpt/d11r_score.py --set dev200 --readers L1_llama70b_awq api_qwen3.8-max
  python eval/error_analysis_tests/efbpt/d11r_score.py --set rdev300 --test     # files under _test/
"""
import argparse
import json
import os
import statistics
import sys
import unicodedata
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from d11r_common import (  # noqa: E402
    ROOT, OUT_ROOT, TEST_ROOT, SCORES_DIR, SCORABLE_SETS, ITEM_SETS, LOCAL_READERS, FREEZE_MD5,
    die, verify_freeze, load_items, read_jsonl, md5_file, latin_letters, has_arabic_script, write_json,
)

os.chdir(ROOT)
PRICES_PATH = "configs/d11r_api_prices.json"
M1_TIE_PP = 1.0
STOPLIST = {"I"}


def norm(t):
    # identical to d3_oracle_retrieval.norm (asserted against the source text below)
    return " ".join(str(t).replace("_", " ").strip().lower().split())


_src = open(os.path.join(_HERE, "d3_oracle_retrieval.py"), encoding="utf-8").read()
if 'def norm(t):\n    return " ".join(str(t).replace("_", " ").strip().lower().split())' not in _src:
    die("d3_oracle_retrieval.norm changed; update d11r_score.norm")


def strip_punct(tok, keep_space=False):
    return "".join(ch for ch in tok if not unicodedata.category(ch).startswith("P") or (keep_space and ch == " "))


def gold_proper_nouns(question):
    out = []
    for i, t in enumerate(question.split()):
        s = strip_punct(t)
        if len(s) < 2:
            continue
        if s.isdigit():
            out.append(s)
            continue
        if i == 0 or s in STOPLIST:
            continue
        if s[0].isalpha() and s[0].isupper():
            out.append(s)
    return out


def m1_row(gold_q, reading):
    pns = gold_proper_nouns(gold_q)
    if not pns:
        return None, 0, 0
    rd_tokens = {strip_punct(t).lower() for t in reading.split()}
    rd_str = "".join(ch for ch in reading if not unicodedata.category(ch).startswith("P")).lower()
    found = sum(1 for p in pns if p.lower() in rd_tokens or p.lower() in rd_str)
    return found / len(pns), found, len(pns)


def required_titles(ev):
    return sorted({pid.rsplit("-", 1)[0] for pid in ev or [] if isinstance(pid, str) and "-" in pid})


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def load_prices():
    if not os.path.exists(PRICES_PATH):
        return {}
    return json.load(open(PRICES_PATH, encoding="utf-8"))


def cost_of(reader, recs, prices):
    p = prices.get(reader.replace("api_", "", 1)) if reader.startswith("api_") else None
    if not p or p.get("input_per_1M") is None or p.get("output_per_1M") is None:
        return None
    pin = sum(r.get("prompt_tokens") or 0 for r in recs)
    pout = sum((r.get("output_tokens") or 0) for r in recs)
    # completion_tokens of thinking models usually include reasoning tokens; if the
    # gateway reports them separately and excludes them, they are added here
    if p.get("reasoning_excluded_from_completion"):
        pout += sum((r.get("reasoning_tokens") or 0) for r in recs)
    return OrderedDict([("currency", p.get("currency")), ("input_tokens", pin), ("output_tokens", pout),
                        ("cost", round(pin / 1e6 * p["input_per_1M"] + pout / 1e6 * p["output_per_1M"], 4)),
                        ("per_1k_rows", round((pin / 1e6 * p["input_per_1M"] + pout / 1e6 * p["output_per_1M"]) / max(len(recs), 1) * 1000, 4))])


def score_reader(reader, path, items, prices):
    from sacrebleu.metrics import CHRF
    recs = {r["qid"]: r for r in read_jsonl(path)}
    hyps, refs, rows = [], [], []
    m1_vals = []
    m1_found = m1_total = 0
    no_pn = 0
    m3 = 0
    m4_in_read = m4_in_gold = m4_total = m4_rows = 0
    garbled = leak = missing = failed = 0
    lat, otok, rtok, ptok = [], [], [], []
    for it in items:
        r = recs.get(it["qid"])
        if r is None:
            missing += 1
            reading = ""
        elif not r.get("ok"):
            failed += 1
            reading = ""
        else:
            reading = r.get("reading") or ""
            if r.get("latency_s") is not None:
                lat.append(r["latency_s"])
            if r.get("output_tokens") is not None:
                otok.append(r["output_tokens"])
            if r.get("prompt_tokens") is not None:
                ptok.append(r["prompt_tokens"])
            if r.get("reasoning_tokens") is not None:
                rtok.append(r["reasoning_tokens"])
        gold = it["question_en"]
        hyps.append(reading)
        refs.append(gold)
        v, f, n = m1_row(gold, reading)
        if v is None:
            no_pn += 1
        else:
            m1_vals.append(v)
            m1_found += f
            m1_total += n
        if norm(reading) == norm(gold):
            m3 += 1
        titles = required_titles(it.get("evidence_paragraph_ids"))
        if titles:
            m4_rows += 1
            for t in titles:
                m4_total += 1
                ig = norm(t) in norm(gold)
                ir = norm(t) in norm(reading)
                m4_in_gold += ig
                m4_in_read += ir
        if latin_letters(reading) < 3:
            garbled += 1
        if has_arabic_script(reading):
            leak += 1
        rows.append(OrderedDict([("qid", it["qid"]), ("m1", None if v is None else round(v, 4)), ("exact", norm(reading) == norm(gold)),
                                 ("reading", reading)]))
    chrf = CHRF()
    corpus = chrf.corpus_score(hyps, [refs])
    sent = [chrf.sentence_score(h, [g]).score for h, g in zip(hyps, refs)]
    ok_recs = [r for r in recs.values() if r.get("ok")]
    out = OrderedDict([
        ("reader", reader), ("file", os.path.relpath(path, ROOT)), ("file_md5", md5_file(path)),
        ("n_items", len(items)), ("n_records", len(recs)), ("missing", missing), ("failed", failed),
        ("M1_proper_noun_recall_macro_pct", round(100.0 * statistics.mean(m1_vals), 2) if m1_vals else None),
        ("M1_micro_pct", pct(m1_found, m1_total)), ("M1_rows_scored", len(m1_vals)), ("M1_rows_without_proper_nouns", no_pn),
        ("M1_gold_proper_nouns_total", m1_total),
        ("M2_chrF_corpus", round(corpus.score, 2)), ("M2_chrF_sentence_mean", round(statistics.mean(sent), 2) if sent else None),
        ("M2_signature", str(chrf.get_signature())),
        ("M3_exact_match_pct", pct(m3, len(items))), ("M3_exact_match_n", m3),
        ("M4_title_presence_ratio_to_gold_pct", pct(m4_in_read, m4_in_gold)),
        ("M4_titles_in_reading", m4_in_read), ("M4_titles_in_gold_question", m4_in_gold), ("M4_titles_total", m4_total),
        ("M4_rows_with_evidence", m4_rows), ("M4_titles_in_reading_share_of_all_pct", pct(m4_in_read, m4_total)),
        ("latency_s", OrderedDict([("mean", round(statistics.mean(lat), 3) if lat else None),
                                   ("median", round(statistics.median(lat), 3) if lat else None),
                                   ("p90", round(sorted(lat)[int(0.9 * (len(lat) - 1))], 3) if lat else None)])),
        ("output_tokens_mean", round(statistics.mean(otok), 1) if otok else None),
        ("prompt_tokens_mean", round(statistics.mean(ptok), 1) if ptok else None),
        ("reasoning_tokens_mean", round(statistics.mean(rtok), 1) if rtok else None),
        ("reasoning_tokens_sum", sum(rtok) if rtok else None),
        ("cost_list_price", cost_of(reader, ok_recs, prices)),
        ("garbled_lt3_latin", garbled), ("urdu_script_leak", leak),
        ("is_local", reader in LOCAL_READERS.values()),
        ("thinking_variant", next((r.get("thinking_variant") for r in ok_recs if r.get("thinking_variant")), None)),
    ])
    return out, rows


def select(scores):
    """§4 pre-declared rule. Returns an OrderedDict with winner, runner_up, trace."""
    cand = [s for s in scores if s["M1_proper_noun_recall_macro_pct"] is not None]
    if len(cand) < 2:
        return OrderedDict([("note", "fewer than 2 scored readers; no selection")])
    cand.sort(key=lambda s: -s["M1_proper_noun_recall_macro_pct"])
    best_m1 = cand[0]["M1_proper_noun_recall_macro_pct"]
    tied = [s for s in cand if best_m1 - s["M1_proper_noun_recall_macro_pct"] <= M1_TIE_PP]
    trace = ["best M1 %.2f (%s); within %.1f pp: %s" % (best_m1, cand[0]["reader"], M1_TIE_PP, ", ".join(s["reader"] for s in tied))]

    def cost_val(s):
        c = s.get("cost_list_price")
        return 0.0 if s["is_local"] else (c["cost"] if c else None)

    def rank_key(s):
        # higher M2 first; then local over API; then cheaper (unknown cost sorts last)
        cv = cost_val(s)
        return (-s["M2_chrF_corpus"], 0 if s["is_local"] else 1, float("inf") if cv is None else cv)

    tied.sort(key=rank_key)
    if len(tied) > 1:
        trace.append("M2 among tied: " + ", ".join("%s %.2f" % (s["reader"], s["M2_chrF_corpus"]) for s in tied))
        if tied[0]["M2_chrF_corpus"] == tied[1]["M2_chrF_corpus"]:
            trace.append("M2 tie -> local over API, cheaper over dearer; cost known: %s"
                         % ", ".join("%s=%s" % (s["reader"], cost_val(s)) for s in tied[:2]))
            if cost_val(tied[0]) is None and cost_val(tied[1]) is None:
                trace.append("UNRESOLVED: both costs unknown (fill configs/d11r_api_prices.json)")
    winner = tied[0]
    rest = [s for s in cand if s["reader"] != winner["reader"]]
    # runner-up: the same procedure on the remaining readers
    best2 = rest[0]["M1_proper_noun_recall_macro_pct"]
    tied2 = [s for s in rest if best2 - s["M1_proper_noun_recall_macro_pct"] <= M1_TIE_PP]
    tied2.sort(key=rank_key)
    runner = tied2[0]
    trace.append("runner-up: %s (M1 %.2f, M2 %.2f)" % (runner["reader"], runner["M1_proper_noun_recall_macro_pct"], runner["M2_chrF_corpus"]))
    return OrderedDict([("winner", winner["reader"]), ("runner_up", runner["reader"]), ("rule", "best M1; within 1.0 pp -> higher M2; ties -> local over API, cheaper over dearer"),
                        ("trace", trace)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=list(ITEM_SETS))
    ap.add_argument("--readers", nargs="*", default=None, help="reader directory names; default: every reader dir with readings_<set>.jsonl")
    ap.add_argument("--test", action="store_true", help="score files under outputs/efbpt/d11r/_test/")
    ap.add_argument("--rows", action="store_true", help="also write per-row m1/exact/reading to the scores JSON")
    args = ap.parse_args()
    if args.set not in SCORABLE_SETS:
        die("refusing to score %s: its readings are sealed unread (freeze §1, §5)" % args.set)
    verify_freeze()
    root = TEST_ROOT if args.test else OUT_ROOT
    if args.readers:
        readers = args.readers
    else:
        readers = sorted(d for d in os.listdir(root) if not d.startswith("_") and
                         os.path.exists(os.path.join(root, d, "readings_%s.jsonl" % args.set))) if os.path.isdir(root) else []
    if not readers:
        die("no reader files for set %s under %s" % (args.set, root))
    items = load_items(args.set, with_gold=True)
    prices = load_prices()
    scores, per_rows = [], OrderedDict()
    for rd in readers:
        path = os.path.join(root, rd, "readings_%s.jsonl" % args.set)
        if not os.path.exists(path):
            die("missing %s" % path)
        s, rows = score_reader(rd, path, items, prices)
        scores.append(s)
        per_rows[rd] = rows
    # gold ceilings for context
    gold_pn_rows = sum(1 for it in items if gold_proper_nouns(it["question_en"]))
    print("\nD11-R scores  set %s  n %d  (rows with >=1 gold proper noun: %d)%s" % (args.set, len(items), gold_pn_rows, "  TEST" if args.test else ""))
    print("%-24s %7s %7s %7s %7s %7s %6s %6s %6s %8s %8s" % ("reader", "M1", "M1mic", "M2chrF", "M3", "M4", "garbl", "leak", "miss", "lat_s", "outtok"))
    for s in scores:
        print("%-24s %7s %7s %7s %7s %7s %6d %6d %6d %8s %8s" % (
            s["reader"], s["M1_proper_noun_recall_macro_pct"], s["M1_micro_pct"], s["M2_chrF_corpus"], s["M3_exact_match_pct"],
            s["M4_title_presence_ratio_to_gold_pct"], s["garbled_lt3_latin"], s["urdu_script_leak"], s["missing"] + s["failed"],
            s["latency_s"]["mean"], s["output_tokens_mean"]))
    sel = select(scores) if args.set == "rdev300" else OrderedDict([("note", "selection is defined on RDEV300 only (freeze §1, §4)")])
    if "winner" in sel:
        print("\nSELECTION (§4): winner %s, runner-up %s" % (sel["winner"], sel["runner_up"]))
        for t in sel["trace"]:
            print("  " + t)
    else:
        print("\n" + sel["note"])
    out = OrderedDict([("set", args.set), ("test", args.test), ("freeze_md5", FREEZE_MD5), ("n_items", len(items)),
                       ("scores", scores), ("selection", sel)])
    if args.rows:
        out["rows"] = per_rows
    os.makedirs(SCORES_DIR, exist_ok=True)
    path = os.path.join(SCORES_DIR, "scores_%s%s.json" % (args.set, "_TEST" if args.test else ""))
    write_json(path, out)
    print("scores -> %s" % path)


if __name__ == "__main__":
    main()
