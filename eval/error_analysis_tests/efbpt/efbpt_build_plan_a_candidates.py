#!/usr/bin/env python3
"""
efbpt_build_plan_a_candidates.py — CPU only. No model, no GPU.

Builds the input file for the Schema C draft generator.

For each of the 100 qids in plan_a_qids_100.txt it joins together:
  - question_ur   (Urdu question, verbatim, from the mapped file)
  - question_en   (English question, context for the model)
  - term          (the gold StrategyQA term entity)
  - steps_en      (official_decomposition, copied verbatim -- AMENDMENT 2b)
  - answer        (true/false mapped to yes/no -- AMENDMENT 2a)
  - candidate_titles  (HINT titles: the gold term first, then evidence pages)

IMPORTANT about candidate_titles:
  These are HINTS ONLY. They are read from the "title" field of each
  paragraph id inside strategyqa_train_paragraphs.json. They are deliberately
  noisy -- a row can list pages for people/things the question never names.
  The draft generator must decide YES/NO for each one, and an entity is kept
  ONLY if the Urdu question actually names it and a verbatim Urdu span exists.
  Nothing in this file is an accepted entity.

Writes exactly one file:
  data/strategyqa_official/efbpt/plan_a_candidates_100.jsonl

Usage:
  python eval/error_analysis_tests/efbpt/efbpt_build_plan_a_candidates.py --dry-run
  python eval/error_analysis_tests/efbpt/efbpt_build_plan_a_candidates.py
  python eval/error_analysis_tests/efbpt/efbpt_build_plan_a_candidates.py --force
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path("/mnt/home/user41/URBench")
EF = BASE / "data/strategyqa_official/efbpt"

QIDS_FILE = EF / "plan_a_qids_100.txt"
MAPPED_FILE = BASE / "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl"
PARAS_FILE = BASE / "data/strategyqa_official/strategyqa_train_paragraphs.json"
OUT_FILE = EF / "plan_a_candidates_100.jsonl"

EXPECTED_N = 100

# AMENDMENT 2a -- the only two answer values allowed in the source data.
ANSWER_MAP = {True: "yes", False: "no"}


def norm(s):
    """Frozen normalisation, same shape as the rest of the project:
    underscores become spaces, whitespace collapses, case is ignored."""
    return re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_", " ")).strip().casefold()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def die(msg):
    print(f"FATAL: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="check everything and print the summary, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing output file")
    ap.add_argument("--limit", type=int, default=0,
                    help="inspect only the first N qids (0 = all). "
                         "Requires --dry-run; a partial file is never written.")
    args = ap.parse_args()

    # --limit is for INSPECTION ONLY. Writing a short file under the name
    # plan_a_candidates_100.jsonl would silently poison every later stage,
    # so a limited run may never write.
    if args.limit and not args.dry_run:
        die("--limit may only be used together with --dry-run. "
            "A partial file must never be written as plan_a_candidates_100.jsonl.")

    # ---------- guard the output file ----------
    if OUT_FILE.exists() and not args.dry_run and not args.force:
        die(f"output already exists: {OUT_FILE}\nRe-run with --force if you really mean to overwrite it.")

    # ---------- load inputs ----------
    for p in (QIDS_FILE, MAPPED_FILE, PARAS_FILE):
        if not p.exists():
            die(f"missing input file: {p}")

    qids = [ln.strip() for ln in QIDS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(qids) != EXPECTED_N:
        die(f"{QIDS_FILE} has {len(qids)} qids, expected {EXPECTED_N}")
    if len(set(qids)) != len(qids):
        die("duplicate qids inside the manifest file")

    mapped_rows = load_jsonl(MAPPED_FILE)
    mapped = {}
    for r in mapped_rows:
        key = r.get("urbench_qid")
        if key is not None:
            mapped[key] = r

    with open(PARAS_FILE, encoding="utf-8") as f:
        paras = json.load(f)

    if args.limit:
        qids = qids[:args.limit]

    # ---------- build ----------
    out_rows = []
    missing_pids = []          # paragraph ids not found in the paragraphs file
    empty_title_pids = []      # found, but no usable title
    title_counts = []
    step_counts = []
    answer_counts = {"yes": 0, "no": 0}
    term_missing_from_evidence = 0
    no_term_rows = []

    for qid in qids:
        row = mapped.get(qid)
        if row is None:
            die(f"qid not found in mapped file: {qid}")

        question_ur = row.get("question_ur")
        question_en = row.get("question_en")
        term = row.get("term") or ""
        steps_en = row.get("official_decomposition")
        raw_answer = row.get("answer")

        if not question_ur or not str(question_ur).strip():
            die(f"empty question_ur for qid {qid}")
        if not isinstance(steps_en, list) or len(steps_en) < 2:
            die(f"official_decomposition is not a list of >=2 steps for qid {qid}: {steps_en!r}")
        if raw_answer not in ANSWER_MAP:
            die(f"unexpected answer value for qid {qid}: {raw_answer!r} "
                f"(AMENDMENT 2a allows only True/False)")

        answer = ANSWER_MAP[raw_answer]
        answer_counts[answer] += 1

        # ---- hint titles from evidence paragraph ids ----
        # Look up the real "title" field. Never split the id on a hyphen:
        # ids such as "Jean-Paul Sartre-3" would break.
        pids = row.get("evidence_paragraph_ids") or []
        seen = set()
        evidence_titles = []
        for pid in pids:
            entry = paras.get(pid)
            if entry is None:
                missing_pids.append((qid, pid))
                continue
            title = entry.get("title") if isinstance(entry, dict) else None
            if not title or not str(title).strip():
                empty_title_pids.append((qid, pid))
                continue
            key = norm(title)
            if key in seen:
                continue
            seen.add(key)
            evidence_titles.append(str(title).strip())

        # The gold StrategyQA "term" is the topic entity of the question and is
        # usually named in the question itself. Evidence pages support the STEPS,
        # so they often do not contain the question's own entities. Putting term
        # first means it goes through the same forced YES/NO decision as the rest
        # instead of relying on the model to think of it unprompted.
        term_in_evidence = bool(term) and norm(term) in seen
        if term and not term_in_evidence:
            candidate_titles = [str(term).strip()] + evidence_titles
            term_missing_from_evidence += 1
        else:
            candidate_titles = list(evidence_titles)

        title_counts.append(len(candidate_titles))
        step_counts.append(len(steps_en))

        out_rows.append({
            "qid": qid,
            "question_ur": question_ur,
            "question_en": question_en,
            "term": term,
            "answer": answer,
            "steps_en": steps_en,
            "candidate_titles": candidate_titles,
            "term_in_evidence_titles": term_in_evidence,
            "n_evidence_pids": len(pids),
        })
        if not term:
            no_term_rows.append(qid)

    # ---------- write ----------
    if args.dry_run:
        print("DRY RUN -- nothing was written.\n")
    else:
        if len(out_rows) != EXPECTED_N:
            die(f"refusing to write: built {len(out_rows)} rows, expected {EXPECTED_N}")
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- summary ----------
    def stats(xs):
        if not xs:
            return "n/a"
        return f"min {min(xs)} / median {sorted(xs)[len(xs)//2]} / max {max(xs)} / mean {sum(xs)/len(xs):.1f}"

    zero_title_rows = [r["qid"] for r in out_rows if not r["candidate_titles"]]

    print("PLAN A CANDIDATES -- build summary")
    print("=" * 60)
    print(f"rows built               : {len(out_rows)}")
    print(f"answer yes / no          : {answer_counts['yes']} / {answer_counts['no']}")
    print(f"steps per row            : {stats(step_counts)}")
    print(f"total steps              : {sum(step_counts)}")
    print(f"hint titles per row      : {stats(title_counts)}")
    print(f"total hint titles        : {sum(title_counts)}")
    print(f"term NOT in evidence pgs : {term_missing_from_evidence}  (term was added as an extra hint)")
    print(f"rows with EMPTY term     : {len(no_term_rows)}")
    if no_term_rows:
        print(f"  qids                   : {no_term_rows[:10]}")
    print(f"rows with ZERO hints     : {len(zero_title_rows)}")
    if zero_title_rows:
        print(f"  qids                   : {zero_title_rows[:10]}")
    print(f"paragraph ids not found  : {len(missing_pids)}")
    if missing_pids:
        print(f"  examples               : {missing_pids[:5]}")
    print(f"paragraph ids w/o title  : {len(empty_title_pids)}")
    if empty_title_pids:
        print(f"  examples               : {empty_title_pids[:5]}")
    print("-" * 60)
    if out_rows:
        ex = out_rows[0]
        print("FIRST ROW PREVIEW")
        print(f"  qid            : {ex['qid']}")
        print(f"  question_ur    : {ex['question_ur']}")
        print(f"  question_en    : {ex['question_en']}")
        print(f"  term           : {ex['term']}")
        print(f"  answer         : {ex['answer']}")
        print(f"  steps_en       : {ex['steps_en']}")
        print(f"  candidate_titles ({len(ex['candidate_titles'])}): {ex['candidate_titles']}")
    print("-" * 60)
    print(f"output file : {OUT_FILE}" + ("  [NOT WRITTEN -- dry run]" if args.dry_run else "  [written]"))


if __name__ == "__main__":
    main()
