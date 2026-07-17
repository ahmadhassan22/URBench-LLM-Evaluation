"""
build_c_probe.py  — assemble the (c) test set. NO MODEL CALL. CPU, <1 min.

WHAT (c) IS
-----------
(c) = does Qwen3-14B corrupt named entities INSIDE its own Urdu multi-hop
reasoning trace (no retrieval involved). This script only BUILDS the test set
we will feed to the model later. It does not call any model.

WHAT THIS SCRIPT DOES (plain words)
-----------------------------------
1. Reads two files you already have:
     - audit30_candidates.jsonl  -> has the Urdu question text (question_ur)
     - audit30_answers.jsonl     -> has your gold entities (canonical_title + urdu_span)
2. For every one of the 30 rows it collects: the Urdu question, and each gold
   entity with its Urdu span.
3. It marks each entity "testable" or not.
     testable  = a proper noun whose Urdu transliteration could drift to a
                 different identity (people, specific places, orgs, works,
                 branded products). These are the only entities where
                 "corruption" is even possible.
     NOT testable = a generic common noun (Hand, Soup, Brain...). A model
                 cannot "corrupt the identity" of a generic word, so scoring
                 it would only pad the denominator and hide the real rate.
4. Writes ONE file that shows ALL 30 rows (nothing hidden), with each entity
   flagged. Later, the corruption rate is computed over testable entities only.

DEFENSIVE
---------
The schema is taken from the 30 lines we built. If a field name differs on
disk, the script prints a clear warning instead of crashing, so you can tell me
the real field name.

HOW TO RUN
----------
    cd /mnt/home/user41/URBench
    python eval/error_analysis_tests/build_c_probe.py

OUTPUT
------
    data/strategyqa_official/efbpt/c_probe_testset.jsonl   <- machine-readable
    data/strategyqa_official/efbpt/c_probe_readable.txt    <- you eyeball this
"""

import json
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
EFBPT = BASE / "data/strategyqa_official/efbpt"

CANDIDATES = EFBPT / "audit30_candidates.jsonl"
ANSWERS    = EFBPT / "audit30_answers.jsonl"
OUT_JSONL  = EFBPT / "c_probe_testset.jsonl"
OUT_TXT    = EFBPT / "c_probe_readable.txt"

# ---------------------------------------------------------------------------
# Fixed testability list, built from the actual 30-row entities.
# NOT testable = generic common nouns (no identity to corrupt).
# Everything else defaults to testable (proper nouns / specific entities).
# This is a pre-declared rule so the denominator is not cherry-picked.
# ---------------------------------------------------------------------------
NOT_TESTABLE = {
    "Hand",
    "Soup",
    "Liquid diet",
    "Brain",
    "Surveillance",
    "Teddy bear",
    "Prehensility",
    "Institutionalisation",
}


def load_jsonl(path):
    """Open a .jsonl file and return a list of dicts (one per non-empty line)."""
    rows = []
    if not path.exists():
        print(f"!! FILE NOT FOUND: {path}")
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_testable(canonical_title):
    """A proper-noun entity is testable; a generic common noun is not."""
    return canonical_title not in NOT_TESTABLE


def main():
    cands = load_jsonl(CANDIDATES)
    answs = load_jsonl(ANSWERS)

    if not cands or not answs:
        print("!! One of the input files was empty or missing. Stopping.")
        return

    # index the Urdu question text by qid, from the candidates file
    q_by_qid = {}
    for r in cands:
        qid = r.get("urbench_qid")
        q_by_qid[qid] = {
            "question_ur": r.get("question_ur"),
            "question_en": r.get("question_en"),
            "answer": r.get("answer"),
        }

    out_rows = []
    txt = open(OUT_TXT, "w", encoding="utf-8")
    txt.write("(c) PROBE TEST SET  —  entity faithfulness in Urdu reasoning\n")
    txt.write("=" * 90 + "\n")
    txt.write("Shows ALL 30 rows. [T] = testable entity, [-] = generic (excluded from rate).\n")
    txt.write("Later: feed the URDU question to Qwen3-14B (thinking ON), read the trace,\n")
    txt.write("check whether each [T] entity is named correctly. Rate = corrupt / total [T].\n")
    txt.write("=" * 90 + "\n\n")

    total_entities = 0
    total_testable = 0
    rows_with_testable = 0

    for i, a in enumerate(answs, 1):
        qid = a.get("urbench_qid")
        meta = q_by_qid.get(qid, {})
        q_ur = meta.get("question_ur", "[[question_ur not found for this qid]]")
        q_en = meta.get("question_en", "")
        ents = a.get("question_entities", [])
        verdict = a.get("verdict", "")

        row_ents = []
        row_has_testable = False
        for e in ents:
            title = e.get("canonical_title", "")
            span  = e.get("urdu_span", "")
            if not title:
                continue
            t = is_testable(title)
            total_entities += 1
            if t:
                total_testable += 1
                row_has_testable = True
            row_ents.append({
                "canonical_title": title,
                "urdu_span": span,
                "testable": t,
            })
        if row_has_testable:
            rows_with_testable += 1

        out_rows.append({
            "urbench_qid": qid,
            "question_ur": q_ur,
            "question_en": q_en,
            "verdict": verdict,
            "entities": row_ents,
        })

        # readable block
        txt.write(f"ROW {i:>2}/30   qid={qid}   verdict={verdict}\n")
        txt.write(f"  UR: {q_ur}\n")
        txt.write(f"  EN: {q_en}\n")
        for re_ in row_ents:
            mark = "T" if re_["testable"] else "-"
            txt.write(f"    [{mark}] {re_['canonical_title']:<45}  span: {re_['urdu_span']}\n")
        txt.write("\n")

    # summary
    txt.write("=" * 90 + "\n")
    txt.write("SUMMARY\n")
    txt.write(f"  total entities across 30 rows : {total_entities}\n")
    txt.write(f"  testable entities [T]         : {total_testable}\n")
    txt.write(f"  generic entities [-]          : {total_entities - total_testable}\n")
    txt.write(f"  rows containing >=1 testable  : {rows_with_testable} / 30\n")
    txt.write("  -> the (c) corruption rate will be measured over the testable entities only.\n")
    txt.close()

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("rows processed        :", len(out_rows))
    print("total entities        :", total_entities)
    print("testable entities [T] :", total_testable)
    print("generic entities  [-] :", total_entities - total_testable)
    print("rows with >=1 testable:", rows_with_testable, "/ 30")
    print()
    print("READ THIS :", OUT_TXT)
    print("MACHINE   :", OUT_JSONL)
    print()
    print("Next: eyeball the .txt. Confirm the [T]/[-] split looks right,")
    print("then we design the model call (still one step at a time).")


if __name__ == "__main__":
    main()