"""
EFBPT AUDIT30 — readable audit viewer (CPU only, ~1 minute).

WHY THIS EXISTS
---------------
The audit file (audit30_candidates.jsonl) refers to evidence as IDs like "Soup-1".
You cannot judge a paragraph you cannot read. This script prints, for each of the
30 rows, a readable page containing:
    - the Urdu question and the English question
    - the entity you must label (and its candidate hard negatives)
    - each plan step, with the FULL TEXT of every candidate evidence paragraph
      printed underneath it
    - blank spaces showing exactly what you must decide

You read the .txt file, decide, then type your answers into the .jsonl file.

HOW TO RUN
----------
    cd /mnt/home/user41/URBench
    python eval/error_analysis_tests/efbpt_audit30_viewer.py

OUTPUT
------
    data/strategyqa_official/efbpt/audit30_readable.txt      <- read this
    data/strategyqa_official/efbpt/audit30_answers.jsonl     <- write answers here
                                                                (pre-created empty
                                                                 template, 30 rows)
"""

import json
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
OFF   = BASE / "data/strategyqa_official"
EFBPT = OFF / "efbpt"

CANDIDATES = EFBPT / "audit30_candidates.jsonl"          # made by the prefill script
PARAGRAPHS = OFF / "strategyqa_train_paragraphs.json"    # official paragraph texts
READABLE   = EFBPT / "audit30_readable.txt"              # you READ this
ANSWERS    = EFBPT / "audit30_answers.jsonl"             # you WRITE here


# ---------------------------------------------------------------------------
# FUNCTION 1: load_jsonl
# Plain English: a .jsonl file is a text file where EVERY LINE is one JSON
# record. This function opens the file, reads it line by line, turns each line
# into a Python dictionary, and returns a list of those dictionaries.
# ---------------------------------------------------------------------------
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():                 # skip empty lines
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# FUNCTION 2: get_paragraph_text
# Plain English: the evidence is stored as an ID, e.g. "Soup-1". The official
# paragraphs file is one big dictionary: ID -> information about that paragraph.
# This function looks up the ID and returns the actual paragraph TEXT.
# If the ID is not found, it says so instead of crashing.
# ---------------------------------------------------------------------------
def get_paragraph_text(para_dict, pid):
    entry = para_dict.get(pid)
    if entry is None:
        return "[[PARAGRAPH ID NOT FOUND: " + pid + "]]"
    # the official file stores a dict with a "content" field (the paragraph text)
    if isinstance(entry, dict):
        return entry.get("content") or entry.get("text") or str(entry)
    return str(entry)


# ---------------------------------------------------------------------------
# FUNCTION 3: wrap
# Plain English: long paragraphs are hard to read on one giant line. This
# breaks the text into lines of about 100 characters so it is comfortable to read.
# ---------------------------------------------------------------------------
def wrap(text, width=100, indent="        "):
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(indent + current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(indent + current)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FUNCTION 4: main
# Plain English: this is the part that actually does the work. It:
#   1. loads the 30 audit rows and the official paragraph texts
#   2. for each row, writes a readable block into audit30_readable.txt
#   3. creates an empty answer template (audit30_answers.jsonl) for you to fill
# ---------------------------------------------------------------------------
def main():
    rows = load_jsonl(CANDIDATES)

    with open(PARAGRAPHS, encoding="utf-8") as f:
        paragraphs = json.load(f)          # one big dictionary: pid -> paragraph

    print("audit rows:", len(rows))
    print("paragraph entries:", len(paragraphs))

    out = open(READABLE, "w", encoding="utf-8")
    ans = open(ANSWERS, "w", encoding="utf-8")

    out.write("EFBPT AUDIT30 — READABLE AUDIT SHEET\n")
    out.write("=" * 100 + "\n")
    out.write("""
HOW TO DECIDE (short guide)

  type:
      RETRIEVE = the step LOOKS UP a fact (even a fact about a previous answer,
                 e.g. "Where was #1 born?" is RETRIEVE)
      REASON   = the step COMPARES / COMPUTES / DECIDES using earlier answers,
                 e.g. "Is #1 before #2?"

  expected_answer_type: BOOLEAN | ENTITY | LOCATION | DATE | NUMBER | SET | SHORT_TEXT

  evidence_ref: read each candidate paragraph below the step. Keep ONLY the ones
                that TRULY answer that step. Drop the rest.

  urdu_span: copy the exact Urdu words for the entity, from the Urdu question.

  verified_incorrect (for each hard negative):
      true  = this title is genuinely a WRONG match (good negative -> keep)
      false = this title could actually be correct/an alias (bad negative -> drop)

  verdict:
      CLEAN            = everything resolved with no problems
      USABLE-WITH-EDIT = small fix needed (under ~2 minutes)
      BROKEN           = Urdu question changed the meaning, OR an entity cannot
                         be found in the Urdu question at all

""")
    out.write("=" * 100 + "\n\n")

    for i, r in enumerate(rows, 1):
        out.write("#" * 100 + "\n")
        out.write(f"ROW {i}/30   qid = {r['urbench_qid']}\n")
        out.write("#" * 100 + "\n\n")

        out.write(f"URDU QUESTION    : {r.get('question_ur')}\n")
        out.write(f"ENGLISH QUESTION : {r.get('question_en')}\n")
        out.write(f"GOLD ANSWER      : {r.get('answer')}\n")
        out.write(f"GOLD TERM        : {r.get('term')}\n\n")

        # ---- entities to label ----
        out.write("-" * 100 + "\n")
        out.write("QUESTION ENTITIES  (label these)\n")
        out.write("-" * 100 + "\n")
        for e in r.get("question_entities", []):
            if not e.get("canonical_title"):
                out.write("\n  [EMPTY SLOT] add any other entity that appears in the Urdu question\n")
                out.write("      canonical_title : ______________________\n")
                out.write("      urdu_span       : ______________________\n")
                continue
            out.write(f"\n  canonical_title : {e['canonical_title']}\n")
            out.write(f"  urdu_span       : ______________________  <-- YOU FILL (copy from Urdu question)\n")
            negs = e.get("hard_negatives", [])
            if negs:
                out.write("  candidate hard negatives (mark each true = truly wrong, false = actually valid):\n")
                for n in negs:
                    out.write(f"      [ ____ ]  {n['title']}\n")
            else:
                out.write("  candidate hard negatives: (none found)\n")
        out.write("\n")

        # ---- plan steps with readable evidence ----
        out.write("-" * 100 + "\n")
        out.write("PLAN STEPS  (type them, pick answer type, keep only true evidence)\n")
        out.write("-" * 100 + "\n")
        for s in r.get("typed_plan", []):
            out.write(f"\n  STEP {s['id']}: {s['question_en']}\n")
            out.write(f"      depends_on          : {s['depends_on']}\n")
            out.write(f"      type_suggested      : {s['type_suggested']}   (hint only)\n")
            out.write(f"      type                : ______________  <-- RETRIEVE or REASON\n")
            out.write(f"      entity_ref          : ______________  <-- which entity is this about\n")
            out.write(f"      expected_answer_type: ______________\n")
            out.write(f"      gold_intermediate_answer: ______________  <-- if a paragraph states it\n")
            out.write("      CANDIDATE EVIDENCE (keep only the ones that truly answer this step):\n")
            cands = s.get("evidence_candidates", [])
            if not cands:
                out.write("          (no evidence candidates — likely an operation/REASON step)\n")
            for pid in cands:
                text = get_paragraph_text(paragraphs, pid)
                out.write(f"\n        [ KEEP? ____ ]  {pid}\n")
                out.write(wrap(text) + "\n")
            out.write("\n")

        # ---- verdict block ----
        out.write("-" * 100 + "\n")
        out.write("VERDICT FOR THIS ROW\n")
        out.write("  semantic_mismatch (Urdu question changed the meaning?) : ______  (true/false)\n")
        out.write("  entity_unresolvable (entity not findable in Urdu?)     : ______  (true/false)\n")
        out.write("  plan_untypeable (a step fits neither RETRIEVE/REASON?) : ______  (true/false)\n")
        out.write("  evidence_unlinked (a RETRIEVE step has no true evidence?): ______  (true/false)\n")
        out.write("  VERDICT : ______________  (CLEAN / USABLE-WITH-EDIT / BROKEN)\n")
        out.write("  NOTES   : ______________________________________________\n")
        out.write("\n\n")

        # ---- empty answer template for this row ----
        ans.write(json.dumps({
            "urbench_qid": r["urbench_qid"],
            "question_entities": [
                {"canonical_title": e.get("canonical_title", ""),
                 "urdu_span": "",
                 "hard_negatives_verified": []}
                for e in r.get("question_entities", []) if e.get("canonical_title")
            ],
            "typed_plan": [
                {"id": s["id"], "type": "", "entity_ref": "",
                 "expected_answer_type": "", "evidence_ref": [],
                 "gold_intermediate_answer": None}
                for s in r.get("typed_plan", [])
            ],
            "flags": {"semantic_mismatch": None, "entity_unresolvable": None,
                      "plan_untypeable": None, "evidence_unlinked": None},
            "verdict": "", "notes": ""
        }, ensure_ascii=False) + "\n")

    out.close()
    ans.close()
    print("\nREAD THIS  :", READABLE)
    print("WRITE HERE :", ANSWERS)
    print("\nStart with ROWS 1-3 only, then send them for calibration review.")


if __name__ == "__main__":
    main()