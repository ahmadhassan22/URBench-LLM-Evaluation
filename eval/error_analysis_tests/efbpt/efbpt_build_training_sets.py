#!/usr/bin/env python3
"""
efbpt_build_training_sets.py

Builds the C1 / C2 / C3 QLoRA training files from the frozen Plan A' gold file.

Contract: docs/EFBPT_PLAN_A_FREEZE.md
  - AMENDMENT 3 Section A : serialization + frozen key order
  - AMENDMENT 4           : fixed system message + frozen Urdu instruction file

C0 is the untrained base model and needs no training file.

Usage (run from ~/URBench):
  python eval/error_analysis_tests/efbpt/efbpt_build_training_sets.py --dry-run
  python eval/error_analysis_tests/efbpt/efbpt_build_training_sets.py

Write-safety:
  --limit N is only allowed together with --dry-run (never writes a partial file).
  Refuses to overwrite existing output files.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict

# ----------------------------------------------------------------------------
# Frozen constants. Do not edit without a new dated amendment.
# ----------------------------------------------------------------------------

GOLD_PATH = "data/strategyqa_official/efbpt/plan_a_gold_100.jsonl"
CANDIDATES_PATH = "data/strategyqa_official/efbpt/plan_a_candidates_100.jsonl"
INSTRUCTION_PATH = "prompts/efbpt/plan_a_instruction_ur.txt"
INSTRUCTION_MD5 = "f3b58d766fe3ec2573ff4f24761cf0c9"
INSTRUCTION_NCHARS = 20
INSTRUCTION_NBYTES = 36

SYSTEM_MESSAGE = "You are a helpful assistant. Answer the user's question."

OUT_DIR = "data/strategyqa_official/efbpt/train"
OUT_FILES = {
    "C1": os.path.join(OUT_DIR, "plan_a_train_c1_100.jsonl"),
    "C2": os.path.join(OUT_DIR, "plan_a_train_c2_100.jsonl"),
    "C3": os.path.join(OUT_DIR, "plan_a_train_c3_100.jsonl"),
}

EXPECTED_ROWS = 100

VALID_TYPES = {"retrieve", "reason"}
VALID_ATYPES = {"BOOLEAN", "ENTITY", "LOCATION", "DATE", "NUMBER", "SET", "SHORT_TEXT"}
VALID_ANSWERS = {"yes", "no"}

# AMENDMENT 2: a bridge entity_ref on a "#N" retrieve step may name an entity
# outside the question's own entity list. It is validated against the candidate
# title universe for that qid. Three refs in the gold file were freely typed
# outside that universe; each was web-verified during human review. They are
# listed here explicitly so the allow-list is tamper-evident: if a fourth ever
# appears, the build fails instead of silently accepting it.
EXTERNAL_ALLOWLIST = {
    "Pantheon",
    "Seoul",
    "List of Hey Arnold! characters",
}

# JSON serialization of the target. Frozen: compact separators, Urdu kept as
# real characters (never \u escapes), key order preserved via OrderedDict.
JSON_KW = dict(ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_instruction():
    """Read the frozen Urdu instruction and verify it byte for byte."""
    if not os.path.exists(INSTRUCTION_PATH):
        die("instruction file missing: " + INSTRUCTION_PATH)

    raw = open(INSTRUCTION_PATH, "rb").read()
    actual_md5 = hashlib.md5(raw).hexdigest()

    if actual_md5 != INSTRUCTION_MD5:
        die(
            "instruction MD5 mismatch.\n"
            "  expected: %s\n"
            "  actual  : %s\n"
            "The frozen instruction file has changed. Stop and investigate."
            % (INSTRUCTION_MD5, actual_md5)
        )
    if len(raw) != INSTRUCTION_NBYTES:
        die("instruction byte length %d, expected %d" % (len(raw), INSTRUCTION_NBYTES))

    text = raw.decode("utf-8")
    if len(text) != INSTRUCTION_NCHARS:
        die("instruction char length %d, expected %d" % (len(text), INSTRUCTION_NCHARS))
    if text != text.strip():
        die("instruction has leading/trailing whitespace")

    return text


def normalize_entity_ref(value):
    """Gold may store an absent entity_ref as null or as an empty string."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


# ----------------------------------------------------------------------------
# Validation of one gold row
# ----------------------------------------------------------------------------

def validate_row(row, idx, candidate_titles, external_hits):
    where = "row %d (qid=%s)" % (idx, row.get("qid", "?"))

    for key in ("qid", "question_ur", "entities", "steps", "answer"):
        if key not in row:
            die("%s: missing top-level key '%s'" % (where, key))

    question = row["question_ur"]
    if not isinstance(question, str) or question.strip() == "":
        die("%s: question_ur is empty or not a string" % where)

    answer = row["answer"]
    if answer not in VALID_ANSWERS:
        die("%s: answer is %r, expected 'yes' or 'no'" % (where, answer))

    entities = row["entities"]
    if not isinstance(entities, list):
        die("%s: entities is not a list" % where)

    titles = []
    for ent in entities:
        for key in ("canonical_title", "urdu_span"):
            if key not in ent:
                die("%s: entity missing '%s'" % (where, key))
        title = ent["canonical_title"]
        span = ent["urdu_span"]
        if not isinstance(title, str) or title.strip() == "":
            die("%s: empty canonical_title" % where)
        if not isinstance(span, str) or span.strip() == "":
            die("%s: empty urdu_span" % where)
        if span not in question:
            die("%s: urdu_span not verbatim in question_ur (title=%s)" % (where, title))
        titles.append(title)

    if len(titles) != len(set(titles)):
        die("%s: duplicate canonical_title in entities" % where)

    steps = row["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        die("%s: steps is empty or not a list" % where)

    title_set = set(titles)
    allowed = title_set | candidate_titles | EXTERNAL_ALLOWLIST

    for pos, step in enumerate(steps):
        for key in ("step_id", "text", "type", "atype"):
            if key not in step:
                die("%s: step %d missing '%s'" % (where, pos + 1, key))

        if step["step_id"] != pos + 1:
            die(
                "%s: step_id %r at position %d — step_id must be 1..N with no gaps"
                % (where, step["step_id"], pos + 1)
            )

        text = step["text"]
        if not isinstance(text, str) or text.strip() == "":
            die("%s: step %d has empty text" % (where, pos + 1))

        stype = step["type"]
        if stype not in VALID_TYPES:
            die("%s: step %d type %r not in %s" % (where, pos + 1, stype, sorted(VALID_TYPES)))

        atype = step["atype"]
        if atype not in VALID_ATYPES:
            die("%s: step %d atype %r not in %s" % (where, pos + 1, atype, sorted(VALID_ATYPES)))

        ref = normalize_entity_ref(step.get("entity_ref"))
        if stype == "retrieve":
            if ref is None:
                die("%s: step %d is 'retrieve' but entity_ref is empty" % (where, pos + 1))
            if ref not in allowed:
                die(
                    "%s: step %d entity_ref %r is outside the allowed universe "
                    "(row entities + candidate_titles + external allow-list)"
                    % (where, pos + 1, ref)
                )
            if ref not in title_set and ref not in candidate_titles:
                external_hits.append((row["qid"], step["step_id"], ref))
        else:  # reason
            if ref is not None:
                die(
                    "%s: step %d is 'reason' but entity_ref is %r (must be empty)"
                    % (where, pos + 1, ref)
                )


# ----------------------------------------------------------------------------
# Target serialization (AMENDMENT 3, Section A). Key order is frozen.
# ----------------------------------------------------------------------------

def target_c1(row):
    obj = OrderedDict()
    obj["answer"] = row["answer"]
    return json.dumps(obj, **JSON_KW)


def target_c2(row):
    steps = []
    for step in row["steps"]:
        s = OrderedDict()
        s["step_id"] = step["step_id"]
        s["text"] = step["text"]
        s["type"] = step["type"]
        s["atype"] = step["atype"]
        steps.append(s)

    obj = OrderedDict()
    obj["steps"] = steps
    obj["answer"] = row["answer"]
    return json.dumps(obj, **JSON_KW)


def target_c3(row):
    entities = []
    for ent in row["entities"]:
        e = OrderedDict()
        e["canonical_title"] = ent["canonical_title"]
        e["urdu_span"] = ent["urdu_span"]
        entities.append(e)

    steps = []
    for step in row["steps"]:
        s = OrderedDict()
        s["step_id"] = step["step_id"]
        s["text"] = step["text"]
        s["type"] = step["type"]
        s["entity_ref"] = normalize_entity_ref(step.get("entity_ref"))
        s["atype"] = step["atype"]
        steps.append(s)

    obj = OrderedDict()
    obj["entities"] = entities
    obj["steps"] = steps
    obj["answer"] = row["answer"]
    return json.dumps(obj, **JSON_KW)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print one example per condition; write nothing")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N rows (requires --dry-run)")
    args = ap.parse_args()

    # Write-safety guard: --limit may never produce a partial output file.
    if args.limit is not None and not args.dry_run:
        die("--limit requires --dry-run (refusing to write a partial file)")

    if not os.path.exists(GOLD_PATH):
        die("gold file missing: " + GOLD_PATH)

    # Refuse to overwrite before doing any work.
    if not args.dry_run:
        for cond, path in OUT_FILES.items():
            if os.path.exists(path):
                die("output already exists, refusing to overwrite: " + path)

    instruction = load_instruction()
    print("instruction verified: %d chars, MD5 %s" % (len(instruction), INSTRUCTION_MD5))
    print("gold file MD5: %s" % md5_of_file(GOLD_PATH))

    # ---- load + validate ----
    rows = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                die("gold line %d is not valid JSON: %s" % (i, e))

    if args.limit is not None:
        rows = rows[: args.limit]
    elif len(rows) != EXPECTED_ROWS:
        die("gold has %d rows, expected %d" % (len(rows), EXPECTED_ROWS))

    qids = [r.get("qid") for r in rows]
    if len(qids) != len(set(qids)):
        die("duplicate qid in gold file")

    # ---- candidate title universe (AMENDMENT 2) ----
    if not os.path.exists(CANDIDATES_PATH):
        die("candidates file missing: " + CANDIDATES_PATH)

    candidates = {}
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                die("candidates line %d is not valid JSON: %s" % (i, e))
            candidates[c["qid"]] = set(c.get("candidate_titles") or [])

    missing = [q for q in qids if q not in candidates]
    if missing:
        die("%d gold qid(s) absent from candidates file, first: %s"
            % (len(missing), missing[0]))

    external_hits = []
    outside_entities = 0

    for i, row in enumerate(rows, start=1):
        validate_row(row, i, candidates[row["qid"]], external_hits)
        row_titles = {e["canonical_title"] for e in row["entities"]}
        for step in row["steps"]:
            ref = normalize_entity_ref(step.get("entity_ref"))
            if ref is not None and ref not in row_titles:
                outside_entities += 1

    # Tamper-evidence: exactly the three documented externals, nothing more.
    used_external = {ref for (_, _, ref) in external_hits}
    if used_external != EXTERNAL_ALLOWLIST:
        die(
            "external entity_ref set changed.\n"
            "  expected: %s\n"
            "  actual  : %s\n"
            "Stop. Do not widen the allow-list without a dated amendment."
            % (sorted(EXTERNAL_ALLOWLIST), sorted(used_external))
        )

    print("bridge refs outside the row's own entities list: %d "
          "(allowed by AMENDMENT 2)" % outside_entities)
    print("refs outside entities + candidate_titles: %d, matching the "
          "documented allow-list exactly" % len(external_hits))

    # ---- build ----
    built = {"C1": [], "C2": [], "C3": []}

    for row in rows:
        user = instruction + "\n\n" + row["question_ur"]

        t1 = target_c1(row)
        t2 = target_c2(row)
        t3 = target_c3(row)

        # AMENDMENT 3: C2 and C3 step text must be byte-identical.
        s2 = [s["text"] for s in json.loads(t2)["steps"]]
        s3 = [s["text"] for s in json.loads(t3)["steps"]]
        if s2 != s3:
            die("qid %s: C2 and C3 step text differ" % row["qid"])
        for a, b in zip(s2, s3):
            if a.encode("utf-8") != b.encode("utf-8"):
                die("qid %s: C2/C3 step text not byte-identical" % row["qid"])

        for cond, target in (("C1", t1), ("C2", t2), ("C3", t3)):
            rec = OrderedDict()
            rec["qid"] = row["qid"]
            rec["condition"] = cond
            rec["system"] = SYSTEM_MESSAGE
            rec["user"] = user
            rec["target"] = target
            built[cond].append(rec)

    # ---- dry run: show one full example per condition ----
    if args.dry_run:
        row = rows[0]
        print("\n" + "=" * 70)
        print("DRY RUN — full example for qid %s" % row["qid"])
        print("=" * 70)
        print("\nSYSTEM:\n%s" % SYSTEM_MESSAGE)
        print("\nUSER (repr, so the Urdu is not mangled by the terminal):")
        print(repr(built["C1"][0]["user"]))
        for cond in ("C1", "C2", "C3"):
            print("\n" + "-" * 70)
            print("TARGET %s:" % cond)
            print(built[cond][0]["target"])
        print("\n" + "=" * 70)
        print("rows validated: %d" % len(rows))
        print("C2/C3 step text byte-identical on all rows: OK")
        print("DRY RUN — nothing was written.")
        return

    # ---- write ----
    os.makedirs(OUT_DIR, exist_ok=True)

    for cond, path in OUT_FILES.items():
        recs = built[cond]
        if len(recs) != EXPECTED_ROWS:
            die("refusing to write %s: %d rows, expected %d" % (path, len(recs), EXPECTED_ROWS))

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, **JSON_KW) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)

    print("\nWROTE:")
    for cond in ("C1", "C2", "C3"):
        path = OUT_FILES[cond]
        n = sum(1 for _ in open(path, "r", encoding="utf-8"))
        print("  %s  rows=%d  md5=%s  %s" % (cond, n, md5_of_file(path), path))
        if n != EXPECTED_ROWS:
            die("post-write row count wrong for %s" % path)

    print("\nRecord these MD5s in experiments.md.")


if __name__ == "__main__":
    main()