#!/usr/bin/env python3
"""
efbpt_plan_a_review.py — CPU. Interactive human reviewer for Plan A' (100 rows).

Reads   : plan_a_drafts_100.jsonl      (model drafts -- NEVER modified)
          plan_a_candidates_100.jsonl  (source of truth for immutable fields)
Writes  : plan_a_gold_100.jsonl        (frozen Schema C only, one line per row)
          plan_a_review_audit.jsonl    (notes, diffs, history -- separate file)

RULES BUILT IN
  - qid / question_ur / answer / step text are IMMUTABLE (AMENDMENT 2b).
    Disagreement goes into a note, never into an edit.
  - Urdu spans are chosen by WORD NUMBERS and sliced from the original
    question string by character offsets. Free typing is allowed but must be
    an exact substring of the question, or it is rejected.
  - Editing or deleting an entity marks every step that referenced it as
    STALE: those steps cannot be accepted with plain Enter -- you must
    explicitly keep or change the reference.
  - reason  -> entity_ref is null, always.
    retrieve -> entity_ref required. It may be (a) an entity, (b) a candidate
    title, or (c) a free-typed external title (AMENDMENT 2c bridge). External
    references always show a warning.
  - Saving appends ONE complete validated JSON line, then flush + fsync.
  - Resume: on start the gold file is validated (parse, duplicate qids,
    qids must belong to the manifest). Review continues at the first row,
    in draft order, that is not yet saved.
  - Reopen a saved row:  --reopen QID   (previous version is preserved in
    the audit file; the gold file is rewritten atomically, no duplicates).
  - Stats:               --stats        (reviewer-draft correction rates)

Usage:
  python eval/error_analysis_tests/efbpt/efbpt_plan_a_review.py
  python eval/error_analysis_tests/efbpt/efbpt_plan_a_review.py --reopen <qid>
  python eval/error_analysis_tests/efbpt/efbpt_plan_a_review.py --stats
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path("/mnt/home/user41/URBench")
EF = BASE / "data/strategyqa_official/efbpt"
DRAFTS_FILE = EF / "plan_a_drafts_100.jsonl"
CANDS_FILE = EF / "plan_a_candidates_100.jsonl"
GOLD_FILE = EF / "plan_a_gold_100.jsonl"
AUDIT_FILE = EF / "plan_a_review_audit.jsonl"

STEP_TYPES = ["retrieve", "reason"]
ANSWER_TYPES = ["BOOLEAN", "ENTITY", "LOCATION", "DATE", "NUMBER", "SET", "SHORT_TEXT"]

HR = "-" * 78
HR2 = "=" * 78


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def norm(s):
    """Frozen normalisation used across the project."""
    return re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_", " ")).strip().casefold()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                die(f"{path} line {i} is not valid JSON: {e}")
    return rows


def append_fsync(path, obj):
    """Append one JSON line, then flush + fsync so a crash cannot lose it."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_rewrite(path, rows):
    """Rewrite a JSONL file safely: temp file in the same folder, fsync, replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def die(msg):
    print(f"\nFATAL: {msg}")
    sys.exit(1)


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        print("\nInput closed. Current row NOT saved. Progress up to the last "
              "'okay' is safe. Bye.")
        sys.exit(0)


# ---------------------------------------------------------------------------
# word tokeniser with character offsets (safeguard 1)
# ---------------------------------------------------------------------------

def tokenise(question):
    """Return [(word_text, start, end), ...] using offsets into the ORIGINAL
    string. A span is always question[start_i:end_j] -- the original text is
    sliced, never rebuilt, so whitespace and punctuation cannot change."""
    return [(m.group(0), m.start(), m.end())
            for m in re.finditer(r"\S+", question)]


def show_words(tokens):
    print("  words (one per line):")
    for i, (w, _s, _e) in enumerate(tokens, start=1):
        print(f"    {i:>2}  {w}")


def pick_span(question, tokens):
    """Ask for a span. Word numbers slice the original string by offsets.
    Free text must be an exact substring of the question.
    Returns the span string, or None if the user cancels."""
    while True:
        raw = ask("  span> (e.g. 4-5, or 4, or t=type text, or c=cancel): ").strip()
        if raw.lower() == "c":
            return None
        if raw.lower() == "t":
            typed = ask("  type the span EXACTLY as it appears in the question:\n  > ").strip()
            if not typed:
                print("  empty -- try again.")
                continue
            if typed in question:
                return typed
            print("  REJECTED: that text is not an exact substring of the question.")
            print("  Use word numbers instead -- they cannot be wrong.")
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", raw)
        if not m:
            print("  did not understand -- examples: 4   or   4-5   or   t   or   c")
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a < 1 or b > len(tokens) or a > b:
            print(f"  out of range -- valid word numbers are 1..{len(tokens)}")
            continue
        span = question[tokens[a - 1][1]:tokens[b - 1][2]]
        print(f"  span = {span}")
        return span


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------

def show_entities(entities):
    if not entities:
        print("  (no entities)")
    for i, e in enumerate(entities, start=1):
        print(f"  E{i}  {e['canonical_title']}  |  {e['urdu_span']}")


def show_step(i, s, stale=False):
    ref = s["entity_ref"] if s["entity_ref"] is not None else "null"
    tag = "  [STALE: entity changed -- ref needs your decision]" if stale else ""
    print(f"  S{i}  type={s['type']}  ref={ref}  atype={s['atype']}{tag}")
    print(f"      \"{s['text']}\"")


def show_row(header, cand, entities, steps, flags, decisions, stale_ids):
    print(HR2)
    print(header)
    print(HR2)
    print("QUESTION (Urdu):")
    print(f"  {cand['question_ur']}")
    print(f"QUESTION (English):  {cand['question_en']}")
    print(f"ANSWER: {cand['answer']}   (frozen)")
    print(HR)
    print("CANDIDATE TITLES (hints; model YES/NO shown):")
    said_yes = set()
    for d in decisions.get("title_decisions", []):
        if d.get("named_in_question") is True:
            said_yes.add(norm(d.get("title", "")))
    for i, t in enumerate(cand["candidate_titles"], start=1):
        mark = "YES" if norm(t) in said_yes else "no "
        print(f"  C{i}  [{mark}] {t}")
    print(HR)
    print("ENTITIES:")
    show_entities(entities)
    print(HR)
    print("STEPS (text is frozen):")
    for i, s in enumerate(steps, start=1):
        show_step(i, s, stale=(i in stale_ids))
    print(HR)
    print("AUTOMATIC FLAGS on this draft:")
    if flags:
        for f in flags:
            print(f"  - {f}")
    else:
        print("  (none -- remember: zero flags means structurally clean, NOT correct)")
    print(HR)
    print("HUMAN-ONLY CHECKS (no script can do these):")
    print("  * Is each Wikipedia page the RIGHT SENSE for this question?")
    print("  * Does each bridge entity_ref point to the entity that step really")
    print("    resolves to? Follow the #N chain yourself.")
    print(HR2)


# ---------------------------------------------------------------------------
# entity review
# ---------------------------------------------------------------------------

def clean_title(raw):
    t = raw.replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def title_warnings(title, entities, exclude_index=None):
    w = []
    if title.lower().startswith("list of"):
        w.append("this is a 'List of ...' page -- not an entity under the policy")
    for j, e in enumerate(entities):
        if exclude_index is not None and j == exclude_index:
            continue
        if norm(e["canonical_title"]) == norm(title):
            w.append(f"duplicate of E{j+1} ({e['canonical_title']})")
        elif e["urdu_span"]:
            pass
    return w


def edit_entity(entity, question, tokens, entities, index):
    """Edit one entity in place. Returns list of (field, before, after)."""
    changes = []
    print(f"\n  editing E{index+1}: {entity['canonical_title']}  |  {entity['urdu_span']}")
    nt = ask(f"  new title (Enter = keep '{entity['canonical_title']}'): ").strip()
    if nt:
        nt = clean_title(nt)
        for w in title_warnings(nt, entities, exclude_index=index):
            print(f"  WARNING: {w}")
        if nt != entity["canonical_title"]:
            changes.append(("canonical_title", entity["canonical_title"], nt))
            entity["canonical_title"] = nt
    resp = ask("  change span? (Enter = keep, s = pick new span): ").strip().lower()
    if resp == "s":
        show_words(tokens)
        span = pick_span(question, tokens)
        if span is not None and span != entity["urdu_span"]:
            changes.append(("urdu_span", entity["urdu_span"], span))
            entity["urdu_span"] = span
    return changes


def add_entity(question, tokens, entities):
    """Create a new entity dict, or None if cancelled."""
    print("\n  adding a new entity")
    title = ask("  Wikipedia page title (Enter = cancel): ").strip()
    if not title:
        return None
    title = clean_title(title)
    for w in title_warnings(title, entities):
        print(f"  WARNING: {w}")
    show_words(tokens)
    span = pick_span(question, tokens)
    if span is None:
        return None
    return {"canonical_title": title, "urdu_span": span}


def review_entities(entities, question, tokens):
    """Walk each entity; allow edit/delete; allow adding at the end.
    Returns (entities, changes, touched_norms) where touched_norms are the
    normalised titles of every entity that was renamed or deleted."""
    changes = []
    touched = set()
    i = 0
    while i < len(entities):
        e = entities[i]
        print(f"\nENTITY E{i+1}:  {e['canonical_title']}  |  {e['urdu_span']}")
        cmd = ask("  Enter=accept  e=edit  d=delete  n=note : ").strip().lower()
        if cmd == "":
            i += 1
        elif cmd == "e":
            before_norm = norm(e["canonical_title"])
            ch = edit_entity(e, question, tokens, entities, i)
            if any(f == "canonical_title" for f, _b, _a in ch):
                touched.add(before_norm)
            for f, b, a in ch:
                changes.append({"where": f"entity:{b if f=='canonical_title' else e['canonical_title']}",
                                "field": f, "before": b, "after": a})
            i += 1
        elif cmd == "d":
            sure = ask(f"  delete E{i+1} '{e['canonical_title']}'? type d again: ").strip().lower()
            if sure == "d":
                touched.add(norm(e["canonical_title"]))
                changes.append({"where": f"entity:{e['canonical_title']}",
                                "field": "deleted", "before": e, "after": None})
                entities.pop(i)
            else:
                print("  not deleted.")
        elif cmd == "n":
            note = ask("  note text: ").strip()
            if note:
                changes.append({"where": f"entity:{e['canonical_title']}",
                                "field": "note", "before": None, "after": note})
            i += 1
        else:
            print("  did not understand.")
    while True:
        cmd = ask("\n  add an entity? a=add  Enter=done : ").strip().lower()
        if cmd == "a":
            ne = add_entity(question, tokens, entities)
            if ne is not None:
                entities.append(ne)
                changes.append({"where": f"entity:{ne['canonical_title']}",
                                "field": "added", "before": None, "after": ne})
                print(f"  added E{len(entities)}: {ne['canonical_title']} | {ne['urdu_span']}")
        elif cmd == "":
            break
        else:
            print("  did not understand.")
    return entities, changes, touched


# ---------------------------------------------------------------------------
# step review
# ---------------------------------------------------------------------------

def classify_ref(ref, entities, cand_titles):
    if ref is None:
        return "null"
    n = norm(ref)
    if any(norm(e["canonical_title"]) == n for e in entities):
        return "entity"
    if any(norm(t) == n for t in cand_titles):
        return "candidate"
    return "external"


def choose_ref(entities, cand_titles):
    """Pick an entity_ref. Returns (ref_string or None-if-cancel, kind)."""
    print("  choose the reference:")
    for i, e in enumerate(entities, start=1):
        print(f"    E{i}  {e['canonical_title']}")
    for i, t in enumerate(cand_titles, start=1):
        print(f"    C{i}  {t}")
    while True:
        raw = ask("  ref> (E2 / C3 / t=type external title / c=cancel): ").strip()
        low = raw.lower()
        if low == "c":
            return None, None
        if low == "t":
            t = ask("  external Wikipedia title (AMENDMENT 2c bridge): ").strip()
            if not t:
                continue
            t = clean_title(t)
            print("  WARNING: external reference -- not an entity of this question and")
            print("           not in the candidate list. Allowed for #N bridges, but")
            print("           double-check the page really exists and is the right one.")
            return t, "external"
        m = re.fullmatch(r"[eE](\d+)", raw)
        if m and 1 <= int(m.group(1)) <= len(entities):
            return entities[int(m.group(1)) - 1]["canonical_title"], "entity"
        m = re.fullmatch(r"[cC](\d+)", raw)
        if m and 1 <= int(m.group(1)) <= len(cand_titles):
            return cand_titles[int(m.group(1)) - 1], "candidate"
        print("  did not understand.")


def review_steps(steps, entities, cand_titles, stale_ids):
    """Walk each step. Steps in stale_ids cannot be accepted with plain Enter."""
    changes = []
    i = 0
    while i < len(steps):
        s = steps[i]
        sid = i + 1
        stale = sid in stale_ids
        print()
        show_step(sid, s, stale=stale)
        kind = classify_ref(s["entity_ref"], entities, cand_titles)
        if kind == "external":
            print("  WARNING: ref is EXTERNAL (not an entity here, not a candidate).")
        if s["type"] == "retrieve" and s["entity_ref"] is None:
            print("  PROBLEM: retrieve step with no reference -- must be fixed.")
            stale = True
        if stale:
            prompt = "  k=keep ref  r=new ref  t=type  y=atype  n=note : "
        else:
            prompt = "  Enter=accept  t=type  r=ref  y=atype  n=note : "
        cmd = ask(prompt).strip().lower()

        if cmd == "" and not stale:
            i += 1
            continue
        if cmd == "k" and stale:
            if s["type"] == "retrieve" and s["entity_ref"] is None:
                print("  cannot keep: retrieve needs a reference.")
                continue
            print("  reference kept by your explicit decision.")
            stale_ids.discard(sid)
            i += 1
            continue
        if cmd == "t":
            new_type = "reason" if s["type"] == "retrieve" else "retrieve"
            sure = ask(f"  change type {s['type']} -> {new_type}? (y/Enter=no): ").strip().lower()
            if sure == "y":
                changes.append({"where": f"step{sid}", "field": "type",
                                "before": s["type"], "after": new_type})
                s["type"] = new_type
                if new_type == "reason" and s["entity_ref"] is not None:
                    changes.append({"where": f"step{sid}", "field": "entity_ref",
                                    "before": s["entity_ref"], "after": None})
                    s["entity_ref"] = None
                    print("  reason step -> reference set to null (frozen rule).")
                if new_type == "retrieve" and s["entity_ref"] is None:
                    print("  retrieve step needs a reference now:")
                    ref, _k = choose_ref(entities, cand_titles)
                    if ref is not None:
                        changes.append({"where": f"step{sid}", "field": "entity_ref",
                                        "before": None, "after": ref})
                        s["entity_ref"] = ref
            continue
        if cmd == "r":
            if s["type"] == "reason":
                print("  reason steps always have null reference (frozen rule).")
                continue
            ref, _k = choose_ref(entities, cand_titles)
            if ref is not None and ref != s["entity_ref"]:
                changes.append({"where": f"step{sid}", "field": "entity_ref",
                                "before": s["entity_ref"], "after": ref})
                s["entity_ref"] = ref
                stale_ids.discard(sid)
            continue
        if cmd == "y":
            print("  atypes: " + "  ".join(f"{i+1}={a}" for i, a in enumerate(ANSWER_TYPES)))
            raw = ask("  atype number (Enter = keep): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(ANSWER_TYPES):
                new_at = ANSWER_TYPES[int(raw) - 1]
                if new_at != s["atype"]:
                    changes.append({"where": f"step{sid}", "field": "atype",
                                    "before": s["atype"], "after": new_at})
                    s["atype"] = new_at
            continue
        if cmd == "n":
            note = ask("  note text: ").strip()
            if note:
                changes.append({"where": f"step{sid}", "field": "note",
                                "before": None, "after": note})
            continue
        print("  did not understand.")
    return steps, changes


# ---------------------------------------------------------------------------
# validation before save (safeguard 6)
# ---------------------------------------------------------------------------

def validate_row(gold, cand):
    """Return (errors, warnings). errors block saving; warnings need eyes."""
    errors, warnings = [], []
    q = cand["question_ur"]

    if gold["qid"] != cand["qid"]:
        errors.append("qid changed -- immutable")
    if gold["question_ur"] != q:
        errors.append("question_ur changed -- immutable")
    if gold["answer"] != cand["answer"]:
        errors.append("answer changed -- immutable")
    if len(gold["steps"]) != len(cand["steps_en"]):
        errors.append("step count changed -- immutable")
    else:
        for i, (s, t) in enumerate(zip(gold["steps"], cand["steps_en"]), start=1):
            if s["text"] != t:
                errors.append(f"step {i} text changed -- immutable (AMENDMENT 2b)")

    if not gold["entities"]:
        warnings.append("no entities at all -- is that truly right for this question?")
    seen = {}
    for i, e in enumerate(gold["entities"], start=1):
        t, sp = e.get("canonical_title", ""), e.get("urdu_span", "")
        if not t or not t.strip():
            errors.append(f"E{i}: empty title")
            continue
        if "_" in t:
            errors.append(f"E{i}: underscore in title '{t}'")
        if t.lower().startswith("list of"):
            warnings.append(f"E{i}: 'List of ...' page ('{t}')")
        if not sp or not sp.strip():
            errors.append(f"E{i}: empty span")
        elif sp not in q:
            errors.append(f"E{i}: span is not an exact substring of the question")
        n = norm(t)
        if n in seen:
            errors.append(f"E{i}: duplicate of E{seen[n]} ('{t}')")
        else:
            seen[n] = i
    by_span = {}
    for e in gold["entities"]:
        by_span.setdefault(e.get("urdu_span", ""), []).append(e.get("canonical_title", ""))
    for sp, titles in by_span.items():
        if sp and len(set(titles)) > 1:
            warnings.append(f"two entities share one span '{sp}': {sorted(set(titles))}")

    ent_norms = {norm(e["canonical_title"]) for e in gold["entities"]}
    cand_norms = {norm(t) for t in cand["candidate_titles"]}
    for i, s in enumerate(gold["steps"], start=1):
        if s.get("step_id") != i:
            errors.append(f"step {i}: step_id must be {i}")
        if s.get("type") not in STEP_TYPES:
            errors.append(f"step {i}: bad type {s.get('type')!r}")
        if s.get("atype") not in ANSWER_TYPES:
            errors.append(f"step {i}: bad atype {s.get('atype')!r}")
        ref = s.get("entity_ref")
        if s.get("type") == "reason" and ref is not None:
            errors.append(f"step {i}: reason step must have null reference")
        if s.get("type") == "retrieve":
            if ref is None or not str(ref).strip():
                errors.append(f"step {i}: retrieve step needs a reference")
            elif "_" in ref:
                errors.append(f"step {i}: underscore in reference '{ref}'")
            elif norm(ref) not in ent_norms and norm(ref) not in cand_norms:
                warnings.append(f"step {i}: EXTERNAL reference '{ref}' "
                                f"(AMENDMENT 2c bridge -- confirm the page is real and right)")
    return errors, warnings


# ---------------------------------------------------------------------------
# one row, start to finish
# ---------------------------------------------------------------------------

def build_baseline(draft_rec, cand):
    """Starting point for review: the model draft if it parsed, otherwise an
    empty skeleton built from the candidates file."""
    if draft_rec is not None and not draft_rec.get("parse_failed"):
        d = draft_rec["draft"]
        entities = [dict(e) for e in d["entities"]]
        steps = [dict(s) for s in d["steps"]]
        flags = list(draft_rec.get("flags", []))
        decisions = draft_rec.get("model_decisions", {})
    else:
        entities = []
        steps = [{"step_id": i, "text": t, "type": None, "entity_ref": None,
                  "atype": None} for i, t in enumerate(cand["steps_en"], start=1)]
        flags = ["parse_failure -- everything below must be filled by hand"]
        decisions = {}
    return entities, steps, flags, decisions


def review_one_row(pos, total, draft_rec, cand, reopen_prev=None):
    """Full interactive review of one row. Returns (gold_row, audit_record)
    or (None, None) if the user quits before saving."""
    qid = cand["qid"]
    question = cand["question_ur"]
    tokens = tokenise(question)

    if reopen_prev is not None:
        entities = [dict(e) for e in reopen_prev["entities"]]
        steps = [dict(s) for s in reopen_prev["steps"]]
        flags = ["(reopened -- baseline is your previously saved gold row)"]
        decisions = draft_rec.get("model_decisions", {}) if draft_rec else {}
    else:
        entities, steps, flags, decisions = build_baseline(draft_rec, cand)

    header = f"ROW {pos} / {total}    qid {qid}" + ("    [REOPENED]" if reopen_prev else "")
    all_changes = []
    while True:
        show_row(header, cand, entities, steps, flags, decisions, set())
        show_words(tokens)

        print("\n--- ENTITY REVIEW ---")
        entities, ch_e, touched = review_entities(entities, question, tokens)
        all_changes.extend(ch_e)

        stale_ids = set()
        for i, s in enumerate(steps, start=1):
            if s["entity_ref"] is not None and norm(s["entity_ref"]) in touched:
                stale_ids.add(i)
        if stale_ids:
            print(f"\n  NOTE: entities changed -- steps {sorted(stale_ids)} reference them")
            print("  and now require your explicit decision (no silent keeping).")

        print("\n--- STEP REVIEW ---")
        steps, ch_s = review_steps(steps, entities, cand["candidate_titles"], stale_ids)
        all_changes.extend(ch_s)

        gold = {
            "qid": qid,
            "question_ur": question,
            "entities": entities,
            "steps": [{"step_id": i, "text": s["text"], "type": s["type"],
                       "entity_ref": s["entity_ref"], "atype": s["atype"]}
                      for i, s in enumerate(steps, start=1)],
            "answer": cand["answer"],
        }
        errors, warnings = validate_row(gold, cand)

        print("\n" + HR2)
        print("FINAL ROW AS IT WILL BE SAVED")
        print(HR2)
        print(f"qid    : {gold['qid']}")
        print(f"answer : {gold['answer']}")
        print("entities:")
        show_entities(gold["entities"])
        print("steps:")
        for i, s in enumerate(gold["steps"], start=1):
            show_step(i, s)
        print(HR)
        real_changes = [c for c in all_changes if c["field"] != "note"]
        notes = [c for c in all_changes if c["field"] == "note"]
        print(f"CHANGES you made ({len(real_changes)}):")
        for c in real_changes:
            print(f"  - {c['where']}: {c['field']}  {c['before']!r} -> {c['after']!r}")
        if not real_changes:
            print("  (none -- draft accepted as-is)")
        if notes:
            print(f"NOTES ({len(notes)}):")
            for c in notes:
                print(f"  - {c['where']}: {c['after']}")
        if errors:
            print("BLOCKING ERRORS -- cannot save until fixed:")
            for e in errors:
                print(f"  !! {e}")
        if warnings:
            print("WARNINGS -- saving means you confirm each one is intended:")
            for w in warnings:
                print(f"  ?? {w}")
        print(HR)

        if errors:
            ask("press Enter to go back and fix the errors...")
            continue
        resp = ask("type exactly 'okay' to save, or 'back' to review again: ").strip()
        if resp == "okay":
            audit = {
                "qid": qid,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "action": "reopen" if reopen_prev is not None else "review",
                "changes": real_changes,
                "notes": [c["after"] for c in notes],
                "warnings_confirmed": warnings,
                "draft_flags": flags,
            }
            if reopen_prev is not None:
                audit["previous_gold"] = reopen_prev
            return gold, audit
        if resp == "back":
            print("\ngoing around again -- everything you set is kept as the new baseline.\n")
            all_changes = list(all_changes)
            continue
        print("not saved. Type 'okay' or 'back'.")


# ---------------------------------------------------------------------------
# gold file loading / resume (safeguard 4)
# ---------------------------------------------------------------------------

def load_gold(valid_qids):
    if not GOLD_FILE.exists():
        return []
    rows = load_jsonl(GOLD_FILE)
    seen = set()
    for i, r in enumerate(rows, start=1):
        qid = r.get("qid")
        if qid not in valid_qids:
            die(f"gold file line {i}: qid {qid!r} is not in the 100-row manifest")
        if qid in seen:
            die(f"gold file line {i}: duplicate qid {qid}")
        seen.add(qid)
        for k in ("qid", "question_ur", "entities", "steps", "answer"):
            if k not in r:
                die(f"gold file line {i}: missing field {k!r}")
    return rows


# ---------------------------------------------------------------------------
# stats (reviewer-draft agreement / correction rates)
# ---------------------------------------------------------------------------

def print_stats():
    if not AUDIT_FILE.exists():
        print("no audit file yet -- nothing reviewed.")
        return
    audits = load_jsonl(AUDIT_FILE)
    latest = {}
    for a in audits:
        latest[a["qid"]] = a          # last action per qid wins
    n = len(latest)
    field_counts = {}
    rows_untouched = 0
    for a in latest.values():
        if not a["changes"]:
            rows_untouched += 1
        for c in a["changes"]:
            field_counts[c["field"]] = field_counts.get(c["field"], 0) + 1
    print("REVIEWER-DRAFT CORRECTION RATES")
    print(HR2)
    print("These are correction counts against the drafts, NOT model accuracy.")
    print("Accuracy needs a defined denominator and entity-matching rule first.")
    print(HR)
    print(f"rows reviewed              : {n}")
    print(f"rows accepted without edits: {rows_untouched}")
    print(f"corrections by field       : {field_counts}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reopen", type=str, default=None,
                    help="qid of a previously saved row to review again")
    ap.add_argument("--stats", action="store_true",
                    help="show reviewer-draft correction rates and exit")
    args = ap.parse_args()

    if args.stats:
        print_stats()
        return

    for p in (DRAFTS_FILE, CANDS_FILE):
        if not p.exists():
            die(f"missing input: {p}")
    drafts = load_jsonl(DRAFTS_FILE)
    cands = load_jsonl(CANDS_FILE)
    if len(cands) != 100:
        die(f"candidates file has {len(cands)} rows, expected 100")
    cand_by_qid = {c["qid"]: c for c in cands}
    draft_by_qid = {d["qid"]: d for d in drafts}
    order = [c["qid"] for c in cands]          # draft order preserved

    gold_rows = load_gold(set(order))
    done = {g["qid"] for g in gold_rows}

    # ---------------- reopen mode (safeguard 5) ----------------
    if args.reopen:
        qid = args.reopen.strip()
        if qid not in cand_by_qid:
            die(f"qid {qid} is not in the manifest")
        if qid not in done:
            die(f"qid {qid} has not been reviewed yet -- run without --reopen")
        prev = next(g for g in gold_rows if g["qid"] == qid)
        pos = order.index(qid) + 1
        gold, audit = review_one_row(pos, len(order), draft_by_qid.get(qid),
                                     cand_by_qid[qid], reopen_prev=prev)
        if gold is None:
            return
        append_fsync(AUDIT_FILE, audit)        # history first, incl. previous version
        new_rows = [gold if g["qid"] == qid else g for g in gold_rows]
        atomic_rewrite(GOLD_FILE, new_rows)
        print(f"\nrow {qid} updated. Gold file rewritten safely "
              f"({len(new_rows)} rows, no duplicates).")
        return

    # ---------------- normal resume loop ----------------
    remaining = [q for q in order if q not in done]
    print(f"\nPlan A' review. {len(done)} of {len(order)} rows already saved. "
          f"{len(remaining)} to go.")
    if not remaining:
        print("All 100 rows are done. Use --stats for correction rates, "
              "or --reopen <qid> to revisit one.")
        return
    print("Ctrl+C between rows is always safe -- saved rows are never lost.\n")

    for qid in remaining:
        pos = order.index(qid) + 1
        try:
            gold, audit = review_one_row(pos, len(order), draft_by_qid.get(qid),
                                         cand_by_qid[qid])
        except KeyboardInterrupt:
            print("\n\nstopped. Current row NOT saved; everything before it is safe.")
            return
        if gold is None:
            return
        append_fsync(GOLD_FILE, gold)
        append_fsync(AUDIT_FILE, audit)
        done.add(qid)
        print(f"\nsaved. {len(done)} / {len(order)} rows complete.\n")

    print("\nALL 100 ROWS REVIEWED. Gold file complete.")
    print("Run with --stats for reviewer-draft correction rates.")


if __name__ == "__main__":
    main()
