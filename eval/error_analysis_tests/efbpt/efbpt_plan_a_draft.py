#!/usr/bin/env python3
"""
efbpt_plan_a_draft.py — GPU. Schema C draft generator for Plan A.

Reads  : data/strategyqa_official/efbpt/plan_a_candidates_100.jsonl
Writes : data/strategyqa_official/efbpt/plan_a_drafts_100.jsonl   (full run)
         data/strategyqa_official/efbpt/plan_a_drafts_testN.jsonl (--limit N)

WHAT THE MODEL DOES (only four jobs):
  1. For every candidate title: is it named in the Urdu question? yes/no
     plus the verbatim Urdu span.
  2. extra_entities: entities clearly named in the question that are missing
     from the candidate list.
  3. Per step: type (retrieve / reason).
  4. Per step: entity_ref and atype.

WHAT THE SCRIPT DOES MECHANICALLY (the model never touches these):
  - qid, question_ur, answer  -> copied from the candidates file
  - steps[].text              -> copied verbatim from official_decomposition
                                 (AMENDMENT 2b, English step text)
  - entities list             -> built from the title decisions marked true,
                                 plus extra_entities (always flagged)

FROZEN SETTINGS (freeze doc): thinking OFF, temperature 0, max_tokens 1024.
finish_reason is recorded for every row so truncation cannot pass unnoticed.

THIS SCRIPT DOES NOT JUDGE QUALITY. It drafts and flags. Structural validation
and human review are separate steps. Mixing them is how Stage 3 went wrong.

Usage:
  python eval/error_analysis_tests/efbpt/efbpt_plan_a_draft.py --dry-run
  python eval/error_analysis_tests/efbpt/efbpt_plan_a_draft.py --limit 5
  python eval/error_analysis_tests/efbpt/efbpt_plan_a_draft.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path("/mnt/home/user41/URBench")
EF = BASE / "data/strategyqa_official/efbpt"

CANDS_FILE = EF / "plan_a_candidates_100.jsonl"
OUT_FULL = EF / "plan_a_drafts_100.jsonl"
SUMMARY_FULL = EF / "plan_a_drafts_100_summary.txt"

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
EXPECTED_N = 100

# Frozen enums (freeze doc 1.5 / 1.6)
STEP_TYPES = ["retrieve", "reason"]
ANSWER_TYPES = ["BOOLEAN", "ENTITY", "LOCATION", "DATE", "NUMBER", "SET", "SHORT_TEXT"]

# Frozen decoding (freeze doc 4.6)
TEMPERATURE = 0.0
MAX_TOKENS = 1024
ENABLE_THINKING = False


def norm(s):
    """Frozen normalisation: underscores -> spaces, collapse whitespace, casefold."""
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


# --------------------------------------------------------------------------
# PROMPT
# --------------------------------------------------------------------------

def build_prompt(row):
    titles = row["candidate_titles"]
    titles_block = "\n".join(f"- {t}" for t in titles)

    steps_block = "\n".join(
        f"STEP {i}: {t}" for i, t in enumerate(row["steps_en"], start=1)
    )
    n_steps = len(row["steps_en"])

    return f"""You are annotating one multi-hop question for a research dataset.
Return ONLY a JSON object. No prose. No markdown fences.

URDU QUESTION: {row['question_ur']}
ENGLISH QUESTION: {row['question_en']}

CANDIDATE PAGE TITLES (hints only -- some are WRONG or IRRELEVANT):
{titles_block}

PLAN STEPS (already written, do not rewrite them):
{steps_block}

Produce this exact JSON structure:
{{
  "title_decisions": [
    {{"title": "<copy one CANDIDATE PAGE TITLE exactly>",
      "named_in_question": true or false,
      "urdu_span": "<the exact Urdu words for it, copied character-for-character from the URDU QUESTION, or empty string>"}}
  ],
  "extra_entities": [
    {{"canonical_title": "<Wikipedia page title of an entity clearly named in the question but MISSING from the candidate list>",
      "urdu_span": "<the exact Urdu words, copied character-for-character from the URDU QUESTION>"}}
  ],
  "steps": [
    {{"step_id": <1..{n_steps}>,
      "type": "retrieve" or "reason",
      "entity_ref": "<see the entity_ref rules>" or null,
      "atype": one of {ANSWER_TYPES}}}
  ]
}}

RULES FOR title_decisions
- Output EXACTLY ONE decision for EVERY candidate title, in the same order. Never skip one.
- The candidate list is a HINT LIST built from evidence pages. It is NOT a list of
  correct answers. Several titles are usually NOT named in the question. Say false for those.
- named_in_question is true ONLY if the URDU QUESTION itself refers to that entity --
  by name, by an Urdu word for it, or by a different word-form
  (for example "body builder" for the page Bodybuilding).
- Being merely related to the topic is NOT enough. If the question does not name it, say false.
- If named_in_question is true, urdu_span MUST be a character-for-character copy of
  words that appear in the URDU QUESTION. Never translate, never reword, never invent.
- If named_in_question is false, urdu_span is an empty string.

RULES FOR extra_entities
- The candidate list is often INCOMPLETE. Read the URDU QUESTION word by word and check
  whether it names any entity that is not in the candidate list. Most questions name
  two or more entities in total.
- Add those here, with the real Wikipedia page title and a verbatim Urdu span.
- Do NOT repeat anything already in the candidate list.
- If the question truly names nothing extra, use an empty list.

RULES FOR canonical titles
- Use the real Wikipedia page name, with normal spaces. Never use underscores.
- Keep the real page name even when the Urdu span is a different word-form.
- A plural span maps to the singular page.

RULES FOR steps
- Output exactly {n_steps} step objects, step_id 1 to {n_steps}, in order.
- type = "retrieve" when the step looks up an external fact. This INCLUDES a lookup about
  the answer of an earlier step (for example "When did #1 develop?" or an age/date lookup).
- type = "reason" when the step only compares, counts, computes, or decides using earlier answers.
- atype = the kind of answer THIS step returns: {ANSWER_TYPES}.

RULES FOR entity_ref
- entity_ref is the entity the step asks ABOUT. It is NEVER the answer the step returns.
  Example: for "Is there a god of #1 in Greek mythology?" the answer may be Hephaestus,
  but the step asks about Greek mythology / about #1 -- so Hephaestus is WRONG here.
- "reason" step -> entity_ref must be null.
- "retrieve" step that names an entity directly -> entity_ref is that entity's canonical title.
- "retrieve" step written about "#N" -> entity_ref is the canonical title of the entity
  that step N's answer resolved to. Follow the chain. This entity does not have to appear
  in the question. Use a candidate page title when one fits.
- If a step names an entity directly AND also mentions "#N", use the directly named entity.
- Never use a "List of ..." page as an entity or as an entity_ref.

Output the JSON object only."""


# --------------------------------------------------------------------------
# PARSING AND MECHANICAL ASSEMBLY
# --------------------------------------------------------------------------

def parse_llm_json(text):
    t = text.strip()
    t = re.sub(r"^```(json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def assemble(row, parsed, finish_reason):
    """Turn one model reply into a Schema C draft plus review flags.
    Nothing is rejected here -- problems become flags for the human reviewer."""
    flags = []
    qid = row["qid"]
    question_ur = row["question_ur"]
    ur_norm = norm(question_ur)
    steps_en = row["steps_en"]
    n_steps = len(steps_en)

    if finish_reason != "stop":
        flags.append(f"finish_reason:{finish_reason}")

    # ---- title decisions: completeness is enforced ----
    supplied = row["candidate_titles"]
    supplied_norm = [norm(t) for t in supplied]
    decisions = parsed.get("title_decisions") or []
    seen_norm = [norm(d.get("title", "")) for d in decisions]

    for t, tn in zip(supplied, supplied_norm):
        c = seen_norm.count(tn)
        if c == 0:
            flags.append(f"title_missing:{t}")
        elif c > 1:
            flags.append(f"title_duplicate:{t}")
    for d, dn in zip(decisions, seen_norm):
        if dn not in supplied_norm:
            flags.append(f"title_unknown:{d.get('title')}")

    # Authoritative spelling map: normalised form -> exact candidate title.
    # The model's own spelling is NEVER stored when an authoritative one exists.
    canon_map = {}
    for t in supplied:
        canon_map.setdefault(norm(t), t)

    # ---- entities: built mechanically, never by the model ----
    entities = []
    for d, dn in zip(decisions, seen_norm):
        if d.get("named_in_question") is not True:
            continue
        if dn not in supplied_norm:
            continue
        span = (d.get("urdu_span") or "").strip()
        title = canon_map[dn]                      # authoritative spelling
        if str(d.get("title") or "").strip() != title:
            flags.append(f"title_respelled:{d.get('title')}->{title}")
        if not span:
            flags.append(f"empty_span:{title}")
        elif norm(span) not in ur_norm:
            flags.append(f"span_not_in_question:{title}")
        if title.lower().startswith("list of"):
            flags.append(f"list_page:{title}")
        entities.append({"canonical_title": title, "urdu_span": span})

    # Titles the model explicitly said are NOT named in the question.
    # Used later to catch it contradicting itself.
    not_named_norm = {
        dn for d, dn in zip(decisions, seen_norm)
        if d.get("named_in_question") is not True and dn in supplied_norm
    }

    extras = parsed.get("extra_entities") or []
    for e in extras:
        title = str(e.get("canonical_title") or "").replace("_", " ").strip()
        span = (e.get("urdu_span") or "").strip()
        if not title:
            continue
        already = {norm(x["canonical_title"]) for x in entities}
        if norm(title) in already:
            # Never store the same entity twice. Skip it, but record it.
            flags.append(f"extra_duplicates_entity:{title}")
            continue
        if norm(title) in supplied_norm:
            flags.append(f"extra_duplicates_candidate:{title}")
        if title.lower().startswith("list of"):
            flags.append(f"list_page:{title}")
        if not span:
            flags.append(f"empty_span:{title}")
        elif norm(span) not in ur_norm:
            flags.append(f"span_not_in_question:{title}")
        entities.append({"canonical_title": title, "urdu_span": span})
    if extras:
        # Model-proposed entities are never trusted. Always human-checked.
        flags.append(f"extra_entities:{len(extras)}")

    if not entities:
        flags.append("no_entities")

    # duplicate entities
    ent_norms = [norm(e["canonical_title"]) for e in entities]
    for n_ in set(ent_norms):
        if ent_norms.count(n_) > 1:
            flags.append(f"duplicate_entity:{n_}")

    # Two different titles on the SAME Urdu span means one of them is wrong.
    # Which one is a human decision, but it must never go unnoticed.
    by_span = {}
    for e in entities:
        by_span.setdefault(norm(e["urdu_span"]), []).append(e["canonical_title"])
    for span_n, titles in by_span.items():
        uniq = sorted(set(titles))
        if span_n and len(uniq) > 1:
            flags.append(f"shared_span:{span_n}:{'|'.join(uniq)}")

    # extra_entities have no authoritative source, so their own spelling is kept
    # (and they are already flagged for human verification above).
    for e in entities:
        canon_map.setdefault(norm(e["canonical_title"]), e["canonical_title"])

    title_universe = set(supplied_norm) | set(ent_norms)

    # ---- steps: text copied, model supplies type / entity_ref / atype ----
    model_steps = parsed.get("steps") or []
    by_id = {}
    for s in model_steps:
        try:
            sid = int(s.get("step_id"))
        except (TypeError, ValueError):
            flags.append("bad_step_id")
            continue
        if sid in by_id:
            flags.append(f"duplicate_step_id:{sid}")
        by_id[sid] = s

    if sorted(by_id.keys()) != list(range(1, n_steps + 1)):
        flags.append(f"step_ids_mismatch:got_{sorted(by_id.keys())}_expected_1..{n_steps}")

    steps = []
    for i, text in enumerate(steps_en, start=1):
        s = by_id.get(i, {})

        stype = str(s.get("type") or "").strip().lower()
        if stype not in STEP_TYPES:
            flags.append(f"bad_type:step{i}:{s.get('type')!r}")
            stype = stype or None

        atype = str(s.get("atype") or "").strip().upper()
        if atype not in ANSWER_TYPES:
            flags.append(f"bad_atype:step{i}:{s.get('atype')!r}")
            atype = atype or None

        ref_raw = s.get("entity_ref")
        ref = None if ref_raw is None else str(ref_raw).replace("_", " ").strip()
        if ref == "":
            ref = None

        if stype == "reason" and ref is not None:
            flags.append(f"reason_has_ref:step{i}")
            ref = None                      # AMENDMENT 2c: reason -> always null
        if stype == "retrieve":
            if ref is None:
                flags.append(f"retrieve_empty_ref:step{i}")
            elif norm(ref) in canon_map:
                canon = canon_map[norm(ref)]       # authoritative spelling
                if ref != canon:
                    flags.append(f"ref_respelled:step{i}:{ref}->{canon}")
                ref = canon
                if norm(ref) not in title_universe:
                    flags.append(f"ref_outside_universe:step{i}:{ref}")
            else:
                # No authoritative match: keep the model's text unchanged, flag it.
                flags.append(f"ref_outside_universe:step{i}:{ref}")

            # Self-contradiction: the model said this title is NOT named in the
            # question, then used it as the subject of a step that has no "#N".
            # (A "#N" step may legitimately point outside the question.)
            if ref is not None and "#" not in text and norm(ref) in not_named_norm:
                flags.append(f"ref_marked_not_named:step{i}:{ref}")

        steps.append({
            "step_id": i,
            "text": text,                   # verbatim English, AMENDMENT 2b
            "type": stype,
            "entity_ref": ref,
            "atype": atype,
        })

    draft = {
        "qid": qid,
        "question_ur": question_ur,
        "entities": entities,
        "steps": steps,
        "answer": row["answer"],            # already mapped yes/no, AMENDMENT 2a
    }
    return draft, sorted(set(flags))


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and print the first one; load no model, write nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N rows; output goes to a separate test file")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing output file")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor parallel size = how many GPUs to split the model across. "
                         "Use 1 on a 48GB card (L20/A6000), 2 on 24GB cards (4090).")
    args = ap.parse_args()

    if not CANDS_FILE.exists():
        die(f"missing input: {CANDS_FILE}")
    rows = load_jsonl(CANDS_FILE)
    if len(rows) != EXPECTED_N:
        die(f"{CANDS_FILE} has {len(rows)} rows, expected {EXPECTED_N}")

    if args.limit:
        rows = rows[:args.limit]
        out_file = EF / f"plan_a_drafts_test{args.limit}.jsonl"
        summary_file = EF / f"plan_a_drafts_test{args.limit}_summary.txt"
    else:
        out_file = OUT_FULL
        summary_file = SUMMARY_FULL

    if out_file.exists() and not args.dry_run and not args.force:
        die(f"output already exists: {out_file}\nRe-run with --force to overwrite.")

    prompts_text = [build_prompt(r) for r in rows]

    if args.dry_run:
        print("DRY RUN -- no model loaded, nothing written.\n")
        print(f"rows prepared : {len(rows)}")
        print(f"would write   : {out_file}")
        print("=" * 70)
        print("PROMPT FOR ROW 1")
        print("=" * 70)
        print(prompts_text[0])
        print("=" * 70)
        lens = [len(p) for p in prompts_text]
        print(f"prompt length in characters: min {min(lens)} / max {max(lens)}")
        return

    # heavy imports only when we really run
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.85, tensor_parallel_size=args.tp)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                        stop=["<|im_end|>"])

    chat_prompts = []
    for p in prompts_text:
        messages = [{"role": "user", "content": p}]
        chat_prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING))

    outputs = llm.generate(chat_prompts, sp)

    results = []
    n_parse_fail = 0
    n_clean = 0
    flag_counts = {}
    truncated = []

    for row, out in zip(rows, outputs):
        gen = out.outputs[0].text.strip()
        finish_reason = getattr(out.outputs[0], "finish_reason", "unknown")
        parsed = parse_llm_json(gen)

        if parsed is None:
            n_parse_fail += 1
            # A broken row is kept for the SAME qid. Never skipped, never swapped.
            results.append({
                "qid": row["qid"],
                "parse_failed": True,
                "finish_reason": finish_reason,
                "raw": gen[:3000],
                "flags": ["parse_failure"],
            })
            flag_counts["parse_failure"] = flag_counts.get("parse_failure", 0) + 1
            if finish_reason != "stop":
                truncated.append(row["qid"])
            continue

        draft, flags = assemble(row, parsed, finish_reason)
        if not flags:
            n_clean += 1
        if finish_reason != "stop":
            truncated.append(row["qid"])
        for f in flags:
            key = f.split(":")[0]
            flag_counts[key] = flag_counts.get(key, 0) + 1

        results.append({
            "qid": row["qid"],
            "parse_failed": False,
            "finish_reason": finish_reason,
            "draft": draft,
            "flags": flags,
            # Raw model output, kept so the human reviewer can see exactly what
            # the model decided -- not only the assembled result.
            "model_decisions": {
                "title_decisions": parsed.get("title_decisions") or [],
                "extra_entities": parsed.get("extra_entities") or [],
                "steps": parsed.get("steps") or [],
            },
            "candidate_titles": row["candidate_titles"],
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ent = sum(len(r["draft"]["entities"]) for r in results if not r["parse_failed"])
    n_rows_ok = sum(1 for r in results if not r["parse_failed"])
    ent_per_row = (n_ent / n_rows_ok) if n_rows_ok else 0
    n_true = n_ent
    n_titles = sum(len(r["candidate_titles"]) for r in rows)

    lines = [
        "PLAN A -- Schema C DRAFT GENERATION",
        "=" * 60,
        f"rows processed          : {len(results)}",
        f"JSON parse failures     : {n_parse_fail}",
        f"rows with zero flags    : {n_clean}",
        f"truncated (finish!=stop): {len(truncated)} {truncated[:10]}",
        f"candidate titles shown  : {n_titles}",
        f"entities kept           : {n_ent}   ({ent_per_row:.2f} per parsed row)",
        f"flag counts             : {flag_counts}",
        "",
        "NOTE: every row still goes to human review (Plan A'). Zero flags means",
        "structurally clean, NOT correct.",
    ]
    summary = "\n".join(lines)
    summary_file.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\ndrafts  : {out_file}\nsummary : {summary_file}")


if __name__ == "__main__":
    main()
