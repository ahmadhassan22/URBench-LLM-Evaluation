#!/usr/bin/env python3
"""Phase-R v3: leak-free atomic fact planning and offline oracle audit.

This program intentionally does NOT load the Wikipedia index and does NOT run
R1-R4 retrieval.  It repairs the failed v2 planner before any expensive search.

Stages:
  self-test  Standard-library unit checks only.
  generate   Qwen3-14B sees only {urbench_qid, question_ur}.  It creates atomic
             external-fact goals, reviews them using the same legal input, and
             creates independent direct-view and translate-first queries.
  audit      CPU-only.  Official decomposition/evidence enter only here, after
             generation, to prepare count diagnostics and a fixed 12-item
             oracle-review packet.  Audit data never flows back into prompts.

Outputs use data/strategyqa_official/phase_r_v3, leaving v2 untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


BASE = Path("/mnt/home/user41/URBench")
OFFICIAL_DIR = BASE / "data" / "strategyqa_official"
DEV50_PATH = OFFICIAL_DIR / "dev50_seed42.jsonl"
DEV50_QIDS_PATH = OFFICIAL_DIR / "dev50_seed42_qids.txt"
EVAL458_PATH = BASE / "data" / "sdfr_splits" / "strategyqa_eval.jsonl"
PARAGRAPHS_PATH = OFFICIAL_DIR / "strategyqa_train_paragraphs.json"
MODEL_PATH = Path("/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B")

OUT_DIR = OFFICIAL_DIR / "phase_r_v3"
CACHE_DIR = OUT_DIR / "cache" / "generation"

SCHEMA_VERSION = "phase-r-fact-plan-v3.0"
MAX_FACTS = 4
MAX_ENTITY_CANDIDATES = 3
MARKERS = {"operation", "no_evidence"}


class PhaseRGenerationError(RuntimeError):
    pass


def norm_qid(value: Any) -> str:
    return str(value).strip()


def norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PhaseRGenerationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise PhaseRGenerationError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PhaseRGenerationError(f"Expected object in {path}")
    return value


def frozen_qids() -> list[str]:
    qids = [norm_qid(line) for line in DEV50_QIDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(qids) != 50 or len(set(qids)) != 50:
        raise PhaseRGenerationError(f"Frozen DEV50 must contain 50 unique qids, found {len(qids)}/{len(set(qids))}")
    return qids


def eval_qids() -> set[str]:
    qids = {norm_qid(row.get("qid")) for row in load_jsonl(EVAL458_PATH)}
    if len(qids) != 458:
        raise PhaseRGenerationError(f"Expected 458 evaluation qids, found {len(qids)}")
    return qids


def legal_inputs() -> list[dict[str, str]]:
    """Discard all gold fields before returning live-method inputs."""
    frozen = frozen_qids()
    evaluation = eval_qids()
    rows = load_jsonl(DEV50_PATH)
    if len(rows) != 50:
        raise PhaseRGenerationError(f"Expected 50 DEV50 rows, found {len(rows)}")
    by_qid: dict[str, str] = {}
    for row in rows:
        qid = norm_qid(row.get("urbench_qid"))
        question = norm_text(row.get("question_ur"))
        if not qid or not question:
            raise PhaseRGenerationError("DEV50 row missing urbench_qid or question_ur")
        if row.get("is_eval") is not False:
            raise PhaseRGenerationError(f"DEV50 row {qid} is not marked is_eval=false")
        if qid in by_qid:
            raise PhaseRGenerationError(f"Duplicate qid: {qid}")
        by_qid[qid] = question
    if set(by_qid) != set(frozen):
        raise PhaseRGenerationError("DEV50 rows differ from frozen qid file")
    if set(by_qid) & evaluation:
        raise PhaseRGenerationError("DEV50 intersects eval458")
    print("[guard] DEV50=50 unique, frozen order, zero eval458 overlap")
    print("[guard] live model inputs are urbench_qid + question_ur only")
    return [{"urbench_qid": qid, "question_ur": by_qid[qid]} for qid in frozen]


def extract_json(raw: str, expected: type) -> Any | None:
    text = raw.split("</think>")[-1].strip() if "</think>" in raw else raw.strip()
    positions = [position for token in ("{", "[") if (position := text.find(token)) >= 0]
    if not positions:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[min(positions) :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, expected) else None


def unique_texts(values: Any, maximum: int) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = norm_text(value)
        key = clean.casefold()
        if clean and key not in seen:
            output.append(clean)
            seen.add(key)
        if len(output) >= maximum:
            break
    return output


def normalize_fact_plan(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        value = value.get("facts")
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_FACTS:
        return None
    facts: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for position, raw in enumerate(value, 1):
        if not isinstance(raw, dict):
            return None
        try:
            fact_id = int(raw.get("id", position))
        except (TypeError, ValueError):
            return None
        lookup = norm_text(raw.get("lookup_question_ur"))
        entities = unique_texts(raw.get("entity_mentions_ur", []), 3)
        attribute = norm_text(raw.get("required_attribute_ur"))
        if fact_id in seen_ids or not lookup or not attribute:
            return None
        facts.append(
            {
                "id": fact_id,
                "lookup_question_ur": lookup,
                "entity_mentions_ur": entities,
                "required_attribute_ur": attribute,
            }
        )
        seen_ids.add(fact_id)
    return facts


def normalize_review(value: Any) -> tuple[list[dict[str, Any]] | None, list[str]]:
    if not isinstance(value, dict):
        return None, []
    facts = normalize_fact_plan(value)
    issues = unique_texts(value.get("issues", []), 10)
    return facts, issues


def normalize_translation(value: Any, fact_ids: set[int]) -> tuple[str, dict[int, str]] | None:
    if not isinstance(value, dict):
        return None
    question_en = norm_text(value.get("question_en"))
    raw_facts = value.get("facts")
    if not question_en or not isinstance(raw_facts, list):
        return None
    mapping: dict[int, str] = {}
    for raw in raw_facts:
        if not isinstance(raw, dict):
            continue
        try:
            fact_id = int(raw.get("id"))
        except (TypeError, ValueError):
            continue
        lookup_en = norm_text(raw.get("lookup_question_en"))
        if fact_id in fact_ids and lookup_en:
            mapping[fact_id] = lookup_en
    if set(mapping) != fact_ids:
        return None
    return question_en, mapping


def normalize_query_view(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    query = norm_text(value.get("search_query_en"))
    candidates = unique_texts(value.get("entity_candidates_en", []), MAX_ENTITY_CANDIDATES)
    if not query:
        return None
    return {"search_query_en": query, "entity_candidates_en": candidates}


SYNTHETIC_EXAMPLES = """Synthetic examples (not StrategyQA items and not evaluation data):

Example A
Question: کیا ماؤنٹ ایورسٹ برج خلیفہ سے اونچا ہے؟
Correct external facts:
{"facts":[
 {"id":1,"lookup_question_ur":"ماؤنٹ ایورسٹ کی اونچائی کتنی ہے؟","entity_mentions_ur":["ماؤنٹ ایورسٹ"],"required_attribute_ur":"اونچائی"},
 {"id":2,"lookup_question_ur":"برج خلیفہ کی اونچائی کتنی ہے؟","entity_mentions_ur":["برج خلیفہ"],"required_attribute_ur":"اونچائی"}
]}

Example B
Question: کیا جاپان اور فرانس ایک ہی براعظم میں واقع ہیں؟
Correct external facts:
{"facts":[
 {"id":1,"lookup_question_ur":"جاپان کس براعظم میں واقع ہے؟","entity_mentions_ur":["جاپان"],"required_attribute_ur":"براعظم"},
 {"id":2,"lookup_question_ur":"فرانس کس براعظم میں واقع ہے؟","entity_mentions_ur":["فرانس"],"required_attribute_ur":"براعظم"}
]}

Example C
Question: کیا پانی سطح سمندر پر پچاس ڈگری سینٹی گریڈ پر ابلتا ہے؟
Correct external facts:
{"facts":[
 {"id":1,"lookup_question_ur":"سطح سمندر پر پانی کا نقطہ ابال کیا ہے؟","entity_mentions_ur":["پانی"],"required_attribute_ur":"سطح سمندر پر نقطہ ابال"}
]}
"""

PLAN_PROMPT = """You are planning external fact retrieval for an Urdu yes/no reasoning question.

{examples}

Now process this Urdu question:
{question_ur}

List ALL and ONLY the external factual premises that an English Wikipedia reader
must look up before reasoning. Follow these rules strictly:

1. Each item must be one atomic fact lookup about one entity/attribute.
2. Never repeat the original yes/no question as a lookup.
3. Never ask Wikipedia to answer the final yes/no conclusion directly.
4. Do not include comparison, arithmetic, temporal ordering, set inclusion, or
   other operations; those will be reasoned over later.
5. Include every independent external premise needed for that later operation.
6. Return 1-4 facts. Do not answer the question and do not supply fact values.

Return ONLY this JSON schema:
{{"facts":[{{"id":1,"lookup_question_ur":"...","entity_mentions_ur":["..."],"required_attribute_ur":"..."}}]}}"""

REVIEW_PROMPT = """Review an external-fact plan using ONLY the original Urdu question and the proposed plan.

Original Urdu question:
{question_ur}

Proposed external facts:
{plan_json}

Repair the plan if it repeats the final yes/no question, combines multiple facts,
omits an external premise, uses the wrong attribute, or includes a reasoning
operation. Do not answer the question and do not add fact values. Keep 1-4 atomic
external lookups.

Return ONLY JSON:
{{"facts":[{{"id":1,"lookup_question_ur":"...","entity_mentions_ur":["..."],"required_attribute_ur":"..."}}],"issues":["brief repair description"]}}"""

TRANSLATE_PROMPT = """Translate this Urdu question and its atomic fact lookups into English.
Preserve entity identity exactly. Do not answer anything and do not add facts.

Urdu question:
{question_ur}

Atomic Urdu lookups:
{facts_json}

Return ONLY JSON:
{{"question_en":"...","facts":[{{"id":1,"lookup_question_en":"..."}}]}}"""

DIRECT_QUERY_PROMPT = """Create one concise English Wikipedia search query directly from this Urdu fact lookup.
Do not translate or reinterpret the entire original question first.

Original Urdu question: {question_ur}
Atomic Urdu lookup: {lookup_ur}
Urdu entity mentions: {entities_ur}
Required attribute: {attribute_ur}

Return ONLY JSON:
{{"search_query_en":"...","entity_candidates_en":["English entity-name candidate"]}}
Entity candidates must be names only, maximum 3. Do not answer the fact."""

TRANSLATED_QUERY_PROMPT = """Create one concise English Wikipedia search query from this translated English view.

English question: {question_en}
Atomic English lookup: {lookup_en}

Return ONLY JSON:
{{"search_query_en":"...","entity_candidates_en":["English entity-name candidate"]}}
Entity candidates must be names only, maximum 3. Do not answer the fact."""

REPAIR_PROMPT = """Convert the malformed response below into valid JSON matching the schema.
Do not add facts or answers. Output JSON only.

Schema:
{schema}

Malformed response:
{raw}"""


def llm_generate(llm: Any, sampling: Any, prompts: Sequence[str], thinking: bool) -> list[str]:
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(
        conversations,
        sampling,
        chat_template_kwargs={"enable_thinking": thinking},
        use_tqdm=True,
    )
    return [output.outputs[0].text.strip() for output in outputs]


def repair_invalid(
    llm: Any,
    sampling: Any,
    raw_values: list[str],
    parsed_values: list[Any | None],
    expected: type,
    schema: str,
) -> tuple[list[str], list[Any | None]]:
    failed = [index for index, value in enumerate(parsed_values) if value is None]
    if not failed:
        return raw_values, parsed_values
    prompts = [REPAIR_PROMPT.format(schema=schema, raw=raw_values[index][:8000]) for index in failed]
    repaired = llm_generate(llm, sampling, prompts, thinking=False)
    for index, raw in zip(failed, repaired):
        raw_values[index] = raw
        parsed_values[index] = extract_json(raw, expected)
    return raw_values, parsed_values


def cache_path(qid: str) -> Path:
    return CACHE_DIR / f"{qid}.json"


def valid_cache(path: Path, qid: str, question_ur: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = read_json_object(path)
    except (OSError, json.JSONDecodeError, PhaseRGenerationError):
        return None
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("urbench_qid") != qid
        or record.get("input_question_sha256") != sha256_text(question_ur)
    ):
        return None
    return record


def token_set(text: str) -> set[str]:
    clean = re.sub(r"[^\w\u0600-\u06ff]+", " ", norm_text(text).casefold())
    return {token for token in clean.split() if token}


def question_similarity(question: str, lookup: str) -> float:
    first, second = token_set(question), token_set(lookup)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def stage_generate(args: argparse.Namespace) -> None:
    inputs = legal_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [
        row
        for row in inputs
        if valid_cache(cache_path(row["urbench_qid"]), row["urbench_qid"], row["question_ur"]) is None
    ]
    print(f"[generate] valid cached={len(inputs) - len(todo)} todo={len(todo)}")

    if todo:
        if not MODEL_PATH.exists():
            raise PhaseRGenerationError(f"Missing model: {MODEL_PATH}")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(MODEL_PATH),
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=6144,
            trust_remote_code=True,
        )
        think_sampling = SamplingParams(temperature=0.0, max_tokens=2300, seed=0)
        short_sampling = SamplingParams(temperature=0.0, max_tokens=500, seed=0)

        for offset in range(0, len(todo), args.batch_size):
            batch = todo[offset : offset + args.batch_size]
            print(f"[generate] batch {offset + 1}-{offset + len(batch)} / {len(todo)}")

            plan_raw = llm_generate(
                llm,
                think_sampling,
                [PLAN_PROMPT.format(examples=SYNTHETIC_EXAMPLES, question_ur=row["question_ur"]) for row in batch],
                thinking=True,
            )
            plan_values: list[Any | None] = [extract_json(raw, dict) for raw in plan_raw]
            plan_values = [value if normalize_fact_plan(value) else None for value in plan_values]
            plan_raw, plan_values = repair_invalid(
                llm,
                short_sampling,
                plan_raw,
                plan_values,
                dict,
                '{"facts":[{"id":1,"lookup_question_ur":"...","entity_mentions_ur":["..."],"required_attribute_ur":"..."}]}',
            )
            original_plans = [normalize_fact_plan(value) for value in plan_values]

            review_prompts: list[str] = []
            review_owners: list[int] = []
            for index, (row, facts) in enumerate(zip(batch, original_plans)):
                if facts:
                    review_prompts.append(
                        REVIEW_PROMPT.format(
                            question_ur=row["question_ur"],
                            plan_json=json.dumps({"facts": facts}, ensure_ascii=False),
                        )
                    )
                    review_owners.append(index)
            review_raw = llm_generate(llm, think_sampling, review_prompts, thinking=True) if review_prompts else []
            review_values: list[Any | None] = [extract_json(raw, dict) for raw in review_raw]
            review_values = [value if normalize_review(value)[0] else None for value in review_values]
            review_raw, review_values = repair_invalid(
                llm,
                short_sampling,
                review_raw,
                review_values,
                dict,
                '{"facts":[{"id":1,"lookup_question_ur":"...","entity_mentions_ur":["..."],"required_attribute_ur":"..."}],"issues":["..."]}',
            )
            reviewed_by_owner: dict[int, tuple[list[dict[str, Any]] | None, list[str]]] = {}
            for owner, value in zip(review_owners, review_values):
                reviewed_by_owner[owner] = normalize_review(value)

            final_plans: list[list[dict[str, Any]] | None] = []
            review_issues: list[list[str]] = []
            review_changed: list[bool] = []
            for index, original in enumerate(original_plans):
                reviewed, issues = reviewed_by_owner.get(index, (None, []))
                final = reviewed or original
                final_plans.append(final)
                review_issues.append(issues)
                review_changed.append(bool(original and final and canonical_hash(original) != canonical_hash(final)))

            translation_prompts: list[str] = []
            translation_owners: list[int] = []
            for index, (row, facts) in enumerate(zip(batch, final_plans)):
                if facts:
                    translation_prompts.append(
                        TRANSLATE_PROMPT.format(
                            question_ur=row["question_ur"],
                            facts_json=json.dumps({"facts": facts}, ensure_ascii=False),
                        )
                    )
                    translation_owners.append(index)
            translation_raw = llm_generate(llm, short_sampling, translation_prompts, thinking=False) if translation_prompts else []
            translation_values: list[Any | None] = [extract_json(raw, dict) for raw in translation_raw]
            validated_translation_values: list[Any | None] = []
            for owner, value in zip(translation_owners, translation_values):
                facts_for_owner = final_plans[owner] or []
                fact_ids_for_owner = {fact["id"] for fact in facts_for_owner}
                validated_translation_values.append(
                    value if normalize_translation(value, fact_ids_for_owner) else None
                )
            translation_values = validated_translation_values
            translation_raw, translation_values = repair_invalid(
                llm,
                short_sampling,
                translation_raw,
                translation_values,
                dict,
                '{"question_en":"...","facts":[{"id":1,"lookup_question_en":"..."}]}',
            )
            translations: dict[int, tuple[str, dict[int, str]] | None] = {}
            for owner, facts, value in zip(
                translation_owners,
                [final_plans[index] for index in translation_owners],
                translation_values,
            ):
                fact_ids = {fact["id"] for fact in facts or []}
                translations[owner] = normalize_translation(value, fact_ids)

            direct_prompts: list[str] = []
            translated_prompts: list[str] = []
            query_owners: list[tuple[int, int]] = []
            for row_index, (row, facts) in enumerate(zip(batch, final_plans)):
                translation = translations.get(row_index)
                if not facts or not translation:
                    continue
                question_en, lookup_map = translation
                for fact_index, fact in enumerate(facts):
                    direct_prompts.append(
                        DIRECT_QUERY_PROMPT.format(
                            question_ur=row["question_ur"],
                            lookup_ur=fact["lookup_question_ur"],
                            entities_ur=json.dumps(fact["entity_mentions_ur"], ensure_ascii=False),
                            attribute_ur=fact["required_attribute_ur"],
                        )
                    )
                    translated_prompts.append(
                        TRANSLATED_QUERY_PROMPT.format(
                            question_en=question_en,
                            lookup_en=lookup_map[fact["id"]],
                        )
                    )
                    query_owners.append((row_index, fact_index))

            direct_raw = llm_generate(llm, short_sampling, direct_prompts, thinking=False) if direct_prompts else []
            translated_query_raw = llm_generate(llm, short_sampling, translated_prompts, thinking=False) if translated_prompts else []
            direct_values: list[Any | None] = [extract_json(raw, dict) for raw in direct_raw]
            translated_query_values: list[Any | None] = [extract_json(raw, dict) for raw in translated_query_raw]
            direct_values = [value if normalize_query_view(value) else None for value in direct_values]
            translated_query_values = [
                value if normalize_query_view(value) else None for value in translated_query_values
            ]
            direct_raw, direct_values = repair_invalid(
                llm,
                short_sampling,
                direct_raw,
                direct_values,
                dict,
                '{"search_query_en":"...","entity_candidates_en":["..."]}',
            )
            translated_query_raw, translated_query_values = repair_invalid(
                llm,
                short_sampling,
                translated_query_raw,
                translated_query_values,
                dict,
                '{"search_query_en":"...","entity_candidates_en":["..."]}',
            )
            query_views: dict[tuple[int, int], tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
            for owner, direct_value, translated_value in zip(query_owners, direct_values, translated_query_values):
                query_views[owner] = (normalize_query_view(direct_value), normalize_query_view(translated_value))

            for row_index, row in enumerate(batch):
                original = original_plans[row_index]
                facts = final_plans[row_index]
                translation = translations.get(row_index)
                enriched: list[dict[str, Any]] = []
                if facts and translation:
                    question_en, lookup_map = translation
                    for fact_index, fact in enumerate(facts):
                        direct_view, translated_view = query_views.get((row_index, fact_index), (None, None))
                        enriched.append(
                            {
                                **fact,
                                "lookup_question_en": lookup_map.get(fact["id"], ""),
                                "direct_view": direct_view,
                                "translated_view": translated_view,
                                "whole_question_similarity": round(
                                    question_similarity(row["question_ur"], fact["lookup_question_ur"]), 4
                                ),
                            }
                        )
                else:
                    question_en = ""

                record = {
                    "schema_version": SCHEMA_VERSION,
                    "urbench_qid": row["urbench_qid"],
                    "input_question_sha256": sha256_text(row["question_ur"]),
                    "live_source_fields_used": ["urbench_qid", "question_ur"],
                    "question_ur": row["question_ur"],
                    "question_en_translation": question_en,
                    "parse_ok": bool(facts),
                    "original_plan": original or [],
                    "review_changed_plan": review_changed[row_index],
                    "review_issues": review_issues[row_index],
                    "facts": enriched,
                    "created_unix": int(time.time()),
                }
                atomic_write_json(cache_path(row["urbench_qid"]), record)

    records: list[dict[str, Any]] = []
    for row in inputs:
        cached = valid_cache(cache_path(row["urbench_qid"]), row["urbench_qid"], row["question_ur"])
        if cached is None:
            raise PhaseRGenerationError(f"Missing or invalid cache for {row['urbench_qid']}")
        records.append(cached)

    parse_failures = sum(not record.get("parse_ok") for record in records)
    facts = [fact for record in records for fact in record.get("facts", [])]
    direct_failures = sum(not fact.get("direct_view") for fact in facts)
    translated_failures = sum(not fact.get("translated_view") for fact in facts)
    high_similarity = sum(float(fact.get("whole_question_similarity", 0.0)) >= 0.80 for fact in facts)
    fact_distribution = dict(sorted(Counter(len(record.get("facts", [])) for record in records).items()))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dev50_items": len(records),
        "parse_failures": parse_failures,
        "atomic_fact_goals": len(facts),
        "fact_count_distribution": fact_distribution,
        "review_changed_items": sum(bool(record.get("review_changed_plan")) for record in records),
        "direct_query_failures": direct_failures,
        "translated_query_failures": translated_failures,
        "whole_question_like_goals_similarity_ge_0_80": high_similarity,
        "ready_for_manual_oracle_audit": (
            parse_failures == 0 and direct_failures == 0 and translated_failures == 0
        ),
        "ready_for_retrieval": False,
        "note": "Manual oracle audit is mandatory; this program never auto-approves retrieval.",
    }
    atomic_write_json(OUT_DIR / "generation_summary.json", summary)
    atomic_write_jsonl(OUT_DIR / "generation_records.jsonl", records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready_for_manual_oracle_audit"]:
        raise PhaseRGenerationError("Mechanical generation checks failed")


def recursively_collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(recursively_collect_strings(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(recursively_collect_strings(item))
        return result
    return []


def official_retrieval_step_indices(row: dict[str, Any], paragraph_ids: set[str]) -> list[int]:
    decomposition = row.get("official_decomposition")
    evidence = row.get("official_evidence")
    if not isinstance(decomposition, list) or not isinstance(evidence, list):
        raise PhaseRGenerationError(f"Malformed official annotations for {row.get('urbench_qid')}")
    flags = [False] * len(decomposition)
    annotations = evidence
    for annotation in annotations:
        if not isinstance(annotation, list) or len(annotation) != len(decomposition):
            raise PhaseRGenerationError(f"Evidence/decomposition mismatch for {row.get('urbench_qid')}")
        for index, step_evidence in enumerate(annotation):
            strings = recursively_collect_strings(step_evidence)
            if any(value.casefold() not in MARKERS and value in paragraph_ids for value in strings):
                flags[index] = True
    return [index + 1 for index, flag in enumerate(flags) if flag]


def stage_audit(_: argparse.Namespace) -> None:
    frozen = frozen_qids()
    if set(frozen) & eval_qids():
        raise PhaseRGenerationError("DEV50 intersects eval458")
    generated = {record["urbench_qid"]: record for record in load_jsonl(OUT_DIR / "generation_records.jsonl")}
    dev = {norm_qid(row["urbench_qid"]): row for row in load_jsonl(DEV50_PATH)}
    if set(generated) != set(frozen) or set(dev) != set(frozen):
        raise PhaseRGenerationError("Audit inputs do not contain exactly frozen DEV50")
    with PARAGRAPHS_PATH.open(encoding="utf-8") as handle:
        paragraphs = json.load(handle)
    if not isinstance(paragraphs, dict) or len(paragraphs) != 9251:
        raise PhaseRGenerationError("Official paragraph dictionary is not the verified 9251-entry file")
    paragraph_ids = set(paragraphs)

    count_rows: list[dict[str, Any]] = []
    deficits = 0
    exact = 0
    surpluses = 0
    for qid in frozen:
        official_r = official_retrieval_step_indices(dev[qid], paragraph_ids)
        generated_count = len(generated[qid].get("facts", []))
        difference = generated_count - len(official_r)
        deficits += int(difference < 0)
        exact += int(difference == 0)
        surpluses += int(difference > 0)
        count_rows.append(
            {
                "urbench_qid": qid,
                "generated_fact_count": generated_count,
                "official_evidence_bearing_step_count": len(official_r),
                "official_evidence_bearing_step_indices": official_r,
                "count_difference": difference,
            }
        )
    atomic_write_jsonl(OUT_DIR / "generation_count_audit.jsonl", count_rows)

    ordered_records = [generated[qid] for qid in sorted(frozen)]
    sample = random.Random(42).sample(ordered_records, 12)
    lines: list[str] = []
    packet: list[dict[str, Any]] = []
    for record in sample:
        qid = record["urbench_qid"]
        row = dev[qid]
        official_r = official_retrieval_step_indices(row, paragraph_ids)
        item = {
            "urbench_qid": qid,
            "question_ur": record["question_ur"],
            "question_en_translation": record["question_en_translation"],
            "generated_facts": record["facts"],
            "review_issues": record.get("review_issues", []),
            "official_decomposition_oracle_only": row["official_decomposition"],
            "official_evidence_bearing_step_indices": official_r,
            "manual_all_external_premises_covered": None,
            "manual_atomic_and_correct_attributes": None,
            "manual_severe_entity_corruption": None,
            "manual_notes": "",
        }
        packet.append(item)
        lines.extend(["=" * 78, f"QID: {qid}", f"UR: {record['question_ur']}", f"EN: {record['question_en_translation']}", "GENERATED ATOMIC FACTS:"])
        for fact in record["facts"]:
            lines.append(f"  {fact['id']}. {fact['lookup_question_ur']}")
            lines.append(f"     entity_ur: {fact['entity_mentions_ur']}")
            lines.append(f"     attribute: {fact['required_attribute_ur']}")
            lines.append(f"     direct: {fact['direct_view']}")
            lines.append(f"     translated: {fact['translated_view']}")
        lines.append(f"REVIEW ISSUES: {record.get('review_issues', [])}")
        lines.append("OFFICIAL DECOMPOSITION — ORACLE AUDIT ONLY:")
        for index, step in enumerate(row["official_decomposition"], 1):
            marker = "R" if index in official_r else "O/no-evidence"
            lines.append(f"  {index}. [{marker}] {step}")
        lines.append("")
    atomic_write_text(OUT_DIR / "generation_oracle_sample_seed42.txt", "\n".join(lines) + "\n")
    atomic_write_jsonl(OUT_DIR / "generation_oracle_sample_seed42.jsonl", packet)

    summary = {
        "dev50_items": 50,
        "items_generated_fewer_facts_than_official_evidence_steps": deficits,
        "items_with_equal_fact_counts": exact,
        "items_generated_more_facts_than_official_evidence_steps": surpluses,
        "fixed_manual_sample_size": 12,
        "ready_for_retrieval": False,
        "note": "Count agreement is not semantic coverage. Complete the fixed 12-item manual oracle audit before any retrieval run.",
    }
    atomic_write_json(OUT_DIR / "generation_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[audit] review packet: {OUT_DIR / 'generation_oracle_sample_seed42.txt'}")


def self_test() -> None:
    plan = normalize_fact_plan(
        {
            "facts": [
                {
                    "id": 1,
                    "lookup_question_ur": "ماؤنٹ ایورسٹ کی اونچائی کتنی ہے؟",
                    "entity_mentions_ur": ["ماؤنٹ ایورسٹ"],
                    "required_attribute_ur": "اونچائی",
                },
                {
                    "id": 2,
                    "lookup_question_ur": "برج خلیفہ کی اونچائی کتنی ہے؟",
                    "entity_mentions_ur": ["برج خلیفہ"],
                    "required_attribute_ur": "اونچائی",
                },
            ]
        }
    )
    assert plan and len(plan) == 2
    assert normalize_query_view({"search_query_en": "Mount Everest height", "entity_candidates_en": ["Mount Everest"]})
    translated = normalize_translation(
        {"question_en": "Is A taller than B?", "facts": [{"id": 1, "lookup_question_en": "Height of A"}, {"id": 2, "lookup_question_en": "Height of B"}]},
        {1, 2},
    )
    assert translated and set(translated[1]) == {1, 2}
    assert question_similarity("کیا الف ب سے اونچا ہے؟", "الف کی اونچائی کتنی ہے؟") < 0.80
    print("SELF-TEST PASS: fact-plan schema, translation alignment, query schema, whole-question similarity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["self-test", "generate", "audit"])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "generate":
        stage_generate(args)
    elif args.stage == "audit":
        stage_audit(args)


if __name__ == "__main__":
    try:
        main()
    except PhaseRGenerationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
