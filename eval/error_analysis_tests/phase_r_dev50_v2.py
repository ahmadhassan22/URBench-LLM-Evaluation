#!/usr/bin/env python3
"""Leak-free Phase-R DEV50 diagnostic for dual-view entity canonicalization.

Stages are deliberately separate:

  generate  - Qwen3-14B only.  Reads only qid + Urdu question, builds one
              shared Urdu step plan and two independently generated English
              retrieval views, then exits (releasing GPU memory).
  retrieve  - Full FAISS index + multilingual reranker only.  Uses the frozen
              generation cache, evaluates R1-R4 with a fixed 40-passage
              evidence budget, and records title-grounding overhead separately.
  score     - CPU-only offline scorer.  Official annotations enter here only.
              Produces strict item-level official evidence page-title recall,
              a manual M1 label file, and an M3 review packet.

No final-answer generation, no C0-C5, and no eval458 retrieval occur here.
All per-qid caches are written atomically.  The output directory is phase_r_v2,
so the cancelled/invalid phase_r run is preserved untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


BASE = Path("/mnt/home/user41/URBench")
OFFICIAL_DIR = BASE / "data" / "strategyqa_official"
DEV50_PATH = OFFICIAL_DIR / "dev50_seed42.jsonl"
DEV50_QIDS_PATH = OFFICIAL_DIR / "dev50_seed42_qids.txt"
EVAL458_PATH = BASE / "data" / "sdfr_splits" / "strategyqa_eval.jsonl"
PARAGRAPHS_PATH = OFFICIAL_DIR / "strategyqa_train_paragraphs.json"

OUT_DIR = OFFICIAL_DIR / "phase_r_v2"
GEN_CACHE_DIR = OUT_DIR / "cache" / "generation"
RET_CACHE_DIR = OUT_DIR / "cache" / "retrieval"

MODEL_PATH = Path("/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B")
RERANKER_PATH = Path("/mnt/home/user41/downloaded_models/BAAI/bge-reranker-v2-m3")

GEN_SCHEMA_VERSION = "phase-r-gen-v2.1"
RET_SCHEMA_VERSION = "phase-r-ret-v2.1"
EVIDENCE_BUDGET = 40
FINAL_TOP_K = 3
GROUND_TOP_K = 10
MAX_ENTITY_CANDIDATES_PER_VIEW = 3
DEFAULT_GROUND_MIN = 0.60
MARKERS = {"operation", "no_evidence"}


class PhaseRError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_hash(value: Any) -> str:
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
                raise PhaseRError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise PhaseRError(f"Expected object at {path}:{line_number}")
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
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PhaseRError(f"Expected JSON object in {path}")
    return value


def norm_qid(value: Any) -> str:
    return str(value).strip()


def norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def norm_title(value: Any) -> str:
    return norm_text(value).replace("_", " ").casefold()


def frozen_qids() -> list[str]:
    qids = [norm_qid(line) for line in DEV50_QIDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(qids) != 50 or len(set(qids)) != 50:
        raise PhaseRError(f"Frozen DEV50 qid file must contain 50 unique qids, found {len(qids)}/{len(set(qids))}")
    return qids


def eval458_qids() -> set[str]:
    qids = {norm_qid(row.get("qid")) for row in load_jsonl(EVAL458_PATH)}
    if len(qids) != 458:
        raise PhaseRError(f"Expected 458 eval qids, found {len(qids)}")
    return qids


def legal_generation_inputs() -> list[dict[str, str]]:
    """Return only the two live-method fields; discard every gold field."""
    frozen = frozen_qids()
    eval_qids = eval458_qids()
    rows = load_jsonl(DEV50_PATH)
    if len(rows) != 50:
        raise PhaseRError(f"Expected 50 DEV50 rows, found {len(rows)}")

    by_qid: dict[str, str] = {}
    for row in rows:
        qid = norm_qid(row.get("urbench_qid"))
        question_ur = norm_text(row.get("question_ur"))
        if not qid or not question_ur:
            raise PhaseRError("DEV50 row has missing urbench_qid or question_ur")
        if row.get("is_eval") is not False:
            raise PhaseRError(f"DEV50 row {qid} is not explicitly marked is_eval=false")
        if qid in by_qid:
            raise PhaseRError(f"Duplicate DEV50 qid: {qid}")
        by_qid[qid] = question_ur

    if set(by_qid) != set(frozen):
        raise PhaseRError("DEV50 JSONL qids differ from frozen qid file")
    if set(by_qid) & eval_qids:
        raise PhaseRError("DEV50 intersects eval458")

    # This is the only object passed into generation code.
    legal = [{"urbench_qid": qid, "question_ur": by_qid[qid]} for qid in frozen]
    print("[guard] DEV50=50 unique; exact frozen order; disjoint from eval458")
    print("[guard] live generation fields: urbench_qid, question_ur only")
    return legal


def retrieval_guard_qids() -> list[str]:
    qids = frozen_qids()
    if set(qids) & eval458_qids():
        raise PhaseRError("DEV50 intersects eval458")
    print("[guard] retrieval qids=50 frozen DEV50; disjoint from eval458")
    return qids


def extract_json_value(raw: str, expected: type) -> Any | None:
    text = raw.split("</think>")[-1].strip() if "</think>" in raw else raw.strip()
    starts = [position for token in ("{", "[") if (position := text.find(token)) >= 0]
    if not starts:
        return None
    start = min(starts)
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, expected) else None


def normalize_steps(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        value = value.get("steps")
    if not isinstance(value, list) or not value:
        return None
    normalized: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for position, raw in enumerate(value, 1):
        if not isinstance(raw, dict):
            return None
        try:
            step_id = int(raw.get("id", position))
        except (TypeError, ValueError):
            return None
        step_type = str(raw.get("type", "")).strip().upper()
        lookup_ur = norm_text(raw.get("lookup_question_ur"))
        if step_id in seen_ids or step_type not in {"RETRIEVE", "REASON"} or not lookup_ur:
            return None
        depends_raw = raw.get("depends_on", [])
        if not isinstance(depends_raw, list):
            return None
        depends: list[int] = []
        for dependency in depends_raw:
            try:
                depends.append(int(dependency))
            except (TypeError, ValueError):
                return None
        mention = norm_text(raw.get("urdu_entity_mention")) or None
        if step_type == "REASON":
            mention = None
        normalized.append(
            {
                "id": step_id,
                "type": step_type,
                "lookup_question_ur": lookup_ur,
                "depends_on": depends,
                "urdu_entity_mention": mention,
            }
        )
        seen_ids.add(step_id)
    return normalized


def normalize_query_view(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    query = norm_text(value.get("search_query_en"))
    candidates_raw = value.get("entity_candidates_en", [])
    if not query or not isinstance(candidates_raw, list):
        return None
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates_raw:
        item = norm_text(candidate)
        key = item.casefold()
        if item and key not in seen:
            candidates.append(item)
            seen.add(key)
        if len(candidates) >= MAX_ENTITY_CANDIDATES_PER_VIEW:
            break
    return {"search_query_en": query, "entity_candidates_en": candidates}


DECOMPOSE_PROMPT = """You are creating a retrieval plan from an Urdu question.

Use ONLY this Urdu question:
{question_ur}

Break it into ordered steps. A RETRIEVE step asks for an external factual premise.
A REASON step performs comparison, arithmetic, temporal ordering, set inclusion,
or another operation over earlier results.

Return ONLY one JSON object with this schema:
{{"steps":[{{"id":1,"type":"RETRIEVE","lookup_question_ur":"...",
"depends_on":[],"urdu_entity_mention":"exact Urdu entity phrase or null"}}]}}

Keep lookup_question_ur in Urdu. Do not answer the question. Do not provide facts."""

TRANSLATE_PLAN_PROMPT = """Translate the supplied Urdu question and its retrieval goals to English.
Preserve named-entity identity; do not add facts and do not answer anything.

Urdu question:
{question_ur}

Urdu retrieval goals:
{goals_json}

Return ONLY JSON:
{{"question_en":"...","goals":[{{"id":1,"lookup_question_en":"..."}}]}}"""

DIRECT_VIEW_PROMPT = """Create one English Wikipedia retrieval query directly from the Urdu view below.
Do not translate or reinterpret the whole question first. Map the Urdu entity and
lookup goal directly into the best concise English search query.

Original Urdu question: {question_ur}
Urdu lookup goal: {lookup_question_ur}
Urdu entity mention: {entity_ur}

Return ONLY JSON:
{{"search_query_en":"...","entity_candidates_en":["English canonical-name candidate"]}}
The candidates must be entity names only (maximum 3), not facts or descriptions.
Do not answer the question."""

TRANSLATED_VIEW_PROMPT = """Create one concise English Wikipedia retrieval query from this translated English view.

English question: {question_en}
English lookup goal: {lookup_question_en}

Return ONLY JSON:
{{"search_query_en":"...","entity_candidates_en":["English canonical-name candidate"]}}
The candidates must be entity names only (maximum 3), not facts or descriptions.
Do not answer the question."""

REPAIR_PROMPT = """Convert the following malformed response into valid JSON matching this schema exactly.
Do not add facts. Output JSON only.

Schema:
{schema}

Malformed response:
{raw}"""


def llm_generate(llm: Any, sampling: Any, prompts: Sequence[str]) -> list[str]:
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(
        conversations,
        sampling,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )
    return [output.outputs[0].text.strip() for output in outputs]


def repair_values(
    llm: Any,
    sampling: Any,
    raw_values: list[str],
    parsed_values: list[Any | None],
    expected: type,
    schema: str,
) -> tuple[list[Any | None], list[str]]:
    failed = [index for index, value in enumerate(parsed_values) if value is None]
    if not failed:
        return parsed_values, raw_values
    repair_prompts = [REPAIR_PROMPT.format(schema=schema, raw=raw_values[index][:6000]) for index in failed]
    repaired_raw = llm_generate(llm, sampling, repair_prompts)
    for index, repaired in zip(failed, repaired_raw):
        parsed_values[index] = extract_json_value(repaired, expected)
        raw_values[index] = repaired
    return parsed_values, raw_values


def generation_cache_path(qid: str) -> Path:
    return GEN_CACHE_DIR / f"{qid}.json"


def retrieval_cache_path(qid: str) -> Path:
    return RET_CACHE_DIR / f"{qid}.json"


def valid_generation_cache(path: Path, qid: str, question_ur: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = read_json_object(path)
    except (OSError, json.JSONDecodeError, PhaseRError):
        return None
    if (
        record.get("schema_version") != GEN_SCHEMA_VERSION
        or record.get("urbench_qid") != qid
        or record.get("input_question_sha256") != sha256_text(question_ur)
    ):
        return None
    return record


def stage_generate(args: argparse.Namespace) -> None:
    legal_inputs = legal_generation_inputs()
    GEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    todo: list[dict[str, str]] = []
    for row in legal_inputs:
        cached = valid_generation_cache(generation_cache_path(row["urbench_qid"]), row["urbench_qid"], row["question_ur"])
        if cached is None:
            todo.append(row)
        else:
            existing[row["urbench_qid"]] = cached

    print(f"[generate] valid cached={len(existing)} todo={len(todo)}")
    if todo:
        if not MODEL_PATH.exists():
            raise PhaseRError(f"Model path missing: {MODEL_PATH}")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(MODEL_PATH),
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=4096,
            trust_remote_code=True,
        )
        sampling_long = SamplingParams(temperature=0.0, max_tokens=900, seed=0)
        sampling_short = SamplingParams(temperature=0.0, max_tokens=300, seed=0)

        for offset in range(0, len(todo), args.batch_size):
            batch = todo[offset : offset + args.batch_size]
            print(f"[generate] batch {offset + 1}-{offset + len(batch)} / {len(todo)}")

            decomp_raw = llm_generate(
                llm,
                sampling_long,
                [DECOMPOSE_PROMPT.format(question_ur=row["question_ur"]) for row in batch],
            )
            decomp_parsed: list[Any | None] = [extract_json_value(raw, dict) for raw in decomp_raw]
            decomp_parsed, decomp_raw = repair_values(
                llm,
                sampling_long,
                decomp_raw,
                decomp_parsed,
                dict,
                '{"steps":[{"id":1,"type":"RETRIEVE|REASON","lookup_question_ur":"...","depends_on":[],"urdu_entity_mention":"... or null"}]}',
            )

            normalized_per_row = [normalize_steps(value) for value in decomp_parsed]
            translation_prompts: list[str] = []
            translation_owner: list[int] = []
            for row_index, (row, steps) in enumerate(zip(batch, normalized_per_row)):
                if not steps:
                    continue
                goals = [{"id": step["id"], "lookup_question_ur": step["lookup_question_ur"]} for step in steps if step["type"] == "RETRIEVE"]
                translation_prompts.append(
                    TRANSLATE_PLAN_PROMPT.format(
                        question_ur=row["question_ur"],
                        goals_json=json.dumps(goals, ensure_ascii=False),
                    )
                )
                translation_owner.append(row_index)

            translation_raw = llm_generate(llm, sampling_long, translation_prompts) if translation_prompts else []
            translation_parsed: list[Any | None] = [extract_json_value(raw, dict) for raw in translation_raw]
            translation_parsed, translation_raw = repair_values(
                llm,
                sampling_long,
                translation_raw,
                translation_parsed,
                dict,
                '{"question_en":"...","goals":[{"id":1,"lookup_question_en":"..."}]}',
            )
            translations: dict[int, dict[str, Any]] = {}
            for owner, value in zip(translation_owner, translation_parsed):
                translations[owner] = value if isinstance(value, dict) else {}

            direct_prompts: list[str] = []
            translated_prompts: list[str] = []
            step_owners: list[tuple[int, int]] = []
            for row_index, (row, steps) in enumerate(zip(batch, normalized_per_row)):
                if not steps:
                    continue
                trans_obj = translations.get(row_index, {})
                question_en = norm_text(trans_obj.get("question_en"))
                goal_map: dict[int, str] = {}
                goals_raw = trans_obj.get("goals", [])
                if isinstance(goals_raw, list):
                    for goal in goals_raw:
                        if not isinstance(goal, dict):
                            continue
                        try:
                            goal_id = int(goal.get("id"))
                        except (TypeError, ValueError):
                            continue
                        translated_goal = norm_text(goal.get("lookup_question_en"))
                        if translated_goal:
                            goal_map[goal_id] = translated_goal
                for step_index, step in enumerate(steps):
                    if step["type"] != "RETRIEVE":
                        continue
                    lookup_en = goal_map.get(step["id"], "")
                    direct_prompts.append(
                        DIRECT_VIEW_PROMPT.format(
                            question_ur=row["question_ur"],
                            lookup_question_ur=step["lookup_question_ur"],
                            entity_ur=step["urdu_entity_mention"] or "null",
                        )
                    )
                    translated_prompts.append(
                        TRANSLATED_VIEW_PROMPT.format(
                            question_en=question_en,
                            lookup_question_en=lookup_en,
                        )
                    )
                    step_owners.append((row_index, step_index))

            direct_raw = llm_generate(llm, sampling_short, direct_prompts) if direct_prompts else []
            translated_view_raw = llm_generate(llm, sampling_short, translated_prompts) if translated_prompts else []
            direct_values: list[Any | None] = [extract_json_value(raw, dict) for raw in direct_raw]
            translated_values: list[Any | None] = [extract_json_value(raw, dict) for raw in translated_view_raw]
            direct_values, direct_raw = repair_values(
                llm,
                sampling_short,
                direct_raw,
                direct_values,
                dict,
                '{"search_query_en":"...","entity_candidates_en":["..."]}',
            )
            translated_values, translated_view_raw = repair_values(
                llm,
                sampling_short,
                translated_view_raw,
                translated_values,
                dict,
                '{"search_query_en":"...","entity_candidates_en":["..."]}',
            )

            view_by_step: dict[tuple[int, int], tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
            for owner, direct_value, translated_value in zip(step_owners, direct_values, translated_values):
                view_by_step[owner] = (normalize_query_view(direct_value), normalize_query_view(translated_value))

            for row_index, row in enumerate(batch):
                steps = normalized_per_row[row_index]
                trans_obj = translations.get(row_index, {})
                question_en = norm_text(trans_obj.get("question_en"))
                goal_map: dict[int, str] = {}
                if isinstance(trans_obj.get("goals"), list):
                    for goal in trans_obj["goals"]:
                        if isinstance(goal, dict):
                            try:
                                goal_id = int(goal.get("id"))
                            except (TypeError, ValueError):
                                continue
                            goal_map[goal_id] = norm_text(goal.get("lookup_question_en"))

                final_steps: list[dict[str, Any]] = []
                if steps:
                    for step_index, step in enumerate(steps):
                        enriched = dict(step)
                        if step["type"] == "RETRIEVE":
                            direct_view, translated_view = view_by_step.get((row_index, step_index), (None, None))
                            enriched.update(
                                {
                                    "lookup_question_en": goal_map.get(step["id"], ""),
                                    "direct_view": direct_view,
                                    "translated_view": translated_view,
                                }
                            )
                        final_steps.append(enriched)

                record = {
                    "schema_version": GEN_SCHEMA_VERSION,
                    "urbench_qid": row["urbench_qid"],
                    "input_question_sha256": sha256_text(row["question_ur"]),
                    "live_source_fields_used": ["urbench_qid", "question_ur"],
                    "question_ur": row["question_ur"],
                    "question_en_translation": question_en,
                    "parse_ok": bool(steps),
                    "steps": final_steps,
                    "created_unix": int(time.time()),
                }
                atomic_write_json(generation_cache_path(row["urbench_qid"]), record)

    records: list[dict[str, Any]] = []
    for row in legal_inputs:
        cached = valid_generation_cache(generation_cache_path(row["urbench_qid"]), row["urbench_qid"], row["question_ur"])
        if cached is None:
            raise PhaseRError(f"Missing or invalid generation cache for {row['urbench_qid']}")
        records.append(cached)

    parse_failures = sum(not record.get("parse_ok") for record in records)
    retrieve_steps = [step for record in records for step in record.get("steps", []) if step.get("type") == "RETRIEVE"]
    direct_failures = sum(not step.get("direct_view") for step in retrieve_steps)
    translated_failures = sum(not step.get("translated_view") for step in retrieve_steps)
    entity_steps = sum(bool(step.get("urdu_entity_mention")) for step in retrieve_steps)
    summary = {
        "schema_version": GEN_SCHEMA_VERSION,
        "dev50_items": len(records),
        "decomposition_parse_failures": parse_failures,
        "retrieve_steps": len(retrieve_steps),
        "entity_bearing_retrieve_steps": entity_steps,
        "direct_query_failures": direct_failures,
        "translated_query_failures": translated_failures,
        "ready_for_retrieval": parse_failures <= args.max_parse_failures and direct_failures == 0 and translated_failures == 0,
    }
    atomic_write_json(OUT_DIR / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["ready_for_retrieval"]:
        raise PhaseRError("Generation gate failed; inspect cache before loading the full index")


def title_lexical_score(candidate: str, title: str) -> float:
    cand = norm_title(candidate)
    page = norm_title(title)
    if not cand or not page:
        return 0.0
    if cand == page:
        return 1.0
    cand_tokens = cand.split()
    page_tokens = page.split()
    cand_set, page_set = set(cand_tokens), set(page_tokens)
    if cand_set and cand_set.issubset(page_set):
        return 0.95
    if page_set and page_set.issubset(cand_set):
        return 0.90
    overlap = len(cand_set & page_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_set)
    recall = overlap / len(page_set)
    return 2 * precision * recall / (precision + recall)


def select_grounded_title(candidate: str, hits: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for hit in hits:
        title = norm_text(hit.get("title"))
        if not title:
            continue
        lexical = title_lexical_score(candidate, title)
        retrieval = float(hit.get("score", 0.0))
        combined = 0.75 * lexical + 0.25 * max(0.0, min(1.0, retrieval))
        current = {
            "candidate": candidate,
            "title": title,
            "lexical_score": lexical,
            "retrieval_score": retrieval,
            "combined_score": combined,
        }
        if best is None or current["combined_score"] > best["combined_score"]:
            best = current
    if best is None:
        return {"candidate": candidate, "accepted": False, "title": None, "combined_score": 0.0}
    best["accepted"] = bool(best["combined_score"] >= threshold and best["lexical_score"] >= 0.50)
    if not best["accepted"]:
        best["title"] = None
    return best


def choose_view_grounding(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [result for result in results if result.get("accepted")]
    if not accepted:
        return {"title": None, "candidate": None, "score": 0.0, "all_candidates": results}
    best = max(accepted, key=lambda result: float(result.get("combined_score", 0.0)))
    return {
        "title": best.get("title"),
        "candidate": best.get("candidate"),
        "score": float(best.get("combined_score", 0.0)),
        "all_candidates": results,
    }


def canonical_query(raw_query: str, title: str | None) -> str:
    query = norm_text(raw_query)
    page = norm_text(title)
    if not page:
        return query
    if norm_title(page) in norm_title(query):
        return query
    return f"{page} {query}".strip()


def dedupe_queries(queries: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = norm_text(query)
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


def build_pool(queries: Sequence[str], hits_by_query: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Logical evidence budget is exactly 40 requested hits per condition.

    One unique query receives 40. Two unique view queries receive 20 each.
    Deduplication can make the actual unique pool smaller; this is reported and
    never silently refilled with extra hits.
    """
    unique_queries = dedupe_queries(queries)
    if not unique_queries:
        return [], []
    if len(unique_queries) == 1:
        allocations = [EVIDENCE_BUDGET]
    else:
        unique_queries = unique_queries[:2]
        allocations = [EVIDENCE_BUDGET // 2, EVIDENCE_BUDGET - EVIDENCE_BUDGET // 2]
    pool: dict[Any, dict[str, Any]] = {}
    for query, allocation in zip(unique_queries, allocations):
        for hit in hits_by_query.get(query, [])[:allocation]:
            key = hit.get("row", (norm_title(hit.get("title")), norm_text(hit.get("text"))))
            previous = pool.get(key)
            if previous is None or float(hit.get("score", -math.inf)) > float(previous.get("score", -math.inf)):
                pool[key] = hit
    return list(pool.values()), allocations


def rerank(reranker: Any, shared_urdu_goal: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pool:
        return []
    pairs = [[shared_urdu_goal, norm_text(hit.get("text"))] for hit in pool]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    order = sorted(range(len(pool)), key=lambda index: -float(scores[index]))[:FINAL_TOP_K]
    output: list[dict[str, Any]] = []
    for index in order:
        hit = pool[index]
        output.append(
            {
                "row": hit.get("row"),
                "title": norm_text(hit.get("title")),
                "text": norm_text(hit.get("text")),
                "retrieval_score": float(hit.get("score", 0.0)),
                "rerank_score": float(scores[index]),
            }
        )
    return output


def valid_retrieval_cache(path: Path, generation_digest: str, config_digest: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = read_json_object(path)
    except (OSError, json.JSONDecodeError, PhaseRError):
        return None
    if (
        record.get("schema_version") != RET_SCHEMA_VERSION
        or record.get("generation_digest") != generation_digest
        or record.get("config_digest") != config_digest
    ):
        return None
    return record


def generation_records_for_retrieval(qids: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures = 0
    for qid in qids:
        path = generation_cache_path(qid)
        if not path.exists():
            raise PhaseRError(f"Missing generation cache: {path}")
        record = read_json_object(path)
        if record.get("schema_version") != GEN_SCHEMA_VERSION or record.get("urbench_qid") != qid:
            raise PhaseRError(f"Invalid generation cache: {path}")
        if not record.get("parse_ok"):
            failures += 1
        records.append(record)
    if failures:
        raise PhaseRError(f"Generation contains {failures} parse failures; retrieval is blocked")
    return records


def process_retrieval_item(
    retriever: Any,
    reranker: Any,
    generation: dict[str, Any],
    ground_min: float,
    config_digest: str,
) -> dict[str, Any]:
    qid = generation["urbench_qid"]
    generation_digest = canonical_json_hash(generation)
    cache_path = retrieval_cache_path(qid)
    cached = valid_retrieval_cache(cache_path, generation_digest, config_digest)
    if cached is not None:
        return cached

    retrieve_steps = [step for step in generation.get("steps", []) if step.get("type") == "RETRIEVE"]

    # Ground all entity candidates in one batched title-discovery call per item.
    all_candidates: list[str] = []
    for step in retrieve_steps:
        for view_key in ("direct_view", "translated_view"):
            view = step.get(view_key) or {}
            all_candidates.extend(view.get("entity_candidates_en", [])[:MAX_ENTITY_CANDIDATES_PER_VIEW])
    unique_candidates = dedupe_queries(all_candidates)
    grounding_hits: dict[str, list[dict[str, Any]]] = {}
    if unique_candidates:
        batches = retriever.retrieve(unique_candidates, top_k=GROUND_TOP_K)
        grounding_hits = {query: hits for query, hits in zip(unique_candidates, batches)}

    step_plans: list[dict[str, Any]] = []
    all_evidence_queries: list[str] = []
    for step in retrieve_steps:
        direct_view = step.get("direct_view") or {}
        translated_view = step.get("translated_view") or {}
        direct_raw = norm_text(direct_view.get("search_query_en"))
        translated_raw = norm_text(translated_view.get("search_query_en"))

        direct_results = [
            select_grounded_title(candidate, grounding_hits.get(candidate, []), ground_min)
            for candidate in direct_view.get("entity_candidates_en", [])[:MAX_ENTITY_CANDIDATES_PER_VIEW]
        ]
        translated_results = [
            select_grounded_title(candidate, grounding_hits.get(candidate, []), ground_min)
            for candidate in translated_view.get("entity_candidates_en", [])[:MAX_ENTITY_CANDIDATES_PER_VIEW]
        ]
        direct_ground = choose_view_grounding(direct_results)
        translated_ground = choose_view_grounding(translated_results)
        direct_title = direct_ground.get("title")
        translated_title = translated_ground.get("title")

        if direct_title and translated_title and norm_title(direct_title) == norm_title(translated_title):
            state = "agree"
        elif direct_title and translated_title:
            state = "disagree"
        elif direct_title or translated_title:
            state = "partial"
        else:
            state = "ungrounded"

        direct_canonical = canonical_query(direct_raw, direct_title)
        translated_canonical = canonical_query(translated_raw, translated_title)
        condition_queries = {
            "R1": [direct_raw],
            "R2": [translated_raw],
            "R3": [direct_raw, translated_raw],
            "R4": [direct_canonical, translated_canonical],
        }
        all_evidence_queries.extend(query for queries in condition_queries.values() for query in queries)
        step_plans.append(
            {
                "step": step,
                "condition_queries": condition_queries,
                "grounding": {
                    "state": state,
                    "direct": direct_ground,
                    "translated": translated_ground,
                    "candidate_searches": len(dedupe_queries(
                        list(direct_view.get("entity_candidates_en", []))
                        + list(translated_view.get("entity_candidates_en", []))
                    )),
                },
            }
        )

    unique_evidence_queries = dedupe_queries(all_evidence_queries)
    evidence_hits: dict[str, list[dict[str, Any]]] = {}
    if unique_evidence_queries:
        batches = retriever.retrieve(unique_evidence_queries, top_k=EVIDENCE_BUDGET)
        evidence_hits = {query: hits for query, hits in zip(unique_evidence_queries, batches)}

    step_records: list[dict[str, Any]] = []
    for plan in step_plans:
        step = plan["step"]
        conditions: dict[str, Any] = {}
        for condition, queries in plan["condition_queries"].items():
            pool, allocations = build_pool(queries, evidence_hits)
            unique_queries = dedupe_queries(queries)[:2]
            conditions[condition] = {
                "queries": unique_queries,
                "requested_hits_by_query": allocations,
                "logical_evidence_hits_requested": sum(allocations),
                "logical_evidence_searches": len(unique_queries),
                "unique_pool_size": len(pool),
                "top3": rerank(reranker, step["lookup_question_ur"], pool),
            }
        step_records.append(
            {
                "step_id": step["id"],
                "lookup_question_ur": step["lookup_question_ur"],
                "lookup_question_en": step.get("lookup_question_en", ""),
                "urdu_entity_mention": step.get("urdu_entity_mention"),
                "direct_view": step.get("direct_view"),
                "translated_view": step.get("translated_view"),
                "grounding": plan["grounding"],
                "conditions": conditions,
            }
        )

    record = {
        "schema_version": RET_SCHEMA_VERSION,
        "urbench_qid": qid,
        "generation_digest": generation_digest,
        "config_digest": config_digest,
        "grounding_overhead": {
            "physical_candidate_queries": len(unique_candidates),
            "hits_requested_per_candidate": GROUND_TOP_K,
            "total_grounding_hits_requested": len(unique_candidates) * GROUND_TOP_K,
        },
        "physical_unique_evidence_queries": len(unique_evidence_queries),
        "steps": step_records,
        "created_unix": int(time.time()),
    }
    atomic_write_json(cache_path, record)
    return record


def aggregate_retrieval(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    flat_records: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, float]] = {
        condition: defaultdict(float) for condition in ("R1", "R2", "R3", "R4")
    }
    states: defaultdict[str, int] = defaultdict(int)
    total_grounding_queries = 0

    for item in records:
        total_grounding_queries += int(item["grounding_overhead"]["physical_candidate_queries"])
        for step in item.get("steps", []):
            states[step["grounding"]["state"]] += 1
            for condition, result in step["conditions"].items():
                diagnostics[condition]["steps"] += 1
                diagnostics[condition]["logical_evidence_searches"] += result["logical_evidence_searches"]
                diagnostics[condition]["logical_evidence_hits_requested"] += result["logical_evidence_hits_requested"]
                diagnostics[condition]["unique_pool_size"] += result["unique_pool_size"]
                flat_records.append(
                    {
                        "urbench_qid": item["urbench_qid"],
                        "step_id": step["step_id"],
                        "condition": condition,
                        "lookup_question_ur": step["lookup_question_ur"],
                        "lookup_question_en": step.get("lookup_question_en", ""),
                        "urdu_entity_mention": step.get("urdu_entity_mention"),
                        "queries": result["queries"],
                        "requested_hits_by_query": result["requested_hits_by_query"],
                        "logical_evidence_hits_requested": result["logical_evidence_hits_requested"],
                        "logical_evidence_searches": result["logical_evidence_searches"],
                        "unique_pool_size": result["unique_pool_size"],
                        "grounding": step["grounding"],
                        "top3": result["top3"],
                    }
                )

    budget_summary: dict[str, Any] = {}
    for condition, values in diagnostics.items():
        steps = int(values["steps"])
        budget_summary[condition] = {
            "steps": steps,
            "evidence_hits_requested_per_step": values["logical_evidence_hits_requested"] / max(steps, 1),
            "evidence_searches_per_step": values["logical_evidence_searches"] / max(steps, 1),
            "average_unique_pool_size": values["unique_pool_size"] / max(steps, 1),
            "final_top_k": FINAL_TOP_K,
        }

    for condition in ("R1", "R2", "R3", "R4"):
        requested = budget_summary[condition]["evidence_hits_requested_per_step"]
        if requested != EVIDENCE_BUDGET:
            raise PhaseRError(f"Budget violation in {condition}: {requested} != {EVIDENCE_BUDGET}")

    summary = {
        "schema_version": RET_SCHEMA_VERSION,
        "dev50_items": len(records),
        "retrieval_steps": sum(len(item.get("steps", [])) for item in records),
        "evidence_budget": budget_summary,
        "grounding_state_counts": dict(sorted(states.items())),
        "grounding_overhead": {
            "physical_candidate_queries": total_grounding_queries,
            "hits_requested_per_candidate": GROUND_TOP_K,
            "reported_separately_from_evidence_budget": True,
        },
        "config": config,
        "scientific_note": "R1 is direct Urdu-view to English-query generation; R2 is translate-first. R4 has the same evidence-passage budget as R3; title-grounding overhead is separate and explicit.",
    }
    atomic_write_jsonl(OUT_DIR / "r_records.jsonl", flat_records)
    atomic_write_json(OUT_DIR / "retrieval_summary.json", summary)
    return summary


def stage_retrieve(args: argparse.Namespace) -> None:
    qids = retrieval_guard_qids()
    generations = generation_records_for_retrieval(qids)
    RET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": RET_SCHEMA_VERSION,
        "evidence_budget": EVIDENCE_BUDGET,
        "final_top_k": FINAL_TOP_K,
        "ground_top_k": GROUND_TOP_K,
        "ground_min": args.ground_min,
        "grounding_backend": "full chunk index used only to discover candidate page titles; grounding calls are reported separately",
        "reranker_path": str(RERANKER_PATH),
        "embedder_device": args.device_embed,
    }
    config_digest = canonical_json_hash(config)

    sys.path.insert(0, str(BASE / "rag"))
    from retrieve import Retriever
    from sentence_transformers import CrossEncoder

    print("[retrieve] loading full FAISS index on compute node")
    retriever = Retriever(device=args.device_embed)
    print("[retrieve] loading reranker after generation process has exited")
    reranker = CrossEncoder(str(RERANKER_PATH), max_length=512, device="cuda")

    records: list[dict[str, Any]] = []
    for index, generation in enumerate(generations, 1):
        print(f"[retrieve] {index}/50 qid={generation['urbench_qid']}")
        record = process_retrieval_item(retriever, reranker, generation, args.ground_min, config_digest)
        records.append(record)

    summary = aggregate_retrieval(records, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def recursively_collect_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        result.append(value)
    elif isinstance(value, list):
        for item in value:
            result.extend(recursively_collect_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            result.extend(recursively_collect_strings(item))
    return result


def paragraph_page_title(paragraph_id: str) -> str:
    match = re.match(r"^(.*)-(\d+)$", paragraph_id)
    return match.group(1) if match else paragraph_id


def annotator_evidence_sets(row: dict[str, Any], paragraph_ids: set[str]) -> list[set[str]]:
    evidence = row.get("official_evidence")
    annotations = evidence if isinstance(evidence, list) else [evidence]
    result: list[set[str]] = []
    for annotation in annotations:
        ids = {
            value
            for value in recursively_collect_strings(annotation)
            if value.casefold() not in MARKERS and value in paragraph_ids
        }
        titles = {norm_title(paragraph_page_title(paragraph_id)) for paragraph_id in ids}
        if titles:
            result.append(titles)
    return result


def score_evidence_pages(
    dev_by_qid: dict[str, dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
    paragraph_ids: set[str],
) -> dict[str, Any]:
    retrieved: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in retrieval_records:
        qid = record["urbench_qid"]
        condition = record["condition"]
        for hit in record.get("top3", []):
            title = norm_title(hit.get("title"))
            if title:
                retrieved[qid][condition].add(title)

    output: dict[str, Any] = {}
    for condition in ("R1", "R2", "R3", "R4"):
        recalls: list[float] = []
        hits: list[int] = []
        for qid, row in dev_by_qid.items():
            alternatives = annotator_evidence_sets(row, paragraph_ids)
            if not alternatives:
                continue
            titles = retrieved[qid][condition]
            best_recall = max(len(titles & alternative) / len(alternative) for alternative in alternatives)
            recalls.append(best_recall)
            hits.append(int(any(titles & alternative for alternative in alternatives)))
        output[condition] = {
            "items_scored": len(recalls),
            "macro_best_annotator_page_recall_top3_per_generated_step": sum(recalls) / max(len(recalls), 1),
            "any_official_evidence_page_hit_rate_top3_per_generated_step": sum(hits) / max(len(hits), 1),
        }
    return output


def create_or_preserve_m1_labels(retrieval_records: list[dict[str, Any]]) -> Path:
    path = OUT_DIR / "m1_entity_labels.jsonl"
    if path.exists():
        print(f"[score] preserving existing manual labels: {path}")
        return path
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in retrieval_records:
        if not record.get("urdu_entity_mention"):
            continue
        key = (record["urbench_qid"], int(record["step_id"]))
        entry = by_key.setdefault(
            key,
            {
                "urbench_qid": key[0],
                "step_id": key[1],
                "lookup_question_ur": record["lookup_question_ur"],
                "lookup_question_en": record.get("lookup_question_en", ""),
                "urdu_entity_mention": record.get("urdu_entity_mention"),
                "direct_candidates": [],
                "translated_candidates": [],
                "ground_direct": record.get("grounding", {}).get("direct", {}).get("title"),
                "ground_translated": record.get("grounding", {}).get("translated", {}).get("title"),
                "correct_titles": [],
                "label_notes": "",
            },
        )
        if record["condition"] == "R1":
            entry["direct_candidates"] = (record.get("grounding", {}).get("direct", {}).get("all_candidates") or [])
            entry["translated_candidates"] = (record.get("grounding", {}).get("translated", {}).get("all_candidates") or [])
    atomic_write_jsonl(path, [by_key[key] for key in sorted(by_key)])
    return path


def score_m1_if_labeled(labels_path: Path, retrieval_records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = load_jsonl(labels_path)
    correct_by_key: dict[tuple[str, int], set[str]] = {}
    for label in labels:
        correct = {norm_title(title) for title in label.get("correct_titles", []) if norm_title(title)}
        if correct:
            correct_by_key[(label["urbench_qid"], int(label["step_id"]))] = correct
    if not correct_by_key:
        return {"status": "pending_manual_labels", "labeled_steps": 0}

    hits: dict[str, list[int]] = defaultdict(list)
    ground_direct: list[int] = []
    ground_translated: list[int] = []
    seen_ground: set[tuple[str, int]] = set()
    for record in retrieval_records:
        key = (record["urbench_qid"], int(record["step_id"]))
        correct = correct_by_key.get(key)
        if not correct:
            continue
        titles = {norm_title(hit.get("title")) for hit in record.get("top3", [])}
        hits[record["condition"]].append(int(bool(titles & correct)))
        if key not in seen_ground:
            ground = record.get("grounding", {})
            ground_direct.append(int(norm_title(ground.get("direct", {}).get("title")) in correct))
            ground_translated.append(int(norm_title(ground.get("translated", {}).get("title")) in correct))
            seen_ground.add(key)

    accuracy = {condition: sum(values) / max(len(values), 1) for condition, values in hits.items()}
    gates = {
        "R3_beats_R1_and_R2": accuracy.get("R3", 0.0) > max(accuracy.get("R1", 0.0), accuracy.get("R2", 0.0)),
        "R4_improves_R3_by_10pp": accuracy.get("R4", 0.0) >= accuracy.get("R3", 0.0) + 0.10,
    }
    return {
        "status": "scored",
        "labeled_steps": len(correct_by_key),
        "retrieved_correct_entity_title_at_3": accuracy,
        "ground_direct_accuracy": sum(ground_direct) / max(len(ground_direct), 1),
        "ground_translated_accuracy": sum(ground_translated) / max(len(ground_translated), 1),
        "directional_gates": gates,
    }


def build_m3_packet(
    dev_by_qid: dict[str, dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
    paragraphs: dict[str, Any],
) -> None:
    records_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in retrieval_records:
        records_by_qid[record["urbench_qid"]].append(record)
    packet: list[dict[str, Any]] = []
    for qid in frozen_qids():
        row = dev_by_qid[qid]
        evidence_ids = sorted(
            {
                value
                for value in recursively_collect_strings(row.get("official_evidence"))
                if value.casefold() not in MARKERS and value in paragraphs
            }
        )
        official = [
            {
                "paragraph_id": paragraph_id,
                "page_title": paragraph_page_title(paragraph_id),
                "text": paragraphs[paragraph_id],
            }
            for paragraph_id in evidence_ids
        ]
        packet.append(
            {
                "urbench_qid": qid,
                "question_ur": row["question_ur"],
                "official_evidence_paragraphs": official,
                "retrieval_records": records_by_qid.get(qid, []),
                "manual_premise_support_labels": {},
            }
        )
    atomic_write_jsonl(OUT_DIR / "m3_premise_support_review.jsonl", packet)


def stage_score(_: argparse.Namespace) -> None:
    qids = retrieval_guard_qids()
    retrieval_path = OUT_DIR / "r_records.jsonl"
    if not retrieval_path.exists():
        raise PhaseRError(f"Missing retrieval records: {retrieval_path}")
    retrieval_records = load_jsonl(retrieval_path)
    observed_qids = {record["urbench_qid"] for record in retrieval_records}
    if observed_qids != set(qids):
        raise PhaseRError(f"Retrieval output qids mismatch: observed={len(observed_qids)} expected=50")

    dev_rows = load_jsonl(DEV50_PATH)
    dev_by_qid = {norm_qid(row["urbench_qid"]): row for row in dev_rows}
    with PARAGRAPHS_PATH.open(encoding="utf-8") as handle:
        paragraphs = json.load(handle)
    if not isinstance(paragraphs, dict) or len(paragraphs) != 9251:
        raise PhaseRError(f"Expected 9251 paragraph entries, found {len(paragraphs) if isinstance(paragraphs, dict) else type(paragraphs)}")
    paragraph_ids = set(paragraphs)

    page_metrics = score_evidence_pages(dev_by_qid, retrieval_records, paragraph_ids)
    labels_path = create_or_preserve_m1_labels(retrieval_records)
    m1 = score_m1_if_labeled(labels_path, retrieval_records)
    build_m3_packet(dev_by_qid, retrieval_records, paragraphs)

    score_summary = {
        "official_metric_name": "item-level official evidence page-title recall from top-3 passages per generated retrieval step (best annotator alternative)",
        "strict_title_normalization_keeps_parenthetical_qualifiers": True,
        "page_title_metrics": page_metrics,
        "m1_entity_title": m1,
        "R4_page_recall_strictly_improves_R3": (
            page_metrics["R4"]["macro_best_annotator_page_recall_top3_per_generated_step"]
            > page_metrics["R3"]["macro_best_annotator_page_recall_top3_per_generated_step"]
        ),
        "claims_status": "DEV50 development diagnostic only; no significance or final-method claim",
    }
    atomic_write_json(OUT_DIR / "score_summary.json", score_summary)
    print(json.dumps(score_summary, ensure_ascii=False, indent=2))


def self_test() -> None:
    assert normalize_steps({"steps": [{"id": 1, "type": "retrieve", "lookup_question_ur": " سوال ", "depends_on": [], "urdu_entity_mention": " نام "}]})
    assert normalize_query_view({"search_query_en": " Vellore Fort location ", "entity_candidates_en": ["Vellore Fort", "Vellore Fort"]}) == {
        "search_query_en": "Vellore Fort location",
        "entity_candidates_en": ["Vellore Fort"],
    }
    assert paragraph_page_title("Boeing 747-400-3") == "Boeing 747-400"
    hits = [
        {"row": index, "title": f"T{index}", "text": f"text {index}", "score": 1.0 - index / 100}
        for index in range(40)
    ]
    pool, allocations = build_pool(["same", "same"], {"same": hits})
    assert allocations == [40] and len(pool) == 40
    hits2 = [dict(hit, row=index + 100) for index, hit in enumerate(hits)]
    pool, allocations = build_pool(["q1", "q2"], {"q1": hits, "q2": hits2})
    assert allocations == [20, 20] and len(pool) == 40
    grounded = select_grounded_title("Vellore Fort", [{"title": "Vellore Fort", "score": 0.8}], 0.60)
    assert grounded["accepted"] and grounded["title"] == "Vellore Fort"
    print("SELF-TEST PASS: parsers, title derivation, 40-hit budgets, duplicate-query allocation, grounding")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["generate", "retrieve", "score", "self-test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-parse-failures", type=int, default=5)
    parser.add_argument("--device-embed", default="cpu")
    parser.add_argument("--ground-min", type=float, default=DEFAULT_GROUND_MIN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "generate":
        stage_generate(args)
    elif args.stage == "retrieve":
        stage_retrieve(args)
    elif args.stage == "score":
        stage_score(args)


if __name__ == "__main__":
    try:
        main()
    except PhaseRError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
