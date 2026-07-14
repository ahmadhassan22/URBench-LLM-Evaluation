#!/usr/bin/env python3
"""Fixed-12 entity-first grounding probe for Urdu StrategyQA.

Purpose: test the proposed novel component before decomposition.  The live
method receives only the Urdu question.  It independently creates:

  D: literal/direct Urdu-span -> English candidate lattice
  T: translate-first question -> English candidate lattice
  U: union(D, T)

Only ENTITY candidates are searched against the full Wikipedia chunk index.
RELATION candidates are retained for manual lexical audit.  No decomposition,
reranking, answer generation, C0-C5, DEV50-wide retrieval, or eval458 access is
performed.

Stages:
  self-test  Standard-library checks.
  generate   Qwen3-14B candidate generation; question-only input; atomic cache.
  ground     Full-index title discovery; no Qwen process alive; atomic cache.
  audit      Official fields enter only offline; produces strict evidence-page
             proxy metrics and a human-readable 12-item decision packet.

Output: data/strategyqa_official/phase_r_entity_first_probe12_v1/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
MODEL_PATH = Path("/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B")

OUT_DIR = OFFICIAL_DIR / "phase_r_entity_first_probe12_v1"
GEN_CACHE = OUT_DIR / "cache" / "generation"
GROUND_CACHE = OUT_DIR / "cache" / "grounding"

GEN_SCHEMA = "entity-first-probe12-generation-v1.0"
GROUND_SCHEMA = "entity-first-probe12-grounding-v1.0"
SAMPLE_SIZE = 12
SAMPLE_SEED = 42
MAX_ITEMS_PER_VIEW = 6
MAX_CANDIDATES_PER_ITEM = 5
MAX_ENTITY_CANDIDATES_PER_VIEW = 15
GROUND_TOP_K = 10
TITLE_POOL_CAP = 30
MARKERS = {"operation", "no_evidence"}


class ProbeError(RuntimeError):
    pass


def norm_qid(value: Any) -> str:
    return str(value).strip()


def norm_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", value).strip()


def norm_key(value: Any) -> str:
    return norm_text(value).replace("_", " ").casefold()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProbeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ProbeError(f"Expected object at {path}:{line_number}")
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
        raise ProbeError(f"Expected object in {path}")
    return value


def frozen_dev50_qids() -> list[str]:
    qids = [norm_qid(line) for line in DEV50_QIDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(qids) != 50 or len(set(qids)) != 50:
        raise ProbeError("Frozen DEV50 qid list is not exactly 50 unique rows")
    return qids


def eval458_qids() -> set[str]:
    qids = {norm_qid(row.get("qid")) for row in load_jsonl(EVAL458_PATH)}
    if len(qids) != 458:
        raise ProbeError(f"Expected 458 eval qids, found {len(qids)}")
    return qids


def fixed_sample_qids() -> list[str]:
    # Matches the earlier oracle audit: sorted cache filenames, seed 42.
    ordered = sorted(frozen_dev50_qids())
    return random.Random(SAMPLE_SEED).sample(ordered, SAMPLE_SIZE)


def live_inputs() -> list[dict[str, str]]:
    qids = fixed_sample_qids()
    if set(qids) & eval458_qids():
        raise ProbeError("Probe12 intersects eval458")
    dev = load_jsonl(DEV50_PATH)
    by_qid: dict[str, str] = {}
    for row in dev:
        qid = norm_qid(row.get("urbench_qid"))
        if qid in qids:
            if row.get("is_eval") is not False:
                raise ProbeError(f"Probe row {qid} is not marked non-eval")
            question = norm_text(row.get("question_ur"))
            if not question:
                raise ProbeError(f"Missing Urdu question for {qid}")
            by_qid[qid] = question
    if set(by_qid) != set(qids):
        raise ProbeError("Could not resolve every fixed sample qid")
    print("[guard] fixed seed-42 sample=12, DEV50 only, zero eval458 overlap")
    print("[guard] live fields passed to Qwen: urbench_qid + question_ur only")
    return [{"urbench_qid": qid, "question_ur": by_qid[qid]} for qid in qids]


def extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.split("</think>")[-1].strip() if "</think>" in raw else raw.strip()
    position = text.find("{")
    if position < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[position:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def unique_texts(values: Any, cap: int) -> list[str]:
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
        if len(output) >= cap:
            break
    return output


def normalize_items(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        return None
    output: list[dict[str, Any]] = []
    for raw in value["items"]:
        if not isinstance(raw, dict):
            return None
        span = norm_text(raw.get("span_ur"))
        kind = norm_text(raw.get("kind")).upper()
        candidates = unique_texts(raw.get("candidates_en", []), MAX_CANDIDATES_PER_ITEM)
        if not span or kind not in {"ENTITY", "RELATION"} or not candidates:
            return None
        output.append({"span_ur": span, "kind": kind, "candidates_en": candidates})
        if len(output) >= MAX_ITEMS_PER_VIEW:
            break
    return output or None


def normalize_direct(value: Any) -> list[dict[str, Any]] | None:
    return normalize_items(value)


def normalize_translated(value: Any) -> tuple[str, list[dict[str, Any]]] | None:
    if not isinstance(value, dict):
        return None
    question_en = norm_text(value.get("question_en"))
    items = normalize_items(value)
    if not question_en or not items:
        return None
    return question_en, items


DIRECT_EXAMPLES = """Synthetic transliteration examples (not StrategyQA data):
- Urdu entity `ایکس باکس سیریز ایکس` -> candidates `Xbox Series X`, `X Box Series X`
- Urdu entity `جے۔ آر۔ آر۔ ٹولکین` -> candidates `J. R. R. Tolkien`, `JRR Tolkien`
- Urdu relation `رفتار کی حد` -> candidates `speed limit`, `maximum speed`
"""

DIRECT_PROMPT = """Extract lexical anchors directly from this Urdu question without first translating the full question.

{examples}

Urdu question:
{question_ur}

Return named entities (people, places, organizations, works, products, dates,
named concepts) and critical relation/attribute phrases whose mistranslation
would change the question. For ENTITY items, produce up to 5 literal English
transliteration/name candidates. Preserve letters, initials, and spoken digits;
do not replace an unfamiliar name with a more famous similar name. For RELATION
items, produce up to 5 literal English gloss candidates. Do not answer or infer
facts.

Return ONLY JSON:
{{"items":[{{"span_ur":"...","kind":"ENTITY|RELATION","candidates_en":["..."]}}]}}"""

TRANSLATED_PROMPT = """Translate the complete Urdu question to English, then independently extract its named entities and critical relation/attribute phrases.

Urdu question:
{question_ur}

For every item provide up to 5 possible English names or glosses. Preserve
uncertainty by listing alternatives rather than silently substituting a similar
entity. Do not answer the question and do not infer facts.

Return ONLY JSON:
{{"question_en":"...","items":[{{"span_ur":"...","kind":"ENTITY|RELATION","candidates_en":["..."]}}]}}"""

REPAIR_PROMPT = """Convert the malformed response to the required JSON schema. Do not add answers or facts.

Required schema:
{schema}

Malformed response:
{raw}

Output JSON only."""


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
    parsed_values: list[dict[str, Any] | None],
    schema: str,
) -> tuple[list[str], list[dict[str, Any] | None]]:
    failed = [index for index, value in enumerate(parsed_values) if value is None]
    if not failed:
        return raw_values, parsed_values
    prompts = [REPAIR_PROMPT.format(schema=schema, raw=raw_values[index][:8000]) for index in failed]
    repaired = llm_generate(llm, sampling, prompts, thinking=False)
    for index, raw in zip(failed, repaired):
        raw_values[index] = raw
        parsed_values[index] = extract_json(raw)
    return raw_values, parsed_values


def generation_path(qid: str) -> Path:
    return GEN_CACHE / f"{qid}.json"


def grounding_path(qid: str) -> Path:
    return GROUND_CACHE / f"{qid}.json"


def valid_generation(
    path: Path,
    qid: str,
    question: str,
    *,
    require_parse_ok: bool = True,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = read_json_object(path)
    except (OSError, json.JSONDecodeError, ProbeError):
        return None
    if (
        value.get("schema_version") != GEN_SCHEMA
        or value.get("urbench_qid") != qid
        or value.get("input_question_sha256") != sha256_text(question)
    ):
        return None
    if require_parse_ok and value.get("parse_ok") is not True:
        return None
    return value


def stage_generate(args: argparse.Namespace) -> None:
    inputs = live_inputs()
    GEN_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [
        row for row in inputs
        if valid_generation(generation_path(row["urbench_qid"]), row["urbench_qid"], row["question_ur"]) is None
    ]
    print(f"[generate] cached={len(inputs)-len(todo)} todo={len(todo)}")
    if todo:
        if not MODEL_PATH.exists():
            raise ProbeError(f"Missing model: {MODEL_PATH}")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(MODEL_PATH),
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=6144,
            trust_remote_code=True,
        )
        think_sampling = SamplingParams(temperature=0.0, max_tokens=2200, seed=0)
        repair_sampling = SamplingParams(temperature=0.0, max_tokens=700, seed=0)

        for offset in range(0, len(todo), args.batch_size):
            batch = todo[offset:offset + args.batch_size]
            print(f"[generate] batch {offset+1}-{offset+len(batch)} / {len(todo)}")
            direct_raw = llm_generate(
                llm,
                think_sampling,
                [DIRECT_PROMPT.format(examples=DIRECT_EXAMPLES, question_ur=row["question_ur"]) for row in batch],
                thinking=True,
            )
            translated_raw = llm_generate(
                llm,
                think_sampling,
                [TRANSLATED_PROMPT.format(question_ur=row["question_ur"]) for row in batch],
                thinking=True,
            )
            direct_values: list[dict[str, Any] | None] = [extract_json(raw) for raw in direct_raw]
            translated_values: list[dict[str, Any] | None] = [extract_json(raw) for raw in translated_raw]
            direct_values = [value if normalize_direct(value) else None for value in direct_values]
            translated_values = [value if normalize_translated(value) else None for value in translated_values]
            direct_raw, direct_values = repair_invalid(
                llm,
                repair_sampling,
                direct_raw,
                direct_values,
                '{"items":[{"span_ur":"...","kind":"ENTITY|RELATION","candidates_en":["..."]}]}',
            )
            translated_raw, translated_values = repair_invalid(
                llm,
                repair_sampling,
                translated_raw,
                translated_values,
                '{"question_en":"...","items":[{"span_ur":"...","kind":"ENTITY|RELATION","candidates_en":["..."]}]}',
            )

            # A schema-repair prompt can itself remain malformed.  Retry only
            # the failed view from the original question, with thinking off,
            # instead of treating an existing cache as "missing" or borrowing
            # candidates from the other view (which would destroy view
            # independence).
            direct_failed = [
                index for index, value in enumerate(direct_values)
                if normalize_direct(value) is None
            ]
            if direct_failed:
                retry_raw = llm_generate(
                    llm,
                    repair_sampling,
                    [
                        DIRECT_PROMPT.format(
                            examples=DIRECT_EXAMPLES,
                            question_ur=batch[index]["question_ur"],
                        )
                        for index in direct_failed
                    ],
                    thinking=False,
                )
                for index, raw in zip(direct_failed, retry_raw):
                    direct_raw[index] = raw
                    direct_values[index] = extract_json(raw)

            translated_failed = [
                index for index, value in enumerate(translated_values)
                if normalize_translated(value) is None
            ]
            if translated_failed:
                retry_raw = llm_generate(
                    llm,
                    repair_sampling,
                    [
                        TRANSLATED_PROMPT.format(
                            question_ur=batch[index]["question_ur"],
                        )
                        for index in translated_failed
                    ],
                    thinking=False,
                )
                for index, raw in zip(translated_failed, retry_raw):
                    translated_raw[index] = raw
                    translated_values[index] = extract_json(raw)

            for row, direct_value, translated_value, direct_text, translated_text in zip(
                batch,
                direct_values,
                translated_values,
                direct_raw,
                translated_raw,
            ):
                direct_items = normalize_direct(direct_value)
                translated = normalize_translated(translated_value)
                translated_question = translated[0] if translated else ""
                translated_items = translated[1] if translated else []
                record = {
                    "schema_version": GEN_SCHEMA,
                    "urbench_qid": row["urbench_qid"],
                    "input_question_sha256": sha256_text(row["question_ur"]),
                    "live_source_fields_used": ["urbench_qid", "question_ur"],
                    "question_ur": row["question_ur"],
                    "direct_items": direct_items or [],
                    "translated_question_en": translated_question,
                    "translated_items": translated_items,
                    "direct_parse_ok": direct_items is not None,
                    "translated_parse_ok": translated is not None,
                    "parse_ok": bool(direct_items and translated),
                    "direct_raw_tail": direct_text[-8000:],
                    "translated_raw_tail": translated_text[-8000:],
                    "created_unix": int(time.time()),
                }
                atomic_write_json(generation_path(row["urbench_qid"]), record)

    records: list[dict[str, Any]] = []
    for row in inputs:
        record = valid_generation(
            generation_path(row["urbench_qid"]),
            row["urbench_qid"],
            row["question_ur"],
            require_parse_ok=False,
        )
        if record is None:
            raise ProbeError(f"Missing or structurally invalid generation cache for {row['urbench_qid']}")
        records.append(record)
    failures = sum(not record.get("parse_ok") for record in records)
    direct_failures = [
        record["urbench_qid"] for record in records
        if not record.get("direct_items")
    ]
    translated_failures = [
        record["urbench_qid"] for record in records
        if not record.get("translated_question_en") or not record.get("translated_items")
    ]
    direct_entities = sum(item["kind"] == "ENTITY" for record in records for item in record["direct_items"])
    translated_entities = sum(item["kind"] == "ENTITY" for record in records for item in record["translated_items"])
    direct_relations = sum(item["kind"] == "RELATION" for record in records for item in record["direct_items"])
    translated_relations = sum(item["kind"] == "RELATION" for record in records for item in record["translated_items"])
    summary = {
        "schema_version": GEN_SCHEMA,
        "probe_items": len(records),
        "parse_failures": failures,
        "direct_parse_failure_qids": direct_failures,
        "translated_parse_failure_qids": translated_failures,
        "direct_entity_spans": direct_entities,
        "translated_entity_spans": translated_entities,
        "direct_relation_spans": direct_relations,
        "translated_relation_spans": translated_relations,
        "ready_for_grounding": failures == 0 and direct_entities > 0 and translated_entities > 0,
        "ready_for_method_claim": False,
    }
    atomic_write_json(OUT_DIR / "generation_summary.json", summary)
    atomic_write_jsonl(OUT_DIR / "generation_records.jsonl", records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready_for_grounding"]:
        raise ProbeError(
            "Candidate generation gate failed; "
            f"direct_failures={direct_failures}, translated_failures={translated_failures}"
        )


def lexical_title_score(candidate: str, title: str) -> float:
    candidate_key, title_key = norm_key(candidate), norm_key(title)
    if not candidate_key or not title_key:
        return 0.0
    if candidate_key == title_key:
        return 1.0
    candidate_tokens, title_tokens = set(candidate_key.split()), set(title_key.split())
    if candidate_tokens.issubset(title_tokens):
        return 0.95
    if title_tokens.issubset(candidate_tokens):
        return 0.90
    overlap = len(candidate_tokens & title_tokens)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(candidate_tokens), overlap / len(title_tokens)
    return 2 * precision * recall / (precision + recall)


def candidate_title_rows(candidate: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_title: dict[str, dict[str, Any]] = {}
    for hit in hits:
        title = norm_text(hit.get("title"))
        if not title:
            continue
        retrieval_score = float(hit.get("score", 0.0))
        lexical_score = lexical_title_score(candidate, title)
        combined = 0.65 * lexical_score + 0.35 * max(0.0, min(1.0, retrieval_score))
        row = {
            "candidate": candidate,
            "title": title,
            "retrieval_score": retrieval_score,
            "lexical_score": lexical_score,
            "combined_score": combined,
        }
        key = norm_key(title)
        previous = by_title.get(key)
        if previous is None or row["combined_score"] > previous["combined_score"]:
            by_title[key] = row
    return sorted(by_title.values(), key=lambda row: (-row["combined_score"], -row["retrieval_score"]))


def aggregate_title_pool(candidate_rows: Sequence[dict[str, Any]], cap: int = TITLE_POOL_CAP) -> list[dict[str, Any]]:
    by_title: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        key = norm_key(row.get("title"))
        previous = by_title.get(key)
        if key and (previous is None or float(row["combined_score"]) > float(previous["combined_score"])):
            by_title[key] = row
    return sorted(by_title.values(), key=lambda row: (-float(row["combined_score"]), -float(row["retrieval_score"])))[:cap]


def entity_candidates(items: Sequence[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.get("kind") != "ENTITY":
            continue
        for candidate in item.get("candidates_en", []):
            clean = norm_text(candidate)
            key = clean.casefold()
            if clean and key not in seen:
                output.append(clean)
                seen.add(key)
            if len(output) >= MAX_ENTITY_CANDIDATES_PER_VIEW:
                return output
    return output


def valid_grounding(path: Path, generation_digest: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = read_json_object(path)
    except (OSError, json.JSONDecodeError, ProbeError):
        return None
    if value.get("schema_version") != GROUND_SCHEMA or value.get("generation_digest") != generation_digest:
        return None
    return value


def stage_ground(args: argparse.Namespace) -> None:
    inputs = live_inputs()
    GROUND_CACHE.mkdir(parents=True, exist_ok=True)
    generations: list[dict[str, Any]] = []
    for row in inputs:
        record = valid_generation(generation_path(row["urbench_qid"]), row["urbench_qid"], row["question_ur"])
        if record is None or not record.get("parse_ok"):
            raise ProbeError(f"Generation not ready for {row['urbench_qid']}")
        generations.append(record)

    sys.path.insert(0, str(BASE / "rag"))
    from retrieve import Retriever

    print("[ground] loading full FAISS index; Qwen process has already exited")
    retriever = Retriever(device=args.device_embed)
    grounded: list[dict[str, Any]] = []
    for index, generation in enumerate(generations, 1):
        qid = generation["urbench_qid"]
        digest = canonical_hash(generation)
        cached = valid_grounding(grounding_path(qid), digest)
        if cached is not None:
            print(f"[ground] {index}/12 qid={qid} cached")
            grounded.append(cached)
            continue
        print(f"[ground] {index}/12 qid={qid}")
        direct_candidates = entity_candidates(generation["direct_items"])
        translated_candidates = entity_candidates(generation["translated_items"])
        union_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in direct_candidates + translated_candidates:
            key = candidate.casefold()
            if key not in seen:
                union_candidates.append(candidate)
                seen.add(key)
        hits_by_candidate: dict[str, list[dict[str, Any]]] = {}
        if union_candidates:
            hit_batches = retriever.retrieve(union_candidates, top_k=GROUND_TOP_K)
            hits_by_candidate = {candidate: hits for candidate, hits in zip(union_candidates, hit_batches)}
        rows_by_candidate = {
            candidate: candidate_title_rows(candidate, hits_by_candidate.get(candidate, []))
            for candidate in union_candidates
        }
        direct_pool = aggregate_title_pool([row for candidate in direct_candidates for row in rows_by_candidate[candidate]])
        translated_pool = aggregate_title_pool([row for candidate in translated_candidates for row in rows_by_candidate[candidate]])
        union_pool = aggregate_title_pool([row for candidate in union_candidates for row in rows_by_candidate[candidate]])
        record = {
            "schema_version": GROUND_SCHEMA,
            "urbench_qid": qid,
            "generation_digest": digest,
            "direct_entity_candidates": direct_candidates,
            "translated_entity_candidates": translated_candidates,
            "union_entity_candidates": union_candidates,
            "cross_view_exact_candidate_overlap": sorted(set(map(norm_key, direct_candidates)) & set(map(norm_key, translated_candidates))),
            "candidate_title_rows": rows_by_candidate,
            "direct_title_pool": direct_pool,
            "translated_title_pool": translated_pool,
            "union_title_pool": union_pool,
            "physical_grounding_queries": len(union_candidates),
            "hits_requested_per_query": GROUND_TOP_K,
            "created_unix": int(time.time()),
        }
        atomic_write_json(grounding_path(qid), record)
        grounded.append(record)
    summary = {
        "schema_version": GROUND_SCHEMA,
        "probe_items": len(grounded),
        "physical_grounding_queries": sum(row["physical_grounding_queries"] for row in grounded),
        "hits_requested_per_query": GROUND_TOP_K,
        "average_direct_title_pool": sum(len(row["direct_title_pool"]) for row in grounded) / len(grounded),
        "average_translated_title_pool": sum(len(row["translated_title_pool"]) for row in grounded) / len(grounded),
        "average_union_title_pool": sum(len(row["union_title_pool"]) for row in grounded) / len(grounded),
        "ready_for_method_claim": False,
    }
    atomic_write_json(OUT_DIR / "grounding_summary.json", summary)
    atomic_write_jsonl(OUT_DIR / "grounding_records.jsonl", grounded)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def recursively_collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(recursively_collect_strings(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(recursively_collect_strings(item))
        return output
    return []


def paragraph_page_title(paragraph_id: str) -> str:
    match = re.match(r"^(.*)-(\d+)$", paragraph_id)
    return match.group(1) if match else paragraph_id


def official_evidence_title_alternatives(row: dict[str, Any], paragraph_ids: set[str]) -> list[set[str]]:
    evidence = row.get("official_evidence")
    annotations = evidence if isinstance(evidence, list) else [evidence]
    alternatives: list[set[str]] = []
    for annotation in annotations:
        ids = {
            value for value in recursively_collect_strings(annotation)
            if value.casefold() not in MARKERS and value in paragraph_ids
        }
        titles = {norm_key(paragraph_page_title(value)) for value in ids}
        if titles:
            alternatives.append(titles)
    return alternatives


def pool_title_keys(pool: Sequence[dict[str, Any]]) -> set[str]:
    return {norm_key(row.get("title")) for row in pool if norm_key(row.get("title"))}


def stage_audit(_: argparse.Namespace) -> None:
    inputs = live_inputs()
    qids = [row["urbench_qid"] for row in inputs]
    generations = {row["urbench_qid"]: row for row in load_jsonl(OUT_DIR / "generation_records.jsonl")}
    groundings = {row["urbench_qid"]: row for row in load_jsonl(OUT_DIR / "grounding_records.jsonl")}
    dev = {norm_qid(row["urbench_qid"]): row for row in load_jsonl(DEV50_PATH)}
    if set(generations) != set(qids) or set(groundings) != set(qids):
        raise ProbeError("Audit inputs do not exactly match fixed probe12")
    with PARAGRAPHS_PATH.open(encoding="utf-8") as handle:
        paragraphs = json.load(handle)
    if not isinstance(paragraphs, dict) or len(paragraphs) != 9251:
        raise ProbeError("Official paragraph dictionary is not the verified 9251-entry file")
    paragraph_ids = set(paragraphs)

    proxy_hits: dict[str, list[int]] = defaultdict(list)
    proxy_recall: dict[str, list[float]] = defaultdict(list)
    packet: list[dict[str, Any]] = []
    lines: list[str] = []
    for qid in qids:
        generation = generations[qid]
        grounding = groundings[qid]
        row = dev[qid]
        alternatives = official_evidence_title_alternatives(row, paragraph_ids)
        for name, key in (("direct", "direct_title_pool"), ("translated", "translated_title_pool"), ("union", "union_title_pool")):
            titles = pool_title_keys(grounding[key])
            best_recall = max((len(titles & alternative) / len(alternative) for alternative in alternatives), default=0.0)
            proxy_recall[name].append(best_recall)
            proxy_hits[name].append(int(any(titles & alternative for alternative in alternatives)))
        packet.append(
            {
                "urbench_qid": qid,
                "question_ur": generation["question_ur"],
                "official_question_en_oracle_only": row["question_en"],
                "official_decomposition_oracle_only": row["official_decomposition"],
                "direct_items": generation["direct_items"],
                "translated_question_en": generation["translated_question_en"],
                "translated_items": generation["translated_items"],
                "direct_title_pool": grounding["direct_title_pool"],
                "translated_title_pool": grounding["translated_title_pool"],
                "union_title_pool": grounding["union_title_pool"],
                "manual_correct_entity_titles": [],
                "manual_relation_glosses_correct": None,
                "manual_direct_correct_title_present": None,
                "manual_translated_correct_title_present": None,
                "manual_union_correct_title_present": None,
                "manual_notes": "",
            }
        )
        lines.extend(
            [
                "=" * 88,
                f"QID: {qid}",
                f"UR: {generation['question_ur']}",
                f"OFFICIAL EN (ORACLE ONLY): {row['question_en']}",
                f"TRANSLATED VIEW EN: {generation['translated_question_en']}",
                "DIRECT ITEMS:",
            ]
        )
        for item in generation["direct_items"]:
            lines.append(f"  [{item['kind']}] {item['span_ur']} -> {item['candidates_en']}")
        lines.append("TRANSLATED ITEMS:")
        for item in generation["translated_items"]:
            lines.append(f"  [{item['kind']}] {item['span_ur']} -> {item['candidates_en']}")
        for label, key in (("DIRECT", "direct_title_pool"), ("TRANSLATED", "translated_title_pool"), ("UNION", "union_title_pool")):
            lines.append(f"{label} GROUNDED TITLES (top 12 of {len(grounding[key])}):")
            for title_row in grounding[key][:12]:
                lines.append(
                    f"  {title_row['combined_score']:.3f} [{title_row['title']}] via {title_row['candidate']!r}"
                )
        lines.append("OFFICIAL DECOMPOSITION — ORACLE ONLY:")
        for index, step in enumerate(row["official_decomposition"], 1):
            lines.append(f"  {index}. {step}")
        lines.append("")

    metrics = {
        name: {
            "items": len(proxy_hits[name]),
            "any_official_evidence_page_in_title_pool": sum(proxy_hits[name]) / len(proxy_hits[name]),
            "macro_best_annotator_evidence_page_recall": sum(proxy_recall[name]) / len(proxy_recall[name]),
        }
        for name in ("direct", "translated", "union")
    }
    summary = {
        "probe_items": SAMPLE_SIZE,
        "automatic_proxy_only": "official evidence page titles are not always the canonical question entity",
        "evidence_page_title_proxy": metrics,
        "manual_entity_gate": "pending; correct union title coverage must be >=80% of entity-bearing questions and union must beat both single views",
        "manual_relation_gate": "pending; relation/gloss identity must be correct on >=10/12 questions",
        "ready_for_decomposition": False,
        "ready_for_method_claim": False,
    }
    atomic_write_json(OUT_DIR / "audit_summary.json", summary)
    atomic_write_jsonl(OUT_DIR / "manual_audit_packet.jsonl", packet)
    atomic_write_text(OUT_DIR / "manual_audit_report.txt", "\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[audit] report: {OUT_DIR / 'manual_audit_report.txt'}")


def self_test() -> None:
    direct = normalize_direct(
        {"items": [
            {"span_ur": "جے۔ آر۔ آر۔ ٹولکین", "kind": "ENTITY", "candidates_en": ["J. R. R. Tolkien", "JRR Tolkien"]},
            {"span_ur": "رفتار کی حد", "kind": "RELATION", "candidates_en": ["speed limit", "maximum speed"]},
        ]}
    )
    assert direct and len(direct) == 2
    translated = normalize_translated(
        {"question_en": "Example question", "items": [{"span_ur": "نام", "kind": "ENTITY", "candidates_en": ["Name"]}]}
    )
    assert translated and translated[0] == "Example question"
    synthetic_qids = [f"q{index:02d}" for index in range(50)]
    sample_a = random.Random(42).sample(sorted(synthetic_qids), 12)
    sample_b = random.Random(42).sample(sorted(reversed(synthetic_qids)), 12)
    assert sample_a == sample_b and len(sample_a) == len(set(sample_a)) == 12
    rows = candidate_title_rows(
        "Darth Vader",
        [
            {"title": "Darth Vader", "score": 0.8},
            {"title": "Darth Vader", "score": 0.7},
            {"title": "Darth Maul", "score": 0.75},
        ],
    )
    assert rows[0]["title"] == "Darth Vader" and len(rows) == 2
    pool = aggregate_title_pool(rows + rows)
    assert len(pool) == 2
    print("SELF-TEST PASS: fixed sample, candidate schemas, transliteration views, title dedupe/ranking")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["self-test", "generate", "ground", "audit"])
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--device-embed", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "generate":
        stage_generate(args)
    elif args.stage == "ground":
        stage_ground(args)
    elif args.stage == "audit":
        stage_audit(args)


if __name__ == "__main__":
    try:
        main()
    except ProbeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
