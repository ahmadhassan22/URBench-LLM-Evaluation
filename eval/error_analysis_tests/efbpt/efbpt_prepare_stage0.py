#!/usr/bin/env python3
"""Prepare frozen EFBPT Stage-0 structures without making human judgments."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import heapq
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_RELATIVE_PATH = SCRIPT_PATH.relative_to(REPO).as_posix()

FREEZE_PATH = REPO / "docs/EFBPT_STAGE0_SOURCE_ROLE_ATTAINABILITY_FREEZE.md"
DEV200_PATH = REPO / "data/strategyqa_official/dev200_seed4242.jsonl"
MAPPED_PATH = REPO / "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl"
OFFICIAL_TRAIN_PATH = REPO / "data/strategyqa_official/train.json"
OFFICIAL_DEV_PATH = REPO / "data/strategyqa_official/dev.json"
PARAGRAPHS_PATH = REPO / "data/strategyqa_official/strategyqa_train_paragraphs.json"
D2_COVERAGE_PATH = REPO / "outputs/efbpt/d2/d2_title_coverage.json"
D2_RECALL_SPLIT_PATH = REPO / "outputs/efbpt/d2/d2_recall_split.json"
CORPUS_METADATA_RELATIVE_PATH = "rag/index/wikipedia_full_meta.jsonl"
OUTPUT_DIR = REPO / "data/strategyqa_official/efbpt/stage0"

FREEZE_COMMIT = "fc3013afd5631d1979c78c10230c25d2ef11eedf"
EXPECTED_SHA256 = {
    DEV200_PATH: "1ae2cd21c93d1c8d3fda8f6990a183df558e6509d0884fadb29983f5f610d43c",
    D2_COVERAGE_PATH: "334d4de46a2c0d498807c45ef34f466aa17d554559a9e3ac7c7ba6e84479d3e5",
    D2_RECALL_SPLIT_PATH: "a93f9594d9915a4684a8c84e94cb746116da31744a449ba9d9726a961d7402a1",
}
ADDITIONAL_OFFICIAL_INPUTS = (
    MAPPED_PATH,
    OFFICIAL_TRAIN_PATH,
    OFFICIAL_DEV_PATH,
    PARAGRAPHS_PATH,
)

SEED = 20260822
RELIABILITY_SAMPLE_SIZE = 160
CORPUS_UNIQUE_NORMALIZED_TITLES = 6_402_346
MARKERS = {"operation", "no_evidence"}
STRATUM_ORDER = (
    "NO__EXACT_ABSENT",
    "NO__EXACT_PRESENT",
    "YES__EXACT_ABSENT",
    "YES__EXACT_PRESENT",
)
EXPECTED_STRATUM_COUNTS = {
    "NO__EXACT_ABSENT": 83,
    "NO__EXACT_PRESENT": 251,
    "YES__EXACT_ABSENT": 108,
    "YES__EXACT_PRESENT": 194,
}
EXPECTED_STRATUM_QUOTAS = {
    "NO__EXACT_ABSENT": 21,
    "NO__EXACT_PRESENT": 63,
    "YES__EXACT_ABSENT": 27,
    "YES__EXACT_PRESENT": 49,
}
ARTIFACT_NAMES = (
    "source_instance_master.jsonl",
    "official_evidence_links.jsonl",
    "lexical_trace_instances.jsonl",
    "pass1_question_manifest.jsonl",
    "pass2_order_manifest.json",
    "reliability160_manifest.json",
    "human_annotation_log.schema.json",
    "stage0_preparation_summary.json",
    "reproducibility_manifest.json",
)
JSONL_ARTIFACTS = {
    "source_instance_master.jsonl",
    "official_evidence_links.jsonl",
    "lexical_trace_instances.jsonl",
    "pass1_question_manifest.jsonl",
}

HUMAN_NULL_FIELDS = (
    "source_role",
    "explicit_relation_type",
    "urdu_span_if_explicit",
    "confidence",
    "rationale",
    "concrete_intermediate_information",
    "dependency_status",
    "final_source_role",
    "final_explicit_relation_type",
    "final_role_confidence",
    "final_role_rationale",
    "disagreement_flag",
    "adjudication_required",
    "adjudicated_role",
    "adjudication_rationale",
    "adjudication_changed_bridge_eligibility",
)
HUMAN_LIST_FIELDS = (
    "blind_discovery_candidates",
    "proposed_parent_source_titles",
    "parent_source_titles",
    "official_support_path_references",
)


class Stage0Error(RuntimeError):
    """A fatal mismatch with the frozen Stage-0 protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage0Error(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"required input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"required input is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0Error(f"cannot read valid JSON from {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"required input is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                require(line.strip() != "", f"blank JSONL line at {path}:{line_number}")
                value = json.loads(line)
                require(
                    isinstance(value, dict),
                    f"non-object JSONL value at {path}:{line_number}",
                )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0Error(f"cannot read valid JSONL from {path}: {exc}") from exc
    return rows


def git(*arguments: str, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )


def latest_git_commit(path: Path) -> str:
    relative = path.relative_to(REPO).as_posix()
    result = git("log", "-1", "--format=%H", "--", relative)
    require(result.returncode == 0, f"git log failed for {relative}: {result.stderr.strip()}")
    commit = result.stdout.strip()
    require(commit != "", f"no Git commit found for {relative}")
    return commit


def validate_freeze_commit() -> None:
    require(
        latest_git_commit(FREEZE_PATH) == FREEZE_COMMIT,
        "Stage-0 freeze latest commit does not match the frozen commit",
    )


def resolve_preparation_commit(dry_run: bool) -> str:
    tracked = git("ls-files", "--error-unmatch", "--", SCRIPT_RELATIVE_PATH)
    if dry_run and tracked.returncode != 0:
        return "UNCOMMITTED_DRY_RUN"
    require(tracked.returncode == 0, "normal mode requires the preparation script to be tracked")

    worktree_diff = git("diff", "--quiet", "--", SCRIPT_RELATIVE_PATH)
    index_diff = git("diff", "--cached", "--quiet", "--", SCRIPT_RELATIVE_PATH)
    script_clean = worktree_diff.returncode == 0 and index_diff.returncode == 0
    if dry_run and not script_clean:
        return "UNCOMMITTED_DRY_RUN"
    require(script_clean, "normal mode requires the preparation script to equal Git")

    status = git("status", "--porcelain=v1")
    require(status.returncode == 0, f"git status failed: {status.stderr.strip()}")
    if dry_run and status.stdout:
        return "UNCOMMITTED_DRY_RUN"
    require(status.stdout == "", "normal mode requires a clean working tree")

    commit = latest_git_commit(SCRIPT_PATH)
    committed = git("show", f"{commit}:{SCRIPT_RELATIVE_PATH}", binary=True)
    require(committed.returncode == 0, "cannot read the committed preparation script")
    require(
        committed.stdout == SCRIPT_PATH.read_bytes(),
        "working preparation script bytes do not equal its recorded Git commit",
    )
    return commit


def validate_and_hash_inputs() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256_file(path)
        require(actual == expected, f"SHA-256 mismatch for {path}: {actual}")
        hashes[path.relative_to(REPO).as_posix()] = actual
    for path in ADDITIONAL_OFFICIAL_INPUTS:
        hashes[path.relative_to(REPO).as_posix()] = sha256_file(path)
    return hashes


def normalize_d2_title(title: Any) -> str:
    return " ".join(str(title).replace("_", " ").strip().lower().split())


def normalize_lexical(value: Any) -> str:
    lowered = str(value).lower()
    without_punctuation = re.sub(r"[^\w\s]", " ", lowered)
    return " ".join(without_punctuation.split()).strip()


def singularize(token: str) -> str:
    return token[:-1] if token.endswith("s") and len(token) > 3 else token


def token_equal(left: str, right: str) -> bool:
    return singularize(left) == singularize(right)


def classify_lexical_trace(question_en: str, gold_title: str) -> dict[str, Any]:
    normalized_question = normalize_lexical(question_en)
    normalized_title = normalize_lexical(gold_title)
    question_tokens = normalized_question.split()
    title_tokens = normalized_title.split()
    require(title_tokens, f"empty lexical title after normalization: {gold_title!r}")

    exact = any(
        all(token_equal(title_token, question_tokens[start + offset])
            for offset, title_token in enumerate(title_tokens))
        for start in range(len(question_tokens) - len(title_tokens) + 1)
    )
    if exact:
        trace_class = "EXACT"
        head = title_tokens[title_tokens.index("of") - 1] if "of" in title_tokens else title_tokens[-1]
    else:
        if "of" in title_tokens:
            of_index = title_tokens.index("of")
            require(of_index > 0, f"title has no token before first 'of': {gold_title!r}")
            head = title_tokens[of_index - 1]
        else:
            head = title_tokens[-1]
        trace_class = (
            "PARTIAL" if any(token_equal(head, token) for token in question_tokens) else "ABSENT"
        )
    return {
        "normalized_question_en": normalized_question,
        "lexical_normalized_gold_title": normalized_title,
        "question_tokens": question_tokens,
        "title_tokens": title_tokens,
        "title_head_token": head,
        "lexical_trace_class": trace_class,
        "lexical_trace_en": "YES" if trace_class in {"EXACT", "PARTIAL"} else "NO",
    }


def source_instance_id(urbench_qid: str, gold_title: str) -> str:
    payload = json.dumps(
        [urbench_qid, gold_title],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "s0-" + hashlib.sha256(payload).hexdigest()


def require_unique_keyed_rows(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        require(isinstance(value, str) and value != "", f"{label} row has invalid {key}")
        require(value not in result, f"duplicate {key} in {label}: {value}")
        result[value] = row
    return result


def validate_official_inputs(
    dev_rows: list[dict[str, Any]],
    mapped_rows: list[dict[str, Any]],
    train_rows: Any,
    official_dev_rows: Any,
) -> None:
    require(len(dev_rows) == 200, f"DEV200 has {len(dev_rows)} rows, expected 200")
    dev_by_qid = require_unique_keyed_rows(dev_rows, "urbench_qid", "DEV200")
    require(len(dev_by_qid) == 200, "DEV200 must have 200 unique qids")

    mapped_by_qid = require_unique_keyed_rows(mapped_rows, "urbench_qid", "mapped official data")
    require(len(mapped_by_qid) == 2290, "mapped official data must contain 2290 qids")
    for qid, row in dev_by_qid.items():
        require(qid in mapped_by_qid, f"DEV200 qid absent from mapped official data: {qid}")
        require(row == mapped_by_qid[qid], f"DEV200 row differs from mapped row: {qid}")

    require(isinstance(train_rows, list) and len(train_rows) == 2061, "official train count mismatch")
    require(
        isinstance(official_dev_rows, list) and len(official_dev_rows) == 229,
        "official dev count mismatch",
    )
    official: dict[str, tuple[str, dict[str, Any]]] = {}
    for source, rows in (("train", train_rows), ("dev", official_dev_rows)):
        for row in rows:
            require(isinstance(row, dict), f"official {source} contains a non-object row")
            raw_qid = row.get("qid")
            require(isinstance(raw_qid, str), f"official {source} row has invalid qid")
            qid = raw_qid.strip()
            require(qid and qid not in official, f"duplicate normalized official qid: {qid}")
            official[qid] = (source, row)
    require(len(official) == 2290, "combined official qid count mismatch")

    for row in dev_rows:
        official_qid = row.get("official_qid")
        require(official_qid in official, f"unknown official qid: {official_qid}")
        source, raw = official[official_qid]
        require(row.get("official_source") == source, f"official source mismatch: {official_qid}")
        require(row.get("official_decomposition") == raw.get("decomposition"),
                f"official decomposition mismatch: {official_qid}")
        require(row.get("official_evidence") == raw.get("evidence"),
                f"official evidence mismatch: {official_qid}")
        require(row.get("question_en") == raw.get("question"),
                f"official English question mismatch: {official_qid}")
        require(row.get("answer") == raw.get("answer"),
                f"official answer mismatch: {official_qid}")


def paragraph_metadata(paragraph_id: str, paragraphs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    require(paragraph_id in paragraphs, f"unknown official paragraph ID: {paragraph_id}")
    require("-" in paragraph_id, f"paragraph ID has no final numeric suffix: {paragraph_id}")
    title, suffix = paragraph_id.rsplit("-", 1)
    require(suffix.isdigit(), f"paragraph ID suffix is not numeric: {paragraph_id}")
    metadata = paragraphs[paragraph_id]
    require(isinstance(metadata, dict), f"paragraph metadata is not an object: {paragraph_id}")
    require(metadata.get("title") == title, f"paragraph title mismatch: {paragraph_id}")
    require(metadata.get("para_index") == int(suffix), f"paragraph index mismatch: {paragraph_id}")
    require(isinstance(metadata.get("headers"), list), f"paragraph headers are not a list: {paragraph_id}")
    require(isinstance(metadata.get("section"), str), f"paragraph section is not a string: {paragraph_id}")
    return title, metadata


def build_sources_and_evidence(
    dev_rows: list[dict[str, Any]], paragraphs: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(isinstance(paragraphs, dict), "paragraph metadata input must be a JSON object")
    sources: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    id_to_pair: dict[str, tuple[str, str]] = {}
    qid_rows = require_unique_keyed_rows(dev_rows, "urbench_qid", "DEV200")

    for qid in sorted(qid_rows):
        row = qid_rows[qid]
        official_qid = row.get("official_qid")
        question_ur = row.get("question_ur")
        decomposition = row.get("official_decomposition")
        evidence = row.get("official_evidence")
        require(isinstance(official_qid, str) and official_qid, f"invalid official qid for {qid}")
        require(isinstance(question_ur, str) and question_ur.strip(), f"invalid Urdu question for {qid}")
        require(isinstance(decomposition, list) and decomposition, f"invalid decomposition for {qid}")
        require(all(isinstance(step, str) for step in decomposition), f"non-string decomposition for {qid}")
        require(isinstance(evidence, list), f"official evidence is not a list for {qid}")
        require(len(evidence) == 3, f"official evidence annotator count is not 3 for {qid}")

        paragraph_ids: list[str] = []
        titles_in_occurrence_order: list[str] = []
        for annotator_index, annotation in enumerate(evidence):
            require(isinstance(annotation, list), f"evidence annotator is not a list for {qid}")
            require(
                len(annotation) == len(decomposition),
                f"evidence/decomposition step mismatch for {qid}, annotator {annotator_index}",
            )
            for step_index, evidence_step in enumerate(annotation):
                require(isinstance(evidence_step, list), f"evidence step is not a group list for {qid}")
                require(evidence_step, f"empty evidence step for {qid}, step {step_index}")
                for group_index, group in enumerate(evidence_step):
                    common = {
                        "artifact_scope": "NOT ANNOTATOR-FACING",
                        "urbench_qid": qid,
                        "official_qid": official_qid,
                        "official_evidence_annotator_index": annotator_index,
                        "decomposition_step_index": step_index,
                        "decomposition_text": decomposition[step_index],
                        "evidence_group_index": group_index,
                    }
                    if isinstance(group, str):
                        require(group in MARKERS, f"unknown scalar evidence marker for {qid}: {group!r}")
                        links.append(
                            {
                                **common,
                                "record_type": "MARKER",
                                "raw_path": [annotator_index, step_index, group_index],
                                "paragraph_occurrence_index": None,
                                "marker": group,
                                "paragraph_id": None,
                                "source_instance_id": None,
                                "gold_title": None,
                                "paragraph_title": None,
                                "paragraph_index": None,
                                "section": None,
                                "headers": None,
                            }
                        )
                        continue

                    require(isinstance(group, list), f"unknown evidence group type for {qid}")
                    require(group, f"empty paragraph evidence group for {qid}")
                    for occurrence_index, paragraph_id in enumerate(group):
                        require(isinstance(paragraph_id, str), f"non-string paragraph ID for {qid}")
                        require(paragraph_id not in MARKERS, f"marker nested inside paragraph group for {qid}")
                        title, metadata = paragraph_metadata(paragraph_id, paragraphs)
                        sid = source_instance_id(qid, title)
                        pair = (qid, title)
                        require(sid not in id_to_pair or id_to_pair[sid] == pair,
                                f"source-instance SHA-256 collision: {sid}")
                        id_to_pair[sid] = pair
                        paragraph_ids.append(paragraph_id)
                        titles_in_occurrence_order.append(title)
                        links.append(
                            {
                                **common,
                                "record_type": "PARAGRAPH",
                                "raw_path": [annotator_index, step_index, group_index, occurrence_index],
                                "paragraph_occurrence_index": occurrence_index,
                                "marker": None,
                                "paragraph_id": paragraph_id,
                                "source_instance_id": sid,
                                "gold_title": title,
                                "paragraph_title": metadata["title"],
                                "paragraph_index": metadata["para_index"],
                                "section": metadata["section"],
                                "headers": list(metadata["headers"]),
                            }
                        )

        frozen_ids = row.get("evidence_paragraph_ids")
        require(isinstance(frozen_ids, list), f"evidence_paragraph_ids is not a list for {qid}")
        require(
            frozen_ids == sorted(set(paragraph_ids)),
            f"nested evidence does not reconstruct evidence_paragraph_ids for {qid}",
        )

        for title in sorted(set(titles_in_occurrence_order)):
            sid = source_instance_id(qid, title)
            pair = (qid, title)
            require(sid not in id_to_pair or id_to_pair[sid] == pair,
                    f"source-instance SHA-256 collision: {sid}")
            id_to_pair[sid] = pair
            official_step_indices = [
                link["decomposition_step_index"]
                for link in links
                if link["record_type"] == "PARAGRAPH" and link["source_instance_id"] == sid
            ]
            master = {
                "artifact_scope": "NOT ANNOTATOR-FACING",
                "source_instance_id": sid,
                "urbench_qid": qid,
                "official_qid": official_qid,
                "question_ur": question_ur,
                "gold_title": title,
                "normalized_gold_title": normalize_d2_title(title),
                "lexical_trace_en": None,
                "exact_corpus_status": None,
                "official_step_indices": official_step_indices,
            }
            master.update({field: None for field in HUMAN_NULL_FIELDS})
            master.update({field: [] for field in HUMAN_LIST_FIELDS})
            sources.append(master)

    require(len(id_to_pair) == len(sources), "source ID/pair mapping is not bijective")
    require(len({row["source_instance_id"] for row in sources}) == len(sources),
            "source instance IDs are not unique")
    require(
        {(row["urbench_qid"], row["gold_title"]) for row in sources}
        == set(id_to_pair.values()),
        "source ID mapping is not an exact pair bijection",
    )
    return sources, links


def assign_exact_title_status(
    sources: list[dict[str, Any]], coverage: Any, recall_split: Any
) -> dict[str, int]:
    require(isinstance(coverage, dict), "D2 title coverage must be an object")
    require(isinstance(recall_split, dict), "D2 recall split must be an object")
    present_raw = coverage.get("present_titles")
    absent_raw = coverage.get("absent_titles")
    require(isinstance(present_raw, list) and all(isinstance(x, str) for x in present_raw),
            "D2 present_titles is malformed")
    require(isinstance(absent_raw, list) and all(isinstance(x, str) for x in absent_raw),
            "D2 absent_titles is malformed")
    present = {normalize_d2_title(title) for title in present_raw}
    absent = {normalize_d2_title(title) for title in absent_raw}
    require(len(present) == 432, f"distinct normalized present title count is {len(present)}")
    require(len(absent) == 181, f"distinct normalized absent title count is {len(absent)}")
    require(not present & absent, "a normalized title is both present and absent")
    normalized_source_titles = {row["normalized_gold_title"] for row in sources}
    require(len(normalized_source_titles) == 613, "distinct normalized source title count is not 613")
    require(normalized_source_titles == present | absent,
            "D2 title partition does not exactly cover normalized gold titles")
    require(coverage.get("gold_titles_required") == 613, "D2 required-title count mismatch")
    require(coverage.get("gold_titles_present") == 432, "D2 present-title count mismatch")
    require(coverage.get("gold_titles_absent") == 181, "D2 absent-title count mismatch")
    require(coverage.get("corpus_unique_titles") == CORPUS_UNIQUE_NORMALIZED_TITLES,
            "D2 corpus unique-title count mismatch")

    status_counts: Counter[str] = Counter()
    qid_statuses: dict[str, list[str]] = defaultdict(list)
    for row in sources:
        normalized = row["normalized_gold_title"]
        memberships = int(normalized in present) + int(normalized in absent)
        require(memberships == 1, f"title lacks exactly one D2 status: {row['gold_title']}")
        status = "EXACT_PRESENT" if normalized in present else "EXACT_ABSENT"
        row["exact_corpus_status"] = status
        status_counts[status] += 1
        qid_statuses[row["urbench_qid"]].append(status)

    coverage_counts = Counter()
    for statuses in qid_statuses.values():
        present_count = statuses.count("EXACT_PRESENT")
        if present_count == len(statuses):
            coverage_counts["full"] += 1
        elif present_count == 0:
            coverage_counts["zero"] += 1
        else:
            coverage_counts["partial"] += 1
    require(status_counts == Counter({"EXACT_PRESENT": 445, "EXACT_ABSENT": 191}),
            f"exact-title instance counts mismatch: {dict(status_counts)}")
    require(sum(status_counts.values()) == 636 and 445 + 191 == 636,
            "exact-title instance arithmetic mismatch")
    require(coverage_counts == Counter({"full": 71, "partial": 113, "zero": 16}),
            f"qid exact-title coverage mismatch: {dict(coverage_counts)}")
    require(sum(coverage_counts.values()) == 200 and 71 + 113 + 16 == 200,
            "qid coverage arithmetic mismatch")

    require(recall_split.get("n_questions") == 200, "D2 recall-split qid count mismatch")
    require(recall_split.get("questions_all_gold_available") == 71,
            "D2 recall-split full-coverage count mismatch")
    require(recall_split.get("questions_no_gold_available") == 16,
            "D2 recall-split zero-coverage count mismatch")
    by_k = recall_split.get("by_k")
    require(isinstance(by_k, dict) and by_k, "D2 recall-split by_k is malformed")
    for k, values in by_k.items():
        require(isinstance(values, dict), f"D2 recall-split k={k} is malformed")
        require(values.get("required_total") == 636, f"D2 k={k} total mismatch")
        require(values.get("required_available") == 445, f"D2 k={k} available mismatch")
        require(values.get("required_missing_from_corpus") == 191,
                f"D2 k={k} absent mismatch")
    return {
        "distinct_present": len(present),
        "distinct_absent": len(absent),
        "present_instances": status_counts["EXACT_PRESENT"],
        "absent_instances": status_counts["EXACT_ABSENT"],
        "full_qids": coverage_counts["full"],
        "partial_qids": coverage_counts["partial"],
        "zero_qids": coverage_counts["zero"],
    }


def assign_lexical_traces(
    sources: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dev_by_qid = require_unique_keyed_rows(dev_rows, "urbench_qid", "DEV200")
    lexical_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    trace_counts: Counter[str] = Counter()
    qid_traces: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        qid = source["urbench_qid"]
        question_en = dev_by_qid[qid].get("question_en")
        require(isinstance(question_en, str) and question_en.strip(), f"invalid English question for {qid}")
        diagnostic = classify_lexical_trace(question_en, source["gold_title"])
        source["lexical_trace_en"] = diagnostic["lexical_trace_en"]
        class_counts[diagnostic["lexical_trace_class"]] += 1
        trace_counts[diagnostic["lexical_trace_en"]] += 1
        qid_traces[qid].append(diagnostic["lexical_trace_en"])
        lexical_rows.append(
            {
                "artifact_scope": "NOT ANNOTATOR-FACING",
                "source_instance_id": source["source_instance_id"],
                "urbench_qid": qid,
                "official_qid": source["official_qid"],
                "question_en": question_en,
                "gold_title": source["gold_title"],
                **diagnostic,
            }
        )
    all_visible_qids = sum(all(value == "YES" for value in values) for values in qid_traces.values())
    require(class_counts == Counter({"EXACT": 256, "PARTIAL": 46, "ABSENT": 334}),
            f"lexical class counts mismatch: {dict(class_counts)}")
    require(trace_counts == Counter({"YES": 302, "NO": 334}),
            f"lexical YES/NO counts mismatch: {dict(trace_counts)}")
    require(256 + 46 + 334 == 636 and 302 + 334 == 636,
            "lexical arithmetic mismatch")
    require(all_visible_qids == 39, f"all-visible qid count is {all_visible_qids}, expected 39")
    return lexical_rows, {
        "exact": class_counts["EXACT"],
        "partial": class_counts["PARTIAL"],
        "absent": class_counts["ABSENT"],
        "yes": trace_counts["YES"],
        "no": trace_counts["NO"],
        "all_visible_qids": all_visible_qids,
    }


def largest_remainder_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    require(total > 0, "cannot allocate a sample from an empty population")
    quotas = {stratum: sample_size * count // total for stratum, count in counts.items()}
    remainders = {stratum: (sample_size * count) % total for stratum, count in counts.items()}
    slots = sample_size - sum(quotas.values())
    ranked = sorted(counts, key=lambda stratum: (-remainders[stratum], stratum))
    require(0 <= slots <= len(ranked), "invalid largest-remainder slot count")
    for stratum in ranked[:slots]:
        quotas[stratum] += 1
    return quotas


def build_reliability_manifest(
    sources: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    strata: dict[str, list[str]] = {stratum: [] for stratum in STRATUM_ORDER}
    source_by_id = {row["source_instance_id"]: row for row in sources}
    for row in sources:
        stratum = f"{row['lexical_trace_en']}__{row['exact_corpus_status']}"
        require(stratum in strata, f"unknown reliability stratum: {stratum}")
        strata[stratum].append(row["source_instance_id"])
    counts = {stratum: len(strata[stratum]) for stratum in STRATUM_ORDER}
    require(counts == EXPECTED_STRATUM_COUNTS, f"reliability stratum counts mismatch: {counts}")
    require(sum(counts.values()) == 636 and 83 + 251 + 108 + 194 == 636,
            "reliability stratum arithmetic mismatch")

    quotas = largest_remainder_quotas(counts, RELIABILITY_SAMPLE_SIZE)
    require(quotas == EXPECTED_STRATUM_QUOTAS, f"reliability quotas mismatch: {quotas}")
    require(sum(quotas.values()) == 160 and 21 + 63 + 27 + 49 == 160,
            "reliability quota arithmetic mismatch")

    rng = random.Random(SEED)
    sampled: list[dict[str, Any]] = []
    global_draw = 0
    for stratum in STRATUM_ORDER:
        population = sorted(strata[stratum])
        draws = rng.sample(population, quotas[stratum])
        for stratum_draw, sid in enumerate(draws, 1):
            global_draw += 1
            sampled.append(
                {
                    "draw_order": global_draw,
                    "stratum_draw_order": stratum_draw,
                    "stratum": stratum,
                    "source_instance_id": sid,
                    "urbench_qid": source_by_id[sid]["urbench_qid"],
                }
            )
    require(len(sampled) == 160, "reliability sample does not contain 160 rows")
    require(len({row["source_instance_id"] for row in sampled}) == 160,
            "reliability sample contains duplicate source IDs")
    require(all(row["source_instance_id"] in source_by_id for row in sampled),
            "reliability sample contains an unknown source ID")
    return {
        "artifact_scope": "NOT ANNOTATOR-FACING",
        "seed": SEED,
        "sample_size": RELIABILITY_SAMPLE_SIZE,
        "population_size": len(sources),
        "allocation_algorithm": "proportional_largest_remainder_v1",
        "remainder_tie_break": "lexical_stratum_name",
        "sampling_algorithm": "python_random_sample_sorted_source_ids_v1",
        "stratum_order": list(STRATUM_ORDER),
        "stratum_counts": counts,
        "stratum_quotas": quotas,
        "sampled_instances": sampled,
    }, counts


def build_pass1_manifest(dev_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = [
        {"urbench_qid": row["urbench_qid"], "question_ur": row["question_ur"]}
        for row in sorted(dev_rows, key=lambda value: value["urbench_qid"])
    ]
    require(len(rows) == 200, "Pass-1 manifest does not contain 200 rows")
    require(len({row["urbench_qid"] for row in rows}) == 200,
            "Pass-1 manifest does not contain 200 unique qids")
    require(all(set(row) == {"urbench_qid", "question_ur"} for row in rows),
            "Pass-1 manifest contains a forbidden field")
    return rows


def schedule_pass2(grouped_ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    # This scheduler depends only on qid and source_instance_id.
    shuffled_ids: dict[str, list[str]] = {}
    rng = random.Random(SEED)
    for qid in sorted(grouped_ids):
        ids = sorted(grouped_ids[qid])
        rng.shuffle(ids)
        shuffled_ids[qid] = ids
    shuffled_qids = sorted(shuffled_ids)
    rng.shuffle(shuffled_qids)
    tie_priority = {qid: priority for priority, qid in enumerate(shuffled_qids)}

    heap: list[tuple[int, int, str, int]] = [
        (-len(shuffled_ids[qid]), tie_priority[qid], qid, 0)
        for qid in shuffled_ids
    ]
    heapq.heapify(heap)
    held: tuple[int, int, str, int] | None = None
    order: list[dict[str, Any]] = []
    while heap or held is not None:
        if not heap:
            require(held is not None and held[0] == 0,
                    "Pass-2 scheduler cannot avoid a consecutive same-qid collision")
            break
        neg_remaining, priority, qid, cursor = heapq.heappop(heap)
        if held is not None and held[0] < 0:
            heapq.heappush(heap, held)
        sid = shuffled_ids[qid][cursor]
        order.append(
            {
                "rank": len(order) + 1,
                "urbench_qid": qid,
                "source_instance_id": sid,
            }
        )
        next_cursor = cursor + 1
        next_remaining = len(shuffled_ids[qid]) - next_cursor
        held = (-next_remaining, priority, qid, next_cursor)
    return order


def build_pass2_manifest(sources: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in sources:
        grouped[row["urbench_qid"]].append(row["source_instance_id"])
    maximum = max(map(len, grouped.values()))
    require(maximum == 8, f"maximum qid multiplicity is {maximum}, expected 8")
    require(maximum <= (len(sources) - maximum) + 1, "Pass-2 zero-collision schedule is infeasible")
    order = schedule_pass2(dict(grouped))
    expected_ids = {row["source_instance_id"] for row in sources}
    require(len(order) == 636, f"Pass-2 order has {len(order)} positions")
    require([row["rank"] for row in order] == list(range(1, 637)),
            "Pass-2 ranks are not exactly 1..636")
    ordered_ids = [row["source_instance_id"] for row in order]
    require(len(set(ordered_ids)) == 636 and set(ordered_ids) == expected_ids,
            "Pass-2 order is not an exact source-ID permutation")
    require(all(left["urbench_qid"] != right["urbench_qid"]
                for left, right in zip(order, order[1:])),
            "Pass-2 order contains adjacent instances from the same qid")
    return {
        "seed": SEED,
        "population_size": len(sources),
        "maximum_qid_multiplicity": maximum,
        "scheduler_algorithm": "max_remaining_one_iteration_holdout_v1",
        "ordering_inputs_only": ["urbench_qid", "source_instance_id"],
        "items": order,
    }


def human_annotation_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    nullable_timestamp = {"type": ["string", "null"], "format": "date-time"}
    nullable_round = {"type": ["integer", "null"], "minimum": 1}
    nullable_boolean = {"type": ["boolean", "null"]}
    source_role = {"enum": [None, "EXPLICIT", "LATENT_BRIDGE", "AMBIGUOUS"]}
    explicit_relation_type = {
        "enum": [
            None,
            "DIRECT_MENTION",
            "TRANSLITERATION",
            "COMMON_ALIAS_OR_ABBREVIATION",
            "MORPHOLOGICAL_OR_DEMONYM",
            "DIRECT_SPECIFIC_CONCEPT",
        ]
    }
    confidence = {"enum": [None, "HIGH", "MEDIUM", "LOW"]}
    dependency_status = {
        "enum": [
            None,
            "CLEAR_DEPENDENCY",
            "MULTIPLE_PLAUSIBLE_PARENTS",
            "PARALLEL_OR_UNORDERED",
            "UNRESOLVED",
            "NOT_APPLICABLE",
        ]
    }
    string_list = {"type": "array", "items": {"type": "string"}}
    answer_inspection_event = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason", "viewer", "timestamp", "round", "changed_decision"],
        "properties": {
            "reason": {"type": "string", "minLength": 1},
            "viewer": {"type": "string", "minLength": 1},
            "timestamp": nullable_timestamp,
            "round": nullable_round,
            "changed_decision": {"type": "boolean"},
        },
        "allOf": [
            {
                "anyOf": [
                    {"properties": {"timestamp": {"type": "string"}}},
                    {"properties": {"round": {"type": "integer"}}},
                ]
            }
        ],
    }
    properties = {
        "source_instance_id": {"type": "string", "pattern": "^s0-[0-9a-f]{64}$"},
        "annotator_id": {"type": "string", "minLength": 1},
        "pass1_completed": {
            "type": "boolean",
            "description": "False means Pass 1 has not been completed; true freezes its candidates.",
        },
        "blind_discovery_candidates": string_list,
        "pass1_timestamp": nullable_timestamp,
        "pass1_round": nullable_round,
        "pass2_decision": {
            "description": "Null means unfinished Pass 2 and never means NOT_YET_EXPLICIT.",
            "enum": [None, "EXPLICIT", "NOT_YET_EXPLICIT"],
        },
        "pass2_timestamp": nullable_timestamp,
        "pass2_round": nullable_round,
        "pass3_decision": {
            "description": "Null means Pass 3 has not been completed or was not applicable.",
            "enum": [None, "LATENT_BRIDGE", "AMBIGUOUS"],
        },
        "pass3_timestamp": nullable_timestamp,
        "pass3_round": nullable_round,
        # The following remain the annotator's original pre-adjudication record.
        "source_role": {
            **source_role,
            "description": "Original annotator role; never overwritten by adjudication.",
        },
        "explicit_relation_type": explicit_relation_type,
        "urdu_span_if_explicit": nullable_string,
        "confidence": confidence,
        "rationale": nullable_string,
        "concrete_intermediate_information": nullable_string,
        "dependency_status": dependency_status,
        "proposed_parent_source_titles": string_list,
        "parent_source_titles": {
            **string_list,
            "description": "Finalized human parent titles, when defensible under the freeze.",
        },
        "official_support_path_references": string_list,
        "disagreement_flag": nullable_boolean,
        "adjudication_required": nullable_boolean,
        "adjudicated_role": {
            **source_role,
            "description": "Adjudicated role stored separately from the original annotator role.",
        },
        "adjudication_rationale": nullable_string,
        "adjudication_changed_bridge_eligibility": nullable_boolean,
        "adjudicator_id": nullable_string,
        "adjudication_timestamp": nullable_timestamp,
        "adjudication_round": nullable_round,
        "answer_inspection_exception_used": {
            **nullable_boolean,
            "description": "Null means adjudication is unfinished; false/true records exception use.",
        },
        "answer_inspection_exceptions": {
            "type": "array",
            "items": answer_inspection_event,
        },
        "annotation_timestamp": nullable_timestamp,
        "round": nullable_round,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "human_annotation_log.schema.json",
        "title": "EFBPT Stage-0 append-only human annotation log row",
        "description": (
            "One independent source_instance_id x annotator record with lossless "
            "Pass-1, Pass-2, Pass-3, and adjudication state; preparation creates no log rows."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
        "allOf": [
            {
                "if": {
                    "properties": {"pass1_completed": {"const": False}},
                    "required": ["pass1_completed"],
                },
                "then": {
                    "properties": {
                        "blind_discovery_candidates": {"maxItems": 0},
                        "pass1_timestamp": {"type": "null"},
                        "pass1_round": {"type": "null"},
                        "pass2_decision": {"type": "null"},
                        "pass3_decision": {"type": "null"},
                        "source_role": {"type": "null"},
                    }
                },
            },
            {
                "if": {
                    "properties": {"pass2_decision": {"type": "null"}},
                    "required": ["pass2_decision"],
                },
                "then": {
                    "properties": {
                        "pass2_timestamp": {"type": "null"},
                        "pass2_round": {"type": "null"},
                        "pass3_decision": {"type": "null"},
                        "source_role": {"type": "null"},
                        "explicit_relation_type": {"type": "null"},
                        "urdu_span_if_explicit": {"type": "null"},
                        "confidence": {"type": "null"},
                        "rationale": {"type": "null"},
                        "concrete_intermediate_information": {"type": "null"},
                        "dependency_status": {"type": "null"},
                        "proposed_parent_source_titles": {"maxItems": 0},
                        "parent_source_titles": {"maxItems": 0},
                        "official_support_path_references": {"maxItems": 0},
                    }
                },
            },
            {
                "if": {
                    "properties": {"pass2_decision": {"const": "EXPLICIT"}},
                    "required": ["pass2_decision"],
                },
                "then": {
                    "properties": {
                        "pass3_decision": {"type": "null"},
                        "source_role": {"const": "EXPLICIT"},
                        "explicit_relation_type": {
                            "enum": [
                                "DIRECT_MENTION",
                                "TRANSLITERATION",
                                "COMMON_ALIAS_OR_ABBREVIATION",
                                "MORPHOLOGICAL_OR_DEMONYM",
                                "DIRECT_SPECIFIC_CONCEPT",
                            ]
                        },
                        "confidence": {"enum": ["HIGH", "MEDIUM", "LOW"]},
                        "rationale": {"type": "string", "minLength": 1},
                    }
                },
            },
            {
                "if": {
                    "properties": {"pass2_decision": {"const": "NOT_YET_EXPLICIT"}},
                    "required": ["pass2_decision"],
                },
                "then": {
                    "properties": {
                        "explicit_relation_type": {"type": "null"},
                        "urdu_span_if_explicit": {"type": "null"},
                    }
                },
            },
            {
                "if": {
                    "allOf": [
                        {
                            "properties": {
                                "pass2_decision": {"const": "NOT_YET_EXPLICIT"}
                            },
                            "required": ["pass2_decision"],
                        },
                        {
                            "properties": {"pass3_decision": {"type": "null"}},
                            "required": ["pass3_decision"],
                        },
                    ]
                },
                "then": {
                    "properties": {
                        "source_role": {"type": "null"},
                        "confidence": {"type": "null"},
                        "rationale": {"type": "null"},
                        "concrete_intermediate_information": {"type": "null"},
                        "dependency_status": {"type": "null"},
                        "proposed_parent_source_titles": {"maxItems": 0},
                        "parent_source_titles": {"maxItems": 0},
                        "official_support_path_references": {"maxItems": 0},
                    }
                },
            },
            {
                "if": {
                    "properties": {"pass3_decision": {"type": "null"}},
                    "required": ["pass3_decision"],
                },
                "then": {
                    "properties": {
                        "pass3_timestamp": {"type": "null"},
                        "pass3_round": {"type": "null"},
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "pass3_decision": {"enum": ["LATENT_BRIDGE", "AMBIGUOUS"]}
                    },
                    "required": ["pass3_decision"],
                },
                "then": {
                    "properties": {
                        "pass2_decision": {"const": "NOT_YET_EXPLICIT"},
                        "confidence": {"enum": ["HIGH", "MEDIUM", "LOW"]},
                        "rationale": {"type": "string", "minLength": 1},
                    }
                },
            },
            {
                "if": {
                    "properties": {"pass3_decision": {"const": "LATENT_BRIDGE"}},
                    "required": ["pass3_decision"],
                },
                "then": {
                    "properties": {
                        "source_role": {"const": "LATENT_BRIDGE"},
                        "concrete_intermediate_information": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "dependency_status": {
                            "enum": [
                                "CLEAR_DEPENDENCY",
                                "MULTIPLE_PLAUSIBLE_PARENTS",
                                "PARALLEL_OR_UNORDERED",
                                "UNRESOLVED",
                                "NOT_APPLICABLE",
                            ]
                        },
                        "official_support_path_references": {"minItems": 1},
                    }
                },
            },
            {
                "if": {
                    "properties": {"pass3_decision": {"const": "AMBIGUOUS"}},
                    "required": ["pass3_decision"],
                },
                "then": {"properties": {"source_role": {"const": "AMBIGUOUS"}}},
            },
            {
                "if": {
                    "properties": {
                        "adjudicated_role": {
                            "enum": ["EXPLICIT", "LATENT_BRIDGE", "AMBIGUOUS"]
                        }
                    },
                    "required": ["adjudicated_role"],
                },
                "then": {
                    "properties": {
                        "adjudication_required": {"const": True},
                        "adjudication_rationale": {"type": "string", "minLength": 1},
                        "adjudication_changed_bridge_eligibility": {"type": "boolean"},
                        "adjudicator_id": {"type": "string", "minLength": 1},
                    },
                    "anyOf": [
                        {"properties": {"adjudication_timestamp": {"type": "string"}}},
                        {"properties": {"adjudication_round": {"type": "integer"}}},
                    ],
                },
            },
            {
                "if": {
                    "properties": {"adjudication_required": {"const": False}},
                    "required": ["adjudication_required"],
                },
                "then": {
                    "properties": {
                        "adjudicated_role": {"type": "null"},
                        "adjudication_rationale": {"type": "null"},
                        "adjudication_changed_bridge_eligibility": {"type": "null"},
                        "adjudicator_id": {"type": "null"},
                        "adjudication_timestamp": {"type": "null"},
                        "adjudication_round": {"type": "null"},
                        "answer_inspection_exception_used": {"const": False},
                    }
                },
            },
            {
                "if": {
                    "properties": {"answer_inspection_exception_used": {"const": True}},
                    "required": ["answer_inspection_exception_used"],
                },
                "then": {
                    "properties": {
                        "adjudication_required": {"const": True},
                        "answer_inspection_exceptions": {"minItems": 1},
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "answer_inspection_exception_used": {"enum": [None, False]}
                    },
                    "required": ["answer_inspection_exception_used"],
                },
                "then": {"properties": {"answer_inspection_exceptions": {"maxItems": 0}}},
            },
        ],
    }


def reconstruct_evidence_from_links(rows: list[dict[str, Any]]) -> list[Any]:
    require(rows, "cannot reconstruct empty official evidence")
    max_annotator = max(row["raw_path"][0] for row in rows)
    reconstructed: list[Any] = []
    for annotator_index in range(max_annotator + 1):
        annotation_rows = [row for row in rows if row["raw_path"][0] == annotator_index]
        require(annotation_rows, f"missing evidence annotator path {annotator_index}")
        max_step = max(row["raw_path"][1] for row in annotation_rows)
        annotation: list[Any] = []
        for step_index in range(max_step + 1):
            step_rows = [row for row in annotation_rows if row["raw_path"][1] == step_index]
            require(step_rows, f"missing evidence step path {annotator_index}/{step_index}")
            max_group = max(row["raw_path"][2] for row in step_rows)
            groups: list[Any] = [None] * (max_group + 1)
            for group_index in range(max_group + 1):
                group_rows = [row for row in step_rows if row["raw_path"][2] == group_index]
                require(group_rows, f"missing evidence group path {annotator_index}/{step_index}/{group_index}")
                types = {row["record_type"] for row in group_rows}
                require(len(types) == 1, "evidence group mixes marker and paragraph rows")
                if types == {"MARKER"}:
                    require(len(group_rows) == 1 and len(group_rows[0]["raw_path"]) == 3,
                            "marker evidence path is malformed")
                    groups[group_index] = group_rows[0]["marker"]
                else:
                    require(types == {"PARAGRAPH"}, "unknown evidence link record type")
                    ordered = sorted(group_rows, key=lambda row: row["raw_path"][3])
                    require([row["raw_path"][3] for row in ordered] == list(range(len(ordered))),
                            "paragraph occurrence paths are not contiguous")
                    groups[group_index] = [row["paragraph_id"] for row in ordered]
            annotation.append(groups)
        reconstructed.append(annotation)
    return reconstructed


def assert_human_fields_blank(sources: list[dict[str, Any]]) -> None:
    for row in sources:
        for field in HUMAN_NULL_FIELDS:
            require(row.get(field) is None, f"human scalar field was initialized: {field}")
        for field in HUMAN_LIST_FIELDS:
            require(row.get(field) == [], f"human list field was initialized: {field}")


def recursively_forbidden_keys(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(key for key in value if key in forbidden)
        for child in value.values():
            found.update(recursively_forbidden_keys(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(recursively_forbidden_keys(child, forbidden))
    return found


def validate_prepared(prepared: dict[str, Any], dev_rows: list[dict[str, Any]]) -> None:
    sources = prepared["source_instance_master.jsonl"]
    links = prepared["official_evidence_links.jsonl"]
    lexical = prepared["lexical_trace_instances.jsonl"]
    pass1 = prepared["pass1_question_manifest.jsonl"]
    pass2 = prepared["pass2_order_manifest.json"]
    reliability = prepared["reliability160_manifest.json"]
    schema = prepared["human_annotation_log.schema.json"]

    require(len(sources) == 636, f"source master has {len(sources)} rows")
    require(len({row["source_instance_id"] for row in sources}) == 636,
            "source master IDs are not unique")
    require(len({(row["urbench_qid"], row["gold_title"]) for row in sources}) == 636,
            "source master qid/title pairs are not unique")
    require(len({row["gold_title"] for row in sources}) == 613,
            "globally distinct raw title count is not 613")
    require(len({row["normalized_gold_title"] for row in sources}) == 613,
            "globally distinct normalized title count is not 613")
    for row in sources:
        require(row["source_instance_id"] == source_instance_id(row["urbench_qid"], row["gold_title"]),
                "source instance ID does not match its exact frozen payload")
        require(row["artifact_scope"] == "NOT ANNOTATOR-FACING",
                "source master lacks its visibility warning")
    assert_human_fields_blank(sources)

    link_counts = Counter(row["record_type"] for row in links)
    marker_counts = Counter(row["marker"] for row in links if row["record_type"] == "MARKER")
    require(link_counts == Counter({"PARAGRAPH": 1367, "MARKER": 837}),
            f"evidence link counts mismatch: {dict(link_counts)}")
    require(len(links) == 2204 and 1367 + 837 == 2204, "evidence row arithmetic mismatch")
    require(marker_counts == Counter({"operation": 528, "no_evidence": 309}),
            f"marker split mismatch: {dict(marker_counts)}")
    require(528 + 309 == 837, "marker arithmetic mismatch")
    annotator_steps = {
        (row["urbench_qid"], row["official_evidence_annotator_index"], row["decomposition_step_index"])
        for row in links
    }
    require(len(annotator_steps) == 1755, f"annotator-step cell count is {len(annotator_steps)}")
    annotators_by_qid: dict[str, set[int]] = defaultdict(set)
    for row in links:
        annotators_by_qid[row["urbench_qid"]].add(row["official_evidence_annotator_index"])
        require(row["artifact_scope"] == "NOT ANNOTATOR-FACING",
                "evidence link lacks its visibility warning")
        require("content" not in row, "paragraph content entered evidence links")
    require(len(annotators_by_qid) == 200, "evidence links do not cover 200 qids")
    require(all(indices == {0, 1, 2} for indices in annotators_by_qid.values()),
            "not every qid has exactly three evidence annotators")
    dev_by_qid = require_unique_keyed_rows(dev_rows, "urbench_qid", "DEV200")
    links_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in links:
        links_by_qid[row["urbench_qid"]].append(row)
    for qid, original in dev_by_qid.items():
        require(reconstruct_evidence_from_links(links_by_qid[qid]) == original["official_evidence"],
                f"raw_path reconstruction failed for {qid}")

    require(len(lexical) == 636, "lexical trace artifact does not contain 636 rows")
    require(Counter(row["lexical_trace_class"] for row in lexical)
            == Counter({"EXACT": 256, "PARTIAL": 46, "ABSENT": 334}),
            "lexical trace class validation failed")
    require(Counter(row["lexical_trace_en"] for row in lexical)
            == Counter({"YES": 302, "NO": 334}),
            "lexical trace YES/NO validation failed")
    qid_traces: dict[str, list[str]] = defaultdict(list)
    for row in lexical:
        qid_traces[row["urbench_qid"]].append(row["lexical_trace_en"])
        require(row["artifact_scope"] == "NOT ANNOTATOR-FACING",
                "lexical trace lacks its visibility warning")
    require(sum(all(item == "YES" for item in items) for items in qid_traces.values()) == 39,
            "lexical all-visible qid validation failed")

    require(len(pass1) == 200 and len({row["urbench_qid"] for row in pass1}) == 200,
            "Pass-1 validation failed")
    require(all(set(row) == {"urbench_qid", "question_ur"} for row in pass1),
            "Pass-1 contains non-allowed fields")
    require([row["urbench_qid"] for row in pass1] == sorted(row["urbench_qid"] for row in pass1),
            "Pass-1 qids are not sorted")

    order = pass2.get("items")
    require(isinstance(order, list) and len(order) == 636, "Pass-2 item count mismatch")
    require([row["rank"] for row in order] == list(range(1, 637)), "Pass-2 rank validation failed")
    source_ids = {row["source_instance_id"] for row in sources}
    ordered_ids = [row["source_instance_id"] for row in order]
    require(len(set(ordered_ids)) == 636 and set(ordered_ids) == source_ids,
            "Pass-2 source-ID permutation validation failed")
    require(all(a["urbench_qid"] != b["urbench_qid"] for a, b in zip(order, order[1:])),
            "Pass-2 adjacent-qid validation failed")
    require(pass2.get("ordering_inputs_only") == ["urbench_qid", "source_instance_id"],
            "Pass-2 ordering input declaration changed")

    require(reliability.get("stratum_counts") == EXPECTED_STRATUM_COUNTS,
            "reliability stratum validation failed")
    require(reliability.get("stratum_quotas") == EXPECTED_STRATUM_QUOTAS,
            "reliability quota validation failed")
    sample = reliability.get("sampled_instances")
    require(isinstance(sample, list) and len(sample) == 160, "reliability sample validation failed")
    require(len({row["source_instance_id"] for row in sample}) == 160,
            "reliability sampled IDs are not unique")
    require(all(row["source_instance_id"] in source_ids for row in sample),
            "reliability sample contains an unknown source ID")

    require(schema.get("additionalProperties") is False, "human log schema must reject extra fields")
    require("ABSTAIN" not in json.dumps(schema, ensure_ascii=False), "schema contains forbidden ABSTAIN")
    require("CORPUS_ABSENT" not in json.dumps(schema, ensure_ascii=False),
            "schema contains forbidden CORPUS_ABSENT")

    annotation_facing = {"pass1": pass1, "pass2": pass2}
    forbidden_keys = {
        "answer",
        "gold_answer",
        "question_en",
        "official_facts",
        "urbench_facts",
        "d4_facts",
        "paragraph_content",
        "exact_corpus_status",
        "lexical_trace_en",
        "source_role",
        "dependency_status",
        "retrieval_hop",
    }
    require(not recursively_forbidden_keys(annotation_facing, forbidden_keys),
            "forbidden data entered an annotation-facing manifest")
    all_serialized = json.dumps(prepared, ensure_ascii=False)
    require("ABSTAIN" not in all_serialized, "a prepared artifact contains forbidden ABSTAIN")
    require("CORPUS_ABSENT" not in all_serialized,
            "a prepared artifact contains forbidden CORPUS_ABSENT")
    require("retrieval_hop" not in all_serialized,
            "a retrieval hop was inferred from official decomposition")


def build_prepared_structures(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dev_rows = inputs["dev_rows"]
    sources, links = build_sources_and_evidence(dev_rows, inputs["paragraphs"])
    exact_counts = assign_exact_title_status(sources, inputs["coverage"], inputs["recall_split"])
    lexical_rows, lexical_counts = assign_lexical_traces(sources, dev_rows)
    reliability, stratum_counts = build_reliability_manifest(sources)
    pass1 = build_pass1_manifest(dev_rows)
    pass2 = build_pass2_manifest(sources)
    prepared = {
        "source_instance_master.jsonl": sources,
        "official_evidence_links.jsonl": links,
        "lexical_trace_instances.jsonl": lexical_rows,
        "pass1_question_manifest.jsonl": pass1,
        "pass2_order_manifest.json": pass2,
        "reliability160_manifest.json": reliability,
        "human_annotation_log.schema.json": human_annotation_schema(),
    }
    validate_prepared(prepared, dev_rows)
    counts = {
        "dev200_rows": len(dev_rows),
        "dev200_unique_qids": len({row["urbench_qid"] for row in dev_rows}),
        "source_instances": len(sources),
        "distinct_raw_titles": len({row["gold_title"] for row in sources}),
        "distinct_normalized_titles": len({row["normalized_gold_title"] for row in sources}),
        "evidence_annotators_per_qid": 3,
        "annotator_step_cells": 1755,
        "paragraph_occurrence_rows": 1367,
        "marker_rows": 837,
        "operation_markers": 528,
        "no_evidence_markers": 309,
        "evidence_link_rows": len(links),
        "exact_title": exact_counts,
        "lexical_trace": lexical_counts,
        "reliability_strata": stratum_counts,
        "reliability_quotas": reliability["stratum_quotas"],
        "reliability_sample": len(reliability["sampled_instances"]),
        "pass1_qids": len(pass1),
        "pass2_unique_ids": len({row["source_instance_id"] for row in pass2["items"]}),
        "pass2_adjacent_same_qid": sum(
            left["urbench_qid"] == right["urbench_qid"]
            for left, right in zip(pass2["items"], pass2["items"][1:])
        ),
    }
    return prepared, counts


def preparation_summary(
    counts: dict[str, Any], preparation_commit: str
) -> dict[str, Any]:
    return {
        "artifact_scope": "NOT ANNOTATOR-FACING",
        "freeze_commit": FREEZE_COMMIT,
        "preparation_commit": preparation_commit,
        "dataset_status": "DEV200_SPENT_DEVELOPMENT_DATA",
        "counts": counts,
        "human_annotation_log_created": False,
        "human_judgments_inferred": False,
        "retrieval_hop_inferred": False,
        "canonical_artifacts": list(ARTIFACT_NAMES),
    }


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def serialize_artifacts(prepared: dict[str, Any]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name, value in prepared.items():
        payloads[name] = jsonl_bytes(value) if name in JSONL_ARTIFACTS else json_bytes(value)
    return payloads


def decode_artifact(name: str, payload: bytes) -> Any:
    text = payload.decode("utf-8")
    if name in JSONL_ARTIFACTS:
        require(text.endswith("\n"), f"JSONL artifact lacks final newline: {name}")
        return [json.loads(line) for line in text.splitlines()]
    return json.loads(text)


def revalidate_serialized(
    payloads: dict[str, bytes], expected: dict[str, Any], dev_rows: list[dict[str, Any]]
) -> None:
    decoded: dict[str, Any] = {}
    for name, value in expected.items():
        require(name in payloads, f"serialized artifact is missing: {name}")
        decoded[name] = decode_artifact(name, payloads[name])
        require(decoded[name] == value, f"serialized artifact changed in memory: {name}")
    core = {name: decoded[name] for name in expected if name != "stage0_preparation_summary.json"}
    validate_prepared(core, dev_rows)


def reproducibility_manifest(
    input_hashes: dict[str, str],
    preparation_commit: str,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    required_paths = {path.relative_to(REPO).as_posix() for path in EXPECTED_SHA256}
    additional_paths = {path.relative_to(REPO).as_posix() for path in ADDITIONAL_OFFICIAL_INPUTS}
    return {
        "artifact_scope": "NOT ANNOTATOR-FACING",
        "freeze_commit": FREEZE_COMMIT,
        "preparation_script_path": SCRIPT_RELATIVE_PATH,
        "preparation_script_commit": preparation_commit,
        "required_input_sha256": {
            path: input_hashes[path] for path in sorted(required_paths)
        },
        "additional_official_input_sha256": {
            path: input_hashes[path] for path in sorted(additional_paths)
        },
        "corpus_metadata_path": CORPUS_METADATA_RELATIVE_PATH,
        "unique_normalized_corpus_title_count": CORPUS_UNIQUE_NORMALIZED_TITLES,
        "seed": SEED,
        "generation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sys_version": sys.version,
        "unicode_database_version": unicodedata.unidata_version,
        "sampling_algorithm_version": "proportional_largest_remainder_then_python_random_sample_v1",
        "pass2_scheduler_algorithm_version": "max_remaining_one_iteration_holdout_v1",
        "stratum_order": list(STRATUM_ORDER),
        "prng_call_order": {
            "reliability160": [
                "fresh random.Random(20260822)",
                "for each frozen stratum: sort source_instance_id, then sample(quota)",
            ],
            "pass2": [
                "fresh random.Random(20260822)",
                "for each sorted qid: sort source_instance_id, then shuffle list",
                "sort qids, then shuffle qid list once for tie priority",
            ],
        },
        "artifact_sha256": artifact_hashes,
    }


def write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    # Linux renameat2(RENAME_NOREPLACE) provides atomic publication without an
    # overwrite race between the existence check and directory rename.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "atomic no-replace directory rename is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        raise Stage0Error(
            f"atomic publication failed for {destination}: {os.strerror(error_number)}"
        )


def publish_normal_mode(
    prepared: dict[str, Any],
    dev_rows: list[dict[str, Any]],
    input_hashes: dict[str, str],
    preparation_commit: str,
) -> None:
    require(not OUTPUT_DIR.exists(), f"canonical Stage-0 output already exists: {OUTPUT_DIR}")
    require(OUTPUT_DIR.parent.is_dir(), f"canonical output parent is missing: {OUTPUT_DIR.parent}")
    temporary = Path(tempfile.mkdtemp(prefix=".stage0-preparing-", dir=OUTPUT_DIR.parent))
    published = False
    try:
        payloads = serialize_artifacts(prepared)
        for name, payload in payloads.items():
            write_fsynced(temporary / name, payload)
        fsync_directory(temporary)

        reread = {name: (temporary / name).read_bytes() for name in payloads}
        revalidate_serialized(reread, prepared, dev_rows)
        artifact_hashes = {name: sha256_bytes(reread[name]) for name in sorted(reread)}

        manifest = reproducibility_manifest(input_hashes, preparation_commit, artifact_hashes)
        manifest_payload = json_bytes(manifest)
        write_fsynced(temporary / "reproducibility_manifest.json", manifest_payload)
        fsync_directory(temporary)

        final_payloads = {
            name: (temporary / name).read_bytes()
            for name in ARTIFACT_NAMES
        }
        require(decode_artifact("reproducibility_manifest.json", final_payloads["reproducibility_manifest.json"])
                == manifest, "reproducibility manifest reread validation failed")
        revalidate_serialized(final_payloads, prepared, dev_rows)
        for name, expected_hash in artifact_hashes.items():
            require(sha256_bytes(final_payloads[name]) == expected_hash,
                    f"post-write artifact hash mismatch: {name}")

        require(not OUTPUT_DIR.exists(), f"canonical Stage-0 output appeared during preparation: {OUTPUT_DIR}")
        atomic_rename_directory_noreplace(temporary, OUTPUT_DIR)
        published = True
        fsync_directory(OUTPUT_DIR.parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def load_all_inputs() -> dict[str, Any]:
    dev_rows = read_jsonl(DEV200_PATH)
    mapped_rows = read_jsonl(MAPPED_PATH)
    train_rows = read_json(OFFICIAL_TRAIN_PATH)
    official_dev_rows = read_json(OFFICIAL_DEV_PATH)
    paragraphs = read_json(PARAGRAPHS_PATH)
    validate_official_inputs(dev_rows, mapped_rows, train_rows, official_dev_rows)
    return {
        "dev_rows": dev_rows,
        "mapped_rows": mapped_rows,
        "train_rows": train_rows,
        "official_dev_rows": official_dev_rows,
        "paragraphs": paragraphs,
        "coverage": read_json(D2_COVERAGE_PATH),
        "recall_split": read_json(D2_RECALL_SPLIT_PATH),
    }


def print_validation_summary(counts: dict[str, Any], preparation_commit: str) -> None:
    exact = counts["exact_title"]
    lexical = counts["lexical_trace"]
    strata = counts["reliability_strata"]
    quotas = counts["reliability_quotas"]
    print("EFBPT STAGE-0 PREPARATION VALIDATION")
    print(f"PREPARATION_COMMIT = {preparation_commit}")
    print(f"DEV200: rows={counts['dev200_rows']} unique_qids={counts['dev200_unique_qids']}")
    print(
        "SOURCE INSTANCES: "
        f"instances={counts['source_instances']} raw_titles={counts['distinct_raw_titles']} "
        f"normalized_titles={counts['distinct_normalized_titles']} unique_ids=636"
    )
    print(
        "EVIDENCE: annotators/qid=3 annotator_steps=1755 paragraphs=1367 "
        "markers=837 total=2204 operation=528 no_evidence=309"
    )
    print(
        "EXACT TITLE: distinct_present="
        f"{exact['distinct_present']} distinct_absent={exact['distinct_absent']} "
        f"present_instances={exact['present_instances']} absent_instances={exact['absent_instances']}"
    )
    print(
        f"QID COVERAGE: full={exact['full_qids']} partial={exact['partial_qids']} zero={exact['zero_qids']}"
    )
    print(
        "LEXICAL: "
        f"EXACT={lexical['exact']} PARTIAL={lexical['partial']} ABSENT={lexical['absent']} "
        f"YES={lexical['yes']} NO={lexical['no']} all_visible_qids={lexical['all_visible_qids']}"
    )
    print("RELIABILITY STRATA: " + " ".join(f"{name}={strata[name]}" for name in STRATUM_ORDER))
    print("RELIABILITY QUOTAS: " + " ".join(f"{name}={quotas[name]}" for name in STRATUM_ORDER))
    print(f"RELIABILITY SAMPLE: {counts['reliability_sample']}")
    print(
        f"PASS 1: qids={counts['pass1_qids']} | PASS 2: unique_ids={counts['pass2_unique_ids']} "
        f"ranks=1..636 adjacent_same_qid={counts['pass2_adjacent_same_qid']}"
    )
    print("HUMAN FIELDS: all scalar=null and all designated lists=[]")
    print(
        "FORBIDDEN DATA: no gold answer/D4 facts/human role/retrieval hop/ABSTAIN/"
        "CORPUS_ABSENT in annotation-facing preparation"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic EFBPT Stage-0 administrative structures."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and validate in memory; write nothing and create no directories",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_freeze_commit()
        input_hashes = validate_and_hash_inputs()
        inputs = load_all_inputs()
        prepared, counts = build_prepared_structures(inputs)
        preparation_commit = resolve_preparation_commit(args.dry_run)
        prepared["stage0_preparation_summary.json"] = preparation_summary(
            counts, preparation_commit
        )

        # Dry-run exercises canonical serialization and reread validation only
        # in memory; it intentionally does not create a reproducibility manifest.
        in_memory_payloads = serialize_artifacts(prepared)
        revalidate_serialized(in_memory_payloads, prepared, inputs["dev_rows"])
        print_validation_summary(counts, preparation_commit)
        if args.dry_run:
            print("STAGE0 PREPARATION DRY-RUN: SUCCESS")
            return 0

        publish_normal_mode(
            prepared,
            inputs["dev_rows"],
            input_hashes,
            preparation_commit,
        )
        print(f"Published canonical Stage-0 artifacts: {OUTPUT_DIR}")
        print("STAGE0 PREPARATION: SUCCESS")
        return 0
    except (Stage0Error, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"STAGE0 PREPARATION ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
