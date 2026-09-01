#!/usr/bin/env python3
"""Leakage-safe terminal runner for frozen EFBPT Stage-0 human annotation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence


SEED = 20260822
FREEZE_COMMIT = "8ee4f5bb932aa89b5d2bd07523a7492a261e741b"
PREPARATION_COMMIT = "4f405983487a1f789580d7790745a541d59091f3"
ORIGINAL_ARTIFACT_COMMIT = "a6b0e5077ffb91064837bf1c5dd8f8ffc613c865"
AMENDED_ARTIFACT_COMMIT = "5bbb186b2b079c344cd85435f9e8bf676d6401d9"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FREEZE_PATH = REPO_ROOT / "docs/EFBPT_STAGE0_SOURCE_ROLE_ATTAINABILITY_FREEZE.md"
STAGE0_DIR = REPO_ROOT / "data/strategyqa_official/efbpt/stage0"
MASTER_PATH = STAGE0_DIR / "source_instance_master.jsonl"
EVIDENCE_PATH = STAGE0_DIR / "official_evidence_links.jsonl"
LEXICAL_PATH = STAGE0_DIR / "lexical_trace_instances.jsonl"
PASS1_PATH = STAGE0_DIR / "pass1_question_manifest.jsonl"
PASS2_PATH = STAGE0_DIR / "pass2_order_manifest.json"
RELIABILITY_PATH = STAGE0_DIR / "reliability160_manifest.json"
SCHEMA_PATH = STAGE0_DIR / "human_annotation_log.schema.json"
SUMMARY_PATH = STAGE0_DIR / "stage0_preparation_summary.json"
REPRO_PATH = STAGE0_DIR / "reproducibility_manifest.json"
HUMAN_LOG_PATH = STAGE0_DIR / "human_annotation_log.jsonl"
DEV200_PATH = REPO_ROOT / "data/strategyqa_official/dev200_seed4242.jsonl"
PARAGRAPHS_PATH = (
    REPO_ROOT / "data/strategyqa_official/strategyqa_train_paragraphs.json"
)
PREPARATION_SCRIPT_PATH = (
    REPO_ROOT / "eval/error_analysis_tests/efbpt/efbpt_prepare_stage0.py"
)

CANONICAL_ARTIFACTS = {
    "source_instance_master.jsonl",
    "official_evidence_links.jsonl",
    "lexical_trace_instances.jsonl",
    "pass1_question_manifest.jsonl",
    "pass2_order_manifest.json",
    "reliability160_manifest.json",
    "human_annotation_log.schema.json",
    "stage0_preparation_summary.json",
    "reproducibility_manifest.json",
}
AMENDED_ARTIFACTS = {
    "human_annotation_log.schema.json",
    "stage0_preparation_summary.json",
    "reproducibility_manifest.json",
}
EXPECTED_ARTIFACT_COMMITS = {
    filename: (
        AMENDED_ARTIFACT_COMMIT
        if filename in AMENDED_ARTIFACTS
        else ORIGINAL_ARTIFACT_COMMIT
    )
    for filename in CANONICAL_ARTIFACTS
}

SOURCE_ROLES = ("EXPLICIT", "LATENT_BRIDGE", "AMBIGUOUS")
PASS2_DECISIONS = ("EXPLICIT", "NOT_YET_EXPLICIT")
PASS3_DECISIONS = ("LATENT_BRIDGE", "AMBIGUOUS")
EXPLICIT_RELATIONS = (
    "DIRECT_MENTION",
    "TRANSLITERATION",
    "COMMON_ALIAS_OR_ABBREVIATION",
    "MORPHOLOGICAL_OR_DEMONYM",
    "DIRECT_SPECIFIC_CONCEPT",
)
DEPENDENCY_STATUSES = (
    "CLEAR_DEPENDENCY",
    "MULTIPLE_PLAUSIBLE_PARENTS",
    "PARALLEL_OR_UNORDERED",
    "UNRESOLVED",
    "NOT_APPLICABLE",
)
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")
SENSITIVE_EVENT_TYPES = ("SUPPORTING_TEXT_VIEW", "ANSWER_INSPECTION")
SENSITIVE_EVENT_PASSES = ("PASS_3", "ADJUDICATION")

PASS1_VIEW_FIELDS = frozenset({"question_ur", "progress"})
PASS2_VIEW_FIELDS = frozenset(
    {"question_ur", "gold_title", "blind_discovery_candidates", "progress"}
)
PASS3_VIEW_FIELDS = frozenset(
    {
        "question_ur",
        "gold_title",
        "official_decomposition",
        "target_official_evidence",
        "progress",
    }
)
PASS3_EVIDENCE_FIELDS = frozenset(
    {
        "official_evidence_annotator_index",
        "decomposition_step_index",
        "decomposition_text",
        "evidence_group_index",
        "paragraph_occurrence_index",
        "record_type",
        "marker",
        "paragraph_id",
        "paragraph_title",
        "paragraph_index",
        "section",
        "headers",
        "raw_path",
    }
)
ADJUDICATION_VIEW_FIELDS = frozenset(
    {
        "question_ur",
        "gold_title",
        "independent_annotations",
        "official_decomposition",
        "target_official_evidence",
        "progress",
    }
)
INDEPENDENT_ANNOTATION_VIEW_FIELDS = frozenset(
    {
        "annotator_id",
        "source_role",
        "explicit_relation_type",
        "confidence",
        "rationale",
        "concrete_intermediate_information",
        "dependency_status",
        "proposed_parent_source_titles",
        "official_support_path_references",
    }
)

EXPECTED_REQUIRED_INPUT_HASHES = {
    "data/strategyqa_official/dev200_seed4242.jsonl": (
        "1ae2cd21c93d1c8d3fda8f6990a183df558e6509d0884fadb29983f5f610d43c"
    ),
    "outputs/efbpt/d2/d2_title_coverage.json": (
        "334d4de46a2c0d498807c45ef34f466aa17d554559a9e3ac7c7ba6e84479d3e5"
    ),
    "outputs/efbpt/d2/d2_recall_split.json": (
        "a93f9594d9915a4684a8c84e94cb746116da31744a449ba9d9726a961d7402a1"
    ),
}


class Stage0Error(RuntimeError):
    """Fatal validation or protocol error."""


class QuitRequested(Exception):
    """A safe interactive quit requested by the annotator."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage0Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage0Error(f"Cannot read valid JSON from {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                require(bool(line.strip()), f"Blank JSONL row at {path}:{line_number}")
                value = json.loads(line)
                require(
                    isinstance(value, dict),
                    f"Non-object JSONL row at {path}:{line_number}",
                )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage0Error(f"Cannot read valid JSONL from {path}: {exc}") from exc
    return rows


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise Stage0Error(f"Git inspection failed: git {' '.join(arguments)}: {detail}")
    return result


def git_latest_commit(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).as_posix()
    commit = git("log", "-1", "--format=%H", "--", relative).stdout.strip()
    require(bool(commit), f"No Git commit found for {relative}")
    return commit


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Stage0Error(f"Invalid date-time value: {value!r}") from exc
    require(parsed.tzinfo is not None, f"Timestamp lacks timezone: {value!r}")


def assert_exact_keys(value: dict[str, Any], allowed: Iterable[str], context: str) -> None:
    expected = set(allowed)
    actual = set(value)
    require(actual == expected, f"{context} fields mismatch: {sorted(actual ^ expected)}")


def assert_view(view: dict[str, Any], allowed: frozenset[str], name: str) -> None:
    assert_exact_keys(view, allowed, name)


def schema_enum(properties: dict[str, Any], field: str) -> tuple[str, ...]:
    values = properties[field]["enum"]
    return tuple(value for value in values if value is not None)


def split_schema_branches(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    branches = schema.get("oneOf")
    require(
        isinstance(branches, list) and len(branches) == 3,
        "Human schema must have one annotation branch and two sensitive-event branches",
    )
    annotation_branches: list[dict[str, Any]] = []
    event_branches: dict[str, dict[str, Any]] = {}
    for branch in branches:
        require(isinstance(branch, dict), "Human schema branch is not an object")
        require(
            branch.get("type") == "object"
            and branch.get("additionalProperties") is False,
            "Every human schema branch must be a strict object",
        )
        properties = branch.get("properties")
        require(isinstance(properties, dict), "Human schema branch lacks properties")
        event_type = properties.get("event_type", {}).get("const")
        if event_type is None:
            annotation_branches.append(branch)
        else:
            require(
                event_type in SENSITIVE_EVENT_TYPES and event_type not in event_branches,
                "Unknown or duplicate sensitive-event schema branch",
            )
            event_branches[event_type] = branch
    require(len(annotation_branches) == 1, "Schema lacks one ordinary annotation branch")
    require(
        set(event_branches) == set(SENSITIVE_EVENT_TYPES),
        "Sensitive-event schema vocabulary changed",
    )
    return annotation_branches[0], event_branches


def validate_schema_contract(schema: dict[str, Any]) -> None:
    require(schema.get("type") == "object", "Human schema must describe an object")
    require(schema.get("additionalProperties") is False, "Schema must fail on extra fields")
    root_properties = schema.get("properties")
    require(isinstance(root_properties, dict), "Schema root properties are missing")
    annotation_branch, event_branches = split_schema_branches(schema)
    properties = annotation_branch["properties"]
    required = annotation_branch.get("required")
    require(isinstance(required, list), "Annotation schema required fields are missing")
    require(set(required) == set(properties), "Every annotation property must be required")
    require(schema_enum(properties, "source_role") == SOURCE_ROLES, "Role vocabulary changed")
    require(schema_enum(properties, "adjudicated_role") == SOURCE_ROLES, "Adjudicated roles changed")
    require(schema_enum(properties, "pass2_decision") == PASS2_DECISIONS, "Pass-2 vocabulary changed")
    require(schema_enum(properties, "pass3_decision") == PASS3_DECISIONS, "Pass-3 vocabulary changed")
    require(
        schema_enum(properties, "explicit_relation_type") == EXPLICIT_RELATIONS,
        "Explicit-relation vocabulary changed",
    )
    require(schema_enum(properties, "dependency_status") == DEPENDENCY_STATUSES, "Dependency vocabulary changed")
    require(schema_enum(properties, "confidence") == CONFIDENCES, "Confidence vocabulary changed")
    event_schema = properties["answer_inspection_exceptions"]["items"]
    require(event_schema.get("additionalProperties") is False, "Answer exceptions allow extra fields")
    require(
        set(event_schema.get("required", []))
        == {"reason", "viewer", "timestamp", "round", "changed_decision"},
        "Answer-exception schema changed",
    )
    require(
        event_schema["properties"]["changed_decision"] == {"type": "boolean"},
        "Completed answer exceptions must have a final Boolean",
    )
    supporting = event_branches["SUPPORTING_TEXT_VIEW"]
    require(
        set(supporting.get("required", []))
        == {
            "event_type",
            "source_instance_id",
            "annotator_id",
            "timestamp",
            "round",
            "annotation_pass",
            "official_support_path_references",
        },
        "Supporting-text event schema changed",
    )
    answer = event_branches["ANSWER_INSPECTION"]
    require(
        set(answer.get("required", []))
        == {
            "event_type",
            "source_instance_id",
            "annotator_id",
            "timestamp",
            "round",
            "annotation_pass",
            "reason",
            "provisional_role",
            "changed_decision",
        },
        "Answer-inspection event schema changed",
    )
    require(
        answer["properties"]["annotation_pass"] == {"const": "ADJUDICATION"},
        "Answer inspection is not restricted to adjudication",
    )
    require(
        answer["properties"]["changed_decision"]["type"] == ["boolean", "null"],
        "Immediate answer inspection must allow a pending null decision effect",
    )


def validate_frozen_repository_inputs() -> dict[str, Any]:
    required_paths = {
        MASTER_PATH,
        EVIDENCE_PATH,
        LEXICAL_PATH,
        PASS1_PATH,
        PASS2_PATH,
        RELIABILITY_PATH,
        SCHEMA_PATH,
        SUMMARY_PATH,
        REPRO_PATH,
    }
    for path in {FREEZE_PATH, DEV200_PATH, PARAGRAPHS_PATH, *required_paths}:
        require(path.is_file(), f"Required input is missing: {path}")

    actual_names = {path.name for path in STAGE0_DIR.iterdir() if path.is_file()}
    allowed_names = CANONICAL_ARTIFACTS | ({HUMAN_LOG_PATH.name} if HUMAN_LOG_PATH.exists() else set())
    require(actual_names == allowed_names, f"Unexpected Stage-0 file set: {sorted(actual_names ^ allowed_names)}")
    require(
        all(not path.is_symlink() for path in required_paths),
        "Frozen artifacts must not be symbolic links",
    )

    repro = load_json(REPRO_PATH)
    require(repro.get("freeze_commit") == FREEZE_COMMIT, "Freeze commit mismatch in manifest")
    require(
        repro.get("preparation_script_commit") == PREPARATION_COMMIT,
        "Preparation commit mismatch in manifest",
    )
    require(repro.get("seed") == SEED, "Frozen seed mismatch")
    require(
        repro.get("unique_normalized_corpus_title_count") == 6_402_346,
        "Frozen corpus-title count mismatch",
    )
    require(
        repro.get("required_input_sha256") == EXPECTED_REQUIRED_INPUT_HASHES,
        "Required input hashes changed",
    )
    for relative, expected_hash in EXPECTED_REQUIRED_INPUT_HASHES.items():
        require(
            sha256_file(REPO_ROOT / relative) == expected_hash,
            f"Required input hash mismatch: {relative}",
        )
    additional_hashes = repro.get("additional_official_input_sha256")
    require(isinstance(additional_hashes, dict) and additional_hashes, "Additional official-input hashes are missing")
    for relative, expected_hash in additional_hashes.items():
        require(isinstance(relative, str) and isinstance(expected_hash, str), "Malformed additional input hash")
        path = REPO_ROOT / relative
        require(path.is_file(), f"Additional official input is missing: {relative}")
        require(sha256_file(path) == expected_hash, f"Additional official input hash mismatch: {relative}")

    artifact_hashes = repro.get("artifact_sha256")
    require(isinstance(artifact_hashes, dict), "Artifact hash mapping is missing")
    require(
        set(artifact_hashes) == CANONICAL_ARTIFACTS - {REPRO_PATH.name},
        "Artifact hash coverage is not exactly 8 files",
    )
    for filename, expected_hash in artifact_hashes.items():
        require(
            sha256_file(STAGE0_DIR / filename) == expected_hash,
            f"Frozen artifact hash mismatch: {filename}",
        )

    require(git_latest_commit(FREEZE_PATH) == FREEZE_COMMIT, "Freeze Git commit mismatch")
    require(
        git_latest_commit(PREPARATION_SCRIPT_PATH) == PREPARATION_COMMIT,
        "Preparation script Git commit mismatch",
    )
    for path in required_paths:
        require(
            git_latest_commit(path) == EXPECTED_ARTIFACT_COMMITS[path.name],
            f"Frozen artifact commit mismatch: {path.name}",
        )
        relative = path.relative_to(REPO_ROOT).as_posix()
        require(
            git("diff", "--quiet", "--", relative, check=False).returncode == 0,
            f"Frozen artifact has a working-tree modification: {path.name}",
        )
        require(
            git("diff", "--cached", "--quiet", "--", relative, check=False).returncode == 0,
            f"Frozen artifact has an index modification: {path.name}",
        )

    schema = load_json(SCHEMA_PATH)
    validate_schema_contract(schema)
    return {"repro": repro, "schema": schema}


def validate_actual_annotation_runtime() -> str:
    relative = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
    require(
        git("ls-files", "--error-unmatch", "--", relative, check=False).returncode == 0,
        "Actual annotation requires the runner to be tracked and committed",
    )
    require(
        git("diff", "--quiet", "--", relative, check=False).returncode == 0,
        "Actual annotation refuses a modified runner",
    )
    require(
        git("diff", "--cached", "--quiet", "--", relative, check=False).returncode == 0,
        "Actual annotation refuses a staged but uncommitted runner",
    )
    tracked_status = git("status", "--short", "--untracked-files=no").stdout.strip()
    require(not tracked_status, "Actual annotation requires a clean tracked worktree")
    commit = git_latest_commit(SCRIPT_PATH)
    require(commit == git("rev-parse", "HEAD").stdout.strip(), "Runner must be committed at HEAD")
    return commit


def schedule_no_adjacent(items: Sequence[tuple[str, str]]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for qid, source_id in items:
        grouped[qid].append(source_id)
    for qid in grouped:
        grouped[qid].sort()

    generator = random.Random(SEED)
    for qid in sorted(grouped):
        generator.shuffle(grouped[qid])
    qids = sorted(grouped)
    generator.shuffle(qids)
    priority = {qid: index for index, qid in enumerate(qids)}
    cursors = {qid: 0 for qid in grouped}
    heap = [(-len(grouped[qid]), priority[qid], qid) for qid in grouped]
    heapq.heapify(heap)

    ordered: list[str] = []
    held: tuple[int, int, str] | None = None
    while heap:
        remaining_negative, tie_priority, qid = heapq.heappop(heap)
        cursor = cursors[qid]
        ordered.append(grouped[qid][cursor])
        cursors[qid] += 1
        if held is not None:
            heapq.heappush(heap, held)
            held = None
        remaining = -remaining_negative - 1
        if remaining:
            held = (-remaining, tie_priority, qid)
    require(held is None, "No-adjacent scheduler ended with an un-emitted item")
    require(len(ordered) == len(items), "No-adjacent scheduler lost an item")
    require(len(set(ordered)) == len(ordered), "No-adjacent scheduler duplicated an item")
    qid_by_source = {source_id: qid for qid, source_id in items}
    maximum = max(Counter(qid_by_source.values()).values(), default=0)
    feasible = maximum <= (len(items) - maximum) + 1
    if feasible:
        require(
            all(qid_by_source[left] != qid_by_source[right] for left, right in zip(ordered, ordered[1:])),
            "No-adjacent scheduler produced an avoidable same-qid collision",
        )
    return ordered


class FrozenData:
    def __init__(self) -> None:
        validated = validate_frozen_repository_inputs()
        self.schema: dict[str, Any] = validated["schema"]
        self.master_rows = load_jsonl(MASTER_PATH)
        self.evidence_rows = load_jsonl(EVIDENCE_PATH)
        self.lexical_rows = load_jsonl(LEXICAL_PATH)
        self.pass1_rows = load_jsonl(PASS1_PATH)
        self.pass2_manifest = load_json(PASS2_PATH)
        self.reliability_manifest = load_json(RELIABILITY_PATH)
        self.summary = load_json(SUMMARY_PATH)
        self.dev_rows = load_jsonl(DEV200_PATH)

        self.master_by_id = {row["source_instance_id"]: row for row in self.master_rows}
        self.qid_by_source = {
            source_id: row["urbench_qid"] for source_id, row in self.master_by_id.items()
        }
        self.sources_by_qid: dict[str, list[str]] = defaultdict(list)
        for source_id, qid in self.qid_by_source.items():
            self.sources_by_qid[qid].append(source_id)
        for source_ids in self.sources_by_qid.values():
            source_ids.sort()
        self.question_by_qid = {row["urbench_qid"]: row["question_ur"] for row in self.pass1_rows}
        self.dev_by_qid = {row["urbench_qid"]: row for row in self.dev_rows}
        self.pass2_ids = [item["source_instance_id"] for item in self.pass2_manifest["items"]]
        self.reliability_ids = [
            item["source_instance_id"] for item in self.reliability_manifest["sampled_instances"]
        ]
        self.reliability_set = set(self.reliability_ids)
        self.reliability_qids = sorted({self.qid_by_source[source_id] for source_id in self.reliability_ids})
        self.reliability_sources_by_qid: dict[str, list[str]] = defaultdict(list)
        for source_id in self.reliability_ids:
            self.reliability_sources_by_qid[self.qid_by_source[source_id]].append(source_id)
        for source_ids in self.reliability_sources_by_qid.values():
            source_ids.sort()
        self.reliability_order = schedule_no_adjacent(
            [(self.qid_by_source[source_id], source_id) for source_id in self.reliability_ids]
        )
        self.evidence_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.evidence_rows:
            self.evidence_by_qid[row["urbench_qid"]].append(row)
        self._paragraphs: dict[str, dict[str, Any]] | None = None
        self.validate_structures()

    def validate_structures(self) -> None:
        require(len(self.master_rows) == 636, "Source master count changed")
        require(len(self.master_by_id) == 636, "Source IDs are not unique")
        require(len(self.sources_by_qid) == 200, "Source qid count changed")
        require(len(self.pass1_rows) == 200, "Pass-1 question count changed")
        require(len(self.question_by_qid) == 200, "Pass-1 qids are not unique")
        require(set(self.question_by_qid) == set(self.sources_by_qid), "Pass-1/master qids differ")
        require(len(self.dev_rows) == 200 and len(self.dev_by_qid) == 200, "DEV200 count changed")
        require(set(self.dev_by_qid) == set(self.question_by_qid), "DEV200/pass-1 qids differ")
        require(self.pass2_manifest.get("seed") == SEED, "Pass-2 seed changed")
        require(len(self.pass2_ids) == 636, "Pass-2 count changed")
        require(set(self.pass2_ids) == set(self.master_by_id), "Pass-2 source population changed")
        require(len(set(self.pass2_ids)) == 636, "Pass-2 contains duplicate source IDs")
        require(
            [item["rank"] for item in self.pass2_manifest["items"]] == list(range(1, 637)),
            "Pass-2 ranks changed",
        )
        require(
            all(
                self.qid_by_source[left] != self.qid_by_source[right]
                for left, right in zip(self.pass2_ids, self.pass2_ids[1:])
            ),
            "Pass-2 has an adjacent same-qid pair",
        )
        require(self.reliability_manifest.get("seed") == SEED, "Reliability seed changed")
        require(len(self.reliability_ids) == 160, "Reliability sample count changed")
        require(len(self.reliability_set) == 160, "Reliability sample contains duplicates")
        require(self.reliability_set <= set(self.master_by_id), "Reliability sample has unknown IDs")
        require(
            self.reliability_manifest.get("stratum_quotas")
            == {
                "NO__EXACT_ABSENT": 21,
                "NO__EXACT_PRESENT": 63,
                "YES__EXACT_ABSENT": 27,
                "YES__EXACT_PRESENT": 49,
            },
            "Reliability quotas changed",
        )
        require(len(self.evidence_rows) == 2204, "Evidence-link count changed")
        require(len(self.lexical_rows) == 636, "Lexical diagnostic count changed")
        require(
            all(row["question_ur"] == self.dev_by_qid[row["urbench_qid"]]["question_ur"] for row in self.pass1_rows),
            "Pass-1 Urdu question differs from DEV200",
        )

    def population(self, role: str) -> list[str]:
        if role == "PRIMARY":
            return list(self.pass2_ids)
        if role == "RELIABILITY":
            return list(self.reliability_order)
        raise Stage0Error(f"Role {role} has no independent-annotation population")

    def pass1_qids(self, role: str) -> list[str]:
        if role == "PRIMARY":
            return [row["urbench_qid"] for row in self.pass1_rows]
        if role == "RELIABILITY":
            return list(self.reliability_qids)
        raise Stage0Error(f"Role {role} has no Pass-1 population")

    def sources_for_pass1_qid(self, role: str, qid: str) -> list[str]:
        if role == "PRIMARY":
            return self.sources_by_qid[qid]
        if role == "RELIABILITY":
            return self.reliability_sources_by_qid[qid]
        raise Stage0Error(f"Role {role} has no Pass-1 population")

    def paragraphs(self) -> dict[str, dict[str, Any]]:
        if self._paragraphs is None:
            value = load_json(PARAGRAPHS_PATH)
            require(isinstance(value, dict), "Official paragraph store is not an object")
            self._paragraphs = value
        return self._paragraphs


def build_pass1_view(data: FrozenData, qid: str, progress: str) -> dict[str, Any]:
    view = {"question_ur": data.question_by_qid[qid], "progress": progress}
    assert_view(view, PASS1_VIEW_FIELDS, "Pass-1 view")
    return view


def build_pass2_view(
    data: FrozenData,
    source_id: str,
    blind_candidates: list[str],
    progress: str,
) -> dict[str, Any]:
    master = data.master_by_id[source_id]
    view = {
        "question_ur": master["question_ur"],
        "gold_title": master["gold_title"],
        "blind_discovery_candidates": list(blind_candidates),
        "progress": progress,
    }
    assert_view(view, PASS2_VIEW_FIELDS, "Pass-2 view")
    return view


def target_step_evidence(data: FrozenData, source_id: str) -> list[dict[str, Any]]:
    qid = data.qid_by_source[source_id]
    qid_rows = data.evidence_by_qid[qid]
    target_steps = {
        (row["official_evidence_annotator_index"], row["decomposition_step_index"])
        for row in qid_rows
        if row["source_instance_id"] == source_id
    }
    require(target_steps, "Target source has no official step-linked evidence")
    selected = [
        row
        for row in qid_rows
        if (row["official_evidence_annotator_index"], row["decomposition_step_index"])
        in target_steps
    ]
    result: list[dict[str, Any]] = []
    for row in selected:
        evidence_view = {field: deepcopy(row[field]) for field in PASS3_EVIDENCE_FIELDS}
        assert_view(evidence_view, PASS3_EVIDENCE_FIELDS, "Pass-3 evidence row")
        result.append(evidence_view)
    return result


def build_pass3_view(data: FrozenData, source_id: str, progress: str) -> dict[str, Any]:
    master = data.master_by_id[source_id]
    qid = master["urbench_qid"]
    view = {
        "question_ur": master["question_ur"],
        "gold_title": master["gold_title"],
        "official_decomposition": deepcopy(data.dev_by_qid[qid]["official_decomposition"]),
        "target_official_evidence": target_step_evidence(data, source_id),
        "progress": progress,
    }
    assert_view(view, PASS3_VIEW_FIELDS, "Pass-3 view")
    return view


def build_adjudication_view(
    data: FrozenData,
    source_id: str,
    independent_states: Sequence[dict[str, Any]],
    progress: str,
) -> dict[str, Any]:
    base = build_pass3_view(data, source_id, progress)
    annotations: list[dict[str, Any]] = []
    for state in independent_states:
        annotation = {field: deepcopy(state[field]) for field in INDEPENDENT_ANNOTATION_VIEW_FIELDS}
        assert_view(annotation, INDEPENDENT_ANNOTATION_VIEW_FIELDS, "Independent annotation view")
        annotations.append(annotation)
    view = {
        "question_ur": base["question_ur"],
        "gold_title": base["gold_title"],
        "independent_annotations": annotations,
        "official_decomposition": base["official_decomposition"],
        "target_official_evidence": base["target_official_evidence"],
        "progress": progress,
    }
    assert_view(view, ADJUDICATION_VIEW_FIELDS, "Adjudication view")
    return view


def empty_annotation_record(
    schema: dict[str, Any], source_id: str, annotator_id: str
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source_instance_id": source_id,
        "annotator_id": annotator_id,
        "pass1_completed": False,
        "blind_discovery_candidates": [],
        "pass1_timestamp": None,
        "pass1_round": None,
        "pass2_decision": None,
        "pass2_timestamp": None,
        "pass2_round": None,
        "pass3_decision": None,
        "pass3_timestamp": None,
        "pass3_round": None,
        "source_role": None,
        "explicit_relation_type": None,
        "urdu_span_if_explicit": None,
        "confidence": None,
        "rationale": None,
        "concrete_intermediate_information": None,
        "dependency_status": None,
        "proposed_parent_source_titles": [],
        "parent_source_titles": [],
        "official_support_path_references": [],
        "disagreement_flag": None,
        "adjudication_required": None,
        "adjudicated_role": None,
        "adjudication_rationale": None,
        "adjudication_changed_bridge_eligibility": None,
        "adjudicator_id": None,
        "adjudication_timestamp": None,
        "adjudication_round": None,
        "answer_inspection_exception_used": None,
        "answer_inspection_exceptions": [],
        "annotation_timestamp": None,
        "round": None,
    }
    annotation_branch, _ = split_schema_branches(schema)
    assert_exact_keys(values, annotation_branch["required"], "Blank annotation record")
    return values


def is_bool_or_none(value: Any) -> bool:
    return value is None or type(value) is bool


def is_positive_int_or_none(value: Any) -> bool:
    return value is None or (type(value) is int and value >= 1)


def is_string_or_none(value: Any) -> bool:
    return value is None or isinstance(value, str)


def validate_string_list(value: Any, field: str) -> None:
    require(isinstance(value, list), f"{field} must be a list")
    require(all(isinstance(item, str) and item.strip() for item in value), f"{field} has an invalid item")


def validate_annotation_record(
    record: dict[str, Any], schema: dict[str, Any], known_source_ids: set[str]
) -> None:
    require(isinstance(record, dict), "Annotation event must be an object")
    annotation_branch, _ = split_schema_branches(schema)
    assert_exact_keys(record, annotation_branch["required"], "Annotation event")
    source_id = record["source_instance_id"]
    require(
        isinstance(source_id, str) and re.fullmatch(r"s0-[0-9a-f]{64}", source_id),
        "Invalid source_instance_id",
    )
    require(source_id in known_source_ids, "Annotation event has an unknown source_instance_id")
    require(
        isinstance(record["annotator_id"], str) and record["annotator_id"].strip(),
        "annotator_id must be nonempty",
    )

    require(type(record["pass1_completed"]) is bool, "pass1_completed must be boolean")
    for field in (
        "blind_discovery_candidates",
        "proposed_parent_source_titles",
        "parent_source_titles",
        "official_support_path_references",
    ):
        validate_string_list(record[field], field)
    for field in ("answer_inspection_exceptions",):
        require(isinstance(record[field], list), f"{field} must be a list")
    for field in (
        "pass1_round",
        "pass2_round",
        "pass3_round",
        "adjudication_round",
        "round",
    ):
        require(is_positive_int_or_none(record[field]), f"{field} must be null or a positive integer")
    for field in (
        "pass1_timestamp",
        "pass2_timestamp",
        "pass3_timestamp",
        "adjudication_timestamp",
        "annotation_timestamp",
    ):
        require(is_string_or_none(record[field]), f"{field} must be null or a string")
        if record[field] is not None:
            parse_timestamp(record[field])
    for field in (
        "urdu_span_if_explicit",
        "rationale",
        "concrete_intermediate_information",
        "adjudication_rationale",
        "adjudicator_id",
    ):
        require(is_string_or_none(record[field]), f"{field} must be null or a string")
    for field in (
        "disagreement_flag",
        "adjudication_required",
        "adjudication_changed_bridge_eligibility",
        "answer_inspection_exception_used",
    ):
        require(is_bool_or_none(record[field]), f"{field} must be null or boolean")

    require(record["pass2_decision"] in (None, *PASS2_DECISIONS), "Invalid Pass-2 decision")
    require(record["pass3_decision"] in (None, *PASS3_DECISIONS), "Invalid Pass-3 decision")
    require(record["source_role"] in (None, *SOURCE_ROLES), "Invalid source role")
    require(record["adjudicated_role"] in (None, *SOURCE_ROLES), "Invalid adjudicated role")
    require(
        record["explicit_relation_type"] in (None, *EXPLICIT_RELATIONS),
        "Invalid explicit relation",
    )
    require(record["dependency_status"] in (None, *DEPENDENCY_STATUSES), "Invalid dependency status")
    require(record["confidence"] in (None, *CONFIDENCES), "Invalid confidence")

    if not record["pass1_completed"]:
        require(not record["blind_discovery_candidates"], "Unfinished Pass 1 cannot have candidates")
        require(record["pass1_timestamp"] is None and record["pass1_round"] is None, "Unfinished Pass 1 has metadata")
        require(record["pass2_decision"] is None and record["pass3_decision"] is None, "Later pass precedes Pass 1")
        require(record["source_role"] is None, "Unfinished Pass 1 cannot have a source role")
    else:
        require(record["pass1_timestamp"] is not None and record["pass1_round"] is not None, "Completed Pass 1 lacks metadata")

    if record["pass2_decision"] is None:
        require(record["pass2_timestamp"] is None and record["pass2_round"] is None, "Unfinished Pass 2 has metadata")
        require(record["pass3_decision"] is None, "Pass 3 precedes Pass 2")
        for field in (
            "source_role",
            "explicit_relation_type",
            "urdu_span_if_explicit",
            "confidence",
            "rationale",
            "concrete_intermediate_information",
            "dependency_status",
        ):
            require(record[field] is None, f"Unfinished Pass 2 has {field}")
        require(
            not record["proposed_parent_source_titles"]
            and not record["parent_source_titles"]
            and not record["official_support_path_references"],
            "Unfinished Pass 2 has bridge fields",
        )
    else:
        require(record["pass1_completed"], "Pass 2 precedes Pass 1")
        require(record["pass2_timestamp"] is not None and record["pass2_round"] is not None, "Completed Pass 2 lacks metadata")

    if record["pass2_decision"] == "EXPLICIT":
        require(record["pass3_decision"] is None, "EXPLICIT cannot enter Pass 3")
        require(record["source_role"] == "EXPLICIT", "EXPLICIT Pass 2 must set the same source role")
        require(record["explicit_relation_type"] in EXPLICIT_RELATIONS, "EXPLICIT lacks relation type")
        require(record["confidence"] in CONFIDENCES, "EXPLICIT lacks confidence")
        require(isinstance(record["rationale"], str) and record["rationale"].strip(), "EXPLICIT lacks rationale")
        require(
            record["concrete_intermediate_information"] is None
            and record["dependency_status"] is None
            and not record["proposed_parent_source_titles"]
            and not record["parent_source_titles"]
            and not record["official_support_path_references"],
            "EXPLICIT record has bridge-only fields",
        )

    if record["pass2_decision"] == "NOT_YET_EXPLICIT":
        require(record["explicit_relation_type"] is None, "NOT_YET_EXPLICIT has relation type")
        require(record["urdu_span_if_explicit"] is None, "NOT_YET_EXPLICIT has an explicit span")
        if record["pass3_decision"] is None:
            for field in (
                "source_role",
                "confidence",
                "rationale",
                "concrete_intermediate_information",
                "dependency_status",
            ):
                require(record[field] is None, f"Unfinished Pass 3 has {field}")
            require(
                not record["proposed_parent_source_titles"]
                and not record["parent_source_titles"]
                and not record["official_support_path_references"],
                "Unfinished Pass 3 has bridge fields",
            )

    if record["pass3_decision"] is None:
        require(record["pass3_timestamp"] is None and record["pass3_round"] is None, "Unfinished Pass 3 has metadata")
    else:
        require(record["pass2_decision"] == "NOT_YET_EXPLICIT", "Pass 3 lacks NOT_YET_EXPLICIT")
        require(record["pass3_timestamp"] is not None and record["pass3_round"] is not None, "Completed Pass 3 lacks metadata")
        require(record["source_role"] == record["pass3_decision"], "Pass-3 decision/role mismatch")
        require(record["confidence"] in CONFIDENCES, "Pass 3 lacks confidence")
        require(isinstance(record["rationale"], str) and record["rationale"].strip(), "Pass 3 lacks rationale")

    if record["pass3_decision"] == "LATENT_BRIDGE":
        require(
            isinstance(record["concrete_intermediate_information"], str)
            and record["concrete_intermediate_information"].strip(),
            "LATENT_BRIDGE lacks concrete intermediate information",
        )
        require(record["dependency_status"] in DEPENDENCY_STATUSES, "LATENT_BRIDGE lacks dependency status")
        require(record["official_support_path_references"], "LATENT_BRIDGE lacks official support paths")
        parents = record["parent_source_titles"]
        require(parents == record["proposed_parent_source_titles"], "Raw proposed/final parent lists differ")
        if record["dependency_status"] == "MULTIPLE_PLAUSIBLE_PARENTS":
            require(len(parents) >= 2, "Multiple plausible parents requires at least two titles")
        elif record["dependency_status"] in {
            "PARALLEL_OR_UNORDERED",
            "UNRESOLVED",
            "NOT_APPLICABLE",
        }:
            require(not parents, "This dependency status cannot carry parent titles")
        else:
            require(len(parents) <= 1, "Only MULTIPLE_PLAUSIBLE_PARENTS allows multiple titles")

    if record["pass3_decision"] == "AMBIGUOUS":
        require(record["concrete_intermediate_information"] is None, "AMBIGUOUS cannot assert intermediate information")
        require(record["dependency_status"] is None, "AMBIGUOUS cannot assert dependency")
        require(not record["proposed_parent_source_titles"] and not record["parent_source_titles"], "AMBIGUOUS cannot assert parents")

    adjudication_fields = (
        "adjudicated_role",
        "adjudication_rationale",
        "adjudication_changed_bridge_eligibility",
        "adjudicator_id",
        "adjudication_timestamp",
        "adjudication_round",
    )
    if record["adjudication_required"] is not True:
        require(all(record[field] is None for field in adjudication_fields), "Non-adjudication event has adjudication fields")
        require(record["answer_inspection_exception_used"] in (None, False), "Non-adjudication event used answer exception")
        require(not record["answer_inspection_exceptions"], "Non-adjudication event has answer exception history")
    if record["adjudicated_role"] is not None:
        require(record["adjudication_required"] is True, "Adjudicated role is not marked required")
        require(
            isinstance(record["adjudication_rationale"], str) and record["adjudication_rationale"].strip(),
            "Adjudication lacks rationale",
        )
        require(type(record["adjudication_changed_bridge_eligibility"]) is bool, "Adjudication lacks eligibility-change flag")
        require(isinstance(record["adjudicator_id"], str) and record["adjudicator_id"].strip(), "Adjudication lacks adjudicator")
        require(record["adjudication_timestamp"] is not None and record["adjudication_round"] is not None, "Adjudication lacks metadata")
        require(type(record["answer_inspection_exception_used"]) is bool, "Adjudication lacks answer-exception flag")

    if record["answer_inspection_exception_used"] is True:
        require(record["adjudication_required"] is True, "Answer exception occurred outside adjudication")
        require(record["answer_inspection_exceptions"], "Answer exception flag lacks history")
    else:
        require(not record["answer_inspection_exceptions"], "Unused answer exception has history")
    for event in record["answer_inspection_exceptions"]:
        require(isinstance(event, dict), "Answer exception must be an object")
        assert_exact_keys(
            event,
            {"reason", "viewer", "timestamp", "round", "changed_decision"},
            "Answer exception",
        )
        require(isinstance(event["reason"], str) and event["reason"].strip(), "Answer exception lacks reason")
        require(isinstance(event["viewer"], str) and event["viewer"].strip(), "Answer exception lacks viewer")
        require(isinstance(event["timestamp"], str), "Answer exception lacks timestamp")
        parse_timestamp(event["timestamp"])
        require(type(event["round"]) is int and event["round"] >= 1, "Answer exception has invalid round")
        require(type(event["changed_decision"]) is bool, "Answer exception lacks changed-decision flag")

    if record["annotation_timestamp"] is not None:
        require(record["round"] is not None, "Annotation timestamp lacks round")


def parse_support_reference(reference: str) -> tuple[int, ...]:
    try:
        value = json.loads(reference)
    except json.JSONDecodeError as exc:
        raise Stage0Error("Support-path reference is not valid JSON") from exc
    require(isinstance(value, dict), "Support-path reference must be an object")
    assert_exact_keys(
        value,
        {"raw_path", "supporting_text_viewed"},
        "Support-path reference",
    )
    raw_path = value["raw_path"]
    require(
        isinstance(raw_path, list)
        and len(raw_path) == 4
        and all(type(item) is int and item >= 0 for item in raw_path),
        "Support-path reference has an invalid raw path",
    )
    require(
        type(value["supporting_text_viewed"]) is bool,
        "Support-path reference lacks a Boolean view flag",
    )
    return tuple(raw_path)


def validate_sensitive_event(
    record: dict[str, Any], schema: dict[str, Any], known_source_ids: set[str]
) -> None:
    require(isinstance(record, dict), "Sensitive-view event must be an object")
    event_type = record.get("event_type")
    require(event_type in SENSITIVE_EVENT_TYPES, "Unknown sensitive-view event type")
    _, event_branches = split_schema_branches(schema)
    branch = event_branches[event_type]
    assert_exact_keys(record, branch["required"], f"{event_type} event")
    source_id = record["source_instance_id"]
    require(
        isinstance(source_id, str) and re.fullmatch(r"s0-[0-9a-f]{64}", source_id),
        "Sensitive-view event has an invalid source_instance_id",
    )
    require(source_id in known_source_ids, "Sensitive-view event has an unknown source_instance_id")
    require(
        isinstance(record["annotator_id"], str) and record["annotator_id"].strip(),
        "Sensitive-view event annotator_id must be nonempty",
    )
    require(isinstance(record["timestamp"], str), "Sensitive-view event lacks a timestamp")
    parse_timestamp(record["timestamp"])
    require(
        type(record["round"]) is int and record["round"] >= 1,
        "Sensitive-view event round must be positive",
    )
    require(
        record["annotation_pass"] in SENSITIVE_EVENT_PASSES,
        "Sensitive-view event has an invalid pass",
    )
    if event_type == "SUPPORTING_TEXT_VIEW":
        references = record["official_support_path_references"]
        validate_string_list(references, "official_support_path_references")
        require(references, "Supporting-text view event lacks a support reference")
        require(len(references) == len(set(references)), "Supporting-text view event repeats a reference")
        for reference in references:
            parse_support_reference(reference)
            value = json.loads(reference)
            require(
                value["supporting_text_viewed"] is True,
                "Supporting-text view event must record a completed reveal",
            )
    else:
        require(
            record["annotation_pass"] == "ADJUDICATION",
            "Answer inspection is permitted only during adjudication",
        )
        require(
            isinstance(record["reason"], str) and record["reason"].strip(),
            "Answer-inspection event lacks a reason",
        )
        require(
            record["provisional_role"] in SOURCE_ROLES,
            "Answer-inspection event lacks a provisional role",
        )
        require(
            record["changed_decision"] is None
            or type(record["changed_decision"]) is bool,
            "Answer-inspection changed_decision must be null or Boolean",
        )


def validate_log_record(
    record: dict[str, Any], schema: dict[str, Any], known_source_ids: set[str]
) -> None:
    if "event_type" in record:
        validate_sensitive_event(record, schema, known_source_ids)
    else:
        validate_annotation_record(record, schema, known_source_ids)


def validate_transition(previous: dict[str, Any] | None, current: dict[str, Any]) -> None:
    if previous is None:
        pass1_event = current["pass1_completed"] and current["pass2_decision"] is None
        adjudication_event = current["adjudicated_role"] is not None and not current["pass1_completed"]
        require(pass1_event ^ adjudication_event, "First event must complete Pass 1 or adjudication")
        return

    require(previous["source_instance_id"] == current["source_instance_id"], "Transition changes source ID")
    require(previous["annotator_id"] == current["annotator_id"], "Transition changes annotator ID")
    require(previous["adjudicated_role"] is None, "Completed adjudication cannot be revised")
    require(previous["pass3_decision"] is None, "Completed Pass 3 cannot be revised")
    require(previous["pass2_decision"] != "EXPLICIT", "Completed EXPLICIT decision cannot be revised")

    if previous["pass1_completed"]:
        for field in ("pass1_completed", "blind_discovery_candidates", "pass1_timestamp", "pass1_round"):
            require(current[field] == previous[field], f"Pass-1 history changed: {field}")
    require(current["pass1_completed"], "Pass-1 completion cannot regress")

    if previous["pass2_decision"] is None:
        require(current["pass2_decision"] in PASS2_DECISIONS, "Next event must complete Pass 2")
        require(current["pass3_decision"] is None, "Pass 2 and Pass 3 cannot complete together")
    else:
        require(previous["pass2_decision"] == "NOT_YET_EXPLICIT", "Only NOT_YET_EXPLICIT can advance")
        for field in ("pass2_decision", "pass2_timestamp", "pass2_round"):
            require(current[field] == previous[field], f"Pass-2 history changed: {field}")
        require(current["pass3_decision"] in PASS3_DECISIONS, "Next event must complete Pass 3")


def parse_history_bytes(
    payload: bytes, schema: dict[str, Any], known_source_ids: set[str]
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    if not payload:
        return [], {}, []
    require(payload.endswith(b"\n"), "Annotation log has a truncated final line")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage0Error("Annotation log is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    sensitive_events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        require(bool(line), f"Blank annotation-log line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Stage0Error(f"Invalid annotation-log JSON at line {line_number}: {exc}") from exc
        validate_log_record(row, schema, known_source_ids)
        if "event_type" in row:
            sensitive_events.append(row)
        else:
            key = (row["source_instance_id"], row["annotator_id"])
            validate_transition(latest.get(key), row)
            latest[key] = row
        rows.append(row)
    return rows, latest, sensitive_events


def load_history(
    data: FrozenData,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    if not HUMAN_LOG_PATH.exists():
        return [], {}, []
    require(HUMAN_LOG_PATH.is_file() and not HUMAN_LOG_PATH.is_symlink(), "Human log is not a regular file")
    return parse_history_bytes(HUMAN_LOG_PATH.read_bytes(), data.schema, set(data.master_by_id))


def append_log_records(
    log_path: Path,
    schema: dict[str, Any],
    known_source_ids: set[str],
    records: Sequence[dict[str, Any]],
) -> None:
    require(bool(records), "No records supplied for append")
    for record in records:
        validate_log_record(record, schema, known_source_ids)
    if log_path.exists():
        require(log_path.is_file() and not log_path.is_symlink(), "Annotation log is not a regular file")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    descriptor = os.open(log_path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0, closefd=False)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        handle.seek(0)
        existing = handle.read()
        _, latest, _ = parse_history_bytes(existing, schema, known_source_ids)
        prospective = dict(latest)
        for record in records:
            if "event_type" not in record:
                key = (record["source_instance_id"], record["annotator_id"])
                validate_transition(prospective.get(key), record)
                prospective[key] = record
        payload = b"".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for record in records
        )
        written = handle.write(payload)
        require(written == len(payload), "Short atomic append to annotation log")
        handle.flush()
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            handle.close()
            os.close(descriptor)


def append_records(data: FrozenData, records: Sequence[dict[str, Any]]) -> None:
    append_log_records(
        HUMAN_LOG_PATH,
        data.schema,
        set(data.master_by_id),
        records,
    )


def qualified_annotator_id(role: str, annotator: str) -> str:
    clean = annotator.strip()
    require(bool(clean), "--annotator must be nonempty")
    require("\n" not in clean and "\r" not in clean, "--annotator contains a line break")
    return f"{role}:{clean}"


def states_for_annotator(
    latest: dict[tuple[str, str], dict[str, Any]], annotator_id: str
) -> dict[str, dict[str, Any]]:
    return {
        source_id: state
        for (source_id, state_annotator), state in latest.items()
        if state_annotator == annotator_id
    }


def qid_pass1_state(
    data: FrozenData,
    role: str,
    qid: str,
    states: dict[str, dict[str, Any]],
) -> tuple[bool, list[str] | None]:
    source_ids = data.sources_for_pass1_qid(role, qid)
    completed = [states.get(source_id) for source_id in source_ids]
    completed = [state for state in completed if state is not None and state["pass1_completed"]]
    if not completed:
        return False, None
    candidate_sets = {tuple(state["blind_discovery_candidates"]) for state in completed}
    require(len(candidate_sets) == 1, "Pass-1 candidates disagree within one qid")
    candidates = list(next(iter(candidate_sets)))
    return len(completed) == len(source_ids), candidates


def progress_counts(
    data: FrozenData,
    role: str,
    states: dict[str, dict[str, Any]],
) -> dict[str, int]:
    population = data.population(role)
    qids = data.pass1_qids(role)
    pass1_complete = sum(qid_pass1_state(data, role, qid, states)[0] for qid in qids)
    population_states = [states.get(source_id) for source_id in population]
    pass2_complete = sum(state is not None and state["pass2_decision"] is not None for state in population_states)
    explicit = sum(state is not None and state["pass2_decision"] == "EXPLICIT" for state in population_states)
    not_yet = sum(state is not None and state["pass2_decision"] == "NOT_YET_EXPLICIT" for state in population_states)
    pass3_complete = sum(
        state is not None
        and state["pass2_decision"] == "NOT_YET_EXPLICIT"
        and state["pass3_decision"] is not None
        for state in population_states
    )
    latent = sum(state is not None and state["pass3_decision"] == "LATENT_BRIDGE" for state in population_states)
    ambiguous = sum(state is not None and state["pass3_decision"] == "AMBIGUOUS" for state in population_states)
    return {
        "pass1_complete": pass1_complete,
        "pass1_total": len(qids),
        "pass2_complete": pass2_complete,
        "pass2_total": len(population),
        "pass2_explicit": explicit,
        "pass2_not_yet_explicit": not_yet,
        "pass3_complete": pass3_complete,
        "pass3_required": not_yet,
        "pass3_latent_bridge": latent,
        "pass3_ambiguous": ambiguous,
    }


def prompt_raw(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt) as exc:
        print()
        raise QuitRequested from exc


def prompt_nonempty(prompt: str) -> str:
    while True:
        value = prompt_raw(prompt).strip()
        if value.lower() in {":q", "q"}:
            raise QuitRequested
        if value.lower() in {":h", "h"}:
            print("Enter a nonempty response, or :q to quit safely.")
            continue
        if value:
            return value
        print("A nonempty response is required.")


def prompt_optional(prompt: str) -> str | None:
    value = prompt_raw(prompt).strip()
    if value.lower() in {":q", "q"}:
        raise QuitRequested
    if value.lower() in {":h", "h"}:
        print("Enter text, ENTER for none, or :q to quit safely.")
        return prompt_optional(prompt)
    return value or None


def prompt_choice(prompt: str, choices: dict[str, str], help_text: str) -> str:
    while True:
        value = prompt_raw(prompt).strip().upper()
        if value in {"Q", ":Q"}:
            raise QuitRequested
        if value in {"H", ":H", "?"}:
            print(help_text)
            continue
        if value in choices:
            return choices[value]
        print(f"Invalid choice. {help_text}")


def parse_delimited_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split("|")]
    require(all(names), "Names must be nonempty around the '|' delimiter")
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def prompt_candidates() -> list[str]:
    while True:
        value = prompt_raw("Candidates (ENTER=NONE, separate multiple with |, :q quit, :h help): ").strip()
        if value.lower() in {":q", "q"}:
            raise QuitRequested
        if value.lower() in {":h", "h"}:
            print("Record only English source/entity identities hypothesized directly from the Urdu question.")
            continue
        if not value:
            return []
        try:
            candidates = parse_delimited_names(value)
        except Stage0Error as exc:
            print(exc)
            continue
        print("Parsed candidates:")
        for candidate in candidates:
            print(f"  - {candidate}")
        action = prompt_choice(
            "Save these candidates? [Y/R/Q]: ",
            {"Y": "SAVE", "R": "REENTER"},
            "Y saves; R re-enters; Q quits safely.",
        )
        if action == "SAVE":
            return candidates


def prompt_confidence() -> str:
    return prompt_choice(
        "Confidence [H/M/L]: ",
        {"H": "HIGH", "M": "MEDIUM", "L": "LOW"},
        "H=HIGH, M=MEDIUM, L=LOW, Q=quit.",
    )


def print_pass_heading(pass_name: str, progress: str) -> None:
    print(f"\n{'=' * 18} {pass_name} {'=' * 18}")
    print(f"Progress: {progress}")


def display_pass1_view(view: dict[str, Any]) -> None:
    assert_view(view, PASS1_VIEW_FIELDS, "Pass-1 display")
    print_pass_heading("PASS 1 — BLIND DISCOVERY", view["progress"])
    print(f"Urdu question:\n{view['question_ur']}")
    print("\nWhich English source/entity identities can you directly hypothesize from this Urdu question?")


def display_pass2_view(view: dict[str, Any]) -> None:
    assert_view(view, PASS2_VIEW_FIELDS, "Pass-2 display")
    print_pass_heading("PASS 2 — EXPLICITNESS", view["progress"])
    print(f"Urdu question:\n{view['question_ur']}")
    print(f"\nTarget English title:\n{view['gold_title']}")
    candidates = view["blind_discovery_candidates"]
    print("\nYour frozen Pass-1 candidates:")
    if candidates:
        for candidate in candidates:
            print(f"  - {candidate}")
    else:
        print("  NONE")


def display_evidence_structure(evidence_rows: Sequence[dict[str, Any]]) -> None:
    print("\nTarget step-linked official evidence structure:")
    for index, row in enumerate(evidence_rows, 1):
        assert_view(row, PASS3_EVIDENCE_FIELDS, "Pass-3 evidence display")
        location = (
            f"annotator={row['official_evidence_annotator_index']} "
            f"step={row['decomposition_step_index']} group={row['evidence_group_index']}"
        )
        if row["record_type"] == "PARAGRAPH":
            occurrence = row["paragraph_occurrence_index"]
            print(
                f"  [{index}] {location} occurrence={occurrence} | "
                f"paragraph={row['paragraph_id']} | title={row['paragraph_title']} | "
                f"section={row['section']} | headers={row['headers']}"
            )
        else:
            print(f"  [{index}] {location} | marker={row['marker']}")


def display_pass3_view(view: dict[str, Any]) -> None:
    assert_view(view, PASS3_VIEW_FIELDS, "Pass-3 display")
    print_pass_heading("PASS 3 — BRIDGE VALIDATION", view["progress"])
    print(f"Urdu question:\n{view['question_ur']}")
    print(f"\nTarget English title:\n{view['gold_title']}")
    print("\nOfficial decomposition:")
    for index, step in enumerate(view["official_decomposition"], 1):
        print(f"  {index}. {step}")
    display_evidence_structure(view["target_official_evidence"])
    print("\nSupporting paragraph text is hidden. Use V only when needed.")


def display_adjudication_view(view: dict[str, Any]) -> None:
    assert_view(view, ADJUDICATION_VIEW_FIELDS, "Adjudication display")
    print_pass_heading("ADJUDICATION", view["progress"])
    print(f"Urdu question:\n{view['question_ur']}")
    print(f"\nTarget English title:\n{view['gold_title']}")
    print("\nIndependent annotations:")
    for annotation in view["independent_annotations"]:
        assert_view(annotation, INDEPENDENT_ANNOTATION_VIEW_FIELDS, "Independent annotation display")
        print(f"  Annotator: {annotation['annotator_id']}")
        print(f"    role={annotation['source_role']} confidence={annotation['confidence']}")
        print(f"    relation={annotation['explicit_relation_type']}")
        print(f"    rationale={annotation['rationale']}")
        print(f"    intermediate={annotation['concrete_intermediate_information']}")
        print(f"    dependency={annotation['dependency_status']}")
        print(f"    proposed parents={annotation['proposed_parent_source_titles']}")
        print(f"    official support={annotation['official_support_path_references']}")
    print("\nOfficial decomposition:")
    for index, step in enumerate(view["official_decomposition"], 1):
        print(f"  {index}. {step}")
    display_evidence_structure(view["target_official_evidence"])
    print("\nSupporting text and the gold answer are hidden by default.")


def view_supporting_text(
    data: FrozenData,
    source_id: str,
    annotator_id: str,
    annotation_pass: str,
    annotation_round: int,
    evidence_rows: Sequence[dict[str, Any]],
    persisted_references: list[str],
) -> None:
    while True:
        raw = prompt_raw("Paragraph number to view (Q cancel): ").strip()
        if raw.upper() == "Q":
            return
        try:
            index = int(raw)
        except ValueError:
            print("Enter a displayed paragraph number or Q.")
            continue
        if not 1 <= index <= len(evidence_rows):
            print("That number is outside the displayed evidence range.")
            continue
        row = evidence_rows[index - 1]
        if row["record_type"] != "PARAGRAPH":
            print("Markers have no supporting paragraph text.")
            continue
        paragraph_id = row["paragraph_id"]
        paragraph = data.paragraphs().get(paragraph_id)
        require(paragraph is not None, "Displayed paragraph ID is absent from official paragraph metadata")
        require(paragraph["title"] == row["paragraph_title"], "Paragraph title mismatch before text display")
        require(paragraph["para_index"] == row["paragraph_index"], "Paragraph index mismatch before text display")
        validate_supporting_view_permission(
            data,
            source_id,
            annotator_id,
            annotation_pass,
            row,
        )
        reference = evidence_reference(row, supporting_text_viewed=True)
        event = {
            "event_type": "SUPPORTING_TEXT_VIEW",
            "source_instance_id": source_id,
            "annotator_id": annotator_id,
            "timestamp": utc_timestamp(),
            "round": annotation_round,
            "annotation_pass": annotation_pass,
            "official_support_path_references": [reference],
        }
        append_records(data, [event])
        print(f"\n--- SUPPORTING TEXT [{index}] ---")
        print(paragraph["content"])
        print("--- END SUPPORTING TEXT ---")
        persisted_references.append(reference)
        return


def validate_supporting_view_permission(
    data: FrozenData,
    source_id: str,
    annotator_id: str,
    annotation_pass: str,
    evidence_row: dict[str, Any],
) -> None:
    require(source_id in data.master_by_id, "Supporting-text view has an unknown source")
    require(annotation_pass in SENSITIVE_EVENT_PASSES, "Supporting-text view has an invalid pass")
    require(evidence_row["record_type"] == "PARAGRAPH", "Only supporting paragraphs may be revealed")
    allowed_paths = {
        tuple(row["raw_path"])
        for row in target_step_evidence(data, source_id)
        if row["record_type"] == "PARAGRAPH"
    }
    require(
        tuple(evidence_row["raw_path"]) in allowed_paths,
        "Supporting-text view is not permitted for this source instance",
    )
    _, latest, _ = load_history(data)
    if annotation_pass == "PASS_3":
        role = annotator_id.split(":", 1)[0]
        require(role in {"PRIMARY", "RELIABILITY"}, "Pass-3 view requires an independent annotator")
        require(source_id in data.population(role), "Pass-3 view source is outside this annotator population")
        state = latest.get((source_id, annotator_id))
        require(state is not None, "Pass-3 view lacks an annotation state")
        require(
            state["pass2_decision"] == "NOT_YET_EXPLICIT"
            and state["pass3_decision"] is None,
            "Supporting text is available only during unfinished eligible Pass 3",
        )
    else:
        require(
            annotator_id.startswith("ADJUDICATOR:"),
            "Adjudication view requires an adjudicator identity",
        )
        queue, independent = adjudication_queue(data, latest)
        require(
            source_id in queue and bool(independent.get(source_id)),
            "Supporting text is unavailable outside a pending adjudication case",
        )


def sensitive_events_for(
    events: Sequence[dict[str, Any]],
    event_type: str,
    source_id: str,
    annotator_id: str,
    annotation_pass: str,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["event_type"] == event_type
        and event["source_instance_id"] == source_id
        and event["annotator_id"] == annotator_id
        and event["annotation_pass"] == annotation_pass
    ]


def persisted_support_references(
    data: FrozenData,
    source_id: str,
    annotator_id: str,
    annotation_pass: str,
    evidence_rows: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> list[str]:
    allowed_paths = {
        tuple(row["raw_path"])
        for row in evidence_rows
        if row["record_type"] == "PARAGRAPH"
    }
    references: list[str] = []
    for event in sensitive_events_for(
        events,
        "SUPPORTING_TEXT_VIEW",
        source_id,
        annotator_id,
        annotation_pass,
    ):
        for reference in event["official_support_path_references"]:
            require(
                parse_support_reference(reference) in allowed_paths,
                "Persisted supporting-text view does not belong to this item",
            )
            references.append(reference)
    return references


def select_evidence_indices(evidence_rows: Sequence[dict[str, Any]]) -> list[int]:
    paragraph_indices = {
        index for index, row in enumerate(evidence_rows, 1) if row["record_type"] == "PARAGRAPH"
    }
    while True:
        raw = prompt_nonempty("Official support paragraph number(s), comma-separated: ")
        try:
            selected = [int(part.strip()) for part in raw.split(",")]
        except ValueError:
            print("Use comma-separated displayed paragraph numbers.")
            continue
        if not selected or any(index not in paragraph_indices for index in selected):
            print("Every support reference must be a displayed paragraph row.")
            continue
        return list(dict.fromkeys(selected))


def evidence_reference(row: dict[str, Any], supporting_text_viewed: bool) -> str:
    value = {
        "raw_path": row["raw_path"],
        "supporting_text_viewed": supporting_text_viewed,
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_pass1_records(
    data: FrozenData,
    source_ids: Sequence[str],
    annotator_id: str,
    candidates: list[str],
    annotation_round: int,
) -> list[dict[str, Any]]:
    timestamp = utc_timestamp()
    records: list[dict[str, Any]] = []
    for source_id in source_ids:
        record = empty_annotation_record(data.schema, source_id, annotator_id)
        record.update(
            {
                "pass1_completed": True,
                "blind_discovery_candidates": list(candidates),
                "pass1_timestamp": timestamp,
                "pass1_round": annotation_round,
                "annotation_timestamp": timestamp,
                "round": annotation_round,
            }
        )
        records.append(record)
    return records


def repair_partial_pass1(
    data: FrozenData,
    role: str,
    qid: str,
    annotator_id: str,
    states: dict[str, dict[str, Any]],
    candidates: list[str],
) -> None:
    source_ids = data.sources_for_pass1_qid(role, qid)
    existing = [states[source_id] for source_id in source_ids if source_id in states]
    require(existing, "Cannot repair Pass 1 without a persisted source record")
    exemplar = existing[0]
    require(exemplar["pass1_completed"], "Partial Pass-1 exemplar is unfinished")
    missing = [source_id for source_id in source_ids if source_id not in states]
    records = make_pass1_records(
        data,
        missing,
        annotator_id,
        candidates,
        exemplar["pass1_round"],
    )
    for record in records:
        record["pass1_timestamp"] = exemplar["pass1_timestamp"]
        record["annotation_timestamp"] = exemplar["annotation_timestamp"]
        record["round"] = exemplar["round"]
    if records:
        append_records(data, records)


def run_pass1(data: FrozenData, role: str, annotator_id: str, annotation_round: int) -> None:
    _, latest, _ = load_history(data)
    states = states_for_annotator(latest, annotator_id)
    qids = data.pass1_qids(role)
    for position, qid in enumerate(qids, 1):
        complete, persisted_candidates = qid_pass1_state(data, role, qid, states)
        if complete:
            continue
        if persisted_candidates is not None:
            repair_partial_pass1(data, role, qid, annotator_id, states, persisted_candidates)
            _, latest, _ = load_history(data)
            states = states_for_annotator(latest, annotator_id)
            continue
        view = build_pass1_view(data, qid, f"{position}/{len(qids)}")
        display_pass1_view(view)
        candidates = prompt_candidates()
        append_records(
            data,
            make_pass1_records(
                data,
                data.sources_for_pass1_qid(role, qid),
                annotator_id,
                candidates,
                annotation_round,
            ),
        )
        _, latest, _ = load_history(data)
        states = states_for_annotator(latest, annotator_id)
    counts = progress_counts(data, role, states)
    require(counts["pass1_complete"] == counts["pass1_total"], "Pass 1 did not finish")
    print(f"PASS 1 COMPLETE AND LOGICALLY FROZEN: {counts['pass1_complete']}/{counts['pass1_total']}")


def make_pass2_record(
    data: FrozenData,
    previous: dict[str, Any],
    decision: str,
    annotation_round: int,
    relation: str | None = None,
    span: str | None = None,
    rationale: str | None = None,
    confidence: str | None = None,
) -> dict[str, Any]:
    record = deepcopy(previous)
    timestamp = utc_timestamp()
    record.update(
        {
            "pass2_decision": decision,
            "pass2_timestamp": timestamp,
            "pass2_round": annotation_round,
            "annotation_timestamp": timestamp,
            "round": annotation_round,
        }
    )
    if decision == "EXPLICIT":
        record.update(
            {
                "source_role": "EXPLICIT",
                "explicit_relation_type": relation,
                "urdu_span_if_explicit": span,
                "rationale": rationale,
                "confidence": confidence,
            }
        )
    return record


def collect_pass2_record(
    data: FrozenData,
    previous: dict[str, Any],
    view: dict[str, Any],
    annotation_round: int,
) -> dict[str, Any]:
    while True:
        display_pass2_view(view)
        decision = prompt_choice(
            "Decision [E/N/H/Q]: ",
            {"E": "EXPLICIT", "N": "NOT_YET_EXPLICIT"},
            "E=EXPLICIT, N=NOT_YET_EXPLICIT, H=help, Q=quit safely.",
        )
        if decision == "NOT_YET_EXPLICIT":
            return make_pass2_record(data, previous, decision, annotation_round)
        print(
            "Relation: 1 direct mention; 2 transliteration; 3 alias/abbreviation; "
            "4 morphology/demonym; 5 direct specific concept"
        )
        relation = prompt_choice(
            "Relation [1-5]: ",
            {str(index): relation for index, relation in enumerate(EXPLICIT_RELATIONS, 1)},
            "Choose 1-5; Q quits safely.",
        )
        span = prompt_optional("Urdu span if applicable (ENTER=not applicable, :q quit): ")
        rationale = prompt_nonempty("Short rationale (:q quit): ")
        confidence = prompt_confidence()
        action = prompt_choice(
            "Save EXPLICIT decision? [Y/R/Q]: ",
            {"Y": "SAVE", "R": "RESTART"},
            "Y saves; R restarts this item; Q quits safely.",
        )
        if action == "RESTART":
            continue
        return make_pass2_record(
            data,
            previous,
            decision,
            annotation_round,
            relation,
            span,
            rationale,
            confidence,
        )


def run_pass2(data: FrozenData, role: str, annotator_id: str, annotation_round: int) -> None:
    _, latest, _ = load_history(data)
    states = states_for_annotator(latest, annotator_id)
    counts = progress_counts(data, role, states)
    require(
        counts["pass1_complete"] == counts["pass1_total"],
        f"Pass 2 is blocked until all {counts['pass1_total']} required Pass-1 questions are complete",
    )
    population = data.population(role)
    for position, source_id in enumerate(population, 1):
        previous = states.get(source_id)
        require(previous is not None and previous["pass1_completed"], "Pass-2 item lacks frozen Pass 1")
        if previous["pass2_decision"] is not None:
            continue
        view = build_pass2_view(
            data,
            source_id,
            previous["blind_discovery_candidates"],
            f"{position}/{len(population)}",
        )
        record = collect_pass2_record(data, previous, view, annotation_round)
        append_records(data, [record])
        _, latest, _ = load_history(data)
        states = states_for_annotator(latest, annotator_id)
    counts = progress_counts(data, role, states)
    require(counts["pass2_complete"] == counts["pass2_total"], "Pass 2 did not finish")
    print(f"PASS 2 COMPLETE: {counts['pass2_complete']}/{counts['pass2_total']}")


def prompt_parent_titles(dependency: str) -> list[str]:
    if dependency in {"PARALLEL_OR_UNORDERED", "UNRESOLVED", "NOT_APPLICABLE"}:
        return []
    while True:
        value = prompt_raw(
            "Defensible parent source title(s) (ENTER=none, separate multiple with |, :q quit): "
        ).strip()
        if value.lower() in {":q", "q"}:
            raise QuitRequested
        if not value:
            return []
        try:
            parents = parse_delimited_names(value)
        except Stage0Error as exc:
            print(exc)
            continue
        if dependency == "MULTIPLE_PLAUSIBLE_PARENTS" and len(parents) < 2:
            print("MULTIPLE_PLAUSIBLE_PARENTS requires at least two proposed titles.")
            continue
        if dependency != "MULTIPLE_PLAUSIBLE_PARENTS" and len(parents) > 1:
            print("Multiple titles are allowed only for MULTIPLE_PLAUSIBLE_PARENTS.")
            continue
        return parents


def make_pass3_record(
    previous: dict[str, Any],
    decision: str,
    annotation_round: int,
    rationale: str,
    confidence: str,
    evidence_references: list[str],
    concrete_information: str | None = None,
    dependency: str | None = None,
    parents: list[str] | None = None,
) -> dict[str, Any]:
    record = deepcopy(previous)
    timestamp = utc_timestamp()
    parent_values = list(parents or [])
    record.update(
        {
            "pass3_decision": decision,
            "pass3_timestamp": timestamp,
            "pass3_round": annotation_round,
            "source_role": decision,
            "confidence": confidence,
            "rationale": rationale,
            "official_support_path_references": list(evidence_references),
            "annotation_timestamp": timestamp,
            "round": annotation_round,
        }
    )
    if decision == "LATENT_BRIDGE":
        record.update(
            {
                "concrete_intermediate_information": concrete_information,
                "dependency_status": dependency,
                "proposed_parent_source_titles": parent_values,
                "parent_source_titles": parent_values,
            }
        )
    return record


def collect_pass3_record(
    data: FrozenData,
    source_id: str,
    annotator_id: str,
    previous: dict[str, Any],
    view: dict[str, Any],
    annotation_round: int,
    viewed_references: list[str],
) -> dict[str, Any]:
    while True:
        display_pass3_view(view)
        if viewed_references:
            print(f"Persisted supporting-text reveal events for this item: {len(viewed_references)}")
        while True:
            decision = prompt_choice(
                "Decision [L/A/V/H/Q]: ",
                {"L": "LATENT_BRIDGE", "A": "AMBIGUOUS", "V": "VIEW_TEXT"},
                "L=LATENT_BRIDGE, A=AMBIGUOUS, V=view supporting text, H=help, Q=quit safely.",
            )
            if decision == "VIEW_TEXT":
                view_supporting_text(
                    data,
                    source_id,
                    annotator_id,
                    "PASS_3",
                    annotation_round,
                    view["target_official_evidence"],
                    viewed_references,
                )
                continue
            break

        selected_indices: list[int] = []
        concrete: str | None = None
        dependency: str | None = None
        parents: list[str] = []
        if decision == "LATENT_BRIDGE":
            concrete = prompt_nonempty("Concrete intermediate information (:q quit): ")
            print(
                "Dependency: 1 clear; 2 multiple plausible parents; "
                "3 parallel/unordered; 4 unresolved; 5 not applicable"
            )
            dependency = prompt_choice(
                "Dependency [1-5]: ",
                {str(index): status for index, status in enumerate(DEPENDENCY_STATUSES, 1)},
                "Choose 1-5; no dependency is inferred automatically; Q quits safely.",
            )
            parents = prompt_parent_titles(dependency)
            selected_indices = select_evidence_indices(view["target_official_evidence"])

        rationale = prompt_nonempty("Short rationale (:q quit): ")
        confidence = prompt_confidence()
        action = prompt_choice(
            f"Save {decision} decision? [Y/R/Q]: ",
            {"Y": "SAVE", "R": "RESTART"},
            "Y saves; R restarts this item; Q quits safely.",
        )
        if action == "RESTART":
            continue

        viewed_paths = {parse_support_reference(reference) for reference in viewed_references}
        references = list(dict.fromkeys(viewed_references))
        for index in selected_indices:
            evidence_row = view["target_official_evidence"][index - 1]
            reference = evidence_reference(
                evidence_row,
                supporting_text_viewed=tuple(evidence_row["raw_path"]) in viewed_paths,
            )
            if reference not in references:
                references.append(reference)
        return make_pass3_record(
            previous,
            decision,
            annotation_round,
            rationale,
            confidence,
            references,
            concrete,
            dependency,
            parents,
        )


def run_pass3(data: FrozenData, role: str, annotator_id: str, annotation_round: int) -> None:
    _, latest, sensitive_events = load_history(data)
    states = states_for_annotator(latest, annotator_id)
    counts = progress_counts(data, role, states)
    require(
        counts["pass2_complete"] == counts["pass2_total"],
        f"Pass 3 is blocked until all {counts['pass2_total']} required Pass-2 decisions are complete",
    )
    required_ids = [
        source_id
        for source_id in data.population(role)
        if states[source_id]["pass2_decision"] == "NOT_YET_EXPLICIT"
    ]
    for position, source_id in enumerate(required_ids, 1):
        previous = states[source_id]
        if previous["pass3_decision"] is not None:
            continue
        view = build_pass3_view(data, source_id, f"{position}/{len(required_ids)}")
        viewed_references = persisted_support_references(
            data,
            source_id,
            annotator_id,
            "PASS_3",
            view["target_official_evidence"],
            sensitive_events,
        )
        record = collect_pass3_record(
            data,
            source_id,
            annotator_id,
            previous,
            view,
            annotation_round,
            viewed_references,
        )
        append_records(data, [record])
        _, latest, sensitive_events = load_history(data)
        states = states_for_annotator(latest, annotator_id)
    counts = progress_counts(data, role, states)
    require(counts["pass3_complete"] == counts["pass3_required"], "Pass 3 did not finish")
    print(f"PASS 3 COMPLETE: {counts['pass3_complete']}/{counts['pass3_required']}")


def raw_independent_states_by_source(
    latest: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (source_id, _), state in latest.items():
        if state["source_role"] in SOURCE_ROLES and state["adjudicator_id"] is None:
            grouped[source_id].append(state)
    for states in grouped.values():
        states.sort(key=lambda state: state["annotator_id"])
    return grouped


def adjudication_queue(
    data: FrozenData,
    latest: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    independent = raw_independent_states_by_source(latest)
    already_adjudicated = {
        source_id
        for (source_id, _), state in latest.items()
        if state["adjudicated_role"] in SOURCE_ROLES
    }
    eligible: set[str] = set()
    for source_id, states in independent.items():
        labels = {state["source_role"] for state in states}
        disagreement = len(states) >= 2 and len(labels) > 1
        ambiguous = any(state["source_role"] == "AMBIGUOUS" for state in states)
        low_confidence = any(state["confidence"] == "LOW" for state in states)
        if disagreement or ambiguous or low_confidence:
            eligible.add(source_id)
    ordered = [
        source_id
        for source_id in data.pass2_ids
        if source_id in eligible and source_id not in already_adjudicated
    ]
    return ordered, independent


def prompt_yes_no(prompt: str) -> bool:
    return (
        prompt_choice(
            prompt,
            {"Y": "YES", "N": "NO"},
            "Y=yes, N=no, Q=quit safely.",
        )
        == "YES"
    )


def make_adjudication_record(
    data: FrozenData,
    source_id: str,
    adjudicator_id: str,
    annotation_round: int,
    role: str,
    rationale: str,
    changed_eligibility: bool,
    disagreement: bool,
    answer_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    timestamp = utc_timestamp()
    completed_answer_events = [
        {
            "reason": event["reason"],
            "viewer": event["annotator_id"],
            "timestamp": event["timestamp"],
            "round": event["round"],
            "changed_decision": role != event["provisional_role"],
        }
        for event in answer_events
    ]
    record = empty_annotation_record(data.schema, source_id, adjudicator_id)
    record.update(
        {
            "disagreement_flag": disagreement,
            "adjudication_required": True,
            "adjudicated_role": role,
            "adjudication_rationale": rationale,
            "adjudication_changed_bridge_eligibility": changed_eligibility,
            "adjudicator_id": adjudicator_id,
            "adjudication_timestamp": timestamp,
            "adjudication_round": annotation_round,
            "answer_inspection_exception_used": bool(completed_answer_events),
            "answer_inspection_exceptions": completed_answer_events,
            "annotation_timestamp": timestamp,
            "round": annotation_round,
        }
    )
    return record


def validate_answer_inspection_permission(
    data: FrozenData,
    source_id: str,
    adjudicator_id: str,
) -> None:
    require(
        adjudicator_id.startswith("ADJUDICATOR:"),
        "Gold-answer inspection requires an adjudicator identity",
    )
    _, latest, _ = load_history(data)
    queue, independent = adjudication_queue(data, latest)
    require(
        source_id in queue and bool(independent.get(source_id)),
        "Gold-answer inspection is unavailable outside a pending adjudication case",
    )


def persisted_answer_inspections(
    events: Sequence[dict[str, Any]],
    source_id: str,
    adjudicator_id: str,
) -> list[dict[str, Any]]:
    return sensitive_events_for(
        events,
        "ANSWER_INSPECTION",
        source_id,
        adjudicator_id,
        "ADJUDICATION",
    )


def collect_adjudication_record(
    data: FrozenData,
    source_id: str,
    adjudicator_id: str,
    annotation_round: int,
    states: Sequence[dict[str, Any]],
    view: dict[str, Any],
) -> dict[str, Any]:
    _, _, history_events = load_history(data)
    viewed_references = persisted_support_references(
        data,
        source_id,
        adjudicator_id,
        "ADJUDICATION",
        view["target_official_evidence"],
        history_events,
    )
    answer_events = persisted_answer_inspections(
        history_events,
        source_id,
        adjudicator_id,
    )
    while True:
        display_adjudication_view(view)
        if viewed_references:
            print(f"Persisted supporting-text reveal events for this item: {len(viewed_references)}")
        if answer_events:
            print(f"Persisted gold-answer inspection events for this item: {len(answer_events)}")
        while True:
            decision = prompt_choice(
                "Final decision [E/L/A/V/G/H/Q]: ",
                {
                    "E": "EXPLICIT",
                    "L": "LATENT_BRIDGE",
                    "A": "AMBIGUOUS",
                    "V": "VIEW_TEXT",
                    "G": "VIEW_ANSWER",
                },
                "E/L/A decide; V views supporting text; G requests the logged gold-answer exception; Q quits.",
            )
            if decision == "VIEW_TEXT":
                view_supporting_text(
                    data,
                    source_id,
                    adjudicator_id,
                    "ADJUDICATION",
                    annotation_round,
                    view["target_official_evidence"],
                    viewed_references,
                )
                continue
            if decision == "VIEW_ANSWER":
                answer_reason = prompt_nonempty("Reason answer inspection is necessary (:q quit): ")
                provisional_role = prompt_choice(
                    "Record provisional role before inspection [E/L/A]: ",
                    {"E": "EXPLICIT", "L": "LATENT_BRIDGE", "A": "AMBIGUOUS"},
                    "A provisional role is required to log whether answer viewing changed the decision.",
                )
                validate_answer_inspection_permission(data, source_id, adjudicator_id)
                answer_event = {
                    "event_type": "ANSWER_INSPECTION",
                    "source_instance_id": source_id,
                    "annotator_id": adjudicator_id,
                    "timestamp": utc_timestamp(),
                    "round": annotation_round,
                    "annotation_pass": "ADJUDICATION",
                    "reason": answer_reason,
                    "provisional_role": provisional_role,
                    "changed_decision": None,
                }
                append_records(data, [answer_event])
                answer_events.append(answer_event)
                answer = data.dev_by_qid[data.qid_by_source[source_id]]["answer"]
                require(type(answer) is bool, "Gold answer is not boolean")
                print(f"\nGOLD ANSWER EXCEPTION: {'YES' if answer else 'NO'}")
                print("The answer value will not be written to the annotation log.")
                continue
            final_role = decision
            break

        rationale = prompt_nonempty("Adjudication rationale (:q quit): ")
        changed_eligibility = prompt_yes_no("Did adjudication change bridge eligibility? [Y/N]: ")
        action = prompt_choice(
            f"Save adjudicated role {final_role}? [Y/R/Q]: ",
            {"Y": "SAVE", "R": "RESTART"},
            "Y saves; R restarts this item; Q quits safely.",
        )
        if action == "RESTART":
            continue

        labels = {state["source_role"] for state in states}
        return make_adjudication_record(
            data,
            source_id,
            adjudicator_id,
            annotation_round,
            final_role,
            rationale,
            changed_eligibility,
            len(states) >= 2 and len(labels) > 1,
            answer_events,
        )


def run_adjudication(data: FrozenData, adjudicator_id: str, annotation_round: int) -> None:
    _, latest, history_events = load_history(data)
    own_states = states_for_annotator(latest, adjudicator_id)
    require(
        not any(state["source_role"] in SOURCE_ROLES for state in own_states.values()),
        "An independent annotator identity cannot be reused for adjudication",
    )
    queue, independent = adjudication_queue(data, latest)
    require(queue, "No eligible completed independent annotation currently requires adjudication")
    for position, source_id in enumerate(queue, 1):
        pending_adjudicators = {
            event["annotator_id"]
            for event in history_events
            if event["source_instance_id"] == source_id
            and event["annotation_pass"] == "ADJUDICATION"
        }
        require(
            not pending_adjudicators or pending_adjudicators == {adjudicator_id},
            "Pending sensitive-view history requires resuming with its original adjudicator identity",
        )
        states = independent[source_id]
        require(states, "Adjudication is forbidden without independent annotation")
        view = build_adjudication_view(data, source_id, states, f"{position}/{len(queue)}")
        record = collect_adjudication_record(
            data,
            source_id,
            adjudicator_id,
            annotation_round,
            states,
            view,
        )
        append_records(data, [record])
    print("ADJUDICATION QUEUE COMPLETE")


def print_progress(data: FrozenData, role: str, annotator_id: str) -> None:
    require(role in {"PRIMARY", "RELIABILITY"}, "--progress supports PRIMARY or RELIABILITY")
    _, latest, _ = load_history(data)
    states = states_for_annotator(latest, annotator_id)
    counts = progress_counts(data, role, states)
    print(f"ANNOTATOR: {annotator_id}")
    print(f"Pass 1 complete / total: {counts['pass1_complete']} / {counts['pass1_total']}")
    print(f"Pass 2 complete / total: {counts['pass2_complete']} / {counts['pass2_total']}")
    print(f"Pass 2 EXPLICIT: {counts['pass2_explicit']}")
    print(f"Pass 2 NOT_YET_EXPLICIT: {counts['pass2_not_yet_explicit']}")
    print(f"Pass 3 complete / required: {counts['pass3_complete']} / {counts['pass3_required']}")
    print(f"Pass 3 LATENT_BRIDGE: {counts['pass3_latent_bridge']}")
    print(f"Pass 3 AMBIGUOUS: {counts['pass3_ambiguous']}")
    if role == "RELIABILITY":
        print("Primary/reliability agreement is intentionally hidden during independent annotation.")


def audit_frozen_counts(data: FrozenData) -> None:
    require(len({row["urbench_qid"] for row in data.master_rows}) == 200, "Master qid count changed")
    require(len({row["gold_title"] for row in data.master_rows}) == 613, "Distinct raw-title count changed")
    require(
        Counter(row["exact_corpus_status"] for row in data.master_rows)
        == {"EXACT_PRESENT": 445, "EXACT_ABSENT": 191},
        "Exact-title status counts changed",
    )
    evidence_types = Counter(row["record_type"] for row in data.evidence_rows)
    markers = Counter(row["marker"] for row in data.evidence_rows if row["record_type"] == "MARKER")
    require(evidence_types == {"PARAGRAPH": 1367, "MARKER": 837}, "Evidence row counts changed")
    require(markers == {"operation": 528, "no_evidence": 309}, "Evidence marker counts changed")
    lexical_classes = Counter(row["lexical_trace_class"] for row in data.lexical_rows)
    lexical_binary = Counter(row["lexical_trace_en"] for row in data.lexical_rows)
    require(lexical_classes == {"EXACT": 256, "PARTIAL": 46, "ABSENT": 334}, "Lexical counts changed")
    require(lexical_binary == {"YES": 302, "NO": 334}, "Lexical binary counts changed")
    require(
        data.reliability_manifest.get("stratum_counts")
        == {
            "NO__EXACT_ABSENT": 83,
            "NO__EXACT_PRESENT": 251,
            "YES__EXACT_ABSENT": 108,
            "YES__EXACT_PRESENT": 194,
        },
        "Reliability stratum counts changed",
    )


def run_audit(data: FrozenData) -> None:
    audit_frozen_counts(data)
    rows, _, sensitive_events = load_history(data)
    unique_reliability_qids = len(data.reliability_qids)
    multiplicities = Counter(
        Counter(data.qid_by_source[source_id] for source_id in data.reliability_ids).values()
    )
    print("EFBPT STAGE-0 ANNOTATION RUNNER AUDIT")
    print("INPUTS: all required manifests readable; frozen hashes and commits verified")
    print("PRIMARY:")
    print("  Pass1 questions = 200")
    print("  Pass2 instances = 636")
    print("  fixed decisions before Pass3 = 836")
    print("  Pass3 size = UNKNOWN until Pass2")
    print("RELIABILITY:")
    print("  sample instances = 160")
    print(f"  unique reliability qids = {unique_reliability_qids}")
    print(f"  fixed decisions before Pass3 = {unique_reliability_qids + 160}")
    print("  Pass3 size = UNKNOWN until reliability Pass2")
    print("  possible Pass3 workload = 0 to 160")
    print("  qid multiplicity distribution (instances-per-qid -> qid count):")
    for multiplicity in sorted(multiplicities):
        print(f"    {multiplicity} -> {multiplicities[multiplicity]}")
    print("PERMITTED DISPLAY FIELDS:")
    print("  Pass1: progress, question_ur")
    print("  Pass2: progress, question_ur, one gold_title, own blind_discovery_candidates")
    print("  Pass3: progress, question_ur, target gold_title, official_decomposition, target step-linked evidence identity")
    print("  Pass3 supporting text: hidden by default; explicit V command only; viewing is logged")
    print("  Adjudication: independent labels/rationales plus Pass-3-permitted evidence")
    print(f"HUMAN LOG EXISTS: {'YES' if HUMAN_LOG_PATH.exists() else 'NO'}")
    print(f"ANNOTATION EVENTS: {len(rows)}")
    print(f"SENSITIVE-VIEW EVENTS: {len(sensitive_events)}")
    print(f"ANNOTATION STARTED: {'YES' if rows else 'NO'}")
    print("AUTOMATIC LABELING: NONE")
    print("MODEL/LLM ASSISTANCE: NONE")
    print("STAGE0 ANNOTATION RUNNER AUDIT: SUCCESS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Terminal-only, staged EFBPT Stage-0 human annotation runner."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true", help="validate frozen inputs and report aggregate workload; write nothing")
    mode.add_argument("--progress", action="store_true", help="report aggregate progress for one explicit annotator; write nothing")
    parser.add_argument("--annotator", help="explicit human annotator identifier")
    parser.add_argument("--role", choices=("PRIMARY", "RELIABILITY", "ADJUDICATOR"))
    parser.add_argument("--pass", dest="annotation_pass", choices=("1", "2", "3", "adjudication"))
    parser.add_argument("--round", dest="annotation_round", type=int, default=1)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    require(args.annotation_round >= 1, "--round must be a positive integer")
    data = FrozenData()

    if args.audit:
        require(args.annotator is None and args.role is None and args.annotation_pass is None, "--audit takes no annotator, role, or pass")
        run_audit(data)
        return 0

    require(args.annotator is not None, "--annotator is required")
    require(args.role is not None, "--role is required")
    annotator_id = qualified_annotator_id(args.role, args.annotator)

    if args.progress:
        require(args.annotation_pass is None, "--progress does not accept --pass")
        print_progress(data, args.role, annotator_id)
        return 0

    require(args.annotation_pass is not None, "--pass is required for annotation")
    if args.role == "ADJUDICATOR":
        require(args.annotation_pass == "adjudication", "ADJUDICATOR requires --pass adjudication")
    else:
        require(args.annotation_pass in {"1", "2", "3"}, "Independent annotation requires --pass 1, 2, or 3")

    commit = validate_actual_annotation_runtime()
    print(f"ANNOTATION RUNNER COMMIT: {commit}")
    print(f"ANNOTATION IDENTITY: {annotator_id}")
    if args.role == "ADJUDICATOR":
        run_adjudication(data, annotator_id, args.annotation_round)
    elif args.annotation_pass == "1":
        run_pass1(data, args.role, annotator_id, args.annotation_round)
    elif args.annotation_pass == "2":
        run_pass2(data, args.role, annotator_id, args.annotation_round)
    else:
        run_pass3(data, args.role, annotator_id, args.annotation_round)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QuitRequested:
        print("QUIT: all completed annotation events are safely saved.")
        raise SystemExit(0)
    except Stage0Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
