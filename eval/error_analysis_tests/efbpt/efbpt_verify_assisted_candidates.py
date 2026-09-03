#!/usr/bin/env python3
"""Fast human verification of model-assisted EFBPT bridge candidates.

This runner never reads assisted predictions as human labels and never writes the
canonical frozen Stage-0 human annotation log.  Its sole writable artifact is an
append-only, pair-level verification log under ``outputs/efbpt/stage0_assisted``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
ASSISTED_DIR = REPO_ROOT / "outputs/efbpt/stage0_assisted"
PASS2_PATH = ASSISTED_DIR / "assisted_pass2.jsonl"
PASS3_PATH = ASSISTED_DIR / "assisted_pass3.jsonl"
CANDIDATES_PATH = ASSISTED_DIR / "bridge_candidate_qids.jsonl"
OUTPUT_PATH = ASSISTED_DIR / "human_verified_candidates.jsonl"

STAGE0_DIR = REPO_ROOT / "data/strategyqa_official/efbpt/stage0"
MASTER_PATH = STAGE0_DIR / "source_instance_master.jsonl"
EVIDENCE_PATH = STAGE0_DIR / "official_evidence_links.jsonl"
QUESTION_PATH = STAGE0_DIR / "pass1_question_manifest.jsonl"
CANONICAL_HUMAN_LOG_PATH = STAGE0_DIR / "human_annotation_log.jsonl"

MODEL_ASSISTED_STATUS = "MODEL_ASSISTED_EXPLORATORY_NOT_HUMAN"
OUTPUT_STATUS = "HUMAN_VERIFICATION_OF_MODEL_ASSISTED_CANDIDATES"
VERIFIED = "VERIFIED_CLEAN_BRIDGE"
REJECTED = "REJECTED"

EXPECTED_SHA256 = {
    PASS2_PATH: "201e4df1995d0b04c07142246b1f4af02f5d126caae2849514cae4695a297e65",
    PASS3_PATH: "33895e498b2a71ec30ce471f9e77f8822643fa3faaea84cc5623d44877e1b420",
    CANDIDATES_PATH: "410a59d7e5fc0c22b12a947c5a9195847c807b318394e70a1fb9d4333d845b37",
    MASTER_PATH: "5ebd1968a8ac2f8013b595b69f8bd320025ca7b46d50b602e17f248ce739a084",
    EVIDENCE_PATH: "d59b956ca4003a8438653cc9930f810f5eea1e1655aba104e738152b9a137d0e",
    QUESTION_PATH: "73cb5f1b99f0b146c723c2c46aafe1cd4006ba95c251d0def7d463675ae45f18",
}

ANSWER_FIELDS = {
    "parent_directly_identifiable_from_urdu_question",
    "child_not_directly_identifiable_from_urdu_question_alone",
    "stated_intermediate_information_makes_child_identifiable_or_recoverable",
    "dependency_status",
}
BASE_OUTPUT_FIELDS = {
    "annotation_status",
    "qid",
    "parent_source_instance_id",
    "parent_title",
    "child_source_instance_id",
    "child_title",
    "human_answers",
    "verdict",
    "annotator_id",
    "timestamp",
}
ANNOTATOR_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z")


class VerificationError(RuntimeError):
    """A fatal input, audit, or append-safety failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise VerificationError(f"Cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def validate_input_hashes() -> None:
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"Required input is missing: {path}")
        require(not path.is_symlink(), f"Required input must not be a symlink: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"Frozen input hash mismatch: {path}")


def parse_jsonl_text(text: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        require(bool(line.strip()), f"Blank JSONL row at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        require(isinstance(value, dict), f"Non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"Cannot read {path}: {exc}") from exc
    return parse_jsonl_text(text, path)


def unique_index(
    rows: Iterable[dict[str, Any]], field: str, context: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        require(isinstance(value, str) and value, f"Invalid {field} in {context}")
        require(value not in result, f"Duplicate {field} {value!r} in {context}")
        result[value] = row
    return result


def support_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    raw_path = row.get("raw_path")
    require(isinstance(raw_path, list), "Official support raw_path is not a list")
    return (
        row.get("official_evidence_annotator_index"),
        row.get("decomposition_step_index"),
        row.get("evidence_group_index"),
        row.get("record_type"),
        row.get("paragraph_id"),
        row.get("paragraph_title"),
        row.get("marker"),
        tuple(raw_path),
    )


def pair_key(qid: str, pair: dict[str, Any]) -> tuple[str, str, str]:
    return (
        qid,
        str(pair.get("parent_source_instance_id")),
        str(pair.get("child_source_instance_id")),
    )


class ReviewData:
    """Validated assisted candidates joined to frozen administrative evidence."""

    def __init__(self) -> None:
        validate_input_hashes()
        self.pass2_rows = load_jsonl(PASS2_PATH)
        self.pass3_rows = load_jsonl(PASS3_PATH)
        self.candidates = load_jsonl(CANDIDATES_PATH)
        self.master_rows = load_jsonl(MASTER_PATH)
        self.evidence_rows = load_jsonl(EVIDENCE_PATH)
        self.question_rows = load_jsonl(QUESTION_PATH)

        require(len(self.pass2_rows) == 627, "Assisted Pass-2 row count is not 627")
        require(len(self.pass3_rows) == 302, "Assisted Pass-3 row count is not 302")
        require(len(self.candidates) == 30, "Ranked candidate qid count is not 30")
        require(len(self.master_rows) == 636, "Frozen source master row count is not 636")
        require(len(self.evidence_rows) == 2204, "Frozen evidence row count is not 2204")
        require(len(self.question_rows) == 200, "Frozen question row count is not 200")

        self.pass2_by_id = unique_index(self.pass2_rows, "source_instance_id", "Pass-2")
        self.pass3_by_id = unique_index(self.pass3_rows, "source_instance_id", "Pass-3")
        self.master_by_id = unique_index(self.master_rows, "source_instance_id", "source master")
        question_by_qid = unique_index(self.question_rows, "urbench_qid", "question manifest")
        self.question_by_qid = {
            qid: row.get("question_ur") for qid, row in question_by_qid.items()
        }

        self.evidence_signatures: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
        self.official_step_texts: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in self.evidence_rows:
            source_id = row.get("source_instance_id")
            qid = row.get("urbench_qid")
            step = row.get("decomposition_step_index")
            text = row.get("decomposition_text")
            require(isinstance(qid, str), "Frozen evidence qid is invalid")
            require(isinstance(step, int), "Frozen evidence step index is invalid")
            require(isinstance(text, str) and text, "Frozen evidence step text is invalid")
            self.official_step_texts[(qid, step)].add(text)
            if source_id is None:
                require(
                    row.get("record_type") == "MARKER",
                    "Only frozen marker evidence may omit source_instance_id",
                )
                continue
            require(isinstance(source_id, str) and source_id, "Frozen evidence source id is invalid")
            signature = support_signature(row)
            require(
                signature not in self.evidence_signatures[source_id],
                f"Duplicate frozen evidence signature for {source_id}",
            )
            self.evidence_signatures[source_id].add(signature)

        self.pair_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._validate_candidates()

    def _validate_candidates(self) -> None:
        require(
            [row.get("candidate_rank") for row in self.candidates]
            == list(range(1, 31)),
            "Candidate ranks must be ordered 1 through 30",
        )
        require(
            len({row.get("urbench_qid") for row in self.candidates}) == 30,
            "Candidate qids are not unique",
        )
        require(
            sum(row.get("priority_tier") == "HIGH" for row in self.candidates) == 20,
            "HIGH candidate qid count is not 20",
        )
        require(
            all(row.get("priority_tier") == "HIGH" for row in self.candidates[:20]),
            "The first 20 ranked candidate qids are not all HIGH",
        )
        require(
            all(row.get("priority_tier") == "MEDIUM" for row in self.candidates[20:]),
            "The remaining 10 ranked candidate qids are not all MEDIUM",
        )

        for candidate in self.candidates:
            qid = candidate.get("urbench_qid")
            require(isinstance(qid, str) and qid, "Candidate qid is invalid")
            require(qid in self.question_by_qid, f"Candidate qid absent from frozen questions: {qid}")
            require(
                candidate.get("annotation_status") == MODEL_ASSISTED_STATUS,
                f"Candidate is not explicitly marked model-assisted: {qid}",
            )
            require(
                candidate.get("candidate_status") == "HUMAN_VERIFICATION_CANDIDATES",
                f"Unexpected candidate status for {qid}",
            )
            require(
                candidate.get("verified_bridge_qid") is False,
                f"Candidate incorrectly claims prior verification: {qid}",
            )
            pairs = candidate.get("candidate_pairs")
            require(isinstance(pairs, list) and pairs, f"Candidate has no pairs: {qid}")
            for pair in pairs:
                require(isinstance(pair, dict), f"Malformed candidate pair for {qid}")
                self._validate_pair(qid, pair)

        require(len(self.pair_by_key) == 41, "Candidate parent-child pair count is not 41")

    def _validate_pair(self, qid: str, pair: dict[str, Any]) -> None:
        key = pair_key(qid, pair)
        require(all(value and value != "None" for value in key), f"Invalid pair ids for {qid}")
        require(key not in self.pair_by_key, f"Duplicate candidate pair: {key}")
        parent_id, child_id = key[1], key[2]
        require(parent_id != child_id, f"Parent equals child for {qid}")
        require(parent_id in self.pass2_by_id, f"Parent absent from Pass-2: {parent_id}")
        require(child_id in self.pass2_by_id, f"Child absent from Pass-2: {child_id}")
        require(child_id in self.pass3_by_id, f"Child absent from Pass-3: {child_id}")
        require(parent_id in self.master_by_id, f"Parent absent from source master: {parent_id}")
        require(child_id in self.master_by_id, f"Child absent from source master: {child_id}")

        parent = self.pass2_by_id[parent_id]
        child_pass2 = self.pass2_by_id[child_id]
        child = self.pass3_by_id[child_id]
        parent_master = self.master_by_id[parent_id]
        child_master = self.master_by_id[child_id]
        question = self.question_by_qid[qid]

        for row, label in (
            (parent, "parent Pass-2"),
            (child_pass2, "child Pass-2"),
            (child, "child Pass-3"),
        ):
            require(row.get("urbench_qid") == qid, f"Wrong qid in {label}: {key}")
            require(row.get("question_ur") == question, f"Question mismatch in {label}: {key}")
            require(
                row.get("annotation_status") == MODEL_ASSISTED_STATUS,
                f"{label} is not explicitly marked model-assisted: {key}",
            )

        require(parent.get("predicted_pass2") == "EXPLICIT", f"Parent is not predicted EXPLICIT: {key}")
        require(
            child_pass2.get("predicted_pass2") == "NOT_YET_EXPLICIT",
            f"Child was not held out as not-yet-explicit: {key}",
        )
        require(child.get("predicted_pass3") == "LATENT_BRIDGE", f"Child is not predicted latent: {key}")
        require(child.get("dependency_status") == "CLEAR_DEPENDENCY", f"Child dependency is not clear: {key}")

        parent_title = pair.get("parent_title")
        child_title = pair.get("child_title")
        require(parent_title == parent.get("target_english_title"), f"Parent title mismatch: {key}")
        require(child_title == child.get("target_english_title"), f"Child title mismatch: {key}")
        require(child_title == child_pass2.get("target_english_title"), f"Child Pass-2 title mismatch: {key}")
        require(parent_title == parent_master.get("gold_title"), f"Parent master title mismatch: {key}")
        require(child_title == child_master.get("gold_title"), f"Child master title mismatch: {key}")
        require(parent_master.get("urbench_qid") == qid, f"Parent master qid mismatch: {key}")
        require(child_master.get("urbench_qid") == qid, f"Child master qid mismatch: {key}")
        require(parent_master.get("question_ur") == question, f"Parent master question mismatch: {key}")
        require(child_master.get("question_ur") == question, f"Child master question mismatch: {key}")

        require(pair.get("parent_predicted_role") == "EXPLICIT", f"Pair parent role mismatch: {key}")
        require(pair.get("child_predicted_role") == "LATENT_BRIDGE", f"Pair child role mismatch: {key}")
        require(pair.get("parent_role_confidence") == parent.get("confidence"), f"Parent confidence mismatch: {key}")
        require(pair.get("child_role_confidence") == child.get("confidence"), f"Child confidence mismatch: {key}")
        require(
            pair.get("child_pass2_not_explicit_confidence") == child_pass2.get("confidence"),
            f"Child Pass-2 confidence mismatch: {key}",
        )
        require(pair.get("dependency_status") == child.get("dependency_status"), f"Dependency mismatch: {key}")
        require(
            pair.get("dependency_confidence") == child.get("dependency_confidence"),
            f"Dependency confidence mismatch: {key}",
        )
        require(
            pair.get("official_step_indices") == child.get("official_step_indices"),
            f"Official step mismatch: {key}",
        )
        require(
            parent_title in child.get("proposed_parent_source_titles", []),
            f"Proposed parents omit the candidate parent: {key}",
        )
        require(
            parent.get("exact_corpus_status")
            == child.get("exact_corpus_status")
            == pair.get("parent_exact_corpus_status")
            == pair.get("child_exact_corpus_status")
            == "EXACT_PRESENT",
            f"Pair did not pass exact corpus filtering: {key}",
        )

        decomposition = child.get("official_decomposition")
        steps = child.get("official_step_indices")
        supports = child.get("official_target_evidence_support")
        require(isinstance(decomposition, list) and decomposition, f"Missing decomposition: {key}")
        require(isinstance(steps, list) and steps, f"Missing official steps: {key}")
        require(isinstance(supports, list) and supports, f"Missing official support: {key}")
        for step in steps:
            require(isinstance(step, int) and 0 <= step < len(decomposition), f"Invalid official step: {key}")
            require(
                decomposition[step] in self.official_step_texts.get((qid, step), set()),
                f"Decomposition text is not in frozen official evidence: {key}, step {step}",
            )
        for support in supports:
            require(isinstance(support, dict), f"Malformed official support: {key}")
            require(
                support_signature(support) in self.evidence_signatures.get(child_id, set()),
                f"Assisted support is absent from frozen official evidence: {key}",
            )
        require(
            pair.get("official_paragraph_support_count") == len(supports),
            f"Official paragraph support count mismatch: {key}",
        )
        require(
            pair.get("distinct_official_annotators_supporting_child")
            == len({support.get("official_evidence_annotator_index") for support in supports}),
            f"Official annotator support count mismatch: {key}",
        )
        self.pair_by_key[key] = pair

    def items(self, tier: str) -> list[tuple[dict[str, Any], int, dict[str, Any]]]:
        selected = [
            candidate
            for candidate in self.candidates
            if (candidate["priority_tier"] == "HIGH") == (tier == "high")
        ]
        return [
            (candidate, pair_number, pair)
            for candidate in selected
            for pair_number, pair in enumerate(candidate["candidate_pairs"], 1)
        ]


def parse_timestamp(value: Any) -> None:
    require(isinstance(value, str), "Verification timestamp is not a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VerificationError(f"Invalid verification timestamp: {value!r}") from exc
    require(parsed.tzinfo is not None, f"Verification timestamp lacks timezone: {value!r}")


def validate_annotator(value: Any) -> str:
    require(
        isinstance(value, str) and ANNOTATOR_PATTERN.fullmatch(value) is not None,
        "Annotator id must be 1-80 characters using letters, digits, '.', '_', ':', or '-'",
    )
    return value


def validate_existing_rows(
    rows: list[dict[str, Any]], data: ReviewData
) -> dict[tuple[str, str, str], dict[str, Any]]:
    decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        context = f"existing verification row {row_number}"
        verdict = row.get("verdict")
        expected_fields = BASE_OUTPUT_FIELDS | ({"rejection_reason"} if verdict == REJECTED else set())
        require(set(row) == expected_fields, f"Unexpected fields in {context}")
        require(row.get("annotation_status") == OUTPUT_STATUS, f"Wrong annotation status in {context}")
        key = (
            row.get("qid"),
            row.get("parent_source_instance_id"),
            row.get("child_source_instance_id"),
        )
        require(all(isinstance(value, str) for value in key), f"Invalid ids in {context}")
        require(key in data.pair_by_key, f"Unknown candidate pair in {context}: {key}")
        require(key not in decisions, f"Duplicate decision would violate append-only resume safety: {key}")
        pair = data.pair_by_key[key]
        require(row.get("parent_title") == pair.get("parent_title"), f"Parent title mismatch in {context}")
        require(row.get("child_title") == pair.get("child_title"), f"Child title mismatch in {context}")

        answers = row.get("human_answers")
        require(isinstance(answers, dict) and set(answers) == ANSWER_FIELDS, f"Invalid answers in {context}")
        require(
            answers["parent_directly_identifiable_from_urdu_question"] in {"Y", "N"},
            f"Invalid parent answer in {context}",
        )
        require(
            answers["child_not_directly_identifiable_from_urdu_question_alone"] in {"Y", "N"},
            f"Invalid child answer in {context}",
        )
        require(
            answers["stated_intermediate_information_makes_child_identifiable_or_recoverable"] in {"Y", "N"},
            f"Invalid intermediate-information answer in {context}",
        )
        require(answers["dependency_status"] in {"C", "O"}, f"Invalid dependency answer in {context}")
        expected_verdict = (
            VERIFIED
            if list(answers.values()).count("Y") == 3 and answers["dependency_status"] == "C"
            else REJECTED
        )
        require(verdict == expected_verdict, f"Verdict does not follow the four answers in {context}")
        if verdict == REJECTED:
            reason = row.get("rejection_reason")
            require(isinstance(reason, str) and bool(reason.strip()), f"Missing rejection reason in {context}")
        validate_annotator(row.get("annotator_id"))
        parse_timestamp(row.get("timestamp"))
        decisions[key] = row
    return decisions


def ensure_output_target_safe() -> None:
    require(OUTPUT_PATH.parent == ASSISTED_DIR, "Output path escaped the assisted directory")
    require(OUTPUT_PATH != CANONICAL_HUMAN_LOG_PATH, "Output path aliases the canonical human log")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        require(not OUTPUT_PATH.is_symlink(), f"Refusing symlink output: {OUTPUT_PATH}")
        require(OUTPUT_PATH.is_file(), f"Output exists but is not a regular file: {OUTPUT_PATH}")


def load_existing(data: ReviewData) -> dict[tuple[str, str, str], dict[str, Any]]:
    ensure_output_target_safe()
    if not OUTPUT_PATH.exists():
        return {}
    return validate_existing_rows(load_jsonl(OUTPUT_PATH), data)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def output_snapshot() -> tuple[int, int, str] | None:
    ensure_output_target_safe()
    if not OUTPUT_PATH.exists():
        return None
    stat = OUTPUT_PATH.stat()
    return stat.st_size, stat.st_mtime_ns, sha256_file(OUTPUT_PATH)


def canonical_snapshot() -> tuple[int, int, str]:
    require(CANONICAL_HUMAN_LOG_PATH.is_file(), "Canonical human log is missing")
    require(not CANONICAL_HUMAN_LOG_PATH.is_symlink(), "Canonical human log is a symlink")
    stat = CANONICAL_HUMAN_LOG_PATH.stat()
    return stat.st_size, stat.st_mtime_ns, sha256_file(CANONICAL_HUMAN_LOG_PATH)


def append_decision(record: dict[str, Any], data: ReviewData) -> bool:
    """Append and fsync one decision, refusing duplicates under an exclusive lock."""
    ensure_output_target_safe()
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(OUTPUT_PATH, flags, 0o644)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise VerificationError(f"Refusing symlink output: {OUTPUT_PATH}") from exc
        raise VerificationError(f"Cannot open append-only output {OUTPUT_PATH}: {exc}") from exc

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as reader:
            rows = parse_jsonl_text(reader.read(), OUTPUT_PATH)
        existing = validate_existing_rows(rows, data)
        key = (
            record["qid"],
            record["parent_source_instance_id"],
            record["child_source_instance_id"],
        )
        if key in existing:
            return False

        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "Append-only output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(OUTPUT_PATH.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def grouped_support(supports: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[int, str, str], set[int]] = defaultdict(set)
    for support in supports:
        key = (
            support["decomposition_step_index"],
            support["paragraph_title"],
            support["paragraph_id"],
        )
        grouped[key].add(support["official_evidence_annotator_index"])
    lines = []
    for (step, title, paragraph_id), annotators in sorted(grouped.items()):
        joined = ", ".join(str(value) for value in sorted(annotators))
        lines.append(
            f"    - step {step + 1}: {title} [{paragraph_id}]; official annotator(s): {joined}"
        )
    return lines


def display_candidate(
    data: ReviewData,
    candidate: dict[str, Any],
    pair_number: int,
    pair: dict[str, Any],
    decision_number: int,
    total_decisions: int,
) -> None:
    qid = candidate["urbench_qid"]
    parent = data.pass2_by_id[pair["parent_source_instance_id"]]
    child = data.pass3_by_id[pair["child_source_instance_id"]]
    parent_master = data.master_by_id[pair["parent_source_instance_id"]]
    pair_total = len(candidate["candidate_pairs"])

    print("\n" + "=" * 78)
    print(
        f"CANDIDATE {decision_number}/{total_decisions} | qid rank "
        f"{candidate['candidate_rank']}/30 | pair {pair_number}/{pair_total}"
    )
    print(f"qid: {qid}")
    print(f"Urdu question: {data.question_by_qid[qid]}")
    print("\nPARENT:")
    print(f"  Predicted parent English title: {pair['parent_title']}")
    print(f"  Why model predicted EXPLICIT: {parent['rationale']}")
    print(f"  Predicted relation: {parent['likely_relation']}")
    span = parent_master.get("urdu_span_if_explicit")
    if isinstance(span, str) and span.strip():
        print(f"  Relevant Urdu span: {span}")

    print("\nCHILD:")
    print(f"  Predicted latent child English title: {pair['child_title']}")
    print(f"  Concrete intermediate information: {child['concrete_intermediate_information']}")
    print("  Relevant official decomposition:")
    for step in child["official_step_indices"]:
        print(f"    - step {step + 1}: {child['official_decomposition'][step]}")
    print("  Relevant official evidence support:")
    for line in grouped_support(child["official_target_evidence_support"]):
        print(line)
    print(
        "  Proposed dependency/parent relation: "
        f"{pair['parent_title']} -> {pair['child_title']} "
        f"({child['dependency_status']}; confidence {child['dependency_confidence']})"
    )
    print("\nAdministrative corpus presence: PASS (not a human judgment)")


def read_choice(prompt: str, allowed: set[str]) -> str:
    while True:
        value = input(prompt).strip().upper()
        if value in allowed:
            return value
        print(f"Enter {' or '.join(sorted(allowed))}.")


def collect_decision(
    candidate: dict[str, Any], pair: dict[str, Any], annotator: str
) -> dict[str, Any]:
    print("\n1. Parent directly identifiable from Urdu question?")
    answer_1 = read_choice("   Y / N: ", {"Y", "N"})
    print("\n2. Child NOT directly identifiable from Urdu question alone?")
    answer_2 = read_choice("   Y / N: ", {"Y", "N"})
    print("\n3. Does the stated intermediate information genuinely make the child")
    print("   identifiable/recoverable?")
    answer_3 = read_choice("   Y / N: ", {"Y", "N"})
    print("\n4. Dependency status:")
    print("   C = CLEAR_DEPENDENCY")
    print("   O = OTHER / NOT CLEAR")
    answer_4 = read_choice("   C / O: ", {"C", "O"})

    answers = {
        "parent_directly_identifiable_from_urdu_question": answer_1,
        "child_not_directly_identifiable_from_urdu_question_alone": answer_2,
        "stated_intermediate_information_makes_child_identifiable_or_recoverable": answer_3,
        "dependency_status": answer_4,
    }
    verdict = VERIFIED if (answer_1, answer_2, answer_3, answer_4) == ("Y", "Y", "Y", "C") else REJECTED
    record: dict[str, Any] = {
        "annotation_status": OUTPUT_STATUS,
        "qid": candidate["urbench_qid"],
        "parent_source_instance_id": pair["parent_source_instance_id"],
        "parent_title": pair["parent_title"],
        "child_source_instance_id": pair["child_source_instance_id"],
        "child_title": pair["child_title"],
        "human_answers": answers,
        "verdict": verdict,
        "annotator_id": annotator,
        "timestamp": utc_timestamp(),
    }
    if verdict == REJECTED:
        reason = ""
        while not reason:
            reason = input("Short rejection reason (required): ").strip()
            if not reason:
                print("A short rejection reason is required.")
        record["rejection_reason"] = reason
    return record


def completed_qid_count(
    candidates: list[dict[str, Any]], decisions: dict[tuple[str, str, str], dict[str, Any]]
) -> int:
    return sum(
        all(pair_key(candidate["urbench_qid"], pair) in decisions for pair in candidate["candidate_pairs"])
        for candidate in candidates
    )


def print_progress(data: ReviewData, decisions: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    high = data.candidates[:20]
    remaining = data.candidates[20:]
    verdicts = Counter(row["verdict"] for row in decisions.values())
    print("PROGRESS (aggregate counts only)")
    print(f"total ranked candidate qids = {len(data.candidates)}")
    print(f"candidate parent-child pairs = {len(data.pair_by_key)}")
    print(f"human decisions recorded = {len(decisions)}")
    print(f"fully human-verified qids = {completed_qid_count(data.candidates, decisions)}")
    print(f"high-confidence qids complete = {completed_qid_count(high, decisions)} / {len(high)}")
    print(f"remaining qids complete = {completed_qid_count(remaining, decisions)} / {len(remaining)}")
    print(f"{VERIFIED} = {verdicts[VERIFIED]}")
    print(f"{REJECTED} = {verdicts[REJECTED]}")


def run_audit() -> None:
    canonical_before = canonical_snapshot()
    output_before = output_snapshot()
    data = ReviewData()
    decisions = load_existing(data)
    output_after = output_snapshot()
    canonical_after = canonical_snapshot()
    require(output_before == output_after, "Audit changed the assisted human-verification output")
    require(canonical_before == canonical_after, "Audit changed the canonical human log")

    high_count = sum(row["priority_tier"] == "HIGH" for row in data.candidates)
    remaining_count = len(data.candidates) - high_count
    print("AUDIT")
    print(f"total ranked candidate qids = {len(data.candidates)}")
    print(f"high-confidence candidate qids = {high_count}")
    print(f"remaining candidate qids = {remaining_count}")
    print(f"number already human-verified = {len(decisions)}")
    print("canonical human log is NOT modified")
    print("AUDIT PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human verification of model-assisted EFBPT bridge candidates."
    )
    parser.add_argument("--annotator", help="Human annotator id, for example PRIMARY_A")
    parser.add_argument(
        "--tier",
        choices=("high", "remaining"),
        default="high",
        help="Review the first 20 HIGH qids (default) or the remaining 10 ranked qids",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit", action="store_true", help="Validate and report without verification")
    modes.add_argument("--progress", action="store_true", help="Show aggregate counts only")
    return parser


def run_interactive(annotator: str, tier: str) -> None:
    data = ReviewData()
    decisions = load_existing(data)
    items = data.items(tier)
    pending = [
        item
        for item in items
        if pair_key(item[0]["urbench_qid"], item[2]) not in decisions
    ]
    tier_qids = 20 if tier == "high" else 10
    print(
        f"{tier.upper()} tier: {tier_qids} ranked qids, {len(items)} parent-child pairs, "
        f"{len(pending)} pair decision(s) pending."
    )
    print("Ctrl-C or Ctrl-D stops safely before the current decision is appended.")
    if not pending:
        print("No pending candidates in this tier.")
        return

    for decision_number, (candidate, pair_number, pair) in enumerate(pending, 1):
        display_candidate(data, candidate, pair_number, pair, decision_number, len(pending))
        record = collect_decision(candidate, pair, annotator)
        if append_decision(record, data):
            decisions[pair_key(candidate["urbench_qid"], pair)] = record
            print(f"SAVED AND FSYNCED: {record['verdict']}")
        else:
            print("NOT APPENDED: this pair was already decided by another active process.")
    print("Selected tier is complete.")


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.audit:
            require(arguments.annotator is None, "--annotator is not used with --audit")
            run_audit()
            return 0

        data = ReviewData()
        decisions = load_existing(data)
        if arguments.progress:
            require(arguments.annotator is None, "--annotator is not used with --progress")
            print_progress(data, decisions)
            return 0

        require(arguments.annotator is not None, "Actual verification requires --annotator")
        annotator = validate_annotator(arguments.annotator)
        run_interactive(annotator, arguments.tier)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nStopped safely; the current candidate was not appended.", file=sys.stderr)
        return 130
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
