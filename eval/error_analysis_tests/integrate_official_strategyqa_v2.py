"""
Safely integrate official StrategyQA annotations with URBench identifiers.

Safety properties
-----------------
* Reads the original datasets only.
* Never modifies raw inputs, indexes, prompts, outputs, or model files.
* Performs every validation before publishing generated outputs.
* Writes generated files atomically under data/strategyqa_official/.
* Uses conservative normalized-text matching; never uses fuzzy matching.

The official StrategyQA repository and URBench use different qid namespaces.
Rows are therefore matched by normalized English question text. Duplicate
question groups are resolved only by an unambiguous secondary signature made
from answer, term, description, decomposition, and facts.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BASE = Path("/mnt/home/user41/URBench")
OFF = BASE / "data" / "strategyqa_official"
RAW = BASE / "data" / "strategyqa_raw"
SPL = BASE / "data" / "sdfr_splits"

F_OFF_TRAIN = OFF / "train.json"
F_OFF_DEV = OFF / "dev.json"
F_OFF_PARA = OFF / "strategyqa_train_paragraphs.json"
F_URB_EN = RAW / "strategyQA_train.json"
F_URB_UR = RAW / "strategyQA_train_ur2_norm.jsonl"
F_EVAL = SPL / "strategyqa_eval.jsonl"

F_OUT_MAP = OFF / "strategyqa_official_mapped_urbench_qid.jsonl"
F_OUT_DEV50 = OFF / "dev50_seed42.jsonl"
F_OUT_DEVQ = OFF / "dev50_seed42_qids.txt"

MARKERS = {"operation", "no_evidence"}
EXPECTED_OFFICIAL_TRAIN = 2061
EXPECTED_OFFICIAL_DEV = 229
EXPECTED_TOTAL = 2290
EXPECTED_EVAL = 458
EXPECTED_NON_EVAL = 1832
DEV_SEED = 42
DEV_SIZE = 50


def die(message: str) -> None:
    print(f"\n*** FATAL: {message}")
    print("*** No generated output files were published by this run.")
    raise SystemExit(1)


def load_json(path: Path) -> Any:
    if not path.is_file():
        die(f"required input file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        die(f"required input file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                die(f"invalid JSONL at {path}:{line_no}: {exc}")
    return rows


def norm_text(value: Any) -> str:
    """Conservative normalization used only for validation and joining."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return re.sub(r"\s+", " ", text).strip().casefold()


def norm_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = norm_text(value)
    if text in {"true", "yes", "ہاں"}:
        return True
    if text in {"false", "no", "نہیں"}:
        return False
    return None


def norm_sequence(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return tuple()
    return tuple(norm_text(value) for value in values)


def secondary_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    """Strict discriminator for rows sharing the same normalized question."""
    return (
        norm_bool(row.get("answer")),
        norm_text(row.get("term")),
        norm_text(row.get("description")),
        norm_sequence(row.get("decomposition")),
        norm_sequence(row.get("facts")),
    )


def require_unique_ids(rows: Iterable[dict[str, Any]], label: str) -> set[str]:
    ids: list[str] = []
    missing = 0
    for row in rows:
        qid = row.get("qid")
        if qid is None or str(qid).strip() == "":
            missing += 1
        else:
            ids.append(str(qid))
    counts = Counter(ids)
    duplicates = sorted(qid for qid, count in counts.items() if count > 1)
    if missing:
        die(f"{label} has {missing} row(s) without qid")
    if duplicates:
        die(f"{label} has duplicate qids, e.g. {duplicates[:5]}")
    return set(ids)


def group_by_question(
    rows: Iterable[dict[str, Any]], label: str
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = norm_text(row.get("question"))
        if not key:
            die(f"{label} contains an empty English question")
        groups[key].append(row)
    return dict(groups)


def printable_row(row: dict[str, Any], side: str) -> str:
    payload = {
        "side": side,
        "qid": row.get("qid"),
        "source": row.get("_src"),
        "answer": row.get("answer"),
        "term": row.get("term"),
        "description": row.get("description"),
        "facts": row.get("facts"),
        "decomposition": row.get("decomposition"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def match_rows(
    urbench_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], int, int]:
    """Return strict one-to-one matches plus duplicate-group diagnostics."""
    urb_groups = group_by_question(urbench_rows, "URBench English")
    off_groups = group_by_question(official_rows, "official StrategyQA")

    urb_keys = set(urb_groups)
    off_keys = set(off_groups)
    missing_in_official = sorted(urb_keys - off_keys)
    missing_in_urbench = sorted(off_keys - urb_keys)
    if missing_in_official or missing_in_urbench:
        print("\nQuestion-group mismatch:")
        for question in missing_in_official[:10]:
            print("  URBench only:", question)
        for question in missing_in_urbench[:10]:
            print("  Official only:", question)
        die(
            "normalized English question groups differ: "
            f"URBench-only={len(missing_in_official)}, "
            f"official-only={len(missing_in_urbench)}"
        )

    matches: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    duplicate_groups = 0
    resolved_duplicate_groups = 0
    unresolved: list[str] = []

    for question_key in sorted(urb_keys):
        u_group = urb_groups[question_key]
        o_group = off_groups[question_key]
        if len(u_group) != len(o_group):
            unresolved.append(question_key)
            print(
                f"\nDuplicate-group size mismatch for {question_key!r}: "
                f"URBench={len(u_group)}, official={len(o_group)}"
            )
            for row in u_group:
                print(" ", printable_row(row, "URBench"))
            for row in o_group:
                print(" ", printable_row(row, "official"))
            continue

        if len(u_group) == 1:
            matches.append((u_group[0], o_group[0], "normalized_question"))
            continue

        duplicate_groups += 1
        print(
            f"\nResolving duplicate question group ({len(u_group)} rows/side): "
            f"{u_group[0].get('question')}"
        )
        u_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        o_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in u_group:
            u_by_signature[secondary_signature(row)].append(row)
        for row in o_group:
            o_by_signature[secondary_signature(row)].append(row)

        if set(u_by_signature) != set(o_by_signature):
            unresolved.append(question_key)
        else:
            group_pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
            for signature in u_by_signature:
                u_candidates = u_by_signature[signature]
                o_candidates = o_by_signature[signature]
                if len(u_candidates) != 1 or len(o_candidates) != 1:
                    unresolved.append(question_key)
                    group_pairs = []
                    break
                group_pairs.append(
                    (
                        u_candidates[0],
                        o_candidates[0],
                        "normalized_question+secondary_signature",
                    )
                )
            if group_pairs:
                matches.extend(group_pairs)
                resolved_duplicate_groups += 1
                print("  resolved unambiguously by secondary signature")
                continue

        print("  unresolved; refusing to guess")
        for row in u_group:
            print(" ", printable_row(row, "URBench"))
        for row in o_group:
            print(" ", printable_row(row, "official"))

    unresolved_unique = sorted(set(unresolved))
    print(f"\nduplicate question groups discovered = {duplicate_groups}")
    print(f"resolved duplicate question groups   = {resolved_duplicate_groups}")
    print(f"unresolved duplicate groups          = {len(unresolved_unique)}")
    if unresolved_unique:
        die("one or more duplicate question groups remain ambiguous")

    # Preserve original URBench order and prove one-to-one use of both sides.
    by_urbench_qid: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
    used_official_qids: list[str] = []
    for urow, orow, method in matches:
        uqid = str(urow.get("qid"))
        oqid = str(orow.get("qid"))
        if uqid in by_urbench_qid:
            die(f"URBench row matched more than once: {uqid}")
        by_urbench_qid[uqid] = (urow, orow, method)
        used_official_qids.append(oqid)
    if len(set(used_official_qids)) != len(used_official_qids):
        die("an official row was matched more than once")
    ordered = [by_urbench_qid[str(row["qid"])] for row in urbench_rows]
    return ordered, duplicate_groups, resolved_duplicate_groups


def walk_evidence(
    value: Any,
    paragraph_ids: list[str],
    marker_counts: Counter[str],
) -> None:
    if isinstance(value, str):
        if value in MARKERS:
            marker_counts[value] += 1
        else:
            paragraph_ids.append(value)
    elif isinstance(value, list):
        for item in value:
            walk_evidence(item, paragraph_ids, marker_counts)
    elif isinstance(value, dict):
        for item in value.values():
            walk_evidence(item, paragraph_ids, marker_counts)


def evidence_shape(value: Any, depth: int = 0) -> str:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, list):
        if not value:
            return "list[0]"
        child_shapes = sorted({evidence_shape(item, depth + 1) for item in value})
        return f"list[{len(value)}]({', '.join(child_shapes[:4])})"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def validate_evidence_structure(
    row: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    evidence = row.get("evidence")
    decomposition = row.get("decomposition")
    qid = row.get("qid")
    if not isinstance(decomposition, list) or not decomposition:
        return [f"qid={qid}: official decomposition is not a non-empty list"]
    if not isinstance(evidence, list) or not evidence:
        return [f"qid={qid}: official evidence is not a non-empty list"]
    expected_steps = len(decomposition)
    for annotator_index, annotation in enumerate(evidence):
        if not isinstance(annotation, list):
            errors.append(
                f"qid={qid}: annotator {annotator_index} evidence is not a list"
            )
        elif len(annotation) != expected_steps:
            errors.append(
                f"qid={qid}: annotator {annotator_index} has "
                f"{len(annotation)} evidence steps; expected {expected_steps}"
            )
    return errors


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    print("Safe StrategyQA integration: validation runs before atomic output writes")
    print("Original source files are read-only in this script.\n")

    # 1. Load and validate basic counts.
    off_train_raw = load_json(F_OFF_TRAIN)
    off_dev_raw = load_json(F_OFF_DEV)
    if not isinstance(off_train_raw, list) or not isinstance(off_dev_raw, list):
        die("official train.json and dev.json must both contain JSON lists")
    off_train = [{**row, "_src": "train"} for row in off_train_raw]
    off_dev = [{**row, "_src": "dev"} for row in off_dev_raw]
    official = off_train + off_dev
    paragraphs = load_json(F_OFF_PARA)
    urb_en = load_json(F_URB_EN)
    urb_ur = load_jsonl(F_URB_UR)
    eval_rows = load_jsonl(F_EVAL)

    if not isinstance(paragraphs, dict):
        die("strategyqa_train_paragraphs.json must contain a JSON object")
    if not isinstance(urb_en, list):
        die("strategyQA_train.json must contain a JSON list")

    print(
        f"official train={len(off_train)} dev={len(off_dev)} "
        f"combined={len(official)}"
    )
    print(f"paragraph entries={len(paragraphs)}")
    print(
        f"URBench EN rows={len(urb_en)} UR rows={len(urb_ur)} "
        f"eval rows={len(eval_rows)}"
    )
    if len(off_train) != EXPECTED_OFFICIAL_TRAIN:
        die(f"unexpected official train count: {len(off_train)}")
    if len(off_dev) != EXPECTED_OFFICIAL_DEV:
        die(f"unexpected official dev count: {len(off_dev)}")
    if len(official) != EXPECTED_TOTAL or len(urb_en) != EXPECTED_TOTAL:
        die("official and URBench English datasets must each contain 2290 rows")
    if len(urb_ur) != EXPECTED_TOTAL:
        die("URBench Urdu dataset must contain 2290 rows")
    if len(eval_rows) != EXPECTED_EVAL:
        die("URBench evaluation split must contain 458 rows")

    official_qids = require_unique_ids(official, "official StrategyQA")
    urbench_qids = require_unique_ids(urb_en, "URBench English")
    urdu_qids = require_unique_ids(urb_ur, "URBench Urdu")
    eval_qids = require_unique_ids(eval_rows, "URBench evaluation")
    if urdu_qids != urbench_qids:
        die(
            "URBench English and Urdu qid sets differ: "
            f"English-only={len(urbench_qids - urdu_qids)}, "
            f"Urdu-only={len(urdu_qids - urbench_qids)}"
        )
    if not eval_qids.issubset(urbench_qids):
        die(f"{len(eval_qids - urbench_qids)} eval qids are absent from URBench")
    print(
        f"unique qids: official={len(official_qids)} "
        f"URBench={len(urbench_qids)} eval={len(eval_qids)}"
    )

    # 2. Strict grouped matching.
    matched, duplicate_groups, resolved_duplicate_groups = match_rows(
        urb_en, official
    )
    print(f"matched pairs = {len(matched)} / {EXPECTED_TOTAL}")
    if len(matched) != EXPECTED_TOTAL:
        die(f"expected 2290 matches, found {len(matched)}")

    # 3. Cross-validate matched semantics.
    answer_mismatches: list[tuple[Any, ...]] = []
    decomp_length_mismatches: list[tuple[Any, ...]] = []
    decomp_text_differences: list[str] = []
    term_differences: list[str] = []
    description_differences: list[str] = []
    for urow, orow, _ in matched:
        uqid = str(urow.get("qid"))
        if norm_bool(urow.get("answer")) != norm_bool(orow.get("answer")):
            answer_mismatches.append(
                (uqid, urow.get("answer"), orow.get("answer"))
            )
        u_decomp = urow.get("decomposition") or []
        o_decomp = orow.get("decomposition") or []
        if len(u_decomp) != len(o_decomp):
            decomp_length_mismatches.append(
                (uqid, len(u_decomp), len(o_decomp))
            )
        elif norm_sequence(u_decomp) != norm_sequence(o_decomp):
            decomp_text_differences.append(uqid)
        if norm_text(urow.get("term")) != norm_text(orow.get("term")):
            term_differences.append(uqid)
        if norm_text(urow.get("description")) != norm_text(orow.get("description")):
            description_differences.append(uqid)

    print(f"answer mismatches             = {len(answer_mismatches)}")
    print(f"decomposition length mismatch = {len(decomp_length_mismatches)}")
    print(f"decomposition text differs    = {len(decomp_text_differences)}")
    print(f"term differs                  = {len(term_differences)}")
    print(f"description differs           = {len(description_differences)}")
    if answer_mismatches:
        for mismatch in answer_mismatches[:10]:
            print("  ANSWER MISMATCH:", mismatch)
        die("answer mismatches found")
    if decomp_length_mismatches:
        for mismatch in decomp_length_mismatches[:10]:
            print("  DECOMPOSITION LENGTH MISMATCH:", mismatch)
        die("decomposition length mismatches found")

    # 4. Validate evidence shape and paragraph references.
    first_evidence = official[0].get("evidence")
    print(f"first official evidence shape = {evidence_shape(first_evidence)}")
    paragraph_key_set = set(paragraphs.keys())
    all_paragraph_ids: list[str] = []
    all_markers: Counter[str] = Counter()
    evidence_structure_errors: list[str] = []
    source_stats: dict[str, Counter[str]] = {
        "train": Counter(),
        "dev": Counter(),
    }
    evidence_by_official_qid: dict[str, tuple[list[str], Counter[str]]] = {}
    for row in official:
        errors = validate_evidence_structure(row)
        evidence_structure_errors.extend(errors)
        row_ids: list[str] = []
        row_markers: Counter[str] = Counter()
        walk_evidence(row.get("evidence"), row_ids, row_markers)
        oqid = str(row.get("qid"))
        evidence_by_official_qid[oqid] = (row_ids, row_markers)
        all_paragraph_ids.extend(row_ids)
        all_markers.update(row_markers)
        source = str(row.get("_src"))
        source_stats[source]["rows"] += 1
        source_stats[source]["paragraph_references"] += len(row_ids)
        if row_ids:
            source_stats[source]["rows_with_paragraph_evidence"] += 1
        else:
            source_stats[source]["rows_without_paragraph_evidence"] += 1

    print(f"evidence/decomposition structural mismatches = {len(evidence_structure_errors)}")
    if evidence_structure_errors:
        for error in evidence_structure_errors[:20]:
            print("  STRUCTURE:", error)
        die("official evidence structure does not align with official decomposition")

    unique_paragraph_ids = set(all_paragraph_ids)
    missing_paragraph_ids = sorted(unique_paragraph_ids - paragraph_key_set)
    print(f"total evidence paragraph references = {len(all_paragraph_ids)}")
    print(f"unique evidence paragraph IDs       = {len(unique_paragraph_ids)}")
    print(f"operation markers                   = {all_markers['operation']}")
    print(f"no_evidence markers                 = {all_markers['no_evidence']}")
    print(f"missing evidence paragraph IDs      = {len(missing_paragraph_ids)}")
    for source in ("train", "dev"):
        stats = source_stats[source]
        print(
            f"{source} evidence: rows={stats['rows']} "
            f"rows_with_paragraphs={stats['rows_with_paragraph_evidence']} "
            f"rows_without_paragraphs={stats['rows_without_paragraph_evidence']} "
            f"paragraph_refs={stats['paragraph_references']}"
        )
    if missing_paragraph_ids:
        print("  first missing IDs:", missing_paragraph_ids[:20])
        die("official evidence references IDs absent from paragraph dictionary")

    # 5. Build merged rows after all validations have passed.
    ur_by_qid = {str(row["qid"]): row for row in urb_ur}
    merged: list[dict[str, Any]] = []
    for urow, orow, match_method in matched:
        uqid = str(urow["qid"])
        oqid = str(orow["qid"])
        row_ids, _ = evidence_by_official_qid[oqid]
        merged.append(
            {
                "urbench_qid": uqid,
                "official_qid": oqid,
                "official_source": orow.get("_src"),
                "match_method": match_method,
                "question_en": urow.get("question"),
                "question_ur": ur_by_qid[uqid].get("question"),
                "answer": urow.get("answer"),
                "term": urow.get("term"),
                "description": urow.get("description"),
                "official_term": orow.get("term"),
                "official_description": orow.get("description"),
                "urbench_facts": urow.get("facts"),
                "official_facts": orow.get("facts"),
                "urbench_decomposition": urow.get("decomposition"),
                "official_decomposition": orow.get("decomposition"),
                "official_evidence": orow.get("evidence"),
                "evidence_paragraph_ids": sorted(set(row_ids)),
                "has_paragraph_evidence": bool(row_ids),
                "is_eval": uqid in eval_qids,
            }
        )

    mapped_qids = {row["urbench_qid"] for row in merged}
    missing_eval_qids = sorted(eval_qids - mapped_qids)
    non_eval = [row for row in merged if not row["is_eval"]]
    if len(mapped_qids) != EXPECTED_TOTAL:
        die(f"merged output has {len(mapped_qids)} unique URBench qids, not 2290")
    if missing_eval_qids:
        die(f"missing evaluation mappings, e.g. {missing_eval_qids[:10]}")
    if len(non_eval) != EXPECTED_NON_EVAL:
        die(f"expected 1832 non-eval rows, found {len(non_eval)}")
    print(f"evaluation qids mapped = {len(eval_qids)} / {EXPECTED_EVAL}")
    print(f"non-evaluation rows     = {len(non_eval)}")

    # 6. Create deterministic DEV50 from observed evidence availability.
    eligible = sorted(
        (row for row in non_eval if row["has_paragraph_evidence"]),
        key=lambda row: row["urbench_qid"],
    )
    if len(eligible) < DEV_SIZE:
        die(f"only {len(eligible)} eligible non-eval evidence rows; need 50")
    rng = random.Random(DEV_SEED)
    dev50 = rng.sample(eligible, DEV_SIZE)
    dev50_qids = [row["urbench_qid"] for row in dev50]
    dev50_qid_set = set(dev50_qids)
    intersection = dev50_qid_set & eval_qids
    if len(dev50_qid_set) != DEV_SIZE:
        die("DEV50 contains duplicate URBench qids")
    if intersection:
        die(f"DEV50 intersects evaluation qids: {sorted(intersection)[:10]}")
    print(f"DEV50 eligible pool     = {len(eligible)}")
    print(f"DEV50 size              = {len(dev50)}")
    print("DEV50 intersection eval = 0")

    # 7. Publish only validated derived files, atomically.
    atomic_write_jsonl(F_OUT_MAP, merged)
    atomic_write_jsonl(F_OUT_DEV50, dev50)
    atomic_write_text(F_OUT_DEVQ, "\n".join(dev50_qids) + "\n")
    print(f"\nwrote {F_OUT_MAP} ({len(merged)} rows)")
    print(f"wrote {F_OUT_DEV50} ({len(dev50)} rows)")
    print(f"wrote {F_OUT_DEVQ} ({len(dev50_qids)} qids)")

    print("\n" + "=" * 68)
    print("VALIDATION SUMMARY")
    print(f"  mapped official <-> URBench       : {len(merged)}/2290")
    print(f"  duplicate question groups         : {duplicate_groups}")
    print(f"  resolved duplicate groups         : {resolved_duplicate_groups}")
    print("  unresolved duplicate groups       : 0")
    print(f"  evaluation qids mapped            : {len(eval_qids)}/458")
    print(f"  answer mismatches                 : {len(answer_mismatches)}")
    print(f"  decomp length mismatches          : {len(decomp_length_mismatches)}")
    print(f"  evidence structure mismatches     : {len(evidence_structure_errors)}")
    print(f"  missing evidence paragraph IDs    : {len(missing_paragraph_ids)}")
    print(f"  DEV50 size / disjoint             : {len(dev50)} / {not intersection}")
    print("=" * 68)
    print("SUCCESS: official StrategyQA annotations are integrated safely.")


if __name__ == "__main__":
    main()