#!/usr/bin/env python3
"""Read-only qid aggregation for the September 4 URBench handoff.

Repository target: eval/error_analysis_tests/efbpt/efbpt_assisted_qid_audit.py
Usage from repository root:
    python -B eval/error_analysis_tests/efbpt/efbpt_assisted_qid_audit.py --repo .
The script can also live outside the repository; --repo selects the input root.

Stdlib only. Writes JSON to stdout, never writes files or submits jobs. Imports
only the inspected, blob-pinned verifier from commit e4a5992. Reuses its frozen
input hashes and human-decision validation. Does not call its interactive or
append functions. Disables bytecode writes even without Python's -B switch.

An AUDIT_PASS establishes administrative consistency of the existing human-
verified assisted cohort, not independent reliability or current corpus content.
It does not freeze a bridge experiment or select one pair per qid.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import types


VERIFIER_REL = "eval/error_analysis_tests/efbpt/efbpt_verify_assisted_candidates.py"
VERIFIER_BLOB = "1202d612141258ab9ad190263ce500f9db7b232d"
VERIFIER_COMMIT = "e4a599283e7ecaf66486ca89883c67902bf6c716"
HANDOFF_COUNTS = {
    "total_pair_records": 41,
    "verified_pairs": 36,
    "rejected_pairs": 5,
    "total_unique_qids": 30,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def norm(title):
    return " ".join(str(title).replace("_", " ").strip().lower().split())


def snapshot(path):
    require(path.is_file() and not path.is_symlink(),
            "Missing, non-regular, or symlink input: " + str(path))
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    require(fields(before) == fields(after), "Input changed while reading: " + str(path))
    return {"identity": fields(after), "sha256": digest}


def load_verifier(root):
    path = root / VERIFIER_REL
    before = snapshot(path)
    raw = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    require(blob == VERIFIER_BLOB,
            "Verifier differs from inspected e4a5992 blob; inspect drift before using this audit.")
    module = types.ModuleType("_urbench_assisted_verifier_readonly")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    require(snapshot(path) == before, "Verifier changed during import")
    return module, path, before


def aggregate(data, decisions, verified, rejected):
    require(set(decisions) == set(data.pair_by_key),
            "Review is incomplete or contains unknown pairs; ALL_REJECTED is not valid for pending qids.")
    candidate_qids = {r["urbench_qid"] for r in data.candidates}
    by_qid = defaultdict(list)
    for row in decisions.values():
        by_qid[row["qid"]].append(row)
    require(set(by_qid) == candidate_qids, "Candidate/decision qid sets differ")
    counts = Counter(r["verdict"] for r in decisions.values())
    require(set(counts) <= {verified, rejected}, "Unknown verdict")
    accepted = [r for r in decisions.values() if r["verdict"] == verified]
    accepted_qids = {r["qid"] for r in accepted}

    # Existing ReviewData checks candidate/pass2/pass3 fields and frozen hashes.
    # Also explicitly join both accepted endpoints to the frozen source master.
    for row in accepted:
        key = (row["qid"], row["parent_source_instance_id"], row["child_source_instance_id"])
        pair = data.pair_by_key[key]
        for side in ("parent", "child"):
            source_id = row[side + "_source_instance_id"]
            master = data.master_by_id[source_id]
            require(master["urbench_qid"] == row["qid"], "Source master qid mismatch")
            require(master["gold_title"] == row[side + "_title"], "Source master title mismatch")
            require(master["normalized_gold_title"] == norm(row[side + "_title"]),
                    "Source master title normalization mismatch")
            require(master["exact_corpus_status"] == pair[side + "_exact_corpus_status"] == "EXACT_PRESENT",
                    "Accepted endpoint fails administrative exact presence: " + source_id)

    totals = {
        "total_pair_records": len(decisions),
        "verified_pairs": counts[verified],
        "rejected_pairs": counts[rejected],
        "total_unique_qids": len(by_qid),
    }
    tiers = {}
    for tier in sorted({c["priority_tier"] for c in data.candidates}):
        qids = {c["urbench_qid"] for c in data.candidates if c["priority_tier"] == tier}
        rows = [r for q in qids for r in by_qid[q]]
        tiers[tier] = {
            "reviewed_qids": len(qids),
            "pair_records": len(rows),
            "verified_pairs": sum(r["verdict"] == verified for r in rows),
            "qids_with_verified_pair": len(qids & accepted_qids),
        }
    return {
        "cohort_label": "HUMAN_VERIFICATION_OF_MODEL_ASSISTED_CANDIDATES",
        **totals,
        "handoff_counts_match": totals == HANDOFF_COUNTS,
        "handoff_expected_counts": HANDOFF_COUNTS,
        "fully_reviewed_qids": len(by_qid),
        "unique_qids_with_verified_pair": len(accepted_qids),
        "unique_qids_all_rejected": len(by_qid) - len(accepted_qids),
        "qid_aggregate_status": [
            {"qid": qid, "status": "HAS_VERIFIED_PAIR" if qid in accepted_qids else "ALL_REJECTED"}
            for qid in sorted(by_qid)
        ],
        "unique_accepted_parent_source_instance_ids": len({r["parent_source_instance_id"] for r in accepted}),
        "unique_accepted_child_source_instance_ids": len({r["child_source_instance_id"] for r in accepted}),
        "unique_accepted_parent_normalized_titles": len({norm(r["parent_title"]) for r in accepted}),
        "unique_accepted_child_normalized_titles": len({norm(r["child_title"]) for r in accepted}),
        "accepted_pairs_passing_administrative_exact_presence": len(accepted),
        "accepted_pair_administrative_exact_presence_check": "PASS",
        "exact_presence_scope": "Pinned candidate, assisted-label and source-master consistency; no fresh corpus scan or alias resolution.",
        "tier_summary": tiers,
        "assisted_pilot_at_least_30_accepted_qids": len(accepted_qids) >= 30,
        "canonical_reliability_gate": "NOT_ESTABLISHED_BY_THIS_AUDIT",
        "canonical_stage0_feasibility_gate": "NOT_ESTABLISHED_BY_ASSISTED_COHORT",
        "experiment_state": "NOT_FROZEN_NOT_RUN",
    }


def audit(root):
    verifier, verifier_path, verifier_before = load_verifier(root)
    paths = [*verifier.EXPECTED_SHA256, verifier.OUTPUT_PATH, verifier.CANONICAL_HUMAN_LOG_PATH]
    before = {p: snapshot(p) for p in paths}
    data = verifier.ReviewData()
    rows = verifier.load_jsonl(verifier.OUTPUT_PATH)
    decisions = verifier.validate_existing_rows(rows, data)
    report = aggregate(data, decisions, verifier.VERIFIED, verifier.REJECTED)
    after = {p: snapshot(p) for p in paths}
    require(before == after, "An input changed during the audit; no valid snapshot result")
    require(snapshot(verifier_path) == verifier_before, "Verifier changed during audit")
    report.update({
        "audit_status": "AUDIT_PASS",
        "freeze_readiness": "QID_AGGREGATION_VERIFIED" if report["handoff_counts_match"] else "REVIEW_HANDOFF_COUNT_DRIFT",
        "all_input_contents_unchanged": True,
        "input_sha256": {str(p.relative_to(root)): s["sha256"] for p, s in before.items()},
        "verifier_sha256": verifier_before["sha256"],
        "verifier_git_blob": VERIFIER_BLOB,
        "verifier_reference_commit": VERIFIER_COMMIT,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Existing URBench repository root (default: cwd)")
    args = parser.parse_args()
    try:
        report = audit(args.repo.resolve())
    except Exception as exc:
        print(json.dumps({"audit_status": "AUDIT_FAILED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["handoff_counts_match"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
