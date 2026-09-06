#!/usr/bin/env python3
"""Separate-process N25 scorer, v0.2. No generation, retrieval or human logs.

The scoring targets are opened only after the entire chain authenticates:
the prospective activation manifest (from an explicit expected SHA-256), the
preparation seal, the preflight seal, the parent-query seal, the oracle-E
query seal, the predictions seal, and the exact artifact identity of every
file each seal names. A self-consistent seal produced by an arbitrary caller
cannot pass, because its activation SHA-256 must equal the reviewed manifest's
own content hash and every upstream seal hash it declares must match the seal
actually present on disk.

Refuses partial, duplicate, unknown or malformed qid/arm records. Computes
only the frozen statistics in ``bridge_pilot_core.summarize``.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import os
from pathlib import Path

from bridge_pilot_core import (canonical, exact_keys, need, sha256, strict_json,
                               summarize, validate_predictions)
from bridge_pilot_run import (HEX256, LINEAGE, STAGE_SCHEMAS, TARGET_EXPORT, PathGuard,
                              authenticate_upstream, check_seal, jsonl_rows, load_activation,
                              new_dir, no_symlink, read_small, validate_prediction_record,
                              verify_code_identity, write_json)

SCORING_SCHEMA = "urbench.bridge_n25.scoring_seal.v1"

def stage_seal_path(manifest, stage, tag):
    return Path(manifest["roots"]["output_root"]) / (stage + "_" + tag) / "STAGE_SEAL.json"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activation", type=Path, required=True)
    p.add_argument("--activation-sha256", required=True)
    p.add_argument("--predictions-seal", type=Path, required=True)
    p.add_argument("--seal-sha256", required=True)
    p.add_argument("--targets-sha256", required=True)
    p.add_argument("--parent-tag", default="v1")
    p.add_argument("--oracle-e-tag", default="v1")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    for h in (args.activation_sha256, args.seal_sha256, args.targets_sha256):
        need(isinstance(h, str) and HEX256.fullmatch(h), "EXPECTED_SHA256_REQUIRED")

    # 1. Root of trust: the reviewed manifest, addressed by its own content hash.
    manifest, activation = load_activation(args.activation, args.activation_sha256)
    guard = PathGuard("score")
    guard.allow("activation", args.activation)
    guard.allow("preparation_seal", manifest["preparation"]["seal_path"])
    guard.allow("preflight_seal", manifest["preflight"]["seal_path"])
    guard.allow("prompt_preflight_seal", manifest["prompt_preflight"]["seal_path"])
    guard.allow("prompt_preflight_summary", manifest["prompt_preflight"]["summary_path"])
    guard.allow("predictions_seal", args.predictions_seal)
    prediction_dir = no_symlink(args.predictions_seal).parent
    for name in ("predictions.jsonl", "prediction_records.jsonl"):
        guard.allow("prediction_" + name, prediction_dir / name)
    for stage, tag in (("parent", args.parent_tag), ("oracle_e", args.oracle_e_tag)):
        guard.allow(stage + "_seal", stage_seal_path(manifest, stage, tag))

    out = no_symlink(args.out)
    # The scoring destination is the reviewed one, not a free-form argument.
    need(str(out) == manifest["roots"]["score_output_root"], "SCORE_OUTPUT_NOT_REVIEWED")
    need(not out.exists() and not out.is_symlink() and out.parent.is_dir(),
         "FRESH_OUTPUT_WITH_EXISTING_PARENT_REQUIRED")
    need(not any(Path(x).is_relative_to(out) for x in guard.allowed), "OUTPUT_INPUT_OVERLAP")

    # 2. Reviewed code identity, then the preparation and preflight seals.
    code_identity = verify_code_identity(guard, manifest)
    scorer_local, _ = read_small(Path(__file__).resolve())
    need(sha256(scorer_local) == code_identity["bridge_pilot_score.py"],
         "RUNNING_SCORER_NOT_REVIEWED")
    prep_id, pre_id, prompt_id = authenticate_upstream(guard, manifest)

    # 3. Predictions seal. Its declared activation must be the real manifest.
    raw_seal, seal_id = guard.read_small(args.predictions_seal, args.seal_sha256)
    seal = strict_json(raw_seal)
    schema, status = STAGE_SCHEMAS["retrieve"]
    need(seal.get("schema") == schema and seal.get("status") == status, "PREDICTIONS_NOT_SEALED")
    need(seal.get("activation_sha256") == activation["sha256"], "PREDICTIONS_SEAL_ACTIVATION")
    need(seal.get("preparation_seal_sha256") == manifest["preparation"]["seal_sha256"]
         and seal.get("preflight_seal_sha256") == manifest["preflight"]["seal_sha256"],
         "PREDICTIONS_SEAL_UPSTREAM_BINDING")
    need(seal.get("historical_global_lineage") == LINEAGE, "PREDICTIONS_SEAL_LINEAGE")
    need(seal.get("predictions") == 150 and seal.get("targets_opened") is False
         and seal.get("scored_outcomes") == 0, "PREDICTIONS_SEAL_CONTENT")
    manifest_targets = manifest["exports"][TARGET_EXPORT]
    need(seal.get("targets_sha256") == manifest_targets["sha256"] == args.targets_sha256,
         "TARGETS_IDENTITY_DISAGREEMENT")

    # 4. Both upstream query seals, authenticated by the hashes the seal declares.
    upstream = {}
    for stage, tag, count in (("parent", args.parent_tag, 125), ("oracle_e", args.oracle_e_tag, 25)):
        expected = seal[stage + "_seal_sha256"]
        sub_schema, sub_status = STAGE_SCHEMAS[stage]
        sub, sub_id = check_seal(guard, stage.upper(), stage_seal_path(manifest, stage, tag),
                                 expected, sub_status, sub_schema)
        need(sub["activation_sha256"] == activation["sha256"], stage.upper() + "_SEAL_ACTIVATION")
        need(sub["queries"] == count and sub["targets_opened"] is False,
             stage.upper() + "_SEAL_CONTENT")
        upstream[stage] = sub_id
    need(upstream["parent"]["sha256"] == seal["parent_seal_sha256"], "PARENT_SEAL_IDENTITY")

    # 5. Prediction artifacts, by the identities the predictions seal names.
    exact_keys(seal["artifacts"], ("predictions.jsonl", "prediction_records.jsonl"),
               "PREDICTION_ARTIFACTS")
    artifacts = {}
    for name, spec in sorted(seal["artifacts"].items()):
        exact_keys(spec, ("sha256", "bytes"), "PREDICTION_IDENTITY")
        need(isinstance(spec["sha256"], str) and HEX256.fullmatch(spec["sha256"])
             and type(spec["bytes"]) is int, "PREDICTION_IDENTITY_TYPES")
        raw, actual = guard.read_small(prediction_dir / name, spec["sha256"])
        need(actual["bytes"] == spec["bytes"], "PREDICTION_ARTIFACT_BYTES: " + name)
        artifacts[name] = (jsonl_rows(raw), actual)
    records, record_id = artifacts["prediction_records.jsonl"]
    predictions, prediction_id = artifacts["predictions.jsonl"]

    # 6. Completeness and ranking. Targets have NOT been opened at this point.
    need(len(records) == 150, "PREDICTION_RECORD_COUNT")
    seen = set()
    for record, view in zip(records, predictions):
        validate_prediction_record(record, record.get("arm"))
        need(record["activation_sha256"] == activation["sha256"], "RECORD_ACTIVATION")
        need(record["parent_seal_sha256"] == seal["parent_seal_sha256"]
             and record["oracle_e_seal_sha256"] == seal["oracle_e_seal_sha256"],
             "RECORD_UPSTREAM_BINDING")
        need(record["prediction"] == view, "RECORD_VIEW_DRIFT")
        key = (record["qid"], record["arm"])
        need(key not in seen, "DUPLICATE_PREDICTION_RECORD")
        seen.add(key)
    validate_predictions(predictions)

    # 7. Only now may the sealed targets be opened.
    guard.allow("targets", manifest_targets["path"])
    raw_targets, target_id = guard.read_small(manifest_targets["path"], args.targets_sha256)
    need(target_id["bytes"] == manifest_targets["bytes"], "TARGETS_BYTES")
    result = summarize(predictions, jsonl_rows(raw_targets))
    result["provenance"] = {
        "activation": {"sha256": activation["sha256"], "bytes": activation["bytes"]},
        "preparation_seal": prep_id, "preflight_seal": pre_id,
        "parent_seal": upstream["parent"], "oracle_e_seal": upstream["oracle_e"],
        "prompt_preflight": prompt_id,
        "predictions_seal": seal_id, "predictions": prediction_id,
        "prediction_records": record_id, "targets": target_id,
        "scorer_sha256": sha256(Path(__file__).read_bytes()),
        "core_sha256": sha256(Path(__file__).with_name("bridge_pilot_core.py").read_bytes()),
        "code_identity": code_identity,
        "runner_sha256": sha256(Path(__file__).with_name("bridge_pilot_run.py").read_bytes()),
        "historical_global_lineage": LINEAGE,
        "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED",
        "cohort_scope": "ASSISTED_EXPLORATORY_PILOT; CONDITIONAL_EVIDENCE_NOT_POPULATION_PROOF"}
    # Recheck every input before creating output. A partial run is never scored 0.
    read_small(args.predictions_seal, args.seal_sha256)
    for name, spec in sorted(seal["artifacts"].items()):
        read_small(prediction_dir / name, spec["sha256"])
    read_small(manifest_targets["path"], args.targets_sha256)
    read_small(args.activation, args.activation_sha256)

    os.umask(0o077)
    new_dir(out)
    identity = write_json(out / "scores.json", result)
    score_seal = {"schema": SCORING_SCHEMA, "status": "SEALED_SCORES",
                  "activation_sha256": activation["sha256"],
                  "provenance": result["provenance"], "artifacts": {"scores.json": identity},
                  "historical_global_lineage": LINEAGE}
    write_json(out / "SCORING_SEAL.json", score_seal)
    print(canonical({"status": "SCORING_COMPLETE", "accepted_qids": 25, "accepted_pairs": 36,
                     "output": str(out), "decision": result["decision"],
                     "primary_gate_passed": result["primary_gate_passed"],
                     "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED",
                     "experiment_state": manifest["experiment_state"],
                     "historical_global_lineage": LINEAGE}))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(canonical({"status": "STOPPED", "error_type": type(exc).__name__, "error": str(exc),
                         "historical_global_lineage": LINEAGE,
                         "outputs_must_not_be_deleted": True}), file=sys.stderr)
        sys.exit(2)
