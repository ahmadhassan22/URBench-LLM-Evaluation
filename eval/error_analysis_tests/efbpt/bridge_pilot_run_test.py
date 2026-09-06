#!/usr/bin/env python3
"""Synthetic runtime/scorer tests. No real model, index, corpus or pilot export.

Every fixture is generated in a throwaway temporary directory. Nothing here
loads Qwen or MiniLM weights, deserializes the FAISS index, reads the real
metadata/offsets, opens the real preparation exports or scoring targets, or
produces any pilot outcome.

Run: python -B eval/error_analysis_tests/efbpt/bridge_pilot_run_test.py
"""
import sys
sys.dont_write_bytecode = True
import copy
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from bridge_pilot_core import (ARMS, BUDGET, CHILD_COUNTS, DIM, N_VECTORS, SEED, PilotError,
                               aggregate_candidates, assert_prompt_hashes, canonical,
                               cap_query, local_chunks, norm, parse_state, render_prompt,
                               search_to_metadata, sha256, signflip, summarize,
                               validate_predictions, validate_targets)
import bridge_pilot_core as core
import bridge_pilot_run as run
import bridge_pilot_score as score

HERE = Path(__file__).resolve().parent
QIDS = sorted(CHILD_COUNTS)

# ------------------------------------------------------------------ fixtures

def synthetic_page(text="  اردو aaa London is a city.\nParis is another city. ",
                   title="Synthetic Parent"):
    return {"raw_page_id": "synthetic", "raw_url": "https://example.invalid/parent",
            "raw_title": title, "raw_text": text, "raw_text_sha256": sha256(text),
            "locations": [{"shard_index": 0, "blob_path": "/synthetic/unused",
                           "blob_sha256": "0" * 64, "row_group": 0, "row_in_group": 0,
                           "row_in_shard": 0}],
            "lookup_decision": "SYNTHETIC", "normalized_bucket_rows": 1,
            "normalized_bucket_distinct_pages": 1}

def synthetic_parent(qid):
    page = synthetic_page(title="Synthetic Parent " + qid)
    return {"qid": qid, "question_ur": "یہ کیا ہے؟",
            "parent_source_instance_id": "synthetic-parent-" + qid,
            "parent_title": "Synthetic Parent " + qid,
            "parent_normalized_title": norm("Synthetic Parent " + qid),
            "page": page, "chunks": local_chunks(page)}

def synthetic_targets():
    rows = []
    for qid in QIDS:
        rows.append({"qid": qid, "children": [
            {"source_instance_id": qid + "-" + str(j), "title": "Child " + qid + " " + str(j),
             "normalized_title": norm("Child " + qid + " " + str(j))}
            for j in range(CHILD_COUNTS[qid])]})
    return rows

def synthetic_candidates(qid, arm, hit_children=0):
    gold = ["Child " + qid + " " + str(j) for j in range(CHILD_COUNTS[qid])]
    out = []
    for j in range(BUDGET):
        title = gold[j] if j < min(hit_children, len(gold)) else "Distractor " + str(j)
        out.append({"global_row": j + 1, "score": float(BUDGET - j), "title": title})
    return out

def scoring_view(qid, arm, hit_children=0):
    cands = synthetic_candidates(qid, arm, hit_children)
    return {"qid": qid, "arm": arm, "query": "synthetic query", "candidates": cands,
            "ranked_titles": aggregate_candidates(cands),
            "boundary_tie_within_returned": cands[-1]["score"] == cands[-2]["score"],
            "ties_beyond_budget": "UNOBSERVED_NO_OVERFETCH"}

def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    return {"path": str(path), "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size}

def write_json(path, obj):
    return write(path, canonical(obj) + "\n")

def write_jsonl(path, rows):
    return write(path, "".join(canonical(r) + "\n" for r in rows))

class Tree:
    """A complete synthetic activation tree: manifest, seals, exports, stages."""

    def __init__(self, base, quantization="plain_bfloat16"):
        self.quantization = quantization
        self.base = Path(base)
        self.inputs = self.base / "inputs"
        self.out = self.base / "outputs"
        self.archive = self.base / "archive"
        self.out.mkdir(parents=True)
        self.archive.mkdir(parents=True)
        self.exports = {
            run.PARENT_EXPORT: write_jsonl(self.inputs / run.PARENT_EXPORT,
                                           [synthetic_parent(q) for q in QIDS]),
            run.ORACLE_EXPORT: write_jsonl(self.inputs / run.ORACLE_EXPORT,
                                           [{"qid": q, "question_ur": "س", "urbench_facts": ["f1"]}
                                            for q in QIDS]),
            run.TARGET_EXPORT: write_jsonl(self.inputs / run.TARGET_EXPORT, synthetic_targets()),
        }
        self.prep_seal = write_json(self.inputs / "PREPARATION_SEAL.json", {
            "status": run.PREPARATION_STATUS,
            "artifacts": {n: {"sha256": v["sha256"], "bytes": v["bytes"]}
                          for n, v in self.exports.items()}})
        self.pre_seal = write_json(self.inputs / "PREFLIGHT_SEAL.json", {
            "status": run.PREFLIGHT_STATUS, "historical_global_lineage": run.LINEAGE,
            "preparation_seal_sha256": self.prep_seal["sha256"]})
        self.models = {}
        for key in ("qwen", "encoder"):
            root = self.inputs / ("model_" + key)
            files = {}
            for name in ("config.json", "tokenizer.json"):
                files[name] = write(root / name, '{"synthetic":"' + key + name + '"}')
            self.models[key] = {"root": str(root),
                                "files": {n: {"sha256": v["sha256"], "bytes": v["bytes"]}
                                          for n, v in files.items()}}
        self.assets = {k: write(self.inputs / ("asset_" + k), "synthetic asset " + k)
                       for k in run.ASSET_KEYS}
        self.chat_template_sha256 = sha256("synthetic-chat-template")
        # The prompt preflight was sealed by an earlier runner revision on purpose,
        # so the allowed, documented transition is exercised by every test.
        self.historical_runner = sha256("historical-runner-revision")
        self.min_margin = 17126
        self.build_prompt_preflight()
        self.manifest_path = self.base / "ACTIVATION.json"
        self.manifest = self.build_manifest()
        self.identity = write_json(self.manifest_path, self.manifest)

    def build_prompt_preflight(self, **overrides):
        maxima = {"state_C": {"qid": QIDS[0], "arm": "C", "reserve": 1024,
                              "input_tokens": 40960 - 1024 - self.min_margin,
                              "total_tokens": 40960 - self.min_margin,
                              "remaining_margin": self.min_margin, "status": "WITHIN_CONTEXT"}}
        for name, reserve in (("state_D", 1024), ("query_A", 128), ("query_P", 128),
                              ("query_B", 128), ("query_E", 128)):
            maxima[name] = {"qid": QIDS[1], "arm": name[-1], "reserve": reserve,
                            "input_tokens": 100, "total_tokens": 100 + reserve,
                            "remaining_margin": 40960 - 100 - reserve,
                            "status": "WITHIN_CONTEXT"}
        summary = write_json(self.inputs / "prompt_preflight_summary.json",
                             {"schema": "urbench.bridge_n25.prompt_preflight_summary.v1",
                              "checked_records": 150, "maxima": maxima})
        self.pp_summary = summary
        seal = {"schema": "urbench.bridge_n25.prompt_preflight_seal.v1",
                "status": run.PROMPT_PREFLIGHT_STATUS, "version": "0.1",
                "preparation_seal_sha256": self.prep_seal["sha256"],
                "preflight_seal_sha256": self.pre_seal["sha256"],
                "core_sha256": sha256((HERE / "bridge_pilot_core.py").read_bytes()),
                "runner_sha256": self.historical_runner,
                "chat_template_sha256": self.chat_template_sha256,
                "prompt_constants": dict(sorted(assert_prompt_hashes().items())),
                "artifacts": {"prompt_preflight_summary.json":
                              {"sha256": summary["sha256"], "bytes": summary["bytes"]}},
                "checked_records": 150, "categories_checked": 6, "maxima": maxima,
                "all_known_prompts_within_context": True,
                "runtime_dependent": {"scope": "RUNTIME_DEPENDENT",
                                      "categories": ["query_C", "query_D"]},
                "qwen_weight_files_opened": 0, "qwen_generations": 0, "pilot_searches": 0,
                "scoring_targets_opened": False,
                "experiment_state": run.EXPERIMENT_STATE,
                "historical_global_lineage": run.LINEAGE,
                "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED"}
        seal.update(overrides)
        self.pp_seal_obj = seal
        self.pp_seal = write_json(self.inputs / "PROMPT_PREFLIGHT_SEAL.json", seal)
        return self.pp_seal

    def code_section(self):
        section = {}
        for name in run.CODE_FILES:
            raw = (HERE / name).read_bytes()
            section[name] = {"path": str(HERE / name), "sha256": sha256(raw), "bytes": len(raw)}
        return section

    def build_manifest(self):
        template = run.manifest_template()
        # The template ships a placeholder, not a default: the reviewed manifest
        # must pin one of the two allowed loading semantics explicitly.
        template["settings"]["qwen_load"]["quantization"] = self.quantization
        template["protocol"] = dict(self.prep_seal)
        template["amendment"] = dict(self.pre_seal)
        template["code"] = self.code_section()
        template["preparation"] = {"seal_path": self.prep_seal["path"],
                                   "seal_sha256": self.prep_seal["sha256"],
                                   "artifacts": {n: {"sha256": v["sha256"], "bytes": v["bytes"]}
                                                 for n, v in self.exports.items()}}
        template["preflight"] = {"seal_path": self.pre_seal["path"],
                                 "seal_sha256": self.pre_seal["sha256"]}
        template["prompt_preflight"] = {
            "seal_path": self.pp_seal["path"], "seal_sha256": self.pp_seal["sha256"],
            "summary_path": self.pp_summary["path"],
            "summary_sha256": self.pp_summary["sha256"],
            "historical_runner_sha256": self.historical_runner,
            "checked_records": 150, "categories_checked": 6,
            "minimum_remaining_context_margin": self.min_margin,
            "runtime_dependent_categories": ["query_C", "query_D"]}
        template["settings"]["tokenizer"]["chat_template_sha256"] = self.chat_template_sha256
        template["cohort"] = {"qids": 25, "accepted_pairs": 36, "arms": list(ARMS),
                              "label": "HUMAN_VERIFICATION_OF_MODEL_ASSISTED_CANDIDATES",
                              "predictions": 150}
        template["provenance"] = {"git_parent_head": "e" * 40,
                                  "activation_utc": "2026-09-05T00:00:00+00:00",
                                  "amendment_number": 2, "outcomes_viewed": False}
        template["limitations"] = ["ASSISTED_EXPLORATORY_PILOT_NOT_CANONICAL_COMPLETION"]
        template["exports"] = {n: dict(v) for n, v in self.exports.items()}
        template["models"] = self.models
        template["assets"] = {k: dict(v) for k, v in self.assets.items()}
        template["environment"] = {"python": ".".join(str(x) for x in sys.version_info[:3]),
                                   "executable": sys.executable, "packages": {"numpy": _numpy_version()}}
        template["roots"] = {"output_root": str(self.out), "archive_root": str(self.archive),
                             "stage_tag": "v1",
                             "score_output_root": str(self.out / "score_v1")}
        return template

    def rewrite(self, manifest):
        self.manifest = manifest
        self.identity = write(self.manifest_path, canonical(manifest) + "\n")
        return self.identity

    # -- sealed upstream stages, produced without any model or index ----------

    def seal_stage(self, stage, tag, artifact_rows, extra):
        root = self.out / (stage + "_" + tag)
        root.mkdir(parents=True, exist_ok=True)
        schema, status = run.STAGE_SCHEMAS[stage]
        artifacts = {}
        for name, rows in sorted(artifact_rows.items()):
            artifacts[name] = {k: v for k, v in write_jsonl(root / name, rows).items()
                               if k in ("sha256", "bytes")}
        seal = {"schema": schema, "status": status, "stage": stage, "tag": tag,
                "runner_version": run.VERSION, "activation_sha256": self.identity["sha256"],
                "artifacts": artifacts, "historical_global_lineage": run.LINEAGE,
                "experiment_state": run.EXPERIMENT_STATE, **extra}
        return write_json(root / "STAGE_SEAL.json", seal), seal

    def query_record(self, qid, arm):
        row = {"schema": "urbench.bridge_n25.generation.v1", "pass": "query_" + arm, "qid": qid,
               "arm": arm, "activation_sha256": self.identity["sha256"],
               "runner_version": run.VERSION,
               "preparation_seal_sha256": self.prep_seal["sha256"],
               "preflight_seal_sha256": self.pre_seal["sha256"],
               "source_row": {"qid": qid, "row_sha256": "1" * 64},
               "model_identity": {"root": "synthetic"},
               "encoder_tokenizer_identity": {"root": "synthetic"},
               "prompt_constants": dict(sorted(assert_prompt_hashes().items())),
               "prompt_sha256": sha256(qid + arm), "input_tokens": 10, "max_new_tokens": 128,
               "output_tokens": 5, "length_capped": False,
               "decoding": {"do_sample": False, "num_beams": 1, "state_max_new_tokens": 1024,
                            "query_max_new_tokens": 128, "microbatch": 1,
                            "model_context_limit": 40960},
               "raw_output": "synthetic query", "generation_status": "COMPLETED",
               "result": {"raw_output": "synthetic query", "query": "synthetic query",
                          "fallback": None, "model_empty": False, "word_cap_removed": 0,
                          "encoder_cap_removed": 0, "encoder_tokens_before_cap": 4,
                          "encoder_tokens": 4},
               "historical_global_lineage": run.LINEAGE, "utc": "2026-09-05T00:00:00+00:00"}
        run.validate_generation_record(row, arm, "query_" + arm)
        return row

    def prediction_record(self, qid, arm, parent_sha, e_sha, hits=0):
        view = scoring_view(qid, arm, hits)
        query_row = self.query_record(qid, arm)
        return {"schema": "urbench.bridge_n25.prediction.v1", "pass": "prediction_" + arm,
                "qid": qid, "arm": arm, "activation_sha256": self.identity["sha256"],
                "runner_version": run.VERSION,
                "preparation_seal_sha256": self.prep_seal["sha256"],
                "preflight_seal_sha256": self.pre_seal["sha256"],
                "parent_seal_sha256": parent_sha, "oracle_e_seal_sha256": e_sha,
                "query_record": {"pass": "query_" + arm, "qid": qid, "arm": arm,
                                 "prompt_sha256": query_row["prompt_sha256"],
                                 "record_sha256": sha256(canonical(query_row)),
                                 "fallback": None, "encoder_tokens": 4},
                "encoder_identity": {"root": "synthetic"},
                "index_identity": {"sha256": self.assets["index"]["sha256"]},
                "metadata_identity": {"sha256": self.assets["metadata"]["sha256"]},
                "offsets_identity": {"sha256": self.assets["offsets"]["sha256"]},
                "search_budget": BUDGET,
                "candidate_provenance": [{"global_row": c["global_row"], "byte_offset": c["global_row"],
                                          "metadata_line_sha256": sha256(str(c["global_row"]))}
                                         for c in view["candidates"]],
                "prediction": view, "historical_global_lineage": run.LINEAGE,
                "utc": "2026-09-05T00:00:00+00:00"}

    def build_full_chain(self, hits_for=lambda qid, arm: CHILD_COUNTS[qid] if arm == "D" else 0,
                         prediction_rows=None):
        parent_id, _ = self.seal_stage(
            "parent", "v1",
            {"queries.jsonl": [self.query_record(q, a) for a in ("A", "P", "B", "C", "D") for q in QIDS],
             "states.jsonl": []},
            {"states": 50, "queries": 125, "preparation_seal_sha256": self.prep_seal["sha256"],
             "preflight_seal_sha256": self.pre_seal["sha256"], "export_identity": {},
             "oracle_e_opened": False, "targets_opened": False, "pilot_searches": 0,
             "scored_outcomes": 0})
        e_id, _ = self.seal_stage(
            "oracle_e", "v1", {"queries_e.jsonl": [self.query_record(q, "E") for q in QIDS]},
            {"queries": 25, "parent_seal_sha256": parent_id["sha256"],
             "preparation_seal_sha256": self.prep_seal["sha256"],
             "preflight_seal_sha256": self.pre_seal["sha256"], "export_identity": {},
             "parent_only_opened": False, "targets_opened": False, "pilot_searches": 0,
             "scored_outcomes": 0})
        if prediction_rows is None:
            prediction_rows = [self.prediction_record(q, a, parent_id["sha256"], e_id["sha256"],
                                                      hits_for(q, a))
                               for q in QIDS for a in ARMS]
        pred_id, _ = self.seal_stage(
            "retrieve", "v1",
            {"prediction_records.jsonl": prediction_rows,
             "predictions.jsonl": [r["prediction"] for r in prediction_rows]},
            {"predictions": 150, "targets_sha256": self.exports[run.TARGET_EXPORT]["sha256"],
             "parent_seal_sha256": parent_id["sha256"], "oracle_e_seal_sha256": e_id["sha256"],
             "preparation_seal_sha256": self.prep_seal["sha256"],
             "preflight_seal_sha256": self.pre_seal["sha256"], "search_budget": BUDGET,
             "targets_opened": False, "scored_outcomes": 0})
        return parent_id, e_id, pred_id

def _numpy_version():
    import importlib.metadata
    return importlib.metadata.version("numpy")

class TreeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bridge_pilot_test_")
        self.tree = Tree(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

# ----------------------------------------------------- activation authority

class ActivationTests(TreeCase):
    def test_manifest_authenticates_and_binds_lineage(self):
        m, ident = run.load_activation(self.tree.manifest_path, self.tree.identity["sha256"])
        self.assertEqual(m["historical_global_lineage"], "UNESTABLISHED")
        # The manifest declares the activated state; the three pre-activation
        # seals keep their historical NOT_FROZEN_NOT_RUN value unchanged.
        self.assertEqual(m["experiment_state"], "FROZEN_NOT_RUN")
        self.assertEqual(run.ACTIVATED_EXPERIMENT_STATE, "FROZEN_NOT_RUN")
        self.assertEqual(run.EXPERIMENT_STATE, "NOT_FROZEN_NOT_RUN")
        self.assertEqual(self.tree.pp_seal_obj["experiment_state"], "NOT_FROZEN_NOT_RUN")
        self.assertEqual(m["canonical_stage0"], "INCOMPLETE_GATES_UNCHANGED")
        self.assertEqual(ident["sha256"], self.tree.identity["sha256"])

    def test_template_ships_a_placeholder_not_a_default(self):
        template = run.manifest_template()
        self.assertNotIn(template["settings"]["qwen_load"]["quantization"],
                         run.QWEN_LOAD_CHOICES)
        self.assertEqual(template["historical_global_lineage"], "UNESTABLISHED")
        self.assertEqual(template["experiment_state"], "FROZEN_NOT_RUN")

    def test_both_reviewed_loading_semantics_validate(self):
        for choice in run.QWEN_LOAD_CHOICES:
            with self.subTest(choice=choice):
                m = copy.deepcopy(self.tree.manifest)
                m["settings"]["qwen_load"]["quantization"] = choice
                ident = write(self.tree.base / (choice + ".json"), canonical(m) + "\n")
                loaded, _ = run.load_activation(self.tree.base / (choice + ".json"),
                                                ident["sha256"])
                self.assertEqual(loaded["settings"]["qwen_load"]["quantization"], choice)

    def test_arbitrary_activation_refused(self):
        # A self-consistent manifest at an unreviewed hash cannot be used.
        with self.assertRaises(PilotError):
            run.load_activation(self.tree.manifest_path, "0" * 64)
        with self.assertRaises(PilotError):
            run.load_activation(self.tree.manifest_path, "not-a-hash")

    def test_manifest_tampering_refused(self):
        for mutate in (
                lambda m: m.__setitem__("historical_global_lineage", "ESTABLISHED"),
                lambda m: m.__setitem__("experiment_state", "FROZEN_AND_RUN"),
                lambda m: m.__setitem__("status", "SELF_DECLARED"),
                lambda m: m.__setitem__("canonical_stage0", "PASSED"),
                lambda m: m.__setitem__("extra_field", 1),
                lambda m: m["settings"].__setitem__("seed", 1),
                lambda m: m["settings"]["arms"].remove("P"),
                lambda m: m["settings"]["decoding"].__setitem__("do_sample", True),
                lambda m: m["settings"]["decoding"].__setitem__("state_max_new_tokens", 512),
                lambda m: m["settings"]["decoding"].__setitem__("microbatch", 2),
                lambda m: m["settings"]["retrieval"].__setitem__("budget", 50),
                lambda m: m["settings"]["retrieval"].__setitem__("verify_asset_hashes", False),
                lambda m: m["settings"]["encoder"].__setitem__("normalize_embeddings", False),
                lambda m: m["settings"]["qwen_load"].__setitem__("quantization", "fp8_experimental"),
                lambda m: m["settings"]["qwen_load"].__setitem__("trust_remote_code", True),
                lambda m: m["settings"]["aggregation"].__setitem__("max_titles", 20),
                lambda m: m["exports"].pop(run.TARGET_EXPORT),
                lambda m: m["code"].pop("bridge_pilot_run.py"),
        ):
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(self.tree.manifest)
                mutate(bad)
                ident = write(self.tree.base / "bad.json", canonical(bad) + "\n")
                with self.assertRaises(PilotError):
                    run.load_activation(self.tree.base / "bad.json", ident["sha256"])

    def test_search_budget_cannot_be_reduced_to_50(self):
        bad = copy.deepcopy(self.tree.manifest)
        bad["settings"]["retrieval"]["budget"] = 50
        ident = write(self.tree.base / "b50.json", canonical(bad) + "\n")
        with self.assertRaises(PilotError):
            run.load_activation(self.tree.base / "b50.json", ident["sha256"])

    def test_output_input_overlap_refused(self):
        bad = copy.deepcopy(self.tree.manifest)
        bad["roots"]["output_root"] = str(self.tree.inputs)
        ident = write(self.tree.base / "overlap.json", canonical(bad) + "\n")
        with self.assertRaises(PilotError):
            run.load_activation(self.tree.base / "overlap.json", ident["sha256"])

    def test_preparation_preflight_mismatch_refused(self):
        bad_seal = write_json(self.tree.inputs / "PREFLIGHT_BAD.json",
                              {"status": run.PREFLIGHT_STATUS,
                               "historical_global_lineage": run.LINEAGE,
                               "preparation_seal_sha256": "9" * 64})
        m = copy.deepcopy(self.tree.manifest)
        m["preflight"] = {"seal_path": bad_seal["path"], "seal_sha256": bad_seal["sha256"]}
        ident = self.tree.rewrite(m)
        manifest, activation = run.load_activation(self.tree.manifest_path, ident["sha256"])
        with self.assertRaises(PilotError):
            run.check_inputs(manifest, activation, "parent")

    def test_seal_status_tampering_refused(self):
        write_json(self.tree.inputs / "PREPARATION_SEAL.json",
                   {"status": "SELF_DECLARED_OK", "artifacts": {}})
        manifest, activation = run.load_activation(self.tree.manifest_path,
                                                   self.tree.identity["sha256"])
        with self.assertRaises(PilotError):
            run.check_inputs(manifest, activation, "parent")

    def test_code_identity_drift_refused(self):
        m = copy.deepcopy(self.tree.manifest)
        m["code"]["bridge_pilot_run.py"]["sha256"] = "a" * 64
        ident = self.tree.rewrite(m)
        manifest, activation = run.load_activation(self.tree.manifest_path, ident["sha256"])
        with self.assertRaises(PilotError):
            run.check_inputs(manifest, activation, "parent")

# ------------------------------------------- prompt-preflight seal binding

class PromptPreflightBindingTests(TreeCase):
    """The audit seal was produced by an earlier runner. That one transition is
    declared in the manifest and accepted; every other mismatch is refused."""

    def load(self):
        return run.load_activation(self.tree.manifest_path, self.tree.identity["sha256"])

    def rebuild(self, **overrides):
        """Re-seal the synthetic prompt preflight, then re-point the manifest."""
        self.tree.build_prompt_preflight(**overrides)
        m = self.tree.build_manifest()
        return self.tree.rewrite(m)

    def check(self, ident=None):
        ident = ident or self.tree.identity
        manifest, activation = run.load_activation(self.tree.manifest_path, ident["sha256"])
        return run.check_inputs(manifest, activation, "parent")

    def test_documented_runner_transition_is_accepted(self):
        report = self.check()
        pp = report["prompt_preflight"]
        self.assertEqual(pp["historical_runner_sha256"], self.tree.historical_runner)
        self.assertEqual(pp["active_runner_sha256"],
                         sha256((HERE / "bridge_pilot_run.py").read_bytes()))
        self.assertNotEqual(pp["historical_runner_sha256"], pp["active_runner_sha256"])
        self.assertTrue(pp["runner_revision_changed"])

    def test_unrelated_runner_hash_refused(self):
        ident = self.rebuild(runner_sha256="c" * 64)
        with self.assertRaises(PilotError) as ctx:
            self.check(ident)
        self.assertIn("PROMPT_PREFLIGHT_RUNNER_UNKNOWN", str(ctx.exception))

    def test_seal_claiming_the_active_runner_refused(self):
        # Passes the "known value" test but is not the declared historical one.
        active = sha256((HERE / "bridge_pilot_run.py").read_bytes())
        ident = self.rebuild(runner_sha256=active)
        with self.assertRaises(PilotError) as ctx:
            self.check(ident)
        self.assertIn("PROMPT_PREFLIGHT_HISTORICAL_RUNNER_DRIFT", str(ctx.exception))

    def test_core_must_be_unchanged(self):
        ident = self.rebuild(core_sha256="d" * 64)
        with self.assertRaises(PilotError) as ctx:
            self.check(ident)
        self.assertIn("PROMPT_PREFLIGHT_CORE_DRIFT", str(ctx.exception))

    def test_counts_failures_and_margin_are_bound(self):
        for overrides, expected in (
                ({"checked_records": 149}, "PROMPT_PREFLIGHT_RECORD_COUNT"),
                ({"categories_checked": 5}, "PROMPT_PREFLIGHT_CATEGORY_COUNT"),
                ({"all_known_prompts_within_context": False}, "PROMPT_PREFLIGHT_CONTEXT_FAILURE"),
                ({"runtime_dependent": {"scope": "MEASURED", "categories": ["query_C", "query_D"]}},
                 "PROMPT_PREFLIGHT_RUNTIME_DEPENDENT_DRIFT"),
                ({"qwen_generations": 1}, "PROMPT_PREFLIGHT_NOT_OUTCOME_FREE"),
                ({"scoring_targets_opened": True}, "PROMPT_PREFLIGHT_NOT_OUTCOME_FREE"),
                ({"pilot_searches": 3}, "PROMPT_PREFLIGHT_NOT_OUTCOME_FREE"),
                ({"qwen_weight_files_opened": 8}, "PROMPT_PREFLIGHT_NOT_OUTCOME_FREE"),
                ({"historical_global_lineage": "ESTABLISHED"}, "PROMPT_PREFLIGHT_LINEAGE"),
                ({"experiment_state": "FROZEN_NOT_RUN"}, "PROMPT_PREFLIGHT_HISTORICAL_STATE"),
                ({"chat_template_sha256": "f" * 64}, "PROMPT_PREFLIGHT_CHAT_TEMPLATE_DRIFT"),
                ({"prompt_constants": {"system": "0" * 64}},
                 "PROMPT_PREFLIGHT_PROMPT_CONSTANT_DRIFT"),
                ({"preparation_seal_sha256": "a" * 64}, "PROMPT_PREFLIGHT_UPSTREAM_BINDING"),
                ({"artifacts": {"prompt_preflight_summary.json": {"sha256": "b" * 64, "bytes": 1}}},
                 "PROMPT_PREFLIGHT_SUMMARY_NOT_SEALED")):
            with self.subTest(overrides=sorted(overrides)):
                ident = self.rebuild(**overrides)
                with self.assertRaises(PilotError) as ctx:
                    self.check(ident)
                self.assertIn(expected, str(ctx.exception))

    def test_margin_drift_between_seal_and_manifest_refused(self):
        self.tree.min_margin = 17127          # manifest claims a margin the seal denies
        self.tree.build_prompt_preflight()
        self.tree.min_margin = 17126
        m = self.tree.build_manifest()
        m["prompt_preflight"]["minimum_remaining_context_margin"] = 99999
        ident = self.tree.rewrite(m)
        with self.assertRaises(PilotError) as ctx:
            self.check(ident)
        self.assertIn("PROMPT_PREFLIGHT_MARGIN_DRIFT", str(ctx.exception))

    def test_summary_hash_drift_refused(self):
        path = Path(self.tree.pp_summary["path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(PilotError):
            self.check()

    def test_every_stage_and_the_scorer_authenticate_it(self):
        source = (HERE / "bridge_pilot_run.py").read_text()
        # Defined once, called once from the single shared upstream authenticator,
        # so all four call sites and the scorer inherit it.
        self.assertEqual(source.count("def authenticate_prompt_preflight("), 1)
        self.assertEqual(source.count("prompt_id = authenticate_prompt_preflight("), 1)
        self.assertEqual(source.count("prep_id, pre_id, prompt_id = authenticate_upstream("), 4)
        scorer = (HERE / "bridge_pilot_score.py").read_text()
        self.assertIn("authenticate_upstream(guard, manifest)", scorer)
        self.assertIn("prompt_preflight_seal", scorer)

class ActivationRootsTests(TreeCase):
    def test_stage_tag_must_be_the_reviewed_one(self):
        manifest, activation = run.load_activation(self.tree.manifest_path,
                                                   self.tree.identity["sha256"])
        run.StageRoot("parent", manifest, activation, "v1")
        with self.assertRaises(PilotError) as ctx:
            run.StageRoot("parent", manifest, activation, "v2")
        self.assertIn("STAGE_TAG_NOT_REVIEWED", str(ctx.exception))

    def test_all_four_stage_destinations_must_be_fresh(self):
        manifest, activation = run.load_activation(self.tree.manifest_path,
                                                   self.tree.identity["sha256"])
        run.check_inputs(manifest, activation, "parent")
        for root, stage in ((self.tree.out, "retrieve"), (self.tree.archive, "score")):
            target = root / (stage + "_v1")
            target.mkdir(parents=True)
            with self.assertRaises(PilotError) as ctx:
                run.check_inputs(manifest, activation, "parent")
            self.assertIn("STAGE_DESTINATION_NOT_FRESH", str(ctx.exception))
            target.rmdir()

    def test_score_root_must_match_stage_tag(self):
        m = copy.deepcopy(self.tree.manifest)
        m["roots"]["score_output_root"] = str(self.tree.out / "score_elsewhere")
        ident = write(self.tree.base / "badroot.json", canonical(m) + "\n")
        with self.assertRaises(PilotError):
            run.load_activation(self.tree.base / "badroot.json", ident["sha256"])

    def test_cohort_provenance_and_limitations_are_bound(self):
        for mutate in (lambda m: m["cohort"].__setitem__("qids", 24),
                       lambda m: m["cohort"].__setitem__("accepted_pairs", 35),
                       lambda m: m["cohort"].__setitem__("predictions", 149),
                       lambda m: m["cohort"].__setitem__("label", "CANONICAL"),
                       lambda m: m["cohort"]["arms"].remove("E"),
                       lambda m: m["provenance"].__setitem__("outcomes_viewed", True),
                       lambda m: m["provenance"].__setitem__("amendment_number", 1),
                       lambda m: m["provenance"].__setitem__("git_parent_head", "zz"),
                       lambda m: m.__setitem__("limitations", []),
                       lambda m: m["settings"]["statistics"].__setitem__("effect_threshold_pp", 5),
                       lambda m: m["settings"]["statistics"].__setitem__("alpha", 0.1),
                       lambda m: m["settings"]["statistics"].__setitem__("bootstrap_seed", 1),
                       lambda m: m["settings"]["statistics"].__setitem__("bootstrap_resamples", 1000),
                       lambda m: m["settings"]["statistics"].__setitem__("primary", "D-B"),
                       lambda m: m["settings"]["statistics"].__setitem__("sampling_unit", "PAIR"),
                       lambda m: m["settings"]["tokenizer"]["apply_chat_template_kwargs"]
                                  .__setitem__("enable_thinking", True),
                       lambda m: m["settings"]["tokenizer"]["tokenize_kwargs"]
                                  .__setitem__("truncation", True)):
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(self.tree.manifest)
                mutate(bad)
                ident = write(self.tree.base / "m.json", canonical(bad) + "\n")
                with self.assertRaises(PilotError):
                    run.load_activation(self.tree.base / "m.json", ident["sha256"])

    def test_quantization_is_the_reviewed_nf4_choice(self):
        m = copy.deepcopy(self.tree.manifest)
        m["settings"]["qwen_load"]["quantization"] = "bnb_nf4_double_bfloat16"
        ident = write(self.tree.base / "nf4.json", canonical(m) + "\n")
        loaded, _ = run.load_activation(self.tree.base / "nf4.json", ident["sha256"])
        self.assertEqual(loaded["settings"]["qwen_load"]["quantization"],
                         "bnb_nf4_double_bfloat16")
        self.assertEqual(loaded["settings"]["decoding"]["microbatch"], 1)
        self.assertFalse(loaded["settings"]["qwen_load"]["trust_remote_code"])
        self.assertEqual(loaded["settings"]["retrieval"]["budget"], 100)
        self.assertEqual(loaded["settings"]["aggregation"]["max_titles"], 10)

# ------------------------------------------------------------ input boundary

class BoundaryTests(TreeCase):
    def test_every_stage_forbids_the_scoring_targets(self):
        for stage in run.STAGE_UNITS:
            self.assertIn(run.TARGET_EXPORT, run.STAGE_FORBIDDEN[stage])

    def test_parent_stage_cannot_register_oracle_or_targets(self):
        guard = run.stage_guard("parent", self.tree.manifest)
        for name in (run.ORACLE_EXPORT, run.TARGET_EXPORT):
            with self.assertRaises(PilotError):
                guard.allow("sneaky", self.tree.exports[name]["path"])
        guard.allow("ok", self.tree.exports[run.PARENT_EXPORT]["path"])

    def test_oracle_stage_cannot_register_parent_or_targets(self):
        guard = run.stage_guard("oracle_e", self.tree.manifest)
        for name in (run.PARENT_EXPORT, run.TARGET_EXPORT):
            with self.assertRaises(PilotError):
                guard.allow("sneaky", self.tree.exports[name]["path"])
        guard.allow("ok", self.tree.exports[run.ORACLE_EXPORT]["path"])

    def test_retrieve_stage_registers_no_export_at_all(self):
        guard = run.stage_guard("retrieve", self.tree.manifest)
        for name in run.EXPORTS:
            with self.assertRaises(PilotError):
                guard.allow("sneaky", self.tree.exports[name]["path"])

    def test_unregistered_path_cannot_be_read(self):
        guard = run.stage_guard("parent", self.tree.manifest)
        with self.assertRaises(PilotError):
            guard.read_small(self.tree.exports[run.PARENT_EXPORT]["path"])
        guard.allow("ok", self.tree.exports[run.PARENT_EXPORT]["path"])
        raw, _ = guard.read_small(self.tree.exports[run.PARENT_EXPORT]["path"])
        self.assertEqual(len(run.jsonl_rows(raw)), 25)

    def test_check_inputs_per_stage_reports_boundary(self):
        manifest, activation = run.load_activation(self.tree.manifest_path,
                                                   self.tree.identity["sha256"])
        report = run.check_inputs(manifest, activation, "parent")
        self.assertEqual(report["parent_rows"], 25)
        self.assertEqual(report["expected_units"], {"parent": 175, "oracle_e": 25, "retrieve": 150})
        self.assertEqual(report["historical_global_lineage"], "UNESTABLISHED")
        self.assertEqual(report["experiment_state"], "FROZEN_NOT_RUN")
        self.assertEqual(report["canonical_stage0"], "INCOMPLETE_GATES_UNCHANGED")
        self.assertEqual(report["stage_tag"], "v1")
        self.assertFalse(report["scoring_targets_opened"])
        self.assertEqual(report["prompt_preflight"]["historical_runner_sha256"],
                         self.tree.historical_runner)
        self.assertTrue(report["prompt_preflight"]["runner_revision_changed"])
        self.assertEqual((report["files_written"], report["pilot_searches"],
                          report["qwen_generations"]), (0, 0, 0))
        self.assertFalse(report["weights_loaded"] or report["index_loaded"])
        self.assertEqual(sorted(report["forbidden_inputs"]),
                         sorted((run.ORACLE_EXPORT, run.TARGET_EXPORT)))
        e_report = run.check_inputs(manifest, activation, "oracle_e")
        self.assertEqual(e_report["oracle_rows"], 25)
        self.assertNotIn("parent_rows", e_report)

# ---------------------------------------------------- immutable stage output

class StageRootTests(TreeCase):
    def root(self):
        manifest, activation = run.load_activation(self.tree.manifest_path,
                                                   self.tree.identity["sha256"])
        return run.StageRoot("parent", manifest, activation, "v1"), activation

    def record(self, activation, qid, arm="A", pass_name="query_A"):
        row = self.tree.query_record(qid, arm)
        row["pass"] = pass_name
        row["activation_sha256"] = activation["sha256"]
        return row

    def test_create_is_exclusive_and_never_overwrites(self):
        root, activation = self.root()
        root.open_or_resume({"synthetic": True})
        row = self.record(activation, QIDS[0])
        root.create("query_A", QIDS[0], row)
        with self.assertRaises(FileExistsError):
            root.create("query_A", QIDS[0], row)
        changed = dict(row, raw_output="different")
        with self.assertRaises(FileExistsError):
            root.create("query_A", QIDS[0], changed)
        # The original bytes survive the attempted rewrite.
        found, _ = root.existing("query_A", QIDS[0],
                                 lambda r: run.validate_generation_record(r, "A", "query_A"))
        self.assertEqual(found["raw_output"], "synthetic query")

    def test_resume_creates_only_missing_records(self):
        root, activation = self.root()
        inputs = {"synthetic": True}
        self.assertFalse(root.open_or_resume(inputs))
        for qid in QIDS[:3]:
            root.create("query_A", qid, self.record(activation, qid))
        root2 = self.root()[0]
        self.assertTrue(root2.open_or_resume(inputs))
        check = lambda r: run.validate_generation_record(r, "A", "query_A")
        present = [q for q in QIDS if root2.existing("query_A", q, check) is not None]
        self.assertEqual(present, QIDS[:3])

    def test_resume_configuration_drift_refused(self):
        root, _ = self.root()
        root.open_or_resume({"synthetic": True})
        root2 = self.root()[0]
        with self.assertRaises(PilotError):
            root2.open_or_resume({"synthetic": False})

    def test_rerun_after_seal_refused(self):
        root, activation = self.root()
        root.open_or_resume({"synthetic": True})
        write_json(root.seal_path, {"status": "SEALED_PARENT_STATES_AND_QUERIES"})
        root2 = self.root()[0]
        with self.assertRaises(PilotError):
            root2.open_or_resume({"synthetic": True})

    def test_existing_record_with_foreign_activation_refused(self):
        root, activation = self.root()
        root.open_or_resume({"synthetic": True})
        row = self.record(activation, QIDS[0])
        row["activation_sha256"] = "b" * 64
        write_json(root.record_path("query_A", QIDS[0]), row)
        with self.assertRaises(PilotError):
            root.existing("query_A", QIDS[0],
                          lambda r: run.validate_generation_record(r, "A", "query_A"))

    def test_unknown_qid_or_pass_refused(self):
        root, _ = self.root()
        with self.assertRaises(PilotError):
            root.record_path("query_A", "not-a-real-qid")
        with self.assertRaises(PilotError):
            root.record_path("../escape_A", QIDS[0])

    def test_record_path_accepts_every_parent_pass(self):
        """Job 82277 stopped on RECORD_PASS_NAME at query_P: the validator's
        character class omitted the P control arm that Amendment 2 adds. Every
        pass the parent stage actually plans must be accepted."""
        root, _ = self.root()
        for pass_name in run.PARENT_PASSES:
            path = root.record_path(pass_name, QIDS[0])
            self.assertEqual(path.name, pass_name + "__" + QIDS[0] + ".json")
            self.assertEqual(path.parent, root.records)

    def test_record_path_accepts_every_frozen_arm(self):
        """The validator must admit exactly the frozen arms, including P, for
        every pass prefix the three stages construct from ARMS."""
        root, _ = self.root()
        self.assertEqual(list(ARMS), ["A", "P", "B", "C", "D", "E"])
        for arm in ARMS:
            for prefix in ("query_", "state_", "prediction_"):
                name = prefix + arm
                path = root.record_path(name, QIDS[0])
                self.assertEqual(path.name, name + "__" + QIDS[0] + ".json")
                self.assertEqual(path.parent, root.records)

    def test_record_path_accepts_p_arm_by_name(self):
        """Explicit non-parametrised guard for the exact names that failed."""
        root, _ = self.root()
        for name in ("query_P", "prediction_P", "state_P"):
            self.assertEqual(root.record_path(name, QIDS[0]).name,
                             name + "__" + QIDS[0] + ".json")

    def test_record_path_rejects_arms_outside_the_frozen_set(self):
        """Widening the class to admit P must not admit anything further."""
        root, _ = self.root()
        for bad in ("query_F", "query_Q", "query_Z", "prediction_F", "state_G"):
            with self.assertRaises(PilotError):
                root.record_path(bad, QIDS[0])

    def test_record_path_rejects_malformed_names_and_traversal(self):
        """Traversal, separators, case and arity guards all still hold."""
        root, _ = self.root()
        for bad in ("", "_A", "A", "query_", "query_AP", "query_PA", "query_a",
                    "query_p", "Query_A", "query_A ", " query_A", "query-A",
                    "query_A.json", "query_A__x", "1query_A",
                    "../escape_P", "../../etc/passwd_P", "..__P", "/abs_P",
                    "sub/dir_P", "query_P/../../escape_A", "query_P\n",
                    "query_P\x00", "query_É"):
            with self.assertRaises(PilotError):
                root.record_path(bad, QIDS[0])

    def test_record_path_containment_for_every_accepted_name(self):
        """No accepted pass name may resolve outside the stage records dir."""
        root, _ = self.root()
        names = list(run.PARENT_PASSES) + ["query_E"] + [
            "prediction_" + arm for arm in ARMS]
        for name in names:
            path = root.record_path(name, QIDS[0])
            self.assertTrue(path.resolve().is_relative_to(root.records.resolve()))

    def test_archive_must_be_fresh(self):
        root, _ = self.root()
        root.archive.mkdir(parents=True)
        with self.assertRaises(PilotError):
            root.open_or_resume({"synthetic": True})

    def test_write_new_fsyncs_file_and_parent_directory(self):
        calls = []
        real = os.fsync
        try:
            os.fsync = lambda fd: (calls.append(fd), real(fd))[1]
            target = Path(self.tmp) / "fsync_probe" / "artifact.json"
            target.parent.mkdir()
            run.write_json(target, {"a": 1})
        finally:
            os.fsync = real
        # One fsync for the written file, one for its directory.
        self.assertGreaterEqual(len(calls), 2)

    def test_new_dir_fsyncs_directory_and_parent(self):
        calls = []
        real = os.fsync
        try:
            os.fsync = lambda fd: (calls.append(fd), real(fd))[1]
            run.new_dir(Path(self.tmp) / "fresh_dir")
        finally:
            os.fsync = real
        self.assertGreaterEqual(len(calls), 2)
        with self.assertRaises(PilotError):
            run.new_dir(Path(self.tmp) / "fresh_dir")

    def test_symlinked_output_refused(self):
        link = Path(self.tmp) / "linked"
        link.symlink_to(self.tree.out)
        with self.assertRaises(PilotError):
            run.no_symlink(link / "parent_v1")

# ------------------------------------------------------------ record binding

class RecordTests(TreeCase):
    def good_query(self, arm="A"):
        return self.tree.query_record(QIDS[0], arm)

    def test_query_record_round_trip(self):
        row = self.good_query()
        self.assertIs(run.validate_generation_record(row, "A", "query_A"), row)

    def test_chain_of_thought_field_forbidden(self):
        for field in ("chain_of_thought", "thinking", "reasoning", "reasoning_content"):
            row = dict(self.good_query())
            row[field] = "hidden"
            with self.assertRaises(PilotError):
                run.validate_generation_record(row, "A", "query_A")

    def test_prompt_constant_drift_detected_in_record(self):
        row = self.good_query()
        row["prompt_constants"] = dict(row["prompt_constants"], system="0" * 64)
        with self.assertRaises(PilotError):
            run.validate_generation_record(row, "A", "query_A")

    def test_live_prompt_constant_drift_stops_everything(self):
        original = core.C_INSTRUCTION
        try:
            core.C_INSTRUCTION = original + " "
            with self.assertRaises(PilotError):
                assert_prompt_hashes()
        finally:
            core.C_INSTRUCTION = original
        self.assertEqual(assert_prompt_hashes(), core.PROMPT_HASHES)

    def test_context_budget_and_arm_binding(self):
        row = dict(self.good_query(), input_tokens=40833)
        with self.assertRaises(PilotError):
            run.validate_generation_record(row, "A", "query_A")
        with self.assertRaises(PilotError):
            run.validate_generation_record(self.good_query(), "P", "query_P")

    def test_a_cannot_fall_back_to_sealed_a(self):
        row = self.good_query()
        row["result"] = dict(row["result"], fallback="SEALED_A")
        with self.assertRaises(PilotError):
            run.validate_generation_record(row, "A", "query_A")

    def test_query_over_word_or_encoder_cap_refused(self):
        row = self.good_query()
        row["result"] = dict(row["result"], query=" ".join(["w"] * 33))
        with self.assertRaises(PilotError):
            run.validate_generation_record(row, "A", "query_A")
        row = self.good_query()
        row["result"] = dict(row["result"], encoder_tokens=129)
        with self.assertRaises(PilotError):
            run.validate_generation_record(row, "A", "query_A")

    def test_incomplete_generation_status_refused(self):
        row = dict(self.good_query(), generation_status="TIMEOUT")
        with self.assertRaises(PilotError):
            run.validate_generation_record(row, "A", "query_A")

    def test_prediction_record_binding_and_ranking_drift(self):
        rec = self.tree.prediction_record(QIDS[0], "D", "a" * 64, "b" * 64, CHILD_COUNTS[QIDS[0]])
        run.validate_prediction_record(rec, "D")
        bad = copy.deepcopy(rec)
        bad["prediction"]["ranked_titles"].reverse()
        with self.assertRaises(PilotError):
            run.validate_prediction_record(bad, "D")
        bad = copy.deepcopy(rec)
        bad["prediction"]["candidates"] = bad["prediction"]["candidates"][:99]
        with self.assertRaises(PilotError):
            run.validate_prediction_record(bad, "D")
        bad = copy.deepcopy(rec)
        bad["candidate_provenance"][0]["global_row"] += 1000
        with self.assertRaises(PilotError):
            run.validate_prediction_record(bad, "D")
        bad = copy.deepcopy(rec)
        bad["query_record"]["arm"] = "A"
        with self.assertRaises(PilotError):
            run.validate_prediction_record(bad, "D")

    def test_query_and_prompt_provenance_is_bound(self):
        rec = self.tree.prediction_record(QIDS[0], "A", "a" * 64, "b" * 64)
        query_row = self.tree.query_record(QIDS[0], "A")
        self.assertEqual(rec["query_record"]["record_sha256"], sha256(canonical(query_row)))
        self.assertEqual(rec["query_record"]["prompt_sha256"], query_row["prompt_sha256"])
        self.assertEqual(rec["prediction"]["query"], query_row["result"]["query"])

# --------------------------------------------------- retrieval-side failures

class RetrievalSafetyTests(unittest.TestCase):
    def reader(self, rows, total=None, size=None, offsets=None):
        import numpy as np
        tmp = Path(tempfile.mkdtemp(prefix="bridge_meta_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        raw = b"".join((canonical(r) + "\n").encode("utf-8") for r in rows)
        (tmp / "meta.jsonl").write_bytes(raw)
        if offsets is None:
            offsets, cursor = [], 0
            for r in rows:
                offsets.append(cursor)
                cursor += len((canonical(r) + "\n").encode("utf-8"))
        np.save(tmp / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
        return run.MetadataReader(str(tmp / "meta.jsonl"), str(tmp / "offsets.npy"),
                                  size if size is not None else len(raw),
                                  total if total is not None else len(rows))

    def test_bounded_lookup_returns_metadata(self):
        r = self.reader([{"title": "A", "text": "x"}, {"title": "B", "text": "y"}])
        self.assertEqual(r.lookup(1)["title"], "B")
        self.assertEqual(r.provenance[1]["byte_offset"], len(canonical({"title": "A", "text": "x"})) + 1)
        r.close()

    def test_offsets_shape_and_endpoint_failures(self):
        with self.assertRaises(PilotError):
            self.reader([{"title": "A", "text": "x"}], total=2)
        with self.assertRaises(PilotError):
            self.reader([{"title": "A", "text": "x"}], offsets=[7])

    def test_offset_not_at_line_start(self):
        r = self.reader([{"title": "A", "text": "x"}, {"title": "B", "text": "y"}])
        r.offsets = r.offsets.copy()
        r.offsets[1] += 1
        with self.assertRaises(PilotError):
            r.lookup(1)
        r.close()

    def test_metadata_schema_failures_stop_the_stage(self):
        for rows in ([{"title": "A", "text": "x", "extra": 1}],
                     [{"title": "   ", "text": "x"}],
                     [{"title": "A"}]):
            r = self.reader(rows)
            with self.assertRaises((PilotError, ValueError)):
                r.lookup(0)
            r.close()

    def test_out_of_range_row_refused(self):
        r = self.reader([{"title": "A", "text": "x"}])
        for row in (-1, 1, 1.0, True):
            with self.assertRaises(PilotError):
                r.lookup(row)
        r.close()

    def test_every_id_validated_before_any_metadata_dereference(self):
        r = self.reader([{"title": "A", "text": "x"}, {"title": "B", "text": "y"}])
        touched = []
        original = r.lookup
        r.lookup = lambda i: (touched.append(i), original(i))[1]
        bad = ([0, -1], [0, 5], [0, 1.5], [0, True], [0, 0])
        for ids in bad:
            with self.assertRaises(PilotError):
                search_to_metadata(ids, [1.0, 0.5], r.lookup, ntotal=2, budget=2)
        with self.assertRaises(PilotError):
            search_to_metadata([0, 1], [1.0, math.nan], r.lookup, ntotal=2, budget=2)
        with self.assertRaises(PilotError):
            search_to_metadata([0, 1], [0.5, 1.0], r.lookup, ntotal=2, budget=2)
        self.assertEqual(touched, [])
        result = search_to_metadata([0, 1], [1.0, 0.5], r.lookup, ntotal=2, budget=2)
        self.assertEqual(touched, [0, 1])
        self.assertEqual(len(result["ranked_titles"]), 2)
        r.close()

    def test_duplicate_normalized_titles_collapse_and_short_list_kept(self):
        candidates = [{"global_row": 3, "score": 1.0, "title": " Same Title "},
                      {"global_row": 2, "score": 1.0, "title": "same_title"},
                      {"global_row": 9, "score": 0.5, "title": "Other"}]
        ranked = aggregate_candidates(candidates)
        self.assertEqual([r["normalized_title"] for r in ranked], ["same title", "other"])
        self.assertEqual(ranked[0]["best_global_row"], 2)
        self.assertLess(len(ranked), 10)

    def test_no_python_loop_over_the_corpus(self):
        source = (HERE / "bridge_pilot_run.py").read_text()
        self.assertNotIn("range(N_VECTORS)", source)
        self.assertNotIn("for row in range(23963971)", source)
        self.assertIn("index.search(", source)
        # The only search call sits outside any per-qid loop over the universe.
        self.assertEqual(source.count("index.search("), 1)
        self.assertEqual(source.count("faiss.read_index("), 1)
        self.assertEqual(source.count("SentenceTransformer("), 1)
        self.assertEqual(source.count("AutoModelForCausalLM.from_pretrained"), 1)

    def test_no_offset_autobuild_or_write_path(self):
        source = (HERE / "bridge_pilot_run.py").read_text()
        for forbidden in ("np.save", "_load_or_build_offsets", "from rag", "import rag",
                          "Retriever", "shutil", "os.remove", "rmtree"):
            self.assertNotIn(forbidden, source)
        self.assertIn('mmap_mode="r"', source)
        self.assertIn("allow_pickle=False", source)

# ------------------------------------------------------ frozen state parsing

class StateTests(unittest.TestCase):
    def parent(self, text=None):
        page = synthetic_page(text) if text else synthetic_page()
        return {"qid": QIDS[0], "question_ur": "س", "parent_source_instance_id": "p",
                "parent_title": "Synthetic Parent",
                "parent_normalized_title": "synthetic parent", "page": page,
                "chunks": local_chunks(page)}

    def test_malformed_and_duplicate_json_keys_become_format_failures(self):
        for raw in ('{"facts":[],"facts":[]}', '{"facts":[],"extra":0}', '{"facts":NaN}',
                    '{"facts":[{"quote":"a","quote":"b"}]}', '{"facts":{}}', 'not json', '[]'):
            with self.subTest(raw=raw):
                self.assertTrue(parse_state(raw, self.parent(), "D")["format_failure"])

    def test_invalid_and_unsupported_items_do_not_consume_allowance(self):
        row = self.parent(" ".join("w%d" % i for i in range(20)))
        cid = row["chunks"][0]["chunk_id"]
        items = ([{"chunk_id": cid, "quote": "text not in the page"}]
                 + [{"chunk_id": cid, "quote": "w%d" % i} for i in range(8)]
                 + [{"chunk_id": cid, "quote": "w0"}])
        state = parse_state(canonical({"facts": items}), row, "D")
        self.assertEqual((len(state["items"]), state["invalid_items"],
                          state["duplicate_items"], state["excess_items"]), (6, 1, 1, 2))

    def test_c_item_requires_support_inside_its_quote(self):
        row = self.parent()
        cid = row["chunks"][0]["chunk_id"]
        state = parse_state(canonical({"entities": [
            {"chunk_id": cid, "quote": "London is a city.", "text": "Paris"},
            {"chunk_id": cid, "quote": "London is a city.", "text": "London"}]}), row, "C")
        self.assertEqual(len(state["items"]), 1)
        self.assertEqual(state["invalid_items"], 1)

    def test_empty_state_falls_back_to_parent_title_only(self):
        row = self.parent()
        for arm, raw in (("C", '{"entities":[]}'), ("D", "broken")):
            self.assertEqual(core.parent_query_messages(row, arm, parse_state(raw, row, arm)),
                             core.parent_query_messages(row, "P"))

    def test_token_cap_and_a_fallback_chain(self):
        def tok(text, **kw):
            assert kw == {"add_special_tokens": True, "truncation": False}
            return {"input_ids": list(range(2 + 5 * len(text.split())))}
        capped = cap_query(" ".join(["word"] * 40), "سوال", "A", tok)
        self.assertEqual(len(capped["query"].split()), 25)
        self.assertEqual(capped["word_cap_removed"], 8)
        self.assertEqual(cap_query("   ", "سوال", "A", tok)["fallback"], "URDU_QUESTION")
        self.assertEqual(cap_query("   ", "سوال", "D", tok, "sealed a")["fallback"], "SEALED_A")
        with self.assertRaises(PilotError):
            cap_query("   ", "سوال", "D", tok)
        def crash(text, **kw):
            raise RuntimeError("infrastructure")
        with self.assertRaises(RuntimeError):
            cap_query("word", "سوال", "A", crash)

# ------------------------------------------------------- frozen mathematics

class MathTests(unittest.TestCase):
    def predictions(self, hits):
        return [scoring_view(q, a, hits(q, a)) for q in QIDS for a in ARMS]

    def test_cohort_multiplicities_are_exactly_25_and_36(self):
        targets = synthetic_targets()
        validate_targets(targets)
        self.assertEqual(len(targets), 25)
        self.assertEqual(sum(len(t["children"]) for t in targets), 36)
        counts = sorted(CHILD_COUNTS.values())
        self.assertEqual((counts.count(1), counts.count(2), counts.count(3), counts.count(4)),
                         (17, 6, 1, 1))

    def test_all_six_arms_and_150_completeness(self):
        rows = self.predictions(lambda q, a: 0)
        self.assertEqual(len(rows), 150)
        validate_predictions(rows)
        with self.assertRaises(PilotError):
            validate_predictions(rows[:-1])
        duplicated = rows[:-1] + [copy.deepcopy(rows[0])]
        with self.assertRaises(PilotError):
            validate_predictions(duplicated)
        unknown = copy.deepcopy(rows)
        unknown[0] = dict(unknown[0], qid="0" * 20)
        with self.assertRaises(PilotError):
            validate_predictions(unknown)
        malformed = copy.deepcopy(rows)
        malformed[0]["unexpected"] = 1
        with self.assertRaises(PilotError):
            validate_predictions(malformed)

    def test_macro_and_micro_differ_with_fractional_sibling_hits(self):
        # D recovers exactly one child per qid: fractional credit for multi-child qids.
        result = summarize(self.predictions(lambda q, a: 1 if a == "D" else 0), synthetic_targets())
        macro = result["arms"]["D"]["10"]["qid_macro_recall"]
        micro = result["arms"]["D"]["10"]["pair_micro_recall"]
        self.assertAlmostEqual(macro, (17 + 6 / 2 + 1 / 3 + 1 / 4) / 25)
        self.assertAlmostEqual(micro, 25 / 36)
        self.assertNotAlmostEqual(macro, micro)
        self.assertAlmostEqual(result["arms"]["D"]["10"]["any_verified_child_coverage"], 1.0)
        self.assertAlmostEqual(result["arms"]["D"]["10"]["all_verified_child_coverage"], 17 / 25)

    def test_full_recovery_gives_macro_one_and_all_child_coverage_one(self):
        result = summarize(self.predictions(lambda q, a: CHILD_COUNTS[q] if a == "D" else 0),
                           synthetic_targets())
        self.assertEqual(result["arms"]["D"]["10"]["qid_macro_recall"], 1.0)
        self.assertEqual(result["arms"]["D"]["10"]["all_verified_child_coverage"], 1.0)
        self.assertEqual(result["comparisons"]["D-A"]["effect_pp"], 100.0)

    def test_integer_dp_signflip_matches_brute_force(self):
        for weights in ([], [0] * 25, [12] * 6, [12] * 7 + [-12], [3, -4, 6, 12, 0],
                        [-3, -4, 6], [6, 6, 4, 3, -12], [12, -6, 4, 3, 0, 0]):
            nonzero = [abs(w) for w in weights if w]
            values = [sum(s * w for s, w in zip(signs, nonzero))
                      for signs in itertools.product((-1, 1), repeat=len(nonzero))]
            expected = sum(abs(v) >= abs(sum(weights)) for v in values) / len(values)
            with self.subTest(weights=weights):
                self.assertEqual(signflip(weights)["p_two_sided"], expected)
        self.assertEqual(signflip([0] * 25)["p_two_sided"], 1.0)
        with self.assertRaises(PilotError):
            signflip([1.0, 2.0])

    def test_weights_are_integer_derived_not_rounded_floats(self):
        source = (HERE / "bridge_pilot_core.py").read_text()
        self.assertIn("(12 // CHILD_COUNTS[q])", source)
        self.assertNotIn("round(", source)
        # 2^25 enumeration must never appear.
        self.assertNotIn("2 ** 25", source)
        self.assertNotIn("product((-1, 1), repeat=25)", source)

    def test_bootstrap_indices_shared_across_comparisons(self):
        import numpy as np
        result = summarize(self.predictions(lambda q, a: CHILD_COUNTS[q] if a == "D" else 0),
                           synthetic_targets())
        boot = result["bootstrap"]
        self.assertEqual((boot["resamples"], boot["seed"], boot["rng"]), (20000, SEED, "PCG64"))
        self.assertTrue(boot["same_indices_across_comparisons"])
        self.assertEqual(boot["percentile_method"], "linear")
        expected = np.random.Generator(np.random.PCG64(SEED)).integers(
            0, 25, size=(20000, 25), dtype=np.int64)
        self.assertEqual(expected.shape, (20000, 25))
        for name in ("D-A", "D-P", "D-B", "D-C"):
            self.assertIn("paired_qid_bootstrap_95ci_pp", result["comparisons"][name])

    def test_fixed_sequence_gate_blocks_followup(self):
        # D identical to A: no effect, p = 1, follow-up stays exploratory.
        result = summarize(self.predictions(lambda q, a: 1 if a in ("D", "A") else 0),
                           synthetic_targets())
        self.assertFalse(result["primary_gate_passed"])
        self.assertEqual(result["comparisons"]["D-A"]["p_two_sided"], 1.0)
        self.assertEqual(result["comparisons"]["D-P"]["inference"],
                         "EXPLORATORY_PRIMARY_GATE_NOT_PASSED")
        self.assertEqual(result["comparisons"]["D-B"]["inference"], "DESCRIPTIVE_COMPARISON")
        self.assertEqual(result["decision"], "INSUFFICIENT_EVIDENCE_FOR_FULL_BRIDGE_METHOD")
        self.assertEqual(result["canonical_stage0"], "INCOMPLETE_GATES_UNCHANGED")

    def test_ten_point_threshold_uses_integer_weights(self):
        # Three single-child qids improved = +12 pp > 10 pp, but p = .25.
        improved = set(QIDS[:3])
        result = summarize(
            self.predictions(lambda q, a: 1 if (a == "D" and q in improved
                                                and CHILD_COUNTS[q] == 1) else 0),
            synthetic_targets())
        primary = result["comparisons"]["D-A"]
        self.assertAlmostEqual(primary["effect_pp"], 12.0)
        self.assertTrue(primary["positive_practical_threshold_met"])
        self.assertEqual(primary["p_two_sided"], 0.25)
        self.assertFalse(result["primary_gate_passed"])

    def test_short_ranked_lists_are_counted(self):
        rows = self.predictions(lambda q, a: 0)
        for row in rows:
            for c in row["candidates"]:
                c["title"] = "Only One Title"
            row["ranked_titles"] = aggregate_candidates(row["candidates"])
        result = summarize(rows, synthetic_targets())
        for arm in ARMS:
            self.assertEqual(result["arms"][arm]["short_ranked_lists"], 25)

# ------------------------------------------------------------ scorer chain

class ScorerTests(TreeCase):
    def run_scorer(self, pred_seal, out_name=None, **overrides):
        out = (self.tree.out / "score_v1") if out_name is None else (self.tree.base / out_name)
        argv = ["bridge_pilot_score.py",
                "--activation", str(self.tree.manifest_path),
                "--activation-sha256", overrides.get("activation_sha256",
                                                     self.tree.identity["sha256"]),
                "--predictions-seal", str(overrides.get("seal_path", pred_seal["path"])),
                "--seal-sha256", overrides.get("seal_sha256", pred_seal["sha256"]),
                "--targets-sha256", overrides.get("targets_sha256",
                                                  self.tree.exports[run.TARGET_EXPORT]["sha256"]),
                "--out", str(out)]
        old, sys.argv = sys.argv, argv
        buffer, out = io.StringIO(), sys.stdout
        try:
            sys.stdout = buffer
            score.main()
        finally:
            sys.argv, sys.stdout = old, out
        return json.loads(buffer.getvalue())

    def test_full_chain_scores_and_seals(self):
        _, _, pred = self.tree.build_full_chain()
        result = self.run_scorer(pred)
        self.assertEqual(result["status"], "SCORING_COMPLETE")
        self.assertEqual((result["accepted_qids"], result["accepted_pairs"]), (25, 36))
        self.assertEqual(result["canonical_stage0"], "INCOMPLETE_GATES_UNCHANGED")
        self.assertEqual(result["historical_global_lineage"], "UNESTABLISHED")
        out = self.tree.out / "score_v1"
        scores = json.loads((out / "scores.json").read_text())
        self.assertEqual(scores["arms"]["D"]["10"]["qid_macro_recall"], 1.0)
        self.assertEqual(scores["provenance"]["historical_global_lineage"], "UNESTABLISHED")
        seal = json.loads((out / "SCORING_SEAL.json").read_text())
        self.assertEqual(seal["status"], "SEALED_SCORES")
        self.assertEqual(seal["activation_sha256"], self.tree.identity["sha256"])

    def test_arbitrary_activation_hash_refused(self):
        _, _, pred = self.tree.build_full_chain()
        with self.assertRaises(PilotError):
            self.run_scorer(pred, activation_sha256="c" * 64)

    def test_tampered_predictions_seal_refused(self):
        _, _, pred = self.tree.build_full_chain()
        seal = json.loads(Path(pred["path"]).read_text())
        seal["predictions"] = 149
        forged = write_json(Path(pred["path"]).parent / "FORGED_SEAL.json", seal)
        with self.assertRaises(PilotError):
            self.run_scorer(pred, seal_path=forged["path"], seal_sha256=forged["sha256"])

    def test_self_consistent_foreign_seal_refused(self):
        # A caller re-seals everything under a manifest hash nobody reviewed.
        _, _, pred = self.tree.build_full_chain()
        seal = json.loads(Path(pred["path"]).read_text())
        seal["activation_sha256"] = "d" * 64
        forged = write_json(Path(pred["path"]).parent / "SELF_SEAL.json", seal)
        with self.assertRaises(PilotError):
            self.run_scorer(pred, seal_path=forged["path"], seal_sha256=forged["sha256"])

    def test_prediction_artifact_hash_drift_refused(self):
        _, _, pred = self.tree.build_full_chain()
        path = self.tree.out / "retrieve_v1" / "predictions.jsonl"
        path.write_bytes(path.read_bytes().replace(b"synthetic query", b"tampered query"))
        with self.assertRaises(PilotError):
            self.run_scorer(pred)

    def test_upstream_seal_tampering_refused(self):
        _, _, pred = self.tree.build_full_chain()
        seal_path = self.tree.out / "parent_v1" / "STAGE_SEAL.json"
        seal = json.loads(seal_path.read_text())
        seal["queries"] = 124
        seal_path.write_text(canonical(seal) + "\n")
        with self.assertRaises(PilotError):
            self.run_scorer(pred)

    def test_partial_predictions_refused_before_targets_are_opened(self):
        rows = [self.tree.prediction_record(q, a, "x" * 64, "y" * 64)
                for q in QIDS for a in ARMS][:-1]
        parent_id, e_id, _ = self.tree.build_full_chain()
        for r in rows:
            r["parent_seal_sha256"], r["oracle_e_seal_sha256"] = parent_id["sha256"], e_id["sha256"]
        pred, _ = self.tree.seal_stage(
            "retrieve", "v1b",
            {"prediction_records.jsonl": rows, "predictions.jsonl": [r["prediction"] for r in rows]},
            {"predictions": 150, "targets_sha256": self.tree.exports[run.TARGET_EXPORT]["sha256"],
             "parent_seal_sha256": parent_id["sha256"], "oracle_e_seal_sha256": e_id["sha256"],
             "preparation_seal_sha256": self.tree.prep_seal["sha256"],
             "preflight_seal_sha256": self.tree.pre_seal["sha256"], "search_budget": BUDGET,
             "targets_opened": False, "scored_outcomes": 0})
        # Remove the targets entirely: the run must still fail on completeness,
        # proving the targets are opened only after prediction validation.
        Path(self.tree.exports[run.TARGET_EXPORT]["path"]).unlink()
        with self.assertRaises(PilotError) as ctx:
            self.run_scorer(pred)
        self.assertIn("PREDICTION_RECORD_COUNT", str(ctx.exception))

    def test_targets_are_needed_only_at_the_end(self):
        _, _, pred = self.tree.build_full_chain()
        Path(self.tree.exports[run.TARGET_EXPORT]["path"]).unlink()
        with self.assertRaises((PilotError, OSError)) as ctx:
            self.run_scorer(pred)
        self.assertNotIn("PREDICTION_RECORD_COUNT", str(ctx.exception))

    def test_existing_output_directory_refused(self):
        _, _, pred = self.tree.build_full_chain()
        (self.tree.out / "score_v1").mkdir(parents=True)
        with self.assertRaises(PilotError):
            self.run_scorer(pred)

    def test_unreviewed_score_output_refused(self):
        _, _, pred = self.tree.build_full_chain()
        with self.assertRaises(PilotError) as ctx:
            self.run_scorer(pred, out_name="somewhere_else")
        self.assertIn("SCORE_OUTPUT_NOT_REVIEWED", str(ctx.exception))

    def test_targets_multiplicity_tampering_refused(self):
        rows = synthetic_targets()
        rows[0]["children"] = rows[0]["children"] * 2
        ident = write_jsonl(Path(self.tree.exports[run.TARGET_EXPORT]["path"]), rows)
        m = copy.deepcopy(self.tree.manifest)
        m["exports"][run.TARGET_EXPORT] = dict(ident)
        m["preparation"]["artifacts"][run.TARGET_EXPORT] = {"sha256": ident["sha256"],
                                                            "bytes": ident["bytes"]}
        self.tree.rewrite(m)
        _, _, pred = self.tree.build_full_chain()
        with self.assertRaises(PilotError):
            self.run_scorer(pred, targets_sha256=ident["sha256"])

# ------------------------------------------------------------- job scripts

class JobScriptTests(unittest.TestCase):
    SCRIPTS = ("bridge_pilot_parent.sbatch", "bridge_pilot_oracle_e.sbatch",
               "bridge_pilot_retrieve.sbatch", "bridge_pilot_score.sbatch")

    def test_scripts_require_the_reviewed_activation_identity(self):
        for name in self.SCRIPTS:
            text = (HERE / name).read_text()
            with self.subTest(name=name):
                self.assertIn('test -n "${BRIDGE_ACTIVATION:-}"', text)
                self.assertIn('test -n "${BRIDGE_ACTIVATION_SHA256:-}"', text)
                self.assertIn("^[0-9a-f]{64}$", text)
                self.assertIn("set -euo pipefail", text)
                self.assertIn("/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B", text)
                self.assertIn("cd /mnt/home/user41/URBench", text)
                self.assertIn("/mnt/home/user41/URBench/logs/", text)
                # No default activation identity and no package installation.
                self.assertNotIn("BRIDGE_ACTIVATION:-/", text)
                self.assertNotIn("pip install", text)
                self.assertNotIn("conda install", text)

    def test_gpu_and_memory_requests(self):
        for name in ("bridge_pilot_parent.sbatch", "bridge_pilot_oracle_e.sbatch",
                     "bridge_pilot_retrieve.sbatch"):
            text = (HERE / name).read_text()
            self.assertIn("#SBATCH --gres=gpu:1", text)
            self.assertIn("#SBATCH --partition=q_intel_share_L20", text)
        retrieve = (HERE / "bridge_pilot_retrieve.sbatch").read_text()
        self.assertIn("#SBATCH --mem=80G", retrieve)
        self.assertIn("#SBATCH --cpus-per-task=8", retrieve)
        scorer = (HERE / "bridge_pilot_score.sbatch").read_text()
        self.assertNotIn("--gres=gpu", scorer)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', scorer)

    def test_runner_refuses_login_node_and_missing_job(self):
        source = (HERE / "bridge_pilot_run.py").read_text()
        self.assertIn('need(os.environ.get("SLURM_JOB_ID"), "COMPUTE_JOB_REQUIRED")', source)
        self.assertIn("LOGIN_NODE_REFUSED", source)
        self.assertIn("OFFLINE_REQUIRED", source)

if __name__ == "__main__":
    unittest.main(verbosity=2)
