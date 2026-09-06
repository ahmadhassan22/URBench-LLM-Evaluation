#!/usr/bin/env python3
"""Bounded N25 bridge-pilot stage runner, v0.1. Does NOT activate the pilot.

Four separately launched stages -- ``parent``, ``oracle_e``, ``retrieve`` and
(in ``bridge_pilot_score.py``) ``score``. Every stage authenticates one
prospective activation manifest from an explicit expected SHA-256 before it
opens anything else, and every file read passes through a deny-by-default path
guard built from that manifest. The guard also carries a per-stage forbidden
set, so registering the oracle-E export inside the A/P/B/C/D worker, or the
scoring targets inside any generation or retrieval stage, raises at
registration time rather than at read time.

Frozen pure logic (prompt constants, serialization, chat template call, state
parser, query cap/fallback, search validation, title aggregation, statistics)
lives in ``bridge_pilot_core.py`` and is imported, never re-implemented here.
This module adds only model I/O, retrieval I/O, immutable record keeping,
resume, sealing and archiving.

Nothing here generates, retrieves or scores until an activation manifest
reviewed under the protocol exists; there is no default manifest and no bypass.
``historical_global_lineage`` is carried through as UNESTABLISHED and is never
upgraded by the runtime.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import socket
import stat
import time

from bridge_pilot_core import (ARMS, BUDGET, CHILD_COUNTS, DIM, N_VECTORS, SEED,
                               aggregate_candidates, assert_prompt_hashes,
                               canonical, cap_query, exact_keys, need, norm,
                               oracle_query_messages, parent_query_messages, parse_state,
                               render_prompt, search_to_metadata, sha256, state_messages,
                               strict_json, validate_parent, validate_predictions)

VERSION = "0.1"
ACTIVATION_SCHEMA = "urbench.bridge_n25.activation.v1"
STATE_ARMS = ("C", "D")
QID_PATTERN = re.compile(r"[0-9a-f]{20}")
HEX256 = re.compile(r"[0-9a-f]{64}")
LINEAGE = "UNESTABLISHED"
# Historical state carried by the three pre-activation seals. Kept unchanged so
# the already-sealed prompt-preflight script, which imports this name, keeps its
# exact recorded semantics.
EXPERIMENT_STATE = "NOT_FROZEN_NOT_RUN"
# State the reviewed activation manifest must declare, and that stage artifacts
# stamp: frozen by Amendment 2, no outcome produced yet.
ACTIVATED_EXPERIMENT_STATE = "FROZEN_NOT_RUN"

# Ordered passes. Every A query is sealed on disk before any arm that may fall
# back to it runs, and each state precedes its own query pass.
PARENT_PASSES = ("query_A", "state_C", "state_D", "query_P", "query_B", "query_C", "query_D")
STAGE_UNITS = {"parent": 25 * len(PARENT_PASSES), "oracle_e": 25, "retrieve": 150}
STAGE_SCHEMAS = {
    "parent": ("urbench.bridge_n25.parent_seal.v1", "SEALED_PARENT_STATES_AND_QUERIES"),
    "oracle_e": ("urbench.bridge_n25.oracle_e_seal.v1", "SEALED_ORACLE_E_QUERIES"),
    "retrieve": ("urbench.bridge_n25.predictions_seal.v1", "SEALED_PREDICTIONS"),
}
PREPARATION_STATUS = "SEALED_PREPARATION_NOT_EXPERIMENT_FREEZE"
PREFLIGHT_STATUS = "SEALED_OUTCOME_FREE_PREFLIGHT_NOT_ACTIVATION"
PROMPT_PREFLIGHT_STATUS = "SEALED_OUTCOME_FREE_PROMPT_PREFLIGHT_NOT_ACTIVATION"
PROMPT_PREFLIGHT_RECORDS = 150
PROMPT_PREFLIGHT_CATEGORIES = 6
QWEN_LOAD_CHOICES = ("bnb_nf4_double_bfloat16", "plain_bfloat16")
PARENT_EXPORT = "runtime_parent_only.jsonl"
ORACLE_EXPORT = "runtime_oracle_e.jsonl"
TARGET_EXPORT = "scoring_targets.jsonl"
EXPORTS = (PARENT_EXPORT, ORACLE_EXPORT, TARGET_EXPORT)
ASSET_KEYS = ("index", "metadata", "offsets")
CODE_FILES = ("bridge_pilot_core.py", "bridge_pilot_core_test.py",
              "bridge_pilot_run.py", "bridge_pilot_run_test.py",
              "bridge_pilot_score.py",
              "bridge_pilot_prepare.py", "bridge_pilot_prepare.sbatch",
              "bridge_pilot_preflight.py", "bridge_pilot_preflight.sbatch",
              "bridge_pilot_prompt_preflight.py", "bridge_pilot_prompt_preflight_test.py",
              "bridge_pilot_prompt_preflight.sbatch",
              "efbpt_assisted_qid_audit.py",
              "bridge_pilot_parent.sbatch", "bridge_pilot_oracle_e.sbatch",
              "bridge_pilot_retrieve.sbatch", "bridge_pilot_score.sbatch")
# Stages whose roots this activation will create. No declared input may lie
# inside any of them.
STAGE_ROOT_NAMES = ("parent", "oracle_e", "retrieve", "score")
# Inputs each stage must never be able to register, by manifest export name.
STAGE_FORBIDDEN = {"parent": (ORACLE_EXPORT, TARGET_EXPORT),
                   "oracle_e": (PARENT_EXPORT, TARGET_EXPORT),
                   "retrieve": (PARENT_EXPORT, ORACLE_EXPORT, TARGET_EXPORT)}

# ---------------------------------------------------------------- filesystem

def no_symlink(path):
    p = Path(path).absolute()
    for part in (p, *p.parents):
        need(not part.is_symlink(), "SYMLINK_REFUSED: " + str(part))
    return p

def stamp(path):
    s = no_symlink(path).stat()
    need(stat.S_ISREG(s.st_mode), "REGULAR_FILE_REQUIRED: " + str(path))
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

def read_small(path, expected=None, limit=64 * 1024**2):
    p = no_symlink(path)
    before = stamp(p)
    need(before[2] <= limit, "SMALL_FILE_TOO_LARGE: " + str(p))
    raw = p.read_bytes()
    need(before == stamp(p) and len(raw) == before[2], "SMALL_FILE_CHANGED: " + str(p))
    need(expected is None or sha256(raw) == expected, "SMALL_FILE_HASH: " + str(p))
    return raw, {"path": str(p), "sha256": sha256(raw), "bytes": len(raw)}

def fingerprint(path, expected_hash=None, expected_bytes=None):
    """Streaming hash for large pinned assets. Never a corpus scan."""
    p = no_symlink(path)
    before = stamp(p)
    need(expected_bytes is None or before[2] == expected_bytes, "ASSET_SIZE_DRIFT: " + str(p))
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    need(stamp(p) == before, "ASSET_CHANGED_DURING_HASH: " + str(p))
    need(expected_hash is None or h.hexdigest() == expected_hash, "ASSET_HASH_DRIFT: " + str(p))
    return {"path": str(p), "sha256": h.hexdigest(), "bytes": before[2]}

def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def new_dir(path):
    p = no_symlink(path)
    need(p.parent.is_dir() and not p.exists(), "FRESH_DIRECTORY_WITH_EXISTING_PARENT_REQUIRED: " + str(p))
    p.mkdir(mode=0o700)
    sync_dir(p)
    sync_dir(p.parent)
    return p

def write_new(path, raw):
    """Exclusive create, flush, fsync file, fsync parent, then verify readback."""
    p = no_symlink(path)
    with p.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    sync_dir(p.parent)
    check, _ = read_small(p)
    need(check == raw, "WRITTEN_FILE_MISMATCH: " + str(p))
    return {"sha256": sha256(raw), "bytes": len(raw)}

def write_json(path, obj):
    return write_new(path, (canonical(obj) + "\n").encode("utf-8"))

def jsonl_rows(raw):
    lines = raw.decode("utf-8").splitlines()
    need(lines and all(line.strip() for line in lines), "JSONL_BLANK_OR_EMPTY")
    rows = [strict_json(line) for line in lines]
    need(all(isinstance(r, dict) for r in rows), "JSONL_OBJECT_ROWS")
    return rows

def jsonl_bytes(rows):
    return ("".join(canonical(r) + "\n" for r in rows)).encode("utf-8")

def event(stage, **fields):
    """Structured log line. Prompt/page/fact/query/prediction text never enters."""
    print(canonical({"stage": stage, **fields}), flush=True)

# ------------------------------------------------------------- path guarding

class PathGuard:
    """Deny-by-default read allowlist with a per-stage forbidden set.

    ``allow`` refuses a forbidden path at registration, so a stage cannot even
    describe an input it is not permitted to see. Every runner read goes
    through ``check``/``read_small``, so audits compare actual opened paths.
    """

    def __init__(self, stage, forbidden=()):
        self.stage = stage
        self.allowed = {}
        self.forbidden = {str(no_symlink(p)): label for label, p in forbidden}

    def allow(self, label, path):
        p = no_symlink(path)
        need(str(p) not in self.forbidden,
             "FORBIDDEN_INPUT_FOR_STAGE " + self.stage + ": " + self.forbidden.get(str(p), label))
        self.allowed[str(p)] = label
        return p

    def check(self, path):
        p = no_symlink(path)
        need(str(p) in self.allowed,
             "PATH_NOT_ALLOWED_FOR_STAGE " + self.stage + ": " + str(p))
        return p

    def read_small(self, path, expected=None, limit=64 * 1024**2):
        return read_small(self.check(path), expected, limit)

    def labels(self):
        return sorted(self.allowed.values())

def stage_guard(stage, manifest):
    forbidden = [(name, manifest["exports"][name]["path"]) for name in STAGE_FORBIDDEN[stage]]
    return PathGuard(stage, forbidden)

# ----------------------------------------------------- activation manifest

def _identity(value, label, require_path=False):
    keys = ("path", "sha256", "bytes") if require_path else ("sha256", "bytes")
    exact_keys(value, keys, label)
    need(isinstance(value["sha256"], str) and HEX256.fullmatch(value["sha256"]), label + "_SHA256")
    need(type(value["bytes"]) is int and value["bytes"] > 0, label + "_BYTES")
    if require_path:
        need(isinstance(value["path"], str) and value["path"].startswith("/"), label + "_PATH")
    return value

def _check_settings(s):
    exact_keys(s, ("version", "seed", "qwen_load", "decoding", "encoder", "retrieval",
                   "aggregation", "arms", "tokenizer", "statistics"), "ACTIVATION_SETTINGS")
    exact_keys(s["tokenizer"], ("chat_template_sha256", "tokenizer_class",
                                "apply_chat_template_kwargs", "tokenize_kwargs",
                                "prompt_constants"), "ACTIVATION_TOKENIZER")
    need(HEX256.fullmatch(s["tokenizer"]["chat_template_sha256"]), "TOKENIZER_CHAT_TEMPLATE")
    need(s["tokenizer"]["apply_chat_template_kwargs"] ==
         {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
         "TOKENIZER_CHAT_TEMPLATE_KWARGS")
    need(s["tokenizer"]["tokenize_kwargs"] ==
         {"add_special_tokens": False, "truncation": False}, "TOKENIZER_TOKENIZE_KWARGS")
    need(s["tokenizer"]["prompt_constants"] == dict(sorted(assert_prompt_hashes().items())),
         "TOKENIZER_PROMPT_CONSTANT_DRIFT")
    exact_keys(s["statistics"], ("primary", "followup", "descriptive", "cutoffs",
                                 "effect_threshold_pp", "alpha", "bootstrap_resamples",
                                 "bootstrap_rng", "bootstrap_seed", "percentile_method",
                                 "signflip", "sampling_unit"), "ACTIVATION_STATISTICS")
    need(s["statistics"]["primary"] == "D-A" and s["statistics"]["followup"] == "D-P",
         "STATISTICS_SEQUENCE")
    need(s["statistics"]["descriptive"] == ["D-B", "D-C"], "STATISTICS_DESCRIPTIVE")
    need(s["statistics"]["cutoffs"] == [1, 5, 10], "STATISTICS_CUTOFFS")
    need(s["statistics"]["effect_threshold_pp"] == 10 and s["statistics"]["alpha"] == 0.05,
         "STATISTICS_THRESHOLDS")
    need(s["statistics"]["bootstrap_resamples"] == 20000
         and s["statistics"]["bootstrap_rng"] == "PCG64"
         and s["statistics"]["bootstrap_seed"] == SEED, "STATISTICS_BOOTSTRAP")
    need(s["statistics"]["percentile_method"] == "linear", "STATISTICS_PERCENTILE")
    need(s["statistics"]["signflip"] == "EXACT_INTEGER_DP_QID_LEVEL", "STATISTICS_SIGNFLIP")
    need(s["statistics"]["sampling_unit"] == "QID", "STATISTICS_SAMPLING_UNIT")
    need(s["seed"] == SEED and s["arms"] == list(ARMS), "SETTINGS_SEED_OR_ARMS")
    load = s["qwen_load"]
    exact_keys(load, ("quantization", "compute_dtype", "attn_implementation", "device_map_zero",
                      "trust_remote_code", "use_safetensors", "local_files_only"), "QWEN_LOAD")
    # No default: the reviewed manifest must state loading semantics explicitly.
    need(load["quantization"] in QWEN_LOAD_CHOICES, "QWEN_QUANTIZATION_NOT_REVIEWED")
    need(load["compute_dtype"] == "bfloat16" and load["attn_implementation"] == "sdpa", "QWEN_COMPUTE")
    need(load["device_map_zero"] is True and load["trust_remote_code"] is False,
         "QWEN_DEVICE_OR_REMOTE_CODE")
    need(load["use_safetensors"] is True and load["local_files_only"] is True,
         "QWEN_LOCAL_SAFETENSORS_ONLY")
    dec = s["decoding"]
    exact_keys(dec, ("do_sample", "num_beams", "state_max_new_tokens", "query_max_new_tokens",
                     "microbatch", "model_context_limit"), "DECODING")
    need(dec["do_sample"] is False and dec["num_beams"] == 1, "DECODING_NOT_GREEDY")
    need(dec["state_max_new_tokens"] == 1024 and dec["query_max_new_tokens"] == 128, "DECODING_BUDGETS")
    need(dec["model_context_limit"] == 40960, "DECODING_CONTEXT_LIMIT")
    # Padding-free microbatching only. A larger value changes the fed tensor for
    # a frozen deterministic design and needs prospective review, not a default.
    need(dec["microbatch"] == 1, "MICROBATCH_NOT_REVIEWED")
    enc = s["encoder"]
    exact_keys(enc, ("device", "batch_size", "normalize_embeddings", "precision", "dtype",
                     "attn_implementation", "backend", "max_seq_length", "prefix",
                     "query_token_cap"), "ENCODER_SETTINGS")
    need(enc["normalize_embeddings"] is True and enc["precision"] == "float32", "ENCODER_NORMALIZE")
    need(enc["dtype"] == "float32" and enc["backend"] == "torch", "ENCODER_DTYPE_BACKEND")
    need(enc["max_seq_length"] == 128 and enc["query_token_cap"] == 128, "ENCODER_LENGTHS")
    need(enc["prefix"] == "" and enc["attn_implementation"] == "sdpa", "ENCODER_PREFIX")
    need(type(enc["batch_size"]) is int and 1 <= enc["batch_size"] <= 150, "ENCODER_BATCH_SIZE")
    ret = s["retrieval"]
    exact_keys(ret, ("budget", "search_chunk", "cpu_threads", "ntotal", "dim", "metric",
                     "verify_asset_hashes"), "RETRIEVAL_SETTINGS")
    need(ret["budget"] == BUDGET and ret["ntotal"] == N_VECTORS and ret["dim"] == DIM,
         "RETRIEVAL_UNIVERSE")
    need(ret["metric"] == "INNER_PRODUCT", "RETRIEVAL_METRIC")
    need(type(ret["search_chunk"]) is int and 1 <= ret["search_chunk"] <= 150, "SEARCH_CHUNK")
    need(type(ret["cpu_threads"]) is int and 1 <= ret["cpu_threads"] <= 64, "CPU_THREADS")
    need(ret["verify_asset_hashes"] is True, "ASSET_HASH_VERIFICATION_REQUIRED")
    agg = s["aggregation"]
    exact_keys(agg, ("max_titles", "cutoffs"), "AGGREGATION")
    need(agg["max_titles"] == 10 and agg["cutoffs"] == [1, 5, 10], "AGGREGATION_FROZEN")
    return s

def load_activation(path, expected_sha256):
    """The single root of trust. Everything else is checked against this file."""
    need(isinstance(expected_sha256, str) and HEX256.fullmatch(expected_sha256),
         "EXPECTED_ACTIVATION_SHA256_REQUIRED")
    raw, identity = read_small(path, expected_sha256)
    m = strict_json(raw)
    exact_keys(m, ("schema", "status", "protocol", "amendment", "code", "preparation",
                   "preflight", "prompt_preflight", "exports", "models", "assets",
                   "environment", "settings", "roots", "cohort", "provenance",
                   "limitations", "historical_global_lineage", "experiment_state",
                   "canonical_stage0"), "ACTIVATION_MANIFEST")
    need(m["schema"] == ACTIVATION_SCHEMA, "ACTIVATION_SCHEMA")
    need(m["status"] == "PROSPECTIVE_ACTIVATION_REVIEWED", "ACTIVATION_STATUS")
    need(m["historical_global_lineage"] == LINEAGE, "ACTIVATION_LINEAGE_MUST_STAY_UNESTABLISHED")
    need(m["experiment_state"] == ACTIVATED_EXPERIMENT_STATE, "ACTIVATION_EXPERIMENT_STATE")
    need(m["canonical_stage0"] == "INCOMPLETE_GATES_UNCHANGED", "ACTIVATION_CANONICAL_STAGE0")
    _identity(m["protocol"], "ACTIVATION_PROTOCOL", require_path=True)
    _identity(m["amendment"], "ACTIVATION_AMENDMENT", require_path=True)
    exact_keys(m["code"], CODE_FILES, "ACTIVATION_CODE")
    for name, value in m["code"].items():
        _identity(value, "ACTIVATION_CODE_" + name, require_path=True)
    exact_keys(m["preparation"], ("seal_path", "seal_sha256", "artifacts"), "ACTIVATION_PREPARATION")
    exact_keys(m["preflight"], ("seal_path", "seal_sha256"), "ACTIVATION_PREFLIGHT")
    # The prompt preflight was sealed against an earlier runner revision. That
    # transition is recorded explicitly here; it is never treated as identity.
    exact_keys(m["prompt_preflight"], ("seal_path", "seal_sha256", "summary_path",
                                       "summary_sha256", "historical_runner_sha256",
                                       "checked_records", "categories_checked",
                                       "minimum_remaining_context_margin",
                                       "runtime_dependent_categories"),
               "ACTIVATION_PROMPT_PREFLIGHT")
    pp = m["prompt_preflight"]
    need(HEX256.fullmatch(pp["historical_runner_sha256"]), "PROMPT_PREFLIGHT_HISTORICAL_RUNNER")
    need(pp["checked_records"] == PROMPT_PREFLIGHT_RECORDS
         and pp["categories_checked"] == PROMPT_PREFLIGHT_CATEGORIES,
         "PROMPT_PREFLIGHT_COUNTS")
    need(type(pp["minimum_remaining_context_margin"]) is int
         and pp["minimum_remaining_context_margin"] > 0, "PROMPT_PREFLIGHT_MARGIN")
    need(pp["runtime_dependent_categories"] == ["query_C", "query_D"],
         "PROMPT_PREFLIGHT_RUNTIME_DEPENDENT")
    need(HEX256.fullmatch(pp["summary_sha256"]) and pp["summary_path"].startswith("/"),
         "PROMPT_PREFLIGHT_SUMMARY")
    for section in (m["preparation"], m["preflight"], m["prompt_preflight"]):
        need(isinstance(section["seal_sha256"], str)
             and HEX256.fullmatch(section["seal_sha256"]), "SEAL_SHA256")
        need(isinstance(section["seal_path"], str) and section["seal_path"].startswith("/"), "SEAL_PATH")
    need(isinstance(m["preparation"]["artifacts"], dict)
         and m["preparation"]["artifacts"], "PREPARATION_ARTIFACTS")
    for name, value in m["preparation"]["artifacts"].items():
        _identity(value, "PREPARATION_ARTIFACT_" + name)
    exact_keys(m["exports"], EXPORTS, "ACTIVATION_EXPORTS")
    for name, value in m["exports"].items():
        _identity(value, "ACTIVATION_EXPORT_" + name, require_path=True)
    exact_keys(m["models"], ("qwen", "encoder"), "ACTIVATION_MODELS")
    for key, model in m["models"].items():
        exact_keys(model, ("root", "files"), "ACTIVATION_MODEL_" + key)
        need(isinstance(model["root"], str) and model["root"].startswith("/"), "MODEL_ROOT")
        need(isinstance(model["files"], dict) and model["files"], "MODEL_FILES")
        for name, value in model["files"].items():
            need(isinstance(name, str) and not name.startswith("/")
                 and ".." not in name.split("/"), "MODEL_FILE_NAME")
            _identity(value, "MODEL_FILE_" + name)
    exact_keys(m["assets"], ASSET_KEYS, "ACTIVATION_ASSETS")
    for name, value in m["assets"].items():
        _identity(value, "ACTIVATION_ASSET_" + name, require_path=True)
    exact_keys(m["environment"], ("python", "executable", "packages"), "ACTIVATION_ENVIRONMENT")
    need(isinstance(m["environment"]["packages"], dict) and m["environment"]["packages"],
         "ACTIVATION_PACKAGES")
    exact_keys(m["roots"], ("output_root", "archive_root", "stage_tag",
                            "score_output_root"), "ACTIVATION_ROOTS")
    for key in ("output_root", "archive_root", "score_output_root"):
        need(isinstance(m["roots"][key], str) and m["roots"][key].startswith("/"), "ROOT_PATH")
    need(m["roots"]["output_root"] != m["roots"]["archive_root"], "ROOTS_MUST_DIFFER")
    need(isinstance(m["roots"]["stage_tag"], str)
         and re.fullmatch(r"[a-z0-9_]{1,32}", m["roots"]["stage_tag"]), "ROOT_STAGE_TAG")
    need(m["roots"]["score_output_root"] == str(Path(m["roots"]["output_root"])
                                                / ("score_" + m["roots"]["stage_tag"])),
         "SCORE_ROOT_MUST_MATCH_STAGE_TAG")
    exact_keys(m["cohort"], ("qids", "accepted_pairs", "arms", "label",
                             "predictions"), "ACTIVATION_COHORT")
    need(m["cohort"]["qids"] == 25 and m["cohort"]["accepted_pairs"] == 36
         and m["cohort"]["predictions"] == 150, "COHORT_COUNTS")
    need(m["cohort"]["arms"] == list(ARMS), "COHORT_ARMS")
    need(m["cohort"]["label"] == "HUMAN_VERIFICATION_OF_MODEL_ASSISTED_CANDIDATES",
         "COHORT_LABEL")
    exact_keys(m["provenance"], ("git_parent_head", "activation_utc",
                                 "amendment_number", "outcomes_viewed"), "ACTIVATION_PROVENANCE")
    need(re.fullmatch("[0-9a-f]{40}", m["provenance"]["git_parent_head"]), "GIT_PARENT_HEAD")
    need(m["provenance"]["amendment_number"] == 2, "AMENDMENT_NUMBER")
    need(m["provenance"]["outcomes_viewed"] is False, "OUTCOMES_ALREADY_VIEWED")
    need(isinstance(m["limitations"], list) and m["limitations"], "ACTIVATION_LIMITATIONS")
    _check_settings(m["settings"])
    # No declared input may lie inside any stage root this activation creates.
    # Checking the concrete stage roots, not their shared parents, is exact: the
    # preparation exports legitimately live beside the stage roots.
    tag = m["roots"]["stage_tag"]
    stage_roots = [Path(m["roots"][key]) / (stage + "_" + tag)
                   for key in ("output_root", "archive_root")
                   for stage in STAGE_ROOT_NAMES]
    declared = (list(m["exports"].values()) + list(m["assets"].values())
                + list(m["code"].values()) + [m["protocol"], m["amendment"]]
                + [{"path": m[k][p]} for k in ("preparation", "preflight", "prompt_preflight")
                   for p in ("seal_path",)]
                + [{"path": m["prompt_preflight"]["summary_path"]}])
    for spec in declared:
        for root in stage_roots:
            need(not Path(spec["path"]).is_relative_to(root),
                 "OUTPUT_INPUT_OVERLAP: " + spec["path"])
    return m, identity

def check_environment(m):
    need(".".join(str(x) for x in sys.version_info[:3]) == m["environment"]["python"],
         "PYTHON_VERSION_DRIFT")
    observed = {k: importlib.metadata.version(k) for k in m["environment"]["packages"]}
    need(observed == m["environment"]["packages"], "PACKAGE_VERSION_DRIFT")
    return {"python": sys.version, "executable": sys.executable, "packages": observed}

def check_seal(guard, label, seal_path, seal_sha256, status, schema=None):
    raw, identity = guard.read_small(seal_path, seal_sha256)
    seal = strict_json(raw)
    need(seal.get("status") == status, label + "_STATUS")
    need(schema is None or seal.get("schema") == schema, label + "_SCHEMA")
    return seal, identity

def check_model_files(guard, model):
    """Small files by content, large weights by streamed hash. Loads no weights."""
    root = Path(model["root"])
    checked = {}
    for name, value in sorted(model["files"].items()):
        target = guard.check(root / name)
        if value["bytes"] <= 64 * 1024**2:
            _, ident = read_small(target, value["sha256"])
        else:
            ident = fingerprint(target, value["sha256"], value["bytes"])
        checked[name] = {"sha256": ident["sha256"], "bytes": ident["bytes"]}
    need(checked == {k: {"sha256": v["sha256"], "bytes": v["bytes"]}
                     for k, v in model["files"].items()}, "MODEL_IDENTITY_DRIFT")
    return {"root": model["root"], "files": checked}

def allow_model(guard, manifest, key):
    model = manifest["models"][key]
    guard.allow("model_" + key + "_root", model["root"])
    for name in model["files"]:
        guard.allow("model_" + key + "_" + name, Path(model["root"]) / name)

def verify_code_identity(guard, manifest):
    """Every reviewed code and job file must still match the manifest exactly.

    The job scripts receive the activation identity rather than embedding it,
    so there is no manifest/script hash cycle; this check closes the loop from
    the other side and leaves no unreviewed bypass.
    """
    checked = {}
    for name, spec in sorted(manifest["code"].items()):
        path = guard.allow("code_" + name, spec["path"])
        need(path.name == name, "CODE_FILE_NAME_MISMATCH: " + name)
        _, ident = read_small(path, spec["sha256"])
        need(ident["bytes"] == spec["bytes"], "CODE_FILE_BYTES: " + name)
        checked[name] = ident["sha256"]
    # The running module and its frozen core must be the reviewed bytes.
    here = Path(__file__).resolve().parent
    for name in ("bridge_pilot_run.py", "bridge_pilot_core.py"):
        local, _ = read_small(here / name)
        need(sha256(local) == checked[name], "RUNNING_CODE_NOT_REVIEWED: " + name)
    return checked

# ------------------------------------------------------ stage root + resume

class StageRoot:
    """Fresh versioned root, exclusive-create records, rerun refused after seal."""

    def __init__(self, stage, manifest, activation_identity, tag):
        need(stage in STAGE_UNITS, "UNKNOWN_STAGE")
        need(isinstance(tag, str) and re.fullmatch(r"[a-z0-9_]{1,32}", tag), "STAGE_TAG")
        need(tag == manifest["roots"]["stage_tag"], "STAGE_TAG_NOT_REVIEWED")
        self.stage = stage
        self.tag = tag
        self.manifest = manifest
        self.activation = activation_identity
        self.root = no_symlink(Path(manifest["roots"]["output_root"]) / (stage + "_" + tag))
        self.archive = no_symlink(Path(manifest["roots"]["archive_root"]) / (stage + "_" + tag))
        self.records = self.root / "records"
        self.seal_path = self.root / "STAGE_SEAL.json"
        self.start_path = self.root / "STAGE_START.json"

    def open_or_resume(self, inputs):
        need(not self.archive.exists(), "STAGE_ARCHIVE_ALREADY_EXISTS: " + str(self.archive))
        start = {"schema": "urbench.bridge_n25.stage_start.v1", "stage": self.stage, "tag": self.tag,
                 "runner_version": VERSION, "activation_sha256": self.activation["sha256"],
                 "expected_units": STAGE_UNITS[self.stage], "inputs": inputs,
                 "output_root": str(self.root), "archive_root": str(self.archive),
                 "historical_global_lineage": LINEAGE,
                 "experiment_state": self.manifest["experiment_state"],
                 "resume_policy": "VALIDATE_EXISTING_RECORDS; CREATE_ONLY_MISSING; NEVER_OVERWRITE"}
        if self.root.exists():
            need(self.root.is_dir() and not self.root.is_symlink(), "STAGE_ROOT_NOT_DIRECTORY")
            need(not self.seal_path.exists(),
                 "STAGE_ALREADY_SEALED_RERUN_REFUSED: " + str(self.seal_path))
            raw, _ = read_small(self.start_path)
            # A resume may not silently change activation, cohort size or inputs.
            need(strict_json(raw) == start, "STAGE_RESUME_CONFIGURATION_DRIFT")
            need(self.records.is_dir() and not self.records.is_symlink(), "STAGE_RECORDS_MISSING")
            resumed = True
        else:
            new_dir(self.root)
            new_dir(self.records)
            write_json(self.start_path, start)
            resumed = False
        self.start = start
        return resumed

    def record_path(self, pass_name, qid):
        need(re.fullmatch(r"[a-z_]+_[A-E]", pass_name), "RECORD_PASS_NAME")
        need(qid in CHILD_COUNTS and QID_PATTERN.fullmatch(qid), "RECORD_QID")
        return self.records / (pass_name + "__" + qid + ".json")

    def existing(self, pass_name, qid, validate):
        """Return a validated existing record, or None. Never repairs or rewrites."""
        p = self.record_path(pass_name, qid)
        if not p.exists():
            return None
        raw, identity = read_small(p)
        row = strict_json(raw)
        need(row.get("activation_sha256") == self.activation["sha256"],
             "RECORD_ACTIVATION_DRIFT: " + p.name)
        need(row.get("qid") == qid and row.get("pass") == pass_name, "RECORD_KEY_DRIFT: " + p.name)
        validate(row)
        return row, identity

    def create(self, pass_name, qid, row):
        need(row["qid"] == qid and row["pass"] == pass_name, "RECORD_KEY_MISMATCH")
        need(row["activation_sha256"] == self.activation["sha256"], "RECORD_ACTIVATION_MISMATCH")
        return row, write_json(self.record_path(pass_name, qid), row)

    def seal(self, artifact_rows, extra):
        """Canonical JSONL is built only after every expected record validated."""
        schema, status = STAGE_SCHEMAS[self.stage]
        artifacts = {}
        for name, rows in sorted(artifact_rows.items()):
            artifacts[name] = write_new(self.root / name, jsonl_bytes(rows))
        seal = {"schema": schema, "status": status, "stage": self.stage, "tag": self.tag,
                "runner_version": VERSION, "activation_sha256": self.activation["sha256"],
                "artifacts": artifacts, "historical_global_lineage": LINEAGE,
                "experiment_state": self.manifest["experiment_state"], **extra}
        seal_identity = write_json(self.seal_path, seal)
        self.copy_to_archive(sorted(artifacts))
        return seal, seal_identity

    def copy_to_archive(self, artifact_names):
        new_dir(self.archive)
        copied = {}
        for name in ("STAGE_START.json", *artifact_names, "STAGE_SEAL.json"):
            raw, _ = read_small(self.root / name)
            copied[name] = write_new(self.archive / name, raw)
        new_dir(self.archive / "records")
        for p in sorted(self.records.iterdir()):
            raw, _ = read_small(p)
            copied["records/" + p.name] = write_new(self.archive / "records" / p.name, raw)
        write_json(self.archive / "ARCHIVE_INDEX.json",
                   {"schema": "urbench.bridge_n25.stage_archive.v1", "stage": self.stage,
                    "tag": self.tag, "source_root": str(self.root), "files": copied,
                    "historical_global_lineage": LINEAGE})
        return copied

# ------------------------------------------------------------------- models

class QwenWriter:
    """One Qwen load per stage. Greedy, thinking disabled, no cross-item state."""

    def __init__(self, manifest, guard):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        load = manifest["settings"]["qwen_load"]
        self.decoding = manifest["settings"]["decoding"]
        root = str(guard.check(Path(manifest["models"]["qwen"]["root"])))
        need(torch.cuda.is_available(), "GPU_REQUIRED")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        kwargs = dict(dtype=torch.bfloat16, attn_implementation=load["attn_implementation"],
                      device_map={"": 0}, trust_remote_code=False, use_safetensors=True,
                      local_files_only=True)
        if load["quantization"] == "bnb_nf4_double_bfloat16":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(root, **kwargs)
        self.model.eval()
        # Keep the snapshot's stop tokens; force greedy and clear sampling knobs.
        gc = copy.deepcopy(self.model.generation_config)
        gc.do_sample = False
        gc.num_beams = 1
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
        self.generation_config = gc
        self.identity = {"root": root, "quantization": load["quantization"],
                         "compute_dtype": "bfloat16",
                         "attn_implementation": load["attn_implementation"],
                         "tokenizer_class": type(self.tokenizer).__name__,
                         "chat_template_sha256": sha256(self.tokenizer.chat_template or ""),
                         "model_class": type(self.model).__name__,
                         "model_context_limit": int(self.model.config.max_position_embeddings),
                         "vocab_size": int(self.model.config.vocab_size),
                         "eos_token_id": gc.eos_token_id, "pad_token_id": gc.pad_token_id,
                         "do_sample": bool(gc.do_sample), "num_beams": int(gc.num_beams),
                         "decoding": dict(self.decoding)}
        need(self.identity["model_context_limit"] == self.decoding["model_context_limit"],
             "MODEL_CONTEXT_LIMIT_DRIFT")

    def generate(self, messages, max_new_tokens):
        """Render, context-check, greedy generate. Exceptions propagate upward."""
        rendered = render_prompt(self.tokenizer, messages, max_new_tokens)
        ids = self.torch.tensor([rendered["input_ids"]], dtype=self.torch.long,
                                device=self.model.device)
        mask = self.torch.ones_like(ids)
        started = time.monotonic()
        with self.torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=mask,
                                      max_new_tokens=max_new_tokens,
                                      generation_config=self.generation_config,
                                      do_sample=False, num_beams=1, use_cache=True)
        elapsed = time.monotonic() - started
        need(out.shape[0] == 1, "GENERATION_BATCH_SHAPE")
        new_ids = out[0][ids.shape[1]:].tolist()
        return {"prompt_sha256": rendered["prompt_sha256"], "input_tokens": rendered["input_tokens"],
                "max_new_tokens": max_new_tokens, "output_tokens": len(new_ids),
                "length_capped": len(new_ids) >= max_new_tokens,
                "raw_output": self.tokenizer.decode(new_ids, skip_special_tokens=True),
                "generation_status": "COMPLETED", "generation_seconds": round(elapsed, 3)}

def encoder_tokenizer(manifest, guard):
    """MiniLM tokenizer only. Query capping needs no encoder weights."""
    from transformers import AutoTokenizer
    root = str(guard.check(Path(manifest["models"]["encoder"]["root"])))
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    return tok, {"root": root, "tokenizer_class": type(tok).__name__,
                 "model_max_length": int(getattr(tok, "model_max_length", 0))}

# ------------------------------------------------------------ record schemas

GENERATION_KEYS = ("schema", "pass", "qid", "arm", "activation_sha256", "runner_version",
                   "preparation_seal_sha256", "preflight_seal_sha256", "source_row",
                   "model_identity", "encoder_tokenizer_identity", "prompt_constants",
                   "prompt_sha256", "input_tokens", "max_new_tokens", "output_tokens",
                   "length_capped", "decoding", "raw_output", "generation_status",
                   "result", "historical_global_lineage", "utc")

PREDICTION_KEYS = ("schema", "pass", "qid", "arm", "activation_sha256", "runner_version",
                   "preparation_seal_sha256", "preflight_seal_sha256", "parent_seal_sha256",
                   "oracle_e_seal_sha256", "query_record", "encoder_identity", "index_identity",
                   "metadata_identity", "offsets_identity", "search_budget",
                   "candidate_provenance", "prediction", "historical_global_lineage", "utc")

def generation_record(pass_name, qid, arm, base, generated, result):
    row = {"schema": "urbench.bridge_n25.generation.v1", "pass": pass_name, "qid": qid, "arm": arm,
           "prompt_sha256": generated["prompt_sha256"], "input_tokens": generated["input_tokens"],
           "max_new_tokens": generated["max_new_tokens"], "output_tokens": generated["output_tokens"],
           "length_capped": generated["length_capped"], "raw_output": generated["raw_output"],
           "generation_status": generated["generation_status"], "result": result,
           "utc": datetime.now(timezone.utc).isoformat(), **base}
    exact_keys(row, GENERATION_KEYS, "GENERATION_RECORD")
    return row

def validate_generation_record(row, expected_arm, expected_pass):
    exact_keys(row, GENERATION_KEYS, "GENERATION_RECORD")
    need(row["schema"] == "urbench.bridge_n25.generation.v1", "GENERATION_SCHEMA")
    need(row["arm"] == expected_arm and row["pass"] == expected_pass, "GENERATION_ARM_OR_PASS")
    need(row["generation_status"] == "COMPLETED", "GENERATION_NOT_COMPLETED")
    need(isinstance(row["raw_output"], str), "GENERATION_RAW_OUTPUT")
    need(type(row["input_tokens"]) is int and type(row["output_tokens"]) is int,
         "GENERATION_TOKEN_COUNTS")
    need(row["input_tokens"] + row["max_new_tokens"] <= 40960, "GENERATION_CONTEXT_BUDGET")
    need(isinstance(row["prompt_sha256"], str)
         and HEX256.fullmatch(row["prompt_sha256"]), "GENERATION_PROMPT_SHA256")
    need(row["prompt_constants"] == dict(sorted(assert_prompt_hashes().items())),
         "PROMPT_CONSTANT_DRIFT_IN_RECORD")
    need(row["historical_global_lineage"] == LINEAGE, "GENERATION_LINEAGE")
    need(not ({"chain_of_thought", "thinking", "reasoning", "reasoning_content"} & set(row)),
         "CHAIN_OF_THOUGHT_FIELD_FORBIDDEN")
    if expected_pass.startswith("state_"):
        state = row["result"]
        exact_keys(state, ("arm", "raw_output", "format_failure", "items", "invalid_items",
                           "duplicate_items", "excess_items", "raw_item_count"), "STATE_RESULT")
        need(state["arm"] == expected_arm and state["raw_output"] == row["raw_output"],
             "STATE_RESULT_BINDING")
        need(len(state["items"]) <= 6, "STATE_ITEM_CAP")
        need(type(state["format_failure"]) is bool, "STATE_FORMAT_FLAG")
    else:
        q = row["result"]
        exact_keys(q, ("raw_output", "query", "fallback", "model_empty", "word_cap_removed",
                       "encoder_cap_removed", "encoder_tokens_before_cap", "encoder_tokens"),
                   "QUERY_RESULT")
        need(q["raw_output"] == row["raw_output"], "QUERY_RESULT_BINDING")
        need(isinstance(q["query"], str) and q["query"].strip()
             and len(q["query"].split()) <= 32, "QUERY_TEXT")
        need(q["encoder_tokens"] <= 128, "QUERY_ENCODER_CAP")
        need(q["fallback"] in (None, "URDU_QUESTION", "SEALED_A"), "QUERY_FALLBACK_KIND")
        need(q["fallback"] != "SEALED_A" or expected_arm != "A", "A_CANNOT_FALL_BACK_TO_ITSELF")
    return row

def validate_prediction_record(row, expected_arm):
    exact_keys(row, PREDICTION_KEYS, "PREDICTION_RECORD")
    need(row["schema"] == "urbench.bridge_n25.prediction.v1", "PREDICTION_SCHEMA")
    need(row["arm"] == expected_arm and row["pass"] == "prediction_" + expected_arm, "PREDICTION_ARM")
    need(row["search_budget"] == BUDGET, "PREDICTION_BUDGET")
    need(row["historical_global_lineage"] == LINEAGE, "PREDICTION_LINEAGE")
    exact_keys(row["query_record"], ("pass", "qid", "arm", "prompt_sha256", "record_sha256",
                                     "fallback", "encoder_tokens"), "PREDICTION_QUERY_RECORD")
    need(row["query_record"]["qid"] == row["qid"] and row["query_record"]["arm"] == expected_arm,
         "PREDICTION_QUERY_BINDING")
    need(row["query_record"]["pass"] == "query_" + expected_arm, "PREDICTION_QUERY_PASS")
    p = row["prediction"]
    exact_keys(p, ("qid", "arm", "query", "candidates", "ranked_titles",
                   "boundary_tie_within_returned", "ties_beyond_budget"), "PREDICTION_VIEW")
    need(p["qid"] == row["qid"] and p["arm"] == expected_arm, "PREDICTION_VIEW_KEY")
    need(len(p["candidates"]) == BUDGET, "PREDICTION_CANDIDATE_COUNT")
    need(p["ranked_titles"] == aggregate_candidates(p["candidates"]), "PREDICTION_RANKING_DRIFT")
    need(len(p["ranked_titles"]) <= 10, "PREDICTION_TITLE_CAP")
    prov = row["candidate_provenance"]
    need(isinstance(prov, list) and len(prov) == BUDGET, "PREDICTION_PROVENANCE_COUNT")
    for item, cand in zip(prov, p["candidates"]):
        exact_keys(item, ("global_row", "byte_offset", "metadata_line_sha256"), "CANDIDATE_PROVENANCE")
        need(item["global_row"] == cand["global_row"], "CANDIDATE_PROVENANCE_ROW")
    return row

def source_row_identity(row):
    return {"qid": row["qid"], "row_sha256": sha256(canonical(row))}

def record_base(manifest, activation_identity):
    return {"activation_sha256": activation_identity["sha256"], "runner_version": VERSION,
            "preparation_seal_sha256": manifest["preparation"]["seal_sha256"],
            "preflight_seal_sha256": manifest["preflight"]["seal_sha256"],
            "prompt_constants": dict(sorted(assert_prompt_hashes().items())),
            "decoding": dict(manifest["settings"]["decoding"]),
            "historical_global_lineage": LINEAGE}

# ------------------------------------------------------------ export loading

def read_export(guard, manifest, name, validator):
    spec = manifest["exports"][name]
    raw, actual = guard.read_small(Path(spec["path"]), spec["sha256"])
    need(actual["bytes"] == spec["bytes"], "EXPORT_BYTES: " + name)
    rows = jsonl_rows(raw)
    validator(rows)
    return rows, actual

def validate_parent_export(rows):
    need(len(rows) == 25, "PARENT_EXPORT_ROWS")
    need({r.get("qid") for r in rows} == set(CHILD_COUNTS), "PARENT_EXPORT_COHORT")
    for row in rows:
        validate_parent(row)

def validate_oracle_export(rows):
    need(len(rows) == 25, "ORACLE_EXPORT_ROWS")
    need({r.get("qid") for r in rows} == set(CHILD_COUNTS), "ORACLE_EXPORT_COHORT")
    for row in rows:
        oracle_query_messages(row)

def authenticate_prompt_preflight(guard, manifest):
    """Bind the sealed tokenizer-only prompt evidence.

    The seal was produced by an earlier runner revision, so its recorded runner
    hash is checked against the manifest's explicitly declared historical value,
    never against the active runner. Any third value is rejected. The core hash
    is required to be unchanged, because ``bridge_pilot_core`` owns the prompt
    construction shared by the audit and the runtime.
    """
    spec = manifest["prompt_preflight"]
    seal, seal_id = check_seal(guard, "PROMPT_PREFLIGHT", spec["seal_path"],
                               spec["seal_sha256"], PROMPT_PREFLIGHT_STATUS)
    _, summary_id = guard.read_small(spec["summary_path"], spec["summary_sha256"])
    need(seal["artifacts"]["prompt_preflight_summary.json"]["sha256"] == spec["summary_sha256"],
         "PROMPT_PREFLIGHT_SUMMARY_NOT_SEALED")
    need(seal["preparation_seal_sha256"] == manifest["preparation"]["seal_sha256"]
         and seal["preflight_seal_sha256"] == manifest["preflight"]["seal_sha256"],
         "PROMPT_PREFLIGHT_UPSTREAM_BINDING")
    active_core = manifest["code"]["bridge_pilot_core.py"]["sha256"]
    need(seal["core_sha256"] == active_core, "PROMPT_PREFLIGHT_CORE_DRIFT")
    historical = spec["historical_runner_sha256"]
    active_runner = manifest["code"]["bridge_pilot_run.py"]["sha256"]
    need(seal["runner_sha256"] in (historical, active_runner),
         "PROMPT_PREFLIGHT_RUNNER_UNKNOWN")
    need(seal["runner_sha256"] == historical, "PROMPT_PREFLIGHT_HISTORICAL_RUNNER_DRIFT")
    need(seal["checked_records"] == spec["checked_records"] == PROMPT_PREFLIGHT_RECORDS,
         "PROMPT_PREFLIGHT_RECORD_COUNT")
    need(seal["categories_checked"] == spec["categories_checked"] == PROMPT_PREFLIGHT_CATEGORIES,
         "PROMPT_PREFLIGHT_CATEGORY_COUNT")
    need(seal["all_known_prompts_within_context"] is True, "PROMPT_PREFLIGHT_CONTEXT_FAILURE")
    need(min(x["remaining_margin"] for x in seal["maxima"].values())
         == spec["minimum_remaining_context_margin"], "PROMPT_PREFLIGHT_MARGIN_DRIFT")
    need(seal["runtime_dependent"]["scope"] == "RUNTIME_DEPENDENT"
         and seal["runtime_dependent"]["categories"] == spec["runtime_dependent_categories"],
         "PROMPT_PREFLIGHT_RUNTIME_DEPENDENT_DRIFT")
    need(seal["prompt_constants"] == dict(sorted(assert_prompt_hashes().items())),
         "PROMPT_PREFLIGHT_PROMPT_CONSTANT_DRIFT")
    need(seal["chat_template_sha256"]
         == manifest["settings"]["tokenizer"]["chat_template_sha256"],
         "PROMPT_PREFLIGHT_CHAT_TEMPLATE_DRIFT")
    need(seal["qwen_weight_files_opened"] == 0 and seal["qwen_generations"] == 0
         and seal["pilot_searches"] == 0 and seal["scoring_targets_opened"] is False,
         "PROMPT_PREFLIGHT_NOT_OUTCOME_FREE")
    need(seal["historical_global_lineage"] == LINEAGE, "PROMPT_PREFLIGHT_LINEAGE")
    need(seal["experiment_state"] == EXPERIMENT_STATE, "PROMPT_PREFLIGHT_HISTORICAL_STATE")
    return {"seal": seal_id, "summary": summary_id,
            "historical_runner_sha256": historical,
            "active_runner_sha256": active_runner,
            "runner_revision_changed": historical != active_runner}

def authenticate_upstream(guard, manifest):
    _, prep_id = check_seal(guard, "PREPARATION", manifest["preparation"]["seal_path"],
                            manifest["preparation"]["seal_sha256"], PREPARATION_STATUS)
    pre_seal, pre_id = check_seal(guard, "PREFLIGHT", manifest["preflight"]["seal_path"],
                                  manifest["preflight"]["seal_sha256"], PREFLIGHT_STATUS)
    need(pre_seal.get("historical_global_lineage") == LINEAGE, "PREFLIGHT_LINEAGE")
    need(pre_seal.get("preparation_seal_sha256") == manifest["preparation"]["seal_sha256"],
         "PREFLIGHT_PREPARATION_BINDING")
    prompt_id = authenticate_prompt_preflight(guard, manifest)
    return prep_id, pre_id, prompt_id

# ------------------------------------------------------------- parent stage

def stage_parent(manifest, activation_identity, tag):
    guard = stage_guard("parent", manifest)
    guard.allow("export_parent_only", manifest["exports"][PARENT_EXPORT]["path"])
    guard.allow("preparation_seal", manifest["preparation"]["seal_path"])
    guard.allow("preflight_seal", manifest["preflight"]["seal_path"])
    guard.allow("prompt_preflight_seal", manifest["prompt_preflight"]["seal_path"])
    guard.allow("prompt_preflight_summary", manifest["prompt_preflight"]["summary_path"])
    allow_model(guard, manifest, "qwen")
    allow_model(guard, manifest, "encoder")
    environment = check_environment(manifest)
    code_identity = verify_code_identity(guard, manifest)
    prep_id, pre_id, prompt_id = authenticate_upstream(guard, manifest)
    prep_artifacts = manifest["preparation"]["artifacts"]
    need(prep_artifacts[PARENT_EXPORT]["sha256"] == manifest["exports"][PARENT_EXPORT]["sha256"],
         "PREPARATION_EXPORT_BINDING")
    parents, export_id = read_export(guard, manifest, PARENT_EXPORT, validate_parent_export)
    by_qid = {r["qid"]: r for r in parents}
    root = StageRoot("parent", manifest, activation_identity, tag)
    resumed = root.open_or_resume({PARENT_EXPORT: export_id, "preparation_seal": prep_id,
                                   "preflight_seal": pre_id, "environment": environment,
                                   "prompt_preflight": prompt_id,
                                   "code_identity": code_identity})
    event("stage_open", name="parent", tag=tag, resumed=resumed,
          expected_units=STAGE_UNITS["parent"], allowed_inputs=guard.labels())
    qids = sorted(CHILD_COUNTS)
    plan = [(p, q) for p in PARENT_PASSES for q in qids]

    def checker(pass_name):
        arm = pass_name.rsplit("_", 1)[1]
        return lambda r: validate_generation_record(r, arm, pass_name)

    missing = {(p, q) for p, q in plan if root.existing(p, q, checker(p)) is None}
    event("resume_scan", name="parent", existing=len(plan) - len(missing), missing=len(missing))
    base_common = record_base(manifest, activation_identity)
    writer = enc_tok = None
    if missing:
        qwen_identity = check_model_files(guard, manifest["models"]["qwen"])
        encoder_files = check_model_files(guard, manifest["models"]["encoder"])
        event("model_identity_verified", qwen_files=len(qwen_identity["files"]),
              encoder_files=len(encoder_files["files"]))
        writer = QwenWriter(manifest, guard)
        enc_tok, enc_tok_identity = encoder_tokenizer(manifest, guard)
        base_common["model_identity"] = writer.identity
        base_common["encoder_tokenizer_identity"] = enc_tok_identity
        event("models_loaded", name="parent", quantization=writer.identity["quantization"])
    dec = manifest["settings"]["decoding"]
    for pass_name, qid in plan:
        if (pass_name, qid) not in missing:
            continue
        arm = pass_name.rsplit("_", 1)[1]
        row = by_qid[qid]
        base = dict(base_common, source_row=source_row_identity(row))
        if pass_name.startswith("state_"):
            gen = writer.generate(state_messages(row, arm), dec["state_max_new_tokens"])
            result = parse_state(gen["raw_output"], row, arm)
        else:
            state = None
            if arm in STATE_ARMS:
                found = root.existing("state_" + arm, qid, checker("state_" + arm))
                need(found is not None, "STATE_RECORD_REQUIRED_BEFORE_QUERY")
                state = found[0]["result"]
            gen = writer.generate(parent_query_messages(row, arm, state), dec["query_max_new_tokens"])
            sealed_a = None
            if arm != "A":
                found_a = root.existing("query_A", qid, checker("query_A"))
                need(found_a is not None, "SEALED_A_QUERY_REQUIRED")
                sealed_a = found_a[0]["result"]["query"]
            result = cap_query(gen["raw_output"], row["question_ur"], arm, enc_tok, sealed_a)
        record = generation_record(pass_name, qid, arm, base, gen, result)
        validate_generation_record(record, arm, pass_name)
        root.create(pass_name, qid, record)
        event("record_written", name="parent", record_pass=pass_name, qid=qid, arm=arm,
              input_tokens=gen["input_tokens"], output_tokens=gen["output_tokens"],
              length_capped=gen["length_capped"], seconds=gen["generation_seconds"],
              fallback=result.get("fallback"), format_failure=result.get("format_failure"),
              state_items=(len(result["items"]) if "items" in result else None))
    states, queries = [], []
    for pass_name in PARENT_PASSES:
        for qid in qids:
            found = root.existing(pass_name, qid, checker(pass_name))
            need(found is not None, "MISSING_RECORD_AT_SEAL: " + pass_name + " " + qid)
            (states if pass_name.startswith("state_") else queries).append(found[0])
    need(len(states) == 50 and len(queries) == 125, "PARENT_STAGE_COUNTS")
    _, seal_id = root.seal({"states.jsonl": states, "queries.jsonl": queries},
                           {"states": 50, "queries": 125,
                            "preparation_seal_sha256": manifest["preparation"]["seal_sha256"],
                            "preflight_seal_sha256": manifest["preflight"]["seal_sha256"],
                            "export_identity": export_id, "oracle_e_opened": False,
                            "targets_opened": False, "pilot_searches": 0, "scored_outcomes": 0})
    event("stage_sealed", name="parent", tag=tag, seal_sha256=seal_id["sha256"],
          states=50, queries=125, archive=str(root.archive))
    return seal_id

# ----------------------------------------------------------- oracle-E stage

def stage_oracle_e(manifest, activation_identity, tag, parent_tag):
    guard = stage_guard("oracle_e", manifest)
    guard.allow("export_oracle_e", manifest["exports"][ORACLE_EXPORT]["path"])
    parent_root = Path(manifest["roots"]["output_root"]) / ("parent_" + parent_tag)
    guard.allow("parent_seal", parent_root / "STAGE_SEAL.json")
    guard.allow("parent_queries", parent_root / "queries.jsonl")
    guard.allow("preparation_seal", manifest["preparation"]["seal_path"])
    guard.allow("preflight_seal", manifest["preflight"]["seal_path"])
    guard.allow("prompt_preflight_seal", manifest["prompt_preflight"]["seal_path"])
    guard.allow("prompt_preflight_summary", manifest["prompt_preflight"]["summary_path"])
    allow_model(guard, manifest, "qwen")
    allow_model(guard, manifest, "encoder")
    environment = check_environment(manifest)
    code_identity = verify_code_identity(guard, manifest)
    prep_id, pre_id, prompt_id = authenticate_upstream(guard, manifest)
    schema, status = STAGE_SCHEMAS["parent"]
    parent_seal, parent_seal_id = check_seal(guard, "PARENT", parent_root / "STAGE_SEAL.json",
                                             None, status, schema)
    need(parent_seal["activation_sha256"] == activation_identity["sha256"], "PARENT_SEAL_ACTIVATION")
    need(parent_seal["queries"] == 125 and parent_seal["oracle_e_opened"] is False,
         "PARENT_SEAL_CONTENT")
    q_spec = parent_seal["artifacts"]["queries.jsonl"]
    raw_q, actual_q = guard.read_small(parent_root / "queries.jsonl", q_spec["sha256"])
    need(actual_q["bytes"] == q_spec["bytes"], "PARENT_QUERIES_BYTES")
    sealed_a = {}
    for row in jsonl_rows(raw_q):
        if row.get("pass") == "query_A":
            validate_generation_record(row, "A", "query_A")
            sealed_a[row["qid"]] = row["result"]["query"]
    need(set(sealed_a) == set(CHILD_COUNTS), "SEALED_A_COHORT")
    rows, export_id = read_export(guard, manifest, ORACLE_EXPORT, validate_oracle_export)
    by_qid = {r["qid"]: r for r in rows}
    root = StageRoot("oracle_e", manifest, activation_identity, tag)
    resumed = root.open_or_resume({ORACLE_EXPORT: export_id, "preparation_seal": prep_id,
                                   "preflight_seal": pre_id, "parent_seal": parent_seal_id,
                                   "parent_tag": parent_tag, "environment": environment,
                                   "prompt_preflight": prompt_id,
                                   "code_identity": code_identity})
    event("stage_open", name="oracle_e", tag=tag, resumed=resumed, expected_units=25,
          allowed_inputs=guard.labels())
    check = lambda r: validate_generation_record(r, "E", "query_E")
    qids = sorted(CHILD_COUNTS)
    missing = [q for q in qids if root.existing("query_E", q, check) is None]
    event("resume_scan", name="oracle_e", existing=25 - len(missing), missing=len(missing))
    if missing:
        qwen_identity = check_model_files(guard, manifest["models"]["qwen"])
        encoder_files = check_model_files(guard, manifest["models"]["encoder"])
        event("model_identity_verified", qwen_files=len(qwen_identity["files"]),
              encoder_files=len(encoder_files["files"]))
        writer = QwenWriter(manifest, guard)
        enc_tok, enc_tok_identity = encoder_tokenizer(manifest, guard)
        base_common = dict(record_base(manifest, activation_identity),
                           model_identity=writer.identity,
                           encoder_tokenizer_identity=enc_tok_identity)
        event("models_loaded", name="oracle_e", quantization=writer.identity["quantization"])
        dec = manifest["settings"]["decoding"]
        for qid in missing:
            row = by_qid[qid]
            gen = writer.generate(oracle_query_messages(row), dec["query_max_new_tokens"])
            result = cap_query(gen["raw_output"], row["question_ur"], "E", enc_tok, sealed_a[qid])
            base = dict(base_common, source_row=source_row_identity(row))
            record = generation_record("query_E", qid, "E", base, gen, result)
            validate_generation_record(record, "E", "query_E")
            root.create("query_E", qid, record)
            event("record_written", name="oracle_e", record_pass="query_E", qid=qid, arm="E",
                  input_tokens=gen["input_tokens"], output_tokens=gen["output_tokens"],
                  length_capped=gen["length_capped"], seconds=gen["generation_seconds"],
                  fallback=result["fallback"])
    queries = []
    for qid in qids:
        found = root.existing("query_E", qid, check)
        need(found is not None, "MISSING_RECORD_AT_SEAL: query_E " + qid)
        queries.append(found[0])
    _, seal_id = root.seal({"queries_e.jsonl": queries},
                           {"queries": 25, "parent_seal_sha256": parent_seal_id["sha256"],
                            "preparation_seal_sha256": manifest["preparation"]["seal_sha256"],
                            "preflight_seal_sha256": manifest["preflight"]["seal_sha256"],
                            "export_identity": export_id, "parent_only_opened": False,
                            "targets_opened": False, "pilot_searches": 0, "scored_outcomes": 0})
    event("stage_sealed", name="oracle_e", tag=tag, seal_sha256=seal_id["sha256"],
          queries=25, archive=str(root.archive))
    return seal_id

# ----------------------------------------------------------- retrieve stage

class MetadataReader:
    """Bounded offset-seek reader. Read-only; never builds, repairs or writes."""

    def __init__(self, metadata_path, offsets_path, metadata_bytes, total=N_VECTORS):
        import numpy as np
        self.total = total
        self.size = metadata_bytes
        self.offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        need(self.offsets.ndim == 1 and len(self.offsets) == total, "OFFSETS_SHAPE")
        need(self.offsets.dtype.kind in "iu" and self.offsets.dtype.itemsize == 8, "OFFSETS_DTYPE")
        need(int(self.offsets[0]) == 0 and 0 <= int(self.offsets[-1]) < self.size,
             "OFFSETS_ENDPOINTS")
        self.stream = open(metadata_path, "rb")
        self.provenance = {}

    def close(self):
        self.stream.close()

    def lookup(self, row):
        """Reached only after core validated every returned ID in the result."""
        need(type(row) is int and 0 <= row < self.total, "LOOKUP_ROW_RANGE")
        offset = int(self.offsets[row])
        end = int(self.offsets[row + 1]) if row + 1 < self.total else self.size
        need(0 <= offset < end <= self.size and end - offset <= 8 * 1024**2,
             "LOOKUP_OFFSET_BOUNDS")
        if offset:
            self.stream.seek(offset - 1)
            need(self.stream.read(1) == b"\n", "OFFSET_NOT_LINE_START")
        self.stream.seek(offset)
        raw = self.stream.read(end - offset)
        need(len(raw) == end - offset and raw.endswith(b"\n") and raw.count(b"\n") == 1,
             "METADATA_LINE_BOUNDARY")
        obj = strict_json(raw)
        exact_keys(obj, ("title", "text"), "METADATA")
        need(isinstance(obj["title"], str) and norm(obj["title"])
             and isinstance(obj["text"], str), "METADATA_STRINGS")
        self.provenance[row] = {"global_row": row, "byte_offset": offset,
                                "metadata_line_sha256": sha256(raw)}
        return obj

def load_query_seal(guard, manifest, activation_identity, stage, tag, artifact):
    root = Path(manifest["roots"]["output_root"]) / (stage + "_" + tag)
    guard.allow(stage + "_seal", root / "STAGE_SEAL.json")
    guard.allow(stage + "_" + artifact, root / artifact)
    schema, status = STAGE_SCHEMAS[stage]
    seal, seal_id = check_seal(guard, stage.upper(), root / "STAGE_SEAL.json", None, status, schema)
    need(seal["activation_sha256"] == activation_identity["sha256"],
         stage.upper() + "_SEAL_ACTIVATION")
    need(seal["targets_opened"] is False, stage.upper() + "_SEAL_TARGETS_OPENED")
    spec = seal["artifacts"][artifact]
    raw_rows, actual = guard.read_small(root / artifact, spec["sha256"])
    need(actual["bytes"] == spec["bytes"], stage.upper() + "_ARTIFACT_BYTES")
    return seal, seal_id, jsonl_rows(raw_rows)

def stage_retrieve(manifest, activation_identity, tag, parent_tag, oracle_tag):
    guard = stage_guard("retrieve", manifest)
    guard.allow("preparation_seal", manifest["preparation"]["seal_path"])
    guard.allow("preflight_seal", manifest["preflight"]["seal_path"])
    guard.allow("prompt_preflight_seal", manifest["prompt_preflight"]["seal_path"])
    guard.allow("prompt_preflight_summary", manifest["prompt_preflight"]["summary_path"])
    for key in ASSET_KEYS:
        guard.allow("asset_" + key, manifest["assets"][key]["path"])
    allow_model(guard, manifest, "encoder")
    environment = check_environment(manifest)
    code_identity = verify_code_identity(guard, manifest)
    prep_id, pre_id, prompt_id = authenticate_upstream(guard, manifest)
    parent_seal, parent_seal_id, parent_rows = load_query_seal(
        guard, manifest, activation_identity, "parent", parent_tag, "queries.jsonl")
    e_seal, e_seal_id, e_rows = load_query_seal(
        guard, manifest, activation_identity, "oracle_e", oracle_tag, "queries_e.jsonl")
    need(e_seal["parent_seal_sha256"] == parent_seal_id["sha256"], "ORACLE_E_PARENT_BINDING")
    need(parent_seal["queries"] == 125 and e_seal["queries"] == 25, "QUERY_SEAL_COUNTS")
    queries = {}
    for row in parent_rows + e_rows:
        arm = row.get("arm")
        need(arm in ARMS, "QUERY_ARM")
        validate_generation_record(row, arm, "query_" + arm)
        key = (row["qid"], arm)
        need(key not in queries, "DUPLICATE_QUERY_RECORD")
        queries[key] = row
    need(set(queries) == {(q, a) for q in CHILD_COUNTS for a in ARMS} and len(queries) == 150,
         "QUERY_COHORT_INCOMPLETE")
    root = StageRoot("retrieve", manifest, activation_identity, tag)
    resumed = root.open_or_resume({"preparation_seal": prep_id, "preflight_seal": pre_id,
                                   "parent_seal": parent_seal_id, "oracle_e_seal": e_seal_id,
                                   "parent_tag": parent_tag, "oracle_e_tag": oracle_tag,
                                   "environment": environment,
                                   "prompt_preflight": prompt_id,
                                   "code_identity": code_identity})
    event("stage_open", name="retrieve", tag=tag, resumed=resumed, expected_units=150,
          allowed_inputs=guard.labels())
    plan = [(q, a) for q in sorted(CHILD_COUNTS) for a in ARMS]
    missing = [(q, a) for q, a in plan
               if root.existing("prediction_" + a, q,
                                lambda r, a=a: validate_prediction_record(r, a)) is None]
    event("resume_scan", name="retrieve", existing=150 - len(missing), missing=len(missing))
    if missing:
        retrieve_missing(guard, manifest, activation_identity, root, queries, missing,
                         parent_seal_id, e_seal_id)
    records, scoring_view = [], []
    for qid, arm in plan:
        found = root.existing("prediction_" + arm, qid,
                              lambda r, a=arm: validate_prediction_record(r, a))
        need(found is not None, "MISSING_RECORD_AT_SEAL: prediction_" + arm + " " + qid)
        records.append(found[0])
        scoring_view.append(found[0]["prediction"])
    validate_predictions(scoring_view)
    _, seal_id = root.seal({"prediction_records.jsonl": records, "predictions.jsonl": scoring_view},
                           {"predictions": 150,
                            "targets_sha256": manifest["exports"][TARGET_EXPORT]["sha256"],
                            "parent_seal_sha256": parent_seal_id["sha256"],
                            "oracle_e_seal_sha256": e_seal_id["sha256"],
                            "preparation_seal_sha256": manifest["preparation"]["seal_sha256"],
                            "preflight_seal_sha256": manifest["preflight"]["seal_sha256"],
                            "search_budget": BUDGET, "targets_opened": False,
                            "scored_outcomes": 0})
    event("stage_sealed", name="retrieve", tag=tag, seal_sha256=seal_id["sha256"],
          predictions=150, archive=str(root.archive))
    return seal_id

def retrieve_missing(guard, manifest, activation_identity, root, queries, missing,
                     parent_seal_id, e_seal_id):
    import numpy as np
    import torch
    import faiss
    from sentence_transformers import SentenceTransformer
    settings = manifest["settings"]
    ret, enc_cfg = settings["retrieval"], settings["encoder"]
    encoder_files = check_model_files(guard, manifest["models"]["encoder"])
    assets = {}
    for key in ASSET_KEYS:
        spec = manifest["assets"][key]
        event("asset_hash_started", asset=key)
        assets[key] = fingerprint(guard.check(spec["path"]), spec["sha256"], spec["bytes"])
        event("asset_hash_complete", asset=key)
    need(torch.cuda.is_available(), "GPU_REQUIRED")
    torch.set_num_threads(ret["cpu_threads"])
    faiss.omp_set_num_threads(ret["cpu_threads"])
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(SEED)
    model = SentenceTransformer(manifest["models"]["encoder"]["root"], device=enc_cfg["device"],
                                backend=enc_cfg["backend"], local_files_only=True,
                                trust_remote_code=False,
                                model_kwargs={"use_safetensors": True, "dtype": torch.float32,
                                              "attn_implementation": enc_cfg["attn_implementation"]})
    model.eval()
    need(model.max_seq_length == enc_cfg["max_seq_length"], "ENCODER_MAX_SEQ_LENGTH")
    need(model.get_sentence_embedding_dimension() == DIM, "ENCODER_DIMENSION")
    need(all(p.dtype == torch.float32 for p in model.parameters()), "ENCODER_DTYPE")
    need(getattr(model, "default_prompt_name", None) is None, "ENCODER_DEFAULT_PROMPT")
    encoder_runtime = {**encoder_files, "max_seq_length": int(model.max_seq_length),
                       "dimension": int(model.get_sentence_embedding_dimension()),
                       "normalize_embeddings": True, "precision": "float32",
                       "batch_size": enc_cfg["batch_size"], "prefix": ""}
    event("index_load_started")
    index = faiss.read_index(str(guard.check(manifest["assets"]["index"]["path"])))
    observed = {"python_class": type(index).__name__, "python_module": type(index).__module__,
                "is_flat_storage": isinstance(index, faiss.IndexFlat),
                "metric_type": int(index.metric_type), "dimension": int(index.d),
                "vectors": int(index.ntotal), "is_trained": bool(index.is_trained),
                "code_size": int(index.code_size), "faiss_version": faiss.__version__}
    write_json(root.root / "index_structure_observed.json", dict(observed, stage="retrieve"))
    need(observed["is_flat_storage"] and observed["code_size"] == 4 * DIM,
         "INDEX_TYPE_NOT_FLAT_FLOAT32")
    need(observed["metric_type"] == int(faiss.METRIC_INNER_PRODUCT), "INDEX_METRIC_NOT_INNER_PRODUCT")
    need(observed["dimension"] == DIM, "INDEX_DIMENSION_MISMATCH")
    need(observed["vectors"] == N_VECTORS, "INDEX_NTOTAL_MISMATCH")
    need(observed["is_trained"], "INDEX_NOT_TRAINED")
    event("index_structure_validated", python_class=observed["python_class"],
          metric_type=observed["metric_type"], dimension=observed["dimension"],
          vectors=observed["vectors"], code_size=observed["code_size"])
    reader = MetadataReader(str(guard.check(manifest["assets"]["metadata"]["path"])),
                            str(guard.check(manifest["assets"]["offsets"]["path"])),
                            manifest["assets"]["metadata"]["bytes"])
    index_identity = dict(observed, **assets["index"])
    try:
        ordered = sorted(missing)
        chunk = ret["search_chunk"]
        # One encode pass and one batched search per block; never a corpus loop.
        for start in range(0, len(ordered), chunk):
            block = ordered[start:start + chunk]
            texts = [queries[key]["result"]["query"] for key in block]
            with torch.inference_mode():
                vectors = model.encode(texts, batch_size=enc_cfg["batch_size"],
                                       normalize_embeddings=True, device=enc_cfg["device"],
                                       convert_to_numpy=True, show_progress_bar=False,
                                       precision="float32", prompt="")
            need(vectors.shape == (len(block), DIM) and vectors.dtype == np.float32,
                 "ENCODER_OUTPUT_SHAPE_DTYPE")
            need(bool(np.isfinite(vectors).all()), "ENCODER_NONFINITE")
            norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
            need(bool(np.all(np.abs(norms - 1.0) <= 1e-4)), "ENCODER_NOT_UNIT_NORM")
            began = time.monotonic()
            scores, ids = index.search(np.ascontiguousarray(vectors), BUDGET)
            elapsed = time.monotonic() - began
            need(scores.shape == ids.shape == (len(block), BUDGET), "SEARCH_RESULT_SHAPE")
            event("search_block_complete", queries=len(block), budget=BUDGET,
                  seconds=round(elapsed, 3))
            for offset, (qid, arm) in enumerate(block):
                reader.provenance.clear()
                found = search_to_metadata([int(i) for i in ids[offset].tolist()],
                                           [float(s) for s in scores[offset].tolist()],
                                           reader.lookup)
                query_row = queries[qid, arm]
                prediction = {"qid": qid, "arm": arm, "query": query_row["result"]["query"],
                              "candidates": found["candidates"],
                              "ranked_titles": found["ranked_titles"],
                              "boundary_tie_within_returned": found["boundary_tie_within_returned"],
                              "ties_beyond_budget": found["ties_beyond_budget"]}
                record = {
                    "schema": "urbench.bridge_n25.prediction.v1", "pass": "prediction_" + arm,
                    "qid": qid, "arm": arm,
                    "activation_sha256": activation_identity["sha256"], "runner_version": VERSION,
                    "preparation_seal_sha256": manifest["preparation"]["seal_sha256"],
                    "preflight_seal_sha256": manifest["preflight"]["seal_sha256"],
                    "parent_seal_sha256": parent_seal_id["sha256"],
                    "oracle_e_seal_sha256": e_seal_id["sha256"],
                    "query_record": {"pass": query_row["pass"], "qid": query_row["qid"], "arm": arm,
                                     "prompt_sha256": query_row["prompt_sha256"],
                                     "record_sha256": sha256(canonical(query_row)),
                                     "fallback": query_row["result"]["fallback"],
                                     "encoder_tokens": query_row["result"]["encoder_tokens"]},
                    "encoder_identity": encoder_runtime, "index_identity": index_identity,
                    "metadata_identity": assets["metadata"], "offsets_identity": assets["offsets"],
                    "search_budget": BUDGET,
                    "candidate_provenance": [reader.provenance[c["global_row"]]
                                             for c in found["candidates"]],
                    "prediction": prediction, "historical_global_lineage": LINEAGE,
                    "utc": datetime.now(timezone.utc).isoformat()}
                validate_prediction_record(record, arm)
                root.create("prediction_" + arm, qid, record)
                event("record_written", name="retrieve", record_pass="prediction_" + arm, qid=qid,
                      arm=arm, candidates=len(found["candidates"]),
                      ranked_titles=len(found["ranked_titles"]),
                      boundary_tie=found["boundary_tie_within_returned"])
    finally:
        reader.close()

# --------------------------------------------------------------------- main

def manifest_template():
    """Schema skeleton for reviewers. Prints only; writes nothing; not activation."""
    return {"schema": ACTIVATION_SCHEMA, "status": "PROSPECTIVE_ACTIVATION_REVIEWED",
            "protocol": {"path": "<abs>", "sha256": "<64hex>", "bytes": 1},
            "amendment": {"path": "<abs>", "sha256": "<64hex>", "bytes": 1},
            "code": {name: {"path": "<abs>", "sha256": "<64hex>", "bytes": 1} for name in CODE_FILES},
            "preparation": {"seal_path": "<abs>", "seal_sha256": "<64hex>",
                            "artifacts": {name: {"sha256": "<64hex>", "bytes": 1}
                                          for name in EXPORTS}},
            "preflight": {"seal_path": "<abs>", "seal_sha256": "<64hex>"},
            "prompt_preflight": {"seal_path": "<abs>", "seal_sha256": "<64hex>",
                                 "summary_path": "<abs>", "summary_sha256": "<64hex>",
                                 "historical_runner_sha256": "<64hex>",
                                 "checked_records": PROMPT_PREFLIGHT_RECORDS,
                                 "categories_checked": PROMPT_PREFLIGHT_CATEGORIES,
                                 "minimum_remaining_context_margin": 1,
                                 "runtime_dependent_categories": ["query_C", "query_D"]},
            "exports": {name: {"path": "<abs>", "sha256": "<64hex>", "bytes": 1} for name in EXPORTS},
            "models": {"qwen": {"root": "<abs>", "files": {"<relative>": {"sha256": "<64hex>", "bytes": 1}}},
                       "encoder": {"root": "<abs>", "files": {"<relative>": {"sha256": "<64hex>", "bytes": 1}}}},
            "assets": {k: {"path": "<abs>", "sha256": "<64hex>", "bytes": 1} for k in ASSET_KEYS},
            "environment": {"python": "3.10.19", "executable": "<abs>",
                            "packages": {"torch": "2.9.0"}},
            "settings": {"version": VERSION, "seed": SEED,
                         "qwen_load": {"quantization": "|".join(QWEN_LOAD_CHOICES),
                                       "compute_dtype": "bfloat16", "attn_implementation": "sdpa",
                                       "device_map_zero": True, "trust_remote_code": False,
                                       "use_safetensors": True, "local_files_only": True},
                         "decoding": {"do_sample": False, "num_beams": 1,
                                      "state_max_new_tokens": 1024, "query_max_new_tokens": 128,
                                      "microbatch": 1, "model_context_limit": 40960},
                         "encoder": {"device": "cuda:0", "batch_size": 32,
                                     "normalize_embeddings": True, "precision": "float32",
                                     "dtype": "float32", "attn_implementation": "sdpa",
                                     "backend": "torch", "max_seq_length": 128, "prefix": "",
                                     "query_token_cap": 128},
                         "retrieval": {"budget": BUDGET, "search_chunk": 150, "cpu_threads": 8,
                                       "ntotal": N_VECTORS, "dim": DIM, "metric": "INNER_PRODUCT",
                                       "verify_asset_hashes": True},
                         "aggregation": {"max_titles": 10, "cutoffs": [1, 5, 10]},
                         "arms": list(ARMS),
                         "tokenizer": {"chat_template_sha256": "<64hex>",
                                       "tokenizer_class": "Qwen2TokenizerFast",
                                       "apply_chat_template_kwargs": {
                                           "tokenize": False, "add_generation_prompt": True,
                                           "enable_thinking": False},
                                       "tokenize_kwargs": {"add_special_tokens": False,
                                                           "truncation": False},
                                       "prompt_constants": dict(sorted(assert_prompt_hashes().items()))},
                         "statistics": {"primary": "D-A", "followup": "D-P",
                                        "descriptive": ["D-B", "D-C"], "cutoffs": [1, 5, 10],
                                        "effect_threshold_pp": 10, "alpha": 0.05,
                                        "bootstrap_resamples": 20000, "bootstrap_rng": "PCG64",
                                        "bootstrap_seed": SEED, "percentile_method": "linear",
                                        "signflip": "EXACT_INTEGER_DP_QID_LEVEL",
                                        "sampling_unit": "QID"}},
            "roots": {"output_root": "<abs>", "archive_root": "<abs>", "stage_tag": "v1",
                      "score_output_root": "<abs>"},
            "cohort": {"qids": 25, "accepted_pairs": 36, "arms": list(ARMS),
                       "label": "HUMAN_VERIFICATION_OF_MODEL_ASSISTED_CANDIDATES",
                       "predictions": 150},
            "provenance": {"git_parent_head": "<40hex>", "activation_utc": "<iso8601>",
                           "amendment_number": 2, "outcomes_viewed": False},
            "limitations": ["<limitation strings>"],
            "historical_global_lineage": LINEAGE,
            "experiment_state": ACTIVATED_EXPERIMENT_STATE,
            "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED"}

def check_inputs(manifest, activation_identity, for_stage):
    """Read-only. Loads no weights or index; generates, searches and scores nothing.

    Uses the real per-stage guard, so it also demonstrates the input boundary.
    """
    guard = stage_guard(for_stage, manifest)
    guard.allow("preparation_seal", manifest["preparation"]["seal_path"])
    guard.allow("preflight_seal", manifest["preflight"]["seal_path"])
    guard.allow("prompt_preflight_seal", manifest["prompt_preflight"]["seal_path"])
    guard.allow("prompt_preflight_summary", manifest["prompt_preflight"]["summary_path"])
    environment = check_environment(manifest)
    code_identity = verify_code_identity(guard, manifest)
    prep_id, pre_id, prompt_id = authenticate_upstream(guard, manifest)
    counts = {}
    if for_stage == "parent":
        guard.allow("export_parent_only", manifest["exports"][PARENT_EXPORT]["path"])
        rows, _ = read_export(guard, manifest, PARENT_EXPORT, validate_parent_export)
        counts["parent_rows"] = len(rows)
    elif for_stage == "oracle_e":
        guard.allow("export_oracle_e", manifest["exports"][ORACLE_EXPORT]["path"])
        rows, _ = read_export(guard, manifest, ORACLE_EXPORT, validate_oracle_export)
        counts["oracle_rows"] = len(rows)
    tag = manifest["roots"]["stage_tag"]
    for key in ("output_root", "archive_root"):
        for stage in STAGE_ROOT_NAMES:
            candidate = Path(manifest["roots"][key]) / (stage + "_" + tag)
            no_symlink(candidate)
            need(not candidate.exists() and not candidate.is_symlink(),
                 "STAGE_DESTINATION_NOT_FRESH: " + str(candidate))
    return {"for_stage": for_stage, "activation_sha256": activation_identity["sha256"],
            "preparation_seal": prep_id["sha256"], "preflight_seal": pre_id["sha256"],
            "prompt_preflight": prompt_id, "code_identity": code_identity,
            "stage_tag": manifest["roots"]["stage_tag"],
            "score_output_root": manifest["roots"]["score_output_root"],
            "environment": environment, "allowed_inputs": guard.labels(),
            "forbidden_inputs": sorted(STAGE_FORBIDDEN[for_stage]),
            "expected_units": STAGE_UNITS, "arms": list(ARMS),
            "historical_global_lineage": LINEAGE,
            "experiment_state": manifest["experiment_state"],
            "canonical_stage0": manifest["canonical_stage0"],
            "files_written": 0, "weights_loaded": False, "index_loaded": False,
            "pilot_searches": 0, "qwen_generations": 0, "scoring_targets_opened": False,
            **counts}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("parent", "oracle_e", "retrieve", "check-inputs",
                                 "manifest-template"))
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--activation-sha256")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--parent-tag", default="v1")
    parser.add_argument("--oracle-e-tag", default="v1")
    parser.add_argument("--for-stage", choices=tuple(STAGE_UNITS), default="parent")
    args = parser.parse_args()
    if args.stage == "manifest-template":
        print(canonical(manifest_template()))
        return
    need(args.activation is not None and args.activation_sha256 is not None,
         "ACTIVATION_MANIFEST_AND_EXPECTED_SHA256_REQUIRED")
    manifest, identity = load_activation(args.activation, args.activation_sha256)
    if args.stage == "check-inputs":
        event("inputs_checked", **check_inputs(manifest, identity, args.for_stage))
        return
    need(os.environ.get("SLURM_JOB_ID"), "COMPUTE_JOB_REQUIRED")
    need(socket.gethostname().split(".")[0].lower() != "psn001", "LOGIN_NODE_REFUSED")
    need(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "OFFLINE_REQUIRED")
    os.umask(0o077)
    event("stage_begin", name=args.stage, tag=args.tag, job_id=os.environ["SLURM_JOB_ID"],
          node=socket.gethostname(), activation_sha256=identity["sha256"],
          historical_global_lineage=LINEAGE,
          experiment_state=manifest["experiment_state"],
          prompt_preflight_runner_revision_changed=None)
    if args.stage == "parent":
        stage_parent(manifest, identity, args.tag)
    elif args.stage == "oracle_e":
        stage_oracle_e(manifest, identity, args.tag, args.parent_tag)
    else:
        stage_retrieve(manifest, identity, args.tag, args.parent_tag, args.oracle_e_tag)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Operational failure is never a scientific miss. Partial records stay.
        print(canonical({"status": "STOPPED", "error_type": type(exc).__name__, "error": str(exc),
                         "activated_experiment_state": ACTIVATED_EXPERIMENT_STATE,
                         "outcome_produced": False,
                         "historical_global_lineage": LINEAGE,
                         "outputs_must_not_be_deleted": True}), file=sys.stderr)
        sys.exit(2)
