#!/usr/bin/env python3
"""Tokenizer-only prompt preflight, v0.1. Does NOT activate or run the N25 pilot.

Establishes the exact Qwen token length of every pilot prompt whose content is
known before generation, using the pinned tokenizer and the frozen
``bridge_pilot_core.render_prompt`` path -- the identical
``apply_chat_template(..., tokenize=False, add_generation_prompt=True,
enable_thinking=False)`` call the runtime will make.

Six known categories x 25 qids = 150 measured prompts:
    state_C / state_D  reserve 1024      query_A / query_P / query_B / query_E  reserve 128
Every record must satisfy ``input_tokens + reserve <= 40960``.

The C and D *query* prompts are excluded on purpose: their evidence is the
validated model state, which does not exist yet. They are recorded as
RUNTIME_DEPENDENT with their parser bounds, never as a measured length.

Loads the tokenizer once. There is no AutoModel, no weight shard, no torch
model allocation, no BitsAndBytesConfig, no SentenceTransformer, no FAISS, no
index/metadata/offset access, and no scoring-target access anywhere in this
file. Nothing is generated, retrieved or scored.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import socket

from bridge_pilot_core import (CHILD_COUNTS, PilotError, assert_prompt_hashes, canonical,
                               need, oracle_query_messages, parent_query_messages,
                               render_prompt, sha256, state_messages, strict_json)
from bridge_pilot_run import (EXPERIMENT_STATE, LINEAGE, ORACLE_EXPORT, PARENT_EXPORT,
                              PREFLIGHT_STATUS, PREPARATION_STATUS, TARGET_EXPORT,
                              PathGuard, event, jsonl_rows, new_dir, no_symlink,
                              read_small, validate_oracle_export, validate_parent_export,
                              write_json, write_new)

VERSION = "0.1"
REPO = Path("/mnt/home/user41/URBench")
PREP = "outputs/efbpt/bridge_pilot_n25/v1/preparation_r3"
PREP_ARCHIVE = Path("/mnt/home/user41/URBench_pilot_archives/bridge_pilot_n25_preparation_v1_r3")
PREFLIGHT = "outputs/efbpt/bridge_pilot_n25/v1/preflight_v2"
PREFLIGHT_ARCHIVE = Path("/mnt/home/user41/URBench_pilot_archives/bridge_pilot_n25_preflight_v2")
OUT = "outputs/efbpt/bridge_pilot_n25/v1/prompt_preflight_v1"
ARCHIVE = Path("/mnt/home/user41/URBench_pilot_archives/bridge_pilot_n25_prompt_preflight_v1")

PREPARATION_SEAL_HASH = "f976ef54cdec11425cd944e92aea00001b5e3432e19d4d6ad75544b9ba425249"
PREFLIGHT_SEAL_HASH = "35b6eb102ee2335e9298b96847479dc1d871b571dac8e932f6b1542107796359"
CORE_HASH = "0a02ee75ea4b86a501972441db3d6874eeda1f2d4f30c75cf19c18571f7b5a67"
RUNNER_HASH = "9afdf8921b4f5d1dae78aa0546819d233c63d29000cd1c4f1683990d7c2a1748"
PROTOCOL = "docs/EFBPT_BRIDGE_PILOT_N25_PROTOCOL.md"
PROTOCOL_HASH = "c580427ffc53b97253731eb724a1ee8e408f8242d1aba471c30f6d962593450c"

QWEN = Path("/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B")
# Tokenizer and config files only. No weight shard is listed, opened or hashed.
QWEN_TOKENIZER_FILES = {
    "config.json": ("e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a", 728),
    "tokenizer_config.json": ("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", 9732),
    "tokenizer.json": ("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", 11422654),
    "vocab.json": ("ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", 2776833),
    "merges.txt": ("8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", 1671853),
}
EXPORT_IDENTITIES = {
    PARENT_EXPORT: ("460e88faf6dd074c8fbedbc4344ae93b8a8516f35c31c47d0bbc54a13003c326", 1661204),
    ORACLE_EXPORT: ("b3ee497dbd6833a31b93cf5cb8e6e015626be80e16a3b4b2915be6f646af78ad", 9415),
}
# Paths this stage must never be able to register, let alone open.
FORBIDDEN = (
    ("scoring_targets", "outputs/efbpt/bridge_pilot_n25/v1/preparation_r3/scoring_targets.jsonl"),
    ("index", "rag/index/wikipedia_full.index"),
    ("metadata", "rag/index/wikipedia_full_meta.jsonl"),
    ("offsets", "rag/index/wikipedia_full_meta.offsets.npy"),
    ("human_annotation_log", "data/strategyqa_official/efbpt/stage0/human_annotation_log.jsonl"),
    ("dev200", "data/strategyqa_official/dev200_seed4242.jsonl"),
    ("assisted_pass2", "outputs/efbpt/stage0_assisted/assisted_pass2.jsonl"),
    ("assisted_pass3", "outputs/efbpt/stage0_assisted/assisted_pass3.jsonl"),
    ("human_verified_candidates", "outputs/efbpt/stage0_assisted/human_verified_candidates.jsonl"),
    ("d4_extractions", "outputs/efbpt/d4/d4_extractions.jsonl"),
    ("encoder_root", "/mnt/home/user41/downloaded_models/sentence-transformers/"
                     "paraphrase-multilingual-MiniLM-L12-v2"),
)
FORBIDDEN += tuple(("qwen_weight_%02d" % i, QWEN / ("model-%05d-of-00008.safetensors" % i))
                   for i in range(1, 9))

CONTEXT_LIMIT = 40960
STATE_RESERVE = 1024
QUERY_RESERVE = 128
# Ordered measurement plan. Every production loop is bounded by this tuple x 25.
CATEGORIES = (("state_C", "C", STATE_RESERVE), ("state_D", "D", STATE_RESERVE),
              ("query_A", "A", QUERY_RESERVE), ("query_P", "P", QUERY_RESERVE),
              ("query_B", "B", QUERY_RESERVE), ("query_E", "E", QUERY_RESERVE))
# C/D query prompts depend on validated runtime state; recorded, never measured.
RUNTIME_DEPENDENT = {
    "scope": "RUNTIME_DEPENDENT",
    "categories": ["query_C", "query_D"],
    "reserve": QUERY_RESERVE,
    "reason": "Evidence is the validated model state, which does not exist before generation. "
              "No output is fabricated here and no length is claimed as measured.",
    "parser_bounds": {"maximum_accepted_items": 6,
                      "c_item_maximum_whitespace_words": 8,
                      "d_item_maximum_whitespace_words": 40,
                      "evidence_join": "\\n",
                      "empty_state_is_identical_to_query_P": True},
    "lower_bound_note": "An empty or format-invalid state yields exactly the measured query_P "
                        "prompt, so query_P is an exact floor, not an upper bound.",
    "enforcement": "bridge_pilot_core.render_prompt re-checks input_tokens + 128 <= 40960 "
                   "immediately before every C/D query call and stops the stage on failure.",
}

def guard_for(repo):
    forbidden = [(label, repo / p if not str(p).startswith("/") else Path(p))
                 for label, p in FORBIDDEN]
    return PathGuard("prompt_preflight", forbidden)

def small_inputs(repo, guard):
    """Authenticate every protected small input before any export is opened."""
    identities = {}

    def take(label, path, expected=None, expected_bytes=None):
        guard.allow(label, path)
        _, ident = guard.read_small(path, expected)
        need(expected_bytes is None or ident["bytes"] == expected_bytes, "INPUT_BYTES: " + label)
        identities[label] = ident
        return ident

    take("protocol", repo / PROTOCOL, PROTOCOL_HASH)
    source_dir = repo / "eval/error_analysis_tests/efbpt"
    take("core", source_dir / "bridge_pilot_core.py", CORE_HASH)
    take("runner", source_dir / "bridge_pilot_run.py", RUNNER_HASH)
    for name in ("bridge_pilot_prompt_preflight.py", "bridge_pilot_prompt_preflight_test.py",
                 "bridge_pilot_prompt_preflight.sbatch"):
        take("source_" + name, source_dir / name)
    # Preparation and preflight seals, in both their output and archive copies.
    prep = take("preparation_seal", repo / PREP / "PREPARATION_SEAL.json", PREPARATION_SEAL_HASH)
    prep_archived = take("preparation_seal_archived", PREP_ARCHIVE / "PREPARATION_SEAL.json",
                         PREPARATION_SEAL_HASH)
    need(prep["sha256"] == prep_archived["sha256"], "PREPARATION_SEAL_COPIES")
    pre = take("preflight_seal", repo / PREFLIGHT / "PREFLIGHT_SEAL.json", PREFLIGHT_SEAL_HASH)
    pre_archived = take("preflight_seal_archived", PREFLIGHT_ARCHIVE / "PREFLIGHT_SEAL.json",
                        PREFLIGHT_SEAL_HASH)
    need(pre["sha256"] == pre_archived["sha256"], "PREFLIGHT_SEAL_COPIES")
    prep_seal = strict_json((repo / PREP / "PREPARATION_SEAL.json").read_bytes())
    pre_seal = strict_json((repo / PREFLIGHT / "PREFLIGHT_SEAL.json").read_bytes())
    need(prep_seal["status"] == PREPARATION_STATUS, "PREPARATION_STATUS")
    need(pre_seal["status"] == PREFLIGHT_STATUS, "PREFLIGHT_STATUS")
    need(pre_seal["historical_global_lineage"] == LINEAGE, "PREFLIGHT_LINEAGE")
    need(pre_seal["preparation_seal_sha256"] == PREPARATION_SEAL_HASH, "PREFLIGHT_PREPARATION_BINDING")
    # The preparation seal is the authority for the two exports we may read.
    for name, (digest, size) in EXPORT_IDENTITIES.items():
        recorded = prep_seal["artifacts"][name]
        need(recorded["sha256"] == digest and recorded["bytes"] == size,
             "PREPARATION_EXPORT_BINDING: " + name)
    need(prep_seal["artifacts"][TARGET_EXPORT]["sha256"] != "", "TARGET_IDENTITY_PRESENT")
    guard.allow("qwen_root", QWEN)
    for name, (digest, size) in sorted(QWEN_TOKENIZER_FILES.items()):
        take("qwen_" + name, QWEN / name, digest, size)
    config = strict_json((QWEN / "config.json").read_bytes())
    need(config["max_position_embeddings"] == CONTEXT_LIMIT, "MODEL_CONTEXT_LIMIT_DRIFT")
    need(config["model_type"] == "qwen3", "MODEL_TYPE_DRIFT")
    return identities, prep_seal, pre_seal

def load_tokenizer(guard):
    """One tokenizer load. No AutoModel, no weights, no download, no remote code."""
    from transformers import AutoTokenizer
    import transformers
    root = str(guard.check(QWEN))
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True,
                                              trust_remote_code=False)
    need(tokenizer.chat_template, "CHAT_TEMPLATE_MISSING")
    identity = {"root": root, "tokenizer_class": type(tokenizer).__name__,
                "is_fast": bool(tokenizer.is_fast), "vocab_size": int(tokenizer.vocab_size),
                "tokenizer_model_max_length": int(tokenizer.model_max_length),
                "chat_template_sha256": sha256(tokenizer.chat_template),
                "transformers_version": transformers.__version__,
                "apply_chat_template_kwargs": {"tokenize": False, "add_generation_prompt": True,
                                               "enable_thinking": False},
                "tokenize_kwargs": {"add_special_tokens": False, "truncation": False},
                "model_context_limit": CONTEXT_LIMIT,
                "trust_remote_code": False, "local_files_only": True,
                "weights_loaded": False, "weight_files_opened": 0}
    return tokenizer, identity

def measure(tokenizer, messages, reserve):
    """Exact measurement through the frozen renderer. Records, never truncates.

    ``render_prompt`` is the identical call the runtime makes, so the recorded
    prompt hash and token count are the production values. It is invoked with a
    zero reserve first so an over-budget prompt is reported rather than hidden,
    then re-invoked with the real reserve to confirm the frozen check agrees and
    the render is deterministic.
    """
    try:
        base = render_prompt(tokenizer, messages, 0)
    except PilotError:
        return {"input_tokens": None, "prompt_sha256": None, "reserve": reserve,
                "total_tokens": None, "remaining_margin": None,
                "status": "EXCEEDS_CONTEXT_WITHOUT_ANY_RESERVE", "pass": False}
    total = base["input_tokens"] + reserve
    passed = total <= CONTEXT_LIMIT
    if passed:
        confirm = render_prompt(tokenizer, messages, reserve)
        need(confirm["prompt_sha256"] == base["prompt_sha256"]
             and confirm["input_tokens"] == base["input_tokens"], "RENDER_NONDETERMINISM")
    return {"input_tokens": base["input_tokens"], "prompt_sha256": base["prompt_sha256"],
            "reserve": reserve, "total_tokens": total,
            "remaining_margin": CONTEXT_LIMIT - total,
            "status": "WITHIN_CONTEXT" if passed else "EXCEEDS_CONTEXT", "pass": passed}

def messages_for(category, arm, parent_row, oracle_row):
    if category.startswith("state_"):
        return state_messages(parent_row, arm)
    if arm == "E":
        return oracle_query_messages(oracle_row)
    return parent_query_messages(parent_row, arm)

def measure_all(tokenizer, parents, oracles, constants, tokenizer_identity):
    """150 records: six known categories x 25 qids. No corpus or index loop."""
    by_qid = {r["qid"]: r for r in parents}
    oracle_by_qid = {r["qid"]: r for r in oracles}
    records, maxima = [], {}
    for category, arm, reserve in CATEGORIES:
        worst = None
        rank = lambda r: r["input_tokens"] if r["input_tokens"] is not None else CONTEXT_LIMIT * 2
        for qid in sorted(CHILD_COUNTS):
            result = measure(tokenizer, messages_for(category, arm, by_qid[qid],
                                                     oracle_by_qid[qid]), reserve)
            record = {"schema": "urbench.bridge_n25.prompt_length.v1", "category": category,
                      "arm": arm, "qid": qid, "prompt_constants": constants,
                      "tokenizer_sha256": tokenizer_identity["chat_template_sha256"],
                      "context_limit": CONTEXT_LIMIT, **result}
            records.append(record)
            if worst is None or rank(record) > rank(worst):
                worst = record
            event("prompt_measured", category=category, arm=arm, qid=qid,
                  input_tokens=record["input_tokens"], reserve=reserve,
                  total_tokens=record["total_tokens"],
                  remaining_margin=record["remaining_margin"], passed=record["pass"])
        maxima[category] = {"qid": worst["qid"], "arm": arm, "reserve": reserve,
                            "input_tokens": worst["input_tokens"],
                            "total_tokens": worst["total_tokens"],
                            "remaining_margin": worst["remaining_margin"],
                            "status": worst["status"]}
    need(len(records) == 150, "PROMPT_RECORD_COUNT")
    need(len({(r["category"], r["qid"]) for r in records}) == 150, "PROMPT_RECORD_UNIQUENESS")
    return records, maxima

def read_allowed_exports(repo, guard):
    parent_path = guard.allow("export_" + PARENT_EXPORT, repo / PREP / PARENT_EXPORT)
    oracle_path = guard.allow("export_" + ORACLE_EXPORT, repo / PREP / ORACLE_EXPORT)
    identities, rows = {}, {}
    for label, path, validator in ((PARENT_EXPORT, parent_path, validate_parent_export),
                                   (ORACLE_EXPORT, oracle_path, validate_oracle_export)):
        digest, size = EXPORT_IDENTITIES[label]
        raw, ident = guard.read_small(path, digest)
        need(ident["bytes"] == size, "EXPORT_BYTES: " + label)
        parsed = jsonl_rows(raw)
        validator(parsed)
        need({r["qid"] for r in parsed} == set(CHILD_COUNTS) and len(parsed) == 25,
             "EXPORT_COHORT: " + label)
        rows[label], identities[label] = parsed, ident
    return rows[PARENT_EXPORT], rows[ORACLE_EXPORT], identities

def check_small_inputs(repo, load_tokenizer_too=True):
    """Read-only. Opens no target, no weight shard, no index. Writes nothing."""
    guard = guard_for(repo)
    identities, prep_seal, pre_seal = small_inputs(repo, guard)
    constants = assert_prompt_hashes()
    parents, oracles, export_identities = read_allowed_exports(repo, guard)
    tokenizer_identity = None
    if load_tokenizer_too:
        _, tokenizer_identity = load_tokenizer(guard)
    for dest in (repo / OUT, ARCHIVE):
        no_symlink(dest)
        need(not dest.exists() and not dest.is_symlink(), "DESTINATION_MUST_BE_FRESH: " + str(dest))
        need(dest.parent.is_dir() and os.access(dest.parent, os.W_OK | os.X_OK),
             "DESTINATION_PARENT_NOT_READY: " + str(dest.parent))
    return {"version": VERSION, "parent_rows": len(parents), "oracle_rows": len(oracles),
            "expected_prompt_records": 25 * len(CATEGORIES),
            "categories": [c[0] for c in CATEGORIES],
            "reserves": {c[0]: c[2] for c in CATEGORIES},
            "context_limit": CONTEXT_LIMIT,
            "prompt_constants": constants, "small_inputs": len(identities),
            "export_identities": export_identities, "tokenizer": tokenizer_identity,
            "preparation_seal_sha256": PREPARATION_SEAL_HASH,
            "preflight_seal_sha256": PREFLIGHT_SEAL_HASH,
            "runtime_dependent": RUNTIME_DEPENDENT,
            "allowed_inputs": guard.labels(),
            "forbidden_inputs": sorted(label for label, _ in FORBIDDEN),
            "files_written": 0, "qwen_weight_files_opened": 0, "qwen_generations": 0,
            "pilot_searches": 0, "scoring_targets_opened": False,
            "index_loaded": False, "encoder_loaded": False,
            "experiment_state": EXPERIMENT_STATE, "historical_global_lineage": LINEAGE,
            "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED"}

def measure_only(repo):
    """Dry run: exact measurement, aggregate report, zero persistent output."""
    guard = guard_for(repo)
    small_inputs(repo, guard)
    constants = assert_prompt_hashes()
    parents, oracles, _ = read_allowed_exports(repo, guard)
    tokenizer, tokenizer_identity = load_tokenizer(guard)
    records, maxima = measure_all(tokenizer, parents, oracles, constants, tokenizer_identity)
    failed = [{"category": r["category"], "qid": r["qid"], "status": r["status"]}
              for r in records if not r["pass"]]
    return {"checked_records": len(records), "maxima": maxima, "failed": failed,
            "all_within_context": not failed, "context_limit": CONTEXT_LIMIT,
            "tokenizer": tokenizer_identity, "runtime_dependent": RUNTIME_DEPENDENT,
            "files_written": 0, "qwen_weight_files_opened": 0, "qwen_generations": 0,
            "pilot_searches": 0, "scoring_targets_opened": False,
            "experiment_state": EXPERIMENT_STATE, "historical_global_lineage": LINEAGE}

def preflight(repo):
    """Sealed run. Fresh exclusive destinations; no overwrite and no resume."""
    need(os.environ.get("SLURM_JOB_ID"), "COMPUTE_JOB_REQUIRED")
    need(socket.gethostname().split(".")[0].lower() != "psn001", "LOGIN_NODE_REFUSED")
    need(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "OFFLINE_REQUIRED")
    guard = guard_for(repo)
    identities, prep_seal, pre_seal = small_inputs(repo, guard)
    constants = assert_prompt_hashes()
    parents, oracles, export_identities = read_allowed_exports(repo, guard)
    tokenizer, tokenizer_identity = load_tokenizer(guard)
    os.umask(0o077)
    out = repo / OUT
    for dest in (out, ARCHIVE):
        no_symlink(dest)
        need(not dest.exists() and not dest.is_symlink(),
             "FRESH_DESTINATION_REQUIRED_NO_OVERWRITE_NO_RESUME: " + str(dest))
    new_dir(ARCHIVE)
    new_dir(out)
    artifacts = {}

    def save(name, obj):
        identity = write_json(out / name, obj)
        need(write_json(ARCHIVE / name, obj) == identity, "ARCHIVE_OUTPUT_MISMATCH: " + name)
        artifacts[name] = identity
        return identity

    save("prompt_preflight_start.json", {
        "schema": "urbench.bridge_n25.prompt_preflight_start.v1", "version": VERSION,
        "utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
        "node": socket.gethostname(), "categories": [c[0] for c in CATEGORIES],
        "reserves": {c[0]: c[2] for c in CATEGORIES}, "context_limit": CONTEXT_LIMIT,
        "expected_prompt_records": 25 * len(CATEGORIES),
        "output_directory": str(out), "archive_directory": str(ARCHIVE),
        "resume_policy": "NO_RESUME; NO_OVERWRITE; PRESERVE_PARTIAL_FILES_ON_FAILURE",
        "experiment_state": EXPERIMENT_STATE, "historical_global_lineage": LINEAGE})
    archived_inputs = {}
    for label, ident in sorted(identities.items()):
        raw, _ = read_small(ident["path"])
        if len(raw) <= 4 * 1024**2:
            archived_inputs[label] = write_new(ARCHIVE / ("input_" + label.replace("/", "_")), raw)
    save("input_identities.json", {
        "schema": "urbench.bridge_n25.prompt_preflight_inputs.v1",
        "small_inputs": identities, "exports": export_identities,
        "archived_copies": archived_inputs,
        "large_inputs": "Qwen tokenizer.json/vocab.json/merges.txt and the two runtime exports "
                        "are recorded by hash and path only; not copied into the archive",
        "qwen_weight_files_opened": 0, "scoring_targets_opened": False})
    records, maxima = measure_all(tokenizer, parents, oracles, constants, tokenizer_identity)
    save("prompt_lengths.json", {
        "schema": "urbench.bridge_n25.prompt_lengths.v1", "context_limit": CONTEXT_LIMIT,
        "records": records, "maxima": maxima,
        "content_recorded": "NONE; identifiers, token counts and hashes only",
        "runtime_dependent": RUNTIME_DEPENDENT})
    failed = [{"category": r["category"], "qid": r["qid"], "status": r["status"],
               "input_tokens": r["input_tokens"]} for r in records if not r["pass"]]
    summary = {"schema": "urbench.bridge_n25.prompt_preflight_summary.v1",
               "status": "OUTCOME_FREE_PROMPT_PREFLIGHT_COMPLETE" if not failed
                         else "PROMPT_CONTEXT_BLOCKER",
               "checked_records": len(records), "categories_checked": len(CATEGORIES),
               "qids": 25, "accepted_pairs": 36, "maxima": maxima, "failed": failed,
               "all_known_prompts_within_context": not failed,
               "context_limit": CONTEXT_LIMIT,
               "tokenizer": tokenizer_identity, "prompt_constants": constants,
               "runtime_dependent": RUNTIME_DEPENDENT,
               "qwen_weight_files_opened": 0, "qwen_generations": 0, "pilot_searches": 0,
               "scoring_targets_opened": False, "index_loaded": False, "encoder_loaded": False,
               "experiment_state": EXPERIMENT_STATE, "historical_global_lineage": LINEAGE,
               "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED",
               "output_directory": str(out), "archive_directory": str(ARCHIVE)}
    save("prompt_preflight_summary.json", summary)
    # Rehash every protected small input before sealing.
    for label, ident in sorted(identities.items()):
        read_small(ident["path"], ident["sha256"])
    for label, ident in sorted(export_identities.items()):
        read_small(ident["path"], ident["sha256"])
    seal = {"schema": "urbench.bridge_n25.prompt_preflight_seal.v1",
            "status": "SEALED_OUTCOME_FREE_PROMPT_PREFLIGHT_NOT_ACTIVATION",
            "version": VERSION,
            "preparation_seal_sha256": PREPARATION_SEAL_HASH,
            "preflight_seal_sha256": PREFLIGHT_SEAL_HASH,
            "protocol_sha256": PROTOCOL_HASH, "core_sha256": CORE_HASH,
            "runner_sha256": RUNNER_HASH,
            "script_sha256": identities["source_bridge_pilot_prompt_preflight.py"]["sha256"],
            "test_sha256": identities["source_bridge_pilot_prompt_preflight_test.py"]["sha256"],
            "job_sha256": identities["source_bridge_pilot_prompt_preflight.sbatch"]["sha256"],
            "tokenizer": tokenizer_identity,
            "chat_template_sha256": tokenizer_identity["chat_template_sha256"],
            "prompt_constants": constants, "artifacts": artifacts,
            "checked_records": len(records), "categories_checked": len(CATEGORIES),
            "maxima": maxima, "all_known_prompts_within_context": not failed,
            "runtime_dependent": RUNTIME_DEPENDENT,
            "qwen_weight_files_opened": 0, "qwen_generations": 0, "pilot_searches": 0,
            "scoring_targets_opened": False,
            "experiment_state": EXPERIMENT_STATE, "historical_global_lineage": LINEAGE,
            "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED"}
    seal_identity = write_json(out / "PROMPT_PREFLIGHT_SEAL.json", seal)
    need(write_json(ARCHIVE / "PROMPT_PREFLIGHT_SEAL.json", seal) == seal_identity,
         "SEAL_COPIES_NOT_BYTE_IDENTICAL")
    event("prompt_preflight_complete", seal_sha256=seal_identity["sha256"], **summary)
    # A blocker is reported after the evidence is durably sealed, never before.
    need(not failed, "PROMPT_CONTEXT_BLOCKER; see prompt_lengths.json; do not truncate, "
                     "reduce reserves, change prompts or drop a qid")

def self_test():
    """In-memory only. No tokenizer, no export, no file created."""
    checks = []

    def check(name, value):
        need(value, "SELF_TEST: " + name)
        checks.append(name)

    def rejects(name, fn):
        try:
            fn()
        except (PilotError, ValueError, KeyError):
            checks.append(name)
            return
        raise PilotError("SELF_TEST_EXPECTED_REJECTION: " + name)

    check("category_plan", [c[0] for c in CATEGORIES] ==
          ["state_C", "state_D", "query_A", "query_P", "query_B", "query_E"])
    check("reserves", {c[0]: c[2] for c in CATEGORIES} ==
          {"state_C": 1024, "state_D": 1024, "query_A": 128, "query_P": 128,
           "query_B": 128, "query_E": 128})
    check("expected_records", 25 * len(CATEGORIES) == 150)
    check("context_limit", CONTEXT_LIMIT == 40960)
    check("cohort", len(CHILD_COUNTS) == 25 and sum(CHILD_COUNTS.values()) == 36)
    check("prompt_constants", assert_prompt_hashes())
    check("runtime_dependent_scope", RUNTIME_DEPENDENT["scope"] == "RUNTIME_DEPENDENT"
          and RUNTIME_DEPENDENT["parser_bounds"]["maximum_accepted_items"] == 6)
    guard = guard_for(REPO)
    for label, path in FORBIDDEN:
        rejects("forbidden_" + label,
                lambda p=path: guard.allow("sneaky", REPO / p if not str(p).startswith("/") else p))
    check("no_weight_file_allowed", not any("safetensors" in k for k in guard.allowed))

    class FakeTokenizer:
        def __init__(self, per_message):
            self.per_message = per_message
            self.chat_template = "synthetic"
            self.calls = []

        def apply_chat_template(self, messages, **kw):
            need(kw == {"tokenize": False, "add_generation_prompt": True,
                        "enable_thinking": False}, "SELF_TEST_CHAT_TEMPLATE_KWARGS")
            self.calls.append(len(messages))
            return "R" * self.per_message

        def __call__(self, text, **kw):
            need(kw == {"add_special_tokens": False, "truncation": False},
                 "SELF_TEST_TOKENIZE_KWARGS")
            return {"input_ids": list(range(len(text)))}

    ok = measure(FakeTokenizer(100), [], 128)
    check("within_context", ok["pass"] and ok["input_tokens"] == 100
          and ok["total_tokens"] == 228 and ok["remaining_margin"] == 40960 - 228)
    edge = measure(FakeTokenizer(CONTEXT_LIMIT - 1024), [], 1024)
    check("exact_boundary_passes", edge["pass"] and edge["remaining_margin"] == 0)
    over = measure(FakeTokenizer(CONTEXT_LIMIT - 1023), [], 1024)
    check("one_over_fails", not over["pass"] and over["status"] == "EXCEEDS_CONTEXT"
          and over["remaining_margin"] == -1)
    huge = measure(FakeTokenizer(CONTEXT_LIMIT + 1), [], 128)
    check("over_without_reserve", not huge["pass"]
          and huge["status"] == "EXCEEDS_CONTEXT_WITHOUT_ANY_RESERVE"
          and huge["input_tokens"] is None)
    counting = FakeTokenizer(100)
    measure(counting, [], 128)
    check("render_confirmed_twice_when_passing", counting.calls == [0, 0])
    event("self_test_complete", checks_passed=len(checks), checks=checks, files_written=0,
          tokenizer_loaded=False, exports_opened=False, qwen_weight_files_opened=0,
          qwen_generations=0, pilot_searches=0, scoring_targets_opened=False,
          experiment_state=EXPERIMENT_STATE, historical_global_lineage=LINEAGE)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--check-small-inputs", action="store_true")
    mode.add_argument("--measure-only", action="store_true",
                      help="Exact measurement, aggregate report, no persistent output.")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.check_small_inputs:
        event("small_inputs_checked", **check_small_inputs(no_symlink(args.repo)))
    elif args.measure_only:
        event("measure_only_complete", **measure_only(no_symlink(args.repo)))
    else:
        preflight(no_symlink(args.repo))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # A preflight failure is never a scientific miss. Partial artifacts stay.
        print(canonical({"status": "STOPPED", "error_type": type(exc).__name__, "error": str(exc),
                         "experiment_state": EXPERIMENT_STATE,
                         "historical_global_lineage": LINEAGE,
                         "outputs_must_not_be_deleted": True}), file=sys.stderr)
        sys.exit(2)
