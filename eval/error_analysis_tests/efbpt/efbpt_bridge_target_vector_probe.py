#!/usr/bin/env python3
"""EFBPT bridge pilot N=25 target-vector and score-consistency diagnostic.

Post-outcome diagnostic over the FROZEN N=25 assisted bridge pilot. It reads
sealed pilot artifacts, the sealed corpus assets and the frozen encoder, and
writes only into its own new output directory. It never rewrites, repairs or
rescores the pilot, and it never proposes or tests a repair.

Five stages:
  T1  target-row inventory        one streaming metadata pass, all exact-title rows
  T2  target vector alignment     stored index vector vs freshly encoded document
  T3  query-score reproduction    stored candidate scores vs recomputed scores
  T4  target score vs boundaries  raw top-100 and aggregated top-10 boundaries
  T5  title self-retrieval        diagnostic-only oracle control

Scope limits that must survive into every summary: results are cohort specific.
A pass over the 1046 accepted-target rows is NOT a global vector-to-metadata
lineage proof; historical_global_lineage remains UNESTABLISHED for all
23,963,971 vectors.
"""

import argparse
import base64
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "urbench.bridge_n25.target_vector_probe.v1"
VERSION = "1.0"

# ---------------------------------------------------------------- anchors

# Amendment 1: the frozen pilot anchor and the diagnostic code identity are
# separate facts. The pilot anchor is verified from the activation and seals.
# The repository HEAD at run time is RECORDED, never required to equal it,
# because this diagnostic may itself be committed before execution.
FROZEN_PILOT_HEAD = "3de27683e9577ff6db5eed13057c21c3fa9aa9d0"

OUTPUT_RELATIVE = "outputs/efbpt/bridge_pilot_n25/target_vector_probe_v1"

# Predeclared before execution, numerically equal to the frozen maximum L2
# document-alignment threshold, and never relaxed after observing differences.
#
# What this tolerance is NOT: it is not a proof of any bound between the
# original query embedding produced during the frozen retrieval run and the
# query embedding recomputed here. No distance between those two query vectors
# has ever been established, and the document-alignment threshold says nothing
# about them. T3 is therefore an EMPIRICAL score-reproduction check over the
# 15,000 stored candidate-score occurrences, not a derived bound.
#
# A difference above this tolerance is a SUSPECTED INCONSISTENCY under the
# declared tolerance. It is not proof of global index corruption, and must not
# be reported as such.
SCORE_ABS_TOLERANCE = 0.001

# Amendment 2: reused verbatim from bridge_pilot_preflight.py SETTINGS.
MIN_COSINE = 0.99999
MAX_L2_DISTANCE = 0.001
MAX_UNIT_NORM_ERROR = 0.0001

# Frozen document-encoder settings (bridge_pilot_preflight.py SETTINGS).
ENCODER_INPUT_FORMULA = "metadata.title + '. ' + metadata.text"
ENCODER_BATCH_SIZE = 32
ENCODER_MAX_SEQ_LENGTH = 128
ENCODER_DEVICE = "cuda:0"
ENCODER_PREFIX = ""

N_VECTORS = 23963971
DIM = 384
BUDGET = 100
EXPECTED_QIDS = 25
EXPECTED_TARGETS = 36
EXPECTED_TARGET_ROWS = 1046
EXPECTED_PREDICTIONS = 150
EXPECTED_OCCURRENCES = 15000
EXPECTED_UNIQUE_CANDIDATE_ROWS = 8061
EXPECTED_METADATA_ROWS = 23963971
EXPECTED_METADATA_BYTES = 25866666236
ARMS = ("A", "P", "B", "C", "D", "E")

MAX_METADATA_LINE = 8 * 1024 ** 2

# Terminal statuses (Amendment 9).
OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE"
INPUT_DRIFT = "INPUT_DRIFT"
TARGET_ALIGNMENT_FAILURE = "TARGET_ALIGNMENT_FAILURE"
SCORE_REPRODUCTION_FAILURE = "SCORE_REPRODUCTION_FAILURE"
DIAGNOSTIC_COMPLETE = "DIAGNOSTIC_COMPLETE"

# Sealed pilot artifacts pinned by hash in code. Asset and encoder hashes are
# taken from the activation manifest once the manifest itself is verified, so
# they are not duplicated here.
PINNED = {
    "activation": ("docs/EFBPT_BRIDGE_PILOT_N25_ACTIVATION.json",
                   "bcc684cfd3c91ae26839a664a390529d21e40ffa6faea4bd407081829c5dbfe9", 13927),
    "core": ("eval/error_analysis_tests/efbpt/bridge_pilot_core.py",
             "0a02ee75ea4b86a501972441db3d6874eeda1f2d4f30c75cf19c18571f7b5a67", 23584),
    "scoring_targets": ("outputs/efbpt/bridge_pilot_n25/v1/preparation_r3/scoring_targets.jsonl",
                        "5643935822c0a0227a975047b64b23d748eb61b77fa8eecf824f1deab026373b", 6628),
    "preparation_seal": ("outputs/efbpt/bridge_pilot_n25/v1/preparation_r3/PREPARATION_SEAL.json",
                         "f976ef54cdec11425cd944e92aea00001b5e3432e19d4d6ad75544b9ba425249", 1826),
    "predictions": ("outputs/efbpt/bridge_pilot_n25/v1/retrieve_v2/predictions.jsonl",
                    "4604b8354a066fcafbef6e63f7c73328ac0de13e45b44b6c65104308f793311a", 1384225),
    "prediction_records": ("outputs/efbpt/bridge_pilot_n25/v1/retrieve_v2/prediction_records.jsonl",
                           "556fef6357c52a5d3ec3b732a60731ecbbda0b920f53ce3aec29dba4099f5ab2", 3976517),
    "retrieval_seal": ("outputs/efbpt/bridge_pilot_n25/v1/retrieve_v2/STAGE_SEAL.json",
                       "d3ef88f42a911dac9b0a26804f1b16bba797a3a73c11377b526607dc4eaeae15", 1078),
    "scores": ("outputs/efbpt/bridge_pilot_n25/v1/score_v2/scores.json",
               "7233ca175194573791b289fb9cc09cae015dc8c4b72596c9ea93e3bcb4afc332", 24478),
    "scoring_seal": ("outputs/efbpt/bridge_pilot_n25/v1/score_v2/SCORING_SEAL.json",
                     "b62b09a07aaa7809b61e1a63bf42b92a3a9b364080d65c183c03c66a6b83c792", 4862),
    "parent_seal": ("outputs/efbpt/bridge_pilot_n25/v1/parent_v2/STAGE_SEAL.json",
                    "490f5a989f1c8bc7ce95037e4507309360b37b847ff9d27ced5834c2fc683270", 1057),
    "oracle_e_seal": ("outputs/efbpt/bridge_pilot_n25/v1/oracle_e_v2/STAGE_SEAL.json",
                      "cbba8b6eeec9a0faf5dea7669db9e18e178144c4276fea17b7420948b802324f", 1017),
    "preflight_seal": ("outputs/efbpt/bridge_pilot_n25/v1/preflight_v2/PREFLIGHT_SEAL.json",
                       "35b6eb102ee2335e9298b96847479dc1d871b571dac8e932f6b1542107796359", 1330),
    "vector_alignment": ("outputs/efbpt/bridge_pilot_n25/v1/preflight_v2/vector_alignment.json",
                         "8a8d481e42cef734bbf0f8c2925891f199d0ebfcded9d92264a646d37f04e03e", 479076),
    "selected_rows": ("outputs/efbpt/bridge_pilot_n25/v1/preflight_v2/selected_rows.json",
                      "9705de8504f5924450325affabf53367f67bddaa5b7e5bcaca13837968af19fe", 16277),
    "reachability": ("outputs/efbpt/bridge_pilot_n25/target_reachability_diagnostic_v1/reachability.json",
                     "4918f65f8340013577a6be6feb8bffacc3609e29b38d748029a5afad603a845d", 25663),
    "reachability_seal": ("outputs/efbpt/bridge_pilot_n25/target_reachability_diagnostic_v1/DIAGNOSTIC_SEAL.json",
                          "0c1a774dd9c44c526c705d5fad6d6b25d02de6fca6a2f730323c94a9d17ecabd", 3397),
}

# Query-stage artifacts whose hashes live inside their own stage seals.
QUERY_ARTIFACTS = {
    "parent_queries": ("outputs/efbpt/bridge_pilot_n25/v1/parent_v2/queries.jsonl",
                       "parent_seal", "queries.jsonl"),
    "oracle_queries": ("outputs/efbpt/bridge_pilot_n25/v1/oracle_e_v2/queries_e.jsonl",
                       "oracle_e_seal", "queries_e.jsonl"),
}


class ProbeError(RuntimeError):
    """Diagnostic-level failure carrying a terminal status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def need(condition, status, message):
    if not condition:
        raise ProbeError(status, message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(stage, **fields):
    payload = {"stage": stage, "utc": utc_now()}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False), flush=True)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path, chunk=1 << 22, progress_label=None, progress_seconds=60.0):
    digest = hashlib.sha256()
    read = 0
    last = time.monotonic()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if progress_label and time.monotonic() - last >= progress_seconds:
                last = time.monotonic()
                event("asset_hash_progress_not_a_result", asset=progress_label, bytes_read=read)
    return digest.hexdigest(), read


def git(repo, *args):
    """Read-only git invocation. Never uses a shell and never writes refs.

    Returns (returncode, stdout, stderr) or None when the git binary is not
    present, which is the case on the compute nodes of this cluster."""
    try:
        result = subprocess.run(["git", "-C", str(repo), *args],
                                check=False, capture_output=True, text=True)
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def submission_snapshot():
    """Login-node git provenance injected by the wrapper. Returns None when it
       is absent or malformed; a malformed snapshot is never silently used."""
    head = os.environ.get("TVP_GIT_HEAD", "").strip()
    status_b64 = os.environ.get("TVP_GIT_STATUS_B64")
    utc = os.environ.get("TVP_GIT_SNAPSHOT_UTC", "").strip()
    if not head or status_b64 is None or not utc:
        return None
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        return None
    try:
        status = base64.b64decode(status_b64.encode("ascii"), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    lines = [line for line in status.splitlines() if line.strip()]
    return {"head": head, "utc": utc, "status_porcelain": lines,
            "tracked_files_clean": not [l for l in lines if not l.startswith("??")],
            "untracked_entries": [l for l in lines if l.startswith("??")],
            "status_sha256": sha256_text(status)}


# ------------------------------------------------------- output plumbing

class OutputRoot:
    """Every probe write lands here. Writes are atomic and never overwrite."""

    def __init__(self, directory):
        self.dir = directory
        self.written = {}

    def resolve(self, name):
        need("/" not in name and name not in ("", ".", ".."),
             OPERATIONAL_FAILURE, "ARTIFACT_NAME_REFUSED: " + name)
        target = (self.dir / name).resolve()
        root = self.dir.resolve()
        need(target.parent == root, OPERATIONAL_FAILURE,
             "WRITE_TARGET_OUTSIDE_OUTPUT_DIRECTORY: " + str(target))
        return target

    def write(self, name, raw):
        target = self.resolve(name)
        need(not target.exists(), OPERATIONAL_FAILURE, "REFUSING_OVERWRITE: " + str(target))
        temp = self.resolve(name + ".tmp-" + str(os.getpid()))
        need(not temp.exists(), OPERATIONAL_FAILURE, "TEMP_ALREADY_EXISTS: " + str(temp))
        with open(temp, "xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        directory = os.open(str(self.dir), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.written[name] = {"bytes": len(raw), "sha256": sha256_bytes(raw)}
        event("artifact_written", artifact=name, bytes=len(raw),
              sha256=self.written[name]["sha256"])
        return self.written[name]

    def write_json(self, name, obj):
        return self.write(name, (canonical(obj) + "\n").encode("utf-8"))

    def write_jsonl(self, name, rows):
        body = "".join(canonical(row) + "\n" for row in rows)
        return self.write(name, body.encode("utf-8"))


# ----------------------------------------------------------- guard stages

def load_core(repo):
    path = repo / PINNED["core"][0]
    digest, size = sha256_file(path)
    need(digest == PINNED["core"][1] and size == PINNED["core"][2], INPUT_DRIFT,
         "CORE_CODE_DRIFT: " + str(path))
    spec = importlib.util.spec_from_file_location("bridge_pilot_core_frozen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(module.ARMS == ARMS and module.N_VECTORS == N_VECTORS and module.DIM == DIM
         and module.BUDGET == BUDGET, INPUT_DRIFT, "CORE_CONSTANT_DRIFT")
    return module


def repository_state(repo):
    """Record diagnostic code identity; the pilot anchor is established by the
       activation and seal hashes, never by the current HEAD.

       Git provenance is runtime-verified when the git binary exists. On this
       cluster the compute nodes have no git, so the wrapper injects a
       login-node snapshot which is recorded as SUBMISSION_TIME provenance and
       never claimed as runtime-verified. If neither source is available the
       probe aborts rather than proceeding without provenance."""
    snapshot = submission_snapshot()
    head_result = git(repo, "rev-parse", "HEAD")
    status_result = git(repo, "status", "--porcelain")
    runtime = None
    if head_result is not None and status_result is not None:
        code, head, _ = head_result
        status_code, status_out, _ = status_result
        if code == 0 and len(head) == 40 and status_code == 0:
            lines = [line for line in status_out.splitlines() if line.strip()]
            anchor = git(repo, "cat-file", "-e", FROZEN_PILOT_HEAD + "^{commit}")
            runtime = {"head": head, "status_porcelain": lines,
                       "tracked_files_clean": not [l for l in lines if not l.startswith("??")],
                       "untracked_entries": [l for l in lines if l.startswith("??")],
                       "frozen_pilot_head_present_in_repository":
                           (anchor[0] == 0 if anchor is not None else None)}

    need(runtime is not None or snapshot is not None, OPERATIONAL_FAILURE,
         "GIT_PROVENANCE_UNAVAILABLE: no runtime git and no valid submission snapshot")

    chosen = runtime if runtime is not None else snapshot
    source = ("RUNTIME_VERIFIED" if runtime is not None
              else "SUBMISSION_TIME_SNAPSHOT_NOT_RUNTIME_VERIFIED")
    state = {"frozen_pilot_head": FROZEN_PILOT_HEAD,
             "frozen_pilot_head_present_in_repository":
                 chosen.get("frozen_pilot_head_present_in_repository"),
             "diagnostic_code_head": chosen["head"],
             "diagnostic_code_head_equals_frozen_pilot_head":
                 chosen["head"] == FROZEN_PILOT_HEAD,
             "git_status_porcelain": chosen["status_porcelain"],
             "tracked_files_clean": chosen["tracked_files_clean"],
             "untracked_entries": chosen["untracked_entries"],
             "git_provenance_source": source,
             "runtime_git_available": runtime is not None,
             "submission_snapshot_present": snapshot is not None,
             "submission_snapshot": snapshot,
             "note": ("A later commit of this diagnostic changes diagnostic_code_head only. "
                      "It is not a modification of the frozen pilot, whose identity is "
                      "established by the activation and seal hashes verified below.")}
    if runtime is not None and snapshot is not None:
        state["snapshot_agrees_with_runtime_head"] = (snapshot["head"] == runtime["head"])
        state["snapshot_agrees_with_runtime_status"] = (
            snapshot["status_porcelain"] == runtime["status_porcelain"])
    event("git_provenance", source=source, head=state["diagnostic_code_head"],
          tracked_files_clean=state["tracked_files_clean"])
    return state


def verify_pinned_inputs(repo, manifest_holder):
    identities = {}
    for name, (relative, expected, size) in PINNED.items():
        path = repo / relative
        need(path.is_file(), INPUT_DRIFT, "MISSING_INPUT: " + relative)
        digest, actual = sha256_file(path)
        need(digest == expected and actual == size, INPUT_DRIFT,
             "INPUT_HASH_DRIFT: " + relative)
        identities[name] = {"path": str(path), "bytes": actual, "sha256": digest}
        if name == "activation":
            manifest_holder["manifest"] = json.loads(path.read_text(encoding="utf-8"))
    for name, (relative, seal_key, artifact_key) in QUERY_ARTIFACTS.items():
        path = repo / relative
        need(path.is_file(), INPUT_DRIFT, "MISSING_INPUT: " + relative)
        seal = json.loads((repo / PINNED[seal_key][0]).read_text(encoding="utf-8"))
        spec = seal["artifacts"][artifact_key]
        digest, actual = sha256_file(path)
        need(digest == spec["sha256"] and actual == spec["bytes"], INPUT_DRIFT,
             "QUERY_ARTIFACT_DRIFT: " + relative)
        identities[name] = {"path": str(path), "bytes": actual, "sha256": digest}
    return identities


def verify_assets(manifest, skip_index_hash):
    """Corpus assets. Metadata is hashed by the T1 stream, not here."""
    identities = {}
    for key in ("offsets", "index"):
        spec = manifest["assets"][key]
        path = Path(spec["path"])
        need(path.is_file(), INPUT_DRIFT, "MISSING_ASSET: " + str(path))
        size = path.stat().st_size
        need(size == spec["bytes"], INPUT_DRIFT, "ASSET_BYTES_DRIFT: " + str(path))
        if key == "index" and skip_index_hash:
            identities[key] = {"path": str(path), "bytes": size,
                               "sha256": spec["sha256"], "hash_verified": False,
                               "hash_skipped_reason": "operator passed --skip-index-hash"}
            continue
        digest, _ = sha256_file(path, progress_label=key)
        need(digest == spec["sha256"], INPUT_DRIFT, "ASSET_HASH_DRIFT: " + str(path))
        identities[key] = {"path": str(path), "bytes": size, "sha256": digest,
                           "hash_verified": True}
    meta = manifest["assets"]["metadata"]
    meta_path = Path(meta["path"])
    need(meta_path.is_file(), INPUT_DRIFT, "MISSING_ASSET: " + str(meta_path))
    need(meta_path.stat().st_size == meta["bytes"] == EXPECTED_METADATA_BYTES,
         INPUT_DRIFT, "METADATA_BYTES_DRIFT")
    identities["metadata"] = {"path": str(meta_path), "bytes": meta["bytes"],
                              "sha256": meta["sha256"],
                              "hash_verified_by": "T1 full-stream scan"}
    encoder = manifest["models"]["encoder"]
    root = Path(encoder["root"])
    need(root.is_dir(), INPUT_DRIFT, "MISSING_ENCODER_ROOT: " + str(root))
    files = {}
    for name, spec in sorted(encoder["files"].items()):
        path = root / name
        need(path.is_file(), INPUT_DRIFT, "MISSING_ENCODER_FILE: " + name)
        digest, size = sha256_file(path)
        need(digest == spec["sha256"] and size == spec["bytes"], INPUT_DRIFT,
             "ENCODER_FILE_DRIFT: " + name)
        files[name] = {"bytes": size, "sha256": digest}
    identities["encoder"] = {"root": str(root), "files": files}
    return identities


def verify_environment(manifest):
    declared = manifest["environment"]
    observed = {"python": platform.python_version(), "packages": {}}
    need(observed["python"] == declared["python"], INPUT_DRIFT,
         "PYTHON_VERSION_DRIFT: observed " + observed["python"])
    import importlib.metadata as metadata
    for package, version in sorted(declared["packages"].items()):
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            raise ProbeError(INPUT_DRIFT, "PACKAGE_MISSING: " + package)
        need(found == version, INPUT_DRIFT,
             "PACKAGE_VERSION_DRIFT: " + package + " observed " + found)
        observed["packages"][package] = found
    return observed


# --------------------------------------------------------------- T1

def stage_t1(core, manifest, targets, reachability, out):
    """One streaming metadata pass. Collects every exact-title row for all
       36 accepted targets, re-hashing the whole stream as it goes."""
    meta_path = Path(manifest["assets"]["metadata"]["path"])
    wanted = {t["normalized_title"]: t for t in targets}
    collected = {title: [] for title in wanted}
    digest = hashlib.sha256()
    offset = 0
    rows = 0
    started = time.monotonic()
    last = started
    with open(meta_path, "rb") as handle:
        for raw in handle:
            digest.update(raw)
            need(raw.endswith(b"\n"), OPERATIONAL_FAILURE, "METADATA_LINE_BOUNDARY: row=%d" % rows)
            need(len(raw) <= MAX_METADATA_LINE, OPERATIONAL_FAILURE,
                 "METADATA_ROW_TOO_LONG: row=%d" % rows)
            obj = core.strict_json(raw)
            core.exact_keys(obj, ("title", "text"), "METADATA")
            title = core.norm(obj["title"])
            if title in wanted:
                collected[title].append({
                    "qid": wanted[title]["qid"],
                    "normalized_title": title,
                    "raw_title": obj["title"],
                    "source_instance_id": wanted[title]["source_instance_id"],
                    "global_row": rows,
                    "byte_offset": offset,
                    "metadata_line_sha256": sha256_bytes(raw),
                    "encoder_input_sha256": sha256_text(obj["title"] + ". " + obj["text"]),
                    "text": obj["text"],
                })
            offset += len(raw)
            rows += 1
            need(rows <= EXPECTED_METADATA_ROWS and offset <= EXPECTED_METADATA_BYTES,
                 OPERATIONAL_FAILURE, "METADATA_BOUNDARY_EXCEEDED")
            if rows % 10000 == 0 and time.monotonic() - last >= 60.0:
                last = time.monotonic()
                event("t1_scan_progress_not_a_result", rows=rows, bytes_read=offset)
    elapsed = time.monotonic() - started
    need(rows == EXPECTED_METADATA_ROWS, INPUT_DRIFT, "METADATA_ROW_COUNT_DRIFT")
    need(offset == EXPECTED_METADATA_BYTES, INPUT_DRIFT, "METADATA_BYTE_COUNT_DRIFT")
    need(digest.hexdigest() == manifest["assets"]["metadata"]["sha256"], INPUT_DRIFT,
         "METADATA_STREAM_HASH_DRIFT")

    per_target = []
    rows_out = []
    for target in targets:
        title = target["normalized_title"]
        found = sorted(collected[title], key=lambda r: r["global_row"])
        declared = reachability[title]
        need(found, TARGET_ALIGNMENT_FAILURE, "TARGET_TITLE_ABSENT: " + title)
        need(len(found) == declared["matching_rows"], INPUT_DRIFT,
             "TARGET_ROW_COUNT_DRIFT: " + title)
        need(found[0]["global_row"] == declared["first_row"], INPUT_DRIFT,
             "TARGET_FIRST_ROW_DRIFT: " + title)
        need(found[0]["byte_offset"] == declared["first_byte_offset"], INPUT_DRIFT,
             "TARGET_FIRST_OFFSET_DRIFT: " + title)
        distinct_inputs = {r["encoder_input_sha256"] for r in found}
        distinct_raw_titles = sorted({r["raw_title"] for r in found})
        per_target.append({"qid": target["qid"], "normalized_title": title,
                           "rows": len(found), "first_row": found[0]["global_row"],
                           "first_byte_offset": found[0]["byte_offset"],
                           "distinct_encoder_inputs": len(distinct_inputs),
                           "distinct_raw_titles": distinct_raw_titles,
                           "duplicate_normalized_title_rows": len(found) - len(distinct_inputs)})
        rows_out.extend(found)
    need(len(rows_out) == EXPECTED_TARGET_ROWS, INPUT_DRIFT,
         "TARGET_ROW_TOTAL_DRIFT: %d" % len(rows_out))
    out.write_jsonl("target_rows.jsonl", rows_out)
    event("t1_complete", rows_scanned=rows, seconds=round(elapsed, 3),
          target_rows=len(rows_out), targets=len(targets))
    return rows_out, {"rows_scanned": rows, "bytes": offset, "seconds": round(elapsed, 3),
                      "metadata_sha256": digest.hexdigest(), "target_rows": len(rows_out),
                      "per_target": per_target}


# --------------------------------------------------------------- T2

def configure_runtime(manifest, core):
    """Reproduce the determinism setup of the frozen call sites verbatim:
       bridge_pilot_run.py:1140-1146 and bridge_pilot_preflight.py:363-368."""
    import numpy as np
    import torch
    import faiss
    threads = int(manifest["settings"]["retrieval"]["cpu_threads"])
    need(torch.cuda.is_available(), OPERATIONAL_FAILURE, "GPU_REQUIRED")
    torch.set_num_threads(threads)
    faiss.omp_set_num_threads(threads)
    torch.manual_seed(core.SEED)
    torch.cuda.manual_seed_all(core.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(core.SEED)
    return {"cpu_threads": threads, "seed": core.SEED,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "faiss_version": faiss.__version__,
            "faiss_threads": faiss.omp_get_max_threads(),
            "torch_threads": torch.get_num_threads(),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}


def load_encoder(manifest):
    """Construction copied from the frozen call sites, not from the manifest:
       bridge_pilot_run.py:1148-1155 and bridge_pilot_preflight.py:384-389."""
    import torch
    from sentence_transformers import SentenceTransformer
    root = manifest["models"]["encoder"]["root"]
    cfg = manifest["settings"]["encoder"]
    model = SentenceTransformer(root, device=cfg["device"], backend=cfg["backend"],
                                local_files_only=True, trust_remote_code=False,
                                model_kwargs={"use_safetensors": True, "dtype": torch.float32,
                                              "attn_implementation": cfg["attn_implementation"]})
    model.eval()
    need(model.max_seq_length == cfg["max_seq_length"] == ENCODER_MAX_SEQ_LENGTH,
         INPUT_DRIFT, "ENCODER_MAX_SEQ_LENGTH")
    need(model.get_sentence_embedding_dimension() == DIM, INPUT_DRIFT, "ENCODER_DIMENSION")
    need(all(p.dtype == torch.float32 for p in model.parameters()), INPUT_DRIFT,
         "ENCODER_DTYPE")
    need(getattr(model, "default_prompt_name", None) is None, INPUT_DRIFT,
         "ENCODER_DEFAULT_PROMPT")
    # Observed runtime properties, read back off the live object.
    devices = sorted({str(p.device) for p in model.parameters()})
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    identity = {"root": root,
                "observed_max_seq_length": int(model.max_seq_length),
                "observed_dimension": int(model.get_sentence_embedding_dimension()),
                "observed_parameter_devices": devices,
                "observed_parameter_dtypes": dtypes,
                "observed_default_prompt_name": getattr(model, "default_prompt_name", None),
                "observed_training_mode": bool(model.training),
                "declared_device": cfg["device"], "declared_backend": cfg["backend"],
                "declared_attn_implementation": cfg["attn_implementation"],
                "declared_dtype": cfg["dtype"], "declared_precision": cfg["precision"],
                "declared_batch_size": cfg["batch_size"], "declared_prefix": cfg["prefix"],
                "encode_kwargs": {"batch_size": cfg["batch_size"],
                                  "normalize_embeddings": True, "device": cfg["device"],
                                  "convert_to_numpy": True, "show_progress_bar": False,
                                  "precision": "float32", "prompt": ENCODER_PREFIX},
                "encoder_input": ENCODER_INPUT_FORMULA,
                "use_safetensors": True, "local_files_only": True,
                "trust_remote_code": False}
    need(cfg["prefix"] == ENCODER_PREFIX, INPUT_DRIFT, "ENCODER_PREFIX_DRIFT")
    need(int(cfg["batch_size"]) == ENCODER_BATCH_SIZE, INPUT_DRIFT, "ENCODER_BATCH_DRIFT")
    event("encoder_loaded", **{k: identity[k] for k in
                               ("observed_max_seq_length", "observed_dimension",
                                "observed_parameter_devices", "observed_parameter_dtypes")})
    return model, identity


def encode(model, texts, cfg):
    """encode() call copied from bridge_pilot_run.py:1191-1196 and
       bridge_pilot_preflight.py:399-401, including torch.inference_mode."""
    import numpy as np
    import torch
    with torch.inference_mode():
        vectors = model.encode(texts, batch_size=cfg["batch_size"],
                               normalize_embeddings=True, device=cfg["device"],
                               convert_to_numpy=True, show_progress_bar=False,
                               precision="float32", prompt=ENCODER_PREFIX)
    need(vectors.shape == (len(texts), DIM) and vectors.dtype == np.float32,
         OPERATIONAL_FAILURE, "ENCODER_OUTPUT_SHAPE_DTYPE")
    need(bool(np.isfinite(vectors).all()), OPERATIONAL_FAILURE, "ENCODER_NONFINITE")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    need(bool(np.all(np.abs(norms - 1.0) <= 1e-4)), OPERATIONAL_FAILURE,
         "ENCODER_NOT_UNIT_NORM")
    return vectors


def load_index(manifest):
    """Structure validation copied from bridge_pilot_run.py:1163-1176."""
    import faiss
    path = manifest["assets"]["index"]["path"]
    event("index_load_started", path=path)
    index = faiss.read_index(path)
    structure = {"python_class": type(index).__name__,
                 "python_module": type(index).__module__,
                 "is_flat_storage": isinstance(index, faiss.IndexFlat),
                 "metric_type": int(index.metric_type), "dimension": int(index.d),
                 "vectors": int(index.ntotal), "is_trained": bool(index.is_trained),
                 "code_size": int(index.code_size),
                 "faiss_version": faiss.__version__}
    need(structure["is_flat_storage"] and structure["code_size"] == 4 * DIM, INPUT_DRIFT,
         "INDEX_TYPE_NOT_FLAT_FLOAT32")
    need(structure["metric_type"] == int(faiss.METRIC_INNER_PRODUCT), INPUT_DRIFT,
         "INDEX_METRIC_NOT_INNER_PRODUCT")
    need(structure["dimension"] == DIM, INPUT_DRIFT, "INDEX_DIMENSION_MISMATCH")
    need(structure["vectors"] == N_VECTORS, INPUT_DRIFT, "INDEX_NTOTAL_MISMATCH")
    need(structure["is_trained"], INPUT_DRIFT, "INDEX_NOT_TRAINED")
    event("index_load_complete", **structure)
    return index, structure


def reconstruct(index, rows):
    import numpy as np
    out = np.empty((len(rows), DIM), dtype=np.float32)
    for position, row in enumerate(rows):
        out[position] = index.reconstruct(int(row))
    need(bool(np.isfinite(out).all()), OPERATIONAL_FAILURE, "RECONSTRUCT_NONFINITE")
    return out


def stage_t2(index, model, cfg, target_rows, out):
    """Every one of the 1046 rows, frozen gates, no exclusions.
       Alignment failures are findings; this stage never aborts on them."""
    import numpy as np
    records = []
    failed_titles = set()
    for start in range(0, len(target_rows), ENCODER_BATCH_SIZE):
        batch = target_rows[start:start + ENCODER_BATCH_SIZE]
        texts = [r["raw_title"] + ". " + r["text"] for r in batch]
        for row, text in zip(batch, texts):
            need(sha256_text(text) == row["encoder_input_sha256"], OPERATIONAL_FAILURE,
                 "ENCODER_INPUT_HASH_MISMATCH: row=%d" % row["global_row"])
        fresh = encode(model, texts, cfg)
        stored = reconstruct(index, [r["global_row"] for r in batch])
        for position, row in enumerate(batch):
            a = stored[position].astype(np.float64)
            b = fresh[position].astype(np.float64)
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            need(na > 0.0 and nb > 0.0, OPERATIONAL_FAILURE,
                 "ZERO_NORM_VECTOR: row=%d" % row["global_row"])
            cosine = float(np.dot(a, b) / (na * nb))
            l2 = float(np.linalg.norm(a - b))
            unit_norm_error = max(abs(na - 1.0), abs(nb - 1.0))
            passed = (cosine >= MIN_COSINE and l2 <= MAX_L2_DISTANCE
                      and unit_norm_error <= MAX_UNIT_NORM_ERROR)
            if not passed:
                failed_titles.add(row["normalized_title"])
            records.append({"qid": row["qid"], "normalized_title": row["normalized_title"],
                            "global_row": row["global_row"], "byte_offset": row["byte_offset"],
                            "encoder_input_sha256": row["encoder_input_sha256"],
                            "cosine": cosine, "l2_distance": l2,
                            "stored_norm": na, "fresh_norm": nb,
                            "unit_norm_error": unit_norm_error,
                            "max_abs_difference": float(np.max(np.abs(a - b))),
                            "pass": passed})
        if (start // ENCODER_BATCH_SIZE) % 5 == 0:
            event("t2_progress_not_a_result", rows_checked=start + len(batch),
                  rows_total=len(target_rows))
    out.write_jsonl("target_alignment.jsonl", records)
    summary = {"rows_checked": len(records),
               "rows_passed": sum(1 for r in records if r["pass"]),
               "rows_failed": sum(1 for r in records if not r["pass"]),
               "targets_with_any_failed_row": sorted(failed_titles),
               "thresholds": {"minimum_cosine": MIN_COSINE,
                              "maximum_l2_distance": MAX_L2_DISTANCE,
                              "maximum_unit_norm_error": MAX_UNIT_NORM_ERROR},
               "scope": "ACCEPTED_TARGET_ROWS_ONLY; NOT_GLOBAL_LINEAGE_PROOF",
               "policy": "No automatic relaxation, repair, rebuilding or row exclusion"}
    event("t2_complete", **{k: summary[k] for k in ("rows_checked", "rows_passed", "rows_failed")})
    return records, failed_titles, summary


# --------------------------------------------------------------- T3

def stage_t3(index, model, cfg, predictions, sealed_queries, out):
    """Empirical score reproduction. Reconstruct each unique candidate row once
       and compare all 15,000 stored occurrences against recomputed inner
       products under the predeclared SCORE_ABS_TOLERANCE. Differences above
       tolerance are suspected inconsistencies, not proof of corruption."""
    import numpy as np
    unique_rows = sorted({c["global_row"] for p in predictions for c in p["candidates"]})
    need(len(unique_rows) == EXPECTED_UNIQUE_CANDIDATE_ROWS, INPUT_DRIFT,
         "UNIQUE_CANDIDATE_ROW_DRIFT: %d" % len(unique_rows))
    position_of = {row: i for i, row in enumerate(unique_rows)}
    vectors = reconstruct(index, unique_rows)

    texts, keys = [], []
    for prediction in predictions:
        key = (prediction["qid"], prediction["arm"])
        sealed = sealed_queries.get(key)
        need(sealed is not None, INPUT_DRIFT, "SEALED_QUERY_MISSING: %s %s" % key)
        need(sealed == prediction["query"], INPUT_DRIFT,
             "QUERY_TEXT_NOT_BYTE_IDENTICAL: %s %s" % key)
        texts.append(prediction["query"])
        keys.append(key)
    query_vectors = encode(model, texts, cfg)

    rows, occurrences, exceeding_total = [], 0, 0
    global_max = 0.0
    all_differences = []
    for position, prediction in enumerate(predictions):
        query_vector = query_vectors[position].astype(np.float64)
        differences, worst = [], None
        for rank, candidate in enumerate(prediction["candidates"], 1):
            stored_score = float(candidate["score"])
            recomputed = float(np.dot(query_vector,
                                      vectors[position_of[candidate["global_row"]]].astype(np.float64)))
            difference = abs(recomputed - stored_score)
            differences.append(difference)
            if worst is None or difference > worst["difference"]:
                worst = {"difference": difference, "raw_rank": rank,
                         "global_row": candidate["global_row"],
                         "stored_score": stored_score, "recomputed_score": recomputed}
        occurrences += len(differences)
        exceeding = sum(1 for d in differences if d > SCORE_ABS_TOLERANCE)
        exceeding_total += exceeding
        global_max = max(global_max, worst["difference"])
        all_differences.extend(differences)
        rows.append({"qid": prediction["qid"], "arm": prediction["arm"],
                     "query_sha256": sha256_text(prediction["query"]),
                     "byte_identity_confirmed": True,
                     "occurrences_checked": len(differences),
                     "unique_rows_checked": len({c["global_row"] for c in prediction["candidates"]}),
                     "max_abs_score_difference": worst["difference"],
                     "mean_abs_score_difference": float(sum(differences) / len(differences)),
                     "count_exceeding_tolerance": exceeding,
                     "worst_raw_rank": worst["raw_rank"],
                     "worst_global_row": worst["global_row"],
                     "worst_stored_score": worst["stored_score"],
                     "worst_recomputed_score": worst["recomputed_score"],
                     "within_tolerance": exceeding == 0})
    need(occurrences == EXPECTED_OCCURRENCES, INPUT_DRIFT,
         "OCCURRENCE_COUNT_DRIFT: %d" % occurrences)
    out.write_jsonl("query_score_controls.jsonl", rows)
    ordered = sorted(all_differences)
    quantile = lambda q: ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]
    summary = {"occurrences_checked": occurrences,
               "unique_rows_checked": len(unique_rows),
               "tolerance": SCORE_ABS_TOLERANCE,
               "tolerance_declared": "before execution, in START.json",
               "count_exceeding_tolerance": exceeding_total,
               "global_max_abs_difference": global_max,
               "quantiles_abs_difference": {"p50": quantile(0.50), "p90": quantile(0.90),
                                            "p99": quantile(0.99), "p100": ordered[-1]},
               "reproduced": exceeding_total == 0}
    event("t3_complete", occurrences=occurrences, unique_rows=len(unique_rows),
          exceeding=exceeding_total, global_max=global_max)
    return query_vectors, summary


# --------------------------------------------------------------- T4

def untruncated_title_rank(core, candidates, normalized_title):
    """Distinct-title ordering identical to aggregate_candidates but without
       the [:10] truncation. Read-only re-derivation; core is not modified."""
    best = {}
    for candidate in candidates:
        title = core.norm(candidate["title"])
        previous = best.get(title)
        if (previous is None or candidate["score"] > previous["score"]
                or (candidate["score"] == previous["score"]
                    and candidate["global_row"] < previous["best_global_row"])):
            best[title] = {"normalized_title": title, "score": candidate["score"],
                           "best_global_row": candidate["global_row"]}
    ordered = sorted(best.values(),
                     key=lambda r: (-r["score"], r["normalized_title"], r["best_global_row"]))
    for rank, item in enumerate(ordered, 1):
        if item["normalized_title"] == normalized_title:
            return rank, len(ordered)
    return None, len(ordered)


def stage_t4(core, index, query_vectors, predictions, targets_by_qid, rows_by_title,
             failed_titles, out):
    import numpy as np
    vectors_by_title = {}
    for title, rows in rows_by_title.items():
        vectors_by_title[title] = (rows, reconstruct(index, [r["global_row"] for r in rows]))

    records = []
    for position, prediction in enumerate(predictions):
        qid, arm = prediction["qid"], prediction["arm"]
        query_vector = query_vectors[position].astype(np.float64)
        candidates = prediction["candidates"]
        ranked = prediction["ranked_titles"]
        raw_rank100 = float(candidates[BUDGET - 1]["score"])
        title_rank10 = float(ranked[9]["score"]) if len(ranked) >= 10 else None
        for target in targets_by_qid[qid]:
            title = target["normalized_title"]
            rows, matrix = vectors_by_title[title]
            scores = matrix.astype(np.float64) @ query_vector
            best_position = int(np.argmax(scores))
            max_score = float(scores[best_position])
            best_row = rows[best_position]["global_row"]

            stored_positions = [i for i, c in enumerate(candidates)
                                if core.norm(c["title"]) == title]
            present_in_pool = bool(stored_positions)
            stored_rank = stored_positions[0] + 1 if present_in_pool else None
            stored_score = (max(float(candidates[i]["score"]) for i in stored_positions)
                            if present_in_pool else None)
            stored_title_rank = next((i + 1 for i, t in enumerate(ranked)
                                      if t["normalized_title"] == title), None)
            untruncated_rank, distinct_titles = untruncated_title_rank(core, candidates, title)

            # (a) recomputed score of the best STORED target chunk vs its stored score.
            stored_best_row = None
            stored_chunk_reproduced = None
            missing_expected_row = False
            if present_in_pool:
                stored_best_row = min(candidates[i]["global_row"] for i in stored_positions
                                      if float(candidates[i]["score"]) == stored_score)
                position_in_rows = next((k for k, r in enumerate(rows)
                                         if r["global_row"] == stored_best_row), None)
                if position_in_rows is None:
                    # The pool holds a chunk with this exact title that T1 never
                    # collected. That is an inconsistency, never an agreement.
                    missing_expected_row = True
                    stored_chunk_reproduced = False
                else:
                    stored_chunk_reproduced = bool(
                        abs(float(scores[position_in_rows]) - stored_score)
                        <= SCORE_ABS_TOLERANCE)

            # (b) all-chunk maximum vs the best stored score. Kept separate so an
            # omitted higher-scoring chunk cannot be read as an aggregation defect.
            all_max_minus_stored = (max_score - stored_score) if present_in_pool else None
            all_max_agrees = (bool(abs(all_max_minus_stored) <= SCORE_ABS_TOLERANCE)
                              if all_max_minus_stored is not None else None)
            pooled_rows = {c["global_row"] for c in candidates}
            best_chunk_in_pool = bool(best_row in pooled_rows)

            # (c) omitted chunks scoring above the stored raw rank-100 boundary.
            omitted = [(rows[k]["global_row"], float(scores[k])) for k in range(len(rows))
                       if rows[k]["global_row"] not in pooled_rows]
            omitted_above = [row for row, score in omitted
                             if score - raw_rank100 > SCORE_ABS_TOLERANCE]
            max_omitted_score = max((score for _, score in omitted), default=None)

            margin_raw = max_score - raw_rank100
            margin_title = (max_score - title_rank10) if title_rank10 is not None else None
            # Ranking/aggregation consistency uses the best STORED score, never the
            # all-chunk maximum, so an omitted chunk cannot masquerade as a
            # ranking defect.
            margin_stored_title = (stored_score - title_rank10
                                   if (present_in_pool and title_rank10 is not None) else None)
            alignment_failed = title in failed_titles

            if alignment_failed:
                consistency = "TARGET_ALIGNMENT_FAILED"
            elif missing_expected_row:
                consistency = "MISSING_EXPECTED_TARGET_ROW"
            elif present_in_pool and stored_chunk_reproduced is False:
                consistency = "STORED_CHUNK_SCORE_NOT_REPRODUCED"
            elif omitted_above:
                consistency = "OMITTED_TARGET_CHUNK_ABOVE_BOUNDARY"
            elif present_in_pool and all_max_agrees is False:
                consistency = "ALL_CHUNK_MAX_EXCEEDS_STORED_BEST"
            else:
                consistency = "CONSISTENT"

            if alignment_failed:
                candidate_status = "TARGET_ALIGNMENT_FAILED"
            elif present_in_pool:
                candidate_status = "PRESENT_IN_STORED_TOP100"
            elif abs(margin_raw) <= SCORE_ABS_TOLERANCE:
                candidate_status = "WITHIN_TOP100_BOUNDARY_TOLERANCE_UNRESOLVED"
            elif margin_raw > SCORE_ABS_TOLERANCE:
                candidate_status = "ABOVE_BOUNDARY_BUT_ABSENT"
            else:
                candidate_status = "BELOW_STORED_TOP100_BOUNDARY"

            if alignment_failed:
                ranking_status = "TARGET_ALIGNMENT_FAILED"
            elif not present_in_pool:
                ranking_status = "NOT_APPLICABLE_NOT_IN_TOP100"
            elif stored_title_rank is not None:
                ranking_status = "PRESENT_IN_STORED_TOP10"
            elif title_rank10 is None:
                ranking_status = "PRESENT_IN_STORED_TOP10"
            elif abs(margin_stored_title) <= SCORE_ABS_TOLERANCE:
                ranking_status = "WITHIN_TOP10_BOUNDARY_TOLERANCE_UNRESOLVED"
            elif margin_stored_title > SCORE_ABS_TOLERANCE:
                ranking_status = "ABOVE_TOP10_BOUNDARY_BUT_ABSENT"
            else:
                ranking_status = "BELOW_STORED_TOP10_BOUNDARY"

            records.append({
                "qid": qid, "arm": arm, "normalized_title": title,
                "raw_target_title": target["title"],
                "source_instance_id": target["source_instance_id"],
                "target_rows": len(rows),
                "max_target_score": max_score,
                "best_target_global_row": best_row,
                "raw_rank100_score": raw_rank100,
                "margin_to_raw_rank100": margin_raw,
                "title_rank10_score": title_rank10,
                "margin_to_title_rank10": margin_title,
                "margin_stored_best_to_title_rank10": margin_stored_title,
                "stored_best_target_raw_rank": stored_rank,
                "stored_untruncated_distinct_title_rank": untruncated_rank,
                "distinct_titles_in_pool": distinct_titles,
                "stored_top10_title_rank": stored_title_rank,
                "stored_target_score": stored_score,
                "stored_best_target_global_row": stored_best_row,
                "stored_chunk_score_reproduced": stored_chunk_reproduced,
                "all_target_max_minus_stored_best": all_max_minus_stored,
                "all_target_max_agrees_with_stored_best": all_max_agrees,
                "best_target_chunk_present_in_stored_pool": best_chunk_in_pool,
                "omitted_target_chunks_above_boundary_count": len(omitted_above),
                "omitted_target_chunks_above_boundary_rows": sorted(omitted_above),
                "maximum_omitted_target_score": max_omitted_score,
                "missing_expected_target_row": missing_expected_row,
                "target_alignment_status": ("FAILED" if alignment_failed else "PASSED"),
                "candidate_status": candidate_status,
                "ranking_status": ranking_status,
                "retrieval_consistency_status": consistency,
            })
    need(len(records) == EXPECTED_TARGETS * len(ARMS), OPERATIONAL_FAILURE,
         "T4_RECORD_COUNT: %d" % len(records))
    out.write_jsonl("target_score_margins.jsonl", records)
    counts = {}
    for record in records:
        counts[record["candidate_status"]] = counts.get(record["candidate_status"], 0) + 1
    event("t4_complete", records=len(records), **counts)
    return records, {
        "records": len(records),
        "membership_findings": {
            "candidate_status_counts": counts,
            "ranking_status_counts": {
                status: sum(1 for r in records if r["ranking_status"] == status)
                for status in sorted({r["ranking_status"] for r in records})}},
        "consistency_findings": {
            "retrieval_consistency_status_counts": {
                status: sum(1 for r in records if r["retrieval_consistency_status"] == status)
                for status in sorted({r["retrieval_consistency_status"] for r in records})},
            "records_with_omitted_chunk_above_boundary":
                sum(1 for r in records if r["omitted_target_chunks_above_boundary_count"] > 0),
            "records_with_stored_chunk_not_reproduced":
                sum(1 for r in records if r["stored_chunk_score_reproduced"] is False),
            "records_with_missing_expected_row":
                sum(1 for r in records if r["missing_expected_target_row"])},
        "note": ("Ranks below 100 are unobservable; only margins are reported. Membership "
                 "statuses and consistency findings are deliberately separate: ranking_status "
                 "derives from the best STORED target score, while the all-chunk maximum is a "
                 "retrieval diagnostic only, so an omitted higher-scoring chunk is never "
                 "attributed to aggregation.")}


# --------------------------------------------------------------- T5

class MetadataReader:
    """Bounded offset-seek reader; read-only, mirrors the frozen runner."""

    def __init__(self, core, metadata_path, offsets_path, metadata_bytes):
        import numpy as np
        self.core = core
        self.size = metadata_bytes
        self.offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        need(self.offsets.ndim == 1 and len(self.offsets) == N_VECTORS, INPUT_DRIFT,
             "OFFSETS_SHAPE")
        need(self.offsets.dtype.kind in "iu" and self.offsets.dtype.itemsize == 8,
             INPUT_DRIFT, "OFFSETS_DTYPE")
        need(int(self.offsets[0]) == 0 and 0 <= int(self.offsets[-1]) < self.size,
             INPUT_DRIFT, "OFFSETS_ENDPOINTS")
        self.stream = open(metadata_path, "rb")

    def close(self):
        self.stream.close()

    def lookup(self, row):
        need(type(row) is int and 0 <= row < N_VECTORS, OPERATIONAL_FAILURE,
             "LOOKUP_ROW_RANGE")
        offset = int(self.offsets[row])
        end = int(self.offsets[row + 1]) if row + 1 < N_VECTORS else self.size
        need(0 <= offset < end <= self.size and end - offset <= MAX_METADATA_LINE,
             OPERATIONAL_FAILURE, "LOOKUP_OFFSET_BOUNDS")
        if offset:
            self.stream.seek(offset - 1)
            need(self.stream.read(1) == b"\n", OPERATIONAL_FAILURE, "OFFSET_NOT_LINE_START")
        self.stream.seek(offset)
        raw = self.stream.read(end - offset)
        need(len(raw) == end - offset and raw.endswith(b"\n") and raw.count(b"\n") == 1,
             OPERATIONAL_FAILURE, "METADATA_LINE_BOUNDARY")
        obj = self.core.strict_json(raw)
        self.core.exact_keys(obj, ("title", "text"), "METADATA")
        return obj


def stage_t5(core, index, model, cfg, targets, failed_titles, reader, out):
    """Diagnostic-only oracle control. Never used to alter a pilot score."""
    import numpy as np
    titles = [t["title"] for t in targets]
    vectors = encode(model, titles, cfg)
    records = []
    for position, target in enumerate(targets):
        title = target["normalized_title"]
        query = np.ascontiguousarray(vectors[position].reshape(1, DIM), dtype=np.float32)
        scores, ids = index.search(query, BUDGET)
        row_ids = [int(i) for i in ids[0]]
        row_scores = [float(s) for s in scores[0]]
        core.validate_search_arrays(row_ids, row_scores)
        candidates = []
        for row, score in zip(row_ids, row_scores):
            obj = reader.lookup(row)
            candidates.append({"global_row": row, "score": score, "title": obj["title"]})
        aggregated = core.aggregate_candidates(candidates)
        raw_positions = [i + 1 for i, c in enumerate(candidates)
                         if core.norm(c["title"]) == title]
        best_raw_rank = raw_positions[0] if raw_positions else None
        best_score = (max(c["score"] for c in candidates if core.norm(c["title"]) == title)
                      if raw_positions else None)
        aggregated_rank = next((i + 1 for i, a in enumerate(aggregated)
                                if a["normalized_title"] == title), None)
        untruncated_rank, distinct_titles = untruncated_title_rank(core, candidates, title)
        records.append({
            "qid": target["qid"], "normalized_title": title, "raw_target_title": target["title"],
            "best_exact_title_raw_chunk_rank": best_raw_rank,
            "exact_title_aggregated_distinct_title_rank": aggregated_rank,
            "exact_title_untruncated_distinct_title_rank": untruncated_rank,
            "distinct_titles_in_pool": distinct_titles,
            "hit_at_raw_1": bool(best_raw_rank is not None and best_raw_rank <= 1),
            "hit_at_raw_5": bool(best_raw_rank is not None and best_raw_rank <= 5),
            "hit_at_raw_10": bool(best_raw_rank is not None and best_raw_rank <= 10),
            "hit_at_raw_100": bool(best_raw_rank is not None),
            "hit_at_title_1": bool(aggregated_rank is not None and aggregated_rank <= 1),
            "hit_at_title_5": bool(aggregated_rank is not None and aggregated_rank <= 5),
            "hit_at_title_10": bool(aggregated_rank is not None and aggregated_rank <= 10),
            "exact_title_best_score": best_score,
            "top1_raw_returned_title": candidates[0]["title"],
            "top1_aggregated_title": aggregated[0]["normalized_title"] if aggregated else None,
            "target_alignment_status": ("FAILED" if title in failed_titles else "PASSED"),
            "oracle_scope": "DIAGNOSTIC_ONLY_NOT_A_METHOD_RESULT",
        })
    out.write_jsonl("title_self_retrieval.jsonl", records)
    summary = {"targets": len(records),
               "hit_at_raw_1": sum(1 for r in records if r["hit_at_raw_1"]),
               "hit_at_raw_10": sum(1 for r in records if r["hit_at_raw_10"]),
               "hit_at_raw_100": sum(1 for r in records if r["hit_at_raw_100"]),
               "hit_at_title_1": sum(1 for r in records if r["hit_at_title_1"]),
               "hit_at_title_10": sum(1 for r in records if r["hit_at_title_10"]),
               "targets_with_failed_alignment": sorted(failed_titles),
               "oracle_scope": "DIAGNOSTIC_ONLY_NOT_A_METHOD_RESULT",
               "interpretation_limit": (
                   "Amendment 7: if T2 alignment passed, self-retrieval failure must NOT be "
                   "attributed to title truncation, because the title heads the encoder input "
                   "'title + \". \" + text'. Supported readings are limited to weak dense-encoder "
                   "title matching, competition from other corpus vectors, duplicate or ambiguous "
                   "title behaviour, or an insufficient document-vector representation. If T2 "
                   "alignment failed, treat self-retrieval as an infrastructure or document "
                   "alignment issue and draw no encoder-quality conclusion.")}
    event("t5_complete", **{k: summary[k] for k in
                            ("targets", "hit_at_raw_1", "hit_at_raw_100", "hit_at_title_10")})
    return records, summary


# --------------------------------------------------------------- driver

def build_inputs(repo, core, identities):
    targets = []
    for line in (repo / PINNED["scoring_targets"][0]).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for child in row["children"]:
            targets.append({"qid": row["qid"], "title": child["title"],
                            "normalized_title": child["normalized_title"],
                            "source_instance_id": child["source_instance_id"]})
    need(len(targets) == EXPECTED_TARGETS, INPUT_DRIFT, "TARGET_COUNT_DRIFT")
    need(len({t["qid"] for t in targets}) == EXPECTED_QIDS, INPUT_DRIFT, "QID_COUNT_DRIFT")
    for target in targets:
        need(core.norm(target["title"]) == target["normalized_title"], INPUT_DRIFT,
             "TARGET_NORMALIZATION_DRIFT: " + target["title"])

    reach = json.loads((repo / PINNED["reachability"][0]).read_text(encoding="utf-8"))
    need(len(reach["targets"]) == EXPECTED_TARGETS, INPUT_DRIFT, "REACHABILITY_TARGET_DRIFT")
    reachability = {t["normalized_title"]: t for t in reach["targets"]}
    need(set(reachability) == {t["normalized_title"] for t in targets}, INPUT_DRIFT,
         "REACHABILITY_TITLE_SET_DRIFT")
    declared_rows = sum(t["matching_rows"] for t in reach["targets"])
    need(declared_rows == EXPECTED_TARGET_ROWS, INPUT_DRIFT,
         "REACHABILITY_ROW_TOTAL_DRIFT: %d" % declared_rows)

    predictions = [json.loads(line) for line in
                   (repo / PINNED["predictions"][0]).read_text(encoding="utf-8").splitlines()]
    need(len(predictions) == EXPECTED_PREDICTIONS, INPUT_DRIFT, "PREDICTION_COUNT_DRIFT")
    core.validate_predictions(predictions)
    need({(p["qid"], p["arm"]) for p in predictions}
         == {(t["qid"], a) for t in targets for a in ARMS}, INPUT_DRIFT,
         "PREDICTION_COHORT_DRIFT")

    sealed = {}
    for key in ("parent_queries", "oracle_queries"):
        for line in Path(identities[key]["path"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            sealed[(row["qid"], row["arm"])] = row["result"]["query"]
    need(len(sealed) == EXPECTED_PREDICTIONS, INPUT_DRIFT, "SEALED_QUERY_COUNT_DRIFT")
    return targets, reachability, predictions, sealed


def main():
    parser = argparse.ArgumentParser(description="EFBPT N=25 target-vector diagnostic")
    parser.add_argument("--run", action="store_true",
                        help="Required. Without it the probe exits without touching anything.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "UNKNOWN"))
    parser.add_argument("--probe-sha256", default=None,
                        help="Audited SHA-256 of this script; mismatch is a fatal abort.")
    parser.add_argument("--wrapper-sha256", default=None,
                        help="Audited SHA-256 of the sbatch wrapper; mismatch is a fatal abort.")
    parser.add_argument("--skip-index-hash", action="store_true",
                        help="Skip the 34.3 GiB index hash; identity still checked by bytes, "
                             "ntotal, dimension and metric. Recorded in START.json when used.")
    args = parser.parse_args()

    if not args.run:
        event("refused_without_run_flag", reason="--run is required to execute this diagnostic")
        return 1

    repo = Path(args.repo).resolve()
    out_dir = repo / OUTPUT_RELATIVE
    out = None
    stage = "guards"
    completed = []
    try:
        core = load_core(repo)
        stage = "repository_state"
        repo_state = repository_state(repo)
        stage = "pinned_inputs"
        holder = {}
        identities = verify_pinned_inputs(repo, holder)
        manifest = holder["manifest"]
        need(manifest["cohort"]["accepted_pairs"] == EXPECTED_TARGETS
             and manifest["cohort"]["qids"] == EXPECTED_QIDS
             and manifest["cohort"]["predictions"] == EXPECTED_PREDICTIONS, INPUT_DRIFT,
             "ACTIVATION_COHORT_DRIFT")
        stage = "environment"
        environment = verify_environment(manifest)
        stage = "assets"
        assets = verify_assets(manifest, args.skip_index_hash)
        stage = "self_hash"
        script_path = Path(__file__).resolve()
        script_sha, script_bytes = sha256_file(script_path)
        wrapper_path = script_path.with_suffix(".sbatch")
        wrapper_sha, wrapper_bytes = (sha256_file(wrapper_path)
                                      if wrapper_path.is_file() else (None, None))
        if args.probe_sha256 is not None:
            need(args.probe_sha256 == script_sha, INPUT_DRIFT,
                 "PROBE_SELF_HASH_DRIFT: observed " + script_sha)
        if args.wrapper_sha256 is not None:
            need(wrapper_sha is not None, INPUT_DRIFT, "WRAPPER_MISSING_FOR_HASH_CHECK")
            need(args.wrapper_sha256 == wrapper_sha, INPUT_DRIFT,
                 "WRAPPER_SELF_HASH_DRIFT: observed " + wrapper_sha)
        stage = "inputs"
        targets, reachability, predictions, sealed_queries = build_inputs(repo, core, identities)

        stage = "output_absence_guard"
        need(not out_dir.exists(), OPERATIONAL_FAILURE,
             "OUTPUT_DIRECTORY_EXISTS: " + str(out_dir))
        out_dir.mkdir(parents=False, exist_ok=False)
        out = OutputRoot(out_dir)

        start = {"schema": SCHEMA, "version": VERSION, "utc": utc_now(),
                 "job_id": args.job_id, "status": "STARTED",
                 "scope": "POST_OUTCOME_DIAGNOSTIC_NOT_METHOD_EVALUATION",
                 "historical_global_lineage": "UNESTABLISHED",
                 "repository": repo_state,
                 "diagnostic_script": {"path": str(script_path), "bytes": script_bytes,
                                       "sha256": script_sha},
                 "diagnostic_wrapper": {"path": str(wrapper_path), "bytes": wrapper_bytes,
                                        "sha256": wrapper_sha},
                 "inputs": identities, "assets": assets, "environment": environment,
                 "declared_thresholds": {"score_abs_tolerance": SCORE_ABS_TOLERANCE,
                                         "minimum_cosine": MIN_COSINE,
                                         "maximum_l2_distance": MAX_L2_DISTANCE,
                                         "maximum_unit_norm_error": MAX_UNIT_NORM_ERROR},
                 "expected_counts": {"qids": EXPECTED_QIDS, "targets": EXPECTED_TARGETS,
                                     "target_rows": EXPECTED_TARGET_ROWS,
                                     "predictions": EXPECTED_PREDICTIONS,
                                     "occurrences": EXPECTED_OCCURRENCES,
                                     "unique_candidate_rows": EXPECTED_UNIQUE_CANDIDATE_ROWS},
                 "terminal_statuses": [OPERATIONAL_FAILURE, INPUT_DRIFT,
                                       TARGET_ALIGNMENT_FAILURE, SCORE_REPRODUCTION_FAILURE,
                                       DIAGNOSTIC_COMPLETE]}
        out.write_json("START.json", start)
        completed.append("START.json")

        stage = "T1"
        target_rows, t1 = stage_t1(core, manifest, targets, reachability, out)
        completed.append("target_rows.jsonl")
        rows_by_title = {}
        for row in target_rows:
            rows_by_title.setdefault(row["normalized_title"], []).append(row)

        stage = "load_models"
        runtime = configure_runtime(manifest, core)
        index, index_structure = load_index(manifest)
        model, encoder_identity = load_encoder(manifest)
        encoder_identity["runtime"] = runtime
        enc_cfg = manifest["settings"]["encoder"]

        stage = "T2"
        _, failed_titles, t2 = stage_t2(index, model, enc_cfg, target_rows, out)
        completed.append("target_alignment.jsonl")

        stage = "T3"
        query_vectors, t3 = stage_t3(index, model, enc_cfg, predictions, sealed_queries, out)
        completed.append("query_score_controls.jsonl")

        t4 = t5 = None
        if not t3["reproduced"]:
            terminal = SCORE_REPRODUCTION_FAILURE
            event("t3_failed_skipping_t4_t5", exceeding=t3["count_exceeding_tolerance"])
        else:
            stage = "T4"
            targets_by_qid = {}
            for target in targets:
                targets_by_qid.setdefault(target["qid"], []).append(target)
            _, t4 = stage_t4(core, index, query_vectors, predictions, targets_by_qid,
                             rows_by_title, failed_titles, out)
            completed.append("target_score_margins.jsonl")

            stage = "T5"
            reader = MetadataReader(core, manifest["assets"]["metadata"]["path"],
                                    manifest["assets"]["offsets"]["path"],
                                    manifest["assets"]["metadata"]["bytes"])
            try:
                _, t5 = stage_t5(core, index, model, enc_cfg, targets, failed_titles,
                                 reader, out)
            finally:
                reader.close()
            completed.append("title_self_retrieval.jsonl")
            terminal = (TARGET_ALIGNMENT_FAILURE if failed_titles else DIAGNOSTIC_COMPLETE)

        stage = "summary"
        summary = {"schema": SCHEMA, "version": VERSION, "utc": utc_now(),
                   "job_id": args.job_id, "status": terminal,
                   "scope": "POST_OUTCOME_DIAGNOSTIC_NOT_METHOD_EVALUATION",
                   "cohort_scope": "ACCEPTED_TARGET_ROWS_ONLY; NOT_GLOBAL_LINEAGE_PROOF",
                   "historical_global_lineage": "UNESTABLISHED",
                   "frozen_pilot_head": FROZEN_PILOT_HEAD,
                   "diagnostic_code_head": repo_state["diagnostic_code_head"],
                   "index_structure": index_structure, "encoder_identity": encoder_identity,
                   "t1_target_row_inventory": t1, "t2_target_alignment": t2,
                   "t3_query_score_reproduction": t3, "t4_target_score_margins": t4,
                   "t5_title_self_retrieval": t5,
                   "pilot_scores_modified": False, "repair_proposed": False,
                   "frozen_artifacts_written": 0}
        out.write_json("summary.json", summary)
        completed.append("summary.json")

        stage = "seal"
        seal = {"schema": SCHEMA + ".seal", "version": VERSION, "utc": utc_now(),
                "job_id": args.job_id, "status": terminal,
                "scope": "POST_OUTCOME_DIAGNOSTIC_NOT_METHOD_EVALUATION",
                "historical_global_lineage": "UNESTABLISHED",
                "artifacts": {name: spec for name, spec in sorted(out.written.items())},
                "inputs": identities, "assets": assets,
                "diagnostic_script_sha256": script_sha, "diagnostic_wrapper_sha256": wrapper_sha,
                "frozen_pilot_head": FROZEN_PILOT_HEAD,
                "diagnostic_code_head": repo_state["diagnostic_code_head"],
                "declared_thresholds": start["declared_thresholds"],
                "scheduler_logs": "OUTSIDE_SEAL"}
        out.write_json("DIAGNOSTIC_SEAL.json", seal)
        event("diagnostic_sealed", status=terminal, artifacts=len(out.written))
        return 0 if terminal in (DIAGNOSTIC_COMPLETE, TARGET_ALIGNMENT_FAILURE) else 2

    except ProbeError as error:
        return terminate(out, error.status, stage, type(error).__name__, error.message, completed)
    except Exception as error:  # noqa: BLE001 - every failure must be recorded
        return terminate(out, OPERATIONAL_FAILURE, stage, type(error).__name__,
                         str(error), completed)


def terminate(out, status, stage, exception_type, message, completed):
    event("diagnostic_failed", status=status, failed_stage=stage,
          exception_type=exception_type, message=message, completed=completed)
    if out is not None:
        try:
            out.write_json("FAILURE.json", {"schema": SCHEMA + ".failure", "utc": utc_now(),
                                            "status": status, "failed_stage": stage,
                                            "exception_type": exception_type,
                                            "message": message,
                                            "completed_artifacts": completed})
        except Exception as nested:  # noqa: BLE001
            event("failure_record_unwritable", reason=str(nested))
    return 3


if __name__ == "__main__":
    sys.exit(main())
