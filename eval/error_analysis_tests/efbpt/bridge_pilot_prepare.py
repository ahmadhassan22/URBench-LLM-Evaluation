#!/usr/bin/env python3
"""URBench N=25 preparation, revision 0.3. No generation, retrieval or scoring.

--check-small-inputs: read-only hashes, sidecar metadata and environment checks.
--self-test: synthetic in-memory checks; never opens repository/corpus inputs.
--prepare: explicitly authorized compute-node job; creates NEW preparation and
archive directories. Preserves existing inputs. Refuses overwrite or automatic
resume, including after interruption. No model/index deserialization or imports
of rag builders. A completed preparation does NOT activate the experiment.

Revision 0.2 corrects job 80439's asymmetric title-collision comparison.
Metadata correspondence uses only the already-resolved raw parent's exact-case
title; other-case collision rows remain recorded diagnostically. Hash/copy
acceptance, title normalization for retrieval/scoring, and the cohort do not
change.

Revision 0.3 fixes a separate variable-overwrite bug in revision 0.2: the
excluded-cache-blob COUNT was overwritten by excluded parent-collision ROWS,
making the final count comparison always false. Counts and rows now use
distinct names. Both cache inventories and field-level differences are saved
before the final assertion. All strict identity checks, including st_dev, stay
unchanged. Fresh preparation_r3 and archive paths preserve both failed runs.

Target: eval/error_analysis_tests/efbpt/bridge_pilot_prepare.py
Python 3.10+, NumPy 1.24.0, PyArrow 12.0.0 in the inspected urbench_eval env.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import types
from urllib.parse import parse_qsl, unquote, urlsplit

VERSION = "0.3"
ROOT_DEFAULT = Path("/mnt/home/user41/URBench")
CACHE_DEFAULT = Path("/mnt/home/user41/.cache/modelscope/hub/datasets/downloads")
ARCHIVE_DEFAULT = Path("/mnt/home/user41/URBench_pilot_archives/bridge_pilot_n25_preparation_v1_r3")
OUT_REL = "outputs/efbpt/bridge_pilot_n25/v1/preparation_r3"
PREPARATION_CHANGE = {
    "corrects_jobs": ["80439", "80472"],
    "reason": "Preserve exact-case parent fix; separate excluded blob count from excluded parent-row list",
    "cache_policy": "Strict equality retained, including device IDs; persist both observations and differences",
    "device_diagnosis_limit": "Job 80472 did not save its second inventory; cross-namespace device differences do not prove its in-job cause",
    "page_selection": "EXACT_CASE_OF_ALREADY_RESOLVED_RAW_PARENT",
    "excluded_rows": "Preserved in correspondence diagnostics; no retrieval-universe filtering",
    "previous_output_policy": "Preserve original preparation and archive; fresh run with new destinations",
    "experiment_parameters_changed": False,
}
PROTOCOL_REL = "docs/EFBPT_BRIDGE_PILOT_N25_PROTOCOL.md"
AUDIT_REL = "eval/error_analysis_tests/efbpt/efbpt_assisted_qid_audit.py"
SCRIPT_REL = "eval/error_analysis_tests/efbpt/bridge_pilot_prepare.py"
JOB_REL = "eval/error_analysis_tests/efbpt/bridge_pilot_prepare.sbatch"
VERIFIER_REL = "eval/error_analysis_tests/efbpt/efbpt_verify_assisted_candidates.py"
DEV_REL = "data/strategyqa_official/dev200_seed4242.jsonl"
HUMAN_REL = "outputs/efbpt/stage0_assisted/human_verified_candidates.jsonl"
INDEX_REL = "rag/index/wikipedia_full.index"
META_REL = "rag/index/wikipedia_full_meta.jsonl"
OFFSETS_REL = "rag/index/wikipedia_full_meta.offsets.npy"
N_SHARDS, N_PAGES, N_VECTORS, DIM = 41, 6407814, 23963971, 384
INDEX_BYTES = 36808659501
PINS = {
    PROTOCOL_REL: "c580427ffc53b97253731eb724a1ee8e408f8242d1aba471c30f6d962593450c",
    AUDIT_REL: "265b0e050074df27639b89baa7ea11b989fae8277affc0a7d62b1b13cc04954a",
    VERIFIER_REL: "b3e90b474417aa046a4fdfeb1950174109cd33b8f0f1eb8b814bcdf564e7f0a8",
    DEV_REL: "1ae2cd21c93d1c8d3fda8f6990a183df558e6509d0884fadb29983f5f610d43c",
    HUMAN_REL: "fc9fd100a9acb13e1ce0eb255d0f67817d54a809d42f319dcacf621ebcf1c354",
    "outputs/efbpt/stage0_assisted/assisted_pass2.jsonl": "201e4df1995d0b04c07142246b1f4af02f5d126caae2849514cae4695a297e65",
    "outputs/efbpt/stage0_assisted/assisted_pass3.jsonl": "33895e498b2a71ec30ce471f9e77f8822643fa3faaea84cc5623d44877e1b420",
    "outputs/efbpt/stage0_assisted/bridge_candidate_qids.jsonl": "410a59d7e5fc0c22b12a947c5a9195847c807b318394e70a1fb9d4333d845b37",
    "data/strategyqa_official/efbpt/stage0/source_instance_master.jsonl": "5ebd1968a8ac2f8013b595b69f8bd320025ca7b46d50b602e17f248ce739a084",
    "data/strategyqa_official/efbpt/stage0/official_evidence_links.jsonl": "d59b956ca4003a8438653cc9930f810f5eea1e1655aba104e738152b9a137d0e",
    "data/strategyqa_official/efbpt/stage0/pass1_question_manifest.jsonl": "73cb5f1b99f0b146c723c2c46aafe1cd4006ba95c251d0def7d463675ae45f18",
    "data/strategyqa_official/efbpt/stage0/human_annotation_log.jsonl": "a418e62f6288c775979e7042f6902b83d3fabaae15f6c603d27fa9754631154c",
}
BUILDER_BLOBS = {
    "rag/build_chunks.py": "fa4d58387f970eda8693c886fb93b759d764c829",
    "rag/build_index_full.py": "bf2745b3d6fd5213b0550189448f4d6bd4a38961",
}
CHILD_COUNTS = dict(zip([
    "15885efe6f91724c16d9", "25a088d9d2ce674e639a", "3295844627a9bd1b9135",
    "3f8a1bd6bf3a967cdeb6", "48da75d87c66754ccc2e", "5897ec22db850f7b416e",
    "6c43fa359095fd0845f5", "7281474f2760dce03f39", "73c52134ab2a903d86db",
    "73ca2ef1da65b2a2ebe6", "7b84d2bc643ddc2085f0", "7be51fdef30345de666e",
    "7c3759cc1da78e9fbd79", "7f4effbc97ab2b5fd4a7", "80aa769f55b14c1e4d8d",
    "8cfde6ee28d059a5aff6", "8d06b619a7045ed02f51", "8d3ddaee20ad48edc066",
    "a0896de3fd13cd0f3e16", "b747938f597b09e43603", "e6d3973ed3feb8a42928",
    "e87b63e92165b417d37f", "f2859b2ce17b5f5a6ad9", "f43533225534420816d6",
    "ff3811735ededd8ec3a7",
], [1,1,1,1,1,4,1,1,2,1,1,3,1,1,1,1,1,2,2,1,2,2,1,2,1]))


class PrepError(RuntimeError):
    """An integrity/operational failure, never a scientific zero."""


def need(ok, message):
    if not ok:
        raise PrepError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def strict_json(raw):
    def invalid_constant(_):
        raise PrepError("NONFINITE_JSON_CONSTANT")
    try:
        return json.loads(raw, object_pairs_hook=no_duplicate_keys,
                          parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise PrepError("INVALID_JSON") from exc


def norm(title):
    return " ".join(str(title).replace("_", " ").strip().lower().split())


def exact_case(title):
    return " ".join(title.replace("_", " ").split())


def no_symlinks(path):
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        need(not item.is_symlink(), "SYMLINK_PATH: " + str(item))
    return path


def file_stamp(path):
    path = no_symlinks(path)
    s = path.stat()
    need(stat.S_ISREG(s.st_mode), "NOT_REGULAR_FILE: " + str(path))
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read_small(path):
    before = file_stamp(path)
    need(before[2] <= 64 * 1024**2, "SMALL_INPUT_TOO_LARGE: " + str(path))
    raw = path.read_bytes()
    need(file_stamp(path) == before, "INPUT_CHANGED: " + str(path))
    return raw, {"path": str(path), "sha256": digest(raw), "stat": before}


def fingerprint(path):
    before = file_stamp(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    need(file_stamp(path) == before, "INPUT_CHANGED_DURING_HASH: " + str(path))
    return {"path": str(path), "sha256": h.hexdigest(), "stat": before}


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_parents(path):
    path = no_symlinks(path)
    if path.exists():
        need(path.is_dir(), "NOT_DIRECTORY: " + str(path))
        return
    make_parents(path.parent)
    path.mkdir(mode=0o700)
    fsync_dir(path.parent)


def new_directory(path):
    no_symlinks(path)
    need(not path.exists(), "EXISTING_DIRECTORY_PRESERVED; review before recovery: " + str(path))
    make_parents(path.parent)
    path.mkdir(mode=0o700)
    fsync_dir(path.parent)


def write_new(path, raw):
    """Only new files: partial failed writes stay preserved and cannot be reused."""
    no_symlinks(path)
    make_parents(path.parent)
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(path.parent)
    actual = fingerprint(path)
    need(actual["sha256"] == digest(raw), "WRITE_VERIFICATION_FAILED")
    return {"sha256": actual["sha256"], "bytes": len(raw)}


def write_json(path, obj):
    return write_new(path, (canonical(obj) + "\n").encode("utf-8"))


def write_jsonl(path, rows):
    return write_new(path, "".join(canonical(r) + "\n" for r in rows).encode("utf-8"))


def event(stage, **fields):
    # No question, title, passage, facts, answers or annotation contents in logs.
    print(canonical({"stage": stage, **fields}), flush=True)


def environment():
    versions = {}
    for package, expected in (("numpy", "1.24.0"), ("pyarrow", "12.0.0")):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PrepError("MISSING_PACKAGE: " + package) from exc
        need(versions[package] == expected, "PACKAGE_VERSION_DRIFT: " + package)
    need(sys.version_info[:3] == (3, 10, 19), "PYTHON_VERSION_DRIFT")
    return {"python": sys.version, "executable": sys.executable, "packages": versions}


def git_state(repo):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    def call(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        need(r.returncode == 0, "GIT_READ_FAILED")
        return r.stdout.decode("utf-8").strip()
    return {"head": call("rev-parse", "HEAD"),
            "status": call("status", "--porcelain=v1", "--untracked-files=normal")}


def inspect_small_inputs(repo):
    identities, raw_by_rel = {}, {}
    other = [*BUILDER_BLOBS, SCRIPT_REL, JOB_REL,
             "docs/EFBPT_STAGE0_SOURCE_ROLE_ATTAINABILITY_FREEZE.md"]
    for rel in [*PINS, *other]:
        raw, info = read_small(repo / rel)
        if rel in PINS:
            need(info["sha256"] == PINS[rel], "PINNED_INPUT_DRIFT: " + rel)
        if rel in BUILDER_BLOBS:
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            need(blob == BUILDER_BLOBS[rel], "BUILDER_CODE_DRIFT: " + rel)
        identities[rel], raw_by_rel[rel] = info, raw
    need(Path(__file__).resolve() == (repo / SCRIPT_REL).resolve(), "SAVE_SCRIPT_AT_REPOSITORY_TARGET")
    return identities, raw_by_rel


def sidecar_identity(obj):
    need(isinstance(obj, dict) and set(obj) == {"url", "etag"}, "SIDECAR_SCHEMA_DRIFT")
    need(isinstance(obj["url"], str), "SIDECAR_URL_NOT_STRING")
    u = urlsplit(obj["url"])
    path = unquote(u.path)
    if path != "/api/v1/datasets/AI-ModelScope/wikipedia/repo":
        return None
    need(u.scheme == "https" and u.hostname == "www.modelscope.cn" and
         not u.username and not u.password and not u.fragment, "WIKIPEDIA_URL_IDENTITY_DRIFT")
    fields = no_duplicate_keys(parse_qsl(u.query, keep_blank_values=True))
    need(set(fields) == {"Source", "Revision", "FilePath"}, "WIKIPEDIA_QUERY_SCHEMA_DRIFT")
    need(fields["Source"] == "SDK" and fields["Revision"] == "master" and obj["etag"] is None,
         "WIKIPEDIA_SIDECAR_IDENTITY_DRIFT")
    match = re.fullmatch(r"20231101\.en/train-(\d{5})-of-00041\.parquet", fields["FilePath"])
    need(match is not None, "WIKIPEDIA_FILEPATH_DRIFT")
    return {"dataset": "AI-ModelScope/wikipedia", **fields, "shard_index": int(match.group(1))}


def discover_shards(cache):
    no_symlinks(cache)
    need(cache.is_dir(), "CACHE_DIRECTORY_MISSING")
    selected, excluded = [], 0
    for sidecar in sorted(cache.glob("*.json")):
        if not re.fullmatch(r"[0-9a-f]{64}\.json", sidecar.name):
            continue
        raw, info = read_small(sidecar)
        identity = sidecar_identity(strict_json(raw))
        blob = sidecar.with_suffix("")
        if identity is None:
            if blob.exists():
                file_stamp(blob)
                excluded += 1
            continue
        selected.append({**identity, "sidecar": info, "blob_path": str(blob),
                         "blob_stat": file_stamp(blob)})
    need(sorted(r["shard_index"] for r in selected) == list(range(N_SHARDS)),
         "WIKIPEDIA_SHARDS_MUST_BE_00000_THROUGH_00040_ONCE")
    need(len({tuple(r["blob_stat"][:2]) for r in selected}) == N_SHARDS,
         "WIKIPEDIA_BLOB_IDENTITIES_NOT_UNIQUE")
    need(excluded == 8, "NON_WIKIPEDIA_INVENTORY_DRIFT; inspect metadata before rerunning")
    return sorted(selected, key=lambda r: r["shard_index"]), excluded


def cache_recheck_report(initial_shards, initial_excluded_count,
                         final_shards, final_excluded_count):
    """Compare unmodified inventory snapshots without weakening identity rules.

    Device IDs remain part of equality. Cross-namespace diagnostic observations
    cannot establish a same-process device change in a past job. Save both
    actual observations BEFORE asserting so future failures are attributable.
    """
    before = {"shards": initial_shards, "excluded_regular_blobs": initial_excluded_count}
    after = {"shards": final_shards, "excluded_regular_blobs": final_excluded_count}
    differences = []
    def compare(a, b, path):
        if type(a) is not type(b):
            differences.append({"field": path, "initial": a, "final": b,
                                "reason": "TYPE_CHANGED"})
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                field = path + "/" + key
                if key not in a or key not in b:
                    differences.append({"field": field, "initial": a.get(key),
                                        "final": b.get(key), "reason": "KEY_CHANGED"})
                else:
                    compare(a[key], b[key], field)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append({"field": path + "/length", "initial": len(a),
                                    "final": len(b), "reason": "LENGTH_CHANGED"})
            for i, (left, right) in enumerate(zip(a, b)):
                compare(left, right, path + "/" + str(i))
        elif a != b:
            differences.append({"field": path, "initial": a, "final": b,
                                "reason": "VALUE_CHANGED"})
    compare(before, after, "cache")
    counts_valid = (type(initial_excluded_count) is int and type(final_excluded_count) is int
                    and initial_excluded_count == final_excluded_count == 8)
    return {"status": "PASS" if not differences and counts_valid else "FAIL",
            "policy": "STRICT_ALL_FIELDS_INCLUDING_DEVICE_ID", "counts_valid": counts_valid,
            "stat_field_order": ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"],
            "initial": before, "final": after, "differences": differences}


def import_pinned_audit(repo, raw):
    # Only this already inspected audit and its blob-pinned verifier are imported.
    module = types.ModuleType("_bridge_pilot_pinned_audit")
    module.__file__ = str(repo / AUDIT_REL)
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def export_cohort(decisions, verified, questions, dev_rows, expected_counts):
    groups = defaultdict(list)
    for row in decisions.values():
        if row["verdict"] == verified:
            groups[row["qid"]].append(row)
    need({q: len(rs) for q, rs in groups.items()} == expected_counts,
         "ACCEPTED_QIDS_OR_CHILD_MULTIPLICITIES_DRIFT")
    dev = {}
    for row in dev_rows:
        qid = row.get("urbench_qid")
        need(isinstance(qid, str) and qid not in dev, "DEV200_QID_INVALID_OR_DUPLICATE")
        dev[qid] = row
    parents, oracle, targets = [], [], []
    parent_ids, parent_norms, child_ids, child_norms = set(), set(), set(), set()
    for qid, rows in sorted(groups.items()):
        parent_set = {(r["parent_source_instance_id"], r["parent_title"]) for r in rows}
        need(len(parent_set) == 1, "MULTIPLE_PARENTS_FOR_QID: " + qid)
        sid, title = next(iter(parent_set))
        need(isinstance(title, str) and bool(norm(title)), "INVALID_PARENT_TITLE")
        need(sid not in parent_ids and norm(title) not in parent_norms, "REPEATED_PARENT")
        parent_ids.add(sid)
        parent_norms.add(norm(title))
        need(qid in dev and qid in questions, "QUESTION_JOIN_MISSING: " + qid)
        question = questions[qid]
        need(isinstance(question, str) and question.strip() and dev[qid].get("question_ur") == question,
             "URDU_QUESTION_DRIFT: " + qid)
        facts = dev[qid].get("urbench_facts")
        need(isinstance(facts, list) and len(facts) > 0 and
             all(isinstance(f, str) and f.strip() for f in facts), "ORACLE_FACTS_SCHEMA: " + qid)
        parents.append({"qid": qid, "question_ur": question,
                        "parent_source_instance_id": sid, "parent_title": title,
                        "parent_normalized_title": norm(title)})
        oracle.append({"qid": qid, "question_ur": question, "urbench_facts": list(facts)})
        children = []
        for row in sorted(rows, key=lambda r: r["child_source_instance_id"]):
            cid, ct = row["child_source_instance_id"], row["child_title"]
            need(isinstance(ct, str) and bool(norm(ct)), "INVALID_CHILD_TITLE")
            need(cid not in child_ids and norm(ct) not in child_norms and norm(ct) != norm(title),
                 "DUPLICATE_OR_PARENT_EQUALS_CHILD: " + qid)
            child_ids.add(cid)
            child_norms.add(norm(ct))
            children.append({"source_instance_id": cid, "title": ct, "normalized_title": norm(ct)})
        targets.append({"qid": qid, "children": children})
    return parents, oracle, targets


def load_cohort(repo, audit_raw):
    audit = import_pinned_audit(repo, audit_raw)
    report = audit.audit(repo)
    need(report["audit_status"] == "AUDIT_PASS" and report["handoff_counts_match"] and
         report["unique_qids_with_verified_pair"] == 25 and report["verified_pairs"] == 36,
         "ASSISTED_AUDIT_FAILED")
    verifier, _, _ = audit.load_verifier(repo)
    data = verifier.ReviewData()
    decisions = verifier.validate_existing_rows(verifier.load_jsonl(verifier.OUTPUT_PATH), data)
    raw, _ = read_small(repo / DEV_REL)
    rows = [strict_json(line) for line in raw.splitlines()]
    need(len(rows) == 200, "DEV200_ROW_COUNT_DRIFT")
    parents, oracle, targets = export_cohort(decisions, verifier.VERIFIED, data.question_by_qid,
                                            rows, CHILD_COUNTS)
    return parents, oracle, targets, report


def validate_parquet(pf, pa):
    schema = pf.schema_arrow
    need(schema.names == ["id", "url", "title", "text"] and
         all(pa.types.is_string(field.type) for field in schema), "PARQUET_SCHEMA_DRIFT")
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    need(all(n > 0 for n in sizes) and sum(sizes) == pf.metadata.num_rows, "PARQUET_ROW_GROUP_DRIFT")
    return sizes


def matched_pages(pf, row_group_sizes, wanted, shard, text_batch=1024):
    """First read only id/url/title; read text only in matching row groups."""
    matches = defaultdict(list)
    shard_row_start = 0
    for rg, size in enumerate(row_group_sizes):
        selected, row_start = {}, 0
        for batch in pf.iter_batches(batch_size=16384, row_groups=[rg],
                                     columns=["id", "url", "title"], use_threads=False):
            for j, row in enumerate(batch.to_pylist()):
                need(isinstance(row["title"], str), "RAW_TITLE_SCHEMA")
                nt = norm(row["title"])
                if nt in wanted:
                    need(all(isinstance(row[k], str) and row[k] for k in ("id", "url", "title")),
                         "MATCHED_RAW_IDENTITY_SCHEMA")
                    selected[row_start + j] = (nt, row)
            row_start += batch.num_rows
        need(row_start == size, "PARQUET_BATCH_ROW_COUNT")
        if selected:
            need(len(selected) <= 1000, "EXCESSIVE_PARENT_COLLISIONS; inspect before proceeding")
            row_start, seen = 0, set()
            for batch in pf.iter_batches(batch_size=text_batch, row_groups=[rg],
                                         columns=["id", "url", "title", "text"], use_threads=False):
                for pos in sorted(p for p in selected if row_start <= p < row_start + batch.num_rows):
                    full = batch.slice(pos - row_start, 1).to_pylist()[0]
                    nt, small = selected[pos]
                    need({k: full[k] for k in small} == small, "RAW_ROW_CHANGED_BETWEEN_PASSES")
                    need(isinstance(full["text"], str) and full["text"].strip(), "EMPTY_PARENT_PAGE")
                    matches[nt].append({"raw_page_id": full["id"], "raw_url": full["url"],
                        "raw_title": full["title"], "raw_text": full["text"],
                        "raw_text_sha256": digest(full["text"].encode("utf-8")),
                        "location": {"shard_index": shard["shard_index"],
                            "blob_path": shard["blob_path"], "blob_sha256": shard["blob_sha256"],
                            "row_group": rg, "row_in_group": pos,
                            "row_in_shard": shard_row_start + pos}})
                    seen.add(pos)
                row_start += batch.num_rows
            need(row_start == size and seen == set(selected), "MATCHED_TEXT_ROW_MISSING")
        shard_row_start += size
    return matches


def resolve_parent(title, rows):
    need(bool(rows), "PARENT_PAGE_NOT_FOUND")
    groups, identities = {}, {}
    for row in rows:
        identity = (row["raw_page_id"], row["raw_title"])
        signature = (row["raw_text_sha256"], row["raw_url"])
        need(identity not in identities or identities[identity] == signature, "CONFLICTING_RAW_PAGE_COPIES")
        identities[identity] = signature
        key = (*identity, *signature)
        if key not in groups:
            groups[key] = {k: row[k] for k in ("raw_page_id", "raw_url", "raw_title", "raw_text", "raw_text_sha256")}
            groups[key]["locations"] = []
        groups[key]["locations"].append(row["location"])
    choices = list(groups.values())
    if len(choices) == 1:
        chosen, decision = choices[0], "UNIQUE_NORMALIZED_PAGE"
    else:
        exact = [r for r in choices if exact_case(r["raw_title"]) == exact_case(title)]
        need(len(exact) == 1, "AMBIGUOUS_PARENT_PAGE")
        chosen, decision = exact[0], "UNIQUE_EXACT_CASE_PAGE_IN_NORMALIZED_BUCKET"
    chosen = dict(chosen)
    chosen.update({"lookup_decision": decision, "normalized_bucket_rows": len(rows),
                   "normalized_bucket_distinct_pages": len(choices)})
    return chosen


def local_chunks(page):
    text = page["raw_text"]
    words = list(re.finditer(r"\S+", text))
    chunks = []
    page_key = digest(canonical({k: page[k] for k in ("raw_page_id", "raw_title", "raw_text_sha256")}).encode())
    for start in range(0, len(words), 150):
        end = min(start + 200, len(words))
        lo, hi = words[start].start(), words[end - 1].end()
        chunks.append({"chunk_id": f"{page_key}:w{start}-{end}", "word_start": start,
            "word_end": end, "chunk_char_start": lo, "chunk_char_end": hi, "text": text[lo:hi]})
        if end == len(words):
            break
    need(bool(chunks), "NO_PARENT_WORDS")
    return chunks


def legacy_hashes(text):
    words = text.split()
    return [digest(" ".join(words[i:i+200]).encode("utf-8")) for i in range(0, len(words), 150)]


def select_indexed_parent_rows(page, indexed):
    """Resolve metadata to the SAME raw parent before comparing chunk hashes.

    Selection depends only on the already-resolved raw title, never on which
    rows make hashes pass, child targets, or future retrieval outcomes. This
    affects preparation correspondence only. The index and its normalized-title
    retrieval/scoring universe are untouched. Keep every excluded row for audit.
    """
    selected, excluded = [], []
    parent_norm = norm(page["raw_title"])
    parent_case = exact_case(page["raw_title"])
    for row in indexed:
        title = row.get("metadata_title")
        need(isinstance(title, str) and norm(title) == parent_norm,
             "METADATA_ROW_OUTSIDE_PARENT_NORMALIZED_BUCKET")
        (selected if exact_case(title) == parent_case else excluded).append(row)
    return selected, excluded


def correspondence(page, indexed):
    expected = Counter(legacy_hashes(page["raw_text"]))
    actual = Counter(r["text_sha256"] for r in indexed)
    copies = len(page["locations"])
    valid_scales = [k for k in sorted({1, copies}) if actual == Counter({h: n*k for h,n in expected.items()})]
    need(bool(valid_scales), "PARENT_INDEX_TEXT_MISMATCH")
    scale = valid_scales[0]
    return {"status": "PASS", "legacy_algorithm": "split_200_stride150_including_extra_tail_v1",
        "legacy_chunk_hashes_in_page_order": legacy_hashes(page["raw_text"]),
        "indexed_copy_count": scale, "indexed_chunk_count": len(indexed),
        "indexed_rows": indexed, "comparison": "EXACT_SHA256_MULTISET_WITH_COPY_COUNT",
        "scope": "Selected parent text versus persisted metadata; not vector-content or global snapshot proof"}


def scan_metadata(stream, offsets, wanted, total_rows, progress=False):
    """One sequential scan; verify EVERY offset before processing its row."""
    found = defaultdict(list)
    h, position, count = hashlib.sha256(), 0, 0
    for raw in stream:
        need(count < total_rows and int(offsets[count]) == position, "METADATA_OFFSET_MISMATCH")
        h.update(raw)
        row = strict_json(raw)
        need(isinstance(row, dict) and set(row) == {"title", "text"} and
             isinstance(row["title"], str) and bool(norm(row["title"])) and
             isinstance(row["text"], str) and bool(row["text"].strip()), "METADATA_SCHEMA_DRIFT")
        nt = norm(row["title"])
        if nt in wanted:
            found[nt].append({"global_row": count, "byte_offset": position,
                "metadata_title": row["title"], "text_sha256": digest(row["text"].encode("utf-8"))})
        position += len(raw)
        count += 1
        if progress and count % 1000000 == 0:
            event("metadata_scan", rows_checked=count)
    need(count == total_rows and len(offsets) == total_rows, "METADATA_ROW_COUNT_MISMATCH")
    return found, {"sha256": h.hexdigest(), "bytes": position, "rows": count,
                   "all_offsets_match_jsonl_line_starts": True}


def validate_exports(parent_rows, oracle_rows, target_rows):
    parent_keys = {"qid", "question_ur", "parent_source_instance_id", "parent_title",
                   "parent_normalized_title", "page", "chunks"}
    page_keys = {"raw_page_id", "raw_url", "raw_title", "raw_text", "raw_text_sha256",
                 "locations", "lookup_decision", "normalized_bucket_rows", "normalized_bucket_distinct_pages"}
    loc_keys = {"shard_index", "blob_path", "blob_sha256", "row_group", "row_in_group", "row_in_shard"}
    chunk_keys = {"chunk_id", "word_start", "word_end", "chunk_char_start", "chunk_char_end", "text"}
    qsets = []
    for rows, keys in ((parent_rows, parent_keys), (oracle_rows, {"qid", "question_ur", "urbench_facts"}),
                       (target_rows, {"qid", "children"})):
        need(all(isinstance(r, dict) and set(r) == keys for r in rows), "EXPORT_ALLOWLIST_VIOLATION")
        qset = {r["qid"] for r in rows}
        need(len(qset) == len(rows), "DUPLICATE_EXPORT_QID")
        qsets.append(qset)
    need(qsets[0] == qsets[1] == qsets[2], "EXPORT_QID_SET_MISMATCH")
    for row in parent_rows:
        page = row["page"]
        need(set(page) == page_keys and all(set(x) == loc_keys for x in page["locations"]),
             "PAGE_PROVENANCE_ALLOWLIST_VIOLATION")
        need(row["parent_normalized_title"] == norm(row["parent_title"]) == norm(page["raw_title"]),
             "PARENT_TITLE_NORMALIZATION_MISMATCH")
        need(page["raw_text_sha256"] == digest(page["raw_text"].encode()), "PAGE_TEXT_HASH_MISMATCH")
        need(row["chunks"] == local_chunks(page), "CHUNK_RECONSTRUCTION_MISMATCH")
        need(all(set(c) == chunk_keys for c in row["chunks"]), "CHUNK_ALLOWLIST_VIOLATION")
    for row in target_rows:
        need(all(set(c) == {"source_instance_id", "title", "normalized_title"} for c in row["children"]),
             "TARGET_ALLOWLIST_VIOLATION")


def prepare(args):
    repo, cache, archive = args.repo.resolve(), args.cache.absolute(), args.archive.absolute()
    out = repo / OUT_REL
    need(os.environ.get("SLURM_JOB_ID") and socket.gethostname().split(".")[0].lower() != "psn001",
         "PREPARATION_REQUIRES_SLURM_COMPUTE_NODE")
    need(not archive.is_relative_to(repo) and not repo.is_relative_to(archive),
         "ARCHIVE_MUST_BE_SEPARATE_FROM_REPOSITORY")
    need(not archive.is_relative_to(cache) and not cache.is_relative_to(archive), "ARCHIVE_CACHE_OVERLAP")
    for directory in (out, archive):
        no_symlinks(directory)
        need(not directory.exists(), "EXISTING_DIRECTORY_PRESERVED; review before recovery: " + str(directory))
    env = environment()
    small, raw_inputs = inspect_small_inputs(repo)
    initial_git = git_state(repo)
    shards, excluded_blob_count = discover_shards(cache)
    # Later scanning enriches shard dictionaries; retain the original discovery
    # snapshot independently rather than reconstructing it from mutated data.
    initial_cache_inventory = strict_json(canonical(shards))
    # Validate existing large assets by stat before creating output directories.
    index_stamp = file_stamp(repo / INDEX_REL)
    meta_stamp, offsets_stamp = file_stamp(repo / META_REL), file_stamp(repo / OFFSETS_REL)
    need(index_stamp[2] == INDEX_BYTES, "INDEX_FILE_SIZE_DRIFT")
    os.umask(0o077)
    new_directory(archive)
    # Preserve all small inputs BEFORE parsing any annotation or gold material.
    archived_inputs = {}
    for rel, raw in raw_inputs.items():
        archived_inputs[rel] = write_new(archive / "inputs" / rel, raw)
    write_json(archive / "INPUT_SNAPSHOT.json", {"inputs": archived_inputs, "source_identities": small,
        "git": initial_git, "scope": "Independent persistent-home copy; storage backup policy not established"})
    new_directory(out)
    event("input_snapshot_complete", input_files=len(archived_inputs))
    parents, oracle, targets, audit_report = load_cohort(repo, raw_inputs[AUDIT_REL])
    artifacts = {}
    def save(name, obj, jsonl=False):
        artifacts[name] = (write_jsonl if jsonl else write_json)(out / name, obj)
    save("cohort_audit.json", audit_report)
    save("preparation_start.json", {"version": VERSION, "utc": datetime.now(timezone.utc).isoformat(),
        "git": initial_git, "environment": env, "inputs": small,
        "preparation_change": PREPARATION_CHANGE,
        "archive": str(archive), "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
        "experiment_state": "NOT_FROZEN_NOT_RUN", "resume_policy": "REFUSE_EXISTING_OUTPUT; REVIEW_INTERRUPTION"})
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    wanted = {p["parent_normalized_title"] for p in parents}
    raw_matches, total_pages = defaultdict(list), 0
    for shard in shards:
        path = Path(shard["blob_path"])
        need(file_stamp(path) == shard["blob_stat"], "SHARD_CHANGED_BEFORE_HASH")
        info = fingerprint(path)
        shard["blob_sha256"] = info["sha256"]
        pf = pq.ParquetFile(path)
        try:
            sizes = validate_parquet(pf, pa)
            shard["row_group_sizes"], shard["rows"] = sizes, sum(sizes)
            total_pages += sum(sizes)
            matches = matched_pages(pf, sizes, wanted, shard)
            for nt, rows in matches.items():
                raw_matches[nt].extend(rows)
                need(len(raw_matches[nt]) <= 1000, "EXCESSIVE_PARENT_COLLISIONS")
        finally:
            pf.close()
        need(file_stamp(path) == shard["blob_stat"], "SHARD_CHANGED_DURING_SCAN")
        event("raw_shard_complete", shard_index=shard["shard_index"], shards_total=N_SHARDS)
    need(total_pages == N_PAGES, "RAW_WIKIPEDIA_ROW_COUNT_DRIFT")
    save("raw_shard_inventory.json", {"shards": shards, "excluded_regular_blobs": excluded_blob_count,
        "total_rows": total_pages, "historical_global_snapshot_lineage": "NOT_ESTABLISHED_BY_NEW_HASHES"})
    runtime = []
    for parent in parents:
        try:
            page = resolve_parent(parent["parent_title"], raw_matches[parent["parent_normalized_title"]])
        except PrepError as exc:
            raise PrepError(str(exc) + "; qid=" + parent["qid"]) from exc
        runtime.append({**parent, "page": page, "chunks": local_chunks(page)})
    save("resolved_parent_pages.jsonl", runtime, True)
    event("parent_pages_resolved", qids=len(runtime))
    offsets_info = fingerprint(repo / OFFSETS_REL)
    need(offsets_info["stat"] == offsets_stamp, "OFFSETS_CHANGED_BEFORE_SCAN")
    offsets = np.load(repo / OFFSETS_REL, mmap_mode="r", allow_pickle=False)
    need(offsets.ndim == 1 and offsets.shape == (N_VECTORS,) and offsets.dtype.kind in "iu" and
         offsets.dtype.itemsize == 8, "OFFSETS_ARRAY_SCHEMA_DRIFT")
    # Every offset, including monotonicity, is checked against the byte position
    # in the single metadata scan. No helper can create or repair offsets.
    need(file_stamp(repo / META_REL) == meta_stamp, "METADATA_CHANGED_BEFORE_SCAN")
    with (repo / META_REL).open("rb") as f:
        indexed, metadata_info = scan_metadata(f, offsets, wanted, N_VECTORS, True)
    need(file_stamp(repo / META_REL) == meta_stamp and file_stamp(repo / OFFSETS_REL) == offsets_stamp,
         "METADATA_OR_OFFSETS_CHANGED_DURING_SCAN")
    metadata_info.update({"path": str(repo / META_REL), "stat": meta_stamp})
    del offsets
    save("metadata_scan_summary.json", {"metadata": metadata_info, "offsets": offsets_info})
    checks = []
    failures = []
    for row in runtime:
        bucket = indexed[row["parent_normalized_title"]]
        selected, excluded_collision_rows = select_indexed_parent_rows(row["page"], bucket)
        try:
            check = correspondence(row["page"], selected)
        except PrepError:
            check = {"status": "FAIL", "reason": "PARENT_INDEX_TEXT_MISMATCH",
                     "legacy_chunk_count": len(legacy_hashes(row["page"]["raw_text"])),
                     "legacy_chunk_hashes_in_page_order": legacy_hashes(row["page"]["raw_text"]),
                     "indexed_chunk_count": len(selected),
                     "indexed_rows": selected}
            failures.append(row["qid"])
        check.update({"metadata_page_selection": "EXACT_CASE_OF_ALREADY_RESOLVED_RAW_PARENT",
                      "normalized_bucket_indexed_chunk_count": len(bucket),
                      "excluded_collision_row_count": len(excluded_collision_rows),
                      "excluded_collision_rows": excluded_collision_rows})
        checks.append({"qid": row["qid"], **check})
    save("parent_index_correspondence.json", {"parents": checks, "failed_qids": failures})
    need(not failures, "PARENT_INDEX_TEXT_MISMATCH; see counts in parent_index_correspondence.json")
    event("parent_index_text_checks_passed", qids=25)
    index_info = fingerprint(repo / INDEX_REL)
    need(index_info["stat"] == index_stamp, "INDEX_CHANGED_DURING_PREPARATION")
    save("retrieval_asset_fingerprints.json", {"index": index_info, "metadata": metadata_info,
        "offsets": offsets_info, "expected_vectors": N_VECTORS, "expected_dimension": DIM,
        "index_deserialized": False, "vector_metadata_alignment": "PENDING_SEPARATE_VALIDATION",
        "fingerprint_scope": "First measured bytes; not proof of historical build lineage"})
    validate_exports(runtime, oracle, targets)
    save("runtime_parent_only.jsonl", runtime, True)
    save("runtime_oracle_e.jsonl", oracle, True)
    save("scoring_targets.jsonl", targets, True)
    # Re-hash all protected SMALL inputs. Large files were hashed once and read
    # under stable inode/size/mtime/ctime checks; concurrent mutation aborts.
    final_small, _ = inspect_small_inputs(repo)
    need(small == final_small, "PROTECTED_INPUT_CHANGED_DURING_PREPARATION")
    need(git_state(repo)["head"] == initial_git["head"], "GIT_HEAD_CHANGED_DURING_PREPARATION")
    current_shards, current_excluded_blob_count = discover_shards(cache)
    cache_report = cache_recheck_report(initial_cache_inventory, excluded_blob_count,
                                        current_shards, current_excluded_blob_count)
    save("cache_recheck.json", cache_report)
    need(cache_report["status"] == "PASS",
         "CACHE_CHANGED_DURING_PREPARATION; see cache_recheck.json")
    for rel, stamp in ((INDEX_REL, index_stamp), (META_REL, meta_stamp), (OFFSETS_REL, offsets_stamp)):
        need(file_stamp(repo / rel) == stamp, "LARGE_INPUT_CHANGED_BEFORE_SEAL: " + rel)
    summary = {"preparation_status": "COMPLETE", "experiment_state": "NOT_FROZEN_NOT_RUN",
        "cohort_label": "HUMAN_VERIFICATION_OF_MODEL_ASSISTED_CANDIDATES", "accepted_qids": 25,
        "accepted_pairs": 36, "parents_resolved": 25, "parent_metadata_correspondence_passed": 25,
        "canonical_stage0": "INCOMPLETE_GATES_UNCHANGED", "content_displayed": False,
        "output_directory": str(out), "archive_directory": str(archive),
        "remaining_before_outcomes": ["Review actual full-page prompt lengths without generating outcomes",
            "Validate index type/dimension and vector-to-metadata alignment/lineage",
            "Pin model/tokenizer files and final runner/scorer/job configuration",
            "Audit and activate the prospective amendment/protocol with explicit-path commits"],
        "resume_policy": "No automatic resume; preserve partial files and review interrupted jobs"}
    save("preparation_summary.json", summary)
    # Archive every small preparation artifact before creating a completion seal.
    archive_artifacts = {}
    for name, identity in artifacts.items():
        raw, info = read_small(out / name)
        need(info["sha256"] == identity["sha256"], "OUTPUT_CHANGED_BEFORE_ARCHIVE")
        archive_artifacts[name] = write_new(archive / "preparation" / name, raw)
    need(archive_artifacts == artifacts, "ARCHIVE_ARTIFACT_MISMATCH")
    seal = {"status": "SEALED_PREPARATION_NOT_EXPERIMENT_FREEZE", "artifacts": artifacts,
            "input_snapshot": str(archive / "INPUT_SNAPSHOT.json"),
            "parent_chunking": "unicode_codepoints_whitespace200_stride150_stop_at_final_v1",
            "runtime_paths": {"A_P_B_C_D": "runtime_parent_only.jsonl", "E": "runtime_oracle_e.jsonl",
                              "scorer_after_predictions": "scoring_targets.jsonl"}}
    write_json(archive / "PREPARATION_SEAL.json", seal)
    write_json(out / "PREPARATION_SEAL.json", seal)
    event("preparation_complete", **summary)


def _preparation_flow_fixture(prepare_function, cache_mutation=None):
    """Run the REAL prepare() control flow against a tiny in-memory filesystem.

    External corpus/package/filesystem operations are adapters; parent matching,
    export validation, cache comparison, archive copying and sealing use the
    production control flow. Does not touch real files, env vars or sys.modules.
    An earlier prepare function may be passed by an external regression check.
    """
    import builtins
    from types import SimpleNamespace
    files, directories, events = {}, set(), []
    class MemoryPath:
        def __init__(self, value):
            self.value = str(value)
        def __str__(self):
            return self.value
        def __truediv__(self, other):
            return MemoryPath(self.value + "/" + str(other))
        def resolve(self):
            return self
        def absolute(self):
            return self
        def is_relative_to(self, other):
            return self.value == str(other) or self.value.startswith(str(other) + "/")
        def exists(self):
            return self.value in files or self.value in directories
        def open(self, mode):
            need(mode == "rb", "FIXTURE_UNEXPECTED_OPEN_MODE")
            return io.BytesIO(files[self.value])
    repo, cache, archive = (MemoryPath("/fixture/" + x) for x in ("repo", "cache", "archive"))
    # Preserve constants/function bindings of the actual prepare() under test.
    g = dict(prepare_function.__globals__)
    out = repo / g["OUT_REL"]
    def clone(value):
        return strict_json(canonical(value))
    def stamp(path):
        key = str(path)
        return [56, int(digest(key.encode())[:8], 16), len(files[key]), 100, 100]
    def fp(path):
        key = str(path)
        return {"path": key, "stat": stamp(path), "sha256": digest(files[key])}
    def read(path):
        return files[str(path)], fp(path)
    def write(path, raw):
        key = str(path)
        need(key not in files, "FIXTURE_OVERWRITE_REFUSED")
        files[key] = raw
        return {"sha256": digest(raw), "bytes": len(raw)}
    def new_dir(path):
        need(not path.exists(), "FIXTURE_EXISTING_DIRECTORY_REFUSED")
        directories.add(str(path))
    parents, oracle, targets, metadata = [], [], [], []
    matched = defaultdict(list)
    blob_path, sidecar_path = str(cache / "blob"), str(cache / "blob.json")
    files[blob_path], files[sidecar_path] = b"synthetic-parquet-adapter", b"synthetic-sidecar"
    for i, (qid, child_count) in enumerate(sorted(CHILD_COUNTS.items())):
        title, text = "Parent " + str(i), "Synthetic evidence " + str(i)
        parents.append({"qid": qid, "question_ur": "synthetic question",
                        "parent_source_instance_id": "parent" + str(i),
                        "parent_title": title, "parent_normalized_title": norm(title)})
        oracle.append({"qid": qid, "question_ur": "synthetic question", "urbench_facts": ["oracle fixture"]})
        targets.append({"qid": qid, "children": [{"source_instance_id": qid + str(j),
            "title": f"Child {i} {j}", "normalized_title": f"child {i} {j}"} for j in range(child_count)]})
        def add_page(page_title, page_text, row):
            matched[norm(page_title)].append({"raw_page_id": str(row), "raw_url": "fixture:" + str(row),
                "raw_title": page_title, "raw_text": page_text, "raw_text_sha256": digest(page_text.encode()),
                "location": {"shard_index": 0, "blob_path": blob_path, "blob_sha256": digest(files[blob_path]),
                             "row_group": 0, "row_in_group": row, "row_in_shard": row}})
            metadata.append({"title": page_title, "text": " ".join(page_text.split())})
        add_page(title, text, i)
        if i == 0:
            add_page(title.upper(), "Separate colliding page", 25)
    lines = [(canonical(row) + "\n").encode() for row in metadata]
    positions, pos = [], 0
    for line in lines:
        positions.append(pos)
        pos += len(line)
    class MemoryOffsets(list):
        ndim = 1
        dtype = SimpleNamespace(kind="i", itemsize=8)
        @property
        def shape(self):
            return (len(self),)
    files[str(repo / g["META_REL"])] = b"".join(lines)
    files[str(repo / g["OFFSETS_REL"])] = b"synthetic-offset-adapter"
    files[str(repo / g["INDEX_REL"])] = b"synthetic-index-bytes"
    files[str(repo / g["AUDIT_REL"])] = b"synthetic-protected-input"
    small = {g["AUDIT_REL"]: fp(repo / g["AUDIT_REL"])}
    raw_inputs = {g["AUDIT_REL"]: files[str(repo / g["AUDIT_REL"])]}
    initial_sources = dict(files)
    inventory = [{"dataset": "AI-ModelScope/wikipedia", "Source": "SDK", "Revision": "master",
                  "FilePath": "20231101.en/train-00000-of-00041.parquet", "shard_index": 0,
                  "sidecar": fp(MemoryPath(sidecar_path)), "blob_path": blob_path,
                  "blob_stat": stamp(MemoryPath(blob_path))}]
    discoveries = 0
    def discover(_):
        nonlocal discoveries
        discoveries += 1
        rows, excluded_count = clone(inventory), 8
        if discoveries == 2 and cache_mutation:
            rows, excluded_count = cache_mutation(rows, excluded_count)
        return rows, excluded_count
    pq = SimpleNamespace(ParquetFile=lambda _: SimpleNamespace(close=lambda: None))
    pa = SimpleNamespace(parquet=pq)
    np = SimpleNamespace(load=lambda *a, **kw: MemoryOffsets(positions))
    def import_adapter(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("pyarrow", "pyarrow.parquet"):
            return pq if name.endswith(".parquet") and fromlist else pa
        if name == "numpy":
            return np
        return builtins.__import__(name, globals, locals, fromlist, level)
    g.update({"__builtins__": dict(vars(builtins), __import__=import_adapter),
        "os": SimpleNamespace(environ={"SLURM_JOB_ID": "SYNTHETIC"}, umask=lambda _: None),
        "socket": SimpleNamespace(gethostname=lambda: "synthetic_compute"),
        "N_SHARDS": 1, "N_PAGES": 26, "N_VECTORS": len(metadata), "INDEX_BYTES": len(b"synthetic-index-bytes"),
        "no_symlinks": lambda p: p, "file_stamp": stamp, "fingerprint": fp,
        "new_directory": new_dir, "write_new": write, "read_small": read,
        "write_json": lambda p, value: write(p, (canonical(value) + "\n").encode()),
        "write_jsonl": lambda p, rows: write(p, b"".join((canonical(r) + "\n").encode() for r in rows)),
        "environment": lambda: {"fixture": True}, "git_state": lambda _: {"head": "synthetic", "status": ""},
        "inspect_small_inputs": lambda _: (clone(small), dict(raw_inputs)),
        "discover_shards": discover, "validate_parquet": lambda *a: [26],
        "matched_pages": lambda *a: clone(matched),
        "load_cohort": lambda *a: (clone(parents), clone(oracle), clone(targets), {"audit_status": "AUDIT_PASS"}),
        "event": lambda stage, **fields: events.append({"stage": stage, **fields})})
    function = types.FunctionType(prepare_function.__code__, g)
    error = None
    try:
        function(SimpleNamespace(repo=repo, cache=cache, archive=archive))
    except RuntimeError as exc:
        error = str(exc)
    need(all(files[k] == raw for k, raw in initial_sources.items()), "FIXTURE_SOURCE_MUTATED")
    return {"error": error, "files": files, "events": events, "out": str(out),
            "archive": str(archive), "discoveries": discoveries}


def self_test():
    """Small synthetic checks. No files, subprocesses or real material."""
    checks = []
    def check(name, condition):
        need(condition, "SELF_TEST_FAILED: " + name)
        checks.append(name)
    def rejects(name, fn):
        try:
            fn()
        except PrepError:
            checks.append(name)
        else:
            raise PrepError("SELF_TEST_DID_NOT_REJECT: " + name)
    fixture = {"url": "https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/wikipedia/repo?Source=SDK&Revision=master&FilePath=20231101.en%2Ftrain-00000-of-00041.parquet", "etag": None}
    check("strict_wikipedia_selector", sidecar_identity(fixture)["shard_index"] == 0)
    nonwiki = {**fixture, "url": fixture["url"].replace("AI-ModelScope/wikipedia", "google/boolq")}
    check("exclude_non_wikipedia", sidecar_identity(nonwiki) is None)
    rejects("duplicate_url_query", lambda: sidecar_identity({**fixture, "url": fixture["url"] + "&Revision=master"}))
    rejects("duplicate_json_key", lambda: strict_json('{"qid":"a","qid":"b"}'))
    rejects("nonfinite_json", lambda: strict_json('{"v":NaN}'))
    # Cache equality remains strict; a list replacing the count is rejected.
    cache_fixture = [{"shard_index": 0, "blob_path": "/fixture/blob", "blob_stat": [56, 1, 20, 30, 40],
                      "sidecar": {"path": "/fixture/blob.json", "sha256": "a"*64, "stat": [56, 2, 50, 60, 70]}}]
    check("unchanged_cache_passes", cache_recheck_report(cache_fixture, 8, cache_fixture, 8)["status"] == "PASS")
    check("overwritten_count_is_rejected", cache_recheck_report(cache_fixture, [], cache_fixture, 8)["status"] == "FAIL")
    check("changed_excluded_count_rejected", cache_recheck_report(cache_fixture, 8, cache_fixture, 9)["status"] == "FAIL")
    check("changed_shard_count_rejected", cache_recheck_report(cache_fixture, 8, [], 8)["status"] == "FAIL")
    for key in ("blob_stat", "sidecar_stat"):
        for position, label in enumerate(("device", "inode", "size", "mtime", "ctime")):
            altered = strict_json(canonical(cache_fixture))
            stamp_value = altered[0]["blob_stat"] if key == "blob_stat" else altered[0]["sidecar"]["stat"]
            stamp_value[position] += 1
            check(key + "_" + label + "_change_rejected",
                  cache_recheck_report(cache_fixture, 8, altered, 8)["status"] == "FAIL")
    for key in ("blob_path", "sidecar_hash", "sidecar_path", "shard_index"):
        altered = strict_json(canonical(cache_fixture))
        if key == "blob_path":
            altered[0][key] = "/different"
        elif key == "sidecar_hash":
            altered[0]["sidecar"]["sha256"] = "b"*64
        elif key == "sidecar_path":
            altered[0]["sidecar"]["path"] = "/different.json"
        else:
            altered[0][key] = 1
        check(key + "_change_rejected", cache_recheck_report(cache_fixture, 8, altered, 8)["status"] == "FAIL")
    flow = _preparation_flow_fixture(prepare)
    check("full_flow_reaches_completion_after_collision_loop", flow["error"] is None and
          flow["events"][-1]["stage"] == "preparation_complete" and flow["discoveries"] == 2)
    out_seal = flow["files"][flow["out"] + "/PREPARATION_SEAL.json"]
    check("full_flow_both_seals_identical", out_seal == flow["files"][flow["archive"] + "/PREPARATION_SEAL.json"])
    seal = strict_json(out_seal)
    check("full_flow_archive_hashes_verified", all(
        digest(flow["files"][flow["archive"] + "/preparation/" + name]) == identity["sha256"]
        for name, identity in seal["artifacts"].items()))
    def change_device(rows, count):
        rows[0]["blob_stat"][0] += 1
        return rows, count
    changed = _preparation_flow_fixture(prepare, change_device)
    report_path = changed["out"] + "/cache_recheck.json"
    check("full_flow_cache_failure_recorded_before_stop", changed["error"] is not None and
          changed["error"].startswith("CACHE_CHANGED_DURING_PREPARATION") and
          strict_json(changed["files"][report_path])["status"] == "FAIL")
    check("full_flow_failed_cache_has_no_seal", not any(p.endswith("PREPARATION_SEAL.json") for p in changed["files"]))
    def page(text, title="Synthetic Parent", pid="s0", position=0):
        return {"raw_page_id": pid, "raw_url": "https://example.invalid/fixture",
            "raw_title": title, "raw_text": text, "raw_text_sha256": digest(text.encode()),
            "location": {"shard_index": 0, "blob_path": "/synthetic", "blob_sha256": "0"*64,
                         "row_group": 0, "row_in_group": position, "row_in_shard": position}}
    text = "\n" + " \t".join(["اردو", "é", "x\u0301", *[str(i) for i in range(197)]]) + "\n"
    p = resolve_parent("Synthetic Parent", [page(text)])
    chunks = local_chunks(p)
    check("new_chunker_stops_at_final", len(chunks) == 1)
    check("legacy_extra_tail_200_words", len(legacy_hashes(text)) == 2)
    check("unicode_original_whitespace_offsets", chunks[0]["text"] == text[chunks[0]["chunk_char_start"]:chunks[0]["chunk_char_end"]])
    copies = resolve_parent("Synthetic Parent", [page(text), page(text, position=1)])
    check("identical_copies_preserve_locations", len(copies["locations"]) == 2)
    rejects("conflicting_copies", lambda: resolve_parent("Synthetic Parent", [page(text), page("different")]))
    case = resolve_parent("Synthetic Parent", [page(text), page("other", title="SYNTHETIC PARENT", pid="s1")])
    check("unique_exact_case_resolution", case["raw_page_id"] == "s0")
    rejects("remaining_ambiguity", lambda: resolve_parent("synthetic parent", [page(text), page("other", title="SYNTHETIC PARENT", pid="s1")]))
    rejects("missing_parent", lambda: resolve_parent("missing", []))
    indexed = [{"text_sha256": h, "global_row": i} for i,h in enumerate(legacy_hashes(text))]
    check("legacy_correspondence", correspondence(p, indexed)["status"] == "PASS")
    rejects("wrong_tail_not_accepted", lambda: correspondence(p, indexed[:1]))
    rejects("extra_metadata_chunk", lambda: correspondence(p, indexed + indexed[:1]))
    # Reproduce the two observed COUNT patterns with entirely synthetic text.
    # The second pattern also exercises the intentional extra legacy tail.
    for words, extras, expected_n in ((4186, 6, 28), (1983, 1, 14)):
        sample_text = " ".join("word" + str(i) for i in range(words))
        resolved = resolve_parent("Synthetic Parent", [page(sample_text),
            page("different page", title="SYNTHETIC PARENT", pid="s1")])
        own = [{"text_sha256": h, "global_row": i, "metadata_title": "Synthetic Parent"}
               for i,h in enumerate(legacy_hashes(sample_text))]
        other = [{"text_sha256": digest(("other" + str(i)).encode()),
                  "global_row": expected_n + i, "metadata_title": "SYNTHETIC PARENT"}
                 for i in range(extras)]
        bucket = own + other
        rejects(f"old_collision_failure_{expected_n}_plus_{extras}",
                lambda: correspondence(resolved, bucket))
        selected, excluded = select_indexed_parent_rows(resolved, bucket)
        check(f"collision_corrected_{expected_n}_plus_{extras}",
              len(selected) == expected_n and correspondence(resolved, selected)["status"] == "PASS")
        check(f"all_excluded_rows_preserved_{expected_n}_plus_{extras}", excluded == other)
        # An excluded page cannot supply a missing parent hash, even if its hash
        # happens to match. No hash-based selection or mixed-page repair.
        replacement = {**own[0], "metadata_title": "SYNTHETIC PARENT"}
        incomplete, _ = select_indexed_parent_rows(resolved, own[1:] + other + [replacement])
        rejects(f"collision_cannot_fill_missing_parent_chunk_{expected_n}",
                lambda: correspondence(resolved, incomplete))
        duplicate, _ = select_indexed_parent_rows(resolved, own + [own[0]] + other)
        rejects(f"selected_page_extra_chunk_still_rejected_{expected_n}",
                lambda: correspondence(resolved, duplicate))
    own = [{**r, "metadata_title": " Synthetic_Parent "} for r in indexed]
    selected, excluded = select_indexed_parent_rows(p, own)
    check("exact_case_underscore_whitespace_rule", selected == own and not excluded)
    only_other = [{**r, "metadata_title": "SYNTHETIC PARENT"} for r in own]
    selected, excluded = select_indexed_parent_rows(p, only_other)
    rejects("no_exact_case_parent_match", lambda: correspondence(p, selected))
    rejects("unrelated_normalized_bucket", lambda: select_indexed_parent_rows(
        p, [{**own[0], "metadata_title": "Unrelated"}]))
    selected, _ = select_indexed_parent_rows(copies, own + own)
    check("permitted_raw_copy_factor_unchanged", correspondence(copies, selected)["indexed_copy_count"] == 2)
    rows = [{"title": "Synthetic Parent", "text": "fixture"}, {"title": "Other", "text": "other"}]
    lines = [(canonical(r)+"\n").encode() for r in rows]
    found, info = scan_metadata(io.BytesIO(b"".join(lines)), [0, len(lines[0])], {norm("Synthetic Parent")}, 2)
    check("metadata_correct_offset_and_global_row", found[norm("Synthetic Parent")][0]["global_row"] == 0 and info["rows"] == 2)
    rejects("negative_offset", lambda: scan_metadata(io.BytesIO(b"".join(lines)), [-1,len(lines[0])], set(), 2))
    rejects("wrong_offset", lambda: scan_metadata(io.BytesIO(b"".join(lines)), [0,len(lines[0])+1], set(), 2))
    rejects("wrong_row_count", lambda: scan_metadata(io.BytesIO(b"".join(lines)), [0,len(lines[0]),999], set(), 3))
    decisions = {i: {"qid": "q", "verdict": "OK", "parent_source_instance_id": "ps",
        "parent_title": "Synthetic Parent", "child_source_instance_id": "c"+str(i),
        "child_title": "Child " + str(i), "rationale": "FORBIDDEN_RATIONALE"} for i in range(2)}
    dev = [{"urbench_qid": "q", "question_ur": "سوال", "urbench_facts": ["ORACLE_ONLY_SENTINEL"],
            "answer": "ANSWER_FORBIDDEN", "question_en": "ENGLISH_QUESTION_FORBIDDEN"}]
    parents, oracle, targets = export_cohort(decisions, "OK", {"q":"سوال"}, dev, {"q":2})
    runtime = [{**parents[0], "page": p, "chunks": chunks}]
    validate_exports(runtime, oracle, targets)
    check("two_children_one_parent_runtime", len(runtime) == 1 and len(targets[0]["children"]) == 2)
    check("parent_export_no_privileged_fields", all(s not in canonical(runtime) for s in
          ["ORACLE_ONLY_SENTINEL", "ANSWER_FORBIDDEN", "ENGLISH_QUESTION_FORBIDDEN", "FORBIDDEN_RATIONALE", "Child 0"]))
    rejects("unknown_runtime_field", lambda: validate_exports([{**runtime[0], "answer": "bad"}], oracle, targets))
    rejects("cohort_count_drift", lambda: export_cohort(decisions, "OK", {"q":"سوال"}, dev, {"q":1}))
    rejects("Urdu_join_drift", lambda: export_cohort(decisions, "OK", {"q":"different"}, dev, {"q":2}))
    arrow_status = "NOT_AVAILABLE_LOCALLY; REMOTE_ARROW_TEST_REQUIRED"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pass
    else:
        # Exercise real Parquet I/O entirely in memory, including row-group and
        # batch boundaries, without a synthetic-file write in remote Codex.
        source = pa.table({"id": ["0","1","2","3"], "url": ["u0","u1","u2","u3"],
            "title": ["Other", "Synthetic Parent", "Other2", "Synthetic Parent"],
            "text": ["no", "first", "no2", "second"]})
        sink = pa.BufferOutputStream()
        pq.write_table(source, sink, row_group_size=2)
        pf = pq.ParquetFile(pa.BufferReader(sink.getvalue()))
        try:
            sizes = validate_parquet(pf, pa)
            found = matched_pages(pf, sizes, {norm("Synthetic Parent")},
                {"shard_index":0,"blob_path":"/synthetic","blob_sha256":"0"*64}, text_batch=1)
        finally:
            pf.close()
        chosen = found[norm("Synthetic Parent")]
        check("real_arrow_row_group_offsets", [x["location"]["row_in_shard"] for x in chosen] == [1,3])
        check("real_arrow_selected_text", [x["raw_text"] for x in chosen] == ["first","second"])
        arrow_status = "PASS"
    event("self_test", status="PASS", checks_passed=len(checks), checks=checks,
          pyarrow_in_memory=arrow_status, files_written=0)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-small-inputs", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--repo", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DEFAULT)
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        if args.check_small_inputs:
            repo = args.repo.resolve()
            identities, _ = inspect_small_inputs(repo)
            shards, excluded = discover_shards(args.cache.absolute())
            event("small_input_preflight", status="PASS", environment=environment(),
                  sha256={p:i["sha256"] for p,i in identities.items()}, git=git_state(repo),
                  selected_wikipedia_shards=len(shards), excluded_blobs=excluded,
                  files_written=0, experiment_state="NOT_FROZEN_NOT_RUN")
        else:
            prepare(args)
        return 0
    except PrepError as exc:
        print(canonical({"status":"STOPPED", "error":str(exc), "outputs_must_not_be_deleted":True,
                         "experiment_state":"NOT_FROZEN_NOT_RUN"}), file=sys.stderr)
        return 2
    except Exception as exc:
        # Avoid library exceptions revealing raw passages/annotations in logs.
        print(canonical({"status":"STOPPED", "error_type":type(exc).__name__,
                         "message":"Operational error; preserve partial files for scoped review",
                         "experiment_state":"NOT_FROZEN_NOT_RUN"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
