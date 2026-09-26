#!/usr/bin/env python3
"""Post-scoring exact-title metadata coverage. Standard library only; no retrieval.

Production input pins cannot be overridden at the CLI. --check-inputs is read-only.
--run writes only a fresh diagnostic directory; failed runs must be preserved.
See REACHABILITY_PROBE_GUIDE.md for scope and remote verification requirements.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction

VERSION = "urbench.target_reachability.v1"
ROOT = Path("/mnt/home/user41/URBench")
STAGE = "outputs/efbpt/bridge_pilot_n25/v1/"
OUT = "outputs/efbpt/bridge_pilot_n25/target_reachability_diagnostic_v1"
META = "rag/index/wikipedia_full_meta.jsonl"
CODE = "eval/error_analysis_tests/efbpt/"
META_BYTES = 25866666236
META_ROWS = 23963971
MAX_LINE = 8 * 1024 * 1024
MAX_SMALL = 32 * 1024 * 1024
ARMS = ("A", "P", "B", "C", "D", "E")
HEX = re.compile(r"[0-9a-f]{64}\Z")
QIDS = dict(zip(
    ("15885efe6f91724c16d9", "25a088d9d2ce674e639a", "3295844627a9bd1b9135",
     "3f8a1bd6bf3a967cdeb6", "48da75d87c66754ccc2e", "5897ec22db850f7b416e",
     "6c43fa359095fd0845f5", "7281474f2760dce03f39", "73c52134ab2a903d86db",
     "73ca2ef1da65b2a2ebe6", "7b84d2bc643ddc2085f0", "7be51fdef30345de666e",
     "7c3759cc1da78e9fbd79", "7f4effbc97ab2b5fd4a7", "80aa769f55b14c1e4d8d",
     "8cfde6ee28d059a5aff6", "8d06b619a7045ed02f51", "8d3ddaee20ad48edc066",
     "a0896de3fd13cd0f3e16", "b747938f597b09e43603", "e6d3973ed3feb8a42928",
     "e87b63e92165b417d37f", "f2859b2ce17b5f5a6ad9", "f43533225534420816d6",
     "ff3811735ededd8ec3a7"),
    (1, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1)))
# (repository-relative path, SHA-256, bytes). Values are from accepted audits.
PINS = {
    "activation": ("docs/EFBPT_BRIDGE_PILOT_N25_ACTIVATION.json",
        "bcc684cfd3c91ae26839a664a390529d21e40ffa6faea4bd407081829c5dbfe9", 13927),
    "retrieval_seal": (STAGE + "retrieve_v2/STAGE_SEAL.json",
        "d3ef88f42a911dac9b0a26804f1b16bba797a3a73c11377b526607dc4eaeae15", 1078),
    "scoring_seal": (STAGE + "score_v2/SCORING_SEAL.json",
        "b62b09a07aaa7809b61e1a63bf42b92a3a9b364080d65c183c03c66a6b83c792", 4862),
    "scores": (STAGE + "score_v2/scores.json",
        "7233ca175194573791b289fb9cc09cae015dc8c4b72596c9ea93e3bcb4afc332", 24478),
    "records": (STAGE + "retrieve_v2/prediction_records.jsonl",
        "556fef6357c52a5d3ec3b732a60731ecbbda0b920f53ce3aec29dba4099f5ab2", 3976517),
    "predictions": (STAGE + "retrieve_v2/predictions.jsonl",
        "4604b8354a066fcafbef6e63f7c73328ac0de13e45b44b6c65104308f793311a", 1384225),
    "targets": (STAGE + "preparation_r3/scoring_targets.jsonl",
        "5643935822c0a0227a975047b64b23d748eb61b77fa8eecf824f1deab026373b", 6628),
    "core": (CODE + "bridge_pilot_core.py",
        "0a02ee75ea4b86a501972441db3d6874eeda1f2d4f30c75cf19c18571f7b5a67", 23584),
}


class ProbeError(Exception):
    pass


def need(ok, message):
    if not ok:
        raise ProbeError(message)


def norm(title):
    return " ".join(str(title).replace("_", " ").strip().lower().split())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def integer(x):
    return type(x) is int


def utc():
    return datetime.now(timezone.utc).isoformat()


def event(status, **fields):
    print(json.dumps({"status": status, **fields}, ensure_ascii=True,
                     allow_nan=False, sort_keys=True), flush=True)


def strict_json(raw, label):
    def pairs(items):
        obj = {}
        for key, value in items:
            need(key not in obj, f"DUPLICATE_JSON_KEY: {label}")
            obj[key] = value
        return obj

    def constant(_):
        raise ProbeError(f"NONFINITE_JSON: {label}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProbeError(f"INVALID_JSON_OR_UTF8: {label}") from exc


def jsonl(raw, label):
    need(raw.endswith(b"\n"), f"JSONL_FINAL_NEWLINE: {label}")
    rows = raw.split(b"\n")[:-1]
    need(all(rows), f"JSONL_BLANK_LINE: {label}")
    return [strict_json(row, f"{label}:{i}") for i, row in enumerate(rows)]


def plain_path(path):
    """Require an existing regular file and reject symlinks on the whole path."""
    path = Path(os.path.abspath(path))
    for node in reversed((path, *path.parents)):
        info = node.lstat()
        need(not stat.S_ISLNK(info.st_mode), f"SYMLINK_REFUSED: {node}")
        if node != path:
            need(stat.S_ISDIR(info.st_mode), f"NON_DIRECTORY_PARENT: {node}")
    need(stat.S_ISREG(info.st_mode), f"REGULAR_FILE_REQUIRED: {path}")
    return path


def signature(info):
    # Only compare stats within this process/mount, never against old node st_dev.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def open_input(path):
    path = plain_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode), f"REGULAR_FILE_REQUIRED: {path}")
        need(signature(before) == signature(path.stat()), f"INPUT_RACE: {path}")
        return os.fdopen(fd, "rb"), signature(before)
    except BaseException:
        os.close(fd)
        raise


def unchanged(stream, path, before):
    need(signature(os.fstat(stream.fileno())) == before,
         f"INPUT_CHANGED_ON_DESCRIPTOR: {path}")
    need(signature(plain_path(path).stat()) == before, f"INPUT_REPLACED: {path}")


def read_pinned(path, sha, size=None):
    need(isinstance(sha, str) and HEX.fullmatch(sha), "EXPECTED_SHA_REQUIRED")
    stream, before = open_input(path)
    with stream:
        need(before[2] <= MAX_SMALL, f"SMALL_INPUT_TOO_LARGE: {path}")
        raw = stream.read(MAX_SMALL + 1)
        unchanged(stream, path, before)
    need(len(raw) <= MAX_SMALL, f"SMALL_INPUT_TOO_LARGE: {path}")
    need(size is None or len(raw) == size, f"INPUT_SIZE_DRIFT: {path}")
    need(digest(raw) == sha, f"INPUT_HASH_DRIFT: {path}")
    return raw


def resolved_declared(root, value):
    need(isinstance(value, str) and value, "DECLARED_PATH_REQUIRED")
    path = Path(value)
    need(".." not in path.parts, "PARENT_TRAVERSAL_REFUSED")
    return Path(os.path.abspath(path if path.is_absolute() else root / path))


def find_descriptor(container, root, wanted):
    """Discover only exact path descriptors; never infer a hash from a filename."""
    found = []

    def visit(obj, pointer):
        if isinstance(obj, dict):
            if "path" in obj and resolved_declared(root, obj["path"]) == wanted:
                need("sha256" in obj and "bytes" in obj,
                     f"INCOMPLETE_DESCRIPTOR: {pointer}")
                found.append((obj, pointer))
            for key, value in obj.items():
                visit(value, pointer + "/" + str(key))
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                visit(value, pointer + "/" + str(index))

    visit(container, "")
    need(len(found) == 1, f"UNIQUE_EXACT_DESCRIPTOR_REQUIRED: {wanted}")
    spec, pointer = found[0]
    need(isinstance(spec["sha256"], str) and HEX.fullmatch(spec["sha256"]),
         "DESCRIPTOR_SHA_REQUIRED")
    need(integer(spec["bytes"]) and spec["bytes"] > 0, "DESCRIPTOR_BYTES_REQUIRED")
    return {"path": str(wanted), "sha256": spec["sha256"], "bytes": spec["bytes"],
            "descriptor_pointer": pointer}


def bound_artifact(seal, name, pin):
    spec = seal["artifacts"][name]
    need(spec["sha256"] == pin[1] and spec["bytes"] == pin[2],
         f"SEAL_ARTIFACT_DISAGREEMENT: {name}")


def check_normalizer(source):
    """Compare a single return-expression AST, normalizing its parameter name."""
    tree = ast.parse(source)
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "norm"]
    need(len(funcs) == 1, "FROZEN_NORMALIZER_NOT_FOUND")
    func = funcs[0]
    need(len(func.args.args) == 1, "NORMALIZER_SIGNATURE_DRIFT")
    body = [x for x in func.body if not (isinstance(x, ast.Expr)
            and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str))]
    need(len(body) == 1 and isinstance(body[0], ast.Return), "NORMALIZER_BODY_DRIFT")

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == func.args.args[0].arg:
                return ast.copy_location(ast.Name(id="title", ctx=node.ctx), node)
            return node

    expr = Rename().visit(body[0].value)
    wanted = ast.parse('" ".join(str(title).replace("_", " ").strip().lower().split())',
                       mode="eval").body
    need(ast.dump(expr) == ast.dump(wanted), "NORMALIZER_EXPRESSION_DRIFT")


def validate_targets(rows, cohort):
    need(len(rows) == len(cohort), "TARGET_ROW_COUNT")
    titles, ids, qids = {}, set(), set()
    for row in rows:
        need(isinstance(row, dict) and set(row) == {"qid", "children"}, "TARGET_ROW_SCHEMA")
        qid = row["qid"]
        need(isinstance(qid, str) and qid in cohort and qid not in qids, "TARGET_QID")
        qids.add(qid)
        children = row["children"]
        need(isinstance(children, list) and len(children) == cohort[qid], "TARGET_MULTIPLICITY")
        for child in children:
            need(isinstance(child, dict) and set(child) ==
                 {"source_instance_id", "title", "normalized_title"}, "TARGET_CHILD_SCHEMA")
            need(all(isinstance(x, str) and x.strip() for x in child.values()), "TARGET_STRINGS")
            title, sid = child["normalized_title"], child["source_instance_id"]
            need(title == norm(child["title"]) and title, "TARGET_NORMALIZATION")
            need(title not in titles and sid not in ids, "TARGET_DISTINCTNESS")
            ids.add(sid)
            titles[title] = {"qid": qid, "source_instance_id": sid,
                             "normalized_title": title}
    need(qids == set(cohort) and len(titles) == sum(cohort.values()), "TARGET_BOUNDARY")
    return titles


def aggregate(candidates):
    best = {}
    for c in candidates:
        title = norm(c["title"])
        item = {"normalized_title": title, "score": c["score"], "best_global_row": c["global_row"]}
        prev = best.get(title)
        if prev is None or (-item["score"], item["best_global_row"]) < (-prev["score"], prev["best_global_row"]):
            best[title] = item
    return sorted(best.values(), key=lambda x: (-x["score"], x["normalized_title"], x["best_global_row"]))[:10]


def collect_controls(records, predictions, targets, cohort, activation_sha, total_rows):
    need(len(records) == len(cohort) * len(ARMS), "PREDICTION_COUNT")
    need([r["prediction"] for r in records] == predictions, "PREDICTION_VIEW_DRIFT")
    seen, controls, anywhere = set(), {}, set()
    own = {t: set() for t in targets}
    top = {t: set() for t in targets}
    for r in records:
        qid, arm = r["qid"], r["arm"]
        need(isinstance(qid, str) and isinstance(arm, str), "PREDICTION_ID_TYPES")
        need(qid in cohort and arm in ARMS and (qid, arm) not in seen, "PREDICTION_IDENTITY")
        seen.add((qid, arm))
        need(r["activation_sha256"] == activation_sha and r["search_budget"] == 100,
             "PREDICTION_BINDING_OR_BUDGET")
        p = r["prediction"]
        need(p["qid"] == qid and p["arm"] == arm, "NESTED_PREDICTION_IDENTITY")
        candidates, provenance = p["candidates"], r["candidate_provenance"]
        need(len(candidates) == len(provenance) == 100, "CANDIDATE_COUNT")
        row_ids, previous = set(), float("inf")
        for c, proof in zip(candidates, provenance):
            rid, score, title = c["global_row"], c["score"], c["title"]
            need(integer(rid) and 0 <= rid < total_rows and rid not in row_ids, "CANDIDATE_ROW")
            row_ids.add(rid)
            need(type(score) in (int, float) and math.isfinite(score) and score <= previous,
                 "CANDIDATE_SCORE")
            previous = score
            need(isinstance(title, str) and norm(title), "CANDIDATE_TITLE")
            nt = norm(title)
            need(proof["global_row"] == rid and integer(proof["byte_offset"])
                 and proof["byte_offset"] >= 0, "CANDIDATE_PROVENANCE_ROW")
            need(isinstance(proof["metadata_line_sha256"], str)
                 and HEX.fullmatch(proof["metadata_line_sha256"]), "CANDIDATE_PROVENANCE_HASH")
            control = {"title": title, "byte_offset": proof["byte_offset"],
                       "metadata_line_sha256": proof["metadata_line_sha256"]}
            need(rid not in controls or controls[rid] == control, "CANDIDATE_CONTROL_CONFLICT")
            controls[rid] = control
            if nt in targets:
                anywhere.add(nt)
                if targets[nt]["qid"] == qid:
                    own[nt].add(arm)
        ranked = p["ranked_titles"]
        need(len(ranked) == 10 and ranked == aggregate(candidates), "FROZEN_RANKING_DRIFT")
        for item in ranked:
            nt = item["normalized_title"]
            if nt in targets and targets[nt]["qid"] == qid:
                top[nt].add(arm)
    need(seen == {(q, a) for q in cohort for a in ARMS}, "PREDICTION_COMPLETENESS")
    need(anywhere, "NO_OBSERVED_POSITIVE_CONTROLS")
    return controls, anywhere, own, top


def check_output_fresh(root):
    out = root / OUT
    need(not os.path.lexists(out), f"OUTPUT_EXISTS_PRESERVE_AND_STOP: {out}")
    parent = out.parent
    need(parent.is_dir(), f"OUTPUT_PARENT_MISSING: {parent}")
    for node in (parent, *parent.parents):
        need(not node.is_symlink(), f"OUTPUT_PARENT_SYMLINK: {node}")
    need(os.access(parent, os.W_OK | os.X_OK), "OUTPUT_PARENT_NOT_WRITABLE")
    return out


def prepare(root, script_sha, wrapper_sha):
    need(root == ROOT, "PRODUCTION_REPOSITORY_MUST_MATCH_REVIEWED_ROOT")
    need(Path(__file__).absolute() == root / (CODE + "efbpt_target_reachability_probe.py"),
         "INSTALL_PROBE_AT_REVIEWED_PATH")
    out = check_output_fresh(root)
    pins = dict(PINS)
    pins["probe"] = (CODE + "efbpt_target_reachability_probe.py", script_sha, None)
    pins["wrapper"] = (CODE + "efbpt_target_reachability_probe.sbatch", wrapper_sha, None)
    raw = {}
    # Authenticate seals and record bundles before allowing target parsing.
    for name in ("activation", "retrieval_seal", "scoring_seal", "scores", "records",
                 "predictions", "core", "probe", "wrapper"):
        path, sha, size = pins[name]
        raw[name] = read_pinned(root / path, sha, size)
    docs = {n: strict_json(raw[n], n) for n in ("activation", "retrieval_seal", "scoring_seal", "scores")}
    activation, retrieval, scoring = (docs[n] for n in ("activation", "retrieval_seal", "scoring_seal"))
    for seal, status in ((retrieval, "SEALED_PREDICTIONS"), (scoring, "SEALED_SCORES")):
        need(seal["status"] == status and seal["activation_sha256"] == PINS["activation"][1],
             "SEAL_STATUS_OR_ACTIVATION_DRIFT")
    need(retrieval["targets_sha256"] == PINS["targets"][1], "RETRIEVAL_TARGET_BINDING")
    bound_artifact(retrieval, "prediction_records.jsonl", PINS["records"])
    bound_artifact(retrieval, "predictions.jsonl", PINS["predictions"])
    bound_artifact(scoring, "scores.json", PINS["scores"])
    for label, key in (("targets", "targets"), ("predictions_seal", "retrieval_seal"),
                       ("predictions", "predictions"), ("prediction_records", "records")):
        desc = scoring["provenance"][label]
        pin = PINS[key]
        need(resolved_declared(root, desc["path"]) == root / pin[0]
             and desc["sha256"] == pin[1] and desc["bytes"] == pin[2], "SCORING_PROVENANCE_DRIFT")
    metadata = find_descriptor(activation["assets"], root, root / META)
    need(metadata["bytes"] == META_BYTES, "METADATA_DECLARED_SIZE_DRIFT")
    target_spec = find_descriptor(activation["exports"], root, root / PINS["targets"][0])
    need(target_spec["sha256"] == PINS["targets"][1] and target_spec["bytes"] == PINS["targets"][2],
         "ACTIVATION_TARGET_BINDING")
    need(plain_path(root / META).stat().st_size == META_BYTES, "METADATA_STAT_SIZE_DRIFT")
    check_normalizer(raw["core"])
    raw["targets"] = read_pinned(root / PINS["targets"][0], PINS["targets"][1], PINS["targets"][2])
    targets = validate_targets(jsonl(raw["targets"], "targets"), QIDS)
    records, predictions = jsonl(raw["records"], "records"), jsonl(raw["predictions"], "predictions")
    controls, anywhere, own, top = collect_controls(records, predictions, targets, QIDS,
                                                   PINS["activation"][1], META_ROWS)
    need(docs["scores"]["accepted_qids"] == 25 and docs["scores"]["accepted_pairs"] == 36,
         "SCORING_BOUNDARY_DRIFT")
    inputs = {n: {"path": str(root / p), "sha256": sha, "bytes": len(raw[n])}
              for n, (p, sha, _) in pins.items()}
    return {"output": out, "inputs": inputs, "metadata": metadata, "targets": targets,
            "controls": controls, "anywhere": anywhere, "own": own, "top": top}


def scan_metadata(path, expected_sha, expected_bytes, expected_rows, targets, controls,
                  progress_seconds=60):
    counts = {t: {"matching_rows": 0, "first_row": None, "first_byte_offset": None} for t in targets}
    observed_controls = set()
    h, offset, rows = hashlib.sha256(), 0, 0
    began = last = time.monotonic()
    stream, before = open_input(path)
    with stream:
        need(before[2] == expected_bytes, "METADATA_SIZE_BEFORE_SCAN")
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            need(len(line) <= MAX_LINE, f"METADATA_ROW_TOO_LONG: row={rows}")
            need(line.endswith(b"\n") and line.count(b"\n") == 1,
                 f"METADATA_LINE_BOUNDARY: row={rows}")
            h.update(line)
            row = strict_json(line, f"metadata row {rows}")
            need(isinstance(row, dict) and set(row) == {"title", "text"},
                 f"METADATA_SCHEMA: row={rows}")
            need(isinstance(row["title"], str) and norm(row["title"])
                 and isinstance(row["text"], str), f"METADATA_STRINGS: row={rows}")
            nt = norm(row["title"])
            if nt in counts:
                item = counts[nt]
                if item["matching_rows"] == 0:
                    item["first_row"], item["first_byte_offset"] = rows, offset
                item["matching_rows"] += 1
            if rows in controls:
                c = controls[rows]
                need(row["title"] == c["title"] and offset == c["byte_offset"]
                     and digest(line) == c["metadata_line_sha256"],
                     f"OBSERVED_CANDIDATE_ROW_MISMATCH: row={rows}")
                observed_controls.add(rows)
            offset += len(line)
            rows += 1
            need(rows <= expected_rows and offset <= expected_bytes, "METADATA_BOUNDARY_EXCEEDED")
            if rows % 10000 == 0 and time.monotonic() - last >= progress_seconds:
                elapsed = time.monotonic() - began
                event("SCAN_PROGRESS_NOT_A_RESULT", rows=rows, bytes_read=offset,
                      byte_percent=round(100 * offset / expected_bytes, 2), elapsed_seconds=round(elapsed, 1))
                last = time.monotonic()
        unchanged(stream, path, before)
    need(rows == expected_rows, "METADATA_ROW_COUNT_DRIFT")
    need(offset == expected_bytes, "METADATA_BYTE_COUNT_DRIFT")
    need(h.hexdigest() == expected_sha, "METADATA_HASH_DRIFT")
    need(observed_controls == set(controls), "CANDIDATE_CONTROL_ROWS_MISSING")
    return counts, {"sha256": h.hexdigest(), "bytes": offset, "rows": rows,
                    "candidate_rows_verified": len(observed_controls),
                    "elapsed_seconds": round(time.monotonic() - began, 3)}


def make_result(context, counts, scan, cohort):
    targets, anywhere, own, top = (context[k] for k in ("targets", "anywhere", "own", "top"))
    need(all(counts[t]["matching_rows"] > 0 for t in anywhere), "POSITIVE_CONTROL_TITLE_MISSING")
    results, per_qid = [], {}
    for qid, total in cohort.items():
        present = sum(counts[t]["matching_rows"] > 0 for t, info in targets.items() if info["qid"] == qid)
        per_qid[qid] = {"accepted_children": total, "exact_titles_present": present}
    for title, info in sorted(targets.items(), key=lambda x: (x[1]["qid"], x[1]["source_instance_id"])):
        present = counts[title]["matching_rows"] > 0
        results.append({**info, **counts[title], "exact_title_present": present,
                        "status": "PRESENT_EXACT_TITLE" if present else "ABSENT_UNDER_FROZEN_NORM",
                        "observed_anywhere_in_candidates": title in anywhere,
                        "own_question_candidate_arms": sorted(own[title]),
                        "own_question_top10_arms": sorted(top[title])})
    present_total = sum(r["exact_title_present"] for r in results)
    macro = sum((Fraction(x["exact_titles_present"], x["accepted_children"]) for x in per_qid.values()), Fraction()) / len(cohort)
    return {"schema": VERSION + ".result", "status": "VALIDATED_EXACT_TITLE_COVERAGE",
            "scope": "POST_OUTCOME_METADATA_DIAGNOSTIC_NOT_METHOD_EVALUATION",
            "historical_global_lineage": "UNESTABLISHED", "inputs": context["inputs"],
            "metadata_declaration": context["metadata"], "scan": scan,
            "normalization": "underscore_to_space; strip; lower; split_join_whitespace; no Unicode normalization",
            "positive_controls": {"titles_observed_anywhere": len(anywhere),
                "titles_observed_for_own_question": sum(bool(a) for a in own.values()),
                "all_observed_titles_present": True},
            "summary": {"questions": len(cohort), "accepted_titles": len(targets),
                "present_titles": present_total, "absent_titles": len(targets) - present_total,
                "question_macro_exact_title_coverage": float(macro),
                "question_macro_fraction": str(macro), "pair_micro_exact_title_coverage": present_total / len(targets),
                "questions_none_present": sum(x["exact_titles_present"] == 0 for x in per_qid.values()),
                "questions_all_present": sum(x["exact_titles_present"] == x["accepted_children"] for x in per_qid.values()),
                "questions_some_present": sum(0 < x["exact_titles_present"] < x["accepted_children"] for x in per_qid.values())},
            "per_question": per_qid, "targets": results,
            "limitations": ["Exact normalized title identity only; no alias/redirect/semantic absence proof.",
                "No vector-content alignment or global lineage proof.", "No retrieval rerun, changed pilot score, significance test or answer generation.",
                "Inputs are not held under filesystem locks; full stream hash and in-process stat checks cover the bytes read."]}


class Output:
    def __init__(self, path):
        self.path = path
        path.mkdir(mode=0o700)  # exclusive; no parents=True, no exist_ok
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.identity = (os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def write(self, name, value):
        need(name in {"START.json", "reachability.json", "DIAGNOSTIC_SEAL.json"}, "OUTPUT_NAME")
        info = self.path.lstat()
        need(stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == self.identity, "OUTPUT_DIRECTORY_REPLACED")
        data = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(self.fd)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        with os.fdopen(fd, "rb") as stream:
            need(stream.read(len(data) + 1) == data, "OUTPUT_READBACK_MISMATCH")
        return {"sha256": digest(data), "bytes": len(data)}

    def close(self):
        os.close(self.fd)


def stop_signal(signum, _frame):
    raise ProbeError(f"INTERRUPTED_SIGNAL_{signum}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe-sha256", required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    args = parser.parse_args(argv)
    output = None
    try:
        need(sys.version_info[:3] == (3, 10, 19), "PINNED_PYTHON_3_10_19_REQUIRED")
        need(sys.dont_write_bytecode, "RUN_PYTHON_WITH_MINUS_B")
        for value in (args.probe_sha256, args.wrapper_sha256):
            need(HEX.fullmatch(value), "CODE_DIGEST_REQUIRED")
        signal.signal(signal.SIGTERM, stop_signal)
        signal.signal(signal.SIGINT, stop_signal)
        context = prepare(args.repo, args.probe_sha256, args.wrapper_sha256)
        event("INPUTS_VERIFIED_METADATA_NOT_SCANNED", targets=len(context["targets"]),
              candidate_control_rows=len(context["controls"]),
              positive_control_titles=len(context["anywhere"]), metadata=context["metadata"],
              output=str(context["output"]))
        if args.check_inputs:
            return 0
        output = Output(context["output"])
        artifacts = {"START.json": output.write("START.json", {
            "schema": VERSION + ".start", "status": "STARTED_NOT_A_RESULT", "utc": utc(),
            "inputs": context["inputs"], "metadata": context["metadata"],
            "job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
            "expected_rows": META_ROWS, "output": str(context["output"])})}
        metadata = context["metadata"]
        counts, scan = scan_metadata(Path(metadata["path"]), metadata["sha256"], metadata["bytes"],
                                     META_ROWS, context["targets"], context["controls"])
        # Rehash only small inputs; metadata is authenticated during its single pass.
        for item in context["inputs"].values():
            read_pinned(Path(item["path"]), item["sha256"], item["bytes"])
        result = make_result(context, counts, scan, QIDS)
        artifacts["reachability.json"] = output.write("reachability.json", result)
        for name, identity in artifacts.items():
            read_pinned(context["output"] / name, identity["sha256"], identity["bytes"])
        seal = {"schema": VERSION + ".seal", "status": "SEALED_DIAGNOSTIC", "utc": utc(),
                "job_id": os.environ.get("SLURM_JOB_ID"), "artifacts": artifacts,
                "inputs": context["inputs"], "metadata": {**metadata, "observed": scan},
                "scope": result["scope"], "historical_global_lineage": "UNESTABLISHED"}
        seal_identity = output.write("DIAGNOSTIC_SEAL.json", seal)
        event("DIAGNOSTIC_COMPLETE", output=str(context["output"]), seal=seal_identity,
              summary=result["summary"])
        return 0
    except (ProbeError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            SyntaxError, RecursionError, OverflowError) as exc:
        event("STOPPED", error_type=type(exc).__name__, reason=str(exc),
              outputs_must_not_be_deleted=True, automatic_retry_allowed=False)
        return 2
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
