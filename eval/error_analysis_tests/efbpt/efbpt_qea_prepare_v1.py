#!/usr/bin/env python3
"""Prepare the QEA development cohort: exposure audit, eligibility, ordering, partition.

PREPARATION ONLY. No model call, no retrieval, no triage, no scoring. Writes one new
directory, outputs/efbpt/qea_preparation_v1_r2/, and refuses to overwrite it. Run 1
(outputs/efbpt/qea_preparation_v1/) is preserved unchanged and superseded: it scanned this
script's own manifest note as a code mention. Usage: python -B <this> <run-1 script copy>.

Content discipline: evidence rows are validated programmatically (pass/fail and counts
only). Nothing about reserved questions or evidence is printed or copied; the reserve is
recorded by qid only. Annotation inputs (official decomposition/evidence) and future QEA
runtime inputs (Urdu question only) are written to separate subdirectories.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path("/mnt/home/user41/URBench")
ARCHIVES = Path("/mnt/home/user41/URBench_pilot_archives")
OUT = ROOT / "outputs/efbpt/qea_preparation_v1_r2"
SUPERSEDED = ROOT / "outputs/efbpt/qea_preparation_v1"
RUN1_SCRIPT_COPY = Path(sys.argv[1]) if len(sys.argv) > 1 else None
RUN1_SCRIPT_SHA256 = "5b1d66cad968362773bdce11e327c6a8cd60c3c3d62230c1de1711cce5a572bb"
ORDER_PREFIX = "urbench.qea.dev.v1|"
DEV_TRIAGE_SIZE = 160
VERSION = "urbench.qea.preparation.v1"

MAPPED = ROOT / "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl"
RAW_EN = ROOT / "data/strategyqa_raw/strategyQA_train.json"
RAW_UR = ROOT / "data/strategyqa_raw/strategyQA_train_ur2_norm.jsonl"
PARAGRAPHS = ROOT / "data/strategyqa_official/strategyqa_train_paragraphs.json"
STAGE1 = ROOT / "data/strategyqa_official/efbpt/stage1_report.jsonl"
STAGE0_PREP = ROOT / "eval/error_analysis_tests/efbpt/efbpt_prepare_stage0.py"

QID = re.compile(r"(?<![0-9a-f])[0-9a-f]{20}(?![0-9a-f])")
SKIP_SUFFIXES = {".safetensors", ".bin", ".npy", ".index", ".pt", ".pth", ".pkl", ".gz",
                 ".zip", ".png", ".jpg", ".pdf", ".faiss", ".parquet", ".arrow"}
SKIP_BYTES = 150_000_000

# Path rules, first match wins: (set_id, predicate on repo-relative posix path).
SOURCE_DATASET = {
    "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl",
    "data/strategyqa_official/train.json", "data/strategyqa_official/dev.json",
    "data/strategyqa_official/strategyqa_train_paragraphs.json",
    "data/strategyqa_raw/strategyQA_train.json", "data/strategyqa_raw/strategyQA_train_ur2_norm.jsonl",
}
RULES = [
    ("SOURCE_DATASET", lambda p: p in SOURCE_DATASET),
    ("BULK_BASELINE_EVAL", lambda p: p.startswith("outputs/strategyqa/")),
    ("SDFR_DEMONSTRATION_POOL", lambda p: p in {"data/sdfr_splits/strategyqa_pool.jsonl",
                                               "data/sdfr_splits/strategyqa_pool_urdu.jsonl"}
                                          or p.startswith("data/sdfr_indexes/strategyqa")),
    ("STAGE1_AUTOMATED", lambda p: p == "data/strategyqa_official/efbpt/stage1_report.jsonl"),
    ("STAGE1_EXCLUSION_LISTING", lambda p: p == "data/strategyqa_official/efbpt/stage1_summary.txt"),
    ("SDFR_EVALUATION", lambda p: p == "data/sdfr_splits/strategyqa_eval.jsonl"
                                  or (p.startswith("outputs/sdfr/") and not p.endswith(".md"))),
    ("SDFR_ERROR_ANALYSIS", lambda p: p.startswith("outputs/sdfr/") and p.endswith(".md")),
    ("TRAINING", lambda p: p.startswith("data/strategyqa_official/efbpt/train/")),
    ("PLAN_A_REVIEW", lambda p: p in {"data/strategyqa_official/efbpt/plan_a_gold_100.jsonl",
                                      "data/strategyqa_official/efbpt/plan_a_review_audit.jsonl"}),
    ("PLAN_A_DEVELOPMENT", lambda p: "plan_a" in p),
    ("AUDIT_BLIND_REVIEW", lambda p: "audit30" in p or "blind30" in p),
    ("PHASE_R", lambda p: "phase_r" in p),
    ("STAGE2_CPROBE", lambda p: "stage2_" in p or "c_probe" in p),
    ("DEV50_DEVELOPMENT", lambda p: "dev50" in p),
    ("DEV200_DEVELOPMENT", lambda p: "dev200" in p or p.startswith("data/strategyqa_official/efbpt/stage0/")
                                     or p.startswith("outputs/efbpt/")),
    ("N25_ARCHIVES", lambda p: p.startswith("ARCHIVES/")),
    ("JOB_LOGS", lambda p: p.startswith("logs/")),
    ("CODE_OR_DOC_MENTION", lambda p: p.startswith(("eval/", "docs/")) or p in {"experiments.md", "README.md"}),
    ("OTHER_RECORDED_USE", lambda p: True),
]
CATEGORIES = {
    "SOURCE_DATASET": [],
    "BULK_BASELINE_EVAL": ["previous_evaluation_bulk"],
    "SDFR_DEMONSTRATION_POOL": ["demonstration_pool_use"],
    "STAGE1_AUTOMATED": ["automated_processing"],
    "STAGE1_EXCLUSION_LISTING": ["known_researcher_inspection"],
    "SDFR_EVALUATION": ["previous_evaluation_split"],
    "SDFR_ERROR_ANALYSIS": ["known_researcher_inspection"],
    "TRAINING": ["training_use"],
    "PLAN_A_REVIEW": ["known_researcher_inspection"],
    "PLAN_A_DEVELOPMENT": ["previous_evaluation_split"],
    "AUDIT_BLIND_REVIEW": ["known_researcher_inspection", "previous_evaluation_split"],
    "PHASE_R": ["previous_evaluation_split", "known_researcher_inspection"],
    "STAGE2_CPROBE": ["previous_evaluation_split", "known_researcher_inspection"],
    "DEV50_DEVELOPMENT": ["previous_evaluation_split"],
    "DEV200_DEVELOPMENT": ["previous_evaluation_split", "known_researcher_inspection"],
    "N25_ARCHIVES": ["previous_evaluation_split"],
    "JOB_LOGS": ["recorded_run_output"],
    "CODE_OR_DOC_MENTION": ["known_researcher_inspection"],
    "OTHER_RECORDED_USE": ["unclassified_recorded_use"],
}
# Exposure compatible with eligibility. Everything else disqualifies.
ALLOWED_SETS = {"SOURCE_DATASET", "BULK_BASELINE_EVAL", "SDFR_DEMONSTRATION_POOL", "STAGE1_AUTOMATED"}
ANNOTATION_FIELDS = ("urbench_qid", "official_qid", "question_ur", "official_decomposition",
                     "official_evidence", "evidence_paragraph_ids")


def fail(message):
    raise SystemExit("PREPARATION_STOPPED: " + message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def order_key(qid):
    return sha256_bytes((ORDER_PREFIX + qid).encode("utf-8"))


def scan_roots():
    """Yield (label_path, real_path) for every candidate text file."""
    for base, label in ((ROOT / "data", "data"), (ROOT / "outputs", "outputs"), (ROOT / "logs", "logs"),
                        (ROOT / "eval", "eval"), (ROOT / "docs", "docs"), (ARCHIVES, "ARCHIVES")):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in {".git", "__pycache__"})
            rel_dir = Path(dirpath).relative_to(base)
            parts = rel_dir.parts
            if label == "outputs" and len(parts) >= 2 and parts[0] == "efbpt" and parts[1].startswith(("qea_preparation", ".qea_preparation")):
                continue   # this preparation's own outputs are never exposure evidence
            for name in sorted(filenames):
                real = Path(dirpath) / name
                if real.resolve() == Path(__file__).resolve():
                    continue   # the preparation script itself is not exposure evidence
                label_path = (Path(label) / rel_dir / name).as_posix()
                yield label_path, real
    for name in ("experiments.md", "README.md"):
        yield name, ROOT / name


def main():
    if OUT.exists():
        fail(f"output exists, refusing overwrite: {OUT}")
    started = datetime.now(timezone.utc).isoformat()
    script_identity = identity(Path(__file__).resolve())

    # ---- source pool and alignment
    mapped = [json.loads(line) for line in MAPPED.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_qid = {}
    for row in mapped:
        q = row["urbench_qid"]
        if q in by_qid:
            fail(f"duplicate mapped qid {q}")
        by_qid[q] = row
    ALL = set(by_qid)
    raw_en = json.loads(RAW_EN.read_text(encoding="utf-8"))
    raw_en = raw_en if isinstance(raw_en, list) else raw_en["data"]
    en_rows, ur_rows = defaultdict(list), defaultdict(list)
    for r in raw_en:
        en_rows[str(r.get("qid", "")).strip()].append(r)
    for line in RAW_UR.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ur_rows[str(r.get("qid", "")).strip()].append(r)
    stage1 = {json.loads(l)["qid"]: json.loads(l) for l in STAGE1.read_text(encoding="utf-8").splitlines() if l.strip()}

    # ---- exposure scan
    set_qids = defaultdict(set)
    sources = []
    unreadable = []
    for label_path, real in scan_roots():
        if real.suffix.lower() in SKIP_SUFFIXES or real.is_symlink() or not real.is_file():
            continue
        size = real.stat().st_size
        if size > SKIP_BYTES:
            unreadable.append({"path": label_path, "bytes": size, "reason": "SKIPPED_OVER_SIZE_LIMIT"})
            continue
        try:
            text = real.read_bytes().decode("utf-8", errors="ignore")
        except OSError as exc:
            unreadable.append({"path": label_path, "bytes": size, "reason": type(exc).__name__})
            continue
        found = set(QID.findall(text)) & ALL
        if not found:
            continue
        set_id = next(s for s, pred in RULES if pred(label_path))
        if set_id == "JOB_LOGS" and len(found) >= 1000:
            set_id = "BULK_BASELINE_EVAL"      # bulk evaluation log, recorded as such below
            label_note = "JOB_LOG_WITH_BULK_QIDS"
        else:
            label_note = None
        set_qids[set_id] |= found
        sources.append({"path": label_path, "set_id": set_id, "qid_count": len(found),
                        "sha256": sha256_bytes(real.read_bytes()), "bytes": size, "note": label_note})

    # ---- per-qid exposure, eligibility
    member = defaultdict(set)
    for s, qs in set_qids.items():
        for q in qs:
            member[q].add(s)
    sys.path.insert(0, str(STAGE0_PREP.parent))
    sys.dont_write_bytecode = True
    import efbpt_prepare_stage0 as s0
    paragraphs = json.loads(PARAGRAPHS.read_text(encoding="utf-8"))

    rows_out, eligible = [], []
    reasons_count = Counter()
    profile_count = Counter()
    for q in sorted(ALL):
        row = by_qid[q]
        reasons = []
        sets = sorted(member[q])
        disq = sorted(set(sets) - ALLOWED_SETS)
        if disq:
            reasons.append("EXPOSURE:" + "+".join(disq))
        en_ok = len(en_rows[q]) == 1 and en_rows[q][0].get("question") == row.get("question_en")
        ur_ok = len(ur_rows[q]) == 1 and ur_rows[q][0].get("question") == row.get("question_ur")
        if not (isinstance(row.get("question_ur"), str) and row["question_ur"].strip()
                and isinstance(row.get("question_en"), str) and row["question_en"].strip()):
            reasons.append("ALIGNMENT:EMPTY_TEXT")
        if not en_ok:
            reasons.append("ALIGNMENT:RAW_ENGLISH_MISMATCH")
        if not ur_ok:
            reasons.append("ALIGNMENT:RAW_URDU_MISMATCH")
        ws_key = any(str(r.get("qid")) != str(r.get("qid")).strip() for r in ur_rows[q] + en_rows[q])
        if stage1.get(q, {}).get("status") != "RETAINED":
            reasons.append("STRUCTURE:NOT_STAGE1_RETAINED_GE2_STEPS")
        n_titles = None
        try:
            srcs, _links = s0.build_sources_and_evidence([{k: row.get(k) for k in ANNOTATION_FIELDS}], paragraphs)
            n_titles = len({x["normalized_gold_title"] for x in srcs})
            if n_titles < 2:
                reasons.append("STRUCTURE:FEWER_THAN_2_DISTINCT_EVIDENCE_TITLES")
        except s0.Stage0Error:
            reasons.append("STRUCTURE:OFFICIAL_EVIDENCE_INVALID")
        for r in reasons:
            reasons_count["EXPOSURE" if r.startswith("EXPOSURE:") else r] += 1
        profile_count["+".join(sorted({r.split(":")[0] for r in reasons})) or "ELIGIBLE"] += 1
        cats = sorted({c for s in sets for c in CATEGORIES[s]})
        rec = {"urbench_qid": q, "exposure_sets": sets, "exposure_categories": cats,
               "inspection_status": ("NO_RECORDED_INDIVIDUAL_INSPECTION_UNKNOWN" if not disq
                                     else "RECORDED_USE_OR_INSPECTION"),
               "raw_qid_whitespace_defect": ws_key, "latin_letters_in_urdu": bool(re.search("[A-Za-z]{3,}", row["question_ur"])),
               "distinct_evidence_titles": n_titles, "eligible": not reasons, "exclusion_reasons": reasons}
        rows_out.append(rec)
        if not reasons:
            eligible.append(q)

    ordered = sorted(eligible, key=order_key)
    if len(ordered) < DEV_TRIAGE_SIZE:
        fail(f"only {len(ordered)} eligible qids")
    dev, reserve = ordered[:DEV_TRIAGE_SIZE], ordered[DEV_TRIAGE_SIZE:]
    if set(dev) & set(reserve) or len(set(dev) | set(reserve)) != len(ordered):
        fail("partition not disjoint/complete")

    # ---- annotation inputs (dev only) and QEA runtime inputs (dev only), kept separate
    dev_rows = [{k: by_qid[q].get(k) for k in ANNOTATION_FIELDS} for q in dev]
    master, links = s0.build_sources_and_evidence(dev_rows, paragraphs)
    for m in master:
        m["annotation_scope"] = "QEA_TRIAGE_ANNOTATION_INPUT_NOT_QEA_RUNTIME"
        m["exact_corpus_status"] = None          # assigned only by the approved triage packet's metadata scan

    # ---- overlaps
    ids = sorted(set_qids)
    pairwise = {f"{a}&{b}": len(set_qids[a] & set_qids[b]) for a, b in combinations(ids, 2)
                if set_qids[a] & set_qids[b]}
    signature = Counter("+".join(sorted(member[q] - {"SOURCE_DATASET"})) or "NONE" for q in ALL)
    cat_union = defaultdict(set)
    for q in ALL:
        for s in member[q]:
            for c in CATEGORIES[s]:
                cat_union[c].add(q)

    files = {}

    def put(rel, payload):
        files[rel] = payload if isinstance(payload, bytes) else payload.encode("utf-8")

    put("exposure/exposure_sources.jsonl", "".join(dump(s) + "\n" for s in sources))
    put("exposure/exposure_sets.json", json.dumps({
        "sets": {s: {"categories": CATEGORIES[s], "qids": len(set_qids[s]),
                     "qid_list_sha256": sha256_bytes("\n".join(sorted(set_qids[s])).encode()),
                     "files": sum(1 for x in sources if x["set_id"] == s),
                     "eligibility_compatible": s in ALLOWED_SETS} for s in ids},
        "pairwise_overlaps": pairwise,
        "category_unions_distinct_qids": {c: len(v) for c, v in sorted(cat_union.items())},
        "exact_exposure_signature_partition": dict(sorted(signature.items(), key=lambda x: (-x[1], x[0]))),
        "skipped_files": unreadable}, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    put("exposure/qid_exposure_eligibility.jsonl", "".join(dump(r) + "\n" for r in rows_out))
    put("selection/eligible_ordered.jsonl", "".join(dump({"rank": i + 1, "urbench_qid": q, "order_key": order_key(q),
                                                          "partition": "DEV_TRIAGE" if i < DEV_TRIAGE_SIZE else "RESERVE"})
                                                    + "\n" for i, q in enumerate(ordered)))
    put("selection/dev_triage_qids.txt", "".join(q + "\n" for q in dev))
    put("selection/reserve_qids.txt", "".join(q + "\n" for q in reserve))
    put("annotation_inputs/dev_triage_rows.jsonl", "".join(dump(dict(r, annotation_scope="QEA_TRIAGE_ANNOTATION_INPUT_NOT_QEA_RUNTIME")) + "\n" for r in dev_rows))
    put("annotation_inputs/source_instance_master.jsonl", "".join(dump(m) + "\n" for m in master))
    put("annotation_inputs/official_evidence_links.jsonl", "".join(dump(l) + "\n" for l in links))
    put("qea_runtime_inputs/dev_questions_ur.jsonl", "".join(dump({"urbench_qid": q, "question_ur": by_qid[q]["question_ur"],
                                                                   "runtime_scope": "QEA_RUNTIME_INPUT_QUESTION_ONLY"}) + "\n" for q in dev))

    counts = {
        "mapped_pool": len(ALL),
        "eligible": len(ordered), "dev_triage": len(dev), "reserve": len(reserve),
        "excluded": len(ALL) - len(ordered),
        "exclusion_reason_counts_nonexclusive": dict(sorted(reasons_count.items())),
        "exclusion_profile_counts_exclusive": dict(sorted(profile_count.items())),
        "eligible_with_raw_qid_whitespace_defect": sum(1 for r in rows_out if r["eligible"] and r["raw_qid_whitespace_defect"]),
        "eligible_with_latin_letters_in_urdu": sum(1 for r in rows_out if r["eligible"] and r["latin_letters_in_urdu"]),
        "dev_source_instances": len(master), "dev_evidence_links": len(links),
    }
    manifest = {
        "schema": VERSION, "status": "PREPARED_NOT_RUN", "created_utc": started,
        "model_calls": 0, "retrieval": "NONE", "triage_executed": False,
        "script": script_identity,
        "reused_code": {"efbpt_prepare_stage0.build_sources_and_evidence": identity(STAGE0_PREP)},
        "source_inputs": {k: identity(p) for k, p in (("mapped", MAPPED), ("raw_english", RAW_EN), ("raw_urdu", RAW_UR),
                                                       ("official_paragraphs", PARAGRAPHS), ("stage1_report", STAGE1))},
        "eligibility_rules": [
            "Row is in the mapped URBench-official pool with a unique urbench_qid.",
            "Non-empty Urdu and English question text; exactly one raw English and one raw Urdu row after stripping "
            "surrounding whitespace from the qid key, each byte-identical to the mapped text.",
            "Stage-1 status RETAINED (official decomposition has >= 2 steps).",
            "Official evidence passes efbpt_prepare_stage0.build_sources_and_evidence and yields >= 2 distinct "
            "normalized evidence titles (structural check; content not inspected).",
            "Exposure limited to " + ", ".join(sorted(ALLOWED_SETS)) + ". Any other recorded set disqualifies.",
        ],
        "ordering": {"key": "sha256(utf8('" + ORDER_PREFIX + "' + urbench_qid)) ascending",
                     "applied_after_eligibility": True, "dev_triage_first_n": DEV_TRIAGE_SIZE},
        "partition_semantics": {
            "DEV_TRIAGE": "Development triage pool for QEA target construction (AI-assisted annotation).",
            "RESERVE": "Prospectively reserved pool with documented historical exposure (bulk baseline evaluation, "
                       "SDFR demonstration pool, automated Stage-1 step count). NOT an untouched test set. "
                       "Question and evidence content must not be inspected for method design."},
        "counts": counts,
        "files": {rel: {"sha256": sha256_bytes(b), "bytes": len(b)} for rel, b in sorted(files.items())},
        "notes": ["Raw Urdu file stores qid 236c7a57f3788a60e47f with a leading space; text is byte-identical to the "
                  "mapping and the English file has the clean qid; treated as a whitespace-only key defect.",
                  "exact_corpus_status is null in annotation inputs; it is assigned only by the approved triage packet."],
    }
    if RUN1_SCRIPT_COPY is None or sha256_file(RUN1_SCRIPT_COPY) != RUN1_SCRIPT_SHA256:
        fail("run-1 script copy with the recorded hash must be supplied as argv[1]")
    put("superseded_v1/efbpt_qea_prepare_v1.run1.py", RUN1_SCRIPT_COPY.read_bytes())
    v1_rows = {json.loads(l)["urbench_qid"]: json.loads(l) for l in (SUPERSEDED / "exposure/qid_exposure_eligibility.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    changed = sorted(q for q in ALL if v1_rows[q]["eligible"] != next(r for r in rows_out if r["urbench_qid"] == q)["eligible"])
    v1_dev = (SUPERSEDED / "selection/dev_triage_qids.txt").read_text().split()
    manifest["supersedes"] = {
        "path": str(SUPERSEDED), "manifest_sha256": sha256_file(SUPERSEDED / "PREPARATION_MANIFEST.json"),
        "run1_script_sha256": RUN1_SCRIPT_SHA256, "preserved_unchanged": True,
        "reason": "Run 1 scanned its own preparation script, whose manifest note names the resolved whitespace-key qid, "
                  "and so recorded a spurious CODE_OR_DOC_MENTION exclusion. Run 2 excludes the preparation script and "
                  "all qea_preparation outputs from the exposure scan; rules, ordering key and partition size are unchanged.",
        "eligibility_changes_vs_v1": len(changed), "dev_triage_identical_to_v1": v1_dev == dev}
    put("PREPARATION_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n")

    tmp = OUT.parent / (".qea_preparation_v1.tmp-" + str(os.getpid()))
    os.mkdir(tmp)
    for rel, payload in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    if OUT.exists():
        fail("output appeared during preparation; temporary directory left at " + str(tmp))
    os.rename(tmp, OUT)
    for rel, payload in files.items():
        if sha256_file(OUT / rel) != sha256_bytes(payload):
            fail("readback mismatch " + rel)
    print(json.dumps({"output": str(OUT), "counts": counts,
                      "manifest_sha256": sha256_file(OUT / "PREPARATION_MANIFEST.json")}, indent=1))


if __name__ == "__main__":
    main()
