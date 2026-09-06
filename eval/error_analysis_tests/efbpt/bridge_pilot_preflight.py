#!/usr/bin/env python3
"""Outcome-free compute preflight, v0.2. Does not activate the N25 pilot.

Checks current asset hashes, index/offset structure, all recorded parent rows
plus deterministic global sample rows against freshly encoded metadata; hashes
native model files. No Qwen loading/generation, FAISS search, target access or
scoring. Sample success is NOT a proof of global historical vector lineage.
Revision 0.2 corrects job 81798's impossible post-deserialization
``isinstance(IndexFlatIP)`` check under FAISS 1.7.4. It records the observed
structure before separate assertions and accepts only flat float32 inner-product
storage with the pinned dimension/count. All v1 artifacts remain preserved.

All new outputs and their archive are exclusive-create; interrupted runs stay.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import re
import socket
import stat
import subprocess

from bridge_pilot_core import (CHILD_COUNTS, N_VECTORS, DIM, SEED, PilotError,
                               canonical, exact_keys, need, norm, sha256, strict_json)

VERSION = "0.2"
REPO = Path("/mnt/home/user41/URBench")
PREP = "outputs/efbpt/bridge_pilot_n25/v1/preparation_r3"
PREP_ARCHIVE = Path("/mnt/home/user41/URBench_pilot_archives/bridge_pilot_n25_preparation_v1_r3")
OUT = "outputs/efbpt/bridge_pilot_n25/v1/preflight_v2"
ARCHIVE = Path("/mnt/home/user41/URBench_pilot_archives/bridge_pilot_n25_preflight_v2")
SEAL_HASH = "f976ef54cdec11425cd944e92aea00001b5e3432e19d4d6ad75544b9ba425249"
CORE_HASH = "0a02ee75ea4b86a501972441db3d6874eeda1f2d4f30c75cf19c18571f7b5a67"
QWEN = Path("/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B")
ENCODER = Path("/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ASSETS = {
    "index": ("rag/index/wikipedia_full.index", 36808659501,
              "aeb5a87c9eedfc8a0a0f23994f8000ba0b9be043e385e831b2d060bdbb98a767"),
    "metadata": ("rag/index/wikipedia_full_meta.jsonl", 25866666236,
                 "b659788378d98e9551918c920c53d6625b89d6a1463579c52bf7bf02c12389a2"),
    "offsets": ("rag/index/wikipedia_full_meta.offsets.npy", 191711896,
                "2cf46155c3483fad87648c1b65f71316d0b75fe47e1211bf710839a7fda54c66"),
}
PACKAGES = {"torch": "2.9.0", "transformers": "4.57.6", "tokenizers": "0.22.2",
            "sentence-transformers": "5.4.1", "numpy": "1.24.0", "faiss-cpu": "1.7.4",
            "accelerate": "1.12.0", "bitsandbytes": "0.49.1", "safetensors": "0.7.0"}
SETTINGS = {
    "version": VERSION, "uniform_global_rows": 257, "builder_shard_rows": 500000,
    "include_shard_boundaries_and_neighbors": True, "include_all_recorded_parent_rows": True,
    "maximum_selected_rows": 10000, "encoder_batch_size": 32, "seed": SEED,
    "cpu_threads": 8, "encoder_device": "cuda:0", "encoder_dtype": "float32",
    "encoder_attention": "sdpa", "encoder_backend": "torch", "encoder_max_seq_length": 128,
    "encoder_input": "metadata.title + '. ' + metadata.text", "prefix": "",
    "normalize_embeddings": True, "precision": "float32",
    "minimum_cosine": 0.99999, "maximum_l2_distance": 0.001,
    "maximum_unit_norm_error": 0.0001,
    "comparison_scope": "SELECTED_ROWS_ONLY; NOT_GLOBAL_LINEAGE_PROOF",
    "prospective_threshold_policy": "No automatic relaxation, repair, rebuilding or row exclusion",
}

def safe(path):
    p = Path(path).absolute()
    for x in (p, *p.parents):
        need(not x.is_symlink(), "SYMLINK_REFUSED: " + str(x))
    return p

def stamp(path):
    s = safe(path).stat()
    need(stat.S_ISREG(s.st_mode), "REGULAR_FILE_REQUIRED")
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

def read_small(path, expected=None):
    p = safe(path); before = stamp(p)
    need(before[2] <= 64 * 1024**2, "SMALL_FILE_TOO_LARGE")
    raw = p.read_bytes()
    need(before == stamp(p) and len(raw) == before[2], "SMALL_FILE_CHANGED")
    need(expected is None or sha256(raw) == expected, "SMALL_FILE_HASH: " + str(p))
    return raw, {"path": str(p), "sha256": sha256(raw), "bytes": len(raw), "stat": before}

def fingerprint(path, expected_hash=None, expected_bytes=None):
    p = safe(path); before = stamp(p)
    need(expected_bytes is None or before[2] == expected_bytes, "ASSET_SIZE_DRIFT: " + str(p))
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8*1024**2), b""):
            h.update(block)
    need(stamp(p) == before, "ASSET_CHANGED_DURING_HASH: " + str(p))
    need(expected_hash is None or h.hexdigest() == expected_hash, "ASSET_HASH_DRIFT: " + str(p))
    return {"path": str(p), "sha256": h.hexdigest(), "bytes": before[2], "stat": before}

def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def new_dir(path):
    p = safe(path)
    need(p.parent.is_dir() and not p.exists(), "FRESH_DIRECTORY_WITH_EXISTING_PARENT_REQUIRED: " + str(p))
    p.mkdir(mode=0o700)
    sync_dir(p); sync_dir(p.parent)

def write_new(path, raw):
    p = safe(path)
    with p.open("xb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    sync_dir(p.parent)
    check, _ = read_small(p)
    need(check == raw, "WRITTEN_FILE_MISMATCH")
    return {"sha256": sha256(raw), "bytes": len(raw)}

def event(stage, **fields):
    print(canonical({"stage": stage, **fields}), flush=True)

def env_check():
    need(sys.version_info[:3] == (3,10,19), "PYTHON_VERSION_DRIFT")
    observed = {key: importlib.metadata.version(key) for key in PACKAGES}
    need(observed == PACKAGES, "PACKAGE_VERSION_DRIFT")
    return {"python": sys.version, "executable": sys.executable, "packages": observed}

def select_rows(parents, total=N_VECTORS, uniform=257, shard_size=500000):
    need(total >= 2 and uniform >= 2, "SAMPLE_SETTINGS")
    chosen = {(j*(total-1))//(uniform-1) for j in range(uniform)}
    for boundary in range(shard_size, total, shard_size):
        chosen.update(r for r in (boundary-1,boundary,boundary+1) if 0 <= r < total)
    for r in parents:
        need(type(r) is int and 0 <= r < total, "PARENT_ROW_RANGE")
        chosen.add(r)
    return sorted(chosen)

def parent_rows(report):
    exact_keys(report, ("parents", "failed_qids"), "PARENT_CORRESPONDENCE")
    need(report["failed_qids"] == [] and isinstance(report["parents"], list)
         and len(report["parents"]) == 25, "PARENT_CORRESPONDENCE_NOT_PASS")
    need({p["qid"] for p in report["parents"]} == set(CHILD_COUNTS), "PARENT_QID_SET")
    saved = {}
    for parent in report["parents"]:
        need(parent["status"] == "PASS" and parent["indexed_rows"], "PARENT_NOT_PASS")
        need(parent["indexed_chunk_count"] == len(parent["indexed_rows"]), "PARENT_ROW_COUNT")
        for row in parent["indexed_rows"]:
            exact_keys(row, ("global_row", "byte_offset", "metadata_title", "text_sha256"), "PARENT_INDEX_ROW")
            idx = row["global_row"]
            need(type(idx) is int and 0 <= idx < N_VECTORS, "PARENT_ROW_RANGE")
            need(idx not in saved or saved[idx] == row, "CONFLICTING_PARENT_ROW")
            saved[idx] = row
    return saved

def load_preparation(repo):
    raw, identity = read_small(repo/PREP/"PREPARATION_SEAL.json", SEAL_HASH)
    archived, _ = read_small(PREP_ARCHIVE/"PREPARATION_SEAL.json", SEAL_HASH)
    need(raw == archived, "PREPARATION_SEAL_COPIES")
    seal = strict_json(raw)
    need(seal["status"] == "SEALED_PREPARATION_NOT_EXPERIMENT_FREEZE", "PREPARATION_STATUS")
    consumed, docs, info = {"PREPARATION_SEAL.json": raw}, {}, {"PREPARATION_SEAL.json": identity}
    for name in ("parent_index_correspondence.json", "retrieval_asset_fingerprints.json", "preparation_summary.json"):
        expected = seal["artifacts"][name]
        value, ident = read_small(repo/PREP/name, expected["sha256"])
        need(len(value) == expected["bytes"], "PREPARATION_ARTIFACT_SIZE")
        archive_value, _ = read_small(PREP_ARCHIVE/"preparation"/name, expected["sha256"])
        need(value == archive_value, "PREPARATION_ARCHIVE_COPY")
        consumed[name], docs[name], info[name] = value, strict_json(value), ident
    summary = docs["preparation_summary.json"]
    need(summary["preparation_status"] == "COMPLETE" and summary["accepted_qids"] == 25
         and summary["accepted_pairs"] == 36, "PREPARATION_SUMMARY")
    assets = docs["retrieval_asset_fingerprints.json"]
    for key, (rel, size, h) in ASSETS.items():
        rec = assets[key]
        need(rec["sha256"] == h and Path(rec["path"]) == repo/rel, "RECORDED_ASSET_IDENTITY")
        need(rec.get("bytes", rec["stat"][2]) == size, "RECORDED_ASSET_BYTES")
    return consumed, info, parent_rows(docs["parent_index_correspondence.json"])

def model_inventory(root, is_qwen):
    root = safe(root)
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    if is_qwen:
        required += ["generation_config.json", "model.safetensors.index.json", "vocab.json", "merges.txt"]
    else:
        required += ["sentence_bert_config.json", "modules.json", "1_Pooling/config.json", "sentencepiece.bpe.model"]
    # Capture all root JSON configs plus recognized tokenization text/binary files.
    # No recursive alternate ONNX/OpenVINO/TF or pickle weight loading.
    small_names = set(required) | {p.name for p in root.glob("*.json")}
    for name in ("added_tokens.json", "special_tokens_map.json", "vocab.txt", "merges.txt", "chat_template.jinja"):
        if (root/name).exists(): small_names.add(name)
    small, raws = {}, {}
    for name in sorted(small_names):
        raw, identity = read_small(root/name)
        small[name], raws[name] = identity, raw
    if is_qwen:
        cfg = strict_json(raws["model.safetensors.index.json"])
        weights = sorted(set(cfg["weight_map"].values()))
        need(len(weights) == 8 and all(re.fullmatch(r"model-\d{5}-of-00008\.safetensors", n) for n in weights), "QWEN_WEIGHT_LAYOUT")
        expected_sizes = [3841788544,3963750816] + [3963750880]*5 + [1912371880]
        need([stamp(root/n)[2] for n in weights] == expected_sizes, "QWEN_WEIGHT_SIZES")
    else:
        weights = ["model.safetensors"]
        need(stamp(root/weights[0])[2] == 470641600, "ENCODER_WEIGHT_SIZE")
        modules = strict_json(raws["modules.json"])
        need(isinstance(modules, list) and len(modules) == 2
             and modules[0]["path"] == "" and modules[1]["path"] == "1_Pooling"
             and modules[0]["type"] == "sentence_transformers.models.Transformer"
             and modules[1]["type"] == "sentence_transformers.models.Pooling", "ENCODER_MODULE_LAYOUT")
        need(strict_json(raws["sentence_bert_config.json"])["max_seq_length"] == 128, "ENCODER_MAX_SEQ_LENGTH")
    return small, raws, weights

def check_offsets(offsets, n, metadata_bytes):
    import numpy as np
    need(offsets.ndim == 1 and len(offsets) == n and offsets.dtype.kind in "iu"
         and offsets.dtype.itemsize == 8, "OFFSETS_SHAPE_TYPE")
    need(int(offsets[0]) == 0 and 0 <= int(offsets[-1]) < metadata_bytes, "OFFSETS_ENDPOINTS")
    previous = -1
    for start in range(0,n,1000000):
        block = offsets[start:start+1000000]
        need(int(block[0]) > previous and np.all(block[1:] > block[:-1])
             and np.all(block < metadata_bytes), "OFFSETS_MONOTONIC_BOUNDS")
        previous = int(block[-1])

def metadata_row(stream, offsets, idx, size, total, parent=None):
    need(type(idx) is int and 0 <= idx < total, "LOOKUP_ROW_RANGE")
    offset = int(offsets[idx])
    expected_end = int(offsets[idx+1]) if idx+1 < total else size
    need(0 <= offset < expected_end <= size and expected_end-offset <= 8*1024**2, "LOOKUP_OFFSET_BOUNDS")
    if offset:
        stream.seek(offset-1)
        need(stream.read(1) == b"\n", "OFFSET_NOT_LINE_START")
    stream.seek(offset)
    raw = stream.read(expected_end-offset)
    need(len(raw) == expected_end-offset and raw.endswith(b"\n") and raw.count(b"\n") == 1, "METADATA_LINE_BOUNDARY")
    obj = strict_json(raw)
    exact_keys(obj, ("title","text"), "METADATA")
    need(isinstance(obj["title"],str) and norm(obj["title"])
         and isinstance(obj["text"],str) and obj["text"].strip(), "METADATA_STRINGS")
    if parent is not None:
        need(offset == parent["byte_offset"] and obj["title"] == parent["metadata_title"]
             and sha256(obj["text"]) == parent["text_sha256"], "PARENT_METADATA_DRIFT")
    return obj, {"global_row": idx, "byte_offset": offset, "metadata_line_sha256": sha256(raw),
                 "encoder_input_sha256": sha256(obj["title"]+". "+obj["text"]), "is_parent_row": parent is not None}

def vector_comparison(a, b):
    import numpy as np
    a, b = np.asarray(a,dtype=np.float64), np.asarray(b,dtype=np.float64)
    need(a.shape == b.shape == (DIM,), "VECTOR_SHAPE")
    need(np.isfinite(a).all() and np.isfinite(b).all(), "NONFINITE_VECTOR")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    need(na > 0 and nb > 0, "ZERO_VECTOR")
    cosine = float(np.dot(a,b)/(na*nb))
    distance = float(np.linalg.norm(a-b))
    passed = (cosine >= SETTINGS["minimum_cosine"] and distance <= SETTINGS["maximum_l2_distance"]
              and max(abs(na-1),abs(nb-1)) <= SETTINGS["maximum_unit_norm_error"])
    return {"pass": passed, "cosine": cosine, "l2_distance": distance,
            "max_abs_difference": float(np.max(np.abs(a-b))), "stored_norm": na, "fresh_norm": nb}

def index_fourcc(path):
    p = safe(path); before = stamp(p)
    with p.open("rb") as stream:
        raw = stream.read(4)
    need(stamp(p) == before and len(raw) == 4, "INDEX_CHANGED_DURING_FOURCC_READ")
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PilotError("INDEX_FOURCC_NOT_ASCII") from exc
    return value

def observe_index(index, faiss_module, fourcc):
    """Record the SWIG object as returned; do not infer its constructor class."""
    return {"python_class": type(index).__name__, "python_module": type(index).__module__,
        "mro": [c.__name__ for c in type(index).__mro__],
        "is_flat_storage": isinstance(index, faiss_module.IndexFlat),
        "metric_type": int(index.metric_type), "inner_product_metric_value": int(faiss_module.METRIC_INNER_PRODUCT),
        "dimension": int(index.d), "vectors": int(index.ntotal),
        "is_trained": bool(index.is_trained), "code_size": int(index.code_size),
        "file_fourcc": fourcc}

def validate_index_structure(observed):
    """Separate failures make future diagnostics exact; no auto-repair occurs."""
    need(observed["file_fourcc"] == "IxFI", "INDEX_FILE_FOURCC_NOT_FLAT_IP")
    need(observed["is_flat_storage"] and observed["code_size"] == 4 * DIM, "INDEX_TYPE_NOT_FLAT_FLOAT32")
    need(observed["metric_type"] == observed["inner_product_metric_value"], "INDEX_METRIC_NOT_INNER_PRODUCT")
    need(observed["dimension"] == DIM, "INDEX_DIMENSION_MISMATCH")
    need(observed["vectors"] == N_VECTORS, "INDEX_NTOTAL_MISMATCH")
    need(observed["is_trained"], "INDEX_NOT_TRAINED")

def small_check(repo):
    environment = env_check()
    source_dir = repo/"eval/error_analysis_tests/efbpt"
    core, _ = read_small(source_dir/"bridge_pilot_core.py", CORE_HASH)
    consumed, info, parents = load_preparation(repo)
    qsmall, _, qw = model_inventory(QWEN,True)
    esmall, _, ew = model_inventory(ENCODER,False)
    rows = select_rows(parents)
    need(len(rows) <= SETTINGS["maximum_selected_rows"], "PREFLIGHT_SAMPLE_TOO_LARGE")
    for _, (rel,size,_) in ASSETS.items(): need(stamp(repo/rel)[2] == size, "ASSET_SIZE_DRIFT")
    for dest in (repo/OUT,ARCHIVE):
        safe(dest)
        need(not dest.exists() and dest.parent.is_dir() and os.access(dest.parent,os.W_OK|os.X_OK), "PREFLIGHT_DESTINATION_NOT_READY")
    return {"environment":environment,"parent_rows":len(parents),"selected_rows":len(rows),
            "qwen_weight_files":len(qw),"encoder_weight_files":len(ew),
            "small_model_files":len(qsmall)+len(esmall),"preparation_seal_sha256":SEAL_HASH}

def preflight(repo):
    need(os.environ.get("SLURM_JOB_ID") and socket.gethostname().split(".")[0].lower() != "psn001", "COMPUTE_JOB_REQUIRED")
    need(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "OFFLINE_REQUIRED")
    checked = small_check(repo)
    source_dir = repo/"eval/error_analysis_tests/efbpt"
    source_names = ("bridge_pilot_preflight.py","bridge_pilot_preflight.sbatch","bridge_pilot_core.py")
    sources = {name: read_small(source_dir/name) for name in source_names}
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    consumed, prep_info, parents = load_preparation(repo)
    model_small, model_raws, weight_names = {}, {}, {}
    for key,root in (("qwen",QWEN),("encoder",ENCODER)):
        model_small[key],model_raws[key],weight_names[key] = model_inventory(root,key=="qwen")
    os.umask(0o077)
    out = repo/OUT
    new_dir(ARCHIVE); new_dir(out)
    artifacts, archived_inputs = {}, {}
    def save(name,obj):
        raw = (canonical(obj)+"\n").encode("utf-8")
        identity = write_new(out/name,raw)
        need(write_new(ARCHIVE/name,raw) == identity, "ARCHIVE_OUTPUT_MISMATCH")
        artifacts[name] = identity
    for name,(raw,_) in sources.items():
        archived_inputs["source_"+name] = write_new(ARCHIVE/("source_"+name),raw)
    for name,raw in consumed.items():
        archived_inputs["prep_"+name] = write_new(ARCHIVE/("prep_"+name),raw)
    save("preflight_start.json", {"utc":datetime.now(timezone.utc).isoformat(),"job_id":os.environ["SLURM_JOB_ID"],
        "node":socket.gethostname(),"git_head":head,"settings":SETTINGS,"small_check":checked,
        "sources":{name:identity for name,(_,identity) in sources.items()},"preparation":prep_info,
        "experiment_state":"NOT_FROZEN_NOT_RUN","resume_policy":"REFUSE_EXISTING_OUTPUT; PRESERVE_INTERRUPTED_RUN"})
    for key,raws in model_raws.items():
        new_dir(ARCHIVE/(key+"_small_files"))
        for name,raw in raws.items():
            if "/" in name:
                sub = ARCHIVE/(key+"_small_files")/Path(name).parent
                if not sub.exists(): new_dir(sub)
            archive_name = key+"_small_files/"+name
            archived_inputs[archive_name] = write_new(ARCHIVE/archive_name,raw)
    save("archive_inputs.json",{"archive_directory":str(ARCHIVE),"files":archived_inputs,
        "large_assets":"Hashes and source paths only; not copied into archive"})
    model_identities = {}
    for key,root in (("qwen",QWEN),("encoder",ENCODER)):
        weights = {}
        for name in weight_names[key]:
            event("model_weight_hash_started",model=key,file=name)
            weights[name] = fingerprint(root/name)
            event("model_weight_hash_complete",model=key,file=name)
        model_identities[key] = {"root":str(root),"small_files":model_small[key],"weights":weights}
    save("model_identities.json",model_identities)
    current_assets = {}
    for key,(rel,size,h) in ASSETS.items():
        event("asset_hash_started",asset=key)
        current_assets[key] = fingerprint(repo/rel,h,size)
        event("asset_hash_complete",asset=key)
    save("asset_identities.json",current_assets)
    import numpy as np
    import torch
    import faiss
    from sentence_transformers import SentenceTransformer
    need(torch.cuda.is_available(),"GPU_REQUIRED")
    torch.set_num_threads(8); faiss.omp_set_num_threads(8)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(SEED)
    offsets = np.load(repo/ASSETS["offsets"][0],mmap_mode="r",allow_pickle=False)
    check_offsets(offsets,N_VECTORS,ASSETS["metadata"][1])
    event("index_load_started")
    index = faiss.read_index(str(repo/ASSETS["index"][0]))
    observed = observe_index(index, faiss, index_fourcc(repo/ASSETS["index"][0]))
    observed.update({"offsets_shape":list(offsets.shape),"offsets_dtype":str(offsets.dtype),
        "offsets_monotonic_bounds":"PASS","all_offsets_line_starts":"ESTABLISHED_IN_HASH_IDENTICAL_PREPARATION",
        "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"faiss_threads":faiss.omp_get_max_threads(),
        "faiss_version":faiss.__version__,"validation_state":"OBSERVED_BEFORE_ASSERTIONS"})
    save("index_structure_observed.json",observed)
    validate_index_structure(observed)
    save("index_structure.json",{**observed,"validation_state":"PASS"})
    rows = select_rows(parents)
    save("selected_rows.json",{"rows":rows,"selection":SETTINGS,"selected_rows_sha256":sha256(canonical(rows)),
                              "parent_rows":sorted(parents)})
    model = SentenceTransformer(str(ENCODER),device="cuda:0",backend="torch",local_files_only=True,
        trust_remote_code=False,model_kwargs={"use_safetensors":True,"dtype":torch.float32,"attn_implementation":"sdpa"})
    model.eval()
    need(model.max_seq_length == 128 and model.get_sentence_embedding_dimension() == DIM,"ENCODER_STRUCTURE")
    need(all(p.dtype == torch.float32 for p in model.parameters()),"ENCODER_DTYPE")
    need(getattr(model,"default_prompt_name",None) is None,"ENCODER_DEFAULT_PROMPT")
    records = []
    with (repo/ASSETS["metadata"][0]).open("rb") as stream:
        for start in range(0,len(rows),SETTINGS["encoder_batch_size"]):
            batch = rows[start:start+SETTINGS["encoder_batch_size"]]
            texts, identities = [], []
            for row in batch:
                obj,identity = metadata_row(stream,offsets,row,ASSETS["metadata"][1],N_VECTORS,parents.get(row))
                texts.append(obj["title"]+". "+obj["text"]); identities.append(identity)
            with torch.inference_mode():
                fresh = model.encode(texts,batch_size=SETTINGS["encoder_batch_size"],normalize_embeddings=True,
                    device="cuda:0",convert_to_numpy=True,show_progress_bar=False,precision="float32",prompt="")
            need(fresh.shape == (len(batch),DIM) and fresh.dtype == np.float32,"ENCODER_OUTPUT_SHAPE_DTYPE")
            for row,vector,identity in zip(batch,fresh,identities):
                records.append({**identity,**vector_comparison(index.reconstruct(row),vector)})
            event("vector_batch_checked",rows_checked=len(records),rows_total=len(rows))
    failed = [r["global_row"] for r in records if not r["pass"]]
    save("vector_alignment.json",{"status":"SAMPLED_ALIGNMENT_PASS" if not failed else "SAMPLED_ALIGNMENT_FAIL",
        "selected_rows":len(rows),"parent_rows":len(parents),"failed_rows":failed,"records":records,
        "historical_global_lineage":"UNESTABLISHED","scope":SETTINGS["comparison_scope"],
        "thresholds":{k:SETTINGS[k] for k in ("minimum_cosine","maximum_l2_distance","maximum_unit_norm_error")}})
    need(not failed,"SAMPLED_VECTOR_METADATA_MISMATCH; see vector_alignment.json; no auto-repair")
    for identity in current_assets.values(): need(stamp(identity["path"]) == identity["stat"],"ASSET_CHANGED_DURING_PREFLIGHT")
    for obj in model_identities.values():
        for identity in obj["small_files"].values(): read_small(identity["path"],identity["sha256"])
        for identity in obj["weights"].values(): need(stamp(identity["path"]) == identity["stat"],"MODEL_WEIGHT_CHANGED")
    for name,(_,identity) in sources.items(): read_small(source_dir/name,identity["sha256"])
    for identity in prep_info.values(): read_small(identity["path"],identity["sha256"])
    for name,identity in archived_inputs.items():
        raw,_ = read_small(ARCHIVE/name,identity["sha256"])
        need(len(raw) == identity["bytes"],"ARCHIVED_INPUT_SIZE")
    need(subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip() == head,"GIT_HEAD_CHANGED")
    summary = {"status":"OUTCOME_FREE_PREFLIGHT_COMPLETE","index_structure":"PASS","sampled_alignment":"PASS",
        "checked_rows":len(rows),"parent_rows":len(parents),"historical_global_lineage":"UNESTABLISHED",
        "model_files_pinned":True,"experiment_state":"NOT_FROZEN_NOT_RUN","pilot_searches":0,"qwen_generations":0,
        "remaining":"Review sampled-alignment scope; implement/authenticate runtime and scorer; activate prospective protocol/amendment",
        "output_directory":str(out),"archive_directory":str(ARCHIVE)}
    save("preflight_summary.json",summary)
    seal = {"status":"SEALED_OUTCOME_FREE_PREFLIGHT_NOT_ACTIVATION","artifacts":artifacts,
            "preparation_seal_sha256":SEAL_HASH,"settings_sha256":sha256(canonical(SETTINGS)),
            "historical_global_lineage":"UNESTABLISHED"}
    raw = (canonical(seal)+"\n").encode()
    write_new(ARCHIVE/"PREFLIGHT_SEAL.json",raw); write_new(out/"PREFLIGHT_SEAL.json",raw)
    event("preflight_complete",**summary)

def self_test():
    import numpy as np
    checks = []
    def check(name,value): need(value,"SELF_TEST: "+name); checks.append(name)
    def rejects(name,fn):
        try: fn()
        except (PilotError,ValueError): checks.append(name); return
        raise PilotError("SELF_TEST_EXPECTED_REJECTION: "+name)
    check("sample_endpoints_and_parent",select_rows([4],total=20,uniform=3,shard_size=10)==[0,4,9,10,11,19])
    rejects("negative_parent",lambda:select_rows([-1],20,3,10))
    rejects("outside_parent",lambda:select_rows([20],20,3,10))
    lines = [(canonical({"title":"Synthetic","text":x})+"\n").encode() for x in ("one","two")]
    raw = b"".join(lines); offsets = np.array([0,len(lines[0])],dtype=np.int64)
    check_offsets(offsets,2,len(raw)); checks.append("valid_offsets")
    rejects("duplicate_offsets",lambda:check_offsets(np.array([0,0]),2,len(raw)))
    rejects("outside_offset",lambda:check_offsets(np.array([0,len(raw)]),2,len(raw)))
    obj,identity = metadata_row(io.BytesIO(raw),offsets,1,len(raw),2)
    check("metadata_exact_line",obj["text"]=="two" and identity["byte_offset"]==len(lines[0]))
    rejects("lookup_negative_before_seek",lambda:metadata_row(io.BytesIO(raw),offsets,-1,len(raw),2))
    rejects("wrong_line_start",lambda:metadata_row(io.BytesIO(raw),[0,len(lines[0])+1],1,len(raw),2))
    v = np.zeros(DIM,dtype=np.float32); v[0]=1
    check("identical_vector",vector_comparison(v,v)["pass"])
    check("wrong_row_vector",not vector_comparison(v,np.roll(v,1))["pass"])
    check("unnormalized_vector",not vector_comparison(v,2*v)["pass"])
    rejects("nonfinite_vector",lambda:vector_comparison(v,np.full(DIM,np.nan)))
    rejects("zero_vector",lambda:vector_comparison(v,np.zeros(DIM)))
    check("immutable_thresholds",SETTINGS["minimum_cosine"]==.99999 and SETTINGS["maximum_l2_distance"]==.001)
    class FakeFaiss:
        METRIC_INNER_PRODUCT = 0
        class IndexFlat: pass
    class GoodIndex(FakeFaiss.IndexFlat):
        metric_type, d, ntotal, is_trained, code_size = 0, DIM, N_VECTORS, True, 4*DIM
    good = observe_index(GoodIndex(), FakeFaiss, "IxFI")
    validate_index_structure(good); checks.append("deserialized_generic_flat_ip")
    variants = {
        "fourcc": {**good,"file_fourcc":"IxF2"},
        "storage": {**good,"is_flat_storage":False},
        "code_size": {**good,"code_size":DIM},
        "metric": {**good,"metric_type":1},
        "dimension": {**good,"dimension":DIM+1},
        "ntotal": {**good,"vectors":N_VECTORS-1},
        "trained": {**good,"is_trained":False},
    }
    for name,value in variants.items():
        rejects("index_reject_"+name,lambda value=value:validate_index_structure(value))
    event("self_test_complete",checks_passed=len(checks),files_written=0,real_assets_opened=False,
          model_loaded=False,index_loaded=False,checks=checks)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test",action="store_true")
    mode.add_argument("--check-small-inputs",action="store_true")
    mode.add_argument("--preflight",action="store_true")
    parser.add_argument("--repo",type=Path,default=REPO)
    args=parser.parse_args()
    if args.self_test: self_test()
    elif args.check_small_inputs: event("small_inputs_checked",**small_check(safe(args.repo)),files_written=0)
    else: preflight(safe(args.repo))

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        # A preflight failure is never a scientific miss. Preserve partial artifacts.
        print(canonical({"status":"STOPPED","error_type":type(exc).__name__,"error":str(exc),
                         "experiment_state":"NOT_FROZEN_NOT_RUN","outputs_must_not_be_deleted":True}),file=sys.stderr)
        sys.exit(2)
