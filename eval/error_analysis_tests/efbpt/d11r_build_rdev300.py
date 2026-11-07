#!/usr/bin/env python3
"""
d11r_build_rdev300.py — build and seal RDEV300 (docs/EFBPT_D11R_READER_FREEZE.md §1).

Pool = rows of data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl
(file order) whose urbench_qid is in neither DEV200 (data/strategyqa_official/
dev200_seed4242.jsonl) nor eval458 (the is_eval == True rows of the mapped
file), with non-empty question_ur and question_en. Expected pool size 1632.
Sample: random.Random(2026).sample(pool, 300), written in sample order with the
full mapped row, to data/strategyqa_official/efbpt/rdev300_seed2026.jsonl, plus
rdev300_seed2026.SEAL.json (MD5, n, overlap checks, pool size, source MD5s).

Refuses to overwrite an existing RDEV300 file unless --force is given (the set
is sealed once). Prints n, the overlap checks (must be 0) and the MD5.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)

FREEZE_PATH = "docs/EFBPT_D11R_READER_FREEZE.md"
MAPPED_PATH = "data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl"
MAPPED_MD5 = "bb982798279f5d519ddef01f5cb7ba8c"
DEV200_PATH = "data/strategyqa_official/dev200_seed4242.jsonl"
DEV200_MD5 = "4907f292175b543b17799e6e37b6724b"
OUT_PATH = "data/strategyqa_official/efbpt/rdev300_seed2026.jsonl"
SEAL_PATH = "data/strategyqa_official/efbpt/rdev300_seed2026.SEAL.json"
SEED = 2026
N = 300
EXPECTED_POOL = 1632
EXPECTED_EVAL = 458


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def is_true(v):
    return v is True or str(v) == "True"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing RDEV300 (normally never)")
    args = ap.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        die("%s exists; RDEV300 is sealed once. Use --force only to rebuild deliberately." % OUT_PATH)
    for p, want in ((MAPPED_PATH, MAPPED_MD5), (DEV200_PATH, DEV200_MD5)):
        got = md5_file(p)
        if got != want:
            die("%s MD5 %s != pinned %s" % (p, got, want))
        print("[md5] %s  %s" % (got, p))
    if not os.path.exists(FREEZE_PATH):
        die("freeze missing: " + FREEZE_PATH)
    freeze_md5 = md5_file(FREEZE_PATH)
    print("[md5] %s  %s" % (freeze_md5, FREEZE_PATH))

    mapped = read_jsonl(MAPPED_PATH)
    if len({r["urbench_qid"] for r in mapped}) != len(mapped):
        die("mapped file has duplicate urbench_qid")
    dev = [r["urbench_qid"] for r in read_jsonl(DEV200_PATH)]
    if len(dev) != 200 or len(set(dev)) != 200:
        die("DEV200 does not hold 200 unique qids")
    dev = set(dev)
    ev = {r["urbench_qid"] for r in mapped if is_true(r.get("is_eval"))}
    if len(ev) != EXPECTED_EVAL:
        die("eval458 has %d rows, expected %d" % (len(ev), EXPECTED_EVAL))
    if dev & ev:
        die("DEV200 and eval458 overlap: %d" % len(dev & ev))

    pool = [r for r in mapped if r["urbench_qid"] not in dev and r["urbench_qid"] not in ev
            and str(r.get("question_ur", "")).strip() and str(r.get("question_en", "")).strip()]
    print("[pool] %d rows (mapped %d − DEV200 %d − eval458 %d, non-empty ur/en)" % (len(pool), len(mapped), len(dev), len(ev)))
    if len(pool) != EXPECTED_POOL:
        die("pool size %d != expected %d" % (len(pool), EXPECTED_POOL))

    rng = random.Random(SEED)
    sample = rng.sample(pool, N)
    qids = [r["urbench_qid"] for r in sample]
    if len(set(qids)) != N:
        die("sample has duplicate qids")
    ov_dev, ov_ev = len(set(qids) & dev), len(set(qids) & ev)
    print("[check] n = %d   overlap DEV200 = %d   overlap eval458 = %d" % (len(sample), ov_dev, ov_ev))
    if ov_dev or ov_ev:
        die("RDEV300 overlaps a reserved set")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    n_disk = sum(1 for _ in open(OUT_PATH, "r", encoding="utf-8"))
    if n_disk != N:
        die("wrote %d rows but disk has %d lines" % (N, n_disk))
    out_md5 = md5_file(OUT_PATH)
    yes = sum(1 for r in sample if is_true(r.get("answer")))
    seal = OrderedDict([
        ("item_set", "RDEV300"), ("freeze", OrderedDict([("path", FREEZE_PATH), ("md5", freeze_md5)])),
        ("built", time.strftime("%Y-%m-%d %H:%M:%S")), ("seed", SEED), ("sampler", "random.Random(%d).sample(pool, %d)" % (SEED, N)),
        ("sources", OrderedDict([(MAPPED_PATH, MAPPED_MD5), (DEV200_PATH, DEV200_MD5)])),
        ("pool_size", len(pool)), ("eval458_size", len(ev)), ("dev200_size", len(dev)),
        ("n", N), ("overlap_dev200", ov_dev), ("overlap_eval458", ov_ev),
        ("gold_yes", yes), ("gold_no", N - yes),
        ("rows_with_evidence_ids", sum(1 for r in sample if r.get("evidence_paragraph_ids"))),
        ("file", OrderedDict([("path", OUT_PATH), ("md5", out_md5), ("bytes", os.path.getsize(OUT_PATH)), ("lines", n_disk)])),
        ("qids", qids),
    ])
    with open(SEAL_PATH, "w", encoding="utf-8") as f:
        json.dump(seal, f, indent=2, ensure_ascii=False)
    print("[sealed] %s  md5 %s  bytes %d  lines %d  (gold yes %d / no %d)" % (OUT_PATH, out_md5, os.path.getsize(OUT_PATH), n_disk, yes, N - yes))
    print("[sealed] %s" % SEAL_PATH)


if __name__ == "__main__":
    main()
