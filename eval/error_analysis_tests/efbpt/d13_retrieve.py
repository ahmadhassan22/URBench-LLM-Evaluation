#!/usr/bin/env python3
"""
d13_retrieve.py — D13 Group 2 retrieval step (freeze §2, §7.3). No language
model, no generation, no gold label in any query.

Retriever class of rag/retrieve.py, unchanged (MD5 pinned): FAISS flat
inner-product index rag/index/wikipedia_full.index (ntotal asserted =
23,963,971), embedder paraphrase-multilingual-MiniLM-L12-v2, normalized, no
prefix. Query = exactly question_ur (RAG_ur) or exactly the sealed R1 reading
(RAG_en). top_k = 5; no filter, no rerank. One sealed retrieval file per arm
and set; each record holds the 5 hits in rank order and the 5 facts-block
entries "Title: text" (white space runs replaced by one space).

Title map (D8 freeze §3 rule, d8_reachable_ceiling.build_title_map): each gold
evidence title of the set -> a corpus title or ABSENT. Written and sealed here;
the title share itself is computed only in the readout.

  --sets dev200,eval458  --mode TEST|FULL
TEST = DEV200 rows 0-4 under outputs/efbpt/d13/_test/ (never a result).
Run on a GPU node with >= 60 GB memory (rag/retrieve.py: the flat index needs
about 37 GB of RAM).
"""
import argparse
import os
import socket
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d13_common as C  # noqa: E402
from d13_common import die, git_state, md5_file, file_entry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="dev200,eval458")
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    mode = args.mode
    sets = ["dev200"] if mode == "TEST" else args.sets.split(",")
    for s in sets:
        if s not in C.SETS:
            die("unknown set %r" % s)
    print("=" * 78)
    print("D13 retrieval   sets %s   %s" % (sets, "TEST-5 — NOT A RESULT" if mode == "TEST" else "FULL"))
    print("=" * 78, flush=True)
    contracts = C.verify_contracts()
    if mode == "FULL":
        C.require_committed()
    got = md5_file(os.path.join(C.ROOT, C.RETRIEVE_PY[0]))
    if got != C.RETRIEVE_PY[1]:
        die("%s MD5 %s != the pinned %s" % (C.RETRIEVE_PY[0], got, C.RETRIEVE_PY[1]))
    print("[md5] %-18s %s  %s" % ("rag/retrieve.py", got, C.RETRIEVE_PY[0]))

    ctxs, guards = OrderedDict(), OrderedDict()
    for s in sets:
        ctxs[s] = C.make_ctx(s, mode)
        for arm in C.RAG_ARMS:
            guards[(arm, s)] = C.guard_rag_queries(arm, ctxs[s])
            for r in ctxs[s]["sub"]:
                q = C.rag_query(arm, r, ctxs[s])
                if not q or q != q.strip():
                    die("%s %s qid %s: the query is empty or has outer white space" % (arm, s, r["qid"]))
    print("[preflight] PASS", flush=True)
    if args.preflight_only:
        print("[preflight-only] done. No index loaded, nothing written.")
        return

    todo = []
    for s in sets:
        for arm in C.RAG_ARMS:
            pth = C.retrieval_paths(arm, s, mode)
            if os.path.exists(pth["seal"]):
                C.load_retrieval(arm, ctxs[s])
                print("[sealed] retrieval %s %s is already sealed; not run again" % (arm, s))
            else:
                todo.append((arm, s))
    need_map = [s for s in sets if not os.path.exists(C.retrieval_paths("RAG_ur", s, mode)["title_seal"])]

    scripts, git = C.script_md5s(), git_state()
    if todo:
        sys.path.insert(0, os.path.join(C.ROOT, "rag"))
        import retrieve as RET
        if os.path.abspath(RET.__file__) != os.path.join(C.ROOT, C.RETRIEVE_PY[0]):
            die("imported a retrieve module that is not %s" % C.RETRIEVE_PY[0])
        t0 = time.time()
        r = RET.Retriever(device="cuda")
        if r.index.ntotal != C.RAG_NTOTAL or len(r.offsets) != C.RAG_NTOTAL:
            die("index ntotal %d / meta offsets %d, the freeze says %d" % (r.index.ntotal, len(r.offsets), C.RAG_NTOTAL))
        print("[ok] index ntotal %d = meta lines = freeze §2; d=%d; loaded in %.0fs" % (r.index.ntotal, r.index.d, time.time() - t0), flush=True)
        for arm, s in todo:
            ctx = ctxs[s]
            pth = C.retrieval_paths(arm, s, mode)
            os.makedirs(C.guard_out(pth["dir"]), exist_ok=True)
            queries = [C.rag_query(arm, row, ctx) for row in ctx["sub"]]
            t1 = time.time()
            res = r.retrieve(queries, top_k=C.RAG_TOP_K)
            secs = time.time() - t1
            if len(res) != len(queries):
                die("%s %s: %d results for %d queries" % (arm, s, len(res), len(queries)))
            tmp = C.guard_out(pth["rows"]) + ".tmp"
            with open(tmp, "wb") as f:
                for row, q, hits in zip(ctx["sub"], queries, res):
                    if len(hits) != C.RAG_TOP_K:
                        die("%s %s qid %s: %d hits, %d expected" % (arm, s, row["qid"], len(hits), C.RAG_TOP_K))
                    hl = [OrderedDict([("rank", i + 1), ("row", h["row"]), ("score", h["score"]), ("title", h["title"]), ("text", h["text"])])
                          for i, h in enumerate(hits)]
                    f.write(C.record_line(OrderedDict([("qid", row["qid"]), ("arm", arm), ("set", s), ("query", q), ("top_k", C.RAG_TOP_K),
                                                       ("hits", hl), ("entries", [C.rag_entry(h) for h in hl])])))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, pth["rows"])
            seal = OrderedDict([
                ("experiment", "D13"), ("what", "retrieval for arm %s" % arm), ("arm", arm), ("set", s), ("mode", mode),
                ("n_rows", len(queries)), ("top_k", C.RAG_TOP_K), ("index_path", RET.INDEX_PATH), ("index_ntotal", int(r.index.ntotal)),
                ("index_dim", int(r.index.d)), ("index_bytes", os.path.getsize(RET.INDEX_PATH)), ("meta_path", RET.META_PATH),
                ("embedder", RET.MODEL_NAME), ("embedder_device", "cuda"), ("prefix", None), ("normalized", True),
                ("retrieve_py_md5", got), ("query", guards[(arm, s)]["query"]), ("guards", guards[(arm, s)]),
                ("seconds_total", round(secs, 3)), ("seconds_per_query", round(secs / len(queries), 4)),
                ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("hostname", socket.gethostname()), ("slurm_job_id", os.environ.get("SLURM_JOB_ID")),
                ("d13_freeze_md5", C.D13_FREEZE_MD5), ("contracts", contracts), ("scripts", scripts), ("git", git),
                ("files", OrderedDict([(pth["rows"], file_entry(pth["rows"]))]))])
            C.write_json(pth["seal"], seal)
            C.load_retrieval(arm, ctx)             # read back through the loader the student job uses
            print("  [sealed] %s  %d queries, top %d, %.1fs  rows MD5 %s" % (pth["seal"], len(queries), C.RAG_TOP_K, secs, seal["files"][pth["rows"]]["md5"]), flush=True)
        del r

    if need_map:
        from d8_reachable_ceiling import build_title_map
        gold = sorted({t for s in need_map for row in ctxs[s]["sub"] for t in row["required"] if t})
        print("\n[title map] D8 freeze §3 rule over %d distinct gold titles of %s (one pass over the corpus titles)" % (len(gold), need_map), flush=True)
        decisions = build_title_map(gold)
        for s in need_map:
            pth = C.retrieval_paths("RAG_ur", s, mode)
            os.makedirs(C.guard_out(pth["dir"]), exist_ok=True)
            mine = sorted({t for row in ctxs[s]["sub"] for t in row["required"] if t})
            tmp = C.guard_out(pth["title_map"]) + ".tmp"
            with open(tmp, "wb") as f:
                for t in mine:
                    f.write(C.record_line(decisions[t]))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, pth["title_map"])
            n_present = sum(1 for t in mine if decisions[t]["present"])
            C.write_json(pth["title_seal"], OrderedDict([
                ("experiment", "D13"), ("what", "gold evidence title -> corpus title (D8 freeze §3 rule); no model output"), ("set", s), ("mode", mode),
                ("n_gold_titles", len(mine)), ("n_present", n_present), ("rule", "d8_reachable_ceiling.build_title_map"),
                ("end_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("d13_freeze_md5", C.D13_FREEZE_MD5), ("scripts", scripts), ("git", git),
                ("files", OrderedDict([(pth["title_map"], file_entry(pth["title_map"]))]))]))
            print("  [sealed] %s  %d gold titles, %d present in the corpus" % (pth["title_seal"], len(mine), n_present), flush=True)
    print("\n[done] retrieval %s %s" % (sets, mode), flush=True)


if __name__ == "__main__":
    main()
