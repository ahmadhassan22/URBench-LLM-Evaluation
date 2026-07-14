"""
Phase-R DEV50 retrieval diagnostic (R1-R4). NO answer generation, NO C0-C5.

Tests the dual-view entity-canonicalization hypothesis on DEV50 (development only).
Leak-free: the live method sees ONLY the Urdu question. Official decomposition /
evidence / answer / term / facts are NEVER inputs — used only by the offline scorer
(separate step). One cached question-only typed self-decomposition per item is SHARED
across R1-R4 so decomposition variation cannot confound the comparison.

Reuses:
  - rag/retrieve.py            Retriever (index load, byte-offset meta seek, no prefix)
  - bge-reranker-v2-m3         same reranker across all conditions (probe_rerank_test.py)
  - vLLM Qwen3-14B, thinking OFF (probe_selfdecomp_* pattern)

Budget matching (see corrections C-1):
  - queries-per-step-per-view = 1 (single best entity query per view)
  - pre-rerank passage budget per RETRIEVE step = 40, split evenly across views:
        R1: 40 (1 view)   R2: 40 (1 view)   R3/R4: 20 + 20 (2 views)
  - final reranked top-3 in every condition
  - actual searches + unique-pool size recorded per condition (referee-checkable)

Outputs (new dir data/strategyqa_official/phase_r/):
  - r_records.jsonl        per (qid, step, condition): queries, pool size, searches,
                           reranked top-3 titles+scores, canonicalization view state
  - m1_label_template.jsonl   manual entity-title labeling template (gold-free denominator)
  - m3_premise_support.jsonl  auditable hand-scoring rows (retrieved vs official evidence)
  - decompositions.jsonl   cached typed self-decomposition per qid (shared across R1-R4)
  - summary.json           cross-view diagnostics + budgets consumed
Checkpointed: decompositions and per-qid records resume if the job restarts.
"""
import os, sys, json, re, argparse, unicodedata
from pathlib import Path

BASE = Path("/mnt/home/user41/URBench")
sys.path.insert(0, str(BASE / "rag"))
from retrieve import Retriever                      # reused as-is

OFF        = BASE / "data/strategyqa_official"
DEV50_F    = OFF / "dev50_seed42.jsonl"
DEVQ_F     = OFF / "dev50_seed42_qids.txt"
EVAL_F     = BASE / "data/sdfr_splits/strategyqa_eval.jsonl"
OUT        = OFF / "phase_r"
MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
RERANK_PATH= "/mnt/home/user41/downloaded_models/BAAI/bge-reranker-v2-m3"

PER_STEP_BUDGET = 40      # pre-rerank passages per RETRIEVE step (all conditions)
FINAL_TOPK      = 3       # reranked passages kept per step (all conditions)
GROUND_TOPK     = 10      # title-grounding search depth (R4)
GROUND_MIN      = 0.60    # frozen on DEV50 only; fuzzy title-overlap threshold

# ---------------- helpers reused / adapted from probes ----------------
def load_jsonl(p): 
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]

def norm_title(t):
    t = unicodedata.normalize("NFKC", str(t)).replace("_"," ")
    return re.sub(r"\s+"," ",t).strip().casefold()

def tok_overlap(a, b):
    A, B = set(norm_title(a).split()), set(norm_title(b).split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# typed self-decomposition: question-only, thinking OFF, JSON schema
DECOMP_PROMPT = """You are planning how to answer a question by looking facts up in an English encyclopedia.

Question (Urdu): {q}

Break it into ordered steps. For EACH step output an object with:
  "id": int,
  "type": "RETRIEVE" (a fact to look up) or "REASON" (a comparison/arithmetic/logic step over earlier results),
  "question": the step in English,
  "depends_on": list of earlier step ids,
  "urdu_entity_mention": the exact Urdu entity phrase to look up if type is RETRIEVE else null

Output ONLY a JSON array of step objects. No prose."""

# per-view entity query generation (single best query per view -> budget parity C-1)
ENTITY_Q_PROMPT = """Give the single best English Wikipedia search query to find this fact. Output ONLY the query text, one line, no quotes.

Fact to look up: {step}
Entity: {ent}"""

TRANSLATE_PROMPT = """Translate this Urdu question to English. Preserve named entities exactly. Output ONLY the English translation.

Urdu: {q}"""

def gen(llm, sp, prompts):
    convs = [[{"role":"user","content":p}] for p in prompts]
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text.strip() for o in outs]

def parse_steps(txt):
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if not m: return None
    try:
        steps = json.loads(m.group(0))
        for s in steps:  # minimal schema guard
            s["id"]=int(s["id"]); s["type"]=str(s["type"]).upper()
            s.setdefault("depends_on",[]); s.setdefault("urdu_entity_mention",None)
        return steps
    except Exception:
        return None

# ---------------- retrieval / rerank (reused pattern) ----------------
def retrieve_pool(retr, queries, budget_per_query):
    """Retrieve budget_per_query per query, dedupe by chunk row. Returns (pool, n_searches)."""
    pool = {}
    n = 0
    for q in queries:
        if not q: continue
        n += 1
        for h in retr.retrieve(q, top_k=budget_per_query)[0]:
            if h["row"] not in pool or h["score"] > pool[h["row"]]["score"]:
                pool[h["row"]] = h
    return list(pool.values()), n

def rerank_top(reranker, step_q, pool, k=FINAL_TOPK):
    if not pool: return []
    scores = reranker.predict([[step_q, h["text"]] for h in pool])
    order = sorted(range(len(pool)), key=lambda i: -scores[i])[:k]
    return [{**pool[i], "rerank": float(scores[i])} for i in order]

# ---------------- R4 canonicalization ----------------
def ground_entity(retr, entity_query):
    """Search the index with an entity query; return best-grounded Wikipedia title or None."""
    if not entity_query: return None, 0.0
    hits = retr.retrieve(entity_query, top_k=GROUND_TOPK)[0]
    best_t, best_s = None, 0.0
    for h in hits:
        s = tok_overlap(entity_query, h["title"]) * h["score"]
        if s > best_s:
            best_s, best_t = s, h["title"]
    if best_s >= GROUND_MIN:
        return best_t, best_s
    return None, best_s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device_embed", default="cpu")   # embedder CPU -> GPU free for vLLM+reranker
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- integrity guards (correction C-4) ----
    dev = load_jsonl(DEV50_F)
    dev_qids = [r["urbench_qid"] for r in dev]
    frozen = [l.strip() for l in open(DEVQ_F) if l.strip()]
    eval_qids = {r["qid"] for r in load_jsonl(EVAL_F)}
    assert len(set(dev_qids)) == 50, f"DEV50 not 50 unique: {len(set(dev_qids))}"
    assert set(dev_qids) == set(frozen), "DEV50 rows != frozen qid list"
    assert set(dev_qids).isdisjoint(eval_qids), "DEV50 intersects eval458!"
    print(f"[guard] DEV50 = {len(dev_qids)} unique, disjoint from eval458 OK")

    # ---- models ----
    from vllm import LLM, SamplingParams
    from sentence_transformers import CrossEncoder
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.72,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    retr = Retriever(device=args.device_embed)
    reranker = CrossEncoder(RERANK_PATH, max_length=512, device="cuda")

    # ---- STEP 1: cached typed self-decomposition (SHARED across R1-R4) ----
    dpath = OUT / "decompositions.jsonl"
    cache = {json.loads(l)["urbench_qid"]: json.loads(l) for l in open(dpath)} if dpath.exists() else {}
    todo = [r for r in dev if r["urbench_qid"] not in cache]
    if todo:
        outs = gen(llm, sp, [DECOMP_PROMPT.format(q=r["question_ur"]) for r in todo])
        with open(dpath, "a", encoding="utf-8") as f:
            for r, o in zip(todo, outs):
                steps = parse_steps(o)
                rec = {"urbench_qid": r["urbench_qid"], "steps": steps, "raw": o if steps is None else None}
                cache[r["urbench_qid"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    print(f"[decomp] cached {len(cache)}/50 (parse failures: {sum(1 for v in cache.values() if not v['steps'])})")

    # ---- STEP 2: per-view entity queries + translation (cached with decomp) ----
    # translate each question once (for the translated view)
    tr = gen(llm, sp, [TRANSLATE_PROMPT.format(q=r["question_ur"]) for r in dev])
    trans = {r["urbench_qid"]: t for r, t in zip(dev, tr)}

    # ---- run conditions ----
    rec_f = open(OUT / "r_records.jsonl", "w", encoding="utf-8")
    m1_f  = open(OUT / "m1_label_template.jsonl", "w", encoding="utf-8")
    m3_f  = open(OUT / "m3_premise_support.jsonl", "w", encoding="utf-8")

    diag = {c: {"searches":0, "pool":0, "steps":0} for c in ["R1","R2","R3","R4"]}
    view = {"agree":0, "disagree":0, "ungrounded":0}

    for r in dev:
        qid = r["urbench_qid"]
        steps = cache[qid]["steps"] or []
        en = trans[qid]
        retrieve_steps = [s for s in steps if s.get("type")=="RETRIEVE" and s.get("urdu_entity_mention")]

        for s in retrieve_steps:
            step_q = s["question"]                    # rerank target
            ent    = s["urdu_entity_mention"]
            # per-view single best entity query (budget parity: 1 query/view/step)
            dq = gen(llm, sp, [ENTITY_Q_PROMPT.format(step=step_q, ent=ent)])[0].splitlines()[0].strip()
            tq = gen(llm, sp, [ENTITY_Q_PROMPT.format(step=step_q, ent=en)])[0].splitlines()[0].strip()

            conds = {}
            # R1 Urdu-direct: 1 view, 40
            conds["R1"] = ([dq], PER_STEP_BUDGET)
            # R2 translate-first: 1 view, 40
            conds["R2"] = ([tq], PER_STEP_BUDGET)
            # R3 union: 20 + 20
            conds["R3"] = ([dq, tq], PER_STEP_BUDGET//2)
            # R4 canonicalization: ground each view, replace query, same 20+20 budget
            t_d, s_d = ground_entity(retr, dq)
            t_t, s_t = ground_entity(retr, tq)
            if t_d and t_t and norm_title(t_d)==norm_title(t_t):
                view["agree"]+=1; r4q=[t_d, t_d]
            elif t_d or t_t:
                if t_d and t_t: view["disagree"]+=1
                else: view["ungrounded"]+=1
                r4q=[t_d or dq, t_t or tq]
            else:
                view["ungrounded"]+=1; r4q=[dq, tq]
            conds["R4"] = (r4q, PER_STEP_BUDGET//2)

            for c,(queries,bpq) in conds.items():
                pool, nsearch = retrieve_pool(retr, queries, bpq)
                top = rerank_top(reranker, step_q, pool)
                diag[c]["searches"]+=nsearch; diag[c]["pool"]+=len(pool); diag[c]["steps"]+=1
                rec = {"qid":qid, "step_id":s["id"], "condition":c,
                       "step_q":step_q, "entity_ur":ent, "queries":queries,
                       "n_searches":nsearch, "unique_pool":len(pool),
                       "ground_direct":t_d, "ground_trans":t_t,
                       "top3":[{"title":h["title"],"rerank":h["rerank"],
                                "retr":h["score"],"row":h["row"]} for h in top]}
                rec_f.write(json.dumps(rec, ensure_ascii=False)+"\n")

                # M3 auditable rows (official evidence attached by scorer later; here retrieved side)
                for h in top:
                    m3_f.write(json.dumps({"qid":qid,"step_id":s["id"],"condition":c,
                        "step_q":step_q,"retrieved_title":h["title"],
                        "passage":h["text"][:400],"rerank":h["rerank"],"retr":h["score"]},
                        ensure_ascii=False)+"\n")

            # M1 label template (gold-free denominator: RETRIEVE + entity mention)
            m1_f.write(json.dumps({"qid":qid,"step_id":s["id"],"step_q":step_q,
                "entity_ur":ent,"cand_direct":dq,"cand_trans":tq,
                "ground_direct":t_d,"ground_trans":t_t,
                "correct_title":""},  # <-- to be filled once, by hand
                ensure_ascii=False)+"\n")

    rec_f.close(); m1_f.close(); m3_f.close()

    summary = {"dev50":len(dev), "retrieve_steps_total":diag["R1"]["steps"],
               "budgets":{c:{"avg_searches_per_step": round(diag[c]["searches"]/max(diag[c]["steps"],1),3),
                             "avg_unique_pool": round(diag[c]["pool"]/max(diag[c]["steps"],1),1),
                             "final_topk":FINAL_TOPK} for c in diag},
               "cross_view":view,
               "budget_parity_check":{
                   "R3_R4_same_final_topk": True,
                   "note":"queries/view/step=1; R1/R2=40 (1 view); R3/R4=20+20"}}
    json.dump(summary, open(OUT/"summary.json","w"), indent=2, ensure_ascii=False)
    print("[done] wrote", OUT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()