"""
Re-run the 3 retrieval signals split by FLIP vs BOTH-RIGHT (not wrong vs right).
"""
import json, statistics
from sentence_transformers import SentenceTransformer

COT  = "/mnt/home/user41/URBench/outputs/strategyqa/qwen3_14b/strategyqa_cot_qwen3_14b.jsonl"
SDFR = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_qwen3_14b.jsonl"
EVAL = "/mnt/home/user41/URBench/data/sdfr_splits/strategyqa_eval.jsonl"
EMBED_PATH = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm_label(a):
    return "ہاں" if a is True or str(a).lower() in ["true","yes"] else "نہیں"

cot = {r["qid"]: r["correct"] for r in load(COT)}
eval_facts = {r["qid"]: r.get("facts", []) for r in load(EVAL)}
sdfr = load(SDFR)

flip, both_right = [], []
for r in sdfr:
    qid = r["qid"]
    if qid not in cot: continue
    c, s = cot[qid], r["correct"]
    if c and not s: flip.append(r)
    elif c and s: both_right.append(r)

emb = SentenceTransformer(EMBED_PATH)

def feats(rows):
    max_sims, spreads, unanimous, gold_in, cov = [], [], 0, 0, []
    for r in rows:
        sims = [x["sim"] for x in r["retrieved"]]
        labels = [norm_label(x["a"]) for x in r["retrieved"]]
        max_sims.append(max(sims))
        spreads.append(max(sims)-min(sims))
        if len(set(labels)) == 1: unanimous += 1
        if r["gold"] in labels: gold_in += 1
        facts = eval_facts.get(r["qid"], [])
        if facts:
            fv = emb.encode(facts, normalize_embeddings=True, convert_to_numpy=True)
            rv = emb.encode([x["q"] for x in r["retrieved"]], normalize_embeddings=True, convert_to_numpy=True)
            cov.append(float((rv @ fv.T).max()))
    n = len(rows)
    return {
        "n": n,
        "max_sim_mean": round(statistics.mean(max_sims),4),
        "spread_mean": round(statistics.mean(spreads),4),
        "label_unanimous_%": round(100*unanimous/n,1),
        "gold_in_retrieved_%": round(100*gold_in/n,1),
        "fact_coverage_mean": round(statistics.mean(cov),4) if cov else None,
    }

print(f"FLIP (CoT right -> SDFR wrong): n={len(flip)}")
print(f"BOTH-RIGHT: n={len(both_right)}\n")

ff, bf = feats(flip), feats(both_right)
print(f"{'feature':<22}{'FLIP':>12}{'BOTH-RIGHT':>14}")
for k in ff:
    print(f"{k:<22}{str(ff[k]):>12}{str(bf[k]):>14}")