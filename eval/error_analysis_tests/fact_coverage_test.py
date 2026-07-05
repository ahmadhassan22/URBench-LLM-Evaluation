"""
Fact-coverage test (ANALYSIS ONLY — facts never go to the model)
Q: do retrieved examples semantically contain the facts the question needs?
Compares WRONG vs RIGHT cases. StrategyQA only (it has gold `facts`).
No GPU needed.
"""

import json, statistics
from sentence_transformers import SentenceTransformer

EMBED_PATH = "/mnt/home/user41/downloaded_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT     = "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_qwen3_14b.jsonl"
EVAL       = "/mnt/home/user41/URBench/data/sdfr_splits/strategyqa_eval.jsonl"

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# map qid -> gold facts (Urdu)
eval_facts = {r["qid"]: r.get("facts", []) for r in load(EVAL)}

emb = SentenceTransformer(EMBED_PATH)

def coverage(row):
    """max cosine sim between any retrieved example and any gold fact."""
    facts = eval_facts.get(row["qid"], [])
    if not facts:
        return None
    ret_texts = [x["q"] for x in row["retrieved"]]
    fv = emb.encode(facts, normalize_embeddings=True, convert_to_numpy=True)
    rv = emb.encode(ret_texts, normalize_embeddings=True, convert_to_numpy=True)
    sims = rv @ fv.T          # (3 retrieved) x (n facts)
    return float(sims.max())  # best fact-to-retrieved match

data = load(OUTPUT)
cov_wrong, cov_right = [], []
for r in data:
    c = coverage(r)
    if c is None:
        continue
    (cov_right if r["correct"] else cov_wrong).append(c)

def summ(x):
    return f"n={len(x):<4} mean={statistics.mean(x):.4f}  median={statistics.median(x):.4f}"

print("\nFACT-COVERAGE (max retrieved↔gold-fact similarity)")
print(f"  WRONG : {summ(cov_wrong)}")
print(f"  RIGHT : {summ(cov_right)}")
print(f"  gap (right - wrong mean): {statistics.mean(cov_right)-statistics.mean(cov_wrong):+.4f}")

# how many cases have essentially NO fact coverage?
def frac_below(x, t): return 100*sum(v < t for v in x)/len(x)
for t in [0.40, 0.45, 0.50]:
    print(f"  % below {t}:  WRONG {frac_below(cov_wrong,t):.1f}%   RIGHT {frac_below(cov_right,t):.1f}%")