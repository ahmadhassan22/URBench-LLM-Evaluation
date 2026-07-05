"""
Phase 1 Error Analysis — SDFR-UR StrategyQA & CSQA
Layer 1: aggregate retrieval stats over ALL cases (correct vs wrong)
Layer 2: readable failure table for manual reading
"""

import json, statistics

FILES = {
    "strategyqa": "/mnt/home/user41/URBench/outputs/sdfr/sdfr_strategyqa_qwen3_14b.jsonl",
    "csqa":       "/mnt/home/user41/URBench/outputs/sdfr/sdfr_csqa_large_clean_qwen3_14b.jsonl",
}
OUT_DIR = "/mnt/home/user41/URBench/outputs/sdfr"

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm_label(a, dataset):
    # StrategyQA answers are booleans -> map to ہاں/نہیں; CSQA are letters
    if dataset == "strategyqa":
        return "ہاں" if a is True or str(a).lower() in ["true", "yes"] else "نہیں"
    return str(a).strip().upper()

def stats_block(rows, dataset):
    """Compute Layer-1 features for a set of rows."""
    max_sims, mean_sims, spreads = [], [], []
    label_unanimous, gold_in_retrieved, majority_eq_gold = 0, 0, 0
    for r in rows:
        sims   = [x["sim"] for x in r["retrieved"]]
        labels = [norm_label(x["a"], dataset) for x in r["retrieved"]]
        gold   = r["gold"]
        max_sims.append(max(sims))
        mean_sims.append(sum(sims) / len(sims))
        spreads.append(max(sims) - min(sims))
        if len(set(labels)) == 1:
            label_unanimous += 1
        if gold in labels:
            gold_in_retrieved += 1
        # majority label of the 3 retrieved
        maj = max(set(labels), key=labels.count)
        if maj == gold:
            majority_eq_gold += 1
    n = len(rows)
    return {
        "n": n,
        "max_sim_mean":  round(statistics.mean(max_sims), 4),
        "max_sim_median":round(statistics.median(max_sims), 4),
        "mean_sim_mean": round(statistics.mean(mean_sims), 4),
        "spread_mean":   round(statistics.mean(spreads), 4),
        "label_unanimous_%":   round(100 * label_unanimous / n, 1),
        "gold_in_retrieved_%": round(100 * gold_in_retrieved / n, 1),
        "majority_eq_gold_%":  round(100 * majority_eq_gold / n, 1),
    }

for dataset, path in FILES.items():
    data    = load(path)
    wrong   = [r for r in data if not r["correct"]]
    right   = [r for r in data if r["correct"]]

    print(f"\n{'='*60}\n{dataset.upper()}  (total {len(data)}, wrong {len(wrong)}, right {len(right)})\n{'='*60}")
    print(f"{'feature':<22}{'WRONG':>12}{'RIGHT':>12}")
    sw, sr = stats_block(wrong, dataset), stats_block(right, dataset)
    for k in sw:
        print(f"{k:<22}{sw[k]:>12}{sr[k]:>12}")

    # Layer 2: readable failures table (markdown)
    out = f"{OUT_DIR}/errors_{dataset}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {dataset} — SDFR-UR failures ({len(wrong)} cases)\n\n")
        f.write("| # | qid | question | gold | pred | retrieved (q / a / sim) | my category |\n")
        f.write("|---|-----|----------|------|------|--------------------------|-------------|\n")
        for i, r in enumerate(wrong, 1):
            ret = "<br>".join(
                f"{x['q'][:80]} / {norm_label(x['a'], dataset)} / {x['sim']:.3f}"
                for x in r["retrieved"])
            q = r["question"].replace("|", "،")
            f.write(f"| {i} | {r['qid']} | {q} | {r['gold']} | {r['pred']} | {ret} |  |\n")
    print(f"Failure table written: {out}")


# ============================================================
# LABEL-COPYING TEST: does pred follow the majority retrieved label?
# ============================================================
print(f"\n\n{'#'*60}\nLABEL-COPYING TEST\n{'#'*60}")

for dataset, path in FILES.items():
    data = load(path)
    follow_all, follow_wrong, follow_right = 0, 0, 0
    n_wrong, n_right = 0, 0
    for r in data:
        labels = [norm_label(x["a"], dataset) for x in r["retrieved"]]
        maj = max(set(labels), key=labels.count)
        pred = r["pred"]
        followed = (pred == maj)
        follow_all += followed
        if r["correct"]:
            n_right += 1; follow_right += followed
        else:
            n_wrong += 1; follow_wrong += followed
    n = len(data)
    print(f"\n{dataset.upper()}")
    print(f"  pred == majority(retrieved)  overall: {100*follow_all/n:.1f}%")
    print(f"  among WRONG cases:                    {100*follow_wrong/n_wrong:.1f}%")
    print(f"  among RIGHT cases:                    {100*follow_right/n_right:.1f}%")    