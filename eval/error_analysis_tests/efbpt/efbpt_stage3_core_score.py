"""
efbpt_stage3_core_score.py — CPU-only DEVELOPMENT scorer. May read gold
(the verifier never does). Measures whether agreement routing prevents
false accepts. BLIND30 is development data, not validation.
"""
import json, re
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
VERF  = EF / "blind30_core_verified.jsonl"
GOLD  = EF / "blind30_gold.jsonl"
OUT_S = EF / "blind30_core_score_summary.txt"

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm(s):
    s = re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_", " ")).strip().casefold()
    return s.strip(" .,:;\"'")

def gold_core(g):
    ents = {norm(e["canonical_title"]): e.get("urdu_span","")
            for e in g.get("question_entities", []) if e.get("canonical_title")}
    steps = {s["id"]: {"type": norm(s.get("type")), "ref": norm(s.get("entity_ref")),
                       "at": norm(s.get("expected_answer_type"))}
             for s in g["typed_plan"]}
    return ents, steps

def row_core_correct(oc, gc):
    """Original core vs gold: list of differing fields (empty = fully correct)."""
    diffs = []
    oe, os_ = oc["entities"], oc["steps"]
    ge, gs_ = gc
    if set(oe) != set(ge): diffs.append("entities")
    else:
        for n in oe:
            if norm(oe[n]) != norm(ge[n]): diffs.append(f"span:{n}")
    for sid, g in gs_.items():
        o = os_.get(str(sid)) or os_.get(sid) or {}
        if o.get("type") != g["type"]: diffs.append(f"type:step{sid}")
        if o.get("ref")  != g["ref"]:  diffs.append(f"ref:step{sid}")
        if o.get("at")   != g["at"]:   diffs.append(f"atype:step{sid}")
    return diffs

def main():
    verf = load_jsonl(VERF)
    gold = {g["urbench_qid"]: g for g in load_jsonl(GOLD)}
    acc  = [r for r in verf if r["status"] == "ACCEPT"]
    rej  = [r for r in verf if r["status"] == "REJECT"]

    false_accepts, true_accepts = [], []
    for r in acc:
        d = row_core_correct(r["original_core"], gold_core(gold[r["qid"]]))
        (false_accepts if d else true_accepts).append((r["qid"], d))
    false_rejects, true_rejects = [], []
    for r in rej:
        d = row_core_correct(r["original_core"], gold_core(gold[r["qid"]]))
        (true_rejects if d else false_rejects).append((r["qid"], d, r["reasons"][:4]))

    # per-field accuracy among ACCEPTED rows
    F = {k: [0,0] for k in ["ent","span","type","ref","at"]}
    for r in acc:
        ge, gs_ = gold_core(gold[r["qid"]])
        oe, os_ = r["original_core"]["entities"], r["original_core"]["steps"]
        F["ent"][0] += set(oe) == set(ge); F["ent"][1] += 1
        for n in set(oe) & set(ge):
            F["span"][0] += norm(oe[n]) == norm(ge[n]); F["span"][1] += 1
        for sid, g in gs_.items():
            o = os_.get(str(sid)) or os_.get(sid) or {}
            F["type"][0] += o.get("type") == g["type"]; F["type"][1] += 1
            F["ref"][0]  += o.get("ref")  == g["ref"];  F["ref"][1]  += 1
            F["at"][0]   += o.get("at")   == g["at"];   F["at"][1]   += 1

    cov = len(acc)/30
    prec = len(true_accepts)/len(acc) if acc else 0.0
    def pct(k): return f"{100*F[k][0]/F[k][1]:.1f}% (n={F[k][1]})" if F[k][1] else "n/a"

    L = ["STAGE 3 DEV SCORE — agreement routing vs gold (BLIND30 = dev data)",
         "=" * 66,
         f"accepted coverage        : {len(acc)}/30 = {100*cov:.1f}%",
         f"accepted precision (rows): {len(true_accepts)}/{len(acc)} = {100*prec:.1f}%" if acc else "accepted precision: n/a",
         f"FALSE ACCEPTS (primary)  : {len(false_accepts)}",
         *[f"    {q[:8]}: {d}" for q, d in false_accepts],
         f"true rejects             : {len(true_rejects)}",
         f"FALSE REJECTS            : {len(false_rejects)}",
         *[f"    {q[:8]}: reject reasons {rr}" for q, d, rr in false_rejects],
         "",
         "ACCEPTED-ROW FIELD ACCURACY vs gold:",
         f"  entity set exact : {pct('ent')}",
         f"  urdu_span        : {pct('span')}",
         f"  step type        : {pct('type')}",
         f"  entity_ref       : {pct('ref')}",
         f"  answer type      : {pct('at')}",
         "",
         f"projection: accepted pool ~{round(cov*1770)} / 1770 (n=30, operational only)",
         "rejected rows are DROPPED (workload = 0), not manually repaired.",
         "",
         "NOTE: 100% universe recall guaranteed availability, not correctness.",
         "This is development data. Fresh-sample gate (frozen): coverage >=60%",
         "AND accepted core precision >=95% (+ per-field >=95%, no systematic error)."]
    summary = "\n".join(L)
    OUT_S.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\nsummary:", OUT_S)

if __name__ == "__main__":
    main()