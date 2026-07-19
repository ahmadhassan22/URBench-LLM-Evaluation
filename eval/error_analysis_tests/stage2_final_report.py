"""
stage2_final_report.py — CPU-only. Preserves run-56070 outputs under versioned
names, then builds the final Stage 2 pilot report from existing predictions.
Writes: data/strategyqa_official/efbpt/stage2_run56070_final_report.txt
Touches NO gold, NO candidates, runs NO model.
"""
import json, re, shutil
from pathlib import Path

BASE = Path("/mnt/home/user41/URBench")
EF   = BASE / "data/strategyqa_official/efbpt"
RUN  = "run56070"

# ---- 6. PRESERVE (copy, never move; skip if already exists) ----
preserved = []
for f in ["stage2_pilot_predictions.jsonl", "stage2_pilot_diffs.jsonl", "stage2_pilot_summary.txt"]:
    src, dst = EF / f, EF / f.replace("stage2_pilot", f"stage2_pilot_{RUN}")
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst); preserved.append(dst.name)
log_src = BASE / "s2_pilot_56070.log"
log_dst = EF / f"stage2_pilot_{RUN}.log"
if log_src.exists() and not log_dst.exists():
    shutil.copy2(log_src, log_dst); preserved.append(log_dst.name)

def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace("_", " ")).strip().casefold()

def base_form(s):
    return norm(s).split(" (")[0]

def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

preds = {r["qid"]: r for r in load_jsonl(EF / "stage2_pilot_predictions.jsonl")}
gold  = {g["urbench_qid"]: g for g in load_jsonl(EF / "audit30_answers.jsonl")}
cands = {c["urbench_qid"]: c for c in load_jsonl(EF / "audit30_candidates.jsonl")}

L = []  # report lines
L.append(f"STAGE 2 PILOT — FINAL REPORT (run {RUN})")
L.append("=" * 60)
L.append(f"preserved files: {preserved if preserved else 'already existed, untouched'}")

# ---- 1. recall split: gold entity seeded (in evidence_pages) vs not ----
seen_seed = [0, 0]   # [found, total] for gold entities present in evidence_pages
seen_uns  = [0, 0]   # same for gold entities absent
for qid, g in gold.items():
    pj = preds.get(qid, {}).get("llm")
    if pj is None: continue
    seeded = {norm(p.get("title")) for p in cands[qid].get("evidence_pages", []) if p.get("title")}
    pred_e = {norm(e.get("canonical_title","")) for e in pj.get("question_entities", []) if e.get("canonical_title")}
    for e in g.get("question_entities", []):
        k = norm(e["canonical_title"])
        bucket = seen_seed if (k in seeded or base_form(k) in {base_form(s) for s in seeded}) else seen_uns
        bucket[1] += 1
        if k in pred_e: bucket[0] += 1
L.append("")
L.append("1. ENTITY RECALL SPLIT")
L.append(f"   gold entities IN evidence_pages : {seen_seed[0]}/{seen_seed[1]} = {100*seen_seed[0]/max(seen_seed[1],1):.1f}%")
L.append(f"   gold entities NOT in pages      : {seen_uns[0]}/{seen_uns[1]} = {100*seen_uns[0]/max(seen_uns[1],1):.1f}%")

# ---- 2. real errors vs alias/format ----
diffs = load_jsonl(EF / "stage2_pilot_diffs.jsonl")
alias, near, real_miss, real_extra = [], [], [], []
for row in diffs:
    for f in row["fields"]:
        if f["field"] != "entities": continue
        missed = set(f["gold"]) - set(f["pred"]); extra = set(f["pred"]) - set(f["gold"])
        for m in sorted(missed):
            match = [x for x in extra if base_form(x) == base_form(m)]
            sub   = [x for x in extra if m in x or x in m]
            if match:   alias.append((row["qid"][:8], m, match[0])); extra -= set(match)
            elif sub:   near.append((row["qid"][:8], m, sub[0]));    extra -= set(sub)
            else:       real_miss.append((row["qid"][:8], m))
        for x in sorted(extra): real_extra.append((row["qid"][:8], x))
L.append("")
L.append("2. ENTITY ERROR CLASSIFICATION")
L.append(f"   ALIAS (same entity, format diff)  : {len(alias)}  {alias}")
L.append(f"   NEAR-ALIAS (human call needed)    : {len(near)}  {near}")
L.append(f"   REAL missed                       : {len(real_miss)}")
for q, m in real_miss: L.append(f"      {q}  {m}")
L.append(f"   REAL extra                        : {len(real_extra)}  {real_extra}")

# ---- 3. reasons for every review row (recomputed from diffs; fixes blanks) ----
L.append("")
L.append("3. REVIEW REASONS (recomputed — blank-reason bug was: expected_answer_type")
L.append("   diffs were logged but never added to needs_review in the pilot script)")
for row in diffs:
    reasons = sorted({f["field"] + (f":step" + str(f["step"]) if "step" in f else "") for f in row["fields"]})
    L.append(f"   {row['qid'][:8]}: {', '.join(reasons)}")

# ---- 4. gold_intermediate_answer accuracy (previously unscored) ----
gia_all = [0, 0]; gia_nonnull = [0, 0]; gia_diffs = []
for qid, g in gold.items():
    pj = preds.get(qid, {}).get("llm")
    if pj is None: continue
    ps = {s.get("id"): s for s in pj.get("typed_plan", [])}
    for gs in g.get("typed_plan", []):
        p = ps.get(gs["id"], {})
        gv, pv = gs.get("gold_intermediate_answer"), p.get("gold_intermediate_answer")
        ok = norm(gv or "") == norm(pv or "")
        gia_all[1] += 1; gia_all[0] += ok
        if gv is not None:
            gia_nonnull[1] += 1; gia_nonnull[0] += ok
            if not ok: gia_diffs.append((qid[:8], gs["id"], pv, gv))
L.append("")
L.append("4. GOLD_INTERMEDIATE_ANSWER (exact match after norm)")
L.append(f"   all steps       : {gia_all[0]}/{gia_all[1]} = {100*gia_all[0]/max(gia_all[1],1):.1f}%")
L.append(f"   gold non-null   : {gia_nonnull[0]}/{gia_nonnull[1]} = {100*gia_nonnull[0]/max(gia_nonnull[1],1):.1f}%")
L.append("   NOTE: exact-match is a FLOOR — paraphrases score 0. Mismatches (pred vs gold):")
for q, sid, pv, gv in gia_diffs[:40]: L.append(f"      {q} step{sid}: {pv!r} vs {gv!r}")

# ---- 5. per-field routing decision (pre-declared threshold: >=90% auto) ----
L.append("")
L.append("5. ROUTING (threshold 90%, from run-56070 summary)")
for name, score, dec in [
    ("urdu_span", 95.1, "AUTO-ACCEPT"), ("step type (LLM)", 96.6, "AUTO-ACCEPT"),
    ("entity precision", 93.9, "AUTO-ACCEPT (additions rare)"),
    ("entity recall", 70.6, "REVIEW — human adds missed entities"),
    ("entity_ref", 83.0, "REVIEW"), ("evidence P/R", 73.5, "REVIEW"),
    ("answer type (LLM)", 84.1, "REVIEW (light)"),
    ("gold_intermediate_answer", round(100*gia_nonnull[0]/max(gia_nonnull[1],1),1), "see item 4"),
]:
    L.append(f"   {name:26s} {score:5.1f}%  -> {dec}")

report = "\n".join(L)
out = EF / f"stage2_{RUN}_final_report.txt"
out.write_text(report + "\n", encoding="utf-8")
print(report)
print("\nreport:", out)