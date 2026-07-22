"""
efbpt_stage3_universe_preflight.py — CPU-only, DEVELOPMENT-ONLY preflight.

Question: does the GENERAL candidate universe (term + evidence_pages +
title_decisions + extra_entities + original entity_ref values) already contain
every GOLD question entity? If not, the verifier can never surface the missing
entity and would FALSE-ACCEPT incomplete rows.

Reads gold ONLY to test reachability. Does NOT write any universe, does NOT
feed gold anywhere. If recall < 100%, STOP and report which general source is
missing — we fix the general construction, never insert gold-derived titles.
"""
import json, re
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
CANDS = EF / "blind30_candidates.jsonl"
PREDS = EF / "blind30_predictions.jsonl"
GOLD  = EF / "blind30_gold.jsonl"          # reachability test ONLY
OUT_S = EF / "blind30_universe_preflight.txt"

PREV_MISSED_8 = {  # the 8 entity misses from the ablation, to confirm reachability
    "monogamy","tricarboxylic acid","la marseillaise","largemouth bass",
    "windows phone","sonnet","office of migrant education","mike dewine"}

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm(s):
    s = re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_", " ")).strip().casefold()
    return s.strip(" .,:;\"'")

def build_universe(c, p):
    """GENERAL sources only — no gold."""
    titles = set()
    src = {}   # norm -> which source(s) it came from (for reporting)
    def add(t, where):
        if not t: return
        n = norm(t)
        if not n: return
        titles.add(n); src.setdefault(n, set()).add(where)
    add(c.get("term"), "term")
    for ep in c.get("evidence_pages", []): add(ep.get("title"), "evidence_page")
    for d in p.get("title_decisions", []): add(d.get("title"), "title_decision")
    for e in p.get("extra_entities", []):  add(e.get("canonical_title"), "extra_entity")
    for s in (p.get("typed_plan") or []):  add(s.get("entity_ref"), "orig_entity_ref")
    return titles, src

def main():
    cands = {r["urbench_qid"]: r for r in load_jsonl(CANDS)}
    preds = {r["qid"]: r for r in load_jsonl(PREDS)}
    gold  = {r["urbench_qid"]: r for r in load_jsonl(GOLD)}

    total_gold = found = 0
    missing = []          # (qid, gold_title)
    reached_of_8 = set()

    for q in sorted(cands):
        uni, _src = build_universe(cands[q], preds[q])
        for e in gold[q].get("question_entities", []):
            gt = e.get("canonical_title")
            if not gt: continue
            total_gold += 1
            gn = norm(gt)
            if gn in uni:
                found += 1
                if gn in PREV_MISSED_8: reached_of_8.add(gn)
            else:
                missing.append((q, gt))

    recall = 100 * found / total_gold if total_gold else 0.0
    L = []
    L.append("STAGE 3 CANDIDATE-UNIVERSE PREFLIGHT (development-only; gold used")
    L.append("ONLY to test reachability; verifier never reads gold)")
    L.append("=" * 66)
    L.append(f"gold entities total       : {total_gold}")
    L.append(f"reachable in universe     : {found}")
    L.append(f"candidate-universe recall : {recall:.1f}%")
    L.append("")
    L.append("PREVIOUSLY-MISSED 8 — reachability:")
    for t in sorted(PREV_MISSED_8):
        L.append(f"  {'REACHABLE' if t in reached_of_8 else 'NOT reachable':13s}  {t}")
    L.append("")
    if missing:
        L.append(f"!! MISSING gold entities NOT in general universe: {len(missing)}")
        # diagnose which single general source, if added, would recover each
        for q, gt in missing:
            L.append(f"   {q[:8]}  {gt!r}")
        L.append("")
        L.append("VERDICT: recall < 100% -> STOP. Do NOT run the GPU verifier.")
        L.append("Fix the GENERAL universe construction (which source is missing")
        L.append("these titles?), NOT by inserting gold-derived titles.")
    else:
        L.append("VERDICT: recall = 100%. Every gold entity is reachable from the")
        L.append("general universe. Safe to build and run the three Stage 3 scripts.")

    summary = "\n".join(L)
    OUT_S.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\nsummary:", OUT_S)

if __name__ == "__main__":
    main()