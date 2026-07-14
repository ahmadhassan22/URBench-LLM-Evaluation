"""
Strict data-integration & validation for official StrategyQA <-> URBench.

Official and URBench use DIFFERENT qid namespaces for the same questions, so we
join on normalized English question TEXT (conservative; no fuzzy matching).
Preserves BOTH ids; urbench_qid stays the primary key. Does NOT touch raw files.
Creates DEV50 from non-eval rows that have real (train-sourced) evidence.

VALIDATION-ONLY. No retrieval method here.
"""
import json, re, unicodedata, random, sys
from pathlib import Path

BASE = Path("/mnt/home/user41/URBench")
OFF  = BASE / "data/strategyqa_official"
RAW  = BASE / "data/strategyqa_raw"
SPL  = BASE / "data/sdfr_splits"

F_OFF_TRAIN = OFF / "train.json"
F_OFF_DEV   = OFF / "dev.json"
F_OFF_PARA  = OFF / "strategyqa_train_paragraphs.json"
F_URB_EN    = RAW / "strategyQA_train.json"
F_URB_UR    = RAW / "strategyQA_train_ur2_norm.jsonl"
F_EVAL      = SPL / "strategyqa_eval.jsonl"

F_OUT_MAP   = OFF / "strategyqa_official_mapped_urbench_qid.jsonl"
F_OUT_DEV50 = OFF / "dev50_seed42.jsonl"
F_OUT_DEVQ  = OFF / "dev50_seed42_qids.txt"

def die(msg):
    print("\n*** FATAL:", msg, "\n*** Stopping. No files written past this point.")
    sys.exit(1)

def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---- conservative text normalization (join key) ----
def norm_q(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("\u2019", "'").replace("\u2018", "'")
           .replace("\u201c", '"').replace("\u201d", '"')
           .replace("\u2013", "-").replace("\u2014", "-"))
    s = re.sub(r"\s+", " ", s).strip().casefold()
    return s

def norm_bool(a):
    if isinstance(a, bool): return a
    s = str(a).strip().casefold()
    if s in ("true","yes","ہاں"):  return True
    if s in ("false","no","نہیں"): return False
    return None

def main():
    # ---------- 1. LOAD ----------
    off_train = load_json(F_OFF_TRAIN)
    off_dev   = load_json(F_OFF_DEV)
    for r in off_train: r["_src"] = "train"
    for r in off_dev:   r["_src"] = "dev"
    official = off_train + off_dev
    para     = load_json(F_OFF_PARA)   # dict: pid -> {...}
    urb_en   = load_json(F_URB_EN)
    urb_ur   = load_jsonl(F_URB_UR)
    eval_rows= load_jsonl(F_EVAL)

    print(f"official train={len(off_train)} dev={len(off_dev)} combined={len(official)}")
    print(f"paragraph entries={len(para)}")
    print(f"URBench EN rows={len(urb_en)}  UR rows={len(urb_ur)}  eval rows={len(eval_rows)}")

    para_ids = set(para.keys())

    # ---------- 2. UNIQUENESS OF JOIN KEYS ----------
    def build_index(rows, qfield):
        idx, dups = {}, []
        for r in rows:
            k = norm_q(r.get(qfield))
            if k in idx: dups.append(k)
            else: idx[k] = r
        return idx, dups

    off_idx, off_dups = build_index(official, "question")
    urb_idx, urb_dups = build_index(urb_en, "question")
    ur_idx = { norm_q(r.get("question")): r for r in urb_ur }   # urdu side by same EN? no -> keyed by urbench qid below

    print(f"\nduplicate normalized questions -> official={len(off_dups)}  urbench={len(urb_dups)}")
    if off_dups: die(f"official has duplicate normalized questions e.g. {off_dups[:2]}")
    if urb_dups: die(f"urbench has duplicate normalized questions e.g. {urb_dups[:2]}")

    # URBench UR rows are keyed by qid (same namespace as urb_en). Map by qid.
    ur_by_qid = { r.get("qid"): r for r in urb_ur }

    # ---------- 3. ONE-TO-ONE MATCH ----------
    matched, unmatched_urb = [], []
    for k, urow in urb_idx.items():
        orow = off_idx.get(k)
        if orow is None:
            unmatched_urb.append(urow.get("question"))
            continue
        matched.append((urow, orow))
    unmatched_off = [official_row.get("question")
                     for k, official_row in off_idx.items() if k not in urb_idx]

    print(f"\nmatched pairs = {len(matched)} / {len(urb_idx)}")
    print(f"unmatched URBench = {len(unmatched_urb)}   unmatched official = {len(unmatched_off)}")
    if unmatched_urb:
        print("  --- first unmatched URBench questions (manual inspection) ---")
        for q in unmatched_urb[:10]: print("   URB:", q)
    if unmatched_off:
        print("  --- first unmatched official questions ---")
        for q in unmatched_off[:10]: print("   OFF:", q)
    if len(matched) != 2290:
        die(f"expected 2290 one-to-one matches, got {len(matched)}. Inspect unmatched above.")

    # ---------- 4. CROSS-VALIDATION ----------
    ans_mismatch, decomp_len_mismatch, term_diff, decomp_txt_diff = [], [], [], []
    for urow, orow in matched:
        if norm_bool(urow.get("answer")) != norm_bool(orow.get("answer")):
            ans_mismatch.append((urow.get("qid"), urow.get("answer"), orow.get("answer")))
        du, do = urow.get("decomposition") or [], orow.get("decomposition") or []
        if len(du) != len(do):
            decomp_len_mismatch.append((urow.get("qid"), len(du), len(do)))
        else:
            if [norm_q(x) for x in du] != [norm_q(x) for x in do]:
                decomp_txt_diff.append(urow.get("qid"))
        if urow.get("term") and orow.get("term") and norm_q(urow["term"]) != norm_q(orow["term"]):
            term_diff.append(urow.get("qid"))

    print(f"\nanswer mismatches      = {len(ans_mismatch)}")
    print(f"decomp length mismatch = {len(decomp_len_mismatch)}")
    print(f"decomp text differs    = {len(decomp_txt_diff)}  (reported, not fatal)")
    print(f"term differs           = {len(term_diff)}  (reported, not fatal)")
    if ans_mismatch:
        for m in ans_mismatch[:10]: print("   ANS MISMATCH:", m)
        die("answer mismatch(es) found — must be 0.")
    if decomp_len_mismatch:
        for m in decomp_len_mismatch[:10]: print("   DECOMP LEN:", m)
        print("  (decomposition length mismatches reported for inspection; continuing)")

    # ---------- 5. EVIDENCE VALIDATION ----------
    def walk_evidence(ev, ids, ops, noev):
        """Recursively collect paragraph IDs; count operation/no_evidence markers."""
        if isinstance(ev, str):
            if ev == "operation": ops[0]+=1
            elif ev == "no_evidence": noev[0]+=1
            else: ids.append(ev)
        elif isinstance(ev, list):
            for x in ev: walk_evidence(x, ids, ops, noev)
        elif isinstance(ev, dict):
            for x in ev.values(): walk_evidence(x, ids, ops, noev)

    all_ids, ops, noev = [], [0], [0]
    for _, orow in matched:
        walk_evidence(orow.get("evidence"), all_ids, ops, noev)
    uniq_ids = set(all_ids)
    missing  = sorted(uniq_ids - para_ids)
    print(f"\ntotal evidence references = {len(all_ids)}")
    print(f"unique evidence paragraph IDs = {len(uniq_ids)}")
    print(f"operation markers = {ops[0]}   no_evidence markers = {noev[0]}")
    print(f"missing paragraph IDs = {len(missing)}")
    if missing:
        print("   first missing:", missing[:10])
        die("referenced evidence paragraph IDs missing from paragraphs file.")

    # ---------- 6. WRITE MERGED MAP ----------
    eval_qids = { r.get("qid") for r in eval_rows }
    merged = []
    for urow, orow in matched:
        uq = urow.get("qid")
        merged.append({
            "urbench_qid":   uq,
            "official_qid":  orow.get("qid"),
            "official_source": orow.get("_src"),
            "question":      urow.get("question"),
            "question_ur":   (ur_by_qid.get(uq) or {}).get("question"),
            "answer":        urow.get("answer"),
            "term":          urow.get("term"),
            "description":   urow.get("description"),
            "facts":         urow.get("facts"),
            "decomposition": urow.get("decomposition"),
            "official_evidence": orow.get("evidence"),
            "is_eval":       uq in eval_qids,
        })
    with open(F_OUT_MAP, "w", encoding="utf-8") as f:
        for r in merged: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"\nwrote {F_OUT_MAP}  ({len(merged)} rows)")

    # ---------- 6b. EVAL COVERAGE ----------
    mapped_qids = { r["urbench_qid"] for r in merged }
    missing_eval = sorted(eval_qids - mapped_qids)
    print(f"evaluation qids mapped = {len(eval_qids)-len(missing_eval)} / {len(eval_qids)}")
    if missing_eval:
        print("   missing eval qids:", missing_eval[:10])
        die("some evaluation qids not present in merged map.")

    # ---------- 7. DEV50 (non-eval, train-sourced evidence only) ----------
    non_eval = [r for r in merged if not r["is_eval"]]
    non_eval_train = [r for r in non_eval if r["official_source"] == "train"]
    print(f"\nnon-eval rows = {len(non_eval)}  "
          f"(train-sourced w/ evidence = {len(non_eval_train)}, "
          f"dev-sourced = {len(non_eval)-len(non_eval_train)})")
    rng = random.Random(42)
    dev50 = rng.sample(non_eval_train, 50)
    dev50_qids = { r["urbench_qid"] for r in dev50 }
    assert dev50_qids & eval_qids == set(), "DEV50 intersects eval458!"
    with open(F_OUT_DEV50, "w", encoding="utf-8") as f:
        for r in dev50: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(F_OUT_DEVQ, "w", encoding="utf-8") as f:
        for q in sorted(dev50_qids): f.write(q+"\n")
    print(f"wrote {F_OUT_DEV50} (50 rows) and {F_OUT_DEVQ}")
    print(f"DEV50 ∩ eval458 = {len(dev50_qids & eval_qids)}")

    # ---------- SUMMARY ----------
    print("\n" + "="*60 + "\nVALIDATION SUMMARY")
    print(f"  mapped official<->URBench : {len(merged)}/2290")
    print(f"  evaluation qids mapped    : {len(eval_qids)-len(missing_eval)}/458")
    print(f"  answer mismatches         : {len(ans_mismatch)}")
    print(f"  duplicate norm questions  : {len(off_dups)+len(urb_dups)}")
    print(f"  unmatched (ambiguous)     : {len(unmatched_urb)}")
    print(f"  missing eval mappings     : {len(missing_eval)}")
    print(f"  missing evidence para IDs : {len(missing)}")
    print(f"  DEV50 size / disjoint     : {len(dev50_qids)} / {len(dev50_qids & eval_qids)==0}")
    print("="*60)

if __name__ == "__main__":
    main()