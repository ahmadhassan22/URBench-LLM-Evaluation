"""
efbpt_stage2_pilot.py — Stage 2 PILOT on the frozen AUDIT30 rows only.

WHAT IT DOES (plain words)
--------------------------
1. Loads the 30 prefilled candidate rows, the official paragraph texts, and
   YOUR gold answers (audit30_answers.jsonl).
2. Fills fields two ways:
     RULES  : depends_on (regex), step type (regex hint), expected_answer_type
              (pattern rule). Cheap, deterministic.
     LLM    : Qwen3-14B (thinking OFF, temp 0, strict JSON) fills the semantic
              fields: question entities + urdu spans, per-step type, entity_ref,
              expected_answer_type, kept evidence, intermediate answers.
3. Scores BOTH against your gold, field by field, and writes a summary.
   Nothing is modified; only new stage2_pilot_* files are written.
   Does NOT touch the 1,770. Does NOT build hard negatives.

OUTPUT
------
    data/strategyqa_official/efbpt/stage2_pilot_predictions.jsonl
    data/strategyqa_official/efbpt/stage2_pilot_diffs.jsonl
    data/strategyqa_official/efbpt/stage2_pilot_summary.txt
"""

import json, re
from pathlib import Path
from vllm import LLM, SamplingParams

BASE   = Path("/mnt/home/user41/URBench")
OFF    = BASE / "data/strategyqa_official"
EFBPT  = OFF / "efbpt"

CANDS  = EFBPT / "audit30_candidates.jsonl"
GOLD   = EFBPT / "audit30_answers.jsonl"
PARAS  = OFF / "strategyqa_train_paragraphs.json"

OUT_P  = EFBPT / "stage2_pilot_predictions.jsonl"
OUT_D  = EFBPT / "stage2_pilot_diffs.jsonl"
OUT_S  = EFBPT / "stage2_pilot_summary.txt"

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
PARA_TRUNC = 1200   # chars per paragraph fed to the LLM (keep prompts sane)

ANSWER_TYPES = ["BOOLEAN","ENTITY","LOCATION","DATE","NUMBER","SET","SHORT_TEXT"]

REASON_PAT = re.compile(
    r"^(is|are|was|were|does|do|did|can|could|would|will)\b.*#\d"
    r"|#\d.*\b(multiplied|divided|plus|minus|times|less than|greater than|more than"
    r"|equal|same as|part of|included|within|before|after|between)\b"
    r"|\b(what is)\s+#\d", re.IGNORECASE)


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def para_text(paras, pid):
    e = paras.get(pid)
    if e is None:
        return None
    if isinstance(e, dict):
        return e.get("content") or e.get("text") or str(e)
    return str(e)


# ---------------- deterministic rules ----------------

def rule_type(step_text):
    return "REASON" if REASON_PAT.search(step_text.strip()) else "RETRIEVE"

def rule_answer_type(step_text, typ):
    s = step_text.lower()
    if typ == "REASON":
        return "BOOLEAN"
    if s.startswith("when") or "what year" in s or "date" in s:
        return "DATE"
    if s.startswith("where") or "located" in s or "what country" in s:
        return "LOCATION"
    if s.startswith("how many") or "how much" in s or "how long" in s or "how old" in s:
        return "NUMBER"
    if s.startswith("who"):
        return "ENTITY"
    return "SHORT_TEXT"


# ---------------- LLM prompt ----------------

def build_prompt(row, paras):
    steps_txt = []
    for s in row["typed_plan"]:
        cand_blocks = []
        for pid in s.get("evidence_candidates", []):
            t = para_text(paras, pid)
            if t:
                cand_blocks.append(f'    [{pid}] {t[:PARA_TRUNC]}')
        cb = "\n".join(cand_blocks) if cand_blocks else "    (no candidates)"
        steps_txt.append(f'STEP {s["id"]}: {s["question_en"]}\n  CANDIDATE EVIDENCE:\n{cb}')
    steps_block = "\n\n".join(steps_txt)

    term = row.get("term") or ""
    page_titles = [p.get("title") for p in row.get("evidence_pages", []) if p.get("title")]
    titles_block = ", ".join(page_titles) if page_titles else "(none)"
    return f"""You are annotating a multi-hop reasoning question for a dataset. Return ONLY a JSON object, no prose, no markdown fences.

URDU QUESTION: {row.get('question_ur')}
ENGLISH QUESTION: {row.get('question_en')}
GOLD TERM ENTITY: {term}
CANDIDATE PAGE TITLES (Wikipedia pages relevant to this question): {titles_block}

{steps_block}

TASK — produce this exact JSON structure:
{{
  "question_entities": [
    {{"canonical_title": "<Wikipedia page title of an entity NAMED IN THE QUESTION>",
      "urdu_span": "<the exact Urdu words for this entity, copied verbatim from the Urdu question>"}}
  ],
  "typed_plan": [
    {{"id": <step id>,
      "type": "RETRIEVE" or "REASON",
      "entity_ref": "<canonical_title of the entity this step looks up a fact about; for a step that says #N, use the entity step N's answer resolved to (not step N's own entity_ref); empty string for REASON steps>",
      "expected_answer_type": one of {ANSWER_TYPES},
      "evidence_ref": ["<only the candidate paragraph IDs that TRULY answer this step>"],
      "gold_intermediate_answer": "<short answer read from the kept evidence, or null>"}}
  ]
}}

RULES:
- question_entities: For EACH title in CANDIDATE PAGE TITLES, decide YES/NO: is this entity referred to in the question (by its name, a different word-form like "body builder" for Bodybuilding, or an Urdu word)? Include EVERY title you answer YES for. Most questions name 2 or more entities — outputting only one is usually wrong. The gold term entity is always included. Add a non-listed entity only if clearly named in the question.
- urdu_span must be copied verbatim from the Urdu question. If you cannot find it, use empty string.
- type: RETRIEVE = looks up an external fact, INCLUDING bridge lookups about a previous answer (e.g. "Where was #1 born?" or age/date lookups). REASON = compares, computes, decides using earlier answers.
- entity_ref: the entity whose fact THIS step looks up. If the step text names an entity directly, use that. If the step refers to "#N", use the entity that step N's answer RESOLVED TO (the fact step N returned), NOT step N's own entity_ref. Follow the chain. REASON steps use empty string.
- evidence_ref: keep a candidate ONLY if its text actually answers the step. Drop topically-related but non-answering paragraphs. REASON steps usually keep [].
- gold_intermediate_answer: only if stated in kept evidence; else null.
- Output the JSON object only."""

def parse_llm_json(text):
    t = text.strip()
    t = re.sub(r"^```(json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------------- scoring helpers ----------------

def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace("_", " ")).strip().casefold()

def prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    r = tp / len(gold_set) if gold_set else 1.0
    return p, r


def main():
    cands = load_jsonl(CANDS)
    gold  = {g["urbench_qid"]: g for g in load_jsonl(GOLD)}
    with open(PARAS, encoding="utf-8") as f:
        paras = json.load(f)

    assert len(cands) == 30, f"expected 30 candidate rows, got {len(cands)}"
    assert len(gold) == 30, f"expected 30 gold rows, got {len(gold)}"
    if len(paras) != 9251:
        print(f"!! WARNING: paragraph file has {len(paras)} entries, expected 9251")

    # sanity: every evidence candidate id resolvable
    missing = 0
    for r in cands:
        for s in r["typed_plan"]:
            for pid in s.get("evidence_candidates", []):
                if para_text(paras, pid) is None:
                    missing += 1
    if missing:
        print(f"!! WARNING: {missing} evidence candidate IDs not found in paragraph file")

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2000, stop=["<|im_end|>"])

    prompts = []
    for r in cands:
        messages = [{"role": "user", "content": build_prompt(r, paras)}]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    outputs = llm.generate(prompts, sp)

    # scoring accumulators
    S = {k: [0,0] for k in ["ent_p","ent_r","span","type_llm","type_rule",
                             "eref","ev_p","ev_r","atype_llm","atype_rule"]}
    def add(key, val):
        S[key][0] += val; S[key][1] += 1

    preds, diffs = [], []
    rows_full_auto, review_rows = 0, []

    for r, out in zip(cands, outputs):
        qid = r["urbench_qid"]
        g = gold[qid]
        gen = out.outputs[0].text.strip()
        pj = parse_llm_json(gen)

        row_diffs = {"qid": qid, "fields": []}
        needs_review = []

        if pj is None:
            preds.append({"qid": qid, "llm_parse_failed": True, "raw": gen[:2000]})
            review_rows.append({"qid": qid, "reasons": ["LLM JSON parse failure"]})
            diffs.append({"qid": qid, "fields": [{"field":"ALL","note":"parse failure"}]})
            continue

        # ---- entities: precision/recall on canonical titles ----
        gold_ents = {norm(e["canonical_title"]): e for e in g.get("question_entities", [])}
        pred_ents = {norm(e.get("canonical_title","")): e for e in pj.get("question_entities", [])
                     if e.get("canonical_title")}
        p, rr = prf(set(pred_ents), set(gold_ents))
        add("ent_p", p); add("ent_r", rr)
        if p < 1 or rr < 1:
            row_diffs["fields"].append({"field":"entities",
                "pred": sorted(pred_ents), "gold": sorted(gold_ents)})
            needs_review.append("entity mismatch")

        # ---- urdu spans (on entities present in both) ----
        for k in set(pred_ents) & set(gold_ents):
            ok = norm(pred_ents[k].get("urdu_span")) == norm(gold_ents[k].get("urdu_span"))
            add("span", 1 if ok else 0)
            if not ok:
                row_diffs["fields"].append({"field":"urdu_span","entity":k,
                    "pred": pred_ents[k].get("urdu_span"), "gold": gold_ents[k].get("urdu_span")})
                needs_review.append(f"urdu_span:{k}")

        # ---- per-step fields ----
        gold_steps = {s["id"]: s for s in g.get("typed_plan", [])}
        pred_steps = {s.get("id"): s for s in pj.get("typed_plan", [])}
        for s in r["typed_plan"]:
            sid = s["id"]
            gs = gold_steps.get(sid)
            if gs is None:
                continue
            ps = pred_steps.get(sid, {})

            # type: LLM and rule vs gold
            ok_l = norm(ps.get("type")) == norm(gs.get("type"))
            add("type_llm", 1 if ok_l else 0)
            ok_r = norm(rule_type(s["question_en"])) == norm(gs.get("type"))
            add("type_rule", 1 if ok_r else 0)
            if not ok_l:
                row_diffs["fields"].append({"field":"type","step":sid,
                    "pred": ps.get("type"), "gold": gs.get("type")})
                needs_review.append(f"type:step{sid}")

            # entity_ref
            ok = norm(ps.get("entity_ref")) == norm(gs.get("entity_ref"))
            add("eref", 1 if ok else 0)
            if not ok:
                row_diffs["fields"].append({"field":"entity_ref","step":sid,
                    "pred": ps.get("entity_ref"), "gold": gs.get("entity_ref")})
                needs_review.append(f"entity_ref:step{sid}")

            # evidence P/R
            pv = set(ps.get("evidence_ref") or [])
            gv = set(gs.get("evidence_ref") or [])
            ep, er = prf(pv, gv)
            add("ev_p", ep); add("ev_r", er)
            if pv != gv:
                row_diffs["fields"].append({"field":"evidence_ref","step":sid,
                    "pred": sorted(pv), "gold": sorted(gv)})
                needs_review.append(f"evidence:step{sid}")

            # expected_answer_type: LLM and rule vs gold
            ok_al = norm(ps.get("expected_answer_type")) == norm(gs.get("expected_answer_type"))
            add("atype_llm", 1 if ok_al else 0)
            rt = rule_answer_type(s["question_en"], gs.get("type",""))
            add("atype_rule", 1 if norm(rt) == norm(gs.get("expected_answer_type")) else 0)
            if not ok_al:
                row_diffs["fields"].append({"field":"expected_answer_type","step":sid,
                    "pred": ps.get("expected_answer_type"), "gold": gs.get("expected_answer_type")})

        preds.append({"qid": qid, "llm": pj})
        if row_diffs["fields"]:
            diffs.append(row_diffs)
            review_rows.append({"qid": qid, "reasons": sorted(set(needs_review))})
        else:
            rows_full_auto += 1

    # ---- write outputs ----
    with open(OUT_P, "w", encoding="utf-8") as f:
        for p_ in preds: f.write(json.dumps(p_, ensure_ascii=False) + "\n")
    with open(OUT_D, "w", encoding="utf-8") as f:
        for d in diffs: f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pct(key):
        tot = S[key][1]
        return f"{100*S[key][0]/tot:.1f}% (n={tot})" if tot else "n/a"

    lines = [
        "STAGE 2 PILOT — accuracy vs gold (audit30_answers.jsonl)",
        "=" * 60,
        f"entity precision        : {pct('ent_p')}",
        f"entity recall           : {pct('ent_r')}",
        f"urdu_span accuracy      : {pct('span')}",
        f"step type  — LLM        : {pct('type_llm')}",
        f"step type  — rule       : {pct('type_rule')}",
        f"entity_ref accuracy     : {pct('eref')}",
        f"evidence precision      : {pct('ev_p')}",
        f"evidence recall         : {pct('ev_r')}",
        f"answer type — LLM       : {pct('atype_llm')}",
        f"answer type — rule      : {pct('atype_rule')}",
        f"rows fully auto (0 diff): {rows_full_auto} / 30",
        f"rows needing review     : {len(review_rows)} / 30",
        "",
        "REVIEW ROWS + REASONS:",
    ]
    for rr_ in review_rows:
        lines.append(f"  {rr_['qid']}: {', '.join(rr_['reasons'][:8])}")
    summary = "\n".join(lines)
    with open(OUT_S, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)
    print("\npredictions:", OUT_P)
    print("diffs      :", OUT_D)
    print("summary    :", OUT_S)


if __name__ == "__main__":
    main()