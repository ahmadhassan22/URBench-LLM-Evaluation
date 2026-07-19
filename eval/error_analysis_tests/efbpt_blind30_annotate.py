"""
efbpt_blind30_annotate.py — GPU. Annotates the frozen BLIND30 with schema v2.

Schema v2 changes vs the pilot (parser-enforced, per your design):
  - The LLM must output ONE decision object per CANDIDATE PAGE TITLE:
      {"title": ..., "referred_in_question": true/false, "urdu_span": ...}
    It can no longer silently skip titles.
  - Parser verifies every supplied title appears EXACTLY once; missing or
    duplicated -> row routed to review with reason.
  - question_entities is built MECHANICALLY from decisions marked true.
  - Model-proposed entities outside the list go to extra_entities, always
    flagged for verification, never auto-accepted.
Prompt rules otherwise FROZEN from run 56070. No scoring (no gold exists).
Output feeds the human reviewer, which creates the gold.
"""
import json, re, sys
from pathlib import Path
from vllm import LLM, SamplingParams

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
CANDS = EF / "blind30_candidates.jsonl"
PARAS = BASE / "data/strategyqa_official/strategyqa_train_paragraphs.json"
OUT_P = EF / "blind30_predictions.jsonl"
OUT_S = EF / "blind30_annotate_summary.txt"

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
PARA_TRUNC = 1200
ANSWER_TYPES = ["BOOLEAN","ENTITY","LOCATION","DATE","NUMBER","SET","SHORT_TEXT"]

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def para_text(paras, pid):
    e = paras.get(pid)
    if e is None: return None
    if isinstance(e, dict): return e.get("content") or e.get("text") or str(e)
    return str(e)

def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace("_", " ")).strip().casefold()

def build_prompt(row, paras):
    steps_txt = []
    for s in row["typed_plan"]:
        cand_blocks = []
        for pid in s.get("evidence_candidates", []):
            t = para_text(paras, pid)
            if t: cand_blocks.append(f'    [{pid}] {t[:PARA_TRUNC]}')
        cb = "\n".join(cand_blocks) if cand_blocks else "    (no candidates)"
        steps_txt.append(f'STEP {s["id"]}: {s["question_en"]}\n  CANDIDATE EVIDENCE:\n{cb}')
    steps_block = "\n\n".join(steps_txt)
    term = row.get("term") or ""
    page_titles = [p.get("title") for p in row.get("evidence_pages", []) if p.get("title")]
    titles_block = "\n".join(f"- {t}" for t in page_titles) if page_titles else "(none)"
    return f"""You are annotating a multi-hop reasoning question for a dataset. Return ONLY a JSON object, no prose, no markdown fences.

URDU QUESTION: {row.get('question_ur')}
ENGLISH QUESTION: {row.get('question_en')}
GOLD TERM ENTITY: {term}
CANDIDATE PAGE TITLES:
{titles_block}

{steps_block}

TASK — produce this exact JSON structure:
{{
  "title_decisions": [
    {{"title": "<copy one CANDIDATE PAGE TITLE exactly>",
      "referred_in_question": true or false,
      "urdu_span": "<exact Urdu words for this entity copied verbatim from the Urdu question, or empty string>"}}
  ],
  "extra_entities": [
    {{"canonical_title": "<Wikipedia page title of an entity clearly named in the question but NOT in the candidate list>",
      "urdu_span": "<exact Urdu words, or empty string>"}}
  ],
  "typed_plan": [
    {{"id": <step id>,
      "type": "RETRIEVE" or "REASON",
      "entity_ref": "<canonical_title of the entity this step looks up a fact about; for a step that says #N, use the entity step N's answer resolved to (not step N's own entity_ref); empty string for REASON steps>",
      "expected_answer_type": one of {ANSWER_TYPES},
      "evidence_ref": ["<only the candidate paragraph IDs that TRULY answer this step>"],
      "gold_intermediate_answer": "<short answer read from the kept evidence, or null (JSON null, not the string 'null')>"}}
  ]
}}

RULES:
- title_decisions: output EXACTLY ONE decision for EVERY candidate title, in order. referred_in_question is true if the question refers to the entity by its name, a different word-form (e.g. "body builder" for Bodybuilding), or an Urdu word for it. Most questions refer to 2 or more titles.
- urdu_span must be copied verbatim from the Urdu question; empty string if not present.
- extra_entities: only entities clearly named in the question but missing from the candidate list; usually this list is empty.
- type: RETRIEVE = looks up an external fact, INCLUDING bridge lookups about a previous answer (e.g. "Where was #1 born?" or age/date lookups). REASON = compares, computes, decides using earlier answers.
- entity_ref: the entity whose fact THIS step looks up. If the step text names an entity directly, use that. If the step refers to "#N", use the entity that step N's answer RESOLVED TO (the fact step N returned), NOT step N's own entity_ref. Follow the chain. REASON steps use empty string.
- evidence_ref: keep a candidate ONLY if its text actually answers the step. Drop topically-related but non-answering paragraphs. REASON steps usually keep [].
- gold_intermediate_answer: a SHORT answer (a name, date, number, or one short phrase), not a copied paragraph. JSON null if not stated in kept evidence.
- Output the JSON object only."""

def parse_llm_json(text):
    t = text.strip()
    t = re.sub(r"^```(json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def main():
    if OUT_P.exists():
        print(f"OUTPUT EXISTS: {OUT_P}. Refusing to overwrite. Exiting.")
        sys.exit(0)

    cands = load_jsonl(CANDS)
    assert len(cands) == 30, f"expected 30 candidates, got {len(cands)}"
    with open(PARAS, encoding="utf-8") as f:
        paras = json.load(f)

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

    preds, n_review, reasons_count = [], 0, {}
    for r, out in zip(cands, outputs):
        qid = r["urbench_qid"]
        gen = out.outputs[0].text.strip()
        pj = parse_llm_json(gen)
        review = []

        supplied = [p["title"] for p in r.get("evidence_pages", []) if p.get("title")]
        supplied_norm = [norm(t) for t in supplied]

        if pj is None:
            preds.append({"qid": qid, "llm_parse_failed": True, "raw": gen[:2000],
                          "review_reasons": ["parse_failure"]})
            n_review += 1
            reasons_count["parse_failure"] = reasons_count.get("parse_failure", 0) + 1
            continue

        # ---- parser-enforced completeness on title_decisions ----
        decisions = pj.get("title_decisions") or []
        seen = [norm(d.get("title", "")) for d in decisions]
        for t, tn in zip(supplied, supplied_norm):
            c = seen.count(tn)
            if c == 0: review.append(f"title_missing:{t}")
            elif c > 1: review.append(f"title_duplicate:{t}")
        for d, dn in zip(decisions, seen):
            if dn not in supplied_norm:
                review.append(f"title_unknown:{d.get('title')}")

        # ---- mechanical question_entities from decisions marked true ----
        q_entities = [{"canonical_title": d.get("title"),
                       "urdu_span": d.get("urdu_span", "")}
                      for d in decisions
                      if d.get("referred_in_question") is True
                      and norm(d.get("title","")) in supplied_norm]

        # ---- extra entities: always flagged ----
        extras = pj.get("extra_entities") or []
        if extras: review.append(f"extra_entities:{len(extras)}")

        # ---- routing per frozen run-56070 decision: these fields need review ----
        review.append("check:entity_recall")   # human confirms nothing missed
        review.append("check:entity_ref")
        review.append("check:evidence_ref")
        review.append("check:answer_type")
        review.append("check:gia")

        preds.append({"qid": qid,
                      "question_entities": q_entities,
                      "title_decisions": decisions,
                      "extra_entities": extras,
                      "typed_plan": pj.get("typed_plan"),
                      "review_reasons": sorted(set(review))})
        n_review += 1
        for rr in review:
            key = rr.split(":")[0]
            reasons_count[key] = reasons_count.get(key, 0) + 1

    with open(OUT_P, "w", encoding="utf-8") as f:
        for p_ in preds: f.write(json.dumps(p_, ensure_ascii=False) + "\n")

    n_dec = sum(len(p.get("title_decisions") or []) for p in preds)
    n_true = sum(sum(1 for d in (p.get("title_decisions") or [])
                     if d.get("referred_in_question") is True) for p in preds)
    struct_fail = sum(1 for p in preds if any(
        r.startswith(("title_missing","title_duplicate","title_unknown","parse_failure"))
        for r in p.get("review_reasons", [])))
    lines = [
        "BLIND30 ANNOTATION — schema v2 (no gold; feeds human reviewer)",
        "=" * 60,
        f"rows annotated              : {len(preds)} / 30",
        f"structural failures         : {struct_fail} (parse/missing/dup/unknown titles)",
        f"title decisions emitted     : {n_dec}",
        f"decisions marked true       : {n_true}",
        f"rows with extra_entities    : {sum(1 for p in preds if p.get('extra_entities'))}",
        f"reason counts               : {reasons_count}",
    ]
    summary = "\n".join(lines)
    OUT_S.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\npredictions:", OUT_P, "\nsummary:", OUT_S)

if __name__ == "__main__":
    main()