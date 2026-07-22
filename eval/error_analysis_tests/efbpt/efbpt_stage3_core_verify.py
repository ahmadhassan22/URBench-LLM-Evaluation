"""
efbpt_stage3_core_verify.py — GPU. Independent Schema C verifier + agreement routing.

GOLD ISOLATION: this script reads ONLY blind30_candidates.jsonl and
blind30_predictions.jsonl. There is NO code path to blind30_gold.jsonl,
blind30_review*.jsonl, or any scorer output. (Assertion below enforces it.)

Verifier independently re-decides every candidate-universe title and produces
Schema C core fields only (no evidence, no GIA). Routing ACCEPTs a row only if
deterministic rules pass AND the original schema-v2 core prediction and the
verifier agree exactly (frozen normalization). Disagreement = REJECT, never merge.
"""
import json, re, sys
from pathlib import Path
from vllm import LLM, SamplingParams

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
CANDS = EF / "blind30_candidates.jsonl"
PREDS = EF / "blind30_predictions.jsonl"
OUT_V = EF / "blind30_core_verified.jsonl"
OUT_S = EF / "blind30_core_routing_summary.txt"

# GOLD ISOLATION assertion: no forbidden path may appear in this file's source.
_SRC = Path(__file__).read_text(encoding="utf-8")
for _bad in ("blind30_gold", "blind30_review", "score_diffs"):
    assert _SRC.count(_bad) <= 2, f"gold-isolation violated: {_bad} referenced"

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
ANSWER_TYPES = ["BOOLEAN","ENTITY","LOCATION","DATE","NUMBER","SET","SHORT_TEXT"]

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm(s):
    s = re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_", " ")).strip().casefold()
    return s.strip(" .,:;\"'")

def build_universe(c, p):
    """GENERAL sources only: term, evidence pages, title_decisions,
    extra_entities, original entity_ref values. Deduped by frozen norm."""
    uni = {}   # norm -> display title (first seen)
    def add(t):
        if not t: return
        n = norm(t)
        if n and n not in uni: uni[n] = re.sub(r"\s+", " ", str(t).replace("_"," ")).strip()
    add(c.get("term"))
    for ep in c.get("evidence_pages", []): add(ep.get("title"))
    for d in p.get("title_decisions", []): add(d.get("title"))
    for e in p.get("extra_entities", []):  add(e.get("canonical_title"))
    for s in (p.get("typed_plan") or []):  add(s.get("entity_ref"))
    return uni

def build_prompt(c, uni):
    titles_block = "\n".join(f"- {t}" for t in uni.values())
    steps_block = "\n".join(f'STEP {s["id"]}: {s["question_en"]}' for s in c["typed_plan"])
    return f"""You are independently verifying the annotation of a multi-hop reasoning question. Return ONLY a JSON object, no prose, no markdown fences.

URDU QUESTION: {c.get('question_ur')}
ENGLISH QUESTION: {c.get('question_en')}
CANDIDATE TITLES:
{titles_block}

{steps_block}

TASK — produce this exact JSON structure:
{{
  "title_decisions": [
    {{"title": "<copy one CANDIDATE TITLE exactly>",
      "referred_in_question": true or false,
      "urdu_span": "<exact Urdu words for this entity copied verbatim from the Urdu question (a descriptive phrase or pronoun phrase is allowed if the question refers to the entity that way), or empty string>"}}
  ],
  "typed_plan": [
    {{"id": <step id>,
      "type": "RETRIEVE" or "REASON",
      "entity_ref": "<one CANDIDATE TITLE copied exactly, or empty string for REASON steps>",
      "expected_answer_type": one of {ANSWER_TYPES}}}
  ]
}}

RULES:
- title_decisions: EXACTLY ONE decision per candidate title, in order. referred_in_question is true if the question refers to the entity by name, a different word-form, a descriptive phrase, or an Urdu word. Most questions refer to 2 or more titles.
- urdu_span: verbatim substring of the Urdu question; required (non-empty) whenever referred_in_question is true.
- type: RETRIEVE = looks up an external fact, INCLUDING bridge lookups about a previous answer. REASON = compares, computes, decides using earlier answers.
- entity_ref: the entity whose fact THIS step looks up, chosen ONLY from CANDIDATE TITLES (copy exactly; no free-form names). If the step refers to "#N", use the entity that step N's answer RESOLVED TO, not step N's own entity_ref. REASON steps use empty string.
- Output the JSON object only."""

def parse_json(text):
    t = re.sub(r"^```(json)?", "", text.strip()).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def core_from_prediction(p):
    ents = {norm(e["canonical_title"]): e.get("urdu_span","")
            for e in p.get("question_entities", []) if e.get("canonical_title")}
    # Schema-v2: extra_entities are unverified claims that must be checked here,
    # not silently dropped. Fold them into the original core so the agreement
    # comparison sees them. If the verifier independently confirms one (same
    # title marked true), the sets match and it joins the accepted core; if not,
    # the sets differ -> disagree_entity_set -> the row is rejected.
    for e in (p.get("extra_entities") or []):
        t = e.get("canonical_title")
        if t and norm(t) not in ents:
            ents[norm(t)] = e.get("urdu_span","")
    steps = {s.get("id"): {"type": norm(s.get("type")),
                           "ref": norm(s.get("entity_ref")),
                           "at": norm(s.get("expected_answer_type"))}
             for s in (p.get("typed_plan") or [])}
    return ents, steps

def core_from_verifier(vj, uni):
    ents = {}
    for d in (vj.get("title_decisions") or []):
        if d.get("referred_in_question") is True:
            n = norm(d.get("title",""))
            if n in uni: ents[n] = d.get("urdu_span","")
    steps = {s.get("id"): {"type": norm(s.get("type")),
                           "ref": norm(s.get("entity_ref")),
                           "at": norm(s.get("expected_answer_type"))}
             for s in (vj.get("typed_plan") or [])}
    return ents, steps

def main():
    if OUT_V.exists():
        print(f"OUTPUT EXISTS: {OUT_V}. Refusing to overwrite."); sys.exit(0)

    cands = load_jsonl(CANDS)
    preds = {r["qid"]: r for r in load_jsonl(PREDS)}
    assert len(cands) == 30

    universes = {c["urbench_qid"]: build_universe(c, preds[c["urbench_qid"]]) for c in cands}

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2000, stop=["<|im_end|>"])
    prompts = [tok.apply_chat_template(
        [{"role":"user","content": build_prompt(c, universes[c["urbench_qid"]])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in cands]
    outputs = llm.generate(prompts, sp)

    recs, n_acc = [], 0
    reason_counts = {}
    for c, out in zip(cands, outputs):
        qid = c["urbench_qid"]; p = preds[qid]; uni = universes[qid]
        ur_q = norm(c.get("question_ur"))
        vj = parse_json(out.outputs[0].text.strip())
        reasons = []

        if vj is None:
            reasons.append("verifier_parse_failure")
            v_ents, v_steps = {}, {}
        else:
            # parser completeness on title decisions
            seen = [norm(d.get("title","")) for d in (vj.get("title_decisions") or [])]
            for n in uni:
                cnt = seen.count(n)
                if cnt == 0: reasons.append(f"title_missing:{uni[n]}")
                elif cnt > 1: reasons.append(f"title_duplicate:{uni[n]}")
            for n in seen:
                if n not in uni: reasons.append(f"title_unknown:{n}")
            v_ents, v_steps = core_from_verifier(vj, uni)

        o_ents, o_steps = core_from_prediction(p)
        ids = [s["id"] for s in c["typed_plan"]]

        # deterministic rules — applied to BOTH original and verifier cores
        for tag, ents, steps in (("orig", o_ents, o_steps), ("verif", v_ents, v_steps)):
            if sorted(steps.keys()) != sorted(ids):
                reasons.append(f"{tag}_step_ids_mismatch")
                continue
            for n, span in ents.items():
                if not norm(span): reasons.append(f"{tag}_empty_span:{n}")
                elif norm(span) not in ur_q: reasons.append(f"{tag}_span_not_substring:{n}")
            for sid in ids:
                st = steps[sid]
                if st["type"] not in ("retrieve","reason"): reasons.append(f"{tag}_bad_type:step{sid}")
                if st["type"] == "reason" and st["ref"]: reasons.append(f"{tag}_reason_nonempty_ref:step{sid}")
                if st["type"] == "retrieve":
                    if not st["ref"]: reasons.append(f"{tag}_retrieve_empty_ref:step{sid}")
                    elif st["ref"] not in uni: reasons.append(f"{tag}_ref_not_in_universe:step{sid}")
                if st["at"] not in {norm(a) for a in ANSWER_TYPES}: reasons.append(f"{tag}_bad_atype:step{sid}")

        # agreement (only meaningful if no failures so far)
        if not reasons:
            if set(o_ents) != set(v_ents): reasons.append("disagree_entity_set")
            else:
                for n in o_ents:
                    if norm(o_ents[n]) != norm(v_ents[n]): reasons.append(f"disagree_span:{n}")
            for sid in ids:
                for f in ("type","ref","at"):
                    if o_steps[sid][f] != v_steps[sid][f]:
                        reasons.append(f"disagree_{f}:step{sid}")

        status = "ACCEPT" if not reasons else "REJECT"
        if status == "ACCEPT": n_acc += 1
        for r in reasons:
            reason_counts[r.split(":")[0]] = reason_counts.get(r.split(":")[0], 0) + 1
        recs.append({"qid": qid, "status": status, "reasons": sorted(set(reasons)),
                     "original_core": {"entities": o_ents, "steps": o_steps},
                     "verifier_core": {"entities": v_ents, "steps": v_steps}})

    with open(OUT_V, "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    L = ["STAGE 3 CORE VERIFIER — agreement routing on BLIND30 (dev data)",
         "=" * 62,
         f"ACCEPTED : {n_acc}/30 = {100*n_acc/30:.1f}%",
         f"REJECTED : {30-n_acc}/30",
         f"reject/failure reason counts: {reason_counts}",
         "", "REJECTED ROWS:"]
    for r in recs:
        if r["status"] == "REJECT":
            L.append(f"  {r['qid'][:8]}: {', '.join(r['reasons'][:6])}")
    summary = "\n".join(L)
    OUT_S.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\nverified:", OUT_V, "\nsummary:", OUT_S)

if __name__ == "__main__":
    main()