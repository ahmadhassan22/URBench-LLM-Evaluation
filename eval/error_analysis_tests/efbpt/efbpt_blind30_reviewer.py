"""
efbpt_blind30_reviewer.py — v2. CPU-only interactive reviewer for BLIND30.

Changes vs v1 (after the accidental-accept incident):
  - NOTHING saves without typing 'ok' at the end of the row. Enter never saves.
  - 'edit N' or 'edit <qid>' reopens an already-saved row to fix it.
  - Evidence paragraphs shown in FULL; 'e <pid>' re-prints any paragraph.
  - Adding an entity (+Title) REQUIRES an Urdu span (type '-' to leave empty
    deliberately).
  - Review file rewritten atomically on every save (safe for edits).
This is real annotation work, same standard as AUDIT30. Do it in small sessions.
"""
import json, os, sys, tempfile
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
CANDS = EF / "blind30_candidates.jsonl"
PREDS = EF / "blind30_predictions.jsonl"
PARAS = BASE / "data/strategyqa_official/strategyqa_train_paragraphs.json"
OUT_R = EF / "blind30_review.jsonl"
OUT_G = EF / "blind30_gold.jsonl"

def load_jsonl(p):
    if not Path(p).exists(): return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def para_str(paras, pid):
    e = paras.get(pid)
    if e is None: return "(missing)"
    if isinstance(e, dict): return e.get("content") or e.get("text") or str(e)
    return str(e)

def save_all(reviews):
    """Atomic rewrite: temp file then replace, so a crash never corrupts."""
    fd, tmp = tempfile.mkstemp(dir=str(EF), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in reviews.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, str(OUT_R))

def review_row(qid, c, p, paras, existing=None):
    """Returns the reviewed record, or None if user quit/discarded."""
    print("=" * 70)
    print(f"ROW {qid}" + ("   [EDITING SAVED ROW]" if existing else ""))
    print(f"EN: {c['question_en']}")
    print(f"UR: {c['question_ur']}")
    print(f"ANSWER: {c['answer']}   TERM: {c['term']}")

    if existing:
        ents  = [e["canonical_title"] for e in existing["question_entities"]]
        spans = {e["canonical_title"]: e.get("urdu_span","") for e in existing["question_entities"]}
        base_steps = {s["id"]: s for s in existing["typed_plan"]}
    else:
        ents  = [e["canonical_title"] for e in p.get("question_entities", [])]
        spans = {e["canonical_title"]: e.get("urdu_span","") for e in p.get("question_entities", [])}
        base_steps = {}

    extras = p.get("extra_entities") or []
    print(f"\n  ENTITIES: {ents}")
    print(f"  spans   : { {k: v for k, v in spans.items()} }")
    if extras: print(f"  !! MODEL-PROPOSED EXTRAS (verify): {[e.get('canonical_title') for e in extras]}")
    print(f"  ALL TITLES: {[d.get('title') for d in p.get('title_decisions', [])]}")
    while True:
        v = input("  entities (+Add / -Remove / span Title=... / Enter=done / q): ").strip()
        if v == "q": return "QUIT"
        if not v: break
        if v.startswith("+"):
            t = v[1:].strip()
            if t and t not in ents:
                sp = input(f"    Urdu span for '{t}' (verbatim from UR question; '-' = deliberately empty): ").strip()
                if sp == "": print("    span required — use '-' if truly none"); continue
                ents.append(t); spans[t] = "" if sp == "-" else sp
        elif v.startswith("-"):
            t = v[1:].strip(); ents = [e for e in ents if e != t]; spans.pop(t, None)
        elif v.startswith("span "):
            body = v[5:]
            if "=" in body:
                t, sp = body.split("=", 1); t = t.strip()
                if t in ents: spans[t] = sp.strip()
                else: print(f"    '{t}' not in entities")
        print(f"    now: {ents}")
    final_ents = [{"canonical_title": e, "urdu_span": spans.get(e, "")} for e in ents]

    psteps = {s.get("id"): s for s in (p.get("typed_plan") or [])}
    gold_steps = []
    for s in c["typed_plan"]:
        sid = s["id"]
        src = base_steps.get(sid) or psteps.get(sid, {})
        print(f"\n  STEP {sid}: {s['question_en']}")
        for pid in s.get("evidence_candidates", []):
            print(f"      [{pid}]")
            print("        " + para_str(paras, pid).replace("\n", " "))
        step = {"id": sid,
                "type": src.get("type", ""),
                "entity_ref": src.get("entity_ref", ""),
                "expected_answer_type": src.get("expected_answer_type", ""),
                "evidence_ref": list(src.get("evidence_ref") or []),
                "gold_intermediate_answer": src.get("gold_intermediate_answer")}
        while True:
            line = (f"type={step['type']} | ref={step['entity_ref']!r} | "
                    f"at={step['expected_answer_type']} | ev={step['evidence_ref']} | "
                    f"gia={step['gold_intermediate_answer']!r}")
            v = input(f"    [{line}]\n    fix (ref=/type=/at=/ev=/gia=/e <pid>) or Enter=done: ").strip()
            if v == "q": return "QUIT"
            if not v: break
            if v.startswith("e "):
                print("        " + para_str(paras, v[2:].strip()))
            elif v.startswith("ref="):  step["entity_ref"] = v[4:]
            elif v.startswith("type="): step["type"] = v[5:].upper()
            elif v.startswith("at="):   step["expected_answer_type"] = v[3:].upper()
            elif v.startswith("ev="):   step["evidence_ref"] = [x.strip() for x in v[3:].split(",") if x.strip()]
            elif v.startswith("gia="):  g = v[4:]; step["gold_intermediate_answer"] = None if g == "null" else g
            else: print("      unknown command")
        gold_steps.append(step)

    note = input("\n  row note (Enter=none): ").strip()
    print(f"\n  FINAL: entities={[e['canonical_title'] for e in final_ents]}")
    for st in gold_steps:
        print(f"    step {st['id']}: type={st['type']} ref={st['entity_ref']!r} "
              f"at={st['expected_answer_type']} ev={st['evidence_ref']} gia={st['gold_intermediate_answer']!r}")
    while True:
        conf = input("  type 'ok' to SAVE this row, 'redo' to restart it, 'skip' to discard, 'q' to quit without saving: ").strip()
        if conf == "ok":
            return {"qid": qid, "question_entities": final_ents,
                    "typed_plan": gold_steps, "note": note}
        if conf == "redo": return review_row(qid, c, p, paras, existing)
        if conf == "skip": return None
        if conf == "q": return "QUIT"

def main():
    cands = {r["urbench_qid"]: r for r in load_jsonl(CANDS)}
    preds = {r["qid"]: r for r in load_jsonl(PREDS)}
    with open(PARAS, encoding="utf-8") as f:
        paras = json.load(f)
    reviews = {r["qid"]: r for r in load_jsonl(OUT_R)}
    order = list(cands.keys())
    print(f"reviewed {len(reviews)}/30.")
    print("commands: Enter=next unreviewed | edit N | edit <qid> | list | q\n")

    while True:
        todo = [q for q in order if q not in reviews]
        cmd = input(f"[{len(reviews)}/30 done, {len(todo)} left] > ").strip()
        if cmd == "q": break
        if cmd == "list":
            for i, q in enumerate(order, 1):
                print(f"  {i:2d}. {q}  {'DONE' if q in reviews else ''}")
            continue
        if cmd.startswith("edit"):
            arg = cmd[4:].strip()
            qid = order[int(arg)-1] if arg.isdigit() else arg
            if qid not in cands: print("  unknown row"); continue
            res = review_row(qid, cands[qid], preds.get(qid, {}), paras, reviews.get(qid))
        else:
            if not todo:
                print("all 30 reviewed"); break
            qid = todo[0]
            res = review_row(qid, cands[qid], preds.get(qid, {}), paras)
        if res == "QUIT": break
        if res:
            reviews[res["qid"]] = res
            save_all(reviews)
            print(f"  SAVED ({len(reviews)}/30)")

    if len(reviews) == 30:
        with open(OUT_G, "w", encoding="utf-8") as f:
            for q in order:
                r = reviews[q]
                f.write(json.dumps({"urbench_qid": r["qid"],
                                    "question_entities": r["question_entities"],
                                    "typed_plan": r["typed_plan"],
                                    "note": r.get("note", "")}, ensure_ascii=False) + "\n")
        print(f"\nALL 30 REVIEWED — gold written: {OUT_G}")

if __name__ == "__main__":
    main()