"""
efbpt_blind30_gold_audit_packet.py — CPU-only, read-only.
Builds a BLIND gold-audit packet: for each of the 30 rows shows only
QID, EN question, UR question, term, evidence-page titles, current gold
entities. NO predictions, verifier, routing, or scorer data — so the
re-check is not anchored by pipeline output. Modifies nothing.
"""
import json
from pathlib import Path

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
CANDS = EF / "blind30_candidates.jsonl"
GOLD  = EF / "blind30_gold.jsonl"
OUT   = EF / "blind30_gold_audit_packet.txt"

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    cands = {r["urbench_qid"]: r for r in load_jsonl(CANDS)}
    gold  = {r["urbench_qid"]: r for r in load_jsonl(GOLD)}
    order = list(cands.keys())

    L = []
    L.append("BLIND30 GOLD-AUDIT PACKET (gold only; no pipeline output shown)")
    L.append("=" * 66)
    L.append("Policy reminder (frozen Schema-v2): a title is a question_entity if")
    L.append("the question refers to the entity by its name, a different word-form,")
    L.append("a descriptive phrase, or an Urdu word for it. Generic concepts")
    L.append("(e.g. Monogamy, Sonnet, Hand) ARE entities under this rule. The term")
    L.append("entity is (almost) always present. Each accepted entity needs an")
    L.append("Urdu span that occurs verbatim in the Urdu question.")
    L.append("")
    L.append("CHECK EACH ROW:")
    L.append("  [ ] Is the TERM entity present in gold? (usually yes)")
    L.append("  [ ] Does the question name any OTHER entity (incl. generic concept)")
    L.append("      that is missing from gold?")
    L.append("  [ ] Is any listed gold entity NOT actually referred to? (over-include)")
    L.append("  [ ] Empty gold entities: is that truly correct, or an omission?")
    L.append("")
    L.append("=" * 66)

    for i, q in enumerate(order, 1):
        c, g = cands[q], gold[q]
        ents = [e["canonical_title"] for e in g.get("question_entities", [])]
        ev   = [e.get("title") for e in c.get("evidence_pages", []) if e.get("title")]
        L.append(f"\nROW {i:02d}  {q}")
        L.append(f"  EN   : {c.get('question_en')}")
        L.append(f"  UR   : {c.get('question_ur')}")
        L.append(f"  TERM : {c.get('term')}")
        L.append(f"  EVIDENCE PAGES : {ev}")
        L.append(f"  CURRENT GOLD ENTITIES : {ents}"
                 + ("   <-- EMPTY" if not ents else ""))
        L.append(f"  verdict (OK / ADD:<title> / REMOVE:<title>): ____")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\npacket:", OUT)

if __name__ == "__main__":
    main()