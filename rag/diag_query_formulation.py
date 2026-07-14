"""
THROWAWAY DIAGNOSTIC (delete after). Decides step-2 direction.

Question: for an entity+attribute query, is the fact-bearing chunk MISSING from
the candidate set (candidate-generation problem -> need entity/decomposed queries),
or PRESENT but ranked low (ranking problem -> a reranker fixes it)?

Test: retrieve top-20 three ways for the grey-seal case and report where (if
anywhere) a grey-seal chunk appears.
"""
from retrieve import Retriever   # run from rag/ dir, or adjust sys.path

r = Retriever(device="cuda")

variants = {
    "full_question":   "How fast is a grey seal's reaction speed?",
    "entity_only":     "grey seal",
    "entity_attribute":"grey seal reaction time reflexes",
}

for name, q in variants.items():
    hits = r.retrieve(q, top_k=20)[0]
    print("\n" + "=" * 72)
    print(f"VARIANT: {name}   |   query: {q!r}")
    seal_ranks = []
    for rank, h in enumerate(hits, 1):
        blob = (h["title"] + " " + h["text"]).lower()
        is_seal = "seal" in blob
        if is_seal:
            seal_ranks.append(rank)
        # print only seal hits + the top-3 for context
        if is_seal or rank <= 3:
            tag = "  <-- SEAL" if is_seal else ""
            print(f"  #{rank:2d}  {h['score']:.4f}  [{h['title']}]{tag}")
    if seal_ranks:
        print(f"  >> seal chunk(s) found at rank(s): {seal_ranks}  "
              f"(reranker CAN help)")
    else:
        print(f"  >> NO seal chunk in top-20 "
              f"(candidate-generation problem — reranker canNOT help)")