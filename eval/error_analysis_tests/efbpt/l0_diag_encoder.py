#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l0_diag_encoder.py — WHY did L0 Part C fail?

Diagnostic only. Not a declared experiment, produces no gate, changes no
reading, and does not change the completed L0 gate or result. It investigates
why final L0 LINKABLE recall@10 was 11.85% by separating two explanations:

  (a) the pipeline is broken             -> English title queries would ALSO fail
  (b) Urdu spans align poorly with this
      encoder/title representation       -> English works, Urdu performs poorly

Method: build a SMALL universe (all gold titles found in the corpus, plus
N_FILLER random corpus titles) and query it twice — once with the English
gold title itself, once with the Urdu span. If English self-retrieval is not
near-perfect at rank 1, the fault is ours, not the encoder's.

READ-ONLY. Writes nothing.
"""
import json, os, random, re, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from d3_oracle_retrieval import META_PATH, norm          # noqa: E402
from l0_part_a_census import GOLD, classify, translit    # noqa: E402

EMBED_PATH = ("/mnt/home/user41/downloaded_models/sentence-transformers/"
              "paraphrase-multilingual-MiniLM-L12-v2")
TITLE_RE = re.compile(rb'^\{"title":\s*"((?:[^"\\]|\\.)*)"')
N_FILLER = 5000
SEED = 4242


def main():
    pairs, seen = [], set()
    for path, idf, entf in GOLD:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for e in row.get(entf, []):
                sp, ti = e.get("urdu_span", ""), e.get("canonical_title", "")
                if not ti.strip() or (row[idf], sp) in seen:
                    continue
                seen.add((row[idf], sp))
                b, _ = classify(sp, ti)
                pairs.append({"span": sp, "title": ti, "bucket": b})
    want = {norm(p["title"]) for p in pairs}
    print("gold pairs: %d, distinct normalized titles: %d" % (len(pairs), len(want)))

    rng = random.Random(SEED)
    found, filler, seen_titles, n_lines = {}, [], set(), 0
    eligible_seen = 0
    print("[scan] one pass over the corpus for titles ...", flush=True)
    with open(META_PATH, "rb") as f:
        for raw in f:
            n_lines += 1
            m = TITLE_RE.match(raw)
            if not m:
                continue
            t = m.group(1).decode("utf-8", "replace").replace('\\"', '"')
            nt = norm(t)
            if not nt or nt in seen_titles:
                continue
            seen_titles.add(nt)
            if nt in want:
                found[nt] = t
                continue

            eligible_seen += 1
            if len(filler) < N_FILLER:
                filler.append(t)
            else:
                j = rng.randrange(eligible_seen)
                if j < N_FILLER:
                    filler[j] = t
    print("[scan] corpus lines scanned: %d" % n_lines)
    print("[scan] UNIQUE normalized corpus titles seen: %d" % len(seen_titles))
    print("[scan] gold titles located: %d/%d" % (len(found), len(want)))
    print("[scan] filler count: %d" % len(filler))

    titles = list(found.values()) + filler
    tnorm = [norm(t) for t in titles]
    assert len(tnorm) == len(set(tnorm)), "diagnostic title universe is not unique"
    idx_of = {n: i for i, n in enumerate(tnorm)}
    print("final UNIQUE universe size: %d titles" % len(titles))

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(EMBED_PATH)
    tv = emb.encode(titles, normalize_embeddings=True, convert_to_numpy=True,
                    batch_size=256).astype(np.float32)

    usable = [p for p in pairs if norm(p["title"]) in idx_of]
    print("number of usable pair instances: %d" % len(usable))

    def ranks(queries):
        qv = emb.encode(queries, normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=128).astype(np.float32)
        S = qv @ tv.T
        order = np.argsort(-S, axis=1)
        return order

    eng_order = ranks([p["title"] for p in usable])
    urd_order = ranks([p["span"] for p in usable])

    def rank_of(order_row, gold_i):
        pos = np.where(order_row == gold_i)[0]
        return int(pos[0]) + 1 if len(pos) else 10 ** 9

    rows = []
    for i, p in enumerate(usable):
        gi = idx_of[norm(p["title"])]
        rows.append((p, rank_of(eng_order[i], gi), rank_of(urd_order[i], gi),
                     urd_order[i][:5]))

    def hit(rs, k):
        return 100.0 * sum(1 for r in rs if r <= k) / max(len(rs), 1)

    print("\n" + "=" * 74)
    print("SELF-RETRIEVAL CHECK — query = the ENGLISH gold title itself")
    print("=" * 74)
    er = [r[1] for r in rows]
    for k in (1, 5, 10):
        print("  recall@%-3d %6.1f%%" % (k, hit(er, k)))
    print("  -> near-100% recall@1 validates basic title extraction, encoding,"
          " and scoring plumbing")

    print("\n" + "=" * 74)
    print("URDU SPAN QUERIES on the SAME small universe")
    print("=" * 74)
    for name, sel in (("ALL", lambda p: True),
                      ("TRANSLIT", lambda p: p["bucket"] == "TRANSLIT"),
                      ("SEMANTIC", lambda p: p["bucket"] == "SEMANTIC")):
        sub = [r[2] for r in rows if sel(r[0])]
        if not sub:
            continue
        print("  %-9s n=%3d   @1 %5.1f%%   @5 %5.1f%%   @10 %5.1f%%"
              % (name, len(sub), hit(sub, 1), hit(sub, 5), hit(sub, 10)))

    print("\n" + "=" * 74)
    print("WHAT THE ENCODER ACTUALLY RETRIEVES for 15 Urdu spans")
    print("=" * 74)
    for p, er_, ur_, top in rows[:15]:
        print("  span %-22s gold %-26s eng_rank=%-4d urdu_rank=%s"
              % (translit(p["span"])[:22], p["title"][:26], er_,
                 ur_ if ur_ < 10 ** 9 else "miss"))
        print("      top5: %s" % " | ".join(titles[j][:22] for j in top))
    print("\nINTERPRETATION")
    print("  English self-retrieval validates basic extraction/encoding/scoring plumbing.")
    print("  Poor Urdu-span retrieval would show poor span-to-title alignment for this")
    print("  encoder/title representation; it would not show that the encoder generally")
    print("  cannot handle Urdu.")
    print("\n=== done. no files written. ===")


if __name__ == "__main__":
    main()
