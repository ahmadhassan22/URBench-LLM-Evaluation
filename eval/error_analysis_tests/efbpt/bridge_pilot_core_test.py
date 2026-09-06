#!/usr/bin/env python3
"""Synthetic tests only. No real corpus/model/input reads or output files.

Run: python -B eval/error_analysis_tests/efbpt/bridge_pilot_core_test.py
"""
import sys
sys.dont_write_bytecode = True
import copy
import itertools
import math
import unittest
from bridge_pilot_core import *

def parent(text="  اردو aaa London is a city.\nParis is another city. ", qid=None):
    p = {"raw_page_id": "synthetic", "raw_url": "https://example.invalid/parent",
         "raw_title": "Synthetic Parent", "raw_text": text, "raw_text_sha256": sha256(text),
         "locations": [{"shard_index": 0, "blob_path": "/synthetic/unused", "blob_sha256": "0"*64,
                        "row_group": 0, "row_in_group": 0, "row_in_shard": 0}],
         "lookup_decision": "SYNTHETIC", "normalized_bucket_rows": 1, "normalized_bucket_distinct_pages": 1}
    return {"qid": qid or sorted(CHILD_COUNTS)[0], "question_ur": "یہ کیا ہے؟",
            "parent_source_instance_id": "synthetic-parent", "parent_title": "Synthetic Parent",
            "parent_normalized_title": "synthetic parent", "page": p, "chunks": local_chunks(p)}

def fixture():
    targets, predictions = [], []
    for qid, m in sorted(CHILD_COUNTS.items()):
        children = [{"source_instance_id": f"{qid}-{j}", "title": f"Child {qid} {j}",
                     "normalized_title": f"child {qid} {j}"} for j in range(m)]
        targets.append({"qid": qid, "children": children})
        for arm in ARMS:
            candidates = [{"global_row": j, "score": float(100-j),
                           "title": children[j]["title"] if arm == "D" and j < m else f"Distractor {j}"}
                          for j in range(100)]
            predictions.append({"qid": qid, "arm": arm, "query": "synthetic query",
                                "candidates": candidates, "ranked_titles": aggregate_candidates(candidates),
                                "boundary_tie_within_returned": False, "ties_beyond_budget": "UNOBSERVED_NO_OVERFETCH"})
    return predictions, targets

class CoreTests(unittest.TestCase):
    def test_prompt_bytes(self):
        self.assertEqual(assert_prompt_hashes(), PROMPT_HASHES)

    def test_parent_schema_rejects_injected_target(self):
        r = parent(); r["child_title"] = "leak"
        with self.assertRaises(PilotError): validate_parent(r)

    def test_chunks_preserve_unicode_and_tail(self):
        r = parent(" ".join("لفظ" for _ in range(200)))
        self.assertEqual(len(r["chunks"]), 1)
        # Independent legacy algorithm has the extra 50-word tail.
        self.assertEqual(len([r["page"]["raw_text"].split()[i:i+200] for i in range(0,200,150)]), 2)
        r = parent(" ".join("word" for _ in range(1983)))
        self.assertEqual(len(r["chunks"]), 13)

    def test_prompt_allowlists(self):
        r = parent()
        for arm in ("A", "P", "B"):
            ms = parent_query_messages(r, arm)
            payload = strict_json(ms[1]["content"].split("\nReturn")[0])
            self.assertEqual(set(payload), {"question_ur", "known_source_title", "available_evidence"})
            self.assertEqual(payload["known_source_title"], "" if arm == "A" else r["parent_title"])
            self.assertEqual(payload["available_evidence"], r["page"]["raw_text"] if arm == "B" else "")
        for arm in ("C", "D"):
            payload = strict_json(state_messages(r, arm)[1]["content"].split("\n\n")[0])
            self.assertEqual(set(payload), {"question_ur", "parent_title", "chunks"})
            self.assertEqual(set(payload["chunks"][0]), {"chunk_id", "text"})

    def test_root_rejections(self):
        for raw in ('{"facts":[],"facts":[]}', '{"facts":[],"extra":0}', '{"facts":NaN}',
                    '{"facts":[1e999]}', '{"facts":[{"quote":"a","quote":"b"}]}',
                    '{"facts":{}}', '```json\n{}\n```', '[]'):
            with self.subTest(raw=raw):
                self.assertTrue(parse_state(raw, parent(), "D")["format_failure"])

    def test_unicode_offsets_and_overlapping_occurrences(self):
        r = parent(); c = r["chunks"][0]
        s = parse_state(canonical({"facts": [{"chunk_id": c["chunk_id"], "quote": "aa"}]}), r, "D")
        item = s["items"][0]
        self.assertEqual(item["quote_occurrences"], 2)
        self.assertEqual(item["quote_char_start"], r["page"]["raw_text"].index("aa"))
        self.assertEqual(r["page"]["raw_text"][item["quote_char_start"]:item["quote_char_end"]], "aa")

    def test_item_rejects_do_not_consume_allowance(self):
        r = parent(" ".join(f"w{i}" for i in range(20))); cid = r["chunks"][0]["chunk_id"]
        items = [{"chunk_id": cid, "quote": "unsupported"}]
        items += [{"chunk_id": cid, "quote": f"w{i}"} for i in range(8)]
        items += [{"chunk_id": cid, "quote": "w0"}]
        s = parse_state(canonical({"facts": items}), r, "D")
        self.assertEqual((len(s["items"]),s["invalid_items"],s["duplicate_items"],s["excess_items"]), (6,1,1,2))

    def test_c_support_and_unknown_fields(self):
        r = parent(); cid = r["chunks"][0]["chunk_id"]
        items = [{"chunk_id": cid, "quote": "London is a city.", "text": "London"},
                 {"chunk_id": cid, "quote": "London is a city.", "text": "Paris"},
                 {"chunk_id": cid, "quote": "London is a city.", "text": "London", "why": "extra"}]
        s = parse_state(canonical({"entities": items}), r, "C")
        self.assertEqual(len(s["items"]), 1); self.assertEqual(s["invalid_items"], 2)

    def test_empty_state_retains_parent(self):
        r = parent()
        for arm, raw in (("C", '{"entities":[]}'), ("D", "invalid")):
            self.assertEqual(parent_query_messages(r, arm, parse_state(raw,r,arm)), parent_query_messages(r,"P"))

    def test_state_tampering_stops(self):
        r = parent(); s = parse_state('{"facts":[]}',r,"D")
        s["items"] = [{"quote": "injected"}]
        with self.assertRaises(PilotError): parent_query_messages(r,"D",s)

    def test_cap_and_empty_fallback(self):
        def tokens(text, **kw):
            self.assertEqual(kw, {"add_special_tokens": True,"truncation": False})
            return {"input_ids": list(range(2+5*len(text.split())))}
        r = cap_query(" ".join(["word"]*40),"اردو","A",tokens)
        self.assertEqual((len(r["query"].split()),r["word_cap_removed"],r["encoder_cap_removed"]), (25,8,7))
        self.assertEqual(cap_query(" ","اردو","A",tokens)["fallback"],"URDU_QUESTION")
        self.assertEqual(cap_query(" ","اردو","D",tokens,"fixed A")["query"],"fixed A")
        with self.assertRaises(PilotError): cap_query(" ","اردو","D",tokens)
        def too_long(text, **kw): return {"input_ids": list(range(200))}
        with self.assertRaises(PilotError): cap_query("x","اردو","A",too_long)
        def crash(text, **kw): raise RuntimeError("operational")
        with self.assertRaises(RuntimeError): cap_query("x","اردو","A",crash)

    def test_context_failure(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kw):
                assert kw == dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
                return "rendered"
            def __call__(self, text, **kw):
                assert kw == dict(add_special_tokens=False,truncation=False)
                return {"input_ids": [1]*40000}
        with self.assertRaises(PilotError): render_prompt(Tokenizer(),[],1024)
        self.assertEqual(render_prompt(Tokenizer(),[],128)["input_tokens"],40000)

    def test_all_ids_checked_before_lookup(self):
        called = []
        def lookup(i): called.append(i); return {"title":"x","text":"x"}
        for ids,scores in (([0,-1],[1.,0.]), ([0,2],[1.,0.]), ([0,1.5],[1.,0.]),
                           ([0,1],[1.,math.nan]), ([0,0],[1.,0.]), ([0,True],[1.,0.])):
            with self.assertRaises(PilotError): search_to_metadata(ids,scores,lookup,ntotal=2,budget=2)
        self.assertEqual(called,[])

    def test_title_ties_and_short_list(self):
        cs = [{"global_row":4,"score":1.,"title":" B "},
              {"global_row":3,"score":1.,"title":"a"},
              {"global_row":2,"score":1.,"title":"B"},
              {"global_row":1,"score":.5,"title":"a"}]
        ranked = aggregate_candidates(cs)
        self.assertEqual([r["normalized_title"] for r in ranked],["a","b"])
        self.assertEqual([r["best_global_row"] for r in ranked],[3,2])
        self.assertEqual(len(ranked),2)

    def test_signflip_independent_enumeration(self):
        for weights in ([],[0]*25,[12]*6,[12]*7+[-12],[3,-4,6,12,0],[-3,-4,6]):
            nz = [abs(w) for w in weights if w]
            vals = [sum(s*w for s,w in zip(signs,nz)) for signs in itertools.product((-1,1),repeat=len(nz))]
            p = sum(abs(v)>=abs(sum(weights)) for v in vals)/len(vals)
            self.assertEqual(signflip(weights)["p_two_sided"],p)
        self.assertEqual(signflip([12]*6)["p_two_sided"],.03125)

    def test_partial_or_duplicate_predictions_fail(self):
        ps,ts = fixture()
        with self.assertRaises(PilotError): validate_predictions(ps[:-1])
        ps[-1] = copy.deepcopy(ps[0])
        with self.assertRaises(PilotError): validate_predictions(ps)
        ts[0]["children"] = []
        with self.assertRaises(PilotError): validate_targets(ts)

    def test_corrupted_ranking_fails(self):
        ps,_ = fixture(); ps[0]["ranked_titles"].reverse()
        with self.assertRaises(PilotError): validate_predictions(ps)

    def test_all_arms_macro_micro_and_fixed_sequence(self):
        ps,ts = fixture()
        result = summarize(ps,ts)
        self.assertEqual(result["arms"]["D"]["10"]["qid_macro_recall"],1.)
        self.assertEqual(result["comparisons"]["D-A"]["paired_qid_bootstrap_95ci_pp"],[100.,100.])
        self.assertEqual(result["decision"],"PROMISING_ASSISTED_PILOT")
        self.assertEqual(result["arms"]["D"]["1"]["pair_micro_recall"],25/36)
        self.assertAlmostEqual(result["arms"]["D"]["1"]["qid_macro_recall"], (17+6/2+1/3+1/4)/25)
        for row in ps:
            if row["arm"] == "A":
                ref = next(r for r in ps if r["qid"]==row["qid"] and r["arm"]=="D")
                row["candidates"] = copy.deepcopy(ref["candidates"])
                row["ranked_titles"] = copy.deepcopy(ref["ranked_titles"])
        result = summarize(ps,ts)
        self.assertFalse(result["primary_gate_passed"])
        self.assertEqual(result["comparisons"]["D-A"]["p_two_sided"],1.)
        self.assertEqual(result["comparisons"]["D-P"]["inference"],"EXPLORATORY_PRIMARY_GATE_NOT_PASSED")

if __name__ == "__main__":
    unittest.main(verbosity=2)
