#!/usr/bin/env python3
"""Offline tests for efbpt_qea_triage_v1. No model, GPU, retrieval, metadata scan or job.

Synthetic fixtures live in temporary directories. Historical DEV200 records are read only
(never modified) to regression-test the candidate join and the acceptance rule.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import efbpt_qea_triage_v1 as t  # noqa: E402

ROOT = Path("/mnt/home/user41/URBench")
HIST = ROOT / "outputs/efbpt/stage0_assisted"
STAGE0 = ROOT / "data/strategyqa_official/efbpt/stage0"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jl(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def qid(n):
    return f"{n:020x}"


def sid(q, title):
    return "s0-" + sha(json.dumps([q, title], ensure_ascii=False, separators=(",", ":")).encode())


# ------------------------------------------------------------------ synthetic fixture
# Question plans: (titles, explicit titles, {child: parent} CLEAR_DEPENDENCY, ambiguous titles, absent titles)
PLANS = [
    (["Alpha One", "Beta One", "Gamma One"], {"Alpha One"}, {"Beta One": "Alpha One"}, {"Gamma One"}, set()),
    (["Alpha Two", "Beta Two", "Gamma Two"], {"Alpha Two"}, {"Beta Two": "Alpha Two", "Gamma Two": "Alpha Two"}, set(), set()),
    (["Alpha Three", "Delta Three", "Beta Three", "Gamma Three"], {"Alpha Three", "Delta Three"},
     {"Beta Three": "Alpha Three", "Gamma Three": "Delta Three"}, set(), set()),
    (["Alpha Four", "Beta Four"], set(), {}, set(), set()),
    (["Alpha Five", "Beta Five"], {"Alpha Five"}, {"Beta Five": "Alpha Five"}, set(), {"Beta Five"}),
]
QIDS = [qid(0xA0 + i) for i in range(len(PLANS))]
RESERVE = [qid(0xF0 + i) for i in range(3)]


def build_fixture(base, corpus_status_in_master=False):
    base = Path(base)
    prep, v1 = base / "prep", base / "v1"
    master, links, rows, runtime = [], [], [], []
    for q, (titles, _e, _c, _a, _abs) in zip(QIDS, PLANS):
        question = "کیا " + " ".join(titles) + " سے متعلق ہے؟"
        decomposition = [f"Step {i} about {title}?" for i, title in enumerate(titles)]
        rows.append({"urbench_qid": q, "official_qid": "o" + q[:6], "question_ur": question,
                     "official_decomposition": decomposition, "official_evidence": [], "evidence_paragraph_ids": [],
                     "annotation_scope": "QEA_TRIAGE_ANNOTATION_INPUT_NOT_QEA_RUNTIME"})
        runtime.append({"urbench_qid": q, "question_ur": question, "runtime_scope": "QEA_RUNTIME_INPUT_QUESTION_ONLY"})
        for i, title in enumerate(titles):
            s = sid(q, title)
            master.append({"source_instance_id": s, "urbench_qid": q, "gold_title": title, "question_ur": question,
                           "normalized_gold_title": title.lower(),
                           "exact_corpus_status": "EXACT_PRESENT" if corpus_status_in_master else None})
            for annotator in range(2):
                links.append({"record_type": "PARAGRAPH", "source_instance_id": s, "urbench_qid": q,
                              "official_evidence_annotator_index": annotator, "decomposition_step_index": i,
                              "decomposition_text": decomposition[i], "evidence_group_index": 0,
                              "paragraph_occurrence_index": 0, "marker": None, "paragraph_id": f"{title}-1",
                              "paragraph_title": title, "paragraph_index": 1, "section": "", "headers": [],
                              "raw_path": [annotator, i, 0, 0]})
    ordered = QIDS + RESERVE
    files = {
        "selection/dev_triage_qids.txt": "".join(q + "\n" for q in QIDS),
        "selection/reserve_qids.txt": "".join(q + "\n" for q in RESERVE),
        "selection/eligible_ordered.jsonl": jl([{"rank": i + 1, "urbench_qid": q} for i, q in enumerate(ordered)]),
        "annotation_inputs/source_instance_master.jsonl": jl(master),
        "annotation_inputs/official_evidence_links.jsonl": jl(links),
        "annotation_inputs/dev_triage_rows.jsonl": jl(rows),
        "qea_runtime_inputs/dev_questions_ur.jsonl": jl(runtime),
    }
    ident = {}
    for rel, text in files.items():
        (prep / rel).parent.mkdir(parents=True, exist_ok=True)
        (prep / rel).write_text(text, encoding="utf-8")
        ident[rel] = {"sha256": sha(text.encode()), "bytes": len(text.encode())}
    manifest = json.dumps({"files": ident}, sort_keys=True).encode()
    (prep / "PREPARATION_MANIFEST.json").write_bytes(manifest)
    (v1 / "selection").mkdir(parents=True)
    v1_manifest = b'{"run":1}'
    (v1 / "PREPARATION_MANIFEST.json").write_bytes(v1_manifest)
    (v1 / "selection/dev_triage_qids.txt").write_text(files["selection/dev_triage_qids.txt"])
    meta_lines = []
    for titles, _e, _c, _a, absent in PLANS:
        for title in titles:
            if title not in absent:
                meta_lines.append(json.dumps({"title": title.replace(" ", "_").upper(), "text": "x"}))
    meta_lines.append(json.dumps({"title": "Unrelated", "text": "y"}))
    meta = ("\n".join(meta_lines) + "\n").encode()
    (base / "meta.jsonl").write_bytes(meta)
    cfg = t.Config(prep_dir=prep, prep_manifest_sha256=sha(manifest), v1_dir=v1, v1_manifest_sha256=sha(v1_manifest),
                   runtime_input_sha256=ident["qea_runtime_inputs/dev_questions_ur.jsonl"]["sha256"],
                   dev_count=len(QIDS), reserve_count=len(RESERVE), triage_root=base / "triage",
                   metadata=base / "meta.jsonl", metadata_sha256=sha(meta), metadata_bytes=len(meta),
                   sample_cap=20, sample_min=2)
    return cfg


def split_messages(m):
    """(system, user dict) from either chat format; Gemma-2 has no system role (system + blank line + user)."""
    if len(m) == 2:
        return m[0]["content"], json.loads(m[1]["content"])
    for system in (t.PASS2_SYSTEM, t.PASS3_SYSTEM, t.REVIEW_SYSTEM, t.REVIEW_SYSTEM_V2, t.IDENT_SYSTEM):
        if m[0]["content"].startswith(system + "\n\n"):
            return system, json.loads(m[0]["content"][len(system) + 2:])
    raise AssertionError("unknown message format")


def plan_for(title):
    for titles, explicit, children, ambiguous, _absent in PLANS:
        if title in titles:
            return explicit, children, ambiguous
    raise KeyError(title)


P2_NOT_YET = json.dumps({"decision": "NOT_YET_EXPLICIT", "explicit_relation_type": "", "urdu_span": "",
                         "confidence": "HIGH", "rationale": "Needs an intermediate step."})


def p3_obj(**kw):
    """A valid Pass-3 object in schema property order; defaults to an AMBIGUOUS record."""
    base = {"decision": "AMBIGUOUS", "concrete_intermediate_information": "", "official_step_indices": [0],
            "dependency_status": "NOT_APPLICABLE", "proposed_parent_source_titles": [],
            "dependency_confidence": "NOT_APPLICABLE", "confidence": "MEDIUM", "rationale": "Unclear."}
    base.update(kw)
    return base


class FakeBackend(t.Backend):
    """Deterministic canned responses keyed on the prompt; records every message and schema it receives.

    identity() reports the LLAMA_EXECUTION fields unless an explicit identity is given.
    pass2_override / pass3_override: title -> raw text (or a Pass-3 dict); finish_override:
    (kind, title) -> finish_reason.
    """
    key = "fake"

    def __init__(self, review_answers=None, pass2_override=None, pass3_override=None, finish_override=None,
                 identity=None, ident_override=None):
        self.seen = []
        self.schema_args = []
        self.calls = []            # (kind, user dict, schema or None, text) per request
        self.review_answers = review_answers or {}
        self.pass2_override = pass2_override or {}
        self.pass3_override = pass3_override or {}
        self.finish_override = finish_override or {}
        self.ident_override = ident_override or {}
        self._identity = identity

    def identity(self):
        return dict(t.llama_execution_expected(), family="fake") if self._identity is None else self._identity

    def generate(self, batch, max_new_tokens, schemas=None):
        self.schema_args.append(schemas)
        out = []
        for k, m in enumerate(batch):
            self.seen.append(m)
            system, user = split_messages(m)
            finish = "stop"
            if system == t.PASS2_SYSTEM:
                title = user["gold_title"]
                explicit, _c, _a = plan_for(title)
                text = self.pass2_override.get(title) or (json.dumps(
                    {"decision": "EXPLICIT", "explicit_relation_type": "TRANSLITERATION", "urdu_span": "",
                     "confidence": "HIGH", "rationale": "Named directly."}) if title in explicit else P2_NOT_YET)
                kind = "pass2"
            elif system == t.PASS3_SYSTEM:
                title = user["gold_title"]
                _e, children, _a = plan_for(title)
                steps = sorted({e["decomposition_step_index"] for e in user["target_official_evidence"]})
                kind = "pass3"
                if title in self.pass3_override:
                    o = self.pass3_override[title]
                    text = json.dumps(dict(o, official_step_indices=steps)) if isinstance(o, dict) else o
                elif title in children:
                    text = json.dumps({"decision": "LATENT_BRIDGE", "concrete_intermediate_information": f"Via {children[title]}.",
                                       "official_step_indices": steps, "dependency_status": "CLEAR_DEPENDENCY",
                                       "proposed_parent_source_titles": [children[title]], "dependency_confidence": "HIGH",
                                       "confidence": "HIGH", "rationale": "Clear dependency."})
                else:
                    text = json.dumps({"decision": "AMBIGUOUS", "concrete_intermediate_information": "",
                                       "official_step_indices": steps, "dependency_status": "NOT_APPLICABLE",
                                       "proposed_parent_source_titles": [], "dependency_confidence": "NOT_APPLICABLE",
                                       "confidence": "MEDIUM", "rationale": "Unclear."})
            elif system == t.IDENT_SYSTEM:
                title = user["title"]
                text = self.ident_override.get(title) or json.dumps({"answer": "Y", "note": "stub"})
                kind = "ident"
            elif system in (t.REVIEW_SYSTEM, t.REVIEW_SYSTEM_V2):
                key = (user["parent_title"], user["child_title"])
                q1, q2, q3, q4 = self.review_answers.get(key, ("Y", "Y", "Y", "C"))
                text = json.dumps({"q1": q1, "q2": q2, "q3": q3, "q4": q4, "note": "ok"})
                kind, title = "review", None
            else:
                raise AssertionError("unknown system prompt")
            finish = self.finish_override.get((kind, title), finish)
            self.calls.append((kind, user, None if schemas is None else schemas[k], text))
            out.append({"text": text, "finish_reason": finish, "prompt_tokens": 10, "output_tokens": 20})
        return out


def run_to_t4(cfg, backend=None):
    backend = backend or FakeBackend()
    t.t0_corpus_status(cfg, progress_every=0)
    t.t1_pass2(cfg, lambda: backend)
    t.t2_pass3(cfg, lambda: backend)
    t.t3_candidates(cfg)
    t.t4_packets(cfg)
    return backend


def write_r2(cfg, path, answers, omit=(), duplicate=(), malformed=()):
    packets = t.read_jsonl(cfg.stage("t4") / "review_packets.jsonl")
    lines = []
    for p in packets:
        key = (p["parent_title"], p["child_title"])
        if key in omit:
            continue
        q1, q2, q3, q4 = answers.get(key, ("Y", "Y", "Y", "C"))
        line = {"packet_id": p["packet_id"], "q1": q1, "q2": q2, "q3": q3, "q4": q4, "note": "r2"}
        if key in malformed:
            line["extra"] = 1
        lines.append(json.dumps(line))
        if key in duplicate:
            lines.append(json.dumps(line))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def guard_env(tmp):
    """Submission-style expected hashes for the real runner and tests and a stand-in executed wrapper."""
    wrapper = Path(tmp) / "slurm_script"
    wrapper.write_bytes(b"#!/bin/bash\n# stand-in for the Slurm spool copy\n")
    return {"QEA_EXPECT_RUNNER_SHA256": sha(Path(t.__file__).resolve().read_bytes()),
            "QEA_EXPECT_TEST_SHA256": sha(Path(__file__).resolve().read_bytes()),
            "QEA_EXPECT_WRAPPER_SHA256": sha(wrapper.read_bytes()),
            "QEA_EXECUTED_WRAPPER_PATH": str(wrapper)}


class Base(unittest.TestCase):
    """Fixture in a temporary directory; every stage runs under a passing code-identity guard."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = build_fixture(self.tmp.name)
        self.env = guard_env(self.tmp.name)
        self._env_patch = mock.patch.dict(os.environ, self.env)
        self._env_patch.start()
        t.READ_LOG.clear()

    def tearDown(self):
        self._env_patch.stop()
        self.tmp.cleanup()

    def raises(self, code, fn, *args, **kwargs):
        with self.assertRaises(t.TriageError) as ctx:
            fn(*args, **kwargs)
        self.assertTrue(str(ctx.exception).startswith(code), str(ctx.exception))


# ------------------------------------------------------------------ preparation and drift

class PreparationTests(Base):
    def test_real_r2_preparation_verifies(self):
        prep = t.verify_preparation(t.Config())
        self.assertEqual((len(prep["dev"]), len(prep["master"])), (160, 544))
        self.assertEqual(t.RUNTIME_INPUT_SHA256, "d45bfa71839d926eef605ec262f6d09b11c19579b48c3815f3fc7c5ceed6f960")

    def test_manifest_and_file_drift(self):
        t.verify_preparation(self.cfg)
        self.raises("PREP_MANIFEST_HASH_DRIFT", t.verify_preparation, replace(self.cfg, prep_manifest_sha256="0" * 64))
        path = self.cfg.prep_dir / "annotation_inputs/source_instance_master.jsonl"
        path.write_text(path.read_text() + "\n")
        self.raises("PREP_FILE_DRIFT", t.verify_preparation, self.cfg)

    def test_v1_to_r2_dev_change_detected(self):
        (self.cfg.v1_dir / "selection/dev_triage_qids.txt").write_text("\n".join(reversed(QIDS)) + "\n")
        self.raises("V1_TO_R2_DEV_CHANGED", t.verify_preparation, self.cfg)

    def test_runtime_input_hash_pinned(self):
        self.raises("RUNTIME_INPUT_HASH_MISMATCH", t.verify_preparation, replace(self.cfg, runtime_input_sha256="1" * 64))

    def test_premature_corpus_status_in_input(self):
        cfg = build_fixture(Path(self.tmp.name) / "b", corpus_status_in_master=True)
        self.raises("PREMATURE_CORPUS_STATUS_IN_INPUT", t.verify_preparation, cfg)

    def test_prompt_pins(self):
        self.assertEqual(t.PROMPT_SHA256, t.PINNED_PROMPT_SHA256)


# ------------------------------------------------------------------ parsing

P2_OK = {"decision": "EXPLICIT", "explicit_relation_type": "TRANSLITERATION", "urdu_span": "کیا",
         "confidence": "HIGH", "rationale": "Direct."}
P3_OK = {"decision": "LATENT_BRIDGE", "concrete_intermediate_information": "Via A.", "official_step_indices": [1],
         "dependency_status": "CLEAR_DEPENDENCY", "proposed_parent_source_titles": ["A"],
         "dependency_confidence": "HIGH", "confidence": "HIGH", "rationale": "Clear."}


def mod(base, **kw):
    d = dict(base)
    d.update(kw)
    return json.dumps(d)


class ParseTests(unittest.TestCase):
    def test_pass2_cases(self):
        good = json.dumps(P2_OK)
        cases = [(good, "stop", None), ("```json\n" + good + "\n```", "stop", None),
                 ("Answer: " + good, "stop", "NOT_JSON"), (good, "length", "FINISH_LENGTH"), ("", "stop", "NOT_JSON"),
                 ("[1]", "stop", "NOT_OBJECT"), (mod(P2_OK, extra=1), "stop", "KEY_SET"),
                 (mod(P2_OK, confidence="SURE"), "stop", "ENUM"), (mod(P2_OK, rationale=3), "stop", "FIELD_TYPE"),
                 (mod(P2_OK, explicit_relation_type=""), "stop", "EXPLICIT_RELATION"),
                 (mod(P2_OK, decision="NOT_YET_EXPLICIT"), "stop", "NOT_YET_EXPLICIT_WITH_RELATION_OR_SPAN"),
                 (mod(P2_OK, rationale=" "), "stop", "EMPTY_RATIONALE")]
        for text, finish, expected in cases:
            parsed, err = t.parse_pass2(text, finish, "کیا سوال")
            self.assertEqual(err, expected, text)
            self.assertEqual(parsed is None, expected is not None)
        parsed, _ = t.parse_pass2(mod(P2_OK, urdu_span="غائب"), "stop", "کیا سوال")
        self.assertFalse(parsed["urdu_span_found"])

    def test_pass3_cases(self):
        cands = ["A", "B"]
        cases = [(json.dumps(P3_OK), None),
                 (mod(P3_OK, proposed_parent_source_titles=[]), "PARENT_COUNT_FOR_STATUS"),
                 (mod(P3_OK, proposed_parent_source_titles=["Z"]), "PARENT_NOT_IN_CANDIDATES"),
                 (mod(P3_OK, proposed_parent_source_titles=["A", "A"]), "DUPLICATE_PARENT"),
                 (mod(P3_OK, proposed_parent_source_titles=["target_title"]), "SELF_PARENT"),
                 (mod(P3_OK, official_step_indices=[1, 1]), "DUPLICATE_STEP"),
                 (mod(P3_OK, dependency_status="MULTIPLE_PLAUSIBLE_PARENTS"), "PARENT_COUNT_FOR_STATUS"),
                 (mod(P3_OK, dependency_status="UNRESOLVED"), "PARENT_COUNT_FOR_STATUS"),
                 (mod(P3_OK, official_step_indices=[]), "STEP_RANGE"),
                 (mod(P3_OK, official_step_indices=[5]), "STEP_RANGE"),
                 (mod(P3_OK, official_step_indices=[True]), "STEP_TYPE"),
                 (mod(P3_OK, concrete_intermediate_information=""), "EMPTY_INTERMEDIATE"),
                 (mod(P3_OK, dependency_status="NOT_APPLICABLE"), "LATENT_DEPENDENCY_FIELDS"),
                 (mod(P3_OK, decision="AMBIGUOUS"), "AMBIGUOUS_FIELDS"),
                 (mod(P3_OK, decision="BRIDGE"), "ENUM"), (mod(P3_OK, extra=None), "KEY_SET")]
        for text, expected in cases:
            parsed, err = t.parse_pass3(text, "stop", 3, cands, "Target Title")
            self.assertEqual(err, expected, text)
        amb = mod(P3_OK, decision="AMBIGUOUS", concrete_intermediate_information="", dependency_status="NOT_APPLICABLE",
                  proposed_parent_source_titles=[], dependency_confidence="NOT_APPLICABLE")
        self.assertEqual(t.parse_pass3(amb, "stop", 3, cands, "Target Title")[1], None)
        self.assertEqual(t.parse_pass3(json.dumps(P3_OK), "prompt_too_long", 3, cands, "Target Title")[1],
                         "FINISH_PROMPT_TOO_LONG")

    def test_review_cases(self):
        ok = {"q1": "Y", "q2": "Y", "q3": "Y", "q4": "C", "note": "n"}
        self.assertIsNone(t.parse_review(json.dumps(ok), "stop")[1])
        for bad, code in ((dict(ok, q4="Y"), "ENUM"), (dict(ok, q1="y"), "ENUM"), ({k: ok[k] for k in ok if k != "note"}, "KEY_SET"),
                          (dict(ok, q2=1), "FIELD_TYPE")):
            self.assertEqual(t.parse_review(json.dumps(bad), "stop")[1], code)
        self.assertEqual(t.parse_review(json.dumps(ok), "length")[1], "FINISH_LENGTH")


# ------------------------------------------------------------------ stage isolation

class IsolationTests(Base):
    def test_pass2_prompt_carries_only_question_and_title(self):
        backend = FakeBackend()
        t.t1_pass2(self.cfg, lambda: backend)
        for m in backend.seen:
            self.assertEqual(m[0]["content"], t.PASS2_SYSTEM)
            self.assertEqual(set(json.loads(m[1]["content"])), {"gold_title", "question_ur"})
            for forbidden in ("decomposition", "exact_corpus_status", "EXACT_PRESENT", "question_en", "answer"):
                self.assertNotIn(forbidden, m[1]["content"])

    def test_prediction_stages_cannot_read_corpus_status(self):
        backend = FakeBackend()
        t.t0_corpus_status(self.cfg, progress_every=0)
        t0_file = self.cfg.stage("t0") / "corpus_status.jsonl"
        with t.stage_reads(self.cfg.prep_dir):
            self.raises("READ_OUTSIDE_STAGE_ALLOWLIST", t.read_bytes, t0_file)
        t.READ_LOG.clear()
        t.t1_pass2(self.cfg, lambda: backend)
        t.t2_pass3(self.cfg, lambda: backend)
        t0_dir = str(self.cfg.stage("t0").resolve())
        self.assertFalse([p for p in t.READ_LOG if p.startswith(t0_dir)])
        for stage in ("t1", "t2"):
            seal = json.loads((self.cfg.stage(stage) / "SEAL.json").read_text())
            self.assertIs(seal["corpus_status_visible"], False)
            self.assertFalse([p for p in seal["inputs_read"] if p.startswith(t0_dir)])
        for m in backend.seen:
            self.assertNotIn("EXACT_", m[-1]["content"])

    def test_t3_verifies_predictions_before_reading_corpus_status(self):
        run_to_t4(self.cfg)
        log = list(t.READ_LOG)
        # In the T3 run, every T1/T2 read precedes the first T0 read.
        t.READ_LOG.clear()
        cfg2 = replace(self.cfg, triage_root=Path(self.tmp.name) / "triage_b")
        for stage in ("t0", "t1", "t2"):
            import shutil
            shutil.copytree(self.cfg.stage(stage), cfg2.stage(stage))
        t.READ_LOG.clear()
        t.t3_candidates(cfg2)
        idx = lambda s: [i for i, p in enumerate(t.READ_LOG) if p.startswith(str(cfg2.stage(s).resolve()))]
        self.assertLess(max(idx("t1") + idx("t2")), min(idx("t0")))
        self.assertTrue(log)

    def test_t3_refuses_unsealed_or_tampered_predictions(self):
        backend = FakeBackend()
        t.t0_corpus_status(self.cfg, progress_every=0)
        t.t1_pass2(self.cfg, lambda: backend)
        self.raises("MISSING_OR_SYMLINK_INPUT", t.t3_candidates, self.cfg)       # no T2 seal
        t.t2_pass3(self.cfg, lambda: backend)
        p3 = self.cfg.stage("t2") / "pass3_records.jsonl"
        p3.write_text(p3.read_text().replace("CLEAR_DEPENDENCY", "UNRESOLVED", 1))
        t.READ_LOG.clear()
        self.raises("SEALED_OUTPUT_DRIFT", t.t3_candidates, self.cfg)
        self.assertFalse([p for p in t.READ_LOG if "/t0_corpus_status/" in p])

    def test_t3_refuses_prediction_seal_that_saw_corpus_status(self):
        backend = FakeBackend()
        t.t0_corpus_status(self.cfg, progress_every=0)
        t.t1_pass2(self.cfg, lambda: backend)
        t.t2_pass3(self.cfg, lambda: backend)
        seal_path = self.cfg.stage("t1") / "SEAL.json"
        seal = json.loads(seal_path.read_text())
        seal["corpus_status_visible"] = True
        seal_path.write_text(json.dumps(seal))
        self.raises("PREDICTION_STAGE_SAW_CORPUS_STATUS", t.t3_candidates, self.cfg)

    def test_packets_hide_verdicts_and_are_identical_for_reviewers(self):
        run_to_t4(self.cfg)
        packets = t.read_jsonl(self.cfg.stage("t4") / "review_packets.jsonl")
        self.assertTrue(packets)
        for p in packets:
            self.assertEqual(set(p), set(t.PACKET_FIELDS))
            blob = json.dumps(p)
            for token in ("EXPLICIT", "LATENT_BRIDGE", "CLEAR_DEPENDENCY", "EXACT_PRESENT", "HIGH", "confidence",
                          "rationale", "dependency_status", "Named directly", "Clear dependency"):
                self.assertNotIn(token, blob)
        out = Path(self.tmp.name) / "r2_packet.json"
        t.t4_render_r2(self.cfg, out)
        rendered = json.loads(out.read_text())
        self.assertEqual([x["user"] for x in rendered["packets"]], [json.loads(t.review_user(p)) for p in packets])
        self.assertEqual(rendered["instructions"], t.REVIEW_SYSTEM)

    def test_reviewers_cannot_read_each_other(self):
        run_to_t4(self.cfg)
        t.t4_review_r1(self.cfg, FakeBackend, "llama")
        r1_file = self.cfg.stage("r1") / "reviews.jsonl"
        with t.stage_reads(self.cfg.stage("t4")):
            self.raises("READ_OUTSIDE_STAGE_ALLOWLIST", t.read_bytes, r1_file)
        r2 = Path(self.tmp.name) / "r2.jsonl"
        write_r2(self.cfg, r2, {})
        t.READ_LOG.clear()
        t.t4_ingest_r2(self.cfg, r2, "test-reviewer")
        self.assertFalse([p for p in t.READ_LOG if "t4_review_R1" in p])


# ------------------------------------------------------------------ joins

def hist_inputs():
    master = t.read_jsonl(STAGE0 / "source_instance_master.jsonl")
    p2h = t.read_jsonl(HIST / "assisted_pass2.jsonl")
    p3h = t.read_jsonl(HIST / "assisted_pass3.jsonl")
    p2 = [{"source_instance_id": r["source_instance_id"], "urbench_qid": r["urbench_qid"],
           "gold_title": r["target_english_title"], "status": "OK", "parsed": {"decision": r["predicted_pass2"]}}
          for r in p2h]
    p3 = [{"source_instance_id": r["source_instance_id"], "urbench_qid": r["urbench_qid"],
           "gold_title": r["target_english_title"], "status": "OK",
           "parsed": {"decision": r["predicted_pass3"], "dependency_status": r["dependency_status"],
                      "proposed_parent_source_titles": r["proposed_parent_source_titles"],
                      "concrete_intermediate_information": r["concrete_intermediate_information"],
                      "official_step_indices": r["official_step_indices"]}} for r in p3h]
    corpus = [{"source_instance_id": r["source_instance_id"], "urbench_qid": r["urbench_qid"],
               "exact_corpus_status": r["exact_corpus_status"]} for r in master]
    return master, p2, p3, corpus


class JoinTests(Base):
    def test_historical_candidate_reproduction(self):
        master, p2, p3, corpus = hist_inputs()
        pairs = t.build_candidates(master, p2, p3, corpus)
        hist = {(c["urbench_qid"], p["parent_source_instance_id"], p["child_source_instance_id"])
                for c in t.read_jsonl(HIST / "bridge_candidate_qids.jsonl") for p in c["candidate_pairs"]}
        got = {(p["urbench_qid"], p["parent_source_instance_id"], p["child_source_instance_id"]) for p in pairs}
        self.assertEqual(got, hist)
        self.assertEqual((len(got), len({g[0] for g in got})), (41, 30))

    def test_join_guards(self):
        master, p2, p3, corpus = hist_inputs()
        q = master[0]["urbench_qid"]
        other_q_sid = next(r["source_instance_id"] for r in master if r["urbench_qid"] != q)
        bad = [dict(p2[0], urbench_qid="f" * 20)] + p2[1:]
        self.raises("CROSS_QUESTION_PASS2_RECORD", t.build_candidates, master, bad, p3, corpus)
        self.raises("DUPLICATE_PASS2_SID", t.build_candidates, master, p2 + [p2[0]], p3, corpus)
        self.raises("TITLE_MISMATCH_PASS2", t.build_candidates, master, [dict(p2[0], gold_title="Other")] + p2[1:], p3, corpus)
        self.raises("UNKNOWN_PASS3_SID", t.build_candidates, master, p2, p3 + [dict(p3[0], source_instance_id="s0-x")], corpus)
        self.raises("CORPUS_STATUS_INCOMPLETE", t.build_candidates, master, p2, p3, corpus[1:])
        self.assertTrue(other_q_sid)

    def test_no_cross_question_parent_and_no_self_parent(self):
        run_to_t4(self.cfg)
        pairs = t.read_jsonl(self.cfg.stage("t3") / "candidate_pairs.jsonl")
        by_sid = {r["source_instance_id"]: r["urbench_qid"]
                  for r in t.read_jsonl(self.cfg.prep_dir / "annotation_inputs/source_instance_master.jsonl")}
        for p in pairs:
            self.assertEqual(by_sid[p["parent_source_instance_id"]], by_sid[p["child_source_instance_id"]])
            self.assertNotEqual(p["parent_source_instance_id"], p["child_source_instance_id"])
        # A proposed parent title that exists only in another question never joins.
        master = t.read_jsonl(self.cfg.prep_dir / "annotation_inputs/source_instance_master.jsonl")
        p2 = t.read_jsonl(self.cfg.stage("t1") / "pass2_records.jsonl")
        p3 = t.read_jsonl(self.cfg.stage("t2") / "pass3_records.jsonl")
        corpus = t.read_jsonl(self.cfg.stage("t0") / "corpus_status.jsonl")
        child = next(r for r in p3 if r["gold_title"] == "Beta One")
        child["parsed"]["proposed_parent_source_titles"] = ["Alpha Two"]
        self.assertFalse([p for p in t.build_candidates(master, p2, p3, corpus) if p["child_title"] == "Beta One"])
        child["parsed"]["proposed_parent_source_titles"] = ["Beta One"]
        self.raises("PARENT_EQUALS_CHILD", t.build_candidates, master, p2, p3, corpus)

    def test_fixture_candidates(self):
        run_to_t4(self.cfg)
        pairs = {(p["parent_title"], p["child_title"]) for p in t.read_jsonl(self.cfg.stage("t3") / "candidate_pairs.jsonl")}
        self.assertEqual(pairs, {("Alpha One", "Beta One"), ("Alpha Two", "Beta Two"), ("Alpha Two", "Gamma Two"),
                                 ("Alpha Three", "Beta Three"), ("Delta Three", "Gamma Three")})
        # Q4 had no EXPLICIT title, so Pass 3 never ran for it; Q5's child is EXACT_ABSENT.
        p3_titles = {r["gold_title"] for r in t.read_jsonl(self.cfg.stage("t2") / "pass3_records.jsonl")}
        self.assertNotIn("Beta Four", p3_titles)
        self.assertIn("Beta Five", p3_titles)

    def test_pass3_scope_rule(self):
        recs = [{"urbench_qid": "a", "status": "OK", "parsed": {"decision": "EXPLICIT"}, "source_instance_id": "1"},
                {"urbench_qid": "a", "status": "OK", "parsed": {"decision": "NOT_YET_EXPLICIT"}, "source_instance_id": "2"},
                {"urbench_qid": "b", "status": "OK", "parsed": {"decision": "NOT_YET_EXPLICIT"}, "source_instance_id": "3"},
                {"urbench_qid": "b", "status": "MALFORMED", "parsed": None, "source_instance_id": "4"}]
        self.assertEqual([r["source_instance_id"] for r in t.pass3_scope(recs)], ["2"])


# ------------------------------------------------------------------ acceptance and selection

def ok(a):
    return {"status": "OK", "parsed": dict(zip(("q1", "q2", "q3", "q4"), a))}


class AcceptanceTests(Base):
    def test_decision_matrix(self):
        Y = ("Y", "Y", "Y", "C")
        cases = [(ok(Y), ok(Y), "ACCEPTED_AI_ASSISTED"), (ok(Y), ok(("Y", "Y", "N", "C")), "EXCLUDED_DISAGREEMENT"),
                 (ok(("N", "Y", "Y", "C")), ok(("Y", "Y", "Y", "O")), "EXCLUDED_NOT_ACCEPTED_BY_EITHER"),
                 (ok(Y), ok(("Y", "U", "Y", "C")), "EXCLUDED_UNCERTAIN"), (None, ok(Y), "EXCLUDED_MISSING_R1"),
                 (ok(Y), None, "EXCLUDED_MISSING_R2"), ({"status": "MALFORMED", "parsed": None}, ok(Y), "EXCLUDED_MALFORMED_R1"),
                 (ok(Y), {"status": "MALFORMED", "parsed": None}, "EXCLUDED_MALFORMED_R2")]
        for r1, r2, expected in cases:
            self.assertEqual(t.pair_decision(r1, r2)[0], expected)

    def test_end_to_end_fixture(self):
        run_to_t4(self.cfg)
        t.t4_review_r1(self.cfg, lambda: FakeBackend(review_answers={("Alpha Two", "Gamma Two"): ("Y", "Y", "N", "C")}), "llama")
        r2 = Path(self.tmp.name) / "r2.jsonl"
        write_r2(self.cfg, r2, {})
        t.t4_ingest_r2(self.cfg, r2, "test-reviewer")
        t.t5_accept(self.cfg)
        seal = json.loads((self.cfg.stage("t5") / "SEAL.json").read_text())
        self.assertEqual(seal["pair_status_counts"], {"ACCEPTED_AI_ASSISTED": 4, "EXCLUDED_DISAGREEMENT": 1})
        self.assertEqual(seal["question_status_counts"], {"QUALIFIES": 2, "EXCLUDED_MULTIPLE_ACCEPTED_PARENTS": 1})
        sample = t.read_jsonl(self.cfg.stage("t5") / "development_sample.jsonl")
        self.assertEqual([s["urbench_qid"] for s in sample], QIDS[:2])
        self.assertEqual([c["title"] for c in sample[1]["children"]], ["Beta Two"])   # disagreeing child excluded
        self.assertEqual(seal["sample_status"], "SAMPLE_READY")
        self.assertFalse(seal["retrieval_used"])

    def test_missing_duplicate_and_malformed_r2(self):
        run_to_t4(self.cfg)
        t.t4_review_r1(self.cfg, FakeBackend, "llama")
        r2 = Path(self.tmp.name) / "r2.jsonl"
        write_r2(self.cfg, r2, {("Alpha Three", "Beta Three"): ("Y", "Y", "Y", "U")},
                 omit={("Alpha One", "Beta One")}, duplicate={("Alpha Two", "Beta Two")},
                 malformed={("Alpha Two", "Gamma Two")})
        t.t4_ingest_r2(self.cfg, r2, "test-reviewer")
        t.t5_accept(self.cfg)
        rows = {r["pair_id"]: r["status"] for r in t.read_jsonl(self.cfg.stage("t5") / "pair_decisions.jsonl")}
        pairs = {(p["parent_title"], p["child_title"]): p["pair_id"] for p in t.read_jsonl(self.cfg.stage("t3") / "candidate_pairs.jsonl")}
        self.assertEqual(rows[pairs[("Alpha One", "Beta One")]], "EXCLUDED_MISSING_R2")
        self.assertEqual(rows[pairs[("Alpha Two", "Beta Two")]], "EXCLUDED_MALFORMED_R2")
        self.assertEqual(rows[pairs[("Alpha Two", "Gamma Two")]], "EXCLUDED_MALFORMED_R2")
        self.assertEqual(rows[pairs[("Alpha Three", "Beta Three")]], "EXCLUDED_UNCERTAIN")
        self.assertEqual(rows[pairs[("Delta Three", "Gamma Three")]], "ACCEPTED_AI_ASSISTED")
        seal = json.loads((self.cfg.stage("t5") / "SEAL.json").read_text())
        self.assertEqual(seal["sample_status"], "FEASIBILITY_STOP_BELOW_MINIMUM")

    def test_historical_acceptance_reproduces_n25(self):
        sys.path.insert(0, str(HERE))
        from bridge_pilot_core import CHILD_COUNTS
        cands = t.read_jsonl(HIST / "bridge_candidate_qids.jsonl")
        human = {(r["qid"], r["parent_source_instance_id"], r["child_source_instance_id"]): r
                 for r in t.read_jsonl(HIST / "human_verified_candidates.jsonl")}
        pairs, decisions = [], {}
        for c in cands:
            for p in c["candidate_pairs"]:
                key = (c["urbench_qid"], p["parent_source_instance_id"], p["child_source_instance_id"])
                a = human[key]["human_answers"]
                answers = ok((a["parent_directly_identifiable_from_urdu_question"],
                              a["child_not_directly_identifiable_from_urdu_question_alone"],
                              a["stated_intermediate_information_makes_child_identifiable_or_recoverable"],
                              a["dependency_status"]))
                pid = t.pair_id(*key)
                pairs.append({"pair_id": pid, "urbench_qid": key[0], "parent_source_instance_id": key[1],
                              "parent_title": p["parent_title"], "child_source_instance_id": key[2],
                              "child_title": p["child_title"]})
                decisions[pid] = t.pair_decision(answers, answers)[0]
                self.assertEqual(decisions[pid] == "ACCEPTED_AI_ASSISTED",
                                 human[key]["verdict"] == "VERIFIED_CLEAN_BRIDGE")
        rank = {q: i for i, q in enumerate(sorted({p["urbench_qid"] for p in pairs}))}
        questions, sample, status = t.select_sample(pairs, decisions, rank, None, 0)
        self.assertEqual(sum(v == "ACCEPTED_AI_ASSISTED" for v in decisions.values()), 36)
        self.assertEqual({q["urbench_qid"]: len(q["children"]) for q in sample}, dict(CHILD_COUNTS))
        self.assertTrue(all(q["status"] == "QUALIFIES" for q in questions))

    def test_selection_is_deterministic_and_ordered(self):
        pairs = [{"pair_id": f"p{i}", "urbench_qid": QIDS[i % 5], "parent_source_instance_id": f"P{i % 5}",
                  "parent_title": f"Parent {i % 5}", "child_source_instance_id": f"C{i}", "child_title": f"Child {i}"}
                 for i in range(15)]
        decisions = {p["pair_id"]: "ACCEPTED_AI_ASSISTED" for p in pairs}
        rank = {q: i for i, q in enumerate(QIDS)}
        base = t.select_sample(pairs, decisions, rank, 3, 2)
        shuffled = list(pairs)
        random.Random(7).shuffle(shuffled)
        again = t.select_sample(shuffled, decisions, rank, 3, 2)
        self.assertEqual(base[1], again[1])
        self.assertEqual([q["urbench_qid"] for q in base[1]], QIDS[:3])
        self.assertEqual(t.select_sample(pairs, decisions, rank, 3, 4)[2], "FEASIBILITY_STOP_BELOW_MINIMUM")

    def test_stage_outputs_are_deterministic(self):
        run_to_t4(self.cfg)
        cfg2 = replace(self.cfg, triage_root=Path(self.tmp.name) / "triage_again")
        run_to_t4(cfg2)
        for stage, name in (("t0", "corpus_status.jsonl"), ("t3", "candidate_pairs.jsonl"), ("t4", "review_packets.jsonl")):
            self.assertEqual((self.cfg.stage(stage) / name).read_bytes(), (cfg2.stage(stage) / name).read_bytes())


# ------------------------------------------------------------------ T0, overwrite, smoke

class OutputTests(Base):
    def test_t0_statuses_and_normalization(self):
        t.t0_corpus_status(self.cfg, progress_every=0)
        rows = {r["normalized_gold_title"]: r["exact_corpus_status"]
                for r in t.read_jsonl(self.cfg.stage("t0") / "corpus_status.jsonl")}
        self.assertEqual(rows["alpha one"], "EXACT_PRESENT")      # metadata title "ALPHA_ONE"
        self.assertEqual(rows["beta five"], "EXACT_ABSENT")

    def test_t0_metadata_drift_writes_nothing(self):
        self.raises("METADATA_SHA256_DRIFT", t.t0_corpus_status, replace(self.cfg, metadata_sha256="0" * 64), 0)
        self.assertFalse(self.cfg.stage("t0").exists())

    def test_overwrite_refusal(self):
        t.t0_corpus_status(self.cfg, progress_every=0)
        self.raises("OUTPUT_EXISTS_REFUSING_OVERWRITE", t.t0_corpus_status, self.cfg, 0)
        t.t1_pass2(self.cfg, FakeBackend)
        self.raises("OUTPUT_EXISTS_REFUSING_OVERWRITE", t.t1_pass2, self.cfg, FakeBackend)

    def test_malformed_model_responses_are_recorded_not_repaired(self):
        backend = FakeBackend(pass2_override={"Alpha One": "not json", "Beta One": json.dumps(dict(P2_OK, confidence="SURE"))})
        t.t1_pass2(self.cfg, lambda: backend)
        recs = {r["gold_title"]: r for r in t.read_jsonl(self.cfg.stage("t1") / "pass2_records.jsonl")}
        self.assertEqual((recs["Alpha One"]["status"], recs["Alpha One"]["malformed_reason"]), ("MALFORMED", "NOT_JSON"))
        self.assertEqual(recs["Alpha One"]["raw_output"], "not json")
        self.assertEqual(recs["Beta One"]["malformed_reason"], "ENUM")

    def test_smoke_outputs_are_refused_as_annotations(self):
        smoke_dir = Path(self.tmp.name) / "x_smoke" / "llama"
        smoke_dir.mkdir(parents=True)
        self.raises("SMOKE_OUTPUT_IS_NOT_ANNOTATION", t.load_sealed, smoke_dir, "t1")

    def test_smoke_items_are_historical_only(self):
        with t.stage_reads(*t.SMOKE_HISTORICAL.values()):
            p2u, p3u, ctx, ru = t.smoke_items()
        dev = set((ROOT / "outputs/efbpt/qea_preparation_v1_r2/selection/dev_triage_qids.txt").read_text().split())
        master = {r["gold_title"]: r["urbench_qid"] for r in t.read_jsonl(STAGE0 / "source_instance_master.jsonl")}
        self.assertEqual((len(p2u), len(p3u), len(ru)), (4, 2, 2))
        for u in p2u:
            self.assertNotIn(master[json.loads(u)["gold_title"]], dev)
        for u in ru:
            self.assertEqual(set(json.loads(u)), set(t.PACKET_FIELDS) - {"packet_id"})


class CaptureBackend(t.Backend):
    """Stub for the smoke path: valid canned outputs; records calls. Never loads a model."""
    key = "capture"

    def __init__(self):
        self.calls = []

    def identity(self):
        return {"family": "capture", "backend": "stub"}

    def generate(self, batch, max_new_tokens, schemas=None):
        self.schema_batches = getattr(self, "schema_batches", []) + [schemas]
        out = []
        for m in batch:
            system, user = split_messages(m)
            self.calls.append((system, max_new_tokens))
            if system == t.PASS2_SYSTEM:
                text = json.dumps({"decision": "NOT_YET_EXPLICIT", "explicit_relation_type": "", "urdu_span": "",
                                   "confidence": "LOW", "rationale": "stub"})
            elif system == t.REVIEW_SYSTEM:
                text = json.dumps({"q1": "U", "q2": "U", "q3": "U", "q4": "U", "note": "stub"})
                out.append({"text": text, "finish_reason": "stop", "prompt_tokens": 1, "output_tokens": 1,
                            "diagnostics": {"stub": len(out)}})
                continue
            else:
                text = json.dumps({"decision": "AMBIGUOUS", "concrete_intermediate_information": "",
                                   "official_step_indices": [0], "dependency_status": "NOT_APPLICABLE",
                                   "proposed_parent_source_titles": [], "dependency_confidence": "NOT_APPLICABLE",
                                   "confidence": "LOW", "rationale": "stub"})
            out.append({"text": text, "finish_reason": "stop", "prompt_tokens": 1, "output_tokens": 1})
        return out


def capture_make_backend(captured, backend):
    def fake_make_backend(name, max_num_seqs=8, gpu_memory_utilization=None, structured=False):
        captured.update(name=name, max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization,
                        structured=structured)
        return backend
    return fake_make_backend


class MemoryFractionTests(unittest.TestCase):
    def test_t1_t2_use_the_tested_fraction_and_gemma_has_none(self):
        self.assertEqual(t.VLLM_GPU_MEMORY_UTILIZATION, 0.90)                 # protocol rev 0.2 value, recorded only
        self.assertEqual(t.SMOKE_LLAMA_GPU_MEMORY_UTILIZATION, 0.95)
        self.assertEqual(t.LLAMA_EXECUTION["gpu_memory_utilization"], 0.95)
        captured = {}
        with mock.patch.object(t, "make_backend", capture_make_backend(captured, CaptureBackend())):
            t.llama_execution_backend()
        self.assertEqual(captured, {"name": "llama", "max_num_seqs": 4, "gpu_memory_utilization": 0.95,
                                    "structured": True})
        with self.assertRaises(t.TriageError):
            t.make_backend("gemma", gpu_memory_utilization=0.95)

    def test_smoke_llama_uses_override_and_records_it(self):
        captured, backend = {}, CaptureBackend()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, guard_env(tmp)), \
                mock.patch.object(t, "make_backend", capture_make_backend(captured, backend)):
            t.smoke("llama", out_root=Path(tmp))
            seal = json.loads((Path(tmp) / "llama" / "SEAL.json").read_text())
        self.assertEqual(captured, {"name": "llama", "max_num_seqs": 4, "gpu_memory_utilization": 0.95,
                                    "structured": False})
        self.assertEqual(backend.schema_batches, [None, None])        # unconstrained smoke: no schemas
        self.assertIs(seal["constrained"], False)
        self.assertIsNone(seal["structured_outputs"])
        self.assertEqual(sorted(backend.calls, key=lambda c: c[1]),
                         [(t.PASS2_SYSTEM, 256)] * 4 + [(t.PASS3_SYSTEM, 512)] * 2)
        self.assertEqual((seal["scope"], seal["calls"], seal["format_ok"]), ("SMOKE_NOT_ANNOTATION", 6, 6))
        self.assertEqual(seal["smoke_overrides"]["gpu_memory_utilization"], 0.95)
        self.assertEqual((seal["smoke_overrides"]["protocol_r02_value"], seal["smoke_overrides"]["t1_t2_value"]),
                         (0.90, 0.95))


# ------------------------------------------------------------------ code identity (offline)

class CodeIdentityGuardTests(unittest.TestCase):
    def test_guard_passes_and_records_executed_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = guard_env(tmp)
            record = t.code_identity_guard(env)
        self.assertEqual(record["executed_wrapper"]["path"], env["QEA_EXECUTED_WRAPPER_PATH"])
        for name, var in t.CODE_GUARD_ENV.items():
            self.assertEqual(record[name]["observed_sha256"], env[var])

    def test_guard_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = guard_env(tmp)
            cases = [dict(good, **{var: "0" * 64}) for var in t.CODE_GUARD_ENV.values()]          # mismatch
            cases += [dict(good, **{var: good[var].upper()}) for var in t.CODE_GUARD_ENV.values()]  # malformed
            cases += [dict(good, **{var: good[var][:63]}) for var in t.CODE_GUARD_ENV.values()]
            cases += [{k: v for k, v in good.items() if k != drop} for drop in good]                 # missing
            cases += [dict(good, QEA_EXECUTED_WRAPPER_PATH=str(Path(tmp) / "absent")), {}]
            for env in cases:
                with self.assertRaises(t.TriageError):
                    t.code_identity_guard(env)

    def test_constrained_smoke_refuses_before_inputs_or_model_without_guard(self):
        calls, original_backend, original_items = [], t.make_backend, t.smoke_items
        t.make_backend = lambda *a, **k: calls.append("make_backend")
        t.smoke_items = lambda: calls.append("smoke_items")
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ):
                for var in t.CODE_GUARD_ENV.values():
                    os.environ.pop(var, None)
                for backend, constrained in (("llama", True), ("llama", False), ("gemma", False)):
                    root = Path(tmp) / f"fresh_{backend}_{constrained}"
                    with self.assertRaises(t.TriageError) as ctx:
                        t.smoke(backend, out_root=root, constrained=constrained)
                    self.assertTrue(str(ctx.exception).startswith("CODE_IDENTITY_"), str(ctx.exception))
                    self.assertFalse(root.exists())
        finally:
            t.make_backend, t.smoke_items = original_backend, original_items
        self.assertEqual(calls, [])

    def test_interactive_r2_guard_needs_runner_and_tests_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = guard_env(tmp)
            no_wrapper = {k: v for k, v in good.items() if k not in ("QEA_EXPECT_WRAPPER_SHA256", "QEA_EXECUTED_WRAPPER_PATH")}
            record = t.code_identity_guard(no_wrapper, wrapper=False)
            self.assertIn("not_applicable", record["executed_wrapper"])
            for var in ("QEA_EXPECT_RUNNER_SHA256", "QEA_EXPECT_TEST_SHA256"):
                with self.assertRaises(t.TriageError):
                    t.code_identity_guard(dict(no_wrapper, **{var: "0" * 64}), wrapper=False)
            with self.assertRaises(t.TriageError):
                t.code_identity_guard(no_wrapper)                                      # batch stages need the wrapper

    def test_wrappers_guard_before_tests_and_stage(self):
        for name, stages in (("efbpt_qea_triage_gpu_v1.sbatch", ("smoke-gemma", "t1", "t2", "r1", "calib-gemma", "ident-gemma")),
                             ("efbpt_qea_triage_cpu_v1.sbatch", ("t0", "t3", "t4", "t5"))):
            text = (HERE / name).read_text()
            guard = text.index('echo "identity guard: PASS"')
            tests = text.index('"${PY}" -B -s "${DIR}/efbpt_qea_triage_v1_test.py"')
            self.assertLess(guard, tests, name)
            self.assertLess(tests, text.index("exec "), name)
            for var in ("QEA_EXPECT_RUNNER_SHA256", "QEA_EXPECT_TEST_SHA256", "QEA_EXPECT_WRAPPER_SHA256"):
                self.assertIn(var, text[:guard])
            for token in ('readlink -f "$0"', "export QEA_OBSERVED_RUNNER_SHA256", "QEA_EXECUTED_WRAPPER_PATH",
                          "exit 3", "PYTHONDONTWRITEBYTECODE=1", "set -euo pipefail"):
                self.assertIn(token, text, (name, token))
            self.assertNotIn("--batch-size 8", text)
            for stage in stages:
                self.assertIn(stage, text)


class StageGuardTests(Base):
    """Every stage refuses on a code-identity mismatch before reading any input or loading any model."""

    def test_every_stage_refuses_before_reading_or_loading(self):
        loads = []

        def factory():
            loads.append("load")
            return FakeBackend()

        r2 = Path(self.tmp.name) / "r2.jsonl"
        r2.write_text("")
        stages = {"t0": lambda: t.t0_corpus_status(self.cfg, 0), "t1": lambda: t.t1_pass2(self.cfg, factory),
                  "t2": lambda: t.t2_pass3(self.cfg, factory), "t3": lambda: t.t3_candidates(self.cfg),
                  "t4": lambda: t.t4_packets(self.cfg), "r1": lambda: t.t4_review_r1(self.cfg, factory, "gemma"),
                  "t5": lambda: t.t5_accept(self.cfg),
                  "smoke-gemma": lambda: t.smoke("gemma", out_root=Path(self.tmp.name) / "smoke_g"),
                  "smoke-llama-so": lambda: t.smoke("llama", out_root=Path(self.tmp.name) / "smoke", constrained=True)}
        interactive = {"ingest-r2": lambda: t.t4_ingest_r2(self.cfg, r2, "x"),
                       "render-r2": lambda: t.t4_render_r2(self.cfg, Path(self.tmp.name) / "r2_packet.json")}
        bad_envs = [dict(self.env, **{var: "0" * 64}) for var in t.CODE_GUARD_ENV.values()]
        bad_envs.append({k: v for k, v in self.env.items() if k != "QEA_EXPECT_RUNNER_SHA256"})
        with mock.patch.object(t, "make_backend", lambda *a, **k: loads.append("load")):
            for env in bad_envs:
                runner_or_tests_bad = env.get("QEA_EXPECT_WRAPPER_SHA256") == self.env["QEA_EXPECT_WRAPPER_SHA256"]
                todo = dict(stages, **(interactive if runner_or_tests_bad else {}))
                for name, fn in todo.items():
                    t.READ_LOG.clear()
                    with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(t.TriageError) as ctx:
                        fn()
                    self.assertTrue(str(ctx.exception).startswith("CODE_IDENTITY_"), (name, str(ctx.exception)))
                    self.assertEqual(t.READ_LOG, [], name)
        self.assertEqual(loads, [])
        self.assertFalse(self.cfg.triage_root.exists())
        self.assertFalse((Path(self.tmp.name) / "smoke").exists())
        self.assertFalse((Path(self.tmp.name) / "r2_packet.json").exists())

    def test_seals_record_the_guard_and_code(self):
        backend = run_to_t4(self.cfg)
        t.t4_review_r1(self.cfg, lambda: backend, "gemma")
        r2 = Path(self.tmp.name) / "r2.jsonl"
        write_r2(self.cfg, r2, {})
        t.t4_ingest_r2(self.cfg, r2, "test-reviewer")
        t.t5_accept(self.cfg)
        runner = sha(Path(t.__file__).resolve().read_bytes())
        for stage in ("t0", "t1", "t2", "t3", "t4", "r1", "r2", "t5"):
            seal = json.loads((self.cfg.stage(stage) / "SEAL.json").read_text())
            self.assertEqual(seal["code"]["sha256"], runner, stage)
            guard = seal["code_identity_guard"]
            self.assertEqual(guard["runner"]["observed_sha256"], runner, stage)
            self.assertEqual(guard["tests"]["observed_sha256"], self.env["QEA_EXPECT_TEST_SHA256"], stage)
            if stage == "r2":
                self.assertIn("not_applicable", guard["executed_wrapper"])
            else:
                self.assertEqual(guard["executed_wrapper"]["path"], self.env["QEA_EXECUTED_WRAPPER_PATH"], stage)
        for stage in ("t1", "t2"):
            seal = json.loads((self.cfg.stage(stage) / "SEAL.json").read_text())
            self.assertEqual(seal["execution"]["sha256"], t.LLAMA_EXECUTION_SHA256)
            self.assertEqual(seal["execution"]["config"], t.LLAMA_EXECUTION)
        t3 = json.loads((self.cfg.stage("t3") / "SEAL.json").read_text())
        self.assertEqual({k: v["execution_sha256"] for k, v in t3["upstream_identity"].items()},
                         {"t0": None, "t1": t.LLAMA_EXECUTION_SHA256, "t2": t.LLAMA_EXECUTION_SHA256})


# Job-96136 regression values, embedded so that no test depends on a smoke directory existing.
JOB_96136_PASS2_2_RAW = ('{"decision": "EXPLICIT", "explicit_relation_type": "DIRECT_MENTION", "urdu_span": "\u0686\u06a9\u0648\u062a\u0631\u0627", '
                         '"confidence": "HIGH", "rationale": "The Urdu question directly mentions the word "" which is a direct '
                         'mention of the title ""."}')                                  # unescaped quotes (NOT_JSON)
JOB_96136_PASS3_0_RAW = ('{"decision": "LATENT_BRIDGE", "concrete_intermediate_information": "The speed required to break the '
                         'sound barrier is compared to the top speed of an Audi R8 V-10 Plus.", "official_step_indices": [1, 2], '
                         '"dependency_status": "CLEAR_DEPENDENCY", "proposed_parent_source_titles": ["Speed of sound"], '
                         '"dependency_confidence": "HIGH", "confidence": "HIGH", "rationale": "The question compares the speed '
                         'of the Audi R8 V-10 Plus to the speed of sound, requiring the title \'Speed of sound\' as an '
                         'intermediate step."}')                                          # self parent
JOB_96136_PROMPT_TOKENS = [450, 451, 461, 464, 786, 686]
JOB_96151_MODEL = {"path": "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4",
                   "backend": "vllm", "quantization": "awq", "enforce_eager": True, "max_model_len": 4096,
                   "gpu_memory_utilization": 0.95, "max_num_seqs": 4, "decoding": "greedy temperature=0",
                   "structured_outputs": {"backend": "xgrammar", "disable_any_whitespace": True, "disable_fallback": True}}
JOB_96136_SEAL_SHA256 = "f52d236b89c4fb02771e92ff9586ea010b089f507f232fcddc480e1782bc4765"
JOB_96136_RECORDS_SHA256 = "ec23cb819127053d0f2d44442965c36a3cfc43ebce17beb21e1f38218f4acb2b"
JOB_96317_SEAL_SHA256 = "af9dafd399e8d879450caa2b4e05ba601e99962c989dd60eda8a2610babbf52e"
JOB_96317_RECORDS_SHA256 = "49163628e083a914571c1f0237b05c3d1e398408324ddc15481d0ac4dde65b2c"
JOB_96151_RECORDS_SHA256 = "f3b0b4e2bbda45998baec15dcc33ede4523608ee6f6e350fdb0ec27d97873374"
GEMMA_SMOKE_REVIEW_SHA256 = [   # the two review payloads as built by the job-96151 code (0d5570d2...)
    "d1497c376f27b28ac6ac15c1dd87955791d0214f69d54f1bc2a92a5becefa9c8",
    "334e26400b248871e007c044ead675b447c7168ef615d2367fc8d8ee1cb39a55",
]
SMOKE_INPUT_SHA256 = [   # the six smoke user payloads as built by the job-96136 code (seal code 42894c94...)
    "444a49458a91fe3d274a00eedf1777e7c64dcd6d7d628a953eab812a4a626fa6",
    "d7c485e0341b945bc7405fd24dbe543615374083f036fcf4ded01ba1e55131e4",
    "35bfa6470fed34b5d4b221a94966e2fe6231ebcc694b2404bac074ce21af214e",
    "7c97460a3090157f3b7ed288b42ceb5bd0939444339ca8c6f5b4eb6fb8ce98ae",
    "7d14110d3c1ca46dc507b5f1cde9ce20834579ed3f550174f95e34ecda4162be",
    "e18af38c80ac222201f982254248716b68782d2d2ccad37ccf8a48b9e0e7a5b5",
]
SB_TITLES = ["Audi R8", "Audi R8 (Type 4S)", "Audi S8", "Sound barrier", "Speed of sound"]


def grammar(schema):
    import xgrammar
    return xgrammar.Grammar.from_json_schema(json.dumps(schema), any_whitespace=False)


def accepts(schema, text):
    from xgrammar.testing import _is_grammar_accept_string
    return _is_grammar_accept_string(grammar(schema), text)


P2_VALID_EXPLICIT = {"decision": "EXPLICIT", "explicit_relation_type": "DIRECT_MENTION", "urdu_span": "چکوترا",
                     "confidence": "HIGH", "rationale": 'The question says "چکوترا" \\ directly.'}
P2_VALID_NOT_YET = {"decision": "NOT_YET_EXPLICIT", "explicit_relation_type": "", "urdu_span": "",
                    "confidence": "LOW", "rationale": "Only related."}
P3_BASE = {"decision": "LATENT_BRIDGE", "concrete_intermediate_information": "Via the barrier.",
           "official_step_indices": [0, 1], "dependency_status": "CLEAR_DEPENDENCY",
           "proposed_parent_source_titles": ["Sound barrier"], "dependency_confidence": "HIGH",
           "confidence": "HIGH", "rationale": "Clear."}
P3_AMBIGUOUS = dict(P3_BASE, decision="AMBIGUOUS", concrete_intermediate_information="", dependency_status="NOT_APPLICABLE",
                    proposed_parent_source_titles=[], dependency_confidence="NOT_APPLICABLE")


def js(obj, ascii_only=False):
    return json.dumps(obj, ensure_ascii=ascii_only)      # json.dumps separators == xgrammar (", ", ": ")


class StructuredOutputTests(unittest.TestCase):
    def setUp(self):
        self.cands = t.parent_candidates(SB_TITLES, "Speed of sound")
        self.p3 = t.pass3_schema(3, self.cands, "Speed of sound")

    def test_string_content_quotes_backslash_urdu_unicode_escapes(self):
        p2 = t.pass2_schema()
        self.assertTrue(accepts(p2, js(P2_VALID_EXPLICIT)))                    # raw Urdu, \\" and \\\\
        self.assertTrue(accepts(p2, js(P2_VALID_EXPLICIT, ascii_only=True)))   # \\uXXXX escapes
        raw = JOB_96136_PASS2_2_RAW
        self.assertFalse(accepts(p2, raw))                                     # job 96136 unescaped quotes

    def test_xgrammar_minlength_limitation_reproduction(self):
        version = importlib.metadata.version("xgrammar")
        if version != "0.1.27":
            self.skipTest(f"reproduction recorded for xgrammar 0.1.27, installed {version}")
        base = {"type": "object", "properties": {"r": {"type": "string"}}, "required": ["r"], "additionalProperties": False}
        limited = json.loads(json.dumps(base))
        limited["properties"]["r"]["minLength"] = 1
        for text in ('{"r": "a \\"q\\" b"}', '{"r": "\\u0635"}'):
            self.assertTrue(accepts(base, text), text)
            self.assertFalse(accepts(limited, text), text)                    # the version/schema-specific limitation
        self.assertTrue(accepts(limited, '{"r": "abc"}'))
        with self.assertRaises(t.TriageError):
            t.validate_schema(limited)                                        # our allowlist fails closed

    def test_every_valid_branch_is_representable_and_parser_consistent(self):
        p2 = t.pass2_schema()
        for obj in (P2_VALID_EXPLICIT, P2_VALID_NOT_YET, dict(P2_VALID_EXPLICIT, urdu_span="")):
            self.assertTrue(accepts(p2, js(obj)))
            self.assertIsNone(t.parse_pass2(js(obj), "stop", "کیا چکوترا")[1])
        for rel in t.EXPLICIT_RELATIONS:
            self.assertTrue(accepts(p2, js(dict(P2_VALID_EXPLICIT, explicit_relation_type=rel))))
        branches = [P3_BASE, P3_AMBIGUOUS,
                    dict(P3_BASE, dependency_status="UNRESOLVED", proposed_parent_source_titles=[], dependency_confidence="LOW"),
                    dict(P3_BASE, dependency_status="PARALLEL_OR_UNORDERED", proposed_parent_source_titles=[]),
                    dict(P3_BASE, dependency_status="MULTIPLE_PLAUSIBLE_PARENTS", proposed_parent_source_titles=["Audi S8", "Sound barrier"])]
        for obj in branches:
            for conf in t.CONFIDENCES:
                o = dict(obj, confidence=conf)
                self.assertTrue(accepts(self.p3, js(o)), o)
                self.assertIsNone(t.parse_pass3(js(o), "stop", 3, self.cands, "Speed of sound")[1], o)

    def test_candidate_cardinalities(self):
        amb, unresolved = js(P3_AMBIGUOUS), js(dict(P3_BASE, dependency_status="UNRESOLVED", proposed_parent_source_titles=[]))
        empty = t.pass3_schema(3, [], "Speed of sound")
        self.assertEqual(len(empty["anyOf"]), 2)
        self.assertFalse(accepts(empty, js(P3_BASE)))
        self.assertTrue(accepts(empty, amb) and accepts(empty, unresolved))
        one = t.pass3_schema(3, ["Sound barrier"], "Speed of sound")
        self.assertEqual(len(one["anyOf"]), 3)
        self.assertTrue(accepts(one, js(P3_BASE)))
        self.assertFalse(accepts(one, js(dict(P3_BASE, dependency_status="MULTIPLE_PLAUSIBLE_PARENTS",
                                              proposed_parent_source_titles=["Sound barrier", "Sound barrier"]))))
        self.assertEqual(len(self.p3["anyOf"]), 4)
        self.assertTrue(accepts(self.p3, js(dict(P3_BASE, dependency_status="MULTIPLE_PLAUSIBLE_PARENTS",
                                                 proposed_parent_source_titles=["Audi R8", "Audi S8", "Sound barrier"]))))

    def test_self_parent_and_out_of_list_parents_rejected(self):
        self.assertEqual(self.cands, ["Audi R8", "Audi R8 (Type 4S)", "Audi S8", "Sound barrier"])
        self.assertEqual(t.parent_candidates(["Speed_of_Sound", "SPEED OF SOUND", "Sound barrier"], "Speed of sound"),
                         ["Sound barrier"])
        self.assertEqual(t.parent_candidates(["X Film", "x film", "Target"], "Target"), ["X Film", "x film"])  # not merged
        raw_self = JOB_96136_PASS3_0_RAW
        self.assertFalse(accepts(self.p3, raw_self))                           # job 96136 self-parent
        for bad in (["Speed of sound"], ["Speed_of_sound"], ["Unlisted Page"], ["sound barrier"]):
            self.assertFalse(accepts(self.p3, js(dict(P3_BASE, proposed_parent_source_titles=bad))), bad)
        for bad, code in ((["Speed_of_Sound"], "SELF_PARENT"), (["Unlisted Page"], "PARENT_NOT_IN_CANDIDATES")):
            self.assertEqual(t.parse_pass3(js(dict(P3_BASE, proposed_parent_source_titles=bad)), "stop", 3,
                                           self.cands, "Speed of sound")[1], code)
        with self.assertRaises(t.TriageError):
            t.pass3_schema(3, ["Sound barrier", "speed_of_sound"], "Speed of sound")   # PASS3_SELF_CANDIDATE
        with self.assertRaises(t.TriageError):
            t.pass3_schema(3, ["A", "A"], "Speed of sound")                           # duplicate candidates

    def test_parser_only_rules_remain(self):
        dup_parents = js(dict(P3_BASE, dependency_status="MULTIPLE_PLAUSIBLE_PARENTS",
                              proposed_parent_source_titles=["Audi S8", "Audi S8"]))
        cases = [(dup_parents, "DUPLICATE_PARENT"), (js(dict(P3_BASE, official_step_indices=[1, 1])), "DUPLICATE_STEP"),
                 (js(dict(P3_BASE, rationale="")), "EMPTY_RATIONALE"),
                 (js(dict(P3_BASE, concrete_intermediate_information="")), "EMPTY_INTERMEDIATE")]
        for text, code in cases:
            self.assertTrue(accepts(self.p3, text), code)                        # grammar cannot enforce these
            self.assertEqual(t.parse_pass3(text, "stop", 3, self.cands, "Speed of sound")[1], code)
        self.assertTrue(accepts(t.pass2_schema(), js(dict(P2_VALID_NOT_YET, rationale=""))))
        self.assertEqual(t.parse_pass2(js(dict(P2_VALID_NOT_YET, rationale="")), "stop", "x")[1], "EMPTY_RATIONALE")
        self.assertFalse(accepts(self.p3, js(dict(P3_BASE, official_step_indices=[3]))))    # zero-based, range 0..2
        self.assertFalse(accepts(self.p3, js(dict(P3_BASE, official_step_indices=[]))))

    def test_zero_step_input_fails_closed(self):
        with self.assertRaises(t.TriageError) as ctx:
            t.pass3_schema(0, self.cands, "Speed of sound")
        self.assertTrue(str(ctx.exception).startswith("PASS3_ZERO_STEPS"))
        self.assertEqual(t.parse_pass3(js(P3_BASE), "stop", 0, self.cands, "Speed of sound")[1], "STEP_RANGE")

    def test_schemas_pass_validation_and_unsupported_features_fail_closed(self):
        for schema in (t.pass2_schema(), self.p3, t.pass3_schema(2, ["Grapefruit"], "Grapefruit–drug interactions"),
                       t.pass3_schema(3, [], "Speed of sound")):
            self.assertEqual(len(t.validate_schema(schema)), 64)
        for bad in ({"type": "string", "pattern": "a"}, {"type": "array", "uniqueItems": True},
                    {"type": "object", "properties": {"a": {"type": "string", "maxLength": 3}}}, {"enum": []}):
            with self.assertRaises(t.TriageError):
                t.validate_schema(bad)

    def test_token_level_matcher_with_llama_tokenizer(self):
        import xgrammar
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(t.MODELS["llama"]["path"], local_files_only=True)   # tokenizer only
        info = xgrammar.TokenizerInfo.from_huggingface(tok, vocab_size=128256)
        compiled = xgrammar.GrammarCompiler(info).compile_json_schema(json.dumps(self.p3), any_whitespace=False)

        def run(text):
            m = xgrammar.GrammarMatcher(compiled)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            return all(m.accept_token(i) for i in ids), m

        ok, m = run(js(P3_BASE))
        self.assertTrue(ok)
        self.assertTrue(m.accept_token(info.stop_token_ids[0]) and m.is_terminated())
        self.assertFalse(run(js(dict(P3_BASE, proposed_parent_source_titles=["Speed of sound"])))[0])
        self.assertFalse(run(JOB_96136_PASS3_0_RAW)[0])

    def test_vllm_generate_keeps_schema_request_association(self):
        class Tok:
            def apply_chat_template(self, m, tokenize, add_generation_prompt):
                return [1] * (4000 if "TOO_LONG" in m[-1]["content"] else 10)

        class Out:
            def __init__(self, text):
                self.outputs = [type("C", (), {"text": text, "finish_reason": "stop", "token_ids": [1, 2]})()]

        class LLM:
            def generate(self, prompts, params, use_tqdm):
                self.params = params
                return [Out(f"out{i}") for i in range(len(prompts))]

        backend = object.__new__(t.VllmBackend)
        backend.tokenizer, backend.llm, backend.structured = Tok(), LLM(), True
        backend._sampling = lambda **kw: dict(kw)
        backend._so = lambda json: {"json": json}
        batch = [[{"role": "user", "content": c}] for c in ("a", "TOO_LONG", "b", "c")]
        schemas = [{"enum": [f"s{i}"]} for i in range(4)]
        res = backend.generate(batch, 512, schemas=schemas)
        self.assertEqual([r["finish_reason"] for r in res], ["stop", "prompt_too_long", "stop", "stop"])
        self.assertEqual([p["structured_outputs"]["json"] for p in backend.llm.params], [schemas[0], schemas[2], schemas[3]])
        self.assertTrue(all(p["structured_outputs"]["json"] is not s for p, s in zip(backend.llm.params, (schemas[0], schemas[2], schemas[3]))))
        backend.generate(batch[:1], 256)                                       # unconstrained path unchanged
        self.assertEqual(backend.llm.params, {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256, "seed": 0})
        backend.structured = False
        with self.assertRaises(t.TriageError):
            backend.generate(batch[:1], 256, schemas=schemas[:1])              # schemas need a structured engine
        backend.structured = True
        with self.assertRaises(t.TriageError):
            backend.generate(batch, 256, schemas=schemas[:3])                 # count mismatch

    def test_run_calls_slicing_keeps_each_schema_with_its_request(self):
        seen = []

        class B(t.Backend):
            def generate(self, batch, max_new_tokens, schemas=None):
                seen.append((len(batch), schemas))
                return [{"text": "{}", "finish_reason": "stop", "prompt_tokens": 1, "output_tokens": 1} for _ in batch]

        users = [json.dumps({"n": i}) for i in range(5)]
        schemas = [{"enum": [i]} for i in range(5)]
        recs = t.run_calls(B(), "llama", "sys", users, 64, batch_size=2, schemas=schemas)
        self.assertEqual([len(s) for _n, s in seen], [2, 2, 1])
        self.assertEqual([r["schema_sha256"] for r in recs], [t.sha256_text(t.canonical(s)) for s in schemas])
        seen.clear()
        plain = t.run_calls(B(), "llama", "sys", users, 64, batch_size=2)
        self.assertTrue(all(s is None for _n, s in seen) and all("schema_sha256" not in r for r in plain))

    def test_hf_gemma_unchanged_and_prediction_stages_constrained(self):
        hf = object.__new__(t.HfBnbBackend)
        with self.assertRaises(t.TriageError):
            hf.generate([[{"role": "user", "content": "x"}]], 8, schemas=[{}])
        for kwargs in ({"structured": True}, {"gpu_memory_utilization": 0.95}):
            with self.assertRaises(t.TriageError):
                t.make_backend("gemma", **kwargs)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, guard_env(tmp)):
            cfg = build_fixture(tmp)
            backend = FakeBackend()
            t.t1_pass2(cfg, lambda: backend)
            t.t2_pass3(cfg, lambda: backend)
            self.assertTrue(backend.schema_args and all(isinstance(s, list) and s for s in backend.schema_args))
            self.assertTrue(all(len(s) <= t.LLAMA_EXECUTION["request_batch_size"] for s in backend.schema_args))
            reviewer = FakeBackend()
            t.t0_corpus_status(cfg, 0)
            t.t3_candidates(cfg)
            t.t4_packets(cfg)
            t.t4_review_r1(cfg, lambda: reviewer, "gemma")              # R1: no schemas, own backend
            self.assertTrue(reviewer.schema_args and all(s is None for s in reviewer.schema_args))
        with self.assertRaises(t.TriageError):
            t.smoke("gemma", constrained=True)

    def test_constrained_smoke_path_offline(self):
        self.default_root_state = tree_state(t.SMOKE_ROOT_CONSTRAINED)
        captured, backend = {}, CaptureBackend()
        original = t.make_backend

        def fake_make_backend(name, max_num_seqs=8, gpu_memory_utilization=None, structured=False):
            captured.update(name=name, max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization,
                            structured=structured)
            return backend

        t.make_backend = fake_make_backend
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, guard_env(tmp)):
                t.smoke("llama", out_root=Path(tmp), constrained=True)
                seal = json.loads((Path(tmp) / "llama" / "SEAL.json").read_text())
                rows = [json.loads(l) for l in (Path(tmp) / "llama" / "smoke_records.jsonl").read_text().splitlines()]
        finally:
            t.make_backend = original
        guard = seal["code_identity_guard"]
        self.assertEqual({k: guard[k]["observed_sha256"] for k in t.CODE_GUARD_ENV},
                         {k: guard[k]["expected_sha256"] for k in t.CODE_GUARD_ENV})
        self.assertTrue(guard["executed_wrapper"]["path"].endswith("slurm_script"))
        self.assertEqual(captured, {"name": "llama", "max_num_seqs": 4, "gpu_memory_utilization": 0.95, "structured": True})
        self.assertEqual([r["user_sha256"] for r in rows], SMOKE_INPUT_SHA256)
        p2_batch, p3_batch = backend.schema_batches
        self.assertEqual(len(p2_batch), 4)
        self.assertEqual([s["anyOf"][2]["properties"]["proposed_parent_source_titles"]["items"]["enum"] for s in p3_batch],
                         [self.cands, ["Drug overdose", "Grapefruit"]])            # per-request enums, no leakage
        self.assertEqual([r["schema_sha256"] for r in rows],
                         [t.sha256_text(t.canonical(s)) for s in p2_batch + p3_batch])
        self.assertIs(seal["constrained"], True)
        self.assertEqual(seal["structured_outputs"], t.STRUCTURED_OUTPUTS)
        self.assertEqual(seal["structured_outputs_versions"]["xgrammar"], importlib.metadata.version("xgrammar"))
        self.assertEqual(len(seal["schema_sha256"]), 6)
        self.assertEqual(t.SMOKE_ROOT_CONSTRAINED, ROOT / "outputs/efbpt/qea_triage_v1_smoke_v2")
        self.assertEqual(t.SMOKE_ROOT, ROOT / "outputs/efbpt/qea_triage_v1_smoke")            # Gemma root unchanged
        self.assertEqual(tree_state(t.SMOKE_ROOT_CONSTRAINED), self.default_root_state)   # untouched either way

    def test_smoke_inputs_identical_to_job_96136(self):
        with t.stage_reads(*t.SMOKE_HISTORICAL.values()):
            p2u, p3u, ctx, _ru = t.smoke_items()
        self.assertEqual([t.sha256_text(u) for u in p2u + p3u], SMOKE_INPUT_SHA256)
        self.assertEqual([c[0] for c in ctx], [3, 2])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(t.MODELS["llama"]["path"], local_files_only=True)
        counts = [len(list(tok.apply_chat_template(t.messages(s, u, "llama"), tokenize=True, add_generation_prompt=True)))
                  for s, us in ((t.PASS2_SYSTEM, p2u), (t.PASS3_SYSTEM, p3u)) for u in us]
        recorded = JOB_96136_PROMPT_TOKENS
        self.assertEqual(counts, recorded)
        caps = [t.MAX_NEW_TOKENS["pass2"]] * 4 + [t.MAX_NEW_TOKENS["pass3"]] * 2
        self.assertEqual(caps, [256] * 4 + [512] * 2)
        self.assertTrue(all(c + cap <= 4096 for c, cap in zip(counts, caps)))


# ------------------------------------------------------------------ T1/T2 execution (amendment 1)

class PredictionStageTests(Base):
    def test_execution_config_is_the_tested_one_with_unchanged_model_and_prompts(self):
        e = t.LLAMA_EXECUTION
        self.assertEqual((e["max_model_len"], e["gpu_memory_utilization"], e["max_num_seqs"], e["request_batch_size"]),
                         (4096, 0.95, 4, 4))
        self.assertEqual(e["structured_outputs"], {"backend": "xgrammar", "disable_fallback": True,
                                                   "disable_any_whitespace": True})
        self.assertEqual(e["max_new_tokens"], {"pass2": 256, "pass3": 512})
        self.assertEqual(t.LLAMA_EXECUTION_SHA256, t.sha256_text(t.canonical(e)))
        expected = t.llama_execution_expected()                                  # vs job 96151's recorded identity
        self.assertEqual(set(expected) - set(JOB_96151_MODEL), {"dtype"})           # 96151's log: dtype float16
        for key in JOB_96151_MODEL:
            self.assertEqual(JOB_96151_MODEL[key], expected[key], key)
        self.assertEqual(importlib.metadata.version("xgrammar"), "0.1.27")
        with self.assertRaises(SystemExit):
            t.main(["t1", "--batch-size", "8"])                                   # no CLI override of execution

    def test_other_execution_is_refused_before_any_call(self):
        for bad in ({"gpu_memory_utilization": 0.90}, {"structured_outputs": None}, {"max_num_seqs": 8},
                    {"max_model_len": 8192}, {"quantization": "awq_marlin"}):
            backend = FakeBackend(identity=dict(t.llama_execution_expected(), **bad))
            self.raises("LLAMA_EXECUTION_MISMATCH", t.t1_pass2, self.cfg, lambda: backend)
            self.assertEqual(backend.seen, [])
            self.assertFalse(self.cfg.stage("t1").exists())

    def test_schemas_are_validated_before_model_load(self):
        order, original = [], t.validate_schemas
        with mock.patch.object(t, "validate_schemas", lambda s: order.append("validate") or original(s)):
            t.t1_pass2(self.cfg, lambda: order.append("load") or FakeBackend())
            t.t2_pass3(self.cfg, lambda: order.append("load") or FakeBackend())
        self.assertEqual(order, ["validate", "load"] * 2)

    def test_each_request_carries_its_own_schema(self):
        backend = run_to_t4(self.cfg)
        calls = [c for c in backend.calls if c[0] != "review"]
        for kind, user, schema, _text in calls:
            if kind == "pass2":
                self.assertEqual(schema, t.pass2_schema())
            else:
                n = len(user["official_decomposition"])
                self.assertEqual(t.pass3_schema_candidates(schema), user["candidate_parent_titles"])
                self.assertEqual(schema, t.pass3_schema(n, user["candidate_parent_titles"], user["gold_title"]))
                self.assertTrue(all(b["properties"]["official_step_indices"]["items"]["enum"] == list(range(n))
                                    for b in schema["anyOf"]))
        self.assertEqual([len(b) for b in backend.schema_args[:6]], [4, 4, 4, 2, 4, 3])   # 14 T1, 7 T2 requests
        by_user = {t.sha256_text(t.canonical(u)): t.sha256_text(t.canonical(s)) for _k, u, s, _x in calls}
        for stage, name in (("t1", "pass2_records.jsonl"), ("t2", "pass3_records.jsonl")):
            for rec in t.read_jsonl(self.cfg.stage(stage) / name):
                self.assertEqual(rec["schema_sha256"], by_user[rec["user_sha256"]])

    def test_negative_and_uncertain_branches_are_valid_and_schema_admissible(self):
        backend = FakeBackend(pass3_override={
            "Gamma One": p3_obj(decision="LATENT_BRIDGE", concrete_intermediate_information="Some fact.",
                                dependency_status="UNRESOLVED", dependency_confidence="LOW"),
            "Gamma Two": p3_obj(),
            "Gamma Three": p3_obj(decision="LATENT_BRIDGE", concrete_intermediate_information="Parallel use.",
                                  dependency_status="PARALLEL_OR_UNORDERED", dependency_confidence="MEDIUM")})
        run_to_t4(self.cfg, backend)
        for kind, _user, schema, text in backend.calls:
            if kind != "review":
                self.assertTrue(accepts(schema, text), text)
        rows = {r["urbench_qid"]: r for r in t.read_jsonl(self.cfg.stage("t3") / "question_accounting.jsonl")}
        q1, q2, q3, q4 = (rows[q] for q in QIDS[:4])
        self.assertEqual(q1["t2"]["valid_decisions"], {"LATENT_BRIDGE/CLEAR_DEPENDENCY": 1, "LATENT_BRIDGE/UNRESOLVED": 1})
        self.assertEqual(q2["t2"]["valid_decisions"], {"AMBIGUOUS": 1, "LATENT_BRIDGE/CLEAR_DEPENDENCY": 1})
        self.assertEqual(q3["t2"]["valid_decisions"], {"LATENT_BRIDGE/CLEAR_DEPENDENCY": 1,
                                                      "LATENT_BRIDGE/PARALLEL_OR_UNORDERED": 1})
        for r in (q1, q2, q3):
            self.assertEqual(r["t2"]["outcomes"]["valid"], 2)
            self.assertNotIn("T2_ANNOTATION_FAILURE", r["flags"])
        # Q4: two valid NOT_YET_EXPLICIT decisions are negatives, not failures; Pass 3 is not eligible.
        self.assertEqual(q4["t1"]["outcomes"], dict(dict.fromkeys(t.CALL_OUTCOMES, 0), valid=2))
        self.assertEqual((q4["t1"]["valid_not_yet_explicit"], q4["flags"], q4["outcome"]),
                         (2, ["NO_CANDIDATE_PAIR", "NO_VALID_EXPLICIT_TITLE"], "STOP_T1_NO_VALID_EXPLICIT_TITLE"))
        pairs = {(p["parent_title"], p["child_title"]) for p in t.read_jsonl(self.cfg.stage("t3") / "candidate_pairs.jsonl")}
        self.assertEqual(pairs, {("Alpha One", "Beta One"), ("Alpha Two", "Beta Two"), ("Alpha Three", "Beta Three")})

    def test_t3_reparses_predictions_independently_before_corpus_status(self):
        backend = FakeBackend()
        t.t0_corpus_status(self.cfg, 0)
        t.t1_pass2(self.cfg, lambda: backend)
        t.t2_pass3(self.cfg, lambda: backend)

        def tamper(stage, name, edit):
            path = self.cfg.stage(stage) / name
            recs = t.read_jsonl(path)
            edit(recs[0])
            raw = "".join(t.canonical(r) + "\n" for r in recs).encode()
            path.write_bytes(raw)
            seal = json.loads((self.cfg.stage(stage) / "SEAL.json").read_text())
            seal["outputs"][name] = {"sha256": sha(raw), "bytes": len(raw)}         # consistently resealed
            (self.cfg.stage(stage) / "SEAL.json").write_text(json.dumps(seal))
            return sha((self.cfg.stage(stage) / "SEAL.json").read_bytes())

        tamper("t2", "pass3_records.jsonl", lambda r: r["parsed"].update(rationale="edited"))
        t.READ_LOG.clear()
        self.raises("PASS3_REPARSE_MISMATCH", t.t3_candidates, self.cfg)
        self.assertFalse([p for p in t.READ_LOG if "/t0_corpus_status/" in p])
        t1_sha = tamper("t1", "pass2_records.jsonl", lambda r: r.update(schema_sha256="0" * 64))
        self.raises("T2_NOT_BUILT_ON_THIS_T1", t.t3_candidates, self.cfg)          # the T1 seal changed
        t2_seal = json.loads((self.cfg.stage("t2") / "SEAL.json").read_text())
        t2_seal["t1_seal_sha256"] = t1_sha
        (self.cfg.stage("t2") / "SEAL.json").write_text(json.dumps(t2_seal))
        self.raises("PASS2_REQUEST_IDENTITY", t.t3_candidates, self.cfg)

    def test_stages_refuse_predictions_from_another_execution(self):
        backend = FakeBackend()
        t.t1_pass2(self.cfg, lambda: backend)
        seal_path = self.cfg.stage("t1") / "SEAL.json"
        seal = json.loads(seal_path.read_text())
        seal["execution"]["sha256"] = "0" * 64
        seal_path.write_text(json.dumps(seal))
        loads = []
        self.raises("T1_EXECUTION_CONFIG_MISMATCH", t.t2_pass3, self.cfg, lambda: loads.append(1))
        self.assertEqual(loads, [])


# ------------------------------------------------------------------ per-question accounting

class AccountingTests(Base):
    def acc(self, stage):
        return {r["urbench_qid"]: r for r in t.read_jsonl(self.cfg.stage(stage) / "question_accounting.jsonl")}

    def finish_reviews(self):
        t.t4_review_r1(self.cfg, FakeBackend, "gemma")
        r2 = Path(self.tmp.name) / "r2.jsonl"
        write_r2(self.cfg, r2, {})
        t.t4_ingest_r2(self.cfg, r2, "test-reviewer")
        t.t5_accept(self.cfg)

    def test_one_row_per_qid_and_stage_not_run_is_not_zero(self):
        backend = FakeBackend()
        t.t0_corpus_status(self.cfg, 0)
        t.t1_pass2(self.cfg, lambda: backend)
        rows = t.read_jsonl(self.cfg.stage("t1") / "question_accounting.jsonl")
        self.assertEqual([r["urbench_qid"] for r in rows], QIDS)
        for r, (titles, *_rest) in zip(rows, PLANS):
            self.assertEqual((r["source_instances"], r["t1"]["calls_expected"], r["t1"]["calls_recorded"]), (len(titles),) * 3)
            self.assertEqual((r["t2"], r["proposals"], r["corpus"]), ({"stage": "NOT_RUN"}, "NOT_RUN", {"stage": "NOT_JOINED"}))
            for k in ("t3", "review", "t5"):
                self.assertEqual(r[k], {"stage": "NOT_RUN"})
            self.assertTrue(all(x["t2"] == "NOT_RUN" and x["corpus"] == "NOT_JOINED" for x in r["titles"]))
        self.assertEqual([r["outcome"] for r in rows], ["PENDING_T2"] * 3 + ["STOP_T1_NO_VALID_EXPLICIT_TITLE", "PENDING_T2"])
        t.t2_pass3(self.cfg, lambda: backend)
        rows = self.acc("t2")
        self.assertEqual(rows[QIDS[3]]["t2"], {"stage": "RUN", "calls_expected": 0, "calls_recorded": 0,   # zero, run
                                               "outcomes": dict.fromkeys(t.CALL_OUTCOMES, 0), "valid_decisions": {}})
        self.assertEqual(rows[QIDS[3]]["proposals"], [])
        self.assertEqual({p["status"] for p in rows[QIDS[0]]["proposals"]}, {"PENDING_CORPUS_STATUS"})
        t.t3_candidates(self.cfg)
        t.t4_packets(self.cfg)
        self.finish_reviews()
        rows = self.acc("t5")
        self.assertEqual([rows[q]["outcome"] for q in QIDS],
                         ["SELECTED", "SELECTED", "STOP_T5_MULTIPLE_ACCEPTED_PARENTS", "STOP_T1_NO_VALID_EXPLICIT_TITLE",
                          "STOP_T3_NO_CANDIDATE_PAIR"])
        q5 = rows[QIDS[4]]
        self.assertEqual((q5["proposals"][0]["status"], q5["proposals"][0]["exclusions"]), ("EXCLUDED", ["CHILD_EXACT_ABSENT"]))
        self.assertEqual(q5["flags"], ["NO_CANDIDATE_PAIR", "PROPOSAL_CHILD_EXACT_ABSENT"])
        self.assertEqual((q5["t5"]["question_status"], q5["review"]["packets"]), ("NO_CANDIDATE_PAIR", 0))
        self.assertEqual(rows[QIDS[1]]["review"], {"stage": "RUN", "packets": 2, "r1": {"valid": 2, "malformed": 0, "missing": 0},
                                                   "r2": {"valid": 2, "malformed": 0, "missing": 0}})
        summary = json.loads((self.cfg.stage("t5") / "SEAL.json").read_text())["question_accounting"]
        self.assertEqual(sum(summary["outcome_counts_disjoint"].values()), len(QIDS))
        self.assertEqual(summary["t1_calls"], {"calls_expected": 14, "outcomes": dict(dict.fromkeys(t.CALL_OUTCOMES, 0), valid=14)})
        self.assertEqual(summary["t2_calls"]["calls_expected"], 7)
        flagged = sum(bool(r["flags"]) for r in rows.values())
        self.assertGreater(sum(summary["flag_question_counts_overlapping"].values()), flagged)   # flags overlap
        self.assertIn("not additive", summary["flag_note"])

    def test_valid_child_with_non_explicit_parent_stays_excluded_with_visible_reason(self):
        # The Sound-barrier shape: the parent is a valid NOT_YET_EXPLICIT decision and a valid child names it.
        run_to_t4(self.cfg, FakeBackend(pass2_override={"Delta Three": P2_NOT_YET}))
        pairs = {(p["parent_title"], p["child_title"]) for p in t.read_jsonl(self.cfg.stage("t3") / "candidate_pairs.jsonl")}
        self.assertNotIn(("Delta Three", "Gamma Three"), pairs)
        self.assertIn(("Alpha Three", "Beta Three"), pairs)
        q3 = self.acc("t3")[QIDS[2]]
        prop = next(p for p in q3["proposals"] if p["child_title"] == "Gamma Three")
        self.assertEqual((prop["parent_title"], prop["parent_t1"], prop["status"], prop["exclusions"]),
                         ("Delta Three", "NOT_YET_EXPLICIT", "EXCLUDED", ["PARENT_T1_NOT_YET_EXPLICIT"]))
        self.assertIn("PROPOSAL_PARENT_T1_NOT_YET_EXPLICIT", q3["flags"])
        self.assertNotIn("T1_ANNOTATION_FAILURE", q3["flags"])
        self.assertEqual((q3["t1"]["outcomes"]["valid"], q3["outcome"]), (4, "PENDING_REVIEW_AND_ACCEPTANCE"))
        self.assertEqual({x["gold_title"]: x["t2"] for x in q3["titles"]}["Delta Three"], "AMBIGUOUS")

    def test_malformed_length_prompt_and_missing_parent_labels(self):
        run_to_t4(self.cfg, FakeBackend(pass2_override={"Delta Three": "not json"}))
        q3 = self.acc("t3")[QIDS[2]]
        prop = next(p for p in q3["proposals"] if p["child_title"] == "Gamma Three")
        self.assertEqual((prop["parent_t1"], prop["exclusions"]), ("FAILED_MALFORMED", ["PARENT_T1_ANNOTATION_MALFORMED"]))
        self.assertEqual((q3["t1"]["outcomes"]["malformed"], q3["t3"]["candidate_pairs"]), (1, 1))
        self.assertIn("T1_ANNOTATION_FAILURE", q3["flags"])
        prep = t.verify_preparation(self.cfg)
        p2 = t.read_jsonl(self.cfg.stage("t1") / "pass2_records.jsonl")
        p3 = t.read_jsonl(self.cfg.stage("t2") / "pass3_records.jsonl")
        corpus = t.read_jsonl(self.cfg.stage("t0") / "corpus_status.jsonl")
        delta = next(i for i, r in enumerate(p2) if r["gold_title"] == "Delta Three")
        variants = {"LENGTH_REJECTED": dict(p2[delta], finish_reason="length", malformed_reason="FINISH_LENGTH"),
                    "PROMPT_TOO_LONG": dict(p2[delta], finish_reason="prompt_too_long",
                                            malformed_reason="FINISH_PROMPT_TOO_LONG"),
                    "MISSING": None}
        for code, rec in variants.items():
            recs = [r for i, r in enumerate(p2) if i != delta] + ([rec] if rec else [])
            pairs = t.build_candidates(prep["master"], recs, p3, corpus)
            q3 = {r["urbench_qid"]: r for r in t.question_accounting(prep, recs, p3, corpus, pairs)}[QIDS[2]]
            prop = next(p for p in q3["proposals"] if p["child_title"] == "Gamma Three")
            self.assertEqual(prop["exclusions"], ["PARENT_T1_ANNOTATION_" + code])
            self.assertEqual((q3["t1"]["outcomes"][code.lower()], q3["t1"]["outcomes"]["valid"]), (1, 3))
            self.assertEqual((q3["t3"]["candidate_pairs"], q3["outcome"]), (1, "PENDING_REVIEW_AND_ACCEPTANCE"))
            self.assertEqual(q3["t1"]["calls_recorded"], 3 if code == "MISSING" else 4)

    def test_partially_malformed_questions_keep_valid_pairs_through_t5(self):
        run_to_t4(self.cfg, FakeBackend(pass2_override={"Gamma Two": "not json"},
                                        pass3_override={"Gamma One": '{"decision": '},
                                        finish_override={("pass3", "Gamma Three"): "length"}))
        self.finish_reviews()
        rows = self.acc("t5")
        q1, q2, q3 = rows[QIDS[0]], rows[QIDS[1]], rows[QIDS[2]]
        self.assertEqual((q2["outcome"], q2["t1"]["outcomes"]["malformed"]), ("SELECTED", 1))
        self.assertIn("T1_ANNOTATION_FAILURE", q2["flags"])
        self.assertEqual({x["gold_title"]: (x["t1"], x["t2"]) for x in q2["titles"]}["Gamma Two"],
                         ("FAILED_MALFORMED", "NOT_IN_PASS3_SCOPE"))
        self.assertEqual((q1["outcome"], q1["t2"]["outcomes"]["malformed"]), ("SELECTED", 1))
        self.assertEqual(q3["t2"]["outcomes"]["length_rejected"], 1)                  # distinct from malformed
        self.assertEqual((q3["outcome"], q3["t5"]["accepted_pairs"]), ("SELECTED", 1))
        sample = {s["urbench_qid"]: [c["title"] for c in s["children"]]
                  for s in t.read_jsonl(self.cfg.stage("t5") / "development_sample.jsonl")}
        self.assertEqual(sample, {QIDS[0]: ["Beta One"], QIDS[1]: ["Beta Two"], QIDS[2]: ["Beta Three"]})

    def test_accounting_fails_closed_on_join_or_scope_disagreement(self):
        run_to_t4(self.cfg)
        prep = t.verify_preparation(self.cfg)
        p2 = t.read_jsonl(self.cfg.stage("t1") / "pass2_records.jsonl")
        p3 = t.read_jsonl(self.cfg.stage("t2") / "pass3_records.jsonl")
        corpus = t.read_jsonl(self.cfg.stage("t0") / "corpus_status.jsonl")
        pairs = t.read_jsonl(self.cfg.stage("t3") / "candidate_pairs.jsonl")
        self.raises("ACCOUNTING_JOIN_DISAGREEMENT", t.question_accounting, prep, p2, p3, corpus, pairs[1:])
        self.raises("ACCOUNTING_JOIN_DISAGREEMENT", t.question_accounting, prep, p2, p3, corpus, pairs + pairs[:1])
        self.raises("PASS3_RECORDS_NOT_EQUAL_TO_SCOPE", t.question_accounting, prep, p2, p3[1:])


# ------------------------------------------------------------------ smoke artifacts and the Gemma smoke

class SmokeArtifactTests(Base):
    def test_smoke_artifacts_are_refused_as_annotations_by_path_and_content(self):
        for d in (t.SMOKE_ROOT / "llama", t.SMOKE_ROOT_CONSTRAINED / "llama", t.SMOKE_ROOT / "gemma",
                  t.SMOKE_ROOT_GEMMA / "gemma"):
            for stage in ("t1", "t2", "smoke"):
                self.raises("SMOKE_OUTPUT_IS_NOT_ANNOTATION", t.load_sealed, d, stage)
        synthetic = Path(self.tmp.name) / "synthetic_smoke_out"
        with mock.patch.object(t, "make_backend", capture_make_backend({}, CaptureBackend())):
            t.smoke("llama", out_root=synthetic)
        self.cfg.triage_root.mkdir(parents=True)
        shutil.copytree(synthetic / "llama", self.cfg.stage("t1"))                      # copied to a non-smoke path
        self.raises("SMOKE_OUTPUT_IS_NOT_ANNOTATION", t.load_sealed, self.cfg.stage("t1"), "t1")
        loads = []
        self.raises("SMOKE_OUTPUT_IS_NOT_ANNOTATION", t.t2_pass3, self.cfg, lambda: loads.append(1))
        self.assertEqual(loads, [])
        pinned = {t.SMOKE_ROOT / "llama/SEAL.json": JOB_96136_SEAL_SHA256,
                  t.SMOKE_ROOT / "llama/smoke_records.jsonl": JOB_96136_RECORDS_SHA256,
                  t.SMOKE_ROOT_CONSTRAINED / "llama/SEAL.json": t.JOB_96151_SEAL_SHA256,
                  t.SMOKE_ROOT_CONSTRAINED / "llama/smoke_records.jsonl": JOB_96151_RECORDS_SHA256,
                  t.SMOKE_ROOT / "gemma/SEAL.json": JOB_96317_SEAL_SHA256,
                  t.SMOKE_ROOT / "gemma/smoke_records.jsonl": JOB_96317_RECORDS_SHA256}
        for path, digest in pinned.items():                                    # only if present; never required
            if path.exists():
                self.assertEqual(sha(path.read_bytes()), digest, path)


def tree_state(root):
    """Existence and file hashes under root; equal before and after a test whether or not root exists."""
    root = Path(root)
    if not root.exists():
        return None
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def _string_leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_leaves(v)


class GemmaSmokeTests(unittest.TestCase):
    def test_two_review_calls_on_its_own_backend(self):
        captured, backend = {}, CaptureBackend()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, guard_env(tmp)), \
                mock.patch.object(t, "make_backend", capture_make_backend(captured, backend)):
            root = Path(tmp) / "smoke_gemma"
            t.smoke("gemma", out_root=root)
            seal = json.loads((root / "gemma" / "SEAL.json").read_text())
            rows = [json.loads(l) for l in (root / "gemma" / "smoke_records.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["gemma"])
            with self.assertRaises(t.TriageError):                                   # an existing root refuses
                t.smoke("gemma", out_root=root)
        self.assertEqual(captured, {"name": "gemma", "max_num_seqs": 4, "gpu_memory_utilization": None,
                                    "structured": False})
        self.assertEqual(backend.calls, [(t.REVIEW_SYSTEM, 256)] * 2)               # at most 512 new tokens
        self.assertEqual(backend.schema_batches, [None])
        self.assertEqual([r["user_sha256"] for r in rows], GEMMA_SMOKE_REVIEW_SHA256)
        self.assertEqual((seal["calls"], seal["max_new_tokens"], seal["prompt_sha256"]),
                         (2, {"review": 256}, {"review": t.PINNED_PROMPT_SHA256["review"]}))
        self.assertEqual((seal["constrained"], seal["structured_outputs"], seal["smoke_overrides"]), (False, None, {}))
        self.assertEqual(seal["code_identity_guard"]["runner"]["observed_sha256"], sha(Path(t.__file__).read_bytes()))
        self.assertEqual(t.SMOKE_ROOT_GEMMA, ROOT / "outputs/efbpt/qea_triage_v1_smoke_gemma_v2")   # 96317 kept apart
        self.assertEqual([r["diagnostics"] for r in rows], [{"stub": 0}, {"stub": 1}])

    def test_review_packets_hide_verdicts_rationales_status_and_other_reviews(self):
        with t.stage_reads(*t.SMOKE_HISTORICAL.values()):
            _p2u, _p3u, _ctx, reviews = t.smoke_items()
            verified = sorted((r for r in t.read_jsonl(t.SMOKE_HISTORICAL["verified"]) if r["verdict"] == "VERIFIED_CLEAN_BRIDGE"),
                              key=lambda r: (r["qid"], r["parent_source_instance_id"], r["child_source_instance_id"]))[:2]
            hp2 = {r["source_instance_id"]: r for r in t.read_jsonl(t.SMOKE_HISTORICAL["pass2"])}
            hp3 = {r["source_instance_id"]: r for r in t.read_jsonl(t.SMOKE_HISTORICAL["pass3"])}
        self.assertEqual([t.sha256_text(u) for u in reviews], GEMMA_SMOKE_REVIEW_SHA256)
        allowed = {"question_ur", "target_english_title", "official_decomposition", "official_step_indices",
                   "concrete_intermediate_information", "official_target_evidence_support", "urbench_qid", "qid",
                   "source_instance_id", "parent_source_instance_id", "child_source_instance_id", "parent_title",
                   "child_title", "proposed_parent_source_titles"}
        for user, v in zip(reviews, verified):
            packet = json.loads(user)
            self.assertEqual(set(packet), set(t.PACKET_FIELDS) - {"packet_id"})
            self.assertFalse(t.PACKET_FORBIDDEN & set(packet))
            self.assertEqual((packet["parent_title"], packet["child_title"]), (v["parent_title"], v["child_title"]))
            for token in ("VERIFIED", "CLEAN_BRIDGE", "EXPLICIT", "LATENT_BRIDGE", "AMBIGUOUS", "DEPENDENCY",
                          "EXACT_PRESENT", "EXACT_ABSENT", "confidence", "rationale", "verdict", "human_answers",
                          "identifiable", '"q1"', '"q4"', "R1", "R2", "annotator_id"):
                self.assertNotIn(token, user)
            hidden = [hp2[v["parent_source_instance_id"]], hp2[v["child_source_instance_id"]],
                      hp3[v["child_source_instance_id"]], v]
            for rec in hidden:
                for key, value in rec.items():
                    if key in allowed:
                        continue
                    for leaf in _string_leaves(value):
                        if len(leaf) >= 4:
                            self.assertNotIn(leaf, user, (key, leaf))


# ------------------------------------------------------------------ Gemma backend (amendment 2)

def gemma_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(t.MODELS["gemma"]["path"], local_files_only=True)      # tokenizer only


def fake_gemma(script, dtype_name="bfloat16", nan_steps=(), mask_steps=(), prefix_shift=False):
    """CPU stand-in for the loaded Gemma-2 model: real local generation config, scripted continuation, and the
    stopping behaviour of transformers generate (stop when the last token is in eos_token_id, or at
    max_new_tokens). It calls itself once per generated token (so forward hooks fire), optionally applies a
    masking step before the given logits processors, and checks they hand back the same tensor."""
    import torch
    from types import SimpleNamespace
    from transformers import GenerationConfig
    dtype = getattr(torch, dtype_name)

    class Linear4bit(torch.nn.Module):                       # attribute names read by hf_dtype_census
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.uint8), requires_grad=False)
            self.weight.quant_state = SimpleNamespace(dtype=torch.bfloat16)
            self.compute_dtype = torch.bfloat16

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(16, 4, dtype=dtype)
            self.model.norm = torch.nn.Module()
            self.model.norm.weight = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
            self.model.proj = Linear4bit()
            self.lm_head = torch.nn.Linear(4, 16, bias=False, dtype=dtype)
            self.lm_head.weight = self.model.embed_tokens.weight
            self.generation_config = GenerationConfig.from_pretrained(t.MODELS["gemma"]["path"], local_files_only=True)
            self.config = SimpleNamespace(_attn_implementation="eager")
            self.calls = []

        dtype = property(lambda self: self.model.embed_tokens.weight.dtype)
        device = property(lambda self: torch.device("cpu"))

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, step):
            logits = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
            if step in nan_steps:
                logits[0, 0, 0] = float("nan")
            return SimpleNamespace(logits=logits)

        def generate(self, ids, max_new_tokens, do_sample, eos_token_id, logits_processor):
            self.calls.append({"max_new_tokens": max_new_tokens, "do_sample": do_sample, "eos_token_id": eos_token_id})
            seq = ids[0].tolist()
            for step, tok in enumerate(script[:max_new_tokens]):
                scores = self(step).logits[:, -1, :].to(dtype=torch.float32)
                if step in mask_steps:
                    scores[0, 5] = float("-inf")                                     # a processor's masking
                assert logits_processor(torch.tensor([seq]), scores) is scores
                seq.append(tok)
                if tok in eos_token_id:
                    break
            if prefix_shift:
                seq[0] += 1
            return torch.tensor([seq])

    return Model()


def gemma_backend(model, tok):
    backend = object.__new__(t.HfBnbBackend)
    import torch
    backend.torch, backend.tokenizer, backend.model = torch, tok, model
    backend.stop = t.gemma_stop_ids(tok, model.generation_config)
    return backend


REVIEW_JSON = '{"q1": "Y", "q2": "Y", "q3": "N", "q4": "O", "note": "The step is only loosely linked."}'


class GemmaBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = gemma_tokenizer()
        cls.msg = [[{"role": "user", "content": t.REVIEW_SYSTEM + "\n\n" + '{"x": 1}'}]]

    def run_one(self, script, cap=256, **kw):
        model = fake_gemma(script, **kw)
        out = gemma_backend(model, self.tok).generate(self.msg, cap)[0]
        return out, model

    def test_verified_token_ids_and_stop_set(self):
        from transformers import GenerationConfig
        tok = self.tok
        self.assertEqual([(tok.bos_token_id, tok.bos_token), (tok.eos_token_id, tok.eos_token), (tok.pad_token_id, tok.pad_token)],
                         [(2, "<bos>"), (1, "<eos>"), (0, "<pad>")])
        self.assertEqual([tok.convert_tokens_to_ids(s) for s in ("<start_of_turn>", "<end_of_turn>")], [106, 107])
        self.assertEqual(t.gemma_stop_ids(tok, GenerationConfig(eos_token_id=1)), [[1, "<eos>"], [107, "<end_of_turn>"]])
        self.assertEqual(t.gemma_stop_ids(tok, GenerationConfig(eos_token_id=[1, 107])), [[1, "<eos>"], [107, "<end_of_turn>"]])
        with self.assertRaises(t.TriageError):
            t.gemma_stop_ids(tok, GenerationConfig(eos_token_id=5))                      # existing EOS must be <eos>

        class NoTurnTok:
            eos_token_id, unk_token_id = 1, 3
            convert_ids_to_tokens = staticmethod(lambda i: {1: "<eos>", 3: "<unk>"}[i])
            convert_tokens_to_ids = staticmethod(lambda s: 3)
        with self.assertRaises(t.TriageError):
            t.gemma_stop_ids(NoTurnTok(), GenerationConfig(eos_token_id=1))

    def test_load_contract_bf16_nf4_double_quant_bf16_compute(self):
        import torch
        import transformers
        captured = {}

        def load(dtype_name):
            def fake_from_pretrained(path, **kwargs):
                captured.update(kwargs, path=path)
                return fake_gemma([], dtype_name=dtype_name)
            with mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained):
                return t.HfBnbBackend()

        backend = load("bfloat16")
        q = captured["quantization_config"]
        self.assertEqual((captured["dtype"], q.load_in_4bit, q.bnb_4bit_quant_type, q.bnb_4bit_use_double_quant,
                          q.bnb_4bit_compute_dtype), (torch.bfloat16, True, "nf4", True, torch.bfloat16))
        self.assertEqual((captured["attn_implementation"], captured["device_map"], captured["local_files_only"],
                          captured["trust_remote_code"]), ("eager", {"": 0}, True, False))
        self.assertNotIn("torch_dtype", captured)
        self.assertEqual(backend.stop, [[1, "<eos>"], [107, "<end_of_turn>"]])
        census = backend.census
        self.assertEqual((census["embed_tokens"], census["lm_head"], census["final_norm"], census["linear4bit_compute_dtype"],
                          census["linear4bit_weight_storage"], census["lm_head_tied_to_embed_tokens"]),
                         ("torch.bfloat16",) * 4 + ("torch.uint8", True))
        ident = backend.identity()
        self.assertEqual((ident["load_dtype"], ident["generate_call"]["eos_token_id"], ident["stop_tokens"],
                          ident["attn_implementation"], ident["decoding"]),
                         ("torch.bfloat16", [1, 107], [[1, "<eos>"], [107, "<end_of_turn>"]], "eager", "greedy do_sample=False"))
        self.assertNotIn("generation_config_file", ident)                                 # renamed (amendment 3)
        self.assertEqual((ident["generation_config_at_load"]["eos_token_id"],
                          ident["generation_config_at_load"]["cache_implementation"]), (1, "hybrid"))   # file values
        self.assertIn("after generation", ident["generation_config_note"])
        with self.assertRaises(t.TriageError) as ctx:
            load("float16")                                                              # the pre-amendment state
        self.assertTrue(str(ctx.exception).startswith("GEMMA_EFFECTIVE_DTYPE"))

    def test_finish_classification_follows_the_actual_stop_event(self):
        stop = [1, 107]
        self.assertEqual(t.classify_finish([5, 6, 107], stop, 256), "stop")               # end of turn
        self.assertEqual(t.classify_finish([5, 6, 1], stop, 256), "stop")                 # eos
        self.assertEqual(t.classify_finish([5] * 255 + [1], stop, 256), "stop")           # eos on the final allowed token
        self.assertEqual(t.classify_finish([0] * 256, stop, 256), "length")               # genuine exhaustion
        self.assertEqual(t.classify_finish([5, 6], stop, 256), "unexplained_stop")        # short, no stop token
        self.assertEqual(t.classify_finish([5, 1, 6, 1], stop, 256), "unexplained_stop")  # stop token before the end
        self.assertEqual(t.classify_finish([], stop, 256), "unexplained_stop")
        # Classification uses the set generation stopped on: when only <eos> stops generation, a final
        # <end_of_turn> at the cap is length exhaustion (the pre-amendment code called it a stop).
        self.assertEqual(t.classify_finish([5] * 255 + [107], [1], 256), "length")
        self.assertEqual(t.parse_review(REVIEW_JSON, "unexplained_stop")[1], "FINISH_UNEXPLAINED_STOP")

    def test_end_of_turn_stop_slicing_decoding_and_evidence(self):
        body = self.tok(REVIEW_JSON, add_special_tokens=False)["input_ids"]
        out, model = self.run_one(body + [107, 1, 1])                                    # stops at <end_of_turn>
        d = out["diagnostics"]
        prompt = self.tok.apply_chat_template(self.msg[0], add_generation_prompt=True)
        self.assertEqual(model.calls, [{"max_new_tokens": 256, "do_sample": False, "eos_token_id": [1, 107]}])
        self.assertEqual((out["finish_reason"], out["prompt_tokens"], out["output_tokens"]), ("stop", len(prompt), len(body) + 1))
        self.assertEqual((d["generated_token_ids"], d["generated_token_count"], d["input_width"]),
                         (body + [107], len(body) + 1, len(prompt)))
        self.assertEqual(d["input_token_sha256"], t.sha256_text(t.canonical(prompt)))
        self.assertEqual((d["text_special_removed"], d["text_special_retained"], d["text_after_cleanup"]),
                         (REVIEW_JSON, REVIEW_JSON + "<end_of_turn>", REVIEW_JSON))
        self.assertEqual(out["text"], REVIEW_JSON)
        self.assertEqual((d["termination"]["last_token_id"], d["termination"]["last_token"], d["termination"]["reached_new_token_cap"]),
                         (107, "<end_of_turn>", False))
        self.assertEqual((d["numerics"]["raw_logits"]["steps"], d["numerics"]["processed_scores"]["steps"]), (len(body) + 1,) * 2)
        self.assertFalse(d["numerics"]["raw_numerical_failure"] or d["numerics"]["processed_numerical_failure"])
        parsed, err = t.parse_review(out["text"], out["finish_reason"])
        self.assertEqual((err, parsed["q3"], parsed["q4"]), (None, "N", "O"))              # no forced labels
        json.dumps(d)                                                                     # serializable as recorded

    def test_length_exhaustion_early_empty_stop_eos_at_cap_and_fence_cleanup(self):
        out, _m = self.run_one([0] * 300)                                                 # <pad> until the cap
        d = out["diagnostics"]
        self.assertEqual((out["finish_reason"], out["output_tokens"], out["text"]), ("length", 256, ""))
        self.assertEqual((d["text_special_retained"], d["token_frequency"]["top"], d["token_frequency"]["special_token_count"]),
                         ("<pad>" * 256, [{"id": 0, "token": "<pad>", "count": 256}], 256))
        self.assertTrue(d["termination"]["reached_new_token_cap"])
        self.assertEqual(t.parse_review(out["text"], out["finish_reason"])[1], "FINISH_LENGTH")
        out, _m = self.run_one([0, 0, 0, 1])                                              # empty, stopped early
        self.assertEqual((out["finish_reason"], out["text"], out["output_tokens"]), ("stop", "", 4))
        self.assertEqual(t.parse_review(out["text"], out["finish_reason"])[1], "NOT_JSON")    # still a failed review
        filler = self.tok("\n", add_special_tokens=False)["input_ids"]
        out, _m = self.run_one(filler * 255 + [1] + [0] * 5)                              # <eos> on token 256
        self.assertEqual((out["finish_reason"], out["output_tokens"]), ("stop", 256))
        self.assertTrue(out["diagnostics"]["termination"]["reached_new_token_cap"])
        fenced = "```json\n" + REVIEW_JSON + "\n```"
        out, _m = self.run_one(self.tok(fenced, add_special_tokens=False)["input_ids"] + [107])
        d = out["diagnostics"]
        self.assertEqual((d["text_special_removed"], d["text_after_cleanup"]), (fenced, REVIEW_JSON))
        self.assertIsNone(t.parse_review(out["text"], out["finish_reason"])[1])

    def test_numerical_observers_count_without_changing_scores(self):
        import torch
        obs = t.NonFiniteObserver()
        x = torch.tensor([[0.5, float("nan"), float("inf"), float("-inf"), -2.0]])
        before = x.clone()
        self.assertIs(obs(None, x), x)
        self.assertTrue(torch.equal(x.nan_to_num(), before.nan_to_num()) and bool(x.isnan()[0, 1]))
        self.assertIsNone(obs.hook(None, (), type("O", (), {"logits": x[:, None, :]})()))
        s = obs.summary()
        self.assertEqual((s["steps"], s["entries_nan"], s["entries_posinf"], s["entries_neginf"], s["first_nonfinite_step"],
                          s["finite_min"], s["finite_max"]), (2, 2, 2, 2, 0, -2.0, 0.5))
        out, _m = self.run_one([5, 6, 7, 107], nan_steps={2})                            # NaN in raw logits
        n = out["diagnostics"]["numerics"]
        self.assertTrue(n["raw_numerical_failure"] and n["processed_numerical_failure"])
        self.assertEqual((n["raw_logits"]["first_nonfinite_step"], n["raw_logits"]["steps_with_nan"]), (2, 1))
        out, _m = self.run_one([5, 6, 7, 107], mask_steps={0, 1, 2, 3})                  # masking only
        n = out["diagnostics"]["numerics"]
        self.assertFalse(n["raw_numerical_failure"] or n["processed_numerical_failure"])
        self.assertEqual((n["processed_masked_entries"], n["raw_logits"]["entries_neginf"]), (4, 0))

    def test_prefix_mismatch_and_prompt_limit_fail_closed(self):
        with self.assertRaises(t.TriageError) as ctx:
            self.run_one([5, 107], prefix_shift=True)
        self.assertTrue(str(ctx.exception).startswith("GENERATE_PROMPT_PREFIX_MISMATCH"))
        model = fake_gemma([5, 107])
        out = gemma_backend(model, self.tok).generate(self.msg, 8192)[0]
        self.assertEqual((out["finish_reason"], model.calls), ("prompt_too_long", []))


# ------------------------------------------------------------------ review calibration (amendment 3)

FREEZE_DOC = ROOT / "docs/EFBPT_STAGE0_SOURCE_ROLE_ATTAINABILITY_FREEZE.md"
SMOKE_CASE_TERMS = ("sound", "barrier", "grapefruit", "Audi", "صوتی", "رکاوٹ", "چکوترا")


def freeze_definition():
    lines = FREEZE_DOC.read_text().splitlines()
    start = lines.index("### 6.1 `EXPLICIT`")
    return next(l[2:] for l in lines[start:] if l.startswith("> "))


class ReviewPromptTests(unittest.TestCase):
    def test_v1_frozen_and_v2_pinned(self):
        self.assertEqual(t.PROMPT_SHA256["review"], "5c656e40dee28fcb44b2787662c5f33f7332cc8a625f7aee24d80f6a0f9399cf")
        self.assertEqual(t.PROMPT_SHA256["review_v2"], t.PINNED_PROMPT_SHA256["review_v2"])
        self.assertEqual(t.REVIEW_PROMPTS, {"v1": t.REVIEW_SYSTEM, "v2": t.REVIEW_SYSTEM_V2})

    def test_v2_quotes_the_frozen_rule_and_keeps_v1_sentences(self):
        v2 = t.REVIEW_SYSTEM_V2
        self.assertIn('"' + freeze_definition() + '"', v2)                            # verbatim, quoted
        self.assertEqual(v2.count("reasonably"), 1)                                    # only inside the quotation
        for name in t.EXPLICIT_RELATIONS:
            self.assertIn(name, v2)
        for phrase in ("It means that the Urdu question directly expresses the same specific concept represented by "
                       "the page.", "a generally related topic", "broad semantic similarity", "a useful justification page",
                       "a hypernym or superordinate association requiring world knowledge", "an entity inferred from "
                       "another fact", "a page that merely helps answer the question", "a matching base token alone is "
                       "insufficient if the question does not identify the page sense",
                       "The question is written in Urdu and the titles are English Wikipedia page titles"):
            self.assertIn(phrase, v2)
        v1 = t.REVIEW_SYSTEM
        kept = [v1[:v1.index(" Answer four questions.")], v1[v1.index("q3: "):]]      # framing; q3, q4 and output format
        for sentence in kept:
            self.assertIn(sentence, v2)
        self.assertTrue(v2.startswith(kept[0]) and v2.endswith(kept[1]))
        for term in SMOKE_CASE_TERMS:                                                  # no smoke-case examples
            self.assertNotIn(term.lower(), v2.lower())

    def test_q1_q2_scope_and_polarity(self):
        v2 = t.REVIEW_SYSTEM_V2
        q1, q2 = v2[v2.index("q1: "):v2.index("q2: ")], v2[v2.index("q2: "):v2.index("q3: ")]
        self.assertIn("Judge q1 from the Urdu question and the proposed parent title only", q1)
        self.assertIn("the stated intermediate information, the decomposition steps", q1)
        self.assertIn("Answer Y if a permitted mapping supports direct identification of the parent, N if the required "
                      "direct mapping is absent, or U if you cannot understand", q1)
        self.assertTrue(q2.startswith("q2: Is the child NOT directly identifiable from the Urdu question alone"))
        self.assertIn("meaning exactly the same as in q1", q2)
        self.assertIn("Answer Y if the required direct mapping for the child is absent, N if a permitted mapping "
                      "supports direct identification of the child", q2)
        self.assertEqual(t.ACCEPT, ("Y", "Y", "Y", "C"))                               # acceptance unchanged
        ok = {"status": "OK", "parsed": {"q1": "Y", "q2": "Y", "q3": "Y", "q4": "C"}}
        child_identifiable = {"status": "OK", "parsed": {"q1": "Y", "q2": "N", "q3": "Y", "q4": "C"}}
        self.assertEqual(t.pair_decision(ok, ok)[0], "ACCEPTED_AI_ASSISTED")
        self.assertEqual(t.pair_decision(ok, child_identifiable)[0], "EXCLUDED_DISAGREEMENT")


class CalibrationTests(Base):
    @classmethod
    def setUpClass(cls):
        cls.gemma_tok = gemma_tokenizer()

    def prepare(self, tokenizer=None):
        self.root = Path(self.tmp.name) / "calibration"
        result = t.calib_prepare(root=self.root, tokenizer=tokenizer)
        self.inputs = t.read_jsonl(self.root / "packet/inputs.jsonl")
        self.plan = t.read_jsonl(self.root / "packet/call_plan.jsonl")
        self.labels = t.read_jsonl(self.root / "labels/expected_labels.jsonl")
        return result

    def test_complete_coverage_unique_ids_and_unmodified_labels(self):
        result = self.prepare(tokenizer=self.gemma_tok)
        self.assertEqual((result["cases"], result["calls"], len(self.inputs), len(self.plan)), (51, 102, 51, 102))
        ids = [c["case_id"] for c in self.inputs]
        self.assertEqual(len(set(ids)), 51)
        self.assertEqual(len({c["packet_id"] for c in self.inputs}), 51)
        self.assertEqual(ids, [f"H{i:02d}" for i in range(1, 42)] + [f"C{i:02d}" for i in range(1, 11)])
        source = {(r["qid"], r["parent_source_instance_id"], r["child_source_instance_id"]): r
                  for r in t.read_jsonl(t.CALIB_SOURCES["verified"])}
        hist = [l for l in self.labels if l["kind"] != "CONTROL"]
        self.assertEqual({(l["urbench_qid"], l["parent_source_instance_id"], l["child_source_instance_id"]) for l in hist},
                         set(source))                                                  # all 41, no others
        for l in hist:                                                                 # labels copied verbatim
            r = source[(l["urbench_qid"], l["parent_source_instance_id"], l["child_source_instance_id"])]
            self.assertEqual((l["verdict"], l["human_answers"], l["parent_title"], l["child_title"]),
                             (r["verdict"], r["human_answers"], r["parent_title"], r["child_title"]))
        self.assertEqual(Counter(l["kind"] for l in self.labels),
                         Counter({"HISTORICAL_ACCEPTED": 36, "HISTORICAL_REJECTED": 5, "CONTROL": 10}))
        self.assertTrue(all(l["historical_vector"][0] == "Y" for l in hist))
        self.assertEqual({tuple(l["historical_vector"]) for l in hist if l["kind"] == "HISTORICAL_REJECTED"},
                         {("Y", "Y", "N", "O")})                                       # not q1-negative references
        ctrl = [l for l in self.labels if l["kind"] == "CONTROL"]
        self.assertEqual(Counter(l["expected_q1"] for l in ctrl), Counter({"Y": 4, "N": 4, "U": 2}))
        self.assertTrue(all(l["label_status"] == t.CONTROL_STATUS and l["scored"] == ["q1"] for l in ctrl))
        self.assertFalse({c["packet_id"] for c in self.inputs[41:]} & {l["pair_id"] for l in hist})
        by_id = {c["case_id"]: c for c in self.inputs}
        smoke_pairs = [l["case_id"] for l in hist if (l["parent_title"], l["child_title"]) in
                       (("Sound barrier", "Speed of sound"), ("Grapefruit", "Grapefruit–drug interactions"))]
        self.assertEqual([by_id[c]["user_sha256"] for c in smoke_pairs], GEMMA_SMOKE_REVIEW_SHA256)   # same builder
        seal = json.loads((self.root / "packet/SEAL.json").read_text())
        self.assertEqual(seal["budget"], {"calls": 102, "max_new_tokens_per_call": 256, "max_new_tokens_total": 26112,
                                          "model_loads": 1})
        self.assertTrue(all(n + 256 <= 8192 for n in seal["max_prompt_tokens"].values()))
        self.assertEqual(seal["prompt_sha256"], {"v1": t.PROMPT_SHA256["review"], "v2": t.PROMPT_SHA256["review_v2"]})
        self.raises("OUTPUT_EXISTS_REFUSING_OVERWRITE", t.calib_prepare, root=self.root)

    def test_label_and_claim_isolation_from_inputs(self):
        self.prepare()
        self.assertTrue(all(set(c) == {"case_id", "packet_id", "user", "user_sha256"} for c in self.inputs))
        leaks = {"VERIFIED", "REJECTED", "EXPLICIT", "LATENT_BRIDGE", "DEPENDENCY", "AI_AUTHORED", "expected",
                 "verdict", "human_answers", "rationale", "confidence", "justification"}
        for l in self.labels:
            leaks |= {x for x in (l.get("justification"), l.get("gloss_ai_assisted"), l.get("rejection_reason"),
                                  l.get("debatable")) if x}
        for c in self.inputs:
            self.assertEqual(set(json.loads(c["user"])), set(t.PACKET_FIELDS) - {"packet_id"})
            for token in leaks:
                self.assertNotIn(token, c["user"], (c["case_id"], token))
        packet_files = {p.name for p in (self.root / "packet").iterdir()}
        self.assertNotIn("expected_labels.jsonl", packet_files)
        backend = FakeBackend()
        t.READ_LOG.clear()
        t.calib_run(lambda: backend, root=self.root)
        packet_dir = str((self.root / "packet").resolve())
        self.assertTrue(t.READ_LOG and all(p.startswith(packet_dir) for p in t.READ_LOG))
        seal = json.loads((self.root / "gemma_run/SEAL.json").read_text())
        self.assertFalse(seal["labels_read"])
        self.assertFalse([p for p in seal["inputs_read"] if "/labels/" in p])
        with t.stage_reads(self.root / "packet"):
            self.raises("READ_OUTSIDE_STAGE_ALLOWLIST", t.read_bytes, self.root / "labels/expected_labels.jsonl")

    def test_identical_payloads_across_versions_in_frozen_order(self):
        self.prepare()
        backend = FakeBackend()
        t.calib_run(lambda: backend, root=self.root)
        self.assertEqual(len(backend.seen), 102)
        by_id = {c["case_id"]: c for c in self.inputs}
        for i, (msg, plan) in enumerate(zip(backend.seen, self.plan)):
            system, user = split_messages(msg)
            self.assertEqual((plan["call"], plan["version"]), (i, "v1" if i % 2 == 0 else "v2"))
            self.assertEqual(system, t.REVIEW_PROMPTS[plan["version"]])
            self.assertEqual(t.canonical(user), by_id[plan["case_id"]]["user"])
        for a, b in zip(backend.seen[0::2], backend.seen[1::2]):                      # v1 / v2 of the same case
            self.assertEqual(split_messages(a)[1], split_messages(b)[1])
            self.assertNotEqual(split_messages(a)[0], split_messages(b)[0])
        records = t.read_jsonl(self.root / "gemma_run/calibration_records.jsonl")
        self.assertEqual([(r["case_id"], r["version"]) for r in records], [(p["case_id"], p["version"]) for p in self.plan])

    def test_run_fails_closed_before_loading(self):
        self.prepare()
        loads = []
        with mock.patch.dict(t.REVIEW_PROMPTS, {"v2": t.REVIEW_SYSTEM_V2 + " "}):
            self.raises("CALIB_PLAN_IDENTITY", t.calib_run, lambda: loads.append(1), root=self.root)
        path = self.root / "packet/inputs.jsonl"
        path.write_text(path.read_text().replace('"H01"', '"H99"', 1))
        self.raises("SEALED_OUTPUT_DRIFT", t.calib_run, lambda: loads.append(1), root=self.root)
        with mock.patch.dict(os.environ, dict(self.env, QEA_EXPECT_RUNNER_SHA256="0" * 64)):
            self.raises("CODE_IDENTITY_MISMATCH", t.calib_run, lambda: loads.append(1), root=self.root)
        self.assertEqual(loads, [])
        self.assertFalse((self.root / "gemma_run").exists())

    def test_scoring_keeps_failures_and_never_counts_u_or_invalid_as_negative(self):
        self.prepare()
        ok = lambda *a: {"status": "OK", "finish_reason": "stop", "malformed_reason": None, "parsed": dict(zip(t.Q_KEYS, a))}
        bad = {"status": "MALFORMED", "finish_reason": "stop", "malformed_reason": "NOT_JSON", "parsed": None}
        length = {"status": "MALFORMED", "finish_reason": "length", "malformed_reason": "FINISH_LENGTH", "parsed": None}
        v1 = {l["case_id"]: ok("Y", "Y", "Y", "C") for l in self.labels if l["kind"] != "CONTROL"}
        rej = [l["case_id"] for l in self.labels if l["kind"] == "HISTORICAL_REJECTED"]
        v1[rej[1]], v1[rej[2]], v1[rej[3]], v1[rej[4]] = ok("Y", "Y", "U", "O"), bad, ok("Y", "Y", "N", "O"), ok("N", "Y", "N", "O")
        v1.update({"C01": ok("Y", "U", "U", "U"), "C02": ok("U", "Y", "Y", "C"), "C03": bad, "C04": ok("N", "Y", "Y", "C"),
                   "C05": length, "C06": ok("N", "Y", "Y", "O"), "C07": ok("U", "Y", "Y", "C"), "C08": ok("N", "Y", "Y", "C"),
                   "C09": ok("Y", "Y", "Y", "C")})                                     # C10 missing
        records = [dict(r, case_id=c, version="v1") for c, r in v1.items()]
        records += [dict(ok("Y", "Y", "Y", "C"), case_id=l["case_id"], version="v2") for l in self.labels]
        report, rows = t.calibration_score(self.labels, records)
        r1, r2 = report["versions"]["v1"], report["versions"]["v2"]
        self.assertEqual(r1["validity"]["calls_expected"], 51)
        self.assertEqual((r1["validity"]["length_failures"], r1["validity"]["missing"], r1["validity"]["invalid"]),
                         (1, 1, {"NOT_JSON": 2}))
        self.assertEqual(r1["controls_q1"]["denominators"], {"Y": 4, "N": 4, "U": 2})
        self.assertEqual(r1["controls_q1"]["correct"], {"Y": 2, "N": 2, "U": 1})       # U, invalid, missing never correct
        self.assertEqual(r1["controls_q1"]["confusion_expected_by_observed"]["N"], {"U": 1, "N": 2, "MISSING": 1})
        self.assertEqual(r1["historical_q1_positive"], {"denominator": 41, "q1_Y": 39, "q1_N": 1, "q1_U": 0, "failures": 1})
        self.assertEqual(r1["accepted_36"]["exact_acceptance_vector"], 36)
        self.assertEqual(r1["rejected_5"]["classes"], {"ACCEPTANCE_VECTOR": 1, "UNCERTAIN": 1, "INVALID": 1, "VALID_REJECTION": 2})
        self.assertEqual(r2["controls_q1"]["correct"], {"Y": 4, "N": 0, "U": 0})       # all-Y reviewer gains nothing on N/U
        self.assertEqual(report["paired_changes"]["cases_changed"], sum(r["changed"] for r in rows))
        self.assertEqual(sum(q["pairs"] for q in report["paired_changes"]["by_qid"].values()), 41)
        self.assertEqual(len(report["paired_changes"]["by_qid"]), 30)
        self.assertIn("no p-values", report["note"])

    def test_report_stage_chains_the_seals(self):
        self.prepare()
        t.calib_run(FakeBackend, root=self.root)
        t.calib_report(root=self.root)
        seal = json.loads((self.root / "report/SEAL.json").read_text())
        run_seal = json.loads((self.root / "gemma_run/SEAL.json").read_text())
        self.assertEqual(seal["packet_seal_sha256"], run_seal["packet_seal_sha256"])
        report = json.loads((self.root / "report/report.json").read_text())
        self.assertEqual(report["versions"]["v1"]["validity"]["valid"], 51)


# ------------------------------------------------------------------ isolated title identification (amendment 4)

LEGACY_PROMPT_SHA256 = {"pass2": "4a96e8185cbc19cccb09446f9b215cc9ffaaa2e99ad3d5960da59deb72dce98b",
                        "pass3": "0171cc7ce410c4c37124e57c60c1f7e7023cc384106bbb7ec95842ce6db96ebd",
                        "review": "5c656e40dee28fcb44b2787662c5f33f7332cc8a625f7aee24d80f6a0f9399cf",
                        "review_v2": "59e53e83ee05a1a5fb9378ce343b3cdcd2c5c012a198a240d66076623ee9fd16"}


class IdentificationPromptTests(unittest.TestCase):
    def test_prompt_pinned_role_free_and_legacy_unchanged(self):
        p = t.IDENT_SYSTEM
        self.assertEqual(t.PROMPT_SHA256["identification"], t.PINNED_PROMPT_SHA256["identification"])
        self.assertEqual({k: t.PROMPT_SHA256[k] for k in LEGACY_PROMPT_SHA256}, LEGACY_PROMPT_SHA256)
        self.assertEqual(t.REVIEW_PROMPTS, {"v1": t.REVIEW_SYSTEM, "v2": t.REVIEW_SYSTEM_V2})
        self.assertEqual(t.ACCEPT, ("Y", "Y", "Y", "C"))
        self.assertIn('"' + freeze_definition() + '"', p)
        self.assertEqual(p.count("reasonably"), 1)
        self.assertEqual(p.count(t.IDENT_QUESTION), 1)
        self.assertEqual(t.IDENT_QUESTION, "Is this page directly identifiable from the Urdu question?")
        for name in t.EXPLICIT_RELATIONS:
            self.assertIn(name, p)
        v2 = t.REVIEW_SYSTEM_V2
        block = v2[v2.index("The question is written in Urdu"):v2.index(" Answer four questions.")]
        self.assertIn(block.replace("the titles are English Wikipedia page titles", "the title is an English Wikipedia page title"), p)
        import re
        for hint in ("parent", "child", "q1", "q2", "q3", "q4", "dependency", "intermediate information", "reviewer",
                     "accept"):                                                                # whole words only
            self.assertIsNone(re.search(r"\b" + re.escape(hint) + r"\b", p, flags=re.I), hint)
        self.assertTrue(p.endswith("with keys answer (Y, N or U) and note (one sentence of at most 40 words)."))

    def test_mapping_truth_table_and_parser(self):
        table = {"Y": ("Y", "N"), "N": ("N", "Y"), "U": ("U", "U")}
        for answer, (q1, q2) in table.items():
            self.assertEqual((t.ident_to_q(answer, "parent"), t.ident_to_q(answer, "control"), t.ident_to_q(answer, "child")),
                             (q1, q1, q2))
            self.assertEqual(t.q_to_ident(t.ident_to_q(answer, "child"), "child"), answer)       # inverse restatement
        for failure in ("INVALID", "LENGTH", "MISSING"):                                       # never a semantic answer
            self.assertEqual({t.ident_to_q(failure, r) for r in ("parent", "child", "control")}, {failure})
            self.assertEqual(t.q_to_ident(failure, "child"), failure)
        ok = json.dumps({"answer": "N", "note": "No mapping."})
        self.assertEqual(t.parse_identification(ok, "stop"), ({"answer": "N", "note": "No mapping."}, None))
        for text, finish, code in ((ok, "length", "FINISH_LENGTH"), ("no", "stop", "NOT_JSON"),
                                   (json.dumps({"answer": "X", "note": ""}), "stop", "ENUM"),
                                   (json.dumps({"answer": "Y", "note": "", "q1": "Y"}), "stop", "KEY_SET"),
                                   (json.dumps({"answer": "Y"}), "stop", "KEY_SET")):
            self.assertEqual(t.parse_identification(text, finish)[1], code)
        self.assertEqual(t.ident_outcome(None), "MISSING")
        self.assertEqual(t.ident_outcome({"status": "MALFORMED", "finish_reason": "length"}), "LENGTH")
        self.assertEqual(t.ident_outcome({"status": "MALFORMED", "finish_reason": "stop"}), "INVALID")
        self.assertEqual(t.ident_outcome({"status": "OK", "finish_reason": "stop", "parsed": {"answer": "U"}}), "U")

    def test_conflict_and_discrepancy_detection(self):
        def case(cid, q, parent, child):
            return {"case_id": cid, "user": t.canonical({"question_ur": q, "parent_title": parent, "child_title": child})}
        cases = [case("H01", "سوال", "Alpha", "Beta"), case("H02", "سوال", "Beta", "Gamma"),
                 case("H03", "سوال ", "Alpha", "Delta")]                                      # trailing space: other text
        labels = [{"case_id": c, "kind": "HISTORICAL_ACCEPTED", "urbench_qid": "q1", "historical_vector": ["Y", "Y", "Y", "C"]}
                  for c in ("H01", "H02", "H03")]
        items, assoc, refs, disc = t.derive_identification_items(cases, labels)
        beta = next(i["item_id"] for i in items if i["title"] == "Beta")
        self.assertEqual(disc["cross_role_items"], [beta])
        self.assertEqual(disc["conflicting_reference_items"], [beta])                           # parent Y vs child N
        self.assertEqual(disc["inconsistent_question_text_qids"], ["q1"])
        self.assertEqual(next(r for r in refs if r["item_id"] == beta)["expected"], "CONFLICT")
        self.assertEqual(len([i for i in items if i["title"] == "Alpha"]), 2)                   # no normalization merge
        u = [{"case_id": "H01", "kind": "HISTORICAL_ACCEPTED", "urbench_qid": "q1", "historical_vector": ["U", "U", "Y", "C"]}]
        _i, _a, refs, _d = t.derive_identification_items(cases[:1], u)
        self.assertEqual([r["expected"] for r in refs], ["U", "U"])                              # U preserved both ways


class IdentificationPacketTests(Base):
    def calibration(self, backend=None, name="calibration"):
        root = Path(self.tmp.name) / name
        t.calib_prepare(root=root)
        t.calib_run(lambda: backend or FakeBackend(), root=root)
        return root

    def prepare(self, source, name="ident", tokenizer=None):
        root = Path(self.tmp.name) / name
        result = t.ident_prepare(root=root, source=source, tokenizer=tokenizer)
        return root, result

    def test_counts_associations_isolation_and_determinism(self):
        source = self.calibration()
        root, result = self.prepare(source, tokenizer=gemma_tokenizer())
        self.assertEqual(result["counts"]["items"], 81)
        self.assertEqual(result["counts"]["by_role"], {"parent": 30, "child": 41, "control": 10})
        self.assertEqual((result["counts"]["occurrences"], result["counts"]["occurrences_by_role"]),
                         (92, {"parent": 41, "child": 41, "control": 10}))
        self.assertEqual(result["counts"]["references"], {"Y": 34, "N": 45, "U": 2})
        self.assertEqual(result["discrepancies"], {"cross_role_items": [], "inconsistent_question_text_qids": [],
                                                   "conflicting_reference_items": [], "items_spanning_several_questions": []})
        inputs = t.read_jsonl(root / "packet/inputs.jsonl")
        items = {i["item_id"]: i for i in t.read_jsonl(root / "labels/items.jsonl")}
        for row in inputs:                                                                   # exact input isolation
            self.assertEqual(set(row), {"item_id", "user", "user_sha256"})
            payload = json.loads(row["user"])
            self.assertEqual(payload, {"question_ur": items[row["item_id"]]["question_ur"], "title": items[row["item_id"]]["title"]})
        self.assertEqual(sorted(p.name for p in (root / "packet").iterdir()),
                         ["SEAL.json", "call_plan.jsonl", "identification_system.txt", "inputs.jsonl"])
        plan = t.read_jsonl(root / "packet/call_plan.jsonl")
        self.assertTrue(all(set(p) == {"call", "item_id", "user_sha256", "system_sha256"} for p in plan))
        seal = json.loads((root / "packet/SEAL.json").read_text())
        self.assertEqual(seal["budget"], {"calls": 81, "max_new_tokens_per_call": 256, "max_new_tokens_total": 20736,
                                          "model_loads": 1})
        self.assertLessEqual(seal["max_prompt_tokens"] + 256, 8192)
        assoc = t.read_jsonl(root / "labels/associations.jsonl")
        cases = t.read_jsonl(source / "packet/inputs.jsonl")
        expected_occ = {(c["case_id"], r) for c in cases for r in (("control",) if c["case_id"].startswith("C") else ("parent", "child"))}
        self.assertEqual([(a["case_id"], a["role"]) for a in assoc].__len__(), len(expected_occ))
        self.assertEqual({(a["case_id"], a["role"]) for a in assoc}, expected_occ)          # every occurrence once
        refs = {r["item_id"]: r for r in t.read_jsonl(root / "labels/references.jsonl")}
        for a in assoc:
            if a["role"] == "child":
                self.assertEqual((a["source_question"], a["source_label"], a["derived_identification"]), ("q2", "Y", "N"))
        ctrl = {a["case_id"]: refs[a["item_id"]] for a in assoc if a["role"] == "control"}
        self.assertEqual((ctrl["C07"]["expected"], ctrl["C08"]["expected"]), ("U", "U"))
        self.assertIn("N is also defensible", ctrl["C07"]["debatable"])
        self.assertEqual({ctrl[c]["label_status"] for c in ctrl}, {t.CONTROL_STATUS})
        root2, _r = self.prepare(source, name="ident_again")                                   # deterministic
        for d in ("packet", "labels", "baseline"):
            for f in (root / d).iterdir():
                if f.name != "SEAL.json":
                    self.assertEqual(f.read_bytes(), (root2 / d / f.name).read_bytes(), f)
        self.raises("OUTPUT_EXISTS_REFUSING_OVERWRITE", t.ident_prepare, root=root, source=source)

    def test_baseline_preserves_inconsistent_old_judgments(self):
        source = self.calibration(FakeBackend(review_answers={("John Key", "Helen Clark"): ("N", "Y", "Y", "C")}))
        root, result = self.prepare(source)
        items = {(i["title"]): i["item_id"] for i in t.read_jsonl(root / "labels/items.jsonl")}
        old_items = {(r["version"], r["item_id"]): r for r in t.read_jsonl(root / "baseline/old_items.jsonl")}
        for v in ("v1", "v2"):
            row = old_items[(v, items["John Key"])]
            self.assertEqual(row["old_identification"], "MIXED")
            self.assertEqual(sorted(u["old_identification"] for u in row["underlying"]), ["N", "Y", "Y", "Y"])
            self.assertEqual(old_items[(v, items["Helen Clark"])]["old_identification"], "N")     # q2=Y inverted
        occ = t.read_jsonl(root / "baseline/old_occurrences.jsonl")
        self.assertEqual(Counter(r["version"] for r in occ), Counter({"v1": 92, "v2": 92}))
        seal = json.loads((root / "baseline/SEAL.json").read_text())
        self.assertEqual(seal["mixed_items"], {"v1": [items["John Key"]], "v2": [items["John Key"]]})

    def test_inference_reads_packet_only_guards_and_identical_payloads(self):
        source = self.calibration()
        root, _r = self.prepare(source)
        with mock.patch.dict(os.environ, dict(self.env, QEA_EXPECT_TEST_SHA256="0" * 64)):
            t.READ_LOG.clear()
            self.raises("CODE_IDENTITY_MISMATCH", t.ident_run, FakeBackend, root=root)
            self.assertEqual(t.READ_LOG, [])
        backend = FakeBackend()
        t.READ_LOG.clear()
        t.ident_run(lambda: backend, root=root)
        packet_dir = str((root / "packet").resolve())
        self.assertTrue(t.READ_LOG and all(p.startswith(packet_dir) for p in t.READ_LOG))
        inputs = t.read_jsonl(root / "packet/inputs.jsonl")
        self.assertEqual(len(backend.seen), len(inputs))
        for msg, row in zip(backend.seen, inputs):
            system, user = split_messages(msg)
            self.assertEqual((system, t.canonical(user)), (t.IDENT_SYSTEM, row["user"]))
        seal = json.loads((root / "gemma_run/SEAL.json").read_text())
        self.assertFalse(seal["labels_read"])
        self.raises("OUTPUT_EXISTS_REFUSING_OVERWRITE", t.ident_run, FakeBackend, root=root)

    def test_report_keeps_failures_separate_and_forms_no_acceptance(self):
        source = self.calibration()
        root, _r = self.prepare(source)
        backend = FakeBackend(ident_override={"Frog": json.dumps({"answer": "Y", "note": "x"}),
                                              "John Key": json.dumps({"answer": "U", "note": "x"}),
                                              "Hedgehog": "not json", "Mercury (planet)": json.dumps({"answer": "U", "note": "x"})},
                              finish_override={("ident", "Lemon"): "length"})
        t.ident_run(lambda: backend, root=root)
        items = t.read_jsonl(root / "labels/items.jsonl")
        records = t.read_jsonl(root / "gemma_run/identification_records.jsonl")
        by_title = {i["title"]: i["item_id"] for i in items}
        records = [r for r in records if r["item_id"] != by_title["Grapefruit"]]              # one missing output
        report, item_rows, occ_rows = t.identification_score(
            items, t.read_jsonl(root / "labels/associations.jsonl"), t.read_jsonl(root / "labels/references.jsonl"),
            t.read_jsonl(root / "baseline/old_items.jsonl"), t.read_jsonl(root / "baseline/old_occurrences.jsonl"), records)
        par, chi = report["historical_parent_items"], report["historical_child_items"]
        self.assertEqual((par["denominator"], chi["denominator"]), (30, 41))
        self.assertEqual({k: par["distribution"][k] for k in ("U", "INVALID", "LENGTH", "MISSING")},
                         {"U": 1, "INVALID": 1, "LENGTH": 1, "MISSING": 1})
        self.assertEqual(par["positive_agreement"], 26)                                      # failures and U never count
        self.assertIn(by_title["Frog"], chi["false_explicit_identifications"])
        self.assertEqual((len(chi["false_explicit_identifications"]), chi["negative_agreement"]), (41, 0))  # stub: Y
        u = report["controls"]["by_expected"]["U"]
        self.assertEqual((u["denominator"], u["strict_agreement"], u["distribution"]["U"]), (2, 1, 1))  # C07 planet U
        self.assertEqual(len(report["controls"]["expected_U_items"]), 2)
        for v in ("v1", "v2"):
            self.assertEqual(report["comparison_with_job_96416"][v]["occurrences"]["count"], 92)
        self.assertEqual(len(occ_rows), 184)
        self.assertEqual(len(report["by_question"]), 35)                                     # 30 questions + 5 control groups
        self.assertNotIn("ACCEPTANCE_VECTOR", json.dumps(report))
        self.assertFalse([r for r in occ_rows if r["new_mapped"] in ("N", "Y") and r["new_identification"] in t.FAILURES])
        t.ident_report(root=root)
        seal = json.loads((root / "report/SEAL.json").read_text())
        self.assertEqual(seal["packet_seal_sha256"], json.loads((root / "gemma_run/SEAL.json").read_text())["packet_seal_sha256"])


# ------------------------------------------------------------------ input contract

class InputContractTests(Base):
    def test_development_requests_fit_and_schemas_match_prompts(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(t.MODELS["llama"]["path"], local_files_only=True)   # tokenizer only
        t.READ_LOG.clear()
        result = t.input_contract(t.Config(), tok)
        self.assertTrue(result["ok"], result["conflicts"])
        self.assertEqual((result["pass2"]["requests"], result["pass3_possible"]["requests"]), (544, 544))
        self.assertEqual((result["pass2"]["prompt_limit"], result["pass3_possible"]["prompt_limit"]), (3840, 3584))
        roots = (str(t.PREP_DIR.resolve()), str(t.V1_DIR.resolve()))
        self.assertTrue(t.READ_LOG and all(p.startswith(roots) for p in t.READ_LOG))

    def test_over_limit_and_contract_failures_are_reported_not_adjusted(self):
        class LongTok:
            def apply_chat_template(self, m, tokenize, add_generation_prompt):
                return [1] * 3841
        result = t.input_contract(self.cfg, LongTok())
        self.assertFalse(result["ok"])
        self.assertEqual((result["pass2"]["over_limit"], result["pass3_possible"]["over_limit"]), (14, 14))
        self.assertEqual(result["pass2"]["max"], 3841)
        link = {k: None for k in t.PASS3_EVIDENCE_FIELDS}
        link.update(record_type="PARAGRAPH", source_instance_id="s1")
        row = {"urbench_qid": "q", "source_instance_id": "s1", "gold_title": "Speed of sound", "question_ur": "x"}
        cases = [({"q": ["Speed of sound", "Speed_of_sound", "Sound barrier"]}, ["a", "b"], [link], "PASS3_SELF_CANDIDATE"),
                 ({"q": ["Speed of sound", "A", "A"]}, ["a", "b"], [link], "PASS3_DUPLICATE_CANDIDATES"),
                 ({"q": ["Speed of sound", "A"]}, [], [link], "PASS3_ZERO_STEPS"),
                 ({"q": ["Speed of sound", "A"]}, ["a"], [], "NO_TARGET_EVIDENCE")]
        for titles, steps, links, code in cases:
            prep = {"rows": {"q": {"official_decomposition": steps}}, "links": links}
            self.raises(code, t.pass3_request, prep, titles, row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
