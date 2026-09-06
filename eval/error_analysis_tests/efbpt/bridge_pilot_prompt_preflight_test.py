#!/usr/bin/env python3
"""Synthetic tests for the tokenizer-only prompt preflight. No real inputs.

Every fixture is built in a throwaway temporary directory with a stub
tokenizer. Nothing here loads the real Qwen tokenizer or any weight shard,
opens the real preparation exports or scoring targets, touches the FAISS index,
metadata or offsets, or produces a pilot outcome.

Run: python -B eval/error_analysis_tests/efbpt/bridge_pilot_prompt_preflight_test.py
"""
import sys
sys.dont_write_bytecode = True
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from bridge_pilot_core import (CHILD_COUNTS, PilotError, assert_prompt_hashes, canonical,
                               local_chunks, norm, sha256)
import bridge_pilot_core as core
import bridge_pilot_run as run
import bridge_pilot_prompt_preflight as pp

HERE = Path(__file__).resolve().parent
QIDS = sorted(CHILD_COUNTS)

# ------------------------------------------------------------------ fixtures

class StubTokenizer:
    """Records the exact kwargs it is called with. One token per character."""

    def __init__(self, tokens_per_prompt=None):
        self.tokens_per_prompt = tokens_per_prompt
        self.chat_template = "stub-template"
        self.is_fast = True
        self.vocab_size = 151643
        self.model_max_length = 131072
        self.template_calls = []
        self.tokenize_calls = []

    def apply_chat_template(self, messages, **kw):
        self.template_calls.append(kw)
        assert kw == {"tokenize": False, "add_generation_prompt": True,
                      "enable_thinking": False}, kw
        payload = "".join(m["content"] for m in messages)
        if self.tokens_per_prompt is not None:
            return "R" * self.tokens_per_prompt
        return "<|im_start|>" + payload + "<|im_end|>"

    def __call__(self, text, **kw):
        self.tokenize_calls.append(kw)
        assert kw == {"add_special_tokens": False, "truncation": False}, kw
        return {"input_ids": list(range(len(text)))}

def synthetic_page(title, text="  aaa London is a city.\nParis is another city. "):
    return {"raw_page_id": "synthetic", "raw_url": "https://example.invalid/p",
            "raw_title": title, "raw_text": text, "raw_text_sha256": sha256(text),
            "locations": [{"shard_index": 0, "blob_path": "/synthetic/unused",
                           "blob_sha256": "0" * 64, "row_group": 0, "row_in_group": 0,
                           "row_in_shard": 0}],
            "lookup_decision": "SYNTHETIC", "normalized_bucket_rows": 1,
            "normalized_bucket_distinct_pages": 1}

def synthetic_parent(qid, text=None):
    page = synthetic_page("Synthetic Parent " + qid,
                          text or "  aaa London is a city.\nParis is another city. ")
    return {"qid": qid, "question_ur": "یہ کیا ہے؟",
            "parent_source_instance_id": "synthetic-" + qid,
            "parent_title": "Synthetic Parent " + qid,
            "parent_normalized_title": norm("Synthetic Parent " + qid),
            "page": page, "chunks": local_chunks(page)}

def synthetic_oracle(qid):
    return {"qid": qid, "question_ur": "س", "urbench_facts": ["fact one", "fact two"]}

def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    return {"path": str(path), "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size}

def write_jsonl(path, rows):
    return write(path, "".join(canonical(r) + "\n" for r in rows))

class Tree:
    """Synthetic repo laid out exactly as the module's pinned constants expect."""

    def __init__(self, base):
        self.base = Path(base)
        self.repo = self.base / "repo"
        self.prep = self.repo / pp.PREP
        self.preflight_dir = self.repo / pp.PREFLIGHT
        self.source_dir = self.repo / "eval/error_analysis_tests/efbpt"
        self.prep_archive = self.base / "archives" / "bridge_pilot_n25_preparation_v1_r3"
        self.preflight_archive = self.base / "archives" / "bridge_pilot_n25_preflight_v2"
        self.qwen = self.base / "models" / "Qwen3-14B"
        self.out = self.repo / pp.OUT
        self.archive = self.base / "archives" / "bridge_pilot_n25_prompt_preflight_v1"
        self.parents = [synthetic_parent(q) for q in QIDS]
        self.oracles = [synthetic_oracle(q) for q in QIDS]
        self.build()

    def build(self):
        self.parent_export = write_jsonl(self.prep / run.PARENT_EXPORT, self.parents)
        self.oracle_export = write_jsonl(self.prep / run.ORACLE_EXPORT, self.oracles)
        self.target_export = write_jsonl(self.prep / run.TARGET_EXPORT,
                                         [{"qid": q, "children": []} for q in QIDS])
        for name in ("bridge_pilot_core.py", "bridge_pilot_run.py",
                     "bridge_pilot_prompt_preflight.py", "bridge_pilot_prompt_preflight_test.py",
                     "bridge_pilot_prompt_preflight.sbatch"):
            write(self.source_dir / name, (HERE / name).read_bytes())
        write(self.repo / pp.PROTOCOL, (HERE.parent.parent.parent / pp.PROTOCOL).read_bytes())
        self.prep_seal_obj = {
            "status": pp.PREPARATION_STATUS,
            "artifacts": {run.PARENT_EXPORT: {"sha256": self.parent_export["sha256"],
                                              "bytes": self.parent_export["bytes"]},
                          run.ORACLE_EXPORT: {"sha256": self.oracle_export["sha256"],
                                              "bytes": self.oracle_export["bytes"]},
                          run.TARGET_EXPORT: {"sha256": self.target_export["sha256"],
                                              "bytes": self.target_export["bytes"]}}}
        self.prep_seal = self.put_seal(self.prep / "PREPARATION_SEAL.json",
                                       self.prep_archive / "PREPARATION_SEAL.json",
                                       self.prep_seal_obj)
        self.pre_seal_obj = {"status": pp.PREFLIGHT_STATUS,
                             "historical_global_lineage": run.LINEAGE,
                             "preparation_seal_sha256": self.prep_seal["sha256"]}
        self.pre_seal = self.put_seal(self.preflight_dir / "PREFLIGHT_SEAL.json",
                                      self.preflight_archive / "PREFLIGHT_SEAL.json",
                                      self.pre_seal_obj)
        self.tokenizer_files = {}
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json",
                     "vocab.json", "merges.txt"):
            body = ('{"max_position_embeddings":40960,"model_type":"qwen3"}'
                    if name == "config.json" else '{"stub":"' + name + '"}')
            self.tokenizer_files[name] = write(self.qwen / name, body)

    def put_seal(self, primary, archived, obj):
        ident = write(primary, canonical(obj) + "\n")
        write(archived, canonical(obj) + "\n")
        return ident

    def pins(self):
        """The module constants this synthetic tree corresponds to."""
        return {
            "PREP_ARCHIVE": self.prep_archive, "PREFLIGHT_ARCHIVE": self.preflight_archive,
            "ARCHIVE": self.archive, "QWEN": self.qwen,
            "PREPARATION_SEAL_HASH": self.prep_seal["sha256"],
            "PREFLIGHT_SEAL_HASH": self.pre_seal["sha256"],
            "CORE_HASH": sha256((HERE / "bridge_pilot_core.py").read_bytes()),
            "RUNNER_HASH": sha256((HERE / "bridge_pilot_run.py").read_bytes()),
            "PROTOCOL_HASH": sha256((HERE.parent.parent.parent / pp.PROTOCOL).read_bytes()),
            "QWEN_TOKENIZER_FILES": {n: (v["sha256"], v["bytes"])
                                     for n, v in self.tokenizer_files.items()},
            "EXPORT_IDENTITIES": {
                run.PARENT_EXPORT: (self.parent_export["sha256"], self.parent_export["bytes"]),
                run.ORACLE_EXPORT: (self.oracle_export["sha256"], self.oracle_export["bytes"])},
            "FORBIDDEN": tuple(
                [("scoring_targets", str(self.prep / run.TARGET_EXPORT)),
                 ("index", "rag/index/wikipedia_full.index"),
                 ("metadata", "rag/index/wikipedia_full_meta.jsonl"),
                 ("offsets", "rag/index/wikipedia_full_meta.offsets.npy")]
                + [("qwen_weight_%02d" % i, self.qwen / ("model-%05d-of-00008.safetensors" % i))
                   for i in range(1, 9)]),
        }

class PinnedCase(unittest.TestCase):
    """Repoints the module's pinned constants at a synthetic tree, then restores."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="prompt_preflight_test_")
        self.tree = Tree(self.tmp)
        self.saved = {}
        for key, value in self.tree.pins().items():
            self.saved[key] = getattr(pp, key)
            setattr(pp, key, value)
        self.saved_tokenizer_loader = pp.load_tokenizer
        self.stub = StubTokenizer()
        pp.load_tokenizer = self.fake_loader

    def fake_loader(self, guard=None):
        guard = guard if guard is not None else None
        if guard is not None:
            guard.check(pp.QWEN)
        self.stub_loads = getattr(self, "stub_loads", 0) + 1
        return self.stub, {"root": str(pp.QWEN), "tokenizer_class": "StubTokenizer",
                           "is_fast": True, "vocab_size": 151643,
                           "tokenizer_model_max_length": 131072,
                           "chat_template_sha256": sha256(self.stub.chat_template),
                           "transformers_version": "stub",
                           "apply_chat_template_kwargs": {"tokenize": False,
                                                          "add_generation_prompt": True,
                                                          "enable_thinking": False},
                           "tokenize_kwargs": {"add_special_tokens": False, "truncation": False},
                           "model_context_limit": pp.CONTEXT_LIMIT,
                           "trust_remote_code": False, "local_files_only": True,
                           "weights_loaded": False, "weight_files_opened": 0}

    def tearDown(self):
        for key, value in self.saved.items():
            setattr(pp, key, value)
        pp.load_tokenizer = self.saved_tokenizer_loader
        shutil.rmtree(self.tmp, ignore_errors=True)

    def repo(self):
        return self.tree.repo

# ------------------------------------------------------------- plan and math

class PlanTests(unittest.TestCase):
    def test_exact_category_count_and_reserves(self):
        self.assertEqual([c[0] for c in pp.CATEGORIES],
                         ["state_C", "state_D", "query_A", "query_P", "query_B", "query_E"])
        self.assertEqual([c[1] for c in pp.CATEGORIES], ["C", "D", "A", "P", "B", "E"])
        self.assertEqual([c[2] for c in pp.CATEGORIES], [1024, 1024, 128, 128, 128, 128])
        self.assertEqual(25 * len(pp.CATEGORIES), 150)
        self.assertEqual(pp.CONTEXT_LIMIT, 40960)

    def test_cohort_is_25_and_36(self):
        self.assertEqual(len(CHILD_COUNTS), 25)
        self.assertEqual(sum(CHILD_COUNTS.values()), 36)

    def test_runtime_dependent_scope_is_not_a_measurement(self):
        rd = pp.RUNTIME_DEPENDENT
        self.assertEqual(rd["scope"], "RUNTIME_DEPENDENT")
        self.assertEqual(rd["categories"], ["query_C", "query_D"])
        self.assertEqual(rd["reserve"], 128)
        self.assertEqual(rd["parser_bounds"]["maximum_accepted_items"], 6)
        self.assertEqual(rd["parser_bounds"]["c_item_maximum_whitespace_words"], 8)
        self.assertEqual(rd["parser_bounds"]["d_item_maximum_whitespace_words"], 40)
        self.assertTrue(rd["parser_bounds"]["empty_state_is_identical_to_query_P"])
        # No measured length is claimed anywhere in the runtime-dependent block.
        for key in ("input_tokens", "total_tokens", "maximum_tokens", "measured"):
            self.assertNotIn(key, rd)

    def test_parser_bounds_match_the_frozen_core(self):
        row = synthetic_parent(QIDS[0], " ".join("w%d" % i for i in range(60)))
        cid = row["chunks"][0]["chunk_id"]
        items = [{"chunk_id": cid, "quote": "w%d" % i} for i in range(10)]
        state = core.parse_state(canonical({"facts": items}), row, "D")
        self.assertEqual(len(state["items"]),
                         pp.RUNTIME_DEPENDENT["parser_bounds"]["maximum_accepted_items"])
        # An empty state really does reproduce the P prompt exactly.
        empty = core.parse_state('{"facts":[]}', row, "D")
        self.assertEqual(core.parent_query_messages(row, "D", empty),
                         core.parent_query_messages(row, "P"))

# ------------------------------------------------------- boundary arithmetic

class MeasureTests(unittest.TestCase):
    def test_exact_chat_template_and_tokenize_kwargs(self):
        tok = StubTokenizer(100)
        pp.measure(tok, [], 128)
        self.assertEqual(tok.template_calls[0],
                         {"tokenize": False, "add_generation_prompt": True,
                          "enable_thinking": False})
        self.assertEqual(tok.tokenize_calls[0],
                         {"add_special_tokens": False, "truncation": False})

    def test_within_context_record(self):
        r = pp.measure(StubTokenizer(100), [], 128)
        self.assertTrue(r["pass"])
        self.assertEqual((r["input_tokens"], r["reserve"], r["total_tokens"]), (100, 128, 228))
        self.assertEqual(r["remaining_margin"], 40960 - 228)
        self.assertEqual(r["status"], "WITHIN_CONTEXT")

    def test_exact_boundary_and_one_over(self):
        edge = pp.measure(StubTokenizer(pp.CONTEXT_LIMIT - 1024), [], 1024)
        self.assertTrue(edge["pass"])
        self.assertEqual(edge["remaining_margin"], 0)
        over = pp.measure(StubTokenizer(pp.CONTEXT_LIMIT - 1023), [], 1024)
        self.assertFalse(over["pass"])
        self.assertEqual(over["status"], "EXCEEDS_CONTEXT")
        self.assertEqual(over["remaining_margin"], -1)
        self.assertEqual(over["input_tokens"], pp.CONTEXT_LIMIT - 1023)

    def test_reserve_is_applied_per_category(self):
        for _, _, reserve in pp.CATEGORIES:
            r = pp.measure(StubTokenizer(1000), [], reserve)
            self.assertEqual(r["reserve"], reserve)
            self.assertEqual(r["total_tokens"], 1000 + reserve)

    def test_over_context_without_reserve_is_reported_not_raised(self):
        r = pp.measure(StubTokenizer(pp.CONTEXT_LIMIT + 1), [], 128)
        self.assertFalse(r["pass"])
        self.assertEqual(r["status"], "EXCEEDS_CONTEXT_WITHOUT_ANY_RESERVE")
        self.assertIsNone(r["input_tokens"])
        self.assertIsNone(r["prompt_sha256"])

    def test_render_is_confirmed_twice_only_when_passing(self):
        ok = StubTokenizer(100)
        pp.measure(ok, [], 128)
        self.assertEqual(len(ok.template_calls), 2)
        bad = StubTokenizer(pp.CONTEXT_LIMIT - 1023)
        pp.measure(bad, [], 1024)
        self.assertEqual(len(bad.template_calls), 1)

    def test_prompt_hash_is_recorded_and_deterministic(self):
        tok = StubTokenizer()
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        a = pp.measure(tok, msgs, 128)
        b = pp.measure(tok, msgs, 128)
        self.assertEqual(a["prompt_sha256"], b["prompt_sha256"])
        self.assertEqual(a["prompt_sha256"], sha256("<|im_start|>SU<|im_end|>"))

# ------------------------------------------------------- full 150-record run

class MeasureAllTests(PinnedCase):
    def test_completeness_and_uniqueness(self):
        constants = assert_prompt_hashes()
        _, identity = self.fake_loader()
        records, maxima = pp.measure_all(self.stub, self.tree.parents, self.tree.oracles,
                                         constants, identity)
        self.assertEqual(len(records), 150)
        self.assertEqual(len({(r["category"], r["qid"]) for r in records}), 150)
        self.assertEqual(sorted(maxima), sorted(c[0] for c in pp.CATEGORIES))
        for category, _, reserve in pp.CATEGORIES:
            rows = [r for r in records if r["category"] == category]
            self.assertEqual(len(rows), 25)
            self.assertEqual({r["qid"] for r in rows}, set(QIDS))
            self.assertEqual({r["reserve"] for r in rows}, {reserve})
            self.assertEqual(maxima[category]["reserve"], reserve)
            self.assertEqual(maxima[category]["input_tokens"],
                             max(r["input_tokens"] for r in rows))
            self.assertIn(maxima[category]["qid"], QIDS)

    def test_records_carry_no_research_content(self):
        constants = assert_prompt_hashes()
        _, identity = self.fake_loader()
        records, _ = pp.measure_all(self.stub, self.tree.parents, self.tree.oracles,
                                    constants, identity)
        allowed = {"schema", "category", "arm", "qid", "prompt_constants", "tokenizer_sha256",
                   "context_limit", "input_tokens", "prompt_sha256", "reserve",
                   "total_tokens", "remaining_margin", "status", "pass"}
        blob = canonical(records)
        for record in records:
            self.assertEqual(set(record), allowed)
        # No prompt, page, title, question or fact text can appear in the records.
        for leak in ("London is a city", "Synthetic Parent", "یہ کیا ہے", "fact one",
                     "<|im_start|>", "im_end"):
            self.assertNotIn(leak, blob)

    def test_missing_or_duplicate_qid_is_rejected(self):
        constants = assert_prompt_hashes()
        _, identity = self.fake_loader()
        short = self.tree.parents[:-1]
        with self.assertRaises((PilotError, KeyError)):
            pp.measure_all(self.stub, short, self.tree.oracles, constants, identity)
        duplicated = self.tree.parents[:-1] + [copy.deepcopy(self.tree.parents[0])]
        with self.assertRaises((PilotError, KeyError)):
            pp.measure_all(self.stub, duplicated, self.tree.oracles, constants, identity)

# --------------------------------------------------- inputs, seals, boundary

class SmallInputTests(PinnedCase):
    def check(self):
        return pp.check_small_inputs(self.repo())

    def test_check_small_inputs_writes_nothing(self):
        before = sorted(p for p in self.tree.repo.rglob("*"))
        report = self.check()
        after = sorted(p for p in self.tree.repo.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(report["files_written"], 0)
        self.assertEqual(report["parent_rows"], 25)
        self.assertEqual(report["oracle_rows"], 25)
        self.assertEqual(report["expected_prompt_records"], 150)
        self.assertEqual(report["qwen_weight_files_opened"], 0)
        self.assertEqual(report["qwen_generations"], 0)
        self.assertEqual(report["pilot_searches"], 0)
        self.assertFalse(report["scoring_targets_opened"])
        self.assertFalse(report["index_loaded"] or report["encoder_loaded"])
        self.assertEqual(report["experiment_state"], "NOT_FROZEN_NOT_RUN")
        self.assertEqual(report["historical_global_lineage"], "UNESTABLISHED")
        self.assertEqual(report["prompt_constants"], core.PROMPT_HASHES)

    def test_preparation_seal_tampering_refused(self):
        write(self.tree.prep / "PREPARATION_SEAL.json",
              canonical({"status": "SELF_DECLARED", "artifacts": {}}) + "\n")
        with self.assertRaises(PilotError):
            self.check()

    def test_seal_archive_copy_divergence_refused(self):
        obj = dict(self.tree.prep_seal_obj)
        obj["artifacts"] = dict(obj["artifacts"])
        write(self.tree.prep_archive / "PREPARATION_SEAL.json", canonical(obj) + "\nX")
        with self.assertRaises(PilotError):
            self.check()

    def test_preflight_must_bind_the_preparation_seal(self):
        write(self.tree.preflight_dir / "PREFLIGHT_SEAL.json",
              canonical({"status": pp.PREFLIGHT_STATUS,
                         "historical_global_lineage": run.LINEAGE,
                         "preparation_seal_sha256": "9" * 64}) + "\n")
        pp.PREFLIGHT_SEAL_HASH = sha256(
            (self.tree.preflight_dir / "PREFLIGHT_SEAL.json").read_bytes())
        write(self.tree.preflight_archive / "PREFLIGHT_SEAL.json",
              (self.tree.preflight_dir / "PREFLIGHT_SEAL.json").read_bytes())
        with self.assertRaises(PilotError):
            self.check()

    def test_lineage_downgrade_in_preflight_seal_refused(self):
        obj = {"status": pp.PREFLIGHT_STATUS, "historical_global_lineage": "ESTABLISHED",
               "preparation_seal_sha256": self.tree.prep_seal["sha256"]}
        write(self.tree.preflight_dir / "PREFLIGHT_SEAL.json", canonical(obj) + "\n")
        write(self.tree.preflight_archive / "PREFLIGHT_SEAL.json", canonical(obj) + "\n")
        pp.PREFLIGHT_SEAL_HASH = sha256(canonical(obj).encode() + b"\n")
        with self.assertRaises(PilotError):
            self.check()

    def test_export_hash_drift_refused(self):
        write_jsonl(self.tree.prep / run.PARENT_EXPORT,
                    [synthetic_parent(q, "different page text here") for q in QIDS])
        with self.assertRaises(PilotError):
            self.check()

    def test_export_unknown_field_refused(self):
        rows = [dict(r, injected_child_title="leak") for r in self.tree.parents]
        ident = write_jsonl(self.tree.prep / run.PARENT_EXPORT, rows)
        pp.EXPORT_IDENTITIES = dict(pp.EXPORT_IDENTITIES,
                                    **{run.PARENT_EXPORT: (ident["sha256"], ident["bytes"])})
        obj = copy.deepcopy(self.tree.prep_seal_obj)
        obj["artifacts"][run.PARENT_EXPORT] = {"sha256": ident["sha256"], "bytes": ident["bytes"]}
        self.tree.put_seal(self.tree.prep / "PREPARATION_SEAL.json",
                           self.tree.prep_archive / "PREPARATION_SEAL.json", obj)
        pp.PREPARATION_SEAL_HASH = sha256(canonical(obj).encode() + b"\n")
        with self.assertRaises(PilotError):
            self.check()

    def test_missing_qid_in_export_refused(self):
        ident = write_jsonl(self.tree.prep / run.ORACLE_EXPORT,
                            [synthetic_oracle(q) for q in QIDS[:-1]])
        pp.EXPORT_IDENTITIES = dict(pp.EXPORT_IDENTITIES,
                                    **{run.ORACLE_EXPORT: (ident["sha256"], ident["bytes"])})
        obj = copy.deepcopy(self.tree.prep_seal_obj)
        obj["artifacts"][run.ORACLE_EXPORT] = {"sha256": ident["sha256"], "bytes": ident["bytes"]}
        self.tree.put_seal(self.tree.prep / "PREPARATION_SEAL.json",
                           self.tree.prep_archive / "PREPARATION_SEAL.json", obj)
        pp.PREPARATION_SEAL_HASH = sha256(canonical(obj).encode() + b"\n")
        with self.assertRaises(PilotError):
            self.check()

    def test_core_or_protocol_drift_refused(self):
        for attr in ("CORE_HASH", "PROTOCOL_HASH", "RUNNER_HASH"):
            with self.subTest(attr=attr):
                saved = getattr(pp, attr)
                setattr(pp, attr, "a" * 64)
                try:
                    with self.assertRaises(PilotError):
                        self.check()
                finally:
                    setattr(pp, attr, saved)

    def test_tokenizer_file_drift_refused(self):
        write(self.tree.qwen / "tokenizer_config.json", '{"stub":"tampered"}')
        with self.assertRaises(PilotError):
            self.check()

    def test_model_context_limit_drift_refused(self):
        ident = write(self.tree.qwen / "config.json",
                      '{"max_position_embeddings":131072,"model_type":"qwen3"}')
        pp.QWEN_TOKENIZER_FILES = dict(pp.QWEN_TOKENIZER_FILES,
                                       **{"config.json": (ident["sha256"], ident["bytes"])})
        with self.assertRaises(PilotError):
            self.check()

    def test_live_prompt_constant_drift_refused(self):
        original = core.QUERY_SYSTEM
        try:
            core.QUERY_SYSTEM = original + " "
            with self.assertRaises(PilotError):
                self.check()
        finally:
            core.QUERY_SYSTEM = original
        self.assertEqual(assert_prompt_hashes(), core.PROMPT_HASHES)

    def test_existing_destination_refused(self):
        (self.tree.repo / pp.OUT).mkdir(parents=True)
        with self.assertRaises(PilotError):
            self.check()

    def test_symlinked_destination_refused(self):
        target = self.tree.base / "elsewhere"
        target.mkdir()
        (self.tree.repo / pp.OUT).parent.mkdir(parents=True, exist_ok=True)
        (self.tree.repo / pp.OUT).symlink_to(target)
        with self.assertRaises(PilotError):
            self.check()

class ForbiddenInputTests(PinnedCase):
    def test_forbidden_paths_cannot_be_registered(self):
        guard = pp.guard_for(self.repo())
        for label, path in pp.FORBIDDEN:
            resolved = Path(path) if str(path).startswith("/") else self.repo() / path
            with self.subTest(label=label):
                with self.assertRaises(PilotError):
                    guard.allow("sneaky", resolved)

    def test_scoring_targets_never_registered_during_a_real_check(self):
        report = pp.check_small_inputs(self.repo())
        targets = str(self.tree.prep / run.TARGET_EXPORT)
        self.assertNotIn(targets, [str(p) for p in [targets]] and [])
        self.assertIn("scoring_targets", report["forbidden_inputs"])
        self.assertNotIn("export_" + run.TARGET_EXPORT, report["allowed_inputs"])
        for label in report["allowed_inputs"]:
            self.assertNotIn("scoring_targets", label)
            self.assertNotIn("safetensors", label)

    @staticmethod
    def code_only(path):
        """Source with comments and string literals removed, so prose cannot mask
        or fake an API call. Only real code tokens survive."""
        import tokenize
        kept = []
        with open(path, "rb") as handle:
            for tok in tokenize.tokenize(handle.readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                kept.append(tok.string)
        return " ".join(kept)

    def test_no_model_index_or_encoder_symbols_in_the_script(self):
        code = self.code_only(HERE / "bridge_pilot_prompt_preflight.py")
        for forbidden in ("AutoModel", "AutoModelForCausalLM", "BitsAndBytesConfig",
                          "SentenceTransformer", "faiss", "read_index", "torch",
                          "np", "numpy", "mmap_mode", "safetensors", "generate",
                          "search", "IndexFlat", "encode"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, code.split())
        source = (HERE / "bridge_pilot_prompt_preflight.py").read_text()
        self.assertEqual(source.count("AutoTokenizer.from_pretrained"), 1)
        self.assertIn("local_files_only=True", source)
        self.assertIn("trust_remote_code=False", source)
    def test_only_the_tokenizer_is_imported_from_transformers(self):
        import ast
        tree = ast.parse((HERE / "bridge_pilot_prompt_preflight.py").read_text())
        from_transformers, plain = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("transformers"):
                from_transformers.update(a.name for a in node.names)
            if isinstance(node, ast.Import):
                plain.update(a.name for a in node.names)
        self.assertEqual(from_transformers, {"AutoTokenizer"})
        self.assertEqual(plain & {"torch", "faiss", "numpy", "sentence_transformers"}, set())
        self.assertIn("transformers", plain)  # imported only to record its version

    def test_no_corpus_or_index_loop(self):
        code = self.code_only(HERE / "bridge_pilot_prompt_preflight.py")
        self.assertNotIn("N_VECTORS", code.split())
        self.assertNotIn("23963971", code.split())
        source = (HERE / "bridge_pilot_prompt_preflight.py").read_text()
        self.assertIn("for qid in sorted(CHILD_COUNTS)", source)

# --------------------------------------------- sealed run, immutability, fsync

class SealedRunTests(PinnedCase):
    def run_preflight(self):
        os.environ["SLURM_JOB_ID"] = "synthetic-1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        buffer, out = io.StringIO(), sys.stdout
        real_hostname = pp.socket.gethostname
        # Substitute a synthetic compute-node name for the harness only. The
        # production psn001 gate is unchanged and is asserted separately below.
        pp.socket.gethostname = lambda: "synthetic-compute-node"
        try:
            sys.stdout = buffer
            pp.preflight(self.repo())
        finally:
            sys.stdout = out
            pp.socket.gethostname = real_hostname
            for key in ("SLURM_JOB_ID", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                os.environ.pop(key, None)
        return buffer.getvalue()

    def test_sealed_run_is_byte_identical_in_both_locations(self):
        self.run_preflight()
        out, archive = self.repo() / pp.OUT, pp.ARCHIVE
        names = ["prompt_preflight_start.json", "input_identities.json", "prompt_lengths.json",
                 "prompt_preflight_summary.json", "PROMPT_PREFLIGHT_SEAL.json"]
        for name in names:
            self.assertEqual((out / name).read_bytes(), (archive / name).read_bytes(), name)
        seal = json.loads((out / "PROMPT_PREFLIGHT_SEAL.json").read_text())
        self.assertEqual(seal["status"],
                         "SEALED_OUTCOME_FREE_PROMPT_PREFLIGHT_NOT_ACTIVATION")
        self.assertEqual(seal["preparation_seal_sha256"], pp.PREPARATION_SEAL_HASH)
        self.assertEqual(seal["preflight_seal_sha256"], pp.PREFLIGHT_SEAL_HASH)
        self.assertEqual(seal["checked_records"], 150)
        self.assertEqual(seal["categories_checked"], 6)
        self.assertEqual(seal["qwen_weight_files_opened"], 0)
        self.assertEqual(seal["qwen_generations"], 0)
        self.assertEqual(seal["pilot_searches"], 0)
        self.assertFalse(seal["scoring_targets_opened"])
        self.assertEqual(seal["experiment_state"], "NOT_FROZEN_NOT_RUN")
        self.assertEqual(seal["historical_global_lineage"], "UNESTABLISHED")
        self.assertEqual(seal["canonical_stage0"], "INCOMPLETE_GATES_UNCHANGED")
        self.assertEqual(seal["runtime_dependent"]["scope"], "RUNTIME_DEPENDENT")
        self.assertEqual(seal["prompt_constants"], core.PROMPT_HASHES)
        self.assertTrue(seal["all_known_prompts_within_context"])
        self.assertEqual(sorted(seal["maxima"]), sorted(c[0] for c in pp.CATEGORIES))
        # Every artifact hash in the seal matches the file actually written.
        for name, ident in seal["artifacts"].items():
            self.assertEqual(sha256((out / name).read_bytes()), ident["sha256"])
            self.assertEqual((out / name).stat().st_size, ident["bytes"])

    def test_sealed_artifacts_carry_no_research_content(self):
        self.run_preflight()
        out = self.repo() / pp.OUT
        blob = "".join((out / p.name).read_text() for p in sorted(out.iterdir()))
        for leak in ("London is a city", "Synthetic Parent", "یہ کیا ہے", "fact one",
                     "<|im_start|>", "im_end", "urbench_facts"):
            self.assertNotIn(leak, blob)

    def test_log_lines_contain_no_content(self):
        log = self.run_preflight()
        for line in log.splitlines():
            row = json.loads(line)
            self.assertIsInstance(row, dict)
            for leak in ("London is a city", "Synthetic Parent", "یہ کیا ہے", "fact one"):
                self.assertNotIn(leak, line)

    def test_rerun_is_refused_and_partial_files_survive(self):
        self.run_preflight()
        seal_before = (self.repo() / pp.OUT / "PROMPT_PREFLIGHT_SEAL.json").read_bytes()
        with self.assertRaises(PilotError):
            self.run_preflight()
        self.assertEqual((self.repo() / pp.OUT / "PROMPT_PREFLIGHT_SEAL.json").read_bytes(),
                         seal_before)

    def test_existing_archive_alone_refuses_the_run(self):
        pp.ARCHIVE.mkdir(parents=True)
        with self.assertRaises(PilotError):
            self.run_preflight()
        self.assertFalse((self.repo() / pp.OUT).exists())

    def test_context_blocker_is_sealed_before_it_is_raised(self):
        self.stub.tokens_per_prompt = pp.CONTEXT_LIMIT - 1023
        with self.assertRaises(PilotError) as ctx:
            self.run_preflight()
        self.assertIn("PROMPT_CONTEXT_BLOCKER", str(ctx.exception))
        out = self.repo() / pp.OUT
        summary = json.loads((out / "prompt_preflight_summary.json").read_text())
        self.assertEqual(summary["status"], "PROMPT_CONTEXT_BLOCKER")
        self.assertFalse(summary["all_known_prompts_within_context"])
        self.assertEqual(len(summary["failed"]), 50)  # both 1024-reserve categories
        self.assertTrue((out / "PROMPT_PREFLIGHT_SEAL.json").exists())

    def test_exclusive_create_prevents_overwrite(self):
        self.run_preflight()
        target = self.repo() / pp.OUT / "prompt_lengths.json"
        with self.assertRaises(FileExistsError):
            run.write_json(target, {"tampered": True})

    def test_fsync_called_for_files_and_directories(self):
        calls = []
        real = os.fsync
        try:
            os.fsync = lambda fd: (calls.append(fd), real(fd))[1]
            self.run_preflight()
        finally:
            os.fsync = real
        # Two fresh directories plus their parents, and every artifact and its parent.
        self.assertGreater(len(calls), 20)

    def test_missing_slurm_job_gate(self):
        os.environ.pop("SLURM_JOB_ID", None)
        with self.assertRaises(PilotError) as ctx:
            pp.preflight(self.repo())
        self.assertIn("COMPUTE_JOB_REQUIRED", str(ctx.exception))
        self.assertFalse((self.repo() / pp.OUT).exists())

    def test_login_node_is_refused(self):
        os.environ["SLURM_JOB_ID"] = "synthetic-1"
        real_hostname = pp.socket.gethostname
        pp.socket.gethostname = lambda: "psn001.example"
        try:
            with self.assertRaises(PilotError) as ctx:
                pp.preflight(self.repo())
            self.assertIn("LOGIN_NODE_REFUSED", str(ctx.exception))
        finally:
            pp.socket.gethostname = real_hostname
            os.environ.pop("SLURM_JOB_ID", None)
        self.assertFalse((self.repo() / pp.OUT).exists())

    def test_offline_environment_is_required(self):
        os.environ["SLURM_JOB_ID"] = "synthetic-1"
        os.environ.pop("HF_HUB_OFFLINE", None)
        real_hostname = pp.socket.gethostname
        pp.socket.gethostname = lambda: "synthetic-compute-node"
        try:
            with self.assertRaises(PilotError) as ctx:
                pp.preflight(self.repo())
            self.assertIn("OFFLINE_REQUIRED", str(ctx.exception))
        finally:
            pp.socket.gethostname = real_hostname
            os.environ.pop("SLURM_JOB_ID", None)

class MeasureOnlyTests(PinnedCase):
    def test_measure_only_writes_nothing(self):
        before = sorted(self.tree.base.rglob("*"))
        report = pp.measure_only(self.repo())
        after = sorted(self.tree.base.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(report["checked_records"], 150)
        self.assertTrue(report["all_within_context"])
        self.assertEqual(report["failed"], [])
        self.assertEqual(report["files_written"], 0)
        self.assertEqual(report["qwen_weight_files_opened"], 0)
        self.assertEqual(report["qwen_generations"], 0)
        self.assertEqual(report["pilot_searches"], 0)
        self.assertFalse(report["scoring_targets_opened"])
        self.assertEqual(sorted(report["maxima"]), sorted(c[0] for c in pp.CATEGORIES))

class JobScriptTests(unittest.TestCase):
    def test_cpu_only_resources(self):
        text = (HERE / "bridge_pilot_prompt_preflight.sbatch").read_text()
        self.assertNotIn("--gres", text)
        self.assertNotIn("gpu", text.replace("q_intel_share_L20", ""))
        self.assertIn("#SBATCH --partition=q_intel_share_L20", text)
        self.assertIn("#SBATCH --cpus-per-task=4", text)
        self.assertIn("#SBATCH --mem=16G", text)
        self.assertIn("#SBATCH --time=00:30:00", text)
        self.assertIn("set -euo pipefail", text)
        self.assertIn("cd /mnt/home/user41/URBench", text)
        self.assertIn("/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B", text)
        self.assertIn("/mnt/home/user41/URBench/logs/bridge_n25_prompt_preflight_v1_%j.log", text)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("conda install", text)
        self.assertIn("--preflight", text)

if __name__ == "__main__":
    unittest.main(verbosity=2)
