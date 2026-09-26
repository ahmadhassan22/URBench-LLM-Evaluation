"""Local synthetic tests. Never run against, patch, or import the remote project.

All fixture writes occur in TemporaryDirectory. Production CLI has no test-mode
or identity-override flags; unittest mocks are confined to this test process.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("efbpt_target_reachability_probe.py")
SPEC = importlib.util.spec_from_file_location("probe_under_test", SOURCE)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n").encode()


def lines(rows):
    return b"".join(encoded(row) for row in rows)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.cohort = dict(p.QIDS)
        self.target_rows = []
        self.titles = []
        for qid, n in self.cohort.items():
            children = []
            for _ in range(n):
                title = f"target {len(self.titles)}"
                children.append({"source_instance_id": f"source-{len(self.titles)}",
                                 "title": title, "normalized_title": title})
                self.titles.append(title)
            self.target_rows.append({"qid": qid, "children": children})
        self.meta = [{"title": self.titles[i] if i < 16 else f"candidate {i}",
                      "text": "اردو text café\nwith escaped newline"} for i in range(120)]
        self.meta[110]["title"] = self.titles[16]  # present but never retrieved
        self.meta[119]["title"] = "TARGET_0"  # second normalized row for target 0
        self.meta_raw = lines(self.meta)
        self.meta_path = root / p.META
        self.meta_path.parent.mkdir(parents=True)
        self.meta_path.write_bytes(self.meta_raw)
        meta_spec = {"path": str(self.meta_path), "sha256": p.digest(self.meta_raw),
                     "bytes": len(self.meta_raw)}
        offset = 0
        proofs = []
        for i, row in enumerate(self.meta):
            raw = encoded(row)
            proofs.append({"global_row": i, "byte_offset": offset, "metadata_line_sha256": p.digest(raw)})
            offset += len(raw)
        candidates = [{"global_row": i, "title": self.meta[i]["title"], "score": 200.0 - i}
                      for i in range(100)]
        self.targets_raw = lines(self.target_rows)
        target_spec = {"path": str(root / p.PINS["targets"][0]), "sha256": p.digest(self.targets_raw),
                       "bytes": len(self.targets_raw)}
        self.activation = {"assets": {"metadata": meta_spec}, "exports": {"targets": target_spec}}
        self.activation_raw = encoded(self.activation)
        self.activation_sha = p.digest(self.activation_raw)
        self.records = []
        for qid in self.cohort:
            for arm in p.ARMS:
                pred = {"qid": qid, "arm": arm, "candidates": copy.deepcopy(candidates),
                        "ranked_titles": p.aggregate(candidates)}
                self.records.append({"qid": qid, "arm": arm, "activation_sha256": self.activation_sha,
                    "search_budget": 100, "prediction": pred, "candidate_provenance": copy.deepcopy(proofs[:100])})
        self.predictions = [r["prediction"] for r in self.records]
        records_raw, predictions_raw = lines(self.records), lines(self.predictions)
        self.scores_raw = encoded({"accepted_qids": 25, "accepted_pairs": 36})
        artifact = lambda raw: {"sha256": p.digest(raw), "bytes": len(raw)}
        retrieval = {"status": "SEALED_PREDICTIONS", "activation_sha256": self.activation_sha,
            "targets_sha256": target_spec["sha256"], "artifacts": {
                "prediction_records.jsonl": artifact(records_raw), "predictions.jsonl": artifact(predictions_raw)}}
        retrieval_raw = encoded(retrieval)
        provenance = {"targets": target_spec}
        for label, key, raw in (("predictions_seal", "retrieval_seal", retrieval_raw),
                                ("predictions", "predictions", predictions_raw),
                                ("prediction_records", "records", records_raw)):
            provenance[label] = {"path": str(root / p.PINS[key][0]), **artifact(raw)}
        scoring = {"status": "SEALED_SCORES", "activation_sha256": self.activation_sha,
                   "artifacts": {"scores.json": artifact(self.scores_raw)}, "provenance": provenance}
        payloads = {"activation": self.activation_raw, "retrieval_seal": retrieval_raw,
            "scoring_seal": encoded(scoring), "scores": self.scores_raw, "records": records_raw,
            "predictions": predictions_raw, "targets": self.targets_raw,
            "core": b'def norm(t):\n    return " ".join(str(t).replace("_", " ").strip().lower().split())\n'}
        self.pins = {}
        for name, raw in payloads.items():
            relative = p.PINS[name][0]
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            self.pins[name] = (relative, p.digest(raw), len(raw))
        self.probe_path = root / p.CODE / "efbpt_target_reachability_probe.py"
        self.wrapper_path = root / p.CODE / "efbpt_target_reachability_probe.sbatch"
        self.probe_path.write_bytes(SOURCE.read_bytes())
        self.wrapper_path.write_bytes(SOURCE.with_suffix(".sbatch").read_bytes())
        self.probe_sha = p.digest(self.probe_path.read_bytes())
        self.wrapper_sha = p.digest(self.wrapper_path.read_bytes())
        self.targets = p.validate_targets(self.target_rows, self.cohort)

    def activate(self, stack):
        for name, value in (("ROOT", self.root), ("PINS", self.pins), ("META_BYTES", len(self.meta_raw)),
                            ("META_ROWS", 120), ("__file__", str(self.probe_path))):
            stack.enter_context(patch.object(p, name, value))
        # The production interpreter gate is covered separately. Test host differs.
        stack.enter_context(patch.object(p.sys, "version_info", (3, 10, 19)))
        stack.enter_context(patch.object(p.sys, "dont_write_bytecode", True))

    def prepared(self):
        return p.prepare(self.root, self.probe_sha, self.wrapper_sha)

    def cli(self, mode):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = p.main([mode, "--repo", str(self.root), "--probe-sha256", self.probe_sha,
                           "--wrapper-sha256", self.wrapper_sha])
        return code, output.getvalue()


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="urbench-probe-test-")
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)
        self.stack = contextlib.ExitStack()
        self.f.activate(self.stack)

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def scanned(self, **kwargs):
        context = self.f.prepared()
        params = dict(path=self.f.meta_path, expected_sha=p.digest(self.f.meta_raw),
            expected_bytes=len(self.f.meta_raw), expected_rows=120, targets=self.f.targets,
            controls=context["controls"], progress_seconds=60)
        params.update(kwargs)
        return p.scan_metadata(**params)

    def test_read_only_preflight_does_not_open_metadata_or_write(self):
        before = {str(x): x.read_bytes() for x in self.root.rglob("*") if x.is_file()}
        real_open = p.open_input
        def guard(path):
            self.assertNotEqual(Path(path), self.f.meta_path)
            return real_open(path)
        with patch.object(p, "open_input", guard):
            code, message = self.f.cli("--check-inputs")
        self.assertEqual(code, 0, message)
        self.assertFalse((self.root / p.OUT).exists())
        self.assertEqual(before, {str(x): x.read_bytes() for x in self.root.rglob("*") if x.is_file()})

    def test_full_run_seal_counts_and_no_other_writes(self):
        before = {str(x): x.read_bytes() for x in self.root.rglob("*") if x.is_file()}
        code, message = self.f.cli("--run")
        self.assertEqual(code, 0, message)
        out = self.root / p.OUT
        self.assertEqual({x.name for x in out.iterdir()}, {"START.json", "reachability.json", "DIAGNOSTIC_SEAL.json"})
        result = json.loads((out / "reachability.json").read_bytes())
        self.assertEqual(result["summary"]["present_titles"], 17)
        self.assertEqual(result["summary"]["absent_titles"], 19)
        self.assertEqual(result["positive_controls"]["titles_observed_anywhere"], 16)
        self.assertEqual(result["scan"]["candidate_rows_verified"], 100)
        self.assertEqual(result["targets"][0]["matching_rows"], 2)
        self.assertEqual(result["targets"][0]["first_row"], 0)
        seal = json.loads((out / "DIAGNOSTIC_SEAL.json").read_bytes())
        for name, expected in seal["artifacts"].items():
            raw = (out / name).read_bytes()
            self.assertEqual(expected, {"sha256": p.digest(raw), "bytes": len(raw)})
        for path, raw in before.items():
            self.assertEqual(Path(path).read_bytes(), raw)
        after_files = {str(x) for x in self.root.rglob("*") if x.is_file()}
        self.assertEqual(after_files - set(before), {str(out / n) for n in seal["artifacts"]} | {str(out / "DIAGNOSTIC_SEAL.json")})

    def test_no_second_pass_over_metadata(self):
        calls = []
        real_open = p.open_input
        def tracker(path):
            if Path(path) == self.f.meta_path:
                calls.append(path)
            return real_open(path)
        with patch.object(p, "open_input", tracker):
            code, message = self.f.cli("--run")
        self.assertEqual(code, 0, message)
        self.assertEqual(len(calls), 1)

    def test_input_hash_drift_stops_before_outputs(self):
        path = self.root / self.f.pins["records"][0]
        raw = path.read_bytes().replace(b"target 0", b"target X", 1)
        path.write_bytes(raw)
        code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("INPUT_HASH_DRIFT", message)
        self.assertFalse((self.root / p.OUT).exists())

    def test_existing_output_and_dangling_symlink_stop(self):
        out = self.root / p.OUT
        out.symlink_to(self.root / "does-not-exist")
        code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("OUTPUT_EXISTS", message)
        self.assertTrue(out.is_symlink())

    def test_partial_write_preserved_and_retry_refused(self):
        original = p.Output.write
        def fail_second(output, name, value):
            if name == "reachability.json":
                raise OSError("synthetic storage failure")
            return original(output, name, value)
        with patch.object(p.Output, "write", fail_second):
            code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        out = self.root / p.OUT
        start = (out / "START.json").read_bytes()
        self.assertFalse((out / "DIAGNOSTIC_SEAL.json").exists())
        code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("OUTPUT_EXISTS", message)
        self.assertEqual((out / "START.json").read_bytes(), start)

    def test_metadata_hash_mismatch_preserves_start_no_verdict(self):
        # Change a text byte outside the sampled raw-row controls, preserving size.
        rows = self.f.meta_raw.splitlines(keepends=True)
        rows[118] = rows[118].replace(b"with escaped", b"With escaped", 1)
        self.f.meta_path.write_bytes(b"".join(rows))
        code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("METADATA_HASH_DRIFT", message)
        out = self.root / p.OUT
        self.assertEqual({x.name for x in out.iterdir()}, {"START.json"})

    def test_metadata_schema_utf8_duplicates_and_blank_rows_rejected(self):
        cases = [(b'{"title":"x","title":"y","text":"z"}\n', "DUPLICATE_JSON_KEY"),
                 (b'{"title":"x","text":"\xff"}\n', "INVALID_JSON_OR_UTF8"),
                 (b'{"title":"x"}\n', "METADATA_SCHEMA"),
                 (b'{"title":"x","text":NaN}\n', "NONFINITE_JSON"),
                 (b'{"title":"___","text":"x"}\n', "METADATA_STRINGS"),
                 (b'\n', "INVALID_JSON_OR_UTF8")]
        for raw, error in cases:
            with self.subTest(error=error):
                self.f.meta_path.write_bytes(raw)
                with self.assertRaisesRegex(p.ProbeError, error):
                    p.scan_metadata(self.f.meta_path, p.digest(raw), len(raw), 1, self.f.targets, {})

    def test_row_limit_and_missing_final_newline(self):
        with patch.object(p, "MAX_LINE", 10):
            with self.assertRaisesRegex(p.ProbeError, "ROW_TOO_LONG"):
                self.scanned()
        raw = encoded(self.f.meta[0]).rstrip(b"\n")
        self.f.meta_path.write_bytes(raw)
        with self.assertRaisesRegex(p.ProbeError, "LINE_BOUNDARY"):
            p.scan_metadata(self.f.meta_path, p.digest(raw), len(raw), 1, self.f.targets, {})

    def test_byte_and_row_count_drifts(self):
        with self.assertRaisesRegex(p.ProbeError, "METADATA_SIZE_BEFORE_SCAN"):
            self.scanned(expected_bytes=len(self.f.meta_raw)+1)
        with self.assertRaisesRegex(p.ProbeError, "METADATA_ROW_COUNT_DRIFT"):
            self.scanned(expected_rows=121)
        with self.assertRaisesRegex(p.ProbeError, "METADATA_BOUNDARY_EXCEEDED"):
            self.scanned(expected_rows=119)

    def test_candidate_raw_row_proof_mismatch(self):
        context = self.f.prepared()
        for field, wrong in (("byte_offset", 99), ("metadata_line_sha256", "0"*64), ("title", "wrong")):
            controls = copy.deepcopy(context["controls"])
            controls[0][field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(p.ProbeError, "OBSERVED_CANDIDATE_ROW_MISMATCH"):
                self.scanned(controls=controls)

    def test_missing_control_row_is_failure(self):
        controls = copy.deepcopy(self.f.prepared()["controls"])
        controls[999] = controls[0]
        with self.assertRaisesRegex(p.ProbeError, "CANDIDATE_CONTROL_ROWS_MISSING"):
            self.scanned(controls=controls)

    def test_positive_title_absence_refused(self):
        context = self.f.prepared()
        counts, scan = self.scanned()
        counts["target 0"]["matching_rows"] = 0
        with self.assertRaisesRegex(p.ProbeError, "POSITIVE_CONTROL_TITLE_MISSING"):
            p.make_result(context, counts, scan, self.f.cohort)

    def test_do_not_infer_all_children_from_successful_question(self):
        context = self.f.prepared()
        # qid with three children includes two observed and one not observed.
        parent = self.f.target_rows[11]
        child_titles = [c["normalized_title"] for c in parent["children"]]
        self.assertTrue(any(t in context["anywhere"] for t in child_titles))
        self.assertTrue(any(t not in context["anywhere"] for t in child_titles))

    def test_title_observed_under_other_question_is_positive_control(self):
        records = copy.deepcopy(self.f.records)
        owner = self.f.targets["target 0"]["qid"]
        replacement = {"global_row": 100, "title": self.f.meta[100]["title"], "score": 200.0}
        offset = sum(len(encoded(row)) for row in self.f.meta[:100])
        for record in records:
            if record["qid"] == owner:
                record["prediction"]["candidates"][0] = replacement
                record["candidate_provenance"][0] = {"global_row": 100, "byte_offset": offset,
                                                     "metadata_line_sha256": p.digest(encoded(self.f.meta[100]))}
                record["prediction"]["ranked_titles"] = p.aggregate(record["prediction"]["candidates"])
        _, anywhere, own, _ = p.collect_controls(records, [r["prediction"] for r in records],
            self.f.targets, self.f.cohort, self.f.activation_sha, 120)
        self.assertIn("target 0", anywhere)
        self.assertEqual(own["target 0"], set())

    def test_target_duplicate_missing_and_wrong_norm(self):
        for mutate, error in ((lambda r: r.pop(), "TARGET_ROW_COUNT"),
            (lambda r: r[0]["children"][0].update(normalized_title="wrong"), "TARGET_NORMALIZATION"),
            (lambda r: r[1]["children"].__setitem__(0, r[0]["children"][0]), "TARGET_DISTINCTNESS")):
            rows = copy.deepcopy(self.f.target_rows)
            mutate(rows)
            with self.assertRaisesRegex(p.ProbeError, error):
                p.validate_targets(rows, self.f.cohort)

    def test_duplicate_prediction_and_ranking_changes(self):
        for change, error in ((lambda r: r.__setitem__(1, copy.deepcopy(r[0])), "PREDICTION_IDENTITY"),
            (lambda r: r[0]["prediction"]["ranked_titles"].reverse(), "FROZEN_RANKING_DRIFT")):
            records = copy.deepcopy(self.f.records)
            change(records)
            with self.assertRaisesRegex(p.ProbeError, error):
                p.collect_controls(records, [r["prediction"] for r in records], self.f.targets,
                                   self.f.cohort, self.f.activation_sha, 120)

    def test_separate_prediction_view_drift(self):
        predictions = copy.deepcopy(self.f.predictions)
        predictions[0]["arm"] = "E"
        with self.assertRaisesRegex(p.ProbeError, "PREDICTION_VIEW_DRIFT"):
            p.collect_controls(self.f.records, predictions, self.f.targets, self.f.cohort, self.f.activation_sha, 120)

    def test_descriptor_no_guessing_and_ambiguity(self):
        wanted = self.root / p.META
        with self.assertRaisesRegex(p.ProbeError, "UNIQUE_EXACT_DESCRIPTOR"):
            p.find_descriptor({}, self.root, wanted)
        spec = self.f.activation["assets"]["metadata"]
        with self.assertRaisesRegex(p.ProbeError, "UNIQUE_EXACT_DESCRIPTOR"):
            p.find_descriptor([spec, spec], self.root, wanted)

    def test_source_normalizer_ast_and_unicode_behavior(self):
        p.check_normalizer(b'def norm(x):\n    return " ".join(str(x).replace("_", " ").strip().lower().split())\n')
        with self.assertRaisesRegex(p.ProbeError, "NORMALIZER_EXPRESSION_DRIFT"):
            p.check_normalizer(b'def norm(x):\n    return x.lower()\n')
        self.assertEqual(p.norm("  HELLO_world\t "), "hello world")
        self.assertNotEqual(p.norm("café"), p.norm("cafe\u0301"))

    def test_input_symlink_refused(self):
        target = self.root / self.f.pins["scores"][0]
        copy_path = target.with_name("other.json")
        copy_path.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(copy_path)
        code, message = self.f.cli("--check-inputs")
        self.assertEqual(code, 2)
        self.assertIn("SYMLINK_REFUSED", message)

    def test_input_mutation_during_scan_is_detected(self):
        original = p.strict_json
        changed = False
        def mutator(raw, label):
            nonlocal changed
            obj = original(raw, label)
            if label == "metadata row 0" and not changed:
                changed = True
                with self.f.meta_path.open("ab") as stream:
                    stream.write(encoded(self.f.meta[0]))
            return obj
        with patch.object(p, "strict_json", mutator), self.assertRaises(p.ProbeError):
            self.scanned()

    def test_small_input_mutation_before_sealing_blocks_result(self):
        original = p.scan_metadata
        def mutator(*args, **kwargs):
            output = original(*args, **kwargs)
            path = self.root / self.f.pins["targets"][0]
            path.write_bytes(path.read_bytes() + b" ")
            return output
        with patch.object(p, "scan_metadata", mutator):
            code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("INPUT_SIZE_DRIFT", message)
        self.assertFalse((self.root / p.OUT / "DIAGNOSTIC_SEAL.json").exists())

    def test_production_has_no_identity_override_flags(self):
        source = SOURCE.read_text()
        for forbidden in ("--fixture", "--test-mode", "--metadata-sha256", "--skip", "--force", "--resume"):
            self.assertNotIn('"' + forbidden + '"', source)
        imports = {n.names[0].name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Import)}
        self.assertFalse(imports & {"torch", "faiss", "numpy", "transformers", "subprocess", "requests"})

    def test_interrupted_scan_preserves_start(self):
        with patch.object(p, "scan_metadata", side_effect=p.ProbeError("INTERRUPTED_SIGNAL_15")):
            code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("INTERRUPTED_SIGNAL_15", message)
        self.assertEqual({x.name for x in (self.root / p.OUT).iterdir()}, {"START.json"})

    def test_partial_result_file_is_preserved(self):
        original = p.Output.write
        def interrupt_write(output, name, value):
            if name == "reachability.json":
                with (output.path / name).open("xb") as stream:
                    stream.write(b'{"partial":')
                raise OSError("synthetic mid-write interruption")
            return original(output, name, value)
        with patch.object(p.Output, "write", interrupt_write):
            code, _ = self.f.cli("--run")
        self.assertEqual(code, 2)
        out = self.root / p.OUT
        self.assertEqual((out / "reachability.json").read_bytes(), b'{"partial":')
        self.assertFalse((out / "DIAGNOSTIC_SEAL.json").exists())
        code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("OUTPUT_EXISTS", message)
        self.assertEqual((out / "reachability.json").read_bytes(), b'{"partial":')

    def test_wrong_interpreter_and_probe_identity_are_rejected(self):
        with patch.object(p.sys, "version_info", (3, 12, 13)):
            code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("PINNED_PYTHON", message)
        self.f.probe_sha = "0" * 64
        code, message = self.f.cli("--run")
        self.assertEqual(code, 2)
        self.assertIn("INPUT_HASH_DRIFT", message)
        self.assertFalse((self.root / p.OUT).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
