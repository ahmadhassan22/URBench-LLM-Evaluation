#!/usr/bin/env python3
"""Offline tests for bridge_v2_query_dev. No model, no index, no corpus scan.

Synthetic fixtures cover prompt identity, message construction, the query cap
and its fallback, blinding invariants, the input-only evidence-relevance sheet,
dev-code identity, Git provenance, failure reporting and search validation.
Tests on the sealed pilot artifacts and the frozen activation read them
read-only when present, and skip when they are not. The environment test must
run under the frozen interpreter.

Run:  /mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B \
          eval/error_analysis_tests/efbpt/bridge_v2_query_dev_test.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge_pilot_core as core
import bridge_v2_query_dev as dev

PASSED, SKIPPED, FAILED = [], [], []


def test(fn):
    name = fn.__name__
    try:
        result = fn()
    except SkipTest as exc:
        SKIPPED.append((name, str(exc)))
        print("SKIP %s  (%s)" % (name, exc))
    except Exception as exc:                                    # noqa: BLE001
        FAILED.append((name, "%s: %s" % (type(exc).__name__, exc)))
        print("FAIL %s  %s: %s" % (name, type(exc).__name__, exc))
    else:
        PASSED.append(name)
        print("ok   %s%s" % (name, "  " + result if result else ""))
    return fn


class SkipTest(Exception):
    pass


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_raises(message, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:                                           # noqa: BLE001
        return
    raise AssertionError("expected failure: " + message)


# ------------------------------------------------------------------ fixtures

class FakeEncoderTokenizer:
    """One id per whitespace word plus two specials. Deterministic, no files."""

    def __call__(self, text, add_special_tokens=True, truncation=False):
        n = len(text.split())
        return {"input_ids": list(range(n + (2 if add_special_tokens else 0)))}


def synthetic_parent(qid="0123456789abcdef0123", words=260):
    text = " ".join("w%d" % i for i in range(words))
    text = text.replace("w7 w8", "Alpha Beta", 1)
    page = {"raw_page_id": "1", "raw_url": "https://en.wikipedia.org/wiki/Synthetic",
            "raw_title": "Synthetic Page", "raw_text": text,
            "raw_text_sha256": core.sha256(text),
            "locations": [{"shard_index": 0, "blob_path": "/dev/null",
                           "blob_sha256": "0" * 64, "row_group": 0,
                           "row_in_group": 0, "row_in_shard": 0}],
            "lookup_decision": "UNIQUE_EXACT_CASE_PAGE_IN_NORMALIZED_BUCKET",
            "normalized_bucket_rows": 1, "normalized_bucket_distinct_pages": 1}
    row = {"qid": qid, "question_ur": "کیا ایک ٹیسٹ؟",
           "parent_source_instance_id": "s0-" + "a" * 64,
           "parent_title": "Synthetic Page",
           "parent_normalized_title": core.norm("Synthetic Page"),
           "page": page, "chunks": core.local_chunks(page)}
    core.validate_parent(row)
    return row


def synthetic_d_state(row, quotes):
    raw = json.dumps({"facts": [{"chunk_id": row["chunks"][0]["chunk_id"], "quote": q}
                                for q in quotes]}, ensure_ascii=False)
    state = core.parse_state(raw, row, "D")
    expect(not state["format_failure"], "fixture state must parse")
    return state


SALT = "0123456789abcdef" * 4


def one_qid_ctx(quotes=("w10 w11 w12",)):
    row = synthetic_parent()
    state = synthetic_d_state(row, list(quotes)) if quotes else core.parse_state(
        '{"facts":[]}', row, "D")
    qid = row["qid"]
    ctx = {"rows": {qid: row}, "states": {qid: state},
           "frozen_queries": {(qid, arm): "frozen query " + arm for arm in dev.ARMS_V2}}
    queries = {(qid, arm): {"result": {"query": "revised query " + arm}}
               for arm in dev.ARMS_V2}
    return row, ctx, queries


def sealed(name):
    path = dev.ROOT / dev.PINS[name][0]
    if not path.is_file():
        raise SkipTest("sealed artifact absent: " + dev.PINS[name][0])
    return path


# --------------------------------------------------------- 1. prompt identity

@test
def test_v2_prompt_hash_pinned():
    expect(core.sha256(dev.QUERY_SYSTEM_V2) == dev.QUERY_SYSTEM_V2_SHA256,
           "QUERY_SYSTEM_V2 does not match its pinned hash")
    return dev.QUERY_SYSTEM_V2_SHA256[:16]


@test
def test_frozen_constants_untouched():
    actual = core.assert_prompt_hashes()
    expect(actual == core.PROMPT_HASHES, "frozen prompt constants drifted")
    expect(actual["query"] == "f4ee67a37eb99000eccd4eba8d609cf6a987d73caec3a0ebcc7e85d7b11e1d25",
           "frozen query constant drifted")
    out = dev.assert_prompt_constants()
    expect(out["frozen"] == dict(sorted(core.PROMPT_HASHES.items())), "frozen set mismatch")
    expect(out["query_v2"] != out["frozen"]["query"], "v2 prompt is not a change")


@test
def test_query_system_v2_word_budget_stated():
    expect("32 whitespace-separated words" in dev.QUERY_SYSTEM_V2, "word budget missing")
    expect("one query only" in dev.QUERY_SYSTEM_V2, "single-query instruction missing")
    for clause in ("negation", "date or time limit", "transliterate",
                   "is not an answer", "does not bear on the question"):
        expect(clause in dev.QUERY_SYSTEM_V2, "missing required clause: " + clause)


# ------------------------------------------------ 2. message construction

@test
def test_arm_envelopes():
    row = synthetic_parent()
    state = synthetic_d_state(row, ["w10 w11 w12", "w20 w21"])
    seen = {}
    for arm in dev.ARMS_V2:
        messages = dev.parent_query_messages_v2(row, arm, state if arm == "D" else None)
        expect(len(messages) == 2, "two messages expected")
        expect(messages[0]["content"] == dev.QUERY_SYSTEM_V2, "system message is not V2")
        head = messages[1]["content"].split("\nReturn the search query only.")[0]
        payload = core.strict_json(head)
        expect(set(payload) == {"available_evidence", "known_source_title", "question_ur"},
               "user payload keys drifted")
        expect(payload["question_ur"] == row["question_ur"], "question altered")
        seen[arm] = payload
    expect(seen["A"]["known_source_title"] == "", "arm A must carry no title")
    expect(seen["A"]["available_evidence"] == "", "arm A must carry no evidence")
    expect(seen["P"]["known_source_title"] == row["parent_title"], "arm P title")
    expect(seen["P"]["available_evidence"] == "", "arm P must carry no evidence")
    expect(seen["B"]["available_evidence"] == row["page"]["raw_text"], "arm B full page")
    expect(seen["D"]["available_evidence"] == "w10 w11 w12\nw20 w21", "arm D fact join")


@test
def test_question_passes_through_in_logical_order():
    row = synthetic_parent()
    messages = dev.parent_query_messages_v2(row, "P")
    head = messages[1]["content"].split("\nReturn the search query only.")[0]
    payload = core.strict_json(head)
    expect(payload["question_ur"] == row["question_ur"], "question not byte-identical")
    expect(payload["question_ur"][0] == "ک", "leading code point reordered")
    expect(payload["question_ur"][-1] == "؟", "trailing code point reordered")


@test
def test_empty_d_state_gives_empty_evidence():
    row = synthetic_parent()
    state = core.parse_state('{"facts":[]}', row, "D")
    expect(state["items"] == [], "fixture should have no accepted items")
    messages = dev.parent_query_messages_v2(row, "D", state)
    payload = core.strict_json(
        messages[1]["content"].split("\nReturn the search query only.")[0])
    expect(payload["available_evidence"] == "", "empty state must yield empty evidence")


@test
def test_tampered_state_is_refused():
    row = synthetic_parent()
    state = synthetic_d_state(row, ["w10 w11 w12"])
    tampered = json.loads(json.dumps(state))
    tampered["items"][0]["quote"] = "w10 w11 w12 INJECTED"
    expect_raises("STATE_RECORD_DRIFT not raised",
                  dev.parent_query_messages_v2, row, "D", tampered)


@test
def test_unknown_arm_refused():
    row = synthetic_parent()
    for arm in ("C", "E", "X", ""):
        expect_raises("arm %r accepted" % arm, dev.parent_query_messages_v2, row, arm)


# -------------------------------------------------------- 3. query cap paths

@test
def test_cap_query_word_budget():
    tok = FakeEncoderTokenizer()
    raw = " ".join("word%d" % i for i in range(50))
    out = core.cap_query(raw, "q", "D", tok, sealed_a="fallback query")
    expect(len(out["query"].split()) == 32, "32-word cap not applied")
    expect(out["word_cap_removed"] == 18, "word_cap_removed wrong")
    expect(out["fallback"] is None, "no fallback expected")


@test
def test_cap_query_empty_uses_sealed_a_for_non_a_arms():
    tok = FakeEncoderTokenizer()
    for arm in ("P", "B", "D"):
        out = core.cap_query("", "question text", arm, tok, sealed_a="sealed a query")
        expect(out["fallback"] == "SEALED_A", "arm %s did not use SEALED_A" % arm)
        expect(out["query"] == "sealed a query", "sealed A query not reused verbatim")
        expect(out["model_empty"] is True, "model_empty flag not set")
    expect_raises("missing sealed_a accepted",
                  core.cap_query, "", "question text", "D", tok)


@test
def test_cap_query_empty_arm_a_uses_question():
    tok = FakeEncoderTokenizer()
    out = core.cap_query("   ", "question text here", "A", tok)
    expect(out["fallback"] == "URDU_QUESTION", "arm A fallback wrong")
    expect(out["query"] == "question text here", "arm A fallback text wrong")


@test
def test_cap_query_encoder_trim():
    tok = FakeEncoderTokenizer()
    raw = " ".join("w%d" % i for i in range(32))
    out = core.cap_query(raw, "q", "A", tok)
    expect(out["encoder_tokens"] <= 128, "encoder token budget exceeded")
    expect(out["query"].strip(), "empty query produced")


# ------------------------------------------------------ 4. blinding and sheet

@test
def test_worksheet_hides_version_and_outcomes():
    row, ctx, queries = one_qid_ctx()
    qid = row["qid"]
    cells = dev.worksheet_cells(ctx, queries, [qid], SALT)
    expect(len(cells) == 8, "expected 8 cells for one qid")
    sheet = dev.worksheet_sheet(cells)
    for entry in sheet:
        expect(not (set(entry) & dev.WORKSHEET_FORBIDDEN_KEYS),
               "worksheet leaks a forbidden key")
        for field in dev.RUBRIC_FIELDS:
            expect(entry[field] is None, "rubric field pre-filled: " + field)
    blob = json.dumps(sheet, ensure_ascii=False)
    for token in ("\"version\"", "frozen\":", "revised\":", "ranked_titles", "recall",
                  SALT):
        expect(token not in blob, "worksheet contains " + token)
    relevance = dev.relevance_cells(ctx, [qid], SALT)
    key = dev.worksheet_key(cells, relevance)
    query_key = [k for k in key if k["sheet"] == "query"]
    expect(len(query_key) == 8 and {k["version"] for k in query_key} == {"frozen", "revised"},
           "key file must carry both versions")
    expect({k["id"] for k in query_key} == {c["cell_id"] for c in cells},
           "key does not cover every cell")
    expect({k["id"] for k in key if k["sheet"] == "evidence_relevance"}
           == {c["relevance_id"] for c in relevance}, "key does not cover relevance rows")
    return "8 cells, no leak"


@test
def test_cell_ids_depend_on_the_per_run_salt():
    row, ctx, queries = one_qid_ctx()
    a = {c["cell_id"] for c in dev.worksheet_cells(ctx, queries, [row["qid"]], SALT)}
    b = {c["cell_id"] for c in dev.worksheet_cells(ctx, queries, [row["qid"]],
                                                   "f" * 64)}
    expect(not (a & b), "cell ids do not change with the salt")
    expect(not hasattr(dev, "REVIEW_SALT"), "a fixed review salt is still in the source")
    expect_raises("short salt accepted", dev.review_id, "short", "query", qid="x")


@test
def test_worksheet_shows_each_condition_actual_input():
    row, ctx, queries = one_qid_ctx()
    sheet = dev.worksheet_sheet(dev.worksheet_cells(ctx, queries, [row["qid"]], SALT))
    evidence = {entry["available_evidence"] for entry in sheet}
    expect(row["page"]["raw_text"] in evidence, "arm B page not shown to reviewer")
    expect("w10 w11 w12" in evidence, "arm D evidence not shown to reviewer")
    expect("" in evidence, "arms A/P empty evidence not shown")
    titles = {entry["known_source_title"] for entry in sheet}
    expect(titles == {"", row["parent_title"]}, "titles not shown as fed")


# ------------------------------------------- 4b. input-only evidence relevance

@test
def test_relevance_sheet_is_input_only():
    row, ctx, _ = one_qid_ctx()
    relevance = dev.relevance_cells(ctx, [row["qid"]], SALT)
    expect(sorted(c["arm"] for c in relevance) == ["B", "D"],
           "only inputs carrying parent evidence get a relevance row")
    sheet = dev.relevance_sheet(relevance)
    for entry in sheet:
        expect(not (set(entry) & dev.RELEVANCE_FORBIDDEN_KEYS), "relevance sheet leaks a key")
        expect("query" not in entry, "relevance sheet shows a query")
        for field in dev.EVIDENCE_RELEVANCE_FIELDS:
            expect(entry[field] is None, "relevance field pre-filled: " + field)
    blob = json.dumps(sheet, ensure_ascii=False)
    for token in ("frozen query", "revised query", "\"version\"", "recall"):
        expect(token not in blob, "relevance sheet contains " + token)
    return "%d rows" % len(sheet)


@test
def test_relevance_skips_empty_d_state():
    row, ctx, _ = one_qid_ctx(quotes=())
    relevance = dev.relevance_cells(ctx, [row["qid"]], SALT)
    expect([c["arm"] for c in relevance] == ["B"], "empty D evidence must get no row")


@test
def test_query_cells_link_to_relevance_rows_by_input_not_version():
    row, ctx, queries = one_qid_ctx()
    cells = dev.worksheet_cells(ctx, queries, [row["qid"]], SALT)
    relevance = {c["arm"]: c["relevance_id"]
                 for c in dev.relevance_cells(ctx, [row["qid"]], SALT)}
    for cell in cells:
        expect(cell["evidence_relevance_id"] == relevance.get(cell["arm"]),
               "query cell not linked to its input's relevance row: " + cell["arm"])
    by_arm = {}
    for cell in cells:
        by_arm.setdefault(cell["arm"], set()).add(cell["evidence_relevance_id"])
    expect(all(len(v) == 1 for v in by_arm.values()),
           "frozen and revised cells of one input must share one relevance id")
    expect(by_arm["A"] == {None} and by_arm["P"] == {None},
           "arms without parent evidence must not link a relevance row")


@test
def test_r2_distinguishes_contributed_from_repeated_terms():
    expect("R2_evidence_contribution" in dev.RUBRIC_FIELDS, "R2 field not replaced")
    expect("R2_evidence_use" not in dev.RUBRIC_FIELDS, "old R2 field still present")
    expect(dev.R2_VALUES == ("USES_CONTRIBUTED_TERM", "QUESTION_OR_TITLE_TERMS_ONLY",
                             "NOT_APPLICABLE"), "R2 values drifted")
    expect(dev.E1_VALUES == ("ADDS_RELEVANT_INFORMATION", "ONLY_RESTATES_QUESTION_OR_TITLE",
                             "OFF_QUESTION"), "E1 values drifted")
    expect(dev.EVIDENCE_RELEVANCE_FIELDS[:2] == ("E1_evidence_contribution",
                                                 "E2_contributed_terms"), "E fields drifted")


@test
def test_build_worksheet_writes_relevance_first_and_key_apart():
    """25 synthetic parents under the real qids; three with an empty D state."""
    qids = sorted(core.CHILD_COUNTS)
    rows, states, frozen, queries = {}, {}, {}, {}
    for i, qid in enumerate(qids):
        row = synthetic_parent(qid=qid)
        rows[qid] = row
        states[qid] = (core.parse_state('{"facts":[]}', row, "D") if i < 3
                       else synthetic_d_state(row, ["w10 w11 w12"]))
        for arm in dev.ARMS_V2:
            frozen[qid, arm] = "frozen query " + arm
            queries[qid, arm] = {"result": {"query": "revised query " + arm}}
    ctx = {"rows": rows, "states": states, "frozen_queries": frozen}
    base = Path(tempfile.mkdtemp(prefix="bridge_v2_dev_test_"))
    try:
        out = dev.build_worksheet(ctx, base, queries)
        expect(out["relevance_cells"] == 25 + 22 and out["cells"] == 200, "sheet totals")
        shared = sorted(p.name for p in (base / dev.REVIEW_DIR).iterdir())
        expect(shared == ["evidence_relevance_worksheet.jsonl", "rubric_worksheet.jsonl"],
               "shared review directory holds more than the two sheets: %s" % shared)
        expect(sorted(p.name for p in (base / dev.REVIEW_KEY_DIR).iterdir())
               == ["worksheet_key.jsonl"], "key not isolated")
        rel = (base / dev.REVIEW_DIR / "evidence_relevance_worksheet.jsonl")
        sheet = (base / dev.REVIEW_DIR / "rubric_worksheet.jsonl")
        expect(rel.stat().st_mtime_ns <= sheet.stat().st_mtime_ns,
               "relevance sheet not written first")
        rel_ids = {json.loads(l)["relevance_id"] for l in rel.open(encoding="utf-8")}
        linked = {json.loads(l)["evidence_relevance_id"] for l in sheet.open(encoding="utf-8")}
        expect(linked - {None} == rel_ids, "query cells and relevance rows do not align")
        for path in (rel, sheet):
            text = path.read_text(encoding="utf-8")
            for token in ("\"version\"", "\"qid\"", "\"arm\"", "frozen\"", "revised\""):
                expect(token not in text, path.name + " contains " + token)
        expect(out["salt"] == "PER_RUN_RANDOM_NOT_STORED", "salt label")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return "47 relevance rows, 200 cells"


@test
def test_worksheet_shuffle_is_seeded_and_stable():
    import random as _random
    order_a = list(range(20))
    order_b = list(range(20))
    _random.Random(dev.REVIEW_SHUFFLE_SEED).shuffle(order_a)
    _random.Random(dev.REVIEW_SHUFFLE_SEED).shuffle(order_b)
    expect(order_a == order_b, "shuffle is not deterministic under the declared seed")


# ------------------------------------------------------- 5. guards and search

@test
def test_generation_guard_refuses_gold_oracle_and_english():
    guard = dev.PathGuard("generation", dev.forbidden_pairs())
    for label, path in dev.forbidden_pairs():
        expect_raises("registered forbidden input " + label, guard.allow, label, path)
    expect(guard.labels() == [], "guard should have registered nothing")


@test
def test_search_validation_rejects_bad_results():
    ids = list(range(dev.BUDGET))
    scores = [1.0 - i * 0.001 for i in range(dev.BUDGET)]
    core.validate_search_arrays(ids, scores)
    expect_raises("ascending scores accepted",
                  core.validate_search_arrays, ids, list(reversed(scores)))
    expect_raises("duplicate rows accepted",
                  core.validate_search_arrays, [0] * dev.BUDGET, scores)
    expect_raises("short result accepted",
                  core.validate_search_arrays, ids[:-1], scores[:-1])


@test
def test_aggregate_candidates_max_per_title_and_cap():
    candidates = [{"global_row": 5, "score": 0.4, "title": "Alpha"},
                  {"global_row": 2, "score": 0.9, "title": "alpha"},
                  {"global_row": 7, "score": 0.9, "title": "Beta"},
                  {"global_row": 3, "score": 0.9, "title": "beta"}]
    ranked = core.aggregate_candidates(candidates)
    expect([r["normalized_title"] for r in ranked] == ["alpha", "beta"], "titles wrong")
    expect(ranked[0]["score"] == 0.9 and ranked[0]["best_global_row"] == 2,
           "max-per-title or tie-break wrong")
    expect(ranked[1]["best_global_row"] == 3, "lower row must win an exact tie")
    many = [{"global_row": i, "score": 1.0 - i * 0.01, "title": "T%02d" % i}
            for i in range(25)]
    expect(len(core.aggregate_candidates(many)) == 10, "top-10 cap not applied")


# ------------------------------------------- 6. sealed-artifact identity tests

@test
def test_sealed_d_states_reproduce_under_frozen_parser():
    parents = sealed("parents")
    states = sealed("states")
    rows = {json.loads(l)["qid"]: json.loads(l) for l in parents.open(encoding="utf-8")}
    core.validate_parents(list(rows.values()))
    checked = 0
    for line in states.open(encoding="utf-8"):
        row = json.loads(line)
        if row["arm"] != "D":
            continue
        expect(core.parse_state(row["raw_output"], rows[row["qid"]], "D") == row["result"],
               "sealed D state drifted: " + row["qid"])
        checked += 1
    expect(checked == 25, "expected 25 sealed D states, saw %d" % checked)
    return "25 states"


@test
def test_all_100_v2_prompts_build_from_sealed_inputs():
    parents = sealed("parents")
    states = sealed("states")
    rows = {json.loads(l)["qid"]: json.loads(l) for l in parents.open(encoding="utf-8")}
    d_states = {}
    for line in states.open(encoding="utf-8"):
        row = json.loads(line)
        if row["arm"] == "D":
            d_states[row["qid"]] = row["result"]
    built = 0
    for qid in sorted(core.CHILD_COUNTS):
        for arm in dev.ARMS_V2:
            messages = dev.parent_query_messages_v2(rows[qid], arm,
                                                    d_states[qid] if arm == "D" else None)
            expect(messages[0]["content"] == dev.QUERY_SYSTEM_V2, "system drift")
            built += 1
    expect(built == 100, "expected 100 cells")
    return "100 cells"


@test
def test_descriptive_scores_reproduce_frozen_on_frozen_predictions():
    """Feeding the frozen predictions back must give zero delta everywhere."""
    targets_path = sealed("targets")
    scores_path = sealed("frozen_scores")
    preds_path = (dev.ROOT /
                  "outputs/efbpt/bridge_pilot_n25/v1/retrieve_v2/predictions.jsonl")
    if not preds_path.is_file():
        raise SkipTest("frozen predictions absent")
    targets = [json.loads(l) for l in targets_path.open(encoding="utf-8")]
    frozen = json.loads(scores_path.read_text(encoding="utf-8"))
    predictions = {}
    for line in preds_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["arm"] in dev.ARMS_V2:
            predictions[row["qid"], row["arm"]] = row
    expect(len(predictions) == 100, "expected 100 frozen A/P/B/D predictions")
    out = dev.descriptive_scores(predictions, targets, frozen)
    for arm in dev.ARMS_V2:
        for k in dev.CUTOFFS:
            cell = out["arms"][arm][str(k)]
            expect(abs(cell["qid_macro_recall_delta_pp"]) < 1e-9,
                   "non-zero delta for arm %s@%d" % (arm, k))
            expect(cell["qids_improved"] == 0 and cell["qids_worsened"] == 0,
                   "spurious improvement for arm %s@%d" % (arm, k))
    expect(out["gate"] == "NONE" and out["inference"] == "NONE", "gate/inference labels")
    expect(out["statistics_performed"].startswith("NONE"), "statistics label")
    blob = json.dumps(out)
    for banned in ("p_value", "p_two_sided", "significance", "confidence",
                   "_ci\"", "bootstrap", "signflip", "alpha"):
        expect(banned not in blob, "scores contain " + banned)
    return "deltas all zero"


@test
def test_module_declares_no_inferential_statistics():
    """Scan CODE, not prose: docstrings and comments are stripped first."""
    import ast

    tree = ast.parse(Path(dev.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(ast.fix_missing_locations(tree))
    for banned in ("signflip", "p_two_sided", "percentile", "bootstrap",
                   "positive_practical_threshold_met", "primary_gate_passed",
                   "scipy", "summarize("):
        expect(banned not in code, "runner references inferential machinery: " + banned)
    expect(not hasattr(dev, "signflip"), "signflip re-exported")
    expect(not hasattr(dev, "summarize"), "frozen summarize re-exported")


@test
def test_output_paths_are_the_agreed_ones():
    expect(str(dev.OUT) == "/mnt/home/user41/URBench/outputs/efbpt/bridge_v2_dev/queries_dev1",
           "output root drifted: " + str(dev.OUT))
    expect(dev.ARMS_V2 == ("A", "P", "B", "D"), "arm set drifted")
    expect(dev.BUDGET == 100 and dev.CUTOFFS == (1, 5, 10), "budget or cutoffs drifted")


@test
def test_new_dir_refuses_existing_directory():
    base = Path(tempfile.mkdtemp(prefix="bridge_v2_dev_test_"))
    try:
        target = base / "fresh"
        dev.new_dir(target)
        expect(target.is_dir(), "directory not created")
        expect_raises("existing directory accepted", dev.new_dir, target)
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ------------------------------- 7. environment, models, code, Git, failures

def frozen_manifest():
    path = sealed("activation")
    manifest, _ = dev.load_activation(path, dev.PINS["activation"][1])
    return manifest


@test
def test_frozen_helpers_are_imported_not_redefined():
    import bridge_pilot_run as frozen_run
    for name in ("load_activation", "check_environment", "check_model_files", "allow_model",
                 "PathGuard", "read_small", "write_json", "new_dir"):
        expect(getattr(dev, name) is getattr(frozen_run, name), name + " is not the frozen helper")


@test
def test_running_under_frozen_environment():
    manifest = frozen_manifest()
    env = dev.verify_environment(manifest)
    expect(env["executable"] == manifest["environment"]["executable"], "executable")
    expect(env["packages"] == manifest["environment"]["packages"], "packages")
    return "python %s" % manifest["environment"]["python"]


@test
def test_environment_drift_is_refused():
    manifest = frozen_manifest()
    for mutate in (lambda m: m["environment"].update(executable="/usr/bin/python3"),
                   lambda m: m["environment"].update(python="3.13.11"),
                   lambda m: m["environment"]["packages"].update(torch="0.0.0")):
        bad = json.loads(json.dumps(manifest))
        mutate(bad)
        expect_raises("environment drift accepted", dev.verify_environment, bad)


@test
def test_model_roots_match_activation_and_drift_is_refused():
    manifest = frozen_manifest()
    expect(dev.verify_model_roots(manifest) == {"qwen": dev.QWEN_ROOT,
                                                "encoder": dev.ENCODER_ROOT}, "roots")
    expect(set(manifest["models"]) == {"qwen", "encoder"}, "model set")
    for key in ("qwen", "encoder"):
        bad = json.loads(json.dumps(manifest))
        bad["models"][key]["root"] = "/nonexistent"
        expect_raises(key + " root drift accepted", dev.verify_model_roots, bad)
        bad = json.loads(json.dumps(manifest))
        bad["models"][key]["root"] = "/nonexistent"
        expect_raises(key + " root drift accepted by file check", dev.verify_model_files, bad)
    return "hashing not exercised offline"


@test
def test_chat_template_pin_is_the_frozen_value():
    manifest = frozen_manifest()
    pin = manifest["settings"]["tokenizer"]
    expect(pin["chat_template_sha256"]
           == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
           "chat template pin differs from the value recorded in the note")
    expect(pin["tokenizer_class"] == "Qwen2TokenizerFast", "tokenizer class pin")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@test
def test_dev_code_identity_verified_against_wrapper_supplied_hashes():
    runner, test_file = dev.DEV_FILES["runner"], dev.DEV_FILES["test"]
    good = {dev.ENV_RUNNER_SHA256: file_sha256(runner),
            dev.ENV_TEST_SHA256: file_sha256(test_file),
            dev.ENV_WRAPPER_EXECUTED_SHA256: file_sha256(dev.DEV_FILES["wrapper"])}
    out = dev.verify_dev_code(good, required=True)
    expect(out["verification"] == {"runner": "VERIFIED_AGAINST_WRAPPER",
                                   "test": "VERIFIED_AGAINST_WRAPPER"}, "verification")
    expect(out["wrapper_executed_matches_repository"] is True, "wrapper match flag")
    expect(set(out["files"]) == {"runner", "test", "wrapper"}, "all three recorded")
    for key in (dev.ENV_RUNNER_SHA256, dev.ENV_TEST_SHA256):
        bad = dict(good, **{key: "0" * 64})
        expect_raises("hash mismatch accepted: " + key, dev.verify_dev_code, bad, True)
        missing = {k: v for k, v in good.items() if k != key}
        expect_raises("missing hash accepted: " + key, dev.verify_dev_code, missing, True)
    lax = dev.verify_dev_code({}, required=False)
    expect(set(lax["verification"].values()) == {"NOT_SUPPLIED"}, "lax mode label")
    src = runner.read_text(encoding="utf-8")
    expect(file_sha256(runner) not in src, "runner embeds its own hash")


@test
def test_git_provenance_runtime():
    try:
        out = dev.git_provenance({}, required=True)
    except Exception as exc:                                    # noqa: BLE001
        raise SkipTest("git unavailable here: " + type(exc).__name__)
    expect(out["source"] == "RUNTIME_GIT" and out["runtime_verified"] is True, "runtime label")
    expect(len(out["head"]) == 40, "head")
    return out["head"][:12]


@test
def test_git_absent_uses_labelled_submission_snapshot():
    base = Path(tempfile.mkdtemp(prefix="bridge_v2_dev_test_"))
    saved_git, saved_dir = dev._git, dev.SNAPSHOT_DIR

    def no_git(*args):
        raise FileNotFoundError("git")

    try:
        dev._git, dev.SNAPSHOT_DIR = no_git, base
        snap = base / "submit.txt"
        snap.write_bytes(("a" * 40 + "\n?? docs/EFBPT_BRIDGE_V2_DEV_NOTE.md\n").encode("utf-8"))
        env = {dev.ENV_SUBMIT_SNAPSHOT: str(snap),
               dev.ENV_SUBMIT_SNAPSHOT_SHA256: file_sha256(snap)}
        out = dev.git_provenance(env, required=True)
        expect(out["source"] == dev.SNAPSHOT_LABEL, "snapshot not labelled")
        expect(out["runtime_verified"] is False, "snapshot described as runtime-verified")
        expect(out["submission_snapshot"]["head"] == "a" * 40, "snapshot head")
        expect_raises("no git and no snapshot accepted", dev.git_provenance, {}, True)
        expect(dev.git_provenance({}, False)["source"] == "UNAVAILABLE", "lax label")
        bad = dict(env, **{dev.ENV_SUBMIT_SNAPSHOT_SHA256: "0" * 64})
        expect_raises("snapshot hash mismatch accepted", dev.git_provenance, bad, True)
        other = Path(tempfile.mkdtemp(prefix="bridge_v2_dev_test_")) / "s.txt"
        other.write_bytes(snap.read_bytes())
        outside = {dev.ENV_SUBMIT_SNAPSHOT: str(other),
                   dev.ENV_SUBMIT_SNAPSHOT_SHA256: file_sha256(other)}
        expect_raises("snapshot outside logs accepted", dev.git_provenance, outside, True)
        shutil.rmtree(other.parent, ignore_errors=True)
    finally:
        dev._git, dev.SNAPSHOT_DIR = saved_git, saved_dir
        shutil.rmtree(base, ignore_errors=True)


@test
def test_failure_report_names_stage_and_preserves_partials():
    import contextlib
    import io
    base = Path(tempfile.mkdtemp(prefix="bridge_v2_dev_test_"))
    saved = dict(dev.STAGE)
    try:
        root = base / "run"
        dev.new_dir(root)
        (root / "partial.json").write_text("{}", encoding="utf-8")
        dev.STAGE.update(name="S3_retrieve", root=root)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = dev.report_failure(RuntimeError("INDEX_SHAPE"))
        expect(code == 2, "exit code")
        report = json.loads(err.getvalue())
        expect(report["stage"] == "S3_retrieve" and report["status"] == "STOPPED", "stage")
        expect(report["outputs_must_not_be_deleted"] is True, "preservation flag")
        expect((root / "partial.json").is_file(), "partial artifact removed")
        expect(json.loads((root / "DEV_FAILURE.json").read_text())["stage"] == "S3_retrieve",
               "failure record")
        dev.STAGE.update(name="S0_verify_inputs", root=None)
        with contextlib.redirect_stderr(io.StringIO()):
            dev.report_failure(RuntimeError("x"))
        expect(sorted(p.name for p in root.iterdir()) == ["DEV_FAILURE.json", "partial.json"],
               "report without a created root wrote somewhere")
    finally:
        dev.STAGE.clear()
        dev.STAGE.update(saved)
        shutil.rmtree(base, ignore_errors=True)


def main():
    print("-" * 68)
    print("passed %d   skipped %d   failed %d" % (len(PASSED), len(SKIPPED), len(FAILED)))
    for name, why in SKIPPED:
        print("  skipped: %s  (%s)" % (name, why))
    for name, why in FAILED:
        print("  FAILED:  %s  %s" % (name, why))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
