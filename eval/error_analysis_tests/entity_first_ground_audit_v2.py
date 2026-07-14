#!/usr/bin/env python3
"""Efficient, ground-and-audit-only entity-first probe for Urdu StrategyQA.

This program deliberately does not generate candidates.  It treats the twelve
validated v1 generation caches as immutable question-only inputs, performs one
budgeted globally-deduplicated FAISS search, and then runs the offline oracle
audit.  It never processes eval458 and never performs answer generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


class ProbeError(RuntimeError):
    pass


class EntityFirstGroundAudit:
    BASE = Path("/mnt/home/user41/URBench")
    OFFICIAL = BASE / "data" / "strategyqa_official"
    DEV50 = OFFICIAL / "dev50_seed42.jsonl"
    DEV50_QIDS = OFFICIAL / "dev50_seed42_qids.txt"
    EVAL458 = BASE / "data" / "sdfr_splits" / "strategyqa_eval.jsonl"
    PARAGRAPHS = OFFICIAL / "strategyqa_train_paragraphs.json"

    SOURCE_GENERATION = OFFICIAL / "phase_r_entity_first_probe12_v1" / "cache" / "generation"
    OUT = OFFICIAL / "phase_r_entity_first_probe12_v2"
    GROUND_CACHE = OUT / "cache" / "grounding"

    SOURCE_SCHEMA = "entity-first-probe12-generation-v1.0"
    GROUND_SCHEMA = "entity-first-probe12-grounding-v2.0"
    SAMPLE_SIZE = 12
    SEED = 42
    QUERY_BUDGET_PER_VIEW = 4
    TOP_K = 10
    TITLE_POOL_CAP = 30
    MARKERS = {"operation", "no_evidence"}

    def __init__(self, embedder_device: str = "cpu") -> None:
        self.embedder_device = embedder_device
        self.qids: list[str] = []
        self.dev: dict[str, dict[str, Any]] = {}
        self.generations: dict[str, dict[str, Any]] = {}

    @staticmethod
    def norm(value: Any) -> str:
        text = unicodedata.normalize("NFKC", str(value or ""))
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def key(cls, value: Any) -> str:
        return cls.norm(value).replace("_", " ").casefold()

    @staticmethod
    def sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def digest(value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProbeError(f"Expected JSON object at {path}:{number}")
                rows.append(value)
        return rows

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ProbeError(f"Expected JSON object in {path}")
        return value

    @staticmethod
    def atomic_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @classmethod
    def atomic_json(cls, path: Path, value: Any) -> None:
        cls.atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def atomic_jsonl(cls, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        cls.atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def load_and_validate(self) -> None:
        dev50_qids = [
            self.norm(line)
            for line in self.DEV50_QIDS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(dev50_qids) != 50 or len(set(dev50_qids)) != 50:
            raise ProbeError("DEV50 qid file is not exactly 50 unique qids")
        self.qids = random.Random(self.SEED).sample(sorted(dev50_qids), self.SAMPLE_SIZE)

        eval_qids = {self.norm(row.get("qid")) for row in self.read_jsonl(self.EVAL458)}
        if len(eval_qids) != 458 or set(self.qids) & eval_qids:
            raise ProbeError("Probe12/eval458 provenance guard failed")

        all_dev = {self.norm(row.get("urbench_qid")): row for row in self.read_jsonl(self.DEV50)}
        self.dev = {qid: all_dev[qid] for qid in self.qids if qid in all_dev}
        if set(self.dev) != set(self.qids):
            raise ProbeError("Could not resolve all fixed probe12 rows")

        for qid in self.qids:
            row = self.dev[qid]
            if row.get("is_eval") is not False:
                raise ProbeError(f"Probe row is not marked non-eval: {qid}")
            question = self.norm(row.get("question_ur"))
            cache = self.read_json(self.SOURCE_GENERATION / f"{qid}.json")
            if (
                cache.get("schema_version") != self.SOURCE_SCHEMA
                or cache.get("urbench_qid") != qid
                or cache.get("input_question_sha256") != self.sha256(question)
                or cache.get("parse_ok") is not True
                or not cache.get("direct_items")
                or not cache.get("translated_question_en")
                or not cache.get("translated_items")
            ):
                raise ProbeError(f"Generation cache failed validation: {qid}")
            self.generations[qid] = cache

        print("[guard] fixed seed-42 probe12; DEV50 only; zero eval458 overlap")
        print("[guard] 12 validated generation caches loaded read-only")

    def select_candidates(self, items: Sequence[dict[str, Any]]) -> list[str]:
        lists: list[list[str]] = []
        for item in items:
            if item.get("kind") == "ENTITY":
                values = [self.norm(value) for value in item.get("candidates_en", [])]
                values = [value for value in values if value]
                if values:
                    lists.append(values)

        selected: list[str] = []
        seen: set[str] = set()
        for rank in range(max((len(values) for values in lists), default=0)):
            for values in lists:
                if rank >= len(values):
                    continue
                value = values[rank]
                canonical = self.key(value)
                if canonical and canonical not in seen:
                    selected.append(value)
                    seen.add(canonical)
                if len(selected) == self.QUERY_BUDGET_PER_VIEW:
                    return selected
        return selected

    def plan(self) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        specs: dict[str, dict[str, Any]] = {}
        global_queries: dict[str, str] = {}
        for index, qid in enumerate(self.qids, 1):
            generation = self.generations[qid]
            direct = self.select_candidates(generation["direct_items"])
            translated = self.select_candidates(generation["translated_items"])
            union_keys: list[str] = []
            for candidate in direct + translated:
                canonical = self.key(candidate)
                if canonical and canonical not in union_keys:
                    union_keys.append(canonical)
                    global_queries.setdefault(canonical, candidate)
            specs[qid] = {
                "generation_digest": self.digest(generation),
                "direct": direct,
                "translated": translated,
                "direct_keys": [self.key(value) for value in direct],
                "translated_keys": [self.key(value) for value in translated],
                "union_keys": union_keys,
            }
            print(
                f"[plan] {index}/12 qid={qid} direct={len(direct)} "
                f"translated={len(translated)} union={len(union_keys)}"
            )

        maximum = self.SAMPLE_SIZE * self.QUERY_BUDGET_PER_VIEW * 2
        if len(global_queries) > maximum:
            raise ProbeError(f"Physical query guard failed: {len(global_queries)} > {maximum}")
        print(f"[plan] globally unique physical queries={len(global_queries)}; hard maximum={maximum}")
        return specs, global_queries

    def title_rows(self, candidate: str, hits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        candidate_key = self.key(candidate)
        best: dict[str, dict[str, Any]] = {}
        for hit in hits:
            title = self.norm(hit.get("title"))
            title_key = self.key(title)
            if not title_key:
                continue
            candidate_tokens, title_tokens = set(candidate_key.split()), set(title_key.split())
            if candidate_key == title_key:
                lexical = 1.0
            elif candidate_tokens and candidate_tokens.issubset(title_tokens):
                lexical = 0.95
            elif title_tokens and title_tokens.issubset(candidate_tokens):
                lexical = 0.90
            else:
                overlap = len(candidate_tokens & title_tokens)
                lexical = 0.0 if not overlap else 2 * overlap / (len(candidate_tokens) + len(title_tokens))
            retrieval = float(hit.get("score", 0.0))
            combined = 0.65 * lexical + 0.35 * max(0.0, min(1.0, retrieval))
            row = {
                "candidate": candidate,
                "title": title,
                "retrieval_score": retrieval,
                "lexical_score": lexical,
                "combined_score": combined,
            }
            if title_key not in best or combined > best[title_key]["combined_score"]:
                best[title_key] = row
        return sorted(best.values(), key=lambda row: (-row["combined_score"], -row["retrieval_score"]))

    def pool(self, keys: Sequence[str], rows_by_key: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for key in keys:
            for row in rows_by_key.get(key, []):
                title_key = self.key(row["title"])
                if title_key not in best or row["combined_score"] > best[title_key]["combined_score"]:
                    best[title_key] = row
        return sorted(best.values(), key=lambda row: (-row["combined_score"], -row["retrieval_score"]))[: self.TITLE_POOL_CAP]

    def ground(self, specs: dict[str, dict[str, Any]], queries: dict[str, str]) -> list[dict[str, Any]]:
        sys.path.insert(0, str(self.BASE / "rag"))
        from retrieve import Retriever

        print(f"[ground] loading CPU FAISS index once; queries={len(queries)}")
        retriever = Retriever(device=self.embedder_device)
        keys, texts = list(queries), list(queries.values())
        started = time.monotonic()
        hit_batches = retriever.retrieve(texts, top_k=self.TOP_K)
        elapsed = time.monotonic() - started
        if len(hit_batches) != len(keys):
            raise ProbeError("Retriever output count does not match query count")
        hits = dict(zip(keys, hit_batches))
        rows_by_key = {key: self.title_rows(queries[key], hits[key]) for key in keys}
        print(f"[ground] one global search completed in {elapsed:.1f}s")

        self.GROUND_CACHE.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for index, qid in enumerate(self.qids, 1):
            spec = specs[qid]
            record = {
                "schema_version": self.GROUND_SCHEMA,
                "urbench_qid": qid,
                "generation_digest": spec["generation_digest"],
                "query_budget_per_view": self.QUERY_BUDGET_PER_VIEW,
                "direct_entity_candidates": spec["direct"],
                "translated_entity_candidates": spec["translated"],
                "union_entity_candidates": [queries[key] for key in spec["union_keys"]],
                "cross_view_exact_candidate_overlap": sorted(
                    set(spec["direct_keys"]) & set(spec["translated_keys"])
                ),
                "direct_title_pool": self.pool(spec["direct_keys"], rows_by_key),
                "translated_title_pool": self.pool(spec["translated_keys"], rows_by_key),
                "union_title_pool": self.pool(spec["union_keys"], rows_by_key),
                "logical_queries": len(spec["union_keys"]),
            }
            self.atomic_json(self.GROUND_CACHE / f"{qid}.json", record)
            records.append(record)
            print(f"[write] {index}/12 qid={qid}")

        summary = {
            "schema_version": self.GROUND_SCHEMA,
            "probe_items": len(records),
            "query_budget_per_view_per_item": self.QUERY_BUDGET_PER_VIEW,
            "logical_queries_across_items": sum(row["logical_queries"] for row in records),
            "physical_queries_after_global_dedupe": len(queries),
            "global_search_seconds": elapsed,
            "hits_requested_per_query": self.TOP_K,
        }
        self.atomic_json(self.OUT / "grounding_summary.json", summary)
        self.atomic_jsonl(self.OUT / "grounding_records.jsonl", records)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return records

    def evidence_alternatives(self, evidence: Any, paragraph_ids: set[str]) -> list[set[str]]:
        def strings(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [text for item in value for text in strings(item)]
            if isinstance(value, dict):
                return [text for item in value.values() for text in strings(item)]
            return []

        annotations = evidence if isinstance(evidence, list) else [evidence]
        alternatives: list[set[str]] = []
        for annotation in annotations:
            ids = {
                value for value in strings(annotation)
                if value.casefold() not in self.MARKERS and value in paragraph_ids
            }
            titles = {self.key(re.sub(r"-\d+$", "", value)) for value in ids}
            if titles:
                alternatives.append(titles)
        return alternatives

    def audit(self, records: Sequence[dict[str, Any]]) -> None:
        by_qid = {row["urbench_qid"]: row for row in records}
        with self.PARAGRAPHS.open(encoding="utf-8") as handle:
            paragraphs = json.load(handle)
        if not isinstance(paragraphs, dict) or len(paragraphs) != 9251:
            raise ProbeError("Verified 9251-entry paragraph dictionary not found")
        paragraph_ids = set(paragraphs)

        hits: dict[str, list[int]] = defaultdict(list)
        recalls: dict[str, list[float]] = defaultdict(list)
        packet: list[dict[str, Any]] = []
        report: list[str] = []
        views = (("direct", "direct_title_pool"), ("translated", "translated_title_pool"), ("union", "union_title_pool"))

        for qid in self.qids:
            generation, grounding, row = self.generations[qid], by_qid[qid], self.dev[qid]
            alternatives = self.evidence_alternatives(row.get("official_evidence"), paragraph_ids)
            for name, field in views:
                titles = {self.key(item["title"]) for item in grounding[field]}
                recalls[name].append(max((len(titles & gold) / len(gold) for gold in alternatives), default=0.0))
                hits[name].append(int(any(titles & gold for gold in alternatives)))

            packet.append({
                "urbench_qid": qid,
                "question_ur": generation["question_ur"],
                "official_question_en_oracle_only": row["question_en"],
                "official_decomposition_oracle_only": row["official_decomposition"],
                "direct_items": generation["direct_items"],
                "translated_question_en": generation["translated_question_en"],
                "translated_items": generation["translated_items"],
                "direct_title_pool": grounding["direct_title_pool"],
                "translated_title_pool": grounding["translated_title_pool"],
                "union_title_pool": grounding["union_title_pool"],
                "manual_correct_entity_titles": [],
                "manual_notes": "",
            })
            report.extend(["=" * 88, f"QID: {qid}", f"UR: {generation['question_ur']}", f"ORACLE EN: {row['question_en']}"])
            for label, field in views:
                report.append(f"{label.upper()} TITLES:")
                report.extend(f"  {item['combined_score']:.3f} [{item['title']}] via {item['candidate']!r}" for item in grounding[field][:12])

        metrics = {
            name: {
                "items": len(hits[name]),
                "any_official_evidence_page_in_title_pool": sum(hits[name]) / len(hits[name]),
                "macro_best_annotator_evidence_page_recall": sum(recalls[name]) / len(recalls[name]),
            }
            for name, _ in views
        }
        summary = {
            "probe_items": self.SAMPLE_SIZE,
            "automatic_proxy_only": "evidence pages are not always canonical question entities",
            "evidence_page_title_proxy": metrics,
            "manual_entity_gate": "pending: union coverage >=80% and union beats both single views",
            "manual_relation_gate": "pending: relation/gloss identity correct on >=10/12",
            "ready_for_decomposition": False,
            "ready_for_method_claim": False,
        }
        self.atomic_json(self.OUT / "audit_summary.json", summary)
        self.atomic_jsonl(self.OUT / "manual_audit_packet.jsonl", packet)
        self.atomic_text(self.OUT / "manual_audit_report.txt", "\n".join(report) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    def run(self) -> None:
        self.load_and_validate()
        specs, queries = self.plan()
        records = self.ground(specs, queries)
        self.audit(records)
        print(f"SUCCESS: efficient probe12 outputs written to {self.OUT}")

    @classmethod
    def self_test(cls) -> None:
        probe = cls()
        selected = probe.select_candidates([
            {"kind": "ENTITY", "candidates_en": ["A1", "A2", "A3"]},
            {"kind": "ENTITY", "candidates_en": ["B1", "B2", "B3"]},
        ])
        assert selected == ["A1", "B1", "A2", "B2"]
        aliases: dict[str, str] = {}
        for value in ("American Presidents", "American presidents"):
            aliases.setdefault(probe.key(value), value)
        assert len(aliases) == 1 and aliases[probe.key("American presidents")] == "American Presidents"
        print("SELF-TEST PASS: v2 OOP, fixed budget, round-robin selection, canonical aliases")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("self-test", "run"), required=True)
    parser.add_argument("--device-embed", default="cpu")
    args = parser.parse_args()
    if args.stage == "self-test":
        EntityFirstGroundAudit.self_test()
    else:
        EntityFirstGroundAudit(args.device_embed).run()


if __name__ == "__main__":
    try:
        main()
    except (ProbeError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)