# QEA triage: amendment 4 (isolated title-identification calibration; calibration only)

**Development note, 2026-09-27. Prepared offline; not executed.** Scope: `POST_OUTCOME_AI_ASSISTED_DEVELOPMENT`.

**Unchanged by this amendment:**
- the protocol, amendments 1–3, and every existing prompt, stage, packet and seal;
- the acceptance rule, q3/q4 and their packets.

**Why.** The read-only audit of job 96416 found:
- In 5 of v2's 11 q2 = N answers, the model's own note contradicted the answer.
- Some q1 judgments drew on fields outside the question.

This amendment tests one change that addresses both: an affirmative, isolated identification task.

## 1. Task and prompt
**Prompt:** `IDENT_SYSTEM`, SHA-256 `1cbc1b4600e56c258a80ab4fb650da197f1321df97ec0afd1b7d3155bf3084d7`.
- The same question is asked for every item: **"Is this page directly identifiable from the Urdu question?"**
- The definition block is the same text as v2's. It includes the verbatim Stage-0 §6.1 quotation, with "reasonably"
  inside it, the permitted mappings, the exclusions and the §6.4 boundaries. The only change is "title" in the singular.
- The prompt contains no wording about parent, child, dependency or reviewer roles.

**Input and output:**
- Model input: `{"question_ur": …, "title": …}` only.
- Output: exactly `{"answer": "Y"|"N"|"U", "note": "<one sentence>"}`.
- Parser `parse_identification`: a non-`stop` finish, non-JSON, a wrong key set, a wrong type or a wrong value is
  a failure, never U.

**Backend:** the corrected Gemma backend from amendment 2, greedy decoding, with a cap of 256 new tokens per call.

## 2. Mapping (applied outside the model)
| Identification | Parent q1 | Child q2 |
|---|---|---|
| Y | Y | N |
| N | N | Y |
| U | U | U |

INVALID, LENGTH and MISSING stay failures and are never mapped to a semantic answer. No saved answer is
reinterpreted. New q1/q2 answers are never combined with old q3/q4 answers, so no accepted pair can be produced
this way.

## 3. Derived item packet (`outputs/efbpt/qea_identification_calibration_v1/`, created exclusively)
**Sources:** only the sealed review calibration:
- packet `353611ea…`;
- labels `1412ee53…`;
- run `21ea719f…`, job 96416.

**Identity rule:** an item is an exact (`question_ur`, `title`) pair, with no normalization or aliases.

**Recomputed counts:**
- 30 historical parent items, 41 historical child items and 10 control items: **81 unique items**.
- **92 source occurrences:** 41 parent, 41 child and 10 control.
- Discrepancies: none found. There are no cross-role duplicates, no inconsistent question text within a qid, no
  conflicting reference labels, and no item that spans several questions.

**References:**
- Parent items take the historical q1: Y on all 30.
- Child items take the complement of the historical q2 (U preserved): N on all 41.
- Controls keep their AI-authored expected q1 (4 Y, 4 N, 2 U). The C07/C08 expected-U labels keep their declared
  qualification that N is also defensible.
- Reference totals: 34 Y, 45 N, 2 U.

**Sealed contents:**

| Part | Seal SHA-256 | Contents |
|---|---|---|
| `packet/` | `7ac77e67759547f765933216861cb2c25274de8b6f4e55d7da258c74f129faaf` | Model inputs, call plan and prompt text only; maximum Gemma prompt width 549 tokens |
| `labels/` | `9b0426a7f273101a3117f8d1528c6c021c8ec185ba7c6e54ae40af36102438c9` | Items, source associations and references, each in a separate file |
| `baseline/` | `586b21767845616f962f8673630c2737d899b0aab0e89e0e268a141b0808cc52` | Job-96416 judgments restated as identification: q1 for parents and controls, inverted q2 for children |

The baseline covers 92 occurrences for each of v1 and v2. It marks an item MIXED where its occurrences disagree:
v2 has one such item, I59 Africanized bee (H34 Y, H35 N); v1 has none. All underlying answers are kept.

## 4. Pre-declared report (`ident-report`)
The report gives:
- unique-item results;
- an occurrence-level comparison with job 96416, for each version separately;
- historical parent positive agreement (denominator 30 items);
- historical child negative agreement, plus a list of false explicit identifications (denominator 41 items);
- controls by expected class, with C07/C08 shown separately (strict U agreement and their Y/N/U outcomes);
- uncertainty, invalid, length-stopped and missing outputs;
- results grouped by original question.

Scoring rules: failures stay in every applicable denominator, and U and failures are never counted as correct
negatives. There is no pooled headline accuracy and no significance test, and reused occurrences are not treated as
independent.

## 5. Proposed later execution (not submitted)
81 calls on one model load, at most 81 × 256 = **20,736** new tokens:

```
sbatch --time=01:30:00 --export=ALL,QEA_EXPECT_RUNNER_SHA256=<runner>,QEA_EXPECT_TEST_SHA256=<tests>,QEA_EXPECT_WRAPPER_SHA256=<gpu wrapper> eval/error_analysis_tests/efbpt/efbpt_qea_triage_gpu_v1.sbatch ident-gemma
```

The job writes `…/gemma_run/`. The offline `ident-report` then writes `…/report/`.

## 6. Limitations
- An answer–note contradiction does not show which of the two was semantically correct.
- Isolation and positive wording are tested together, so their individual effects will not be identified.
- Judging each exact item once guarantees consistency, not correctness.
- Production Pass-3 intermediate text is an unverified claim until it is assessed.
- Unresolved step references (`#n`, with 0-based versus 1-based numbering) and the adequacy of the q3/q4 packets
  remain blockers before full triage. This amendment does not change them.

## 7. Record: identification run, job 96485 (2026-09-27; the single approved run)

**Preflight:** all checks passed.
- Seals and the approved code hashes (runner `267cc070…`, tests `6cd26137…`, wrapper `5a9de75c…`) verified.
- 81 unique fresh single-turn calls, with inputs of only `question_ur` and `title`, and inference reading only
  `packet/`.
- The guard passed both locally and on the node, and all 96 offline tests passed on the node.

**Execution:**
- COMPLETED 0:0 on L20001 in 15 min 22 s.
- Load took 600 s. `peak_cuda_bytes_allocated` was 23,109,114,368, and the maximum sampled GPU use was 22,446 of
  46,068 MiB.
- 3,020 of at most 20,736 new tokens were generated. No warnings.

**Seals:** run `62dd0f96300c177c0846e75b21b5e49ec5ccbe2ac742f3fefdf0e2875c2135d4` (labels never read); report
`008ae9207c701221662c21d0304ba964e93468e99f555b458ddc6ac7521cf084`.

**Validity:** all 81 outputs valid. Every call stopped on `<end_of_turn>`, with no length stops and no non-finite
values. The answers were 34 Y, 47 N and 0 U.

| Items | Reference | New | Old v2 (job 96416) | Old v1 |
|---|---|---:|---:|---:|
| Historical parents (30) | Y | 17 Y, 13 N | 24 Y, 5 N, 1 MIXED | 2 Y |
| Historical children (41) | N | 29 N, 12 Y | 31 N | 41 N |
| Controls expected Y (4) | Y | 3 (C09 NASA N) | 4 | 1 |
| Controls expected N (4) | N | 4 | 3 | 4 |
| Controls expected U (2; N also defensible) | U | 0 (both **Y**) | 0 (both N) | 0 (both N) |

**Against v2, correctness changes by item:**
- Parents: 4 gains, including I59 (MIXED → Y), and 11 losses.
- Children: 3 gains and 5 losses.
- Controls: 1 gain (C04) and 1 loss (C09).

**Against v1:** parents gain 15 and lose 0; children gain 0 and lose 12.

**Occurrence level (92 per version):**
- Parents answered Y on 24 of 41 occurrences, against 35 (v2) and 3 (v1).
- Children answered correctly (q2 = Y) on 29 of 41, against 31 (v2) and 41 (v1).

**Pattern:**
- In 5 of 30 questions, the parent was judged N and the child Y: Harry Potter, the Speaker (Nancy Pelosi),
  Breakfast, Shirley Bassey and Asparagus. This never happened under v1 or v2.
- Pattern-checked notes show no contradiction between answer and note.
- The stated reasons include:
  - treating identification as "is the question about this page";
  - Urdu misreadings (for example Sable read as "Sabretooth");
  - claims that names such as Nancy Pelosi or Padmé Amidala appear in the question, when they do not.

**Conclusion:** isolation with positive wording removed the answer–note contradictions and the scope leakage by
construction. It did **not** improve parent recognition over v2, and it lost child negatives and the handling of
ambiguous titles. It is not adopted.

## 8. Correction to §7 (2026-09-27; §7 is left as written)
- **Parents versus v2:** 3 clear gains, 11 losses, and 1 MIXED → Y resolution (I59 Africanized bee). §7 counted
  4 gains because it included the MIXED case.
- **Controls versus v2:**
  - 1 clear gain (C04) and 1 clear loss (C09).
  - 2 ambiguity regressions: C07 and C08 moved from N, which was declared defensible, to Y, which is neither
    expected nor declared defensible.
- **Scope of the conclusion:** the evidence establishes poor performance of the tested Gemma configuration on these
  items. It does not isolate model capability as the sole cause.

## 9. Claude identification calibration: STOPPED_BEFORE_INFERENCE; status PAUSED_BY_USER (2026-09-27)

**What was approved:** one independent Claude run on the sealed 81-item packet, in which each call would receive
only `IDENT_SYSTEM` and one payload, in a fresh context with no tools, memory or other items.

**Outcome:** stopped before inference, because no available route could verify that isolation.
- **Claude calls:** zero.
- **Collection code:** none written.
- **Outputs:** none.

**Routes inspected (read-only):**

| Route | Limitation |
|---|---|
| Existing R2 (`render-r2` / `ingest-r2`) | A manual batch: all items go into one file for one manually operated session. Model, settings, memory and stop reasons are unverifiable and unrecorded. |
| Anthropic SDK | Installed in `urbench_eval`, but no API key is present, and adding credentials was excluded. |
| Claude Code CLI 2.1.283, `--bare` | Skips CLAUDE.md, auto-memory and hooks, but authenticates only with an API key or API-key helper. |
| Same CLI, normal mode (existing login) | Allows `--system-prompt`, `--tools ""` and `--safe-mode`, but auto-memory exclusion is not documented, there is no temperature or output-cap control, and Claude Code builds the request itself, so the exact contents can't be verified before running. |
| Subagents from the orchestrating session | Fresh context, but each carries the Claude Code system prompt and tools. |

**Unchanged:** the isolation requirements. The sealed packet (`7ac77e67…`), labels (`9b0426a7…`), baseline
(`586b2176…`), Gemma run (`62dd0f96…`) and report (`008ae920…`) all still verify.

**Status:** `PAUSED_BY_USER`. Any further step needs explicit authorization.
