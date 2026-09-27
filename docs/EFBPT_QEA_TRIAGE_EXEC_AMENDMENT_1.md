# QEA triage: operational amendment 1 (T1/T2 execution, provenance, accounting)

**Development note, 2026-09-27. Operational only.** It does not edit the QEA protocol revision 0.2
(`docs/EFBPT_QEA_DEV_PROTOCOL.md`, SHA-256 `776b26f0c35e29f0cefdb11e4c0888a2a8be1a1b51509d5ecec07b566402da21`),
any seal, preparation manifest or prior output. It changes no prompt, candidate rule, acceptance rule,
selection rule, budget or cohort. Scope: `POST_OUTCOME_AI_ASSISTED_DEVELOPMENT`.

## 1. Evidence (re-verified 2026-09-27, all unchanged)

| Job | Stage | Result | Seal SHA-256 |
|---|---|---|---|
| 96120 | smoke-llama | Engine initialization failed at `gpu_memory_utilization` 0.90 (0.83 GiB KV < 1.25 GiB at 4096) | none |
| 96136 | smoke-llama | 6 calls, 4 parser-valid | `f52d236b…4765` |
| 96151 | smoke-llama-so | 6 calls, 6 parser-valid, 0 length stops | `d168bf668d01251fdfd1d55019409966fd5d503c004112b84ef791fb0237037e` |

Job 96151 records `f3b0b4e2…3374` match its seal. Its guard recorded runner `0d5570d2…cf0a6`, tests
`e3756107…4ce4` and executed wrapper `ccd7d307…507f`, which matched those files before this amendment.

## 2. Corrected wording for the job-96151 report

- Changes under constrained decoding were observed on these six requests: pass2[2] and pass3[0] changed
  (the two invalid in 96136), and the other four are byte-identical to 96136.
- Generalization and annotation accuracy remain unestablished.
- The smoke tested components separately. Pass 2 and Pass 3 used fixed historical inputs, no Pass-3 request
  depended on this smoke's Pass-2 output, and no join, review or acceptance ran.
- Sound barrier still disagrees with the explicit-parent rule. pass2[0] judges it NOT_YET_EXPLICIT (historical
  label EXPLICIT) while pass3[0] names it as the CLEAR_DEPENDENCY parent. Under the unchanged T3 rule the pair
  is excluded, with reason `PARENT_T1_NOT_YET_EXPLICIT`. No prompt was tuned to recover it.

## 3. T1/T2 execution configuration `qea_triage_llama_t1t2_exec_v2`

Canonical SHA-256 `94527785dbac85492251a2819de9a04c35acc87c8dd20904d10b3a5c576159db` (`LLAMA_EXECUTION` in the
runner). T1 and T2 build their backend only from it; the CLI has no batch or memory option.

| Setting | Protocol rev 0.2 | Amendment 1 (as in job 96151) |
|---|---|---|
| `gpu_memory_utilization` | 0.90 | **0.95** |
| `max_num_seqs` / request batch | 8 / 8 | **4 / 4** |
| Structured outputs | none | **per-request JSON schema; xgrammar, `disable_fallback`, `disable_any_whitespace`** |
| Model, quantization, dtype | Llama-3.1-70B-Instruct AWQ-INT4, awq, float16, `enforce_eager` | unchanged |
| Context, decoding | 4096; greedy, temperature 0, seed 0 | unchanged |
| Prompts | Pass 2 `4a96e818…`, Pass 3 `0171cc7c…` | unchanged |
| Output caps | 256 (Pass 2), 512 (Pass 3) | unchanged |

- **Schemas.**
  - T1 sends the same `pass2_schema()` with every request.
  - T2 builds each request's schema from the prompt's own `candidate_parent_titles`. Exact equality is enforced.
  - T2 fails closed on zero steps, duplicate candidates, a normalized self candidate, or a target without
    paragraph evidence.
  - Valid NOT_YET_EXPLICIT, AMBIGUOUS, UNRESOLVED and PARALLEL_OR_UNORDERED outputs remain representable.
  - Every schema is validated and compiled before the model loads.
  - Each record carries its own schema hash.
- **Parser and joins remain authoritative.**
  - T3 re-parses every sealed raw output, and re-derives each request and schema identity, before reading
    corpus status.
  - `build_candidates` is unchanged.
  - T2 and T3 refuse predictions sealed under any other execution configuration.
- **Gemma (R1 and smoke-gemma)** keeps its own backend: transformers 4-bit NF4, sequential calls, no schemas and
  no memory fraction.

## 4. Provenance

- **CPU wrapper:** it now has the GPU wrapper's fail-closed guard. It exits 3 before tests or any stage on a
  missing, malformed or mismatched runner, test or executed-wrapper hash.
- **Stage seals:** every stage (T0–T5, R1, all smokes) re-checks the three hashes before reading an input or
  loading a model, and records the result in its seal. The interactive R2 commands require the runner and
  test hashes, and record the wrapper as not applicable.
- **Smoke artifacts:** they are refused as annotations by path and by seal content.

## 5. Per-question accounting (`question_accounting.jsonl` in T1, T2, T3 and T5)

This is reporting only. It has one row per development qid, in development order.

**Per call** (disjoint):
- `valid`
- `malformed` (parser rejection)
- `length_rejected` (stopped by the output cap)
- `prompt_too_long` (input rejected before generation)
- `missing`

Annotation failures are never counted as valid negative decisions.

**Per question:**
- `t1`: calls expected and recorded, outcomes, valid EXPLICIT and valid NOT_YET_EXPLICIT counts.
- `pass3_eligibility`: whether the question is eligible, and how many titles are.
- `t2`: calls and outcomes, plus valid decisions by kind.
- `corpus`: status counts.
- `titles`: each title's T1, T2 and corpus label.
- `proposals`: every parent named by a valid CLEAR_DEPENDENCY child, with every unmet candidate condition,
  for example `PARENT_T1_NOT_YET_EXPLICIT`, `PARENT_T1_ANNOTATION_MALFORMED` or `CHILD_EXACT_ABSENT`.
- `t3`: candidate pairs.
- `review`: packets, and R1 and R2 valid, malformed and missing counts.
- `t5`: pair statuses, accepted pairs, question status and selection.

A stage that has not run is `{"stage": "NOT_RUN"}` (or `"NOT_RUN"`), never zero. Corpus status is `NOT_JOINED`
before T3, by design.

**`flags`** are overlapping reason codes and are never summed as disjoint. `NO_CANDIDATE_PAIR` marks every
question without a pair once T3 has run, and `NO_ACCEPTED_PAIR` marks questions with pairs but none accepted.

**`outcome`** is the single disjoint funnel position: the first stop under the existing rules, or the next stage
not yet run. The seal summaries give disjoint outcome counts, and overlapping flag counts labelled as not
additive.

**What accounting does not do:** a partly failed question keeps its valid pairs, and nothing here rejects a
question. The accounting must agree exactly with `build_candidates`, or T3 fails.

## 6. Development input lengths (Llama tokenizer only; development preparation only)

These are complete chat-formatted token counts from the same builders T1 and T2 use (`input-lengths` command).
All 544 source instances are counted as possible Pass-3 inputs, because entry depends on T1.

| Requests | Count | Min | Median | Max | Prompt limit (4096 − cap) | Over limit |
|---|---:|---:|---:|---:|---:|---:|
| Pass 2 | 544 | 425 | 448 | 492 | 3840 | 0 |
| Pass 3 (possible) | 544 | 627 | 770 | 1248 | 3584 | 0 |

- **Contract conflicts:** 0.
- **Distinct Pass-3 schemas:** 543, all compiled.
- **Candidate parents:** at most 14, and no request has zero.
- **Reads:** no reserve content, metadata, index or weights was read.

## 7. Record: Gemma smoke, job 96317 (2026-09-27; one submission, no retry)

**Execution:**
- COMPLETED 0:0 on L20002 in 9 min 02 s.
- The guard passed on the node (runner `aba9b30c…`, tests `490793e6…`, executed wrapper `7f61db36…`), and all
  73 tests passed there.
- Model load took 456 s.
- `peak_cuda_bytes_allocated` 23,109,114,368 (in-process). The maximum sampled GPU use was 22,450 of 46,068 MiB.

**Seal:** `af9dafd399e8d879450caa2b4e05ba601e99962c989dd60eda8a2610babbf52e`, records `49163628…`. Scope is
`SMOKE_NOT_ANNOTATION`. There were 2 calls, and their inputs match the job-96151 code's review payloads.

**Result: both calls invalid.**
- Each generated 256 of 256 new tokens, finished by `length` and decoded (special tokens skipped) to an empty
  string. The parser gives `FINISH_LENGTH`, and the smoke records `format_ok` 0/2.
- Token IDs are not recorded, so the generated tokens cannot be identified.

**Unverified candidate causes (read from code and config; nothing was changed):**
- (a) `HfBnbBackend` passes no dtype. transformers 4.57.6 then forces float16 for the non-quantized layers of a
  bfloat16 Gemma-2 checkpoint.
- (b) The local `generation_config.json` stops only on `<eos>` (1), not on `<end_of_turn>` (107).

**Consequence:** R1 on this backend is not ready. Any change to it needs separate approval and a new smoke
output directory.
