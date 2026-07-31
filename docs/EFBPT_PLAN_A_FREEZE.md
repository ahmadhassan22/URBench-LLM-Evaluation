# EFBPT Plan A′ — FROZEN CONTRACT

**Status:** FROZEN on 2026-07-24
**Rule:** Nothing in this file may be changed after the first draft row is generated.
If something here turns out to be wrong, do NOT edit it silently. Add a dated
"AMENDMENT" section at the bottom and say why.

---

## 0. What this document is

This is the contract for building EFBPT training data and running the C0–C3
experiments. It exists so that decisions are made **before** seeing results,
not after. Every number here was chosen with no results in hand.

---

## 1. Schema C — the training target

Each training row is a question plus a plan. The plan has a list of entities
and a list of steps.

### 1.1 Row-level fields

| Field | Type | Meaning |
|---|---|---|
| `qid` | string | Question id from the official StrategyQA mapping |
| `question_ur` | string | The Urdu question, verbatim |
| `entities` | list | List of entity objects (see 1.2) |
| `steps` | list | List of step objects (see 1.3) |
| `answer` | `"yes"` / `"no"` | Final answer |

### 1.2 Entity object

| Field | Type | Meaning |
|---|---|---|
| `canonical_title` | string | The real Wikipedia page name |
| `urdu_span` | string | The exact Urdu text from the question that refers to it |

### 1.3 Step object

| Field | Type | Meaning |
|---|---|---|
| `step_id` | int | 1, 2, 3 … in order |
| `text` | string | The step written in natural language |
| `type` | enum | Step type — see 1.5 |
| `entity_ref` | string or null | `canonical_title` of the entity this step is about |
| `atype` | enum | What kind of answer this step returns — see 1.6 |

**Rule linking `type` and `entity_ref`** (enforced by
efbpt_stage3_core_verify.py, lines 182-185):

- If `type` is `retrieve` → `entity_ref` MUST be filled in, and must
  match a `canonical_title` in `entities`.
- If `type` is `reason` → `entity_ref` MUST be empty/null.

### 1.4 Fields that are NOT in Schema C

Dropped on purpose (2026-07-21 decision, logged in `experiments.md`):

- `evidence_ref`
- `gold_intermediate_answer` (GIA)

They caused 77 of 93 corrections in the Stage 2 pilot and are not part of the
EFBPT signal. 

### 1.5 Valid step types

Valid values (from efbpt_stage3_core_verify.py, line 181):

- `retrieve`
- `reason`

### 1.6 Valid answer types (`atype`)

Valid values (from efbpt_stage3_core_verify.py, line 30):

- `BOOLEAN`
- `ENTITY`
- `LOCATION`
- `DATE`
- `NUMBER`
- `SET`
- `SHORT_TEXT`

---

## 2. Entity policy

An entity goes in the `entities` list only if **all three** are true:

1. **The question names it.** Exact word, a different word-form, an Urdu word,
   or a descriptive phrase that points at one specific thing.
2. **A verbatim Urdu span exists** in `question_ur` for it. Copy-paste, not
   reworded.
3. **The Wikipedia page is real and correct.** Verified, not assumed.

Extra rules, already learned the hard way:

- `canonical_title` is the **real Wikipedia page name**, even if the Urdu span
  is a different word-form. Example: span = "bodybuilder" (جسم ساز), title stays
  `Bodybuilding`. Do not rename the title to match the span.
- Plural spans map to the singular page. Example: سوروں → `Pig`.
- A pure description with no matching span is **not** an entity.
  Example: "red fruit" alone is not `Apple`.
- Names in Urdu that look like a famous thing but mean something else are
  **not** that thing. Example: نیو یارکر = New Yorker the person, not
  *The New Yorker* magazine.

---

## 3. Protected data — never used for training


These five sets are off-limits for training, for all of C1/C2/C3, forever.

| Set | Count | File | Use |
|---|---|---|---|
| `AUDIT30` | 30 | `data/strategyqa_official/efbpt/audit30_qids.txt` | spent — development only |
| `BLIND30` | 30 | `data/strategyqa_official/efbpt/blind30_qids.txt` | spent — development only |
| `DEV50` | 50 | `data/strategyqa_official/dev50_seed42_qids.txt` | c-probe mechanism scoring |
| `DEV200` | 200 | `data/strategyqa_official/dev200_seed4242_qids.txt` | accuracy evaluation during development |
| `eval458` | 458 | `data/sdfr_splits/strategyqa_eval.jsonl` | final evaluation — opened ONCE |

Pool structure (verified 2026-07-24):

- 2,290 mapped rows total = 458 eval458 + 1,832 non-eval
- DEV50 was drawn from the 1,832 non-eval rows with seed 42, before
  Stage 1 ran
- Stage 1 pre-filter ran on the remaining 1,782 rows: 1,770 retained,
  12 excluded by RULE_LEN2
- Free pool after removing DEV50, eval458, AUDIT30, BLIND30: 1,712 rows

Verified overlaps: DEV50 vs Stage-1-retained = 0. eval458 vs
Stage-1-retained = 0. AUDIT30 overlaps by 28 (2 of its rows were among
the 12 excluded by RULE_LEN2). BLIND30 overlaps by 30.

Boundary rules:

- `DEV50` is a strict subset of `DEV200`
- `DEV200` is disjoint from the 100 / 250 / 500 training manifests
- `eval458` is disjoint from everything and stays untouched until §12

DEV200 construction rule (frozen):

- DEV200 = the 50 existing DEV50 rows + 150 new rows
- The 150 are drawn from the 1,712 free Stage-1-retained rows
- Plain random draw, seed 4242, not stratified
- Reason: DEV200 measures whether the method worked on the questions it
  targets. eval458 stays the unfiltered honest final metric.
- Verified 2026-07-24: all 50 DEV50 rows satisfy RULE_LEN2, so DEV200 is
  uniform in scope. No filtering mismatch exists.
- Built and verified 2026-07-24. Free pool 1,712 -> 150 sampled. All 18
  checks passed, including DEV50 subset and zero overlap with eval458,
  AUDIT30, BLIND30. Reproducibility confirmed by identical re-run.
  Script: `eval/error_analysis_tests/efbpt/efbpt_build_dev200.py`

---

## 4. Sampling manifests

- Sample from the 1,770 rows retained by the Stage 1 pre-filter.
- Exclude all protected sets from §3.
- Stratify by: answer label (yes/no) x hop count (len of
  official_decomposition). See AMENDMENT 1 — entity count and reasoning
  pattern were removed.
- Sampling seed for the manifest chain: 8888
- Fixed seed. Save the qid list to a file.

**Nesting is mandatory:** 100 ⊂ 250 ⊂ 500.

The 250 sampler must **load the saved 100-qid file** and add 150 new ones.
It must not resample from scratch. Same for 500 loading 250. If this is broken,
the 100→250 trend compares two different populations and means nothing.

Files:

- `data/strategyqa_official/efbpt/plan_a_qids_100.txt`
- `data/strategyqa_official/efbpt/plan_a_qids_250.txt`
- `data/strategyqa_official/efbpt/plan_a_qids_500.txt`

Built and verified 2026-07-24. Script:
`eval/error_analysis_tests/efbpt/efbpt_build_manifests.py`

Free pool: 1,562 rows (Stage-1 RETAINED minus DEV50, DEV200, eval458,
AUDIT30, BLIND30). Strata: 8 (answer label x hop bucket 2/3/4/5+).
Proportional quotas, largest-remainder. Nesting is guaranteed by
construction: each stratum is shuffled once with seed 8888 and the three
manifests take prefixes of the same order.

| Manifest | Rows | MD5 |
|---|---|---|
| `plan_a_qids_100.txt` | 100 | `c6bc963ce67850affc0ecb3d75f02b36` |
| `plan_a_qids_250.txt` | 250 | `76d96cfcb87dc0ec6515981b99f1d5ef` |
| `plan_a_qids_500.txt` | 500 | `3989ae7f9c47c619295492738d8626f0` |

Independently verified by reading the written files (not the generating
script): no duplicates, 100 fully inside 250, 250 fully inside 500, and
zero overlap with DEV50, DEV200, AUDIT30, BLIND30.

Composition of the 100-row manifest: 47 yes / 53 no; hops 2=28, 3=53,
4=15, 5+=4. This mirrors the pool proportions and is not forced to 50/50.

Known limitation: deep chains are rare in the pool (~4%). The n=100
manifest contains only 4 rows with 5+ hops. No claim about EFBPT's
effect on long reasoning chains may be made at the n=100 stage.

---

## 5. How rows are built

1. **Model drafts** a Schema C plan for the question.
2. **Structural validation runs first** (before human eyes). Checks:
   - output is valid JSON
   - all required fields present, no unknown fields
   - `type` and `atype` values are in the frozen enums
   - every `urdu_span` appears verbatim in `question_ur`
   - every non-null `entity_ref` matches a `canonical_title` in `entities`
   - `step_id` values are 1..N with no gaps
   A malformed draft is **repaired for the same qid**. Never swap in an easier
   question — that silently biases the dataset toward easy rows.
3. **Human reviews every row.** No auto-accept. This is the whole point of
   Plan A′ — agreement-based auto-accept was tested and rejected (33% coverage,
   70% precision, structural ceiling from correlated same-model error).
4. **Every correction is logged** (see §6).
5. **Structural validation runs again** after review.
6. Freeze the dataset file.

---

## 6. Correction log

Every human edit is written to
`data/strategyqa_official/efbpt/plan_a_corrections.jsonl`, one line per edit:

```json
{"qid": "...", "field": "entities", "error_type": "missing_entity",
 "before": "...", "after": "...", "note": ""}
```

Error types (frozen list):

| `error_type` | Meaning |
|---|---|
| `missing_entity` | Model left out a valid entity |
| `extra_entity` | Model added something that is not a valid entity |
| `wrong_title` | Entity is right but Wikipedia page is wrong |
| `wrong_span` | Span is wrong, not verbatim, or points at the wrong thing |
| `wrong_step_type` | `type` is wrong |
| `wrong_entity_ref` | Step points at the wrong entity |
| `wrong_atype` | `atype` is wrong |
| `wrong_step_text` | Step text is wrong or the plan structure is wrong |
| `wrong_answer` | Final answer is wrong |

**Why this matters:** this log produces a real number — "model drafts needed
edits in N% of rows, mostly on X" — which is a thesis finding on its own, and
is the only evidence that human review actually did something.

---

## 7. The four conditions (C0–C3)

All four use the same base model, same rows, same LoRA settings, same optimizer
steps, same seeds, same decoding. **Only the supervision target changes.**

| | Trained? | What the target contains |
|---|---|---|
| **C0** | No | Frozen base model. Floor. |
| **C1** | Yes | Final answer only. No plan, no entities. |
| **C2** | Yes | Typed plan steps + `atype` + final answer. **No** `canonical_title`, **no** `urdu_span`, **no** `entity_ref`. |
| **C3** | Yes | Everything in C2 **plus** the explicit entity bindings. Full Schema C. |

**C2 step text is NOT scrubbed.** C2 uses the natural plan text, which may
mention entity names inline. C2 is "a normal plan without explicit entity
fields." So **C3 vs C2 tests the value of explicit entity binding**, not the
value of mentioning entities at all. This is a deliberate choice: scrubbing
would create an unnatural baseline and more work.

**The thesis claim is proven by C3 > C2 and C3 > C1.**
C3 > C0 only shows that fine-tuning does something. That is not the claim.

---

## 8. Evaluation

### 8.1 Format A — schema-neutral (PRIMARY)

Input: the Urdu question only. No schema in the prompt.
Output:

```json
{"reasoning_ur": "...", "answer": "yes"}
```

This measures whether reasoning actually improved — not whether C3 learned to
reproduce its own training format. This is the number that matters.

### 8.2 Format B — structured (SECONDARY)

All conditions are asked to produce frozen Schema C plus the final answer.
Measured:

- schema-valid output rate
- valid step-type rate
- valid `entity_ref` rate
- entity-binding correctness

### 8.3 Rules for both formats

- Identical prompt and identical decoding settings across C0–C3.
- Freeze and record: `max_tokens`, thinking ON/OFF, temperature.
- **Report truncation rate per condition.** Mismatched truncation has already
  produced sign-flipped results twice in this project.

Frozen decoding settings — identical for C0, C1, C2, C3:

- thinking: OFF
- temperature: 0
- max_tokens: 1024

max_tokens is set high deliberately. A 128-token limit previously caused
32% truncation on CSQA and produced an invalid result.

---

## 9. The c-probe (mechanism metric)

This is the metric that decides whether EFBPT worked **for the stated reason**.

For every gold target entity in the model's own Urdu reasoning trace, assign
exactly one status:

- `preserved` — the entity appears and means the right thing
- `corrupted` — it appears but resolves to a wrong or invented identity
- `omitted` — it does not appear at all

Reported:

```
corruption rate   = corrupted / all gold target entities
omission rate     = omitted   / all gold target entities
hallucinated rate = extra invented entities / all gold target entities
```

Omission is counted separately on purpose. Without it, a model could just stop
naming entities and look perfectly faithful.

### 9.1 Testability rule (carried over from the original c-probe)

Only entities with **transliteration drift risk** are counted — where the Urdu
span could plausibly resolve to a *different* identity. Phonetically obvious
entities (Hand, Soup, Brain, Teddy bear) are excluded. Including them inflates
the denominator with guaranteed-faithful cases and fakes a low corruption rate.

### 9.2 Workload rule

- **No c-probe at N=100.** Accuracy only.
- **At N=250: score C2 and C3 only**, on the **median-accuracy seed** of each,
  on the fixed `DEV50`. That is ~100 traces total, not 1,000.
- C0 and C1 get accuracy evaluation only.
- No complicated pre-scorer is built at this stage.

---

## 10. Training and seeds

- C0 is deterministic — no seeds needed.
- C1, C2, C3 are each trained with **3 identical seeds**.
- Report per-seed results and mean ± standard deviation.
- QLoRA on 100 rows is high-variance. A single run cannot tell a real gain
  from seed noise.

**Token budgets are NOT matched.** Same rows, same epochs, same optimizer
steps, same settings. C3 targets are longer than C1 targets, so C3 sees more
supervised tokens. This is recorded, not corrected:

- Record supervised-token count per condition.
- Report the target-length difference as a stated limitation.

Frozen training settings — identical for C1, C2, C3:

- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- learning rate: 2e-4
- epochs: 3
- per-device batch size: 4
- gradient accumulation: 4 (effective batch 16)
- max sequence length: 1024
- seeds: 13, 42, 2026

These are reasonable starting values, not tuned optima. Tuning them is
deliberately out of scope. What matters is that every condition gets
identical settings.

---

## 11. Expansion gate — 250 → 500

Pre-declared. Do not lower after seeing results.

Build the 500-row dataset **only if all five** hold:

1. C3-250 beats **both** C1-250 and C2-250 by **≥ 3 accuracy points** on
   `DEV200` (Format A, mean over seeds).
2. C3 corruption rate is **≥ 20% lower (relative)** than C2 on `DEV50`.
3. C3 omission rate rises by **no more than 2 percentage points absolute**
   versus C2.
4. The improvement appears in **at least 2 of 3 seeds**.
5. C3-250 is **not worse** than C3-100.

If the gate fails: STOP. Do not redesign the method using this data.

**Note on the 3-point threshold:** it is a practical decision about whether
more annotation work is worth it. It is not statistical proof. Final evidence
comes from `eval458`.

### 11.1 What "EFBPT worked" requires

All three must hold, or the claim is not made:

- C3 has **lower corruption** than C2, **and**
- C3 does **not** get there by omitting more, **and**
- C3 beats C2 and C1 on **final-answer accuracy**

If accuracy goes up but corruption does not go down, EFBPT did not work through
entity faithfulness — something else did. That is exactly the StrategyQA
label-copying lesson. Say so honestly.

---

## 12. Final evaluation

- `eval458` is opened **once**, after the final configuration is chosen and
  frozen. Never during development.
- Paired significance testing on final-answer accuracy.
- Mechanism scoring on a **pre-declared random 50-item subset** of `eval458`,
  scored blind.

Seed for the eval458 50-item mechanism subset: 20260724

The qid list must be drawn and saved to a file before any results exist.

---

## 13. Self-consistency check

After finishing the 100-row set, wait about one week, then blind re-review
**15 randomly chosen rows** from it. Report the agreement with the original
review.

This is **intra-annotator consistency**, not inter-annotator agreement. There
is only one annotator, so real IAA is impossible. Report it as the weak measure
it is. Weak and honest beats silent.

---

## 14. Open items — must be filled before the first draft row

- [x] §1.5 valid step types — paste from code
- [x] §1.6 valid `atype` values — paste from code
- [x] §3 build `DEV200`, save qid list
- [x] §8.3 decoding settings — max_tokens, thinking, temperature
- [x] §10 LoRA hyperparameters and the 3 seed values
- [x] §12 seed for the eval458 50-item mechanism subset

When all six boxes are ticked, this document is FROZEN and dataset generation
begins.

---

## AMENDMENTS

### AMENDMENT 1 — 2026-07-24 — stratification fields reduced

Original §4 said: stratify by hop count, entity count, yes/no label,
reasoning pattern.

Changed to: stratify by answer label x hop count only.

Reason:

1. Entity count does not exist before annotation. Entities are the
   output of the human annotation step, so they cannot be an input to
   sampling.
2. Reasoning pattern was never defined. No category list exists in this
   project or in the source dataset.
3. A proxy for entity count was considered and REJECTED. The proposed
   proxy was the number of distinct Wikipedia pages in
   `evidence_paragraph_ids`. It fails construct validity: in the mapped
   dataset, the Genghis Khan / Julius Caesar row names 2 entities but
   lists 7 distinct evidence pages. Per Geva et al., TACL 2021, evidence
   is matched to each decomposition by 3 independent workers, so this
   field is a union of three annotators' retrieval choices. It measures
   evidence breadth, not entity count.
4. A proxy for reasoning pattern using `#N` back-references was also
   REJECTED. `#N` only marks that a step reuses an earlier answer. It
   does not identify comparison, causality, temporal reasoning,
   conjunction, or negation, and its count largely tracks hop count,
   which is already a stratification field.

Decision rule applied: a mislabeled control is worse than a stated
limitation. A broken proxy would bias the sample toward evidence density
while the writeup claimed it controlled entity complexity, and that
error would be undiscoverable later.

Remaining limitation (must be reported): entity-complexity balance in
the training manifests is NOT guaranteed. Mitigation: after the first
100 rows are annotated, measure and report the true entity-count
distribution from the annotations. If it is badly skewed, a further
dated amendment may change how the +150 rows are drawn — decided then,
with real numbers.

## AMENDMENT 2 — 2026-07-25

Three items closed before the first draft row is generated.

### (a) Answer value mapping

The mapped file field `answer` is boolean `true` / `false`. Schema C `answer`
is `"yes"` / `"no"`. The mapping is fixed:

- `true`  -> `"yes"`
- `false` -> `"no"`

The generator asserts that no other value exists. Verified on the 100-row
manifest: 47 true, 53 false, 0 other.

### (b) Step text language — ENGLISH

`steps[].text` is copied verbatim from `official_decomposition` in
`strategyqa_official_mapped_urbench_qid.jsonl`. No translation, no rewriting,
no model generation of step text.

Reasons:
1. The decomposition is human-written gold (Geva et al., TACL 2021).
2. Translating it with Qwen3-14B would inject the exact entity corruption
   EFBPT targets into the training labels.
3. The frozen bilingual signal is: Urdu span -> English Wikipedia title ->
   typed plan. Urdu step text was never part of that signal.
4. C2 and C3 share identical step text, so the comparison still isolates
   explicit entity binding.

Stated limitation for the write-up: the plan is written in English while the
question and the entity spans are Urdu. The claim is "bilingual plan
supervision improves reasoning on Urdu questions", NOT "the model reasons
entirely in Urdu".

### (c) entity_ref on bridge steps — overrides section 1.3

Section 1.3 as written requires every `retrieve` step's `entity_ref` to match
a `canonical_title` in `entities`. But `entities` contains only entities named
in the question, so a step such as "When did #1 develop?" has no legal value.
That rule is replaced by:

- `retrieve`, direct: `entity_ref` = a `canonical_title` present in `entities`.
- `retrieve`, bridge (the step text refers to `#N`): `entity_ref` = the
  canonical_title of the entity that step N's answer resolved to. It need not
  appear in `entities`.
- `reason`: `entity_ref` = JSON `null`.

Validation: a `retrieve` step's `entity_ref` is legal if it matches a
`canonical_title` in `entities` OR a title in that row's permitted candidate
universe (evidence page titles plus the row `term`). This matches the universe
check already implemented in `efbpt_stage3_core_verify.py`.

Justification: BLIND30 `entity_ref` accuracy was 96.6% with only 3
corrections, so this convention is cheap to annotate. `evidence_ref` and
`gold_intermediate_answer` caused 77 of 93 corrections and remain dropped.

Measurement on the 100-row manifest: 295 steps total, 133 contain `#`, and all
100 rows contain at least one. Not all are bridge retrievals — some are
`reason` steps, and some use `#N` as the object while the subject is a
question entity.

## AMENDMENT 3 — Training serialization, 100→250 gate, and entity-faithfulness

## protocol (frozen 2026-07-28, before any training data was generated)

### A. Training serialization (C0–C3)

* Input: identical for all conditions. One fixed system message + one fixed
  short instruction + question_ur. No condition-specific text. No schema in
  the prompt. qid is metadata only, never model input.
* Targets, deterministic JSON, frozen key order, no whitespace variation:

  * C1: {"answer":"yes|no"}
  * C2: {"steps":[{"step_id","text","type","atype"}],"answer":...}
  * C3: {"entities":[{"canonical_title","urdu_span"}],
    "steps":[{"step_id","text","type","atype","entity_ref"}],"answer":...}
* C2 and C3 step text MUST be byte-identical (isolates entity binding).
* Candidate titles, audit data, reviewer notes: never in any target.
* C0 = untrained base, same input format at eval.

### B. Format A amended (extractor-based)

Format A output is free-form; the final answer is recovered by ONE frozen
extractor applied identically to C0–C3. Rationale: targets do not contain
reasoning_ur; requiring it would contradict AMENDMENT 2b (English step text).
Primary gate metric = final-answer accuracy on DEV200 via this extractor.
Urdu-reasoning presence and JSON validity: reported separately, never gating.

### C. 100→250 expansion gate (pre-declared; no auto-expansion)

Order of evaluation:

1. Accuracy first. C3-100 mean DEV200 accuracy must beat BOTH C1-100 and
   C2-100, and beat each in ≥2 of 3 paired seeds (13/42/2026).
   If this fails: stop, log, do not run the faithfulness probe.
2. Entity-faithfulness probe (only if step 1 passes): on DEV50,
   C3 mean corruption < C2 mean corruption, in ≥2 of 3 paired seeds, AND
   C3 mean omission ≤ C2 mean omission + 2pp.
   No ≥3pt margin required at 100 (that threshold belongs to 250→500).
   A C3−C2 difference of ≤2 judgments total = inconclusive, not a pass.
   No condition may be redefined after any number is seen.

### D. Entity-faithfulness probe (frozen instrument)

* Gold list: evaluation-only annotation of testable entities for DEV50
  (exact Urdu span + canonical identity + accepted spelling variants).
  "Testable" reuses the frozen transliteration-drift-risk rule from the
  original (c)-probe — NOT a new rule — so rates remain comparable to the
  ~39% corruption baseline. Never enters any model prompt.
* Probe prompt: one frozen prompt requesting brief Urdu reasoning + yes/no.
  Identical for C2 and C3, all seeds. thinking OFF, temp 0, max_tokens 1024.
* Normalization: strip JSON/schema structure, keep reasoning text + answer;
  shuffle across conditions and seeds; mapping key stored separately and
  never shown to the judge. Blinding acknowledged as imperfect (C3 style
  may leak); recorded as a limitation.
* Judge: LLM judge (Claude, fixed model + fixed prompt), one output per
  call, no batching. Per gold entity, exactly one label:
  faithful | corrupted | omitted, each with a verbatim supporting quote and
  a one-line reason. All raw judgments saved.
* Metrics: corruption = corrupted/required instances;
  omission = omitted/required instances. Same denominator. Per-seed + mean.
* Human audit: pre-declared balanced sample of 60–100 judgments, balanced
  across conditions, seeds, and LABELS (oversampling corrupted/omitted).
  Acceptance: ≥90% human–judge agreement AND no systematic disagreement
  direction against one condition. Below 90%: inspect disagreements and
  expand the audit before using any judgment.
* Limitations recorded in advance: single-LLM judge (vs 3 human readers in
  the original probe), imperfect blinding, no minimum corruption margin —
  results at 100 rows are a directional signal, not proof.
* Format B: secondary diagnostic only (JSON validity, valid entity_ref,
  binding correctness). Never gates expansion — it structurally favors C3.

## AMENDMENT 4 — Fixed prompt (frozen 2026-07-28)

AMENDMENT 3 Section A required "one fixed system message + one fixed short
instruction," but never recorded the actual strings. This amendment fixes them.
Frozen before any training data exists.

### A. Fixed system message (English)

    You are a helpful assistant. Answer the user's question.

### B. Fixed instruction (Urdu)

Stored as a file, NOT as a hardcoded literal:
`prompts/efbpt/plan_a_instruction_ur.txt`

- UTF-8, no BOM, no trailing newline
- 20 characters, 36 bytes
- MD5 (raw bytes): `f3b58d766fe3ec2573ff4f24761cf0c9`
- Codepoints in stored order:
  0627 0633 0020 0633 0648 0627 0644 0020 06A9 0627 0020 062C 0648 0627
  0628 0020 062F 06CC 06BA 06D4
- English gloss (not used in any prompt): "Answer this question."

The file was generated from the codepoint list above, not typed or pasted, to
eliminate copy-paste substitution risk. Every script MUST read this file at
runtime and MUST verify the MD5. No script may hardcode the string.

### C. User message assembly (frozen)

    <instruction><LF><LF><question_ur>

Nothing else. No schema, no facts, no candidate titles, no format constraint,
no qid.

### D. Scope

Identical for C0, C1, C2 and C3, at BOTH training time and DEV200 evaluation
time. No condition-specific prompt text exists anywhere in the pipeline.

### E. Rationale (recorded so it is not relitigated)

1. Existing StrategyQA prompts under `prompts/strategyqa/` could NOT be reused
   verbatim, for two independent reasons: every one contains a `{facts}`
   placeholder (EFBPT supplies no facts, so the wording would promise evidence
   that never arrives), and every one forces single-word output (which
   contradicts free-form Format A under AMENDMENT 3B and contradicts the JSON
   step targets of C2 and C3).
2. The instruction is Urdu, matching URBench/StrategyQA evaluation practice,
   to avoid pushing output language toward English. AMENDMENT 2b freezes plan
   STEP TEXT in English only; it does not govern the user instruction.
3. The system message stays English because it is identical across all four
   conditions and therefore cannot confound the C1/C2/C3 comparison. It may
   influence output language, which is a reported non-gating metric only.
4. The instruction carries no output-format constraint, so it does not favour
   any condition's target shape.

### F. Correction to prior handoff notes

`prompts/strategyqa/cot.txt` does not exist. The real StrategyQA CoT templates
are `cot_p1.txt`, `cot_p2.txt`, `cot_p3.txt` under `prompts/strategyqa/`.

## AMENDMENT 5 — Frozen answer extractor for DEV200 (frozen 2026-07-29)

AMENDMENT 3B requires "ONE frozen extractor applied identically to C0-C3" but
never defined it. This amendment defines it. Frozen BEFORE any DEV200 output
has been generated.

### A. Why the existing StrategyQA extractor cannot be reused as-is

`eval/error_analysis_tests/sdfr_strategyqa_fair.py` (and its siblings) search
for Urdu ہاں / نہیں first, and fall back to English only via a bare substring
test (`"no" in text.lower()`).

C1/C2/C3 are trained to emit English JSON such as `{"answer":"no"}` and contain
no Urdu at all, so the Urdu branch finds nothing. The English fallback then
matches "no" inside ordinary words — not, nothing, north, Nobel — and C3's
targets carry English entity titles and step text, so both "yes" and "no"
substrings frequently co-occur, returning "" (unparsed).

C0 is untrained, answers in Urdu prose, and parses normally.

Reusing that function unchanged would have systematically failed the trained
conditions and handed C0 an artificial advantage. Recorded here because the
defect was found by inspection before any number existed, not after.

### B. The frozen extractor

Applied identically to C0, C1, C2 and C3, all seeds. Ordered; first rule that
matches wins; no later rule may override an earlier one.

1. JSON answer field.
   Regex `"answer"\s*:\s*"(yes|no)"`, case-insensitive.
   Take the LAST match in the output. Return "yes" or "no".
2. Urdu.
   If the marker حتمی جواب occurs, discard everything before its LAST
   occurrence. In the remaining text take rfind of ہاں and rfind of نہیں;
   whichever index is greater wins. Return "yes" for ہاں, "no" for نہیں.
3. English, word-bounded.
   Regex `\b(yes|no)\b`, case-insensitive. Take the LAST match.
   The word boundary is required; it is what fixes the not/Nobel defect in A.
4. Otherwise return None (unparsed).

Rule 1 exists because C1-C3 are trained to emit JSON and C0 is not. An
extractor reading only Urdu would measure output-format compliance, not
reasoning.

Rule 2 keeps the rfind "last mention wins" behaviour of the existing
StrategyQA extractors so EFBPT numbers stay comparable to the project's
earlier StrategyQA results. The "first 20 characters" variant used in the
English-pivot scripts is deliberately NOT used: it assumes the answer
immediately follows the marker, which that prompt enforces and the EFBPT
prompt does not.

### C. Scoring rules

- Gold mapping (AMENDMENT 2): DEV200 `answer: true` -> "yes", `false` -> "no".
- Unparsed (None) is scored INCORRECT. It is not excluded from the
  denominator. Excluding it would reward a model for staying silent on
  questions it cannot answer, and would give conditions different
  denominators, making accuracies non-comparable.
- Accuracy denominator is always 200 for every condition and seed.

### D. Mandatory reporting — per condition, per seed

Reported alongside every accuracy number, without exception:

1. unparsed rate
2. truncation rate (output reached max_tokens = 1024)
3. predicted-yes rate

DEV200 label balance is 87 yes / 113 no. **A constant "no" scores 56.5%.**
Any condition near 56.5% has learned a label prior, not reasoning, and must be
reported as such.

**Validity condition on the 100->250 gate:** if the unparsed rate differs
materially between conditions, the accuracy comparison is not valid and the
gate is void regardless of the accuracy ordering. Diagnose the parsing gap
first. Mismatched truncation has already produced sign-flipped results twice
in this project; unparsed rate is the same class of defect.

### E. Known limitation, accepted deliberately

Rule 2's "last mention wins" misreads negated phrasing: "the answer is not
ہاں" scores yes. This defect is already present in every existing StrategyQA
baseline in this repo. It is retained so that the error is IDENTICAL across
C0-C3 and comparable with prior results. Fixing it now would break
comparability with numbers already logged in experiments.md. Logged as a
limitation, not fixed.

### F. Correction to prior notes

The DEV200 file is `data/strategyqa_official/dev200_seed4242.jsonl`.
It is NOT under `data/strategyqa_official/efbpt/`. Every row has
`is_eval: false`; that field is inherited from the source dataset and carries
no meaning for this evaluation.

Contamination check performed 2026-07-29: 0 of 200 DEV200 `urbench_qid`
values appear as a `qid` in `plan_a_train_c3_100.jsonl`. No overlap.

## DIAGNOSTIC D1 — Knowledge vs language bottleneck (frozen 2026-07-30)

This is NOT part of Plan A'. Plan A' is closed: the 100->250 gate failed and
the counterfactual experiment established why (entity labels are read but
knowledge-inert). D1 is a separate diagnostic, frozen before it is run, to
decide what the next method should attack.

### A. Question

Plan A' tested STRUCTURE (plans, entity blocks, entity binding). None of it
raised accuracy above ~65%. The untested question is whether the ceiling comes
from missing KNOWLEDGE or from inability to REASON IN URDU.

### B. Confound that forces four arms

`urbench_facts` in DEV200 is 100% English: 532 fact strings, 0 Arabic-block
characters, 28,343 ASCII letters. `question_ur` is 195/200 pure Arabic script.
So the dataset pairs an Urdu question with English evidence.

Adding English facts alone cannot distinguish three explanations for any gain:
(1) the model lacked knowledge, (2) the model reasons better in English,
(3) English facts leak the answer via lexical overlap with the English source
question. Four arms are therefore required.

### C. The four arms

All on the same 200 DEV200 rows. Base Qwen3-14B, NO adapters.

| arm | question | facts supplied | isolates |
|---|---|---|---|
| A | Urdu | none | baseline |
| B | Urdu | gold, English (verbatim `urbench_facts`) | knowledge + language |
| C | Urdu | English facts from a DIFFERENT row | whether facts are read at all |
| D | Urdu | gold facts machine-translated to Urdu | knowledge, language fixed |

**B vs D is the primary comparison.** Same knowledge content, different
language. B >> D implies the bottleneck is Urdu reasoning. B ~= D implies the
bottleneck is knowledge and translation is a viable route.

Arm C donor assignment: fixed deterministic shift, same scheme as the
counterfactual experiment (shift 97 over the row list), and the script MUST
report the count of rows whose fact set actually changed.

### D. Frozen decoding — identical to the EFBPT regime, NOT the old baselines

- transformers + 4-bit nf4 (double quant, bf16 compute), attn = sdpa
- thinking OFF, temperature 0 (greedy), max_new_tokens 1024
- prompt = fixed system message + Urdu instruction file
  (`prompts/efbpt/plan_a_instruction_ur.txt`, MD5 verified at runtime)
  + facts block (arms B/C/D) + `question_ur`
- answer extractor: IMPORTED from `efbpt_eval_dev200.py`, the frozen
  AMENDMENT 5 extractor. Never re-implemented.
- gold mapping: `answer: true` -> "yes", `false` -> "no" (AMENDMENT 2)
- unparsed scored INCORRECT; denominator always 200 (AMENDMENT 5C)

Deliberately NOT reusing `cot_strategyqa_nofacts_baseline_fair.py`: it runs
thinking ON, max_tokens 2048, vLLM bf16, on the `data/sdfr_splits` eval set.
Numbers from that regime cannot be compared to C0-C3. Token cost is not a
concern here: facts + question is max 141 tokens, p95 116, so 0/200 rows
approach any limit.

### E. Built-in validity control

**Arm A MUST reproduce C0 (57.50% on DEV200).** Arm A uses the identical
prompt, model, quantization and decoding as C0. A deviation greater than
~2pp means the setup differs from the main evaluation in some unintended way,
and NO arm may be interpreted until that is explained. This is a hard
precondition, not a soft check.

### F. Mandatory reporting, per arm

accuracy, unparsed rate, truncation rate, predicted-yes rate, and the
always-"no" floor (56.50%). Under AMENDMENT 5D, if the unparsed-rate spread
across arms exceeds 5pp the accuracy comparison is VOID and the parsing gap
must be diagnosed first. Any subgroup claim must include a matched control on
the SAME qid set (two analysis errors in the Plan A' read-out came from
comparing groups against their own floor instead of a matched control).

### G. Pre-declared readings — fixed before any number is seen

1. If C ~= B, the model is not reading the facts and no arm is interpretable.
   This check comes FIRST; nothing else is read until it passes.
2. If B and D both reach >= 80%, knowledge is the bottleneck. The next method
   should target getting Urdu entity knowledge to the model, not structuring
   plans. Span extraction at 97.89% is an existing asset for that.
3. If B >= 80% but D <= 70%, the bottleneck is Urdu reasoning, not knowledge.
   Supplying Urdu knowledge would not be sufficient and that route is closed.
4. If B and D are both <= 70%, neither knowledge nor language is sufficient:
   the model cannot combine given facts in this task at all. This closes a
   whole family of retrieval-style approaches and is itself a reportable
   finding.
5. Any gain in B must be checked against answer leakage: report how often the
   gold answer is lexically inferable from the English facts alone.

### H. Translation for arm D

Facts are translated by Qwen3-14B itself in a separate pass (same model, no
new dependency), greedy, thinking OFF. The translated file is written once,
MD5-recorded, and reused unchanged by the evaluation. Translation quality is
a limitation and is recorded as such: a poor translation would depress arm D
and could masquerade as evidence for reading 3. To bound this, a sample of
translations must be human-checked before arm D is interpreted, and the
sample size and outcome recorded here.

### I. Status
Declared before execution. No arm has been run. No numbers exist.

## DIAGNOSTIC D1 — AMENDMENT 1 (2026-07-31): arm D withdrawn, arms E and F added

Amended BEFORE any D1 arm was scored. No D1 numbers exist. The only D1 work
executed so far is the translation TEST pass (job 58857, 12 facts), whose sole
purpose was to check translation quality before committing to arm D.

### A. Why arm D is withdrawn

Arm D required machine-translating the English DEV200 facts into Urdu with
Qwen3-14B. A 12-item TEST pass was human-reviewed by a native Urdu speaker
(the author) and independently inspected. The translations are not usable.

Meaning-changing errors, verbatim from `outputs/efbpt/d1/`:

1. NEGATION DELETED. "A Toyota Supra does NOT have consciousness to recount
   any experiences" was rendered as an affirmative statement. Polarity flipped.
   On a yes/no task this alone flips the gold-supported answer.
2. ENTITY REPLACED. "A goat is a mammal" -> the Urdu word used means "bat".
3. "US Navy plane" -> the Urdu word used means "piano".
4. "baker's dozen" -> rendered as "gator's dozen".
5. "toads / snails" -> rendered as "pills" and unrelated animals.
6. "rodent" -> a non-word.
7. "CPU circuit chip" -> Latin letters spliced inside an Urdu word
   ("سirkvit"), a broken mixed-script token.
8. "unlucky number" -> a phrase meaning "without pleasure"; wrong sense.

Approximately 8-9 of 12 carried real errors; 2 of 12 carried errors that would
change the answer. Only 1 of 12 (the "blacklist" fact) was clean.

The failure mode is note-worthy in itself: the translator commits EXACTLY the
disease this thesis studies — entity corruption during Urdu generation
(plane->piano, mammal->bat, baker->gator), the same pattern as the (c)-probe
corruptions (Roewe 550 -> rowing boat, Hades -> Ukrainian singer). This is
independent confirmation of the core finding at the FACT level rather than the
reasoning level.

Consequence: a low arm-D score would mean "the translator corrupted the facts"
but would READ as "the model cannot reason in Urdu" — pre-declared Reading 3
of Section G. That is an uninterpretable arm and it is withdrawn rather than
run. Section H already warned of exactly this risk; the warning fired.

No alternative translator is available. The project's entire baseline suite is
Qwen3-14B; substituting a different model for translation only would break
comparability with every number already logged in experiments.md, and would
introduce a second, unmeasured system into the causal chain.

### B. The replacement: a 2x2 using gold human data only

DEV200 already contains `question_en`, the original human-written English
StrategyQA question, alongside `question_ur` (URBench's Urdu version) and the
English `urbench_facts`. Language can therefore be varied using human-written
text on BOTH sides, with no machine translation anywhere in the pipeline.

| arm | question | facts supplied | isolates |
|---|---|---|---|
| A | Urdu (`question_ur`) | none | baseline; must reproduce C0 |
| B | Urdu | gold English `urbench_facts` | knowledge, Urdu question |
| C | Urdu | English facts from a DIFFERENT row | are facts read at all |
| E | English (`question_en`) | gold English `urbench_facts` | full-English ceiling |
| F | English (`question_en`) | none | question language alone |

This is a clean 2x2 of question language x facts present, plus control arm C.

Contrasts, declared now:
- F - A : cost of the Urdu question by itself, with no facts involved.
- B - A : value of supplying knowledge, Urdu question held fixed.
- E - B : cost of the Urdu question WHEN knowledge is already given.
- E     : the model's ceiling on this task under ideal conditions.

Arm C remains the first thing read: if C is close to B, facts are not being
used and no other arm is interpretable.

### C. Revised pre-declared readings (replaces Section G readings 2-4)

1. (unchanged) If C ~= B, the model is not reading the facts. Stop; nothing
   else is interpretable.
2. If E and B are both high (>= 80%) and E - B is small (<= 5pp): knowledge is
   the bottleneck and question language costs little. The next method should
   target getting knowledge to the model. Span extraction at 97.89% is an
   existing asset for that.
3. If E is high (>= 80%) but B is much lower (E - B >= 15pp): the Urdu QUESTION
   is the bottleneck even when knowledge is given. Supplying Urdu knowledge
   would not be sufficient; the method must attack Urdu comprehension.
4. If E is also low (<= 70%): neither knowledge nor English phrasing is
   sufficient. The model cannot combine given facts on this task at all. This
   closes retrieval-style approaches as a family and is itself reportable.
5. F - A is reported regardless of the above. It quantifies how much accuracy
   URBench's own Urdu question translation costs relative to the original
   English, which is a standalone Urdu-NLP result.
6. (unchanged) Any gain in B or E must be checked for answer leakage: report
   how often the gold answer is lexically inferable from the facts alone.

### D. Unchanged from the original D1 freeze

Sections D (decoding: 4-bit nf4, thinking OFF, temperature 0,
max_new_tokens 1024, frozen AMENDMENT 5 extractor imported, unparsed scored
incorrect, denominator 200), E (arm A MUST reproduce C0 = 57.50% within ~2pp
or nothing is interpreted), and F (mandatory reporting; >5pp unparsed spread
voids the comparison; subgroup claims require a matched control on the same
qid set) all carry over unchanged and apply to arms A, B, C, E and F.

Arm C donor assignment: fixed deterministic shift of 97 over the row list;
the script must report how many rows' fact sets actually changed.

Note for arms E and F: the fixed Urdu instruction from AMENDMENT 4 is designed
for an Urdu question. For an English question the instruction language becomes
a confound. Decision, frozen here: arms E and F use the SAME Urdu instruction
and the same system message as arms A/B/C, so that the ONLY thing varying
across the 2x2 is the language of the question and the presence of facts. The
instruction is a fixed constant across all five arms and therefore cannot
explain any difference between them. This is recorded as a limitation: a
model may respond differently to an Urdu instruction paired with an English
question, and that interaction is not measured here.

### E. Salvage: the translation failure becomes a measured result

The TEST-pass translations are not discarded. A human-labelled sample of the
machine-translated facts (target: 50 items, labelled clean / awkward but
correct / entity corrupted / meaning flipped) will be recorded, giving a
quantified statement of the form: "Qwen3-14B translation of factual English
sentences into Urdu corrupts X% of statements, including Y% that invert
polarity." Sample size, labeller, and per-category counts to be recorded in
experiments.md. This documents on the record why arm D was withdrawn, and
stands as an Urdu-NLP finding in its own right.

### F. Status
Amended before execution. No D1 arm has been scored. No D1 numbers exist.

## DIAGNOSTIC D1 — AMENDMENT 2 (2026-07-31): Devanagari script drift

Declared BEFORE the full D1 run. The only D1 execution so far is the TEST pass
(job 58880, 20 rows), whose purpose is exactly this kind of check. No D1 result
has been scored.

### A. Observation

In the TEST pass the model produced answers in DEVANAGARI (Hindi script)
instead of Perso-Arabic (Urdu script), but only in certain arms:

| arm | question | unparsed in TEST |
|---|---|---|
| A | Urdu script | 0% |
| B | Urdu script | 0% |
| C | Urdu script | 0% |
| E | English | 10% |
| F | English | 15% |
| G | none | 10% |

Urdu and Hindi are the same spoken language (Hindustani) in two scripts, and
the affirmative/negative words are the same words: Urdu ہاں / نہیں correspond
to Devanagari हां / नहीं. The model therefore answered in the correct LANGUAGE
but the wrong SCRIPT.

The pattern is systematic, not random. When the question is in Urdu script the
model stays in Urdu script. When the instruction is Urdu but the question is
English or absent, there is no script anchor in the input and the model falls
into Devanagari, the dominant script for Hindustani in web-scale training data.

A second, distinct cause of unparsed output also appears in arm F: fluent
English answers that discuss the question at length without ever stating a
yes/no verdict (e.g. the Apollo 15 row). That is not script drift and is not
addressed by this amendment; it is a genuine failure to commit to an answer
and is scored incorrect under AMENDMENT 5C.

### B. Why this threatens D1

The frozen AMENDMENT 5 extractor matches Perso-Arabic ہاں / نہیں and
word-bounded English yes / no. Devanagari matches nothing, so a correct answer
is scored unparsed and therefore incorrect.

TEST unparsed spread across arms is 15pp. AMENDMENT 5D voids the accuracy
comparison above 5pp. Without this amendment the full run would produce a
result the protocol correctly refuses to interpret.

### C. Decision: dual scoring, primary extractor UNCHANGED

The AMENDMENT 5 extractor is NOT modified. It remains the primary scorer, so
D1 numbers stay comparable with C0-C3 and with every StrategyQA number already
in experiments.md.

A SECONDARY score is computed OFFLINE from the saved generations (every
generation is stored in full, so no re-run is needed). The secondary scorer is
the AMENDMENT 5 extractor with one additional rule inserted between its Rule 2
(Urdu) and Rule 3 (English):

  Rule 2b — Devanagari. In the same text segment used by Rule 2, take rfind of
  each of these strings; the greatest index wins:
    yes: U+0939 U+093E U+0902   (हां)
    yes: U+0939 U+093E U+0901   (हाँ)
    no : U+0928 U+0939 U+0940 U+0902   (नहीं)
  Return "yes" or "no" accordingly. Codepoints are listed explicitly and the
  scorer MUST build these strings from codepoints, never from typed text.

Both scores are reported for every arm, always together, never one alone.

### D. How the two scores are read

1. If the PRIMARY score has an unparsed spread <= 5pp across arms, the primary
   score is the result. The secondary is reported as a robustness check only.
2. If the PRIMARY score's unparsed spread exceeds 5pp AND the secondary score's
   spread is <= 5pp, the primary comparison is VOID under AMENDMENT 5D and the
   SECONDARY score becomes the interpretable analysis. This substitution is
   permitted ONLY because it is declared here, before any full-run number
   exists.
3. If BOTH scores show a spread > 5pp, neither comparison is valid. Report the
   numbers, diagnose the remaining parsing gap, and draw no conclusion about
   the knowledge-vs-language question.
4. The DIFFERENCE between primary and secondary, per arm, is itself a reported
   measurement: it is the Devanagari script-drift rate.

### E. This is a finding, not only a nuisance

The script-drift rate per arm is reported as an Urdu-NLP result in its own
right: Qwen3-14B, given an Urdu instruction, holds Urdu script when the
question is in Urdu script, but drifts to Devanagari when the question is
English or absent. This means Urdu prompting is script-fragile in a way that
English prompting is not, and any Urdu evaluation whose answer extractor
assumes Perso-Arabic will silently under-score such outputs. Existing Urdu
extractors in this repository, including the one frozen in AMENDMENT 5, have
this blind spot.

Limitation recorded: the drift was observed on a 20-row TEST pass, so the
rates above are indicative only. The full run supplies the real numbers.

### F. Status
Declared before the full D1 run. No D1 arm has been scored at 200 rows.
