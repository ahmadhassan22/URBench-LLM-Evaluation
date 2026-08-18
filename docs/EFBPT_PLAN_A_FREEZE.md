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

## DIAGNOSTIC D2 — Retrieval re-test on the full index (frozen 2026-07-31)

Declared before execution. No D2 number exists.

### A. Why the earlier RAG result must be set aside

The RAG section of experiments.md reports "RAG reduced StrategyQA accuracy by
27.65 percentage points (83.89% -> 56.24%)". Two defects invalidate that
comparison. Both are established from the repository itself, not inferred.

**Defect 1 — the baseline was given gold facts.**
The RAG section's own Setup states: "Prompt: Same zero-shot Urdu prompt
structure as baseline, with retrieved passages replacing gold facts."
The note added 2026-07-07 at experiments.md:1211 confirms the regime for that
number: "thinking ON, facts included in-prompt via `{facts}`/`{question}`
template". So 83.89% is a GOLD-FACTS score, and the label "(no RAG)" meant
"no retrieval", not "no facts". The -27.65pp therefore compares gold facts
against retrieved passages, which cannot answer whether retrieval helps.

Against a true no-facts baseline the picture reverses:

| setting | accuracy |
|---|---|
| gold facts, thinking ON, 2290 q | 83.89% |
| gold facts, thinking OFF, DEV200 (D1 arm B) | 78.00% |
| retrieved passages (old RAG) | 56.24% |
| NO facts, thinking OFF, DEV200 (D1 arm A) | 57.50% |

Retrieval was worth about -1.3pp, i.e. nothing — not a 27.65pp collapse.
Caveat recorded: the old runs used 2290 questions with thinking ON and the D1
arms used 200 with thinking OFF, so this is indicative, not a head-to-head.
D2 exists to produce the head-to-head.

**Defect 2 — the index was tiny.**
`rag/eval_rag_final.py` loads `rag/index/wikipedia.index`. That file is
81,893,421 bytes. 53,316 chunks x 384 dims x 4 bytes = 81,893,376 bytes, a
45-byte header difference: it is the ENTITY-FILTERED index of 53,316 chunks
over 2,152 unique article titles, as stated in the RAG section's own Corpus
Construction block.

The full index `rag/index/wikipedia_full.index` is 36,808,659,501 bytes;
23,963,971 x 384 x 4 = 36,808,659,456, again a 45-byte header difference. It
was built 2026-07-08/09 and HAS NEVER BEEN USED for any answer-accuracy
experiment. The Phase R dual-view branch was rejected on 2026-07-11 at the
plan-generation stage, explicitly "before loading the 35 GB Wikipedia FAISS
index", so it never tested retrieval either.

Consequently the recorded failure mode "Coverage Gap — Grey seal absent" is a
property of a 2,152-page index, not of Wikipedia or of retrieval as a method.

### B. What D2 asks

Given that D1 established knowledge is worth +20.5pp (arm B 78.00% vs arm A
57.50%, with wrong facts at 57.00% and no leakage), and that structure was
worth approximately zero across the whole EFBPT programme, the open question
is whether RETRIEVAL can deliver that knowledge.

D2 asks two things at once:
1. Does plain retrieval on the FULL index beat a fair no-facts baseline?
2. When it fails, is the failure in RETRIEVAL (wrong passages fetched) or in
   USE (right passages fetched, model still wrong)?

Question 2 is the one the earlier experiment could not answer, because it
never measured retrieval quality against gold evidence.

### C. Arms

Same 200 DEV200 rows. Base Qwen3-14B, no adapters.

| arm | facts supplied |
|---|---|
| R1 | top-3 chunks retrieved from the FULL index |
| R2 | top-10 chunks retrieved from the FULL index |

Reference lines are REUSED from D1, not re-run, because the prompt, model,
quantization and decoding are identical:
- no-facts floor for retrieval: D1 arm A = 57.50%
- gold-facts ceiling for retrieval: D1 arm B = 78.00%
- majority-class floor: 56.50%

Retrieval query is `question_en`, one query per question, matching the earlier
setup so the only changed variable is the index. Embedding model unchanged:
paraphrase-multilingual-MiniLM-L12-v2. Retrieved chunks are formatted into the
same facts block used by D1 arms B/C.

### D. The new instrument: gold-evidence recall

DEV200 rows carry `evidence_paragraph_ids` such as "LendingTree-1", "Retail-6".
The part before the final hyphen is the Wikipedia article title. Define, per
question, the set of REQUIRED TITLES as the distinct titles appearing in
`evidence_paragraph_ids`.

Reported per arm:
- title recall@k = fraction of required titles present among retrieved chunks
- fully-covered rate = share of questions where ALL required titles retrieved
- accuracy on the fully-covered subset vs the not-fully-covered subset

This separates retrieval failure from use failure, which the earlier
experiment could not do.

### E. Decoding, scoring, validity

Identical to D1: 4-bit nf4, thinking OFF, temperature 0, max_new_tokens 1024,
AMENDMENT 5 extractor imported unchanged, dual scoring with the Devanagari
rule per D1 AMENDMENT 2, unparsed scored incorrect, denominator 200.
AMENDMENT 5D applies: an unparsed spread above 5pp across R1/R2 and the reused
D1 arms A/B voids the accuracy comparison. Any subgroup claim requires a
matched control on the same qid set.

### F. Pre-declared readings, fixed before any number is seen

1. If R1 or R2 reaches within 5pp of arm B (78.00%): retrieval on the full
   index substantially delivers the knowledge. Coverage was the whole story
   and the method is retrieval engineering, not entity work.
2. If R1/R2 land near arm A (57.50%) AND title recall is LOW: the bottleneck
   is retrieval quality. Entity-grounded retrieval is then justified by
   evidence — retrieve per entity rather than once per question, and ground
   the entity to a canonical title rather than generating it freely. The two
   failure modes recorded in the old error analysis (entity disambiguation,
   "The Police" -> law enforcement; and partial retrieval, three Genghis Khan
   chunks and zero Julius Caesar) are both single-query artifacts and both
   are structurally addressed by per-entity retrieval.
3. If R1/R2 land near arm A AND title recall is HIGH: retrieval works but the
   model cannot use retrieved passages, even though D1 arm B proves it CAN
   use clean facts. The bottleneck is then passage form — length, noise,
   or position — not retrieval or knowledge. Entity grounding would NOT help
   and must not be pursued on this evidence.
4. If accuracy on the fully-covered subset approaches arm B while the
   not-covered subset sits at arm A, recall is the entire story and maximising
   recall is the method.
5. R2 - R1 measures whether more context helps or dilutes. Report regardless.

### G. Recorded limitations

- Retrieval uses the English question. The index is English Wikipedia. This
  is realistic for the corpus but means D2 does not test Urdu-side retrieval.
- `evidence_paragraph_ids` come from official StrategyQA and are a
  sufficient, not exhaustive, evidence set; recall against them is a lower
  bound on usable retrieval.
- The old 56.24% figure is NOT a D2 arm and is not comparable to D2 numbers
  (different split size, thinking ON, different index). It is quoted in
  Section A only to explain why D2 is necessary.

### H. Status
Declared before execution. No D2 arm has been run.

## DIAGNOSTIC D3 — Oracle retrieval (frozen 2026-08-01)

Declared before execution. No D3 number exists.

### A. The untested wall

D1 established that CLEAN one-line gold facts are worth +20.5pp
(arm A 57.50% -> arm B 78.00%, wrong facts 57.00%, no leakage).
D2 established that retrieval delivers almost none of that knowledge:
title recall 16.19% raw at k=10, 23.15% among pages that exist in the corpus,
and only 8/200 questions fully covered.

Neither tested whether the model can use RAW WIKIPEDIA PARAGRAPHS at all.
Gold facts are one clean sentence each. Wikipedia chunks are 200 words of
mostly irrelevant prose. If the model cannot use paragraphs even when they are
the CORRECT paragraphs, then no amount of retrieval or corpus work can help,
and that must be known before spending a day rebuilding a corpus.

### B. Design

Restricted to the 71/200 DEV200 questions where EVERY required gold evidence
title exists in the corpus (from `d2_title_coverage.json`). On those questions
retrieval is bypassed entirely: pages are fetched by EXACT TITLE match from
`rag/index/wikipedia_full_meta.jsonl`. This simulates perfect entity linking
and perfect retrieval — the ceiling of any entity-grounded method.

| arm | facts supplied | source |
|---|---|---|
| O1 | 1 chunk per gold title (lead section) | new run |
| O2 | 3 chunks per gold title | new run |
| A  | none | REUSED from D1 arm A, same 71 qids |
| B  | clean gold facts | REUSED from D1 arm B, same 71 qids |

Arms A and B are NOT re-run. Their saved generations are re-scored on the 71
qid subset. Prompt, model, quantization and decoding are identical, so the
comparison is matched by construction.

Chunk order note: `build_chunks.py` writes an article's chunks consecutively
in order, so chunk 1 is the article lead. O1 takes the lead only; O2 takes the
first three. O2 - O1 measures whether more context helps or adds noise.

### C. Decoding, scoring, validity

Identical to D1: 4-bit nf4, thinking OFF, temperature 0, max_new_tokens 1024,
AMENDMENT 5 extractor imported unchanged, dual scoring with the Devanagari
rule per D1 AMENDMENT 2, unparsed scored incorrect.
Denominator is the 71-question subset for every arm including the reused ones.
AMENDMENT 5D applies: unparsed spread above 5pp across the four arms voids the
accuracy comparison.

The facts block format and prompt assembly are IMPORTED from `d1_eval_arms.py`,
not re-implemented, so O arms are byte-identical in structure to D1 arms B/C.

### D. Pre-declared readings, fixed before any number is seen

1. If O1 or O2 lands within 5pp of arm B: correct Wikipedia paragraphs work
   nearly as well as clean gold facts. The bottleneck is then entity linking
   and corpus coverage, both of which are addressable. Retrieval is a viable
   method and the pathway is confirmed.
2. If O1 and O2 land within 5pp of arm A: the model cannot use raw paragraphs
   even when they are the correct ones. Fetching PAGES is a dead end
   regardless of coverage or linking quality. The method must instead EXTRACT
   facts from pages rather than pass pages to the model.
3. If O1/O2 sit between A and B: paragraphs partially work. Report the exact
   fraction of the gold-facts gain recovered, and treat fact extraction as an
   improvement to a working pipeline rather than a replacement for it.
4. O2 - O1 is reported regardless. A negative value means additional correct
   context HURTS, which is itself a finding about context noise.

### E. Recorded limitations

- 71 questions is a small sample. A 5pp difference is roughly one standard
  error; only large effects are interpretable. This is a direction-finding
  diagnostic, not a headline result.
- The 71 subset is not random: it is the subset whose gold pages survived in
  the corpus, which may correlate with entity popularity. Arms A and B are
  reused on the SAME subset, so the comparison is internally matched, but the
  subset's absolute accuracy must not be compared to full-DEV200 numbers.
- Chunks are 200-word windows and are not aligned to the original StrategyQA
  evidence paragraph ids, so O arms supply the right ARTICLE, not necessarily
  the right paragraph. This makes D3 an upper bound on article-level retrieval
  and a lower bound on paragraph-perfect retrieval.

### F. Status
Declared before execution. No D3 arm has been run.

## DIAGNOSTIC D4 — Fact extraction from correct articles (frozen 2026-08-01)

Declared before execution. No D4 number exists.

### A. Position established by D1-D3, with statistics

| finding | evidence |
|---|---|
| clean English facts work | 83.10% vs 59.15% on the 71 subset, McNemar p=0.0023 |
| correct PAGES do not measurably work | O1 +9.86pp, p=0.2295 — not significant |
| more correct context hurts slightly | O2 - O1 = -1.40pp |
| embedding retrieval fails | 23.15% recall among pages that exist |
| corpus is broken | 70.47% title coverage; central articles (Cat, Porsche, Douglas Adams) absent in BOTH title and text — "Felis catus" appears under 160 titles, none of them Cat; "Douglas Noel Adams" appears 0 times in 24M chunks |
| model cannot generate its own facts | genread probe (2026-07-08, unlogged): 70.0% = no-facts 70.0%, zero gain |

The correction to the D3 read-out is recorded here: the earlier phrase "41% of
the gold-facts gain recovered" overstated the oracle result. The paired test
shows the O1 gain is not statistically distinguishable from zero at n=71.

### B. The untested step

D3 handed the model whole articles (up to ~4,200 tokens) and it failed to find
the needed sentence inside them. Arm B handed it single clean sentences and it
succeeded. The step in between has never been tested: a SEPARATE extraction
pass that reads the correct article and writes short atomic English facts,
which are then fed to the answering pass in exactly the arm-B format.

Reading (extraction, English->English) is a different capability from knowing
(generation, which genread proved fails) and from finding (retrieval, which
D2 proved fails). Whether Qwen3-14B can do targeted extraction is unknown.

### C. Design

Same 71-question subset as D3 (all gold titles present in corpus), so D3 arms
A, O1, O2, B are reused as matched references on identical qids.

Two-pass pipeline, both passes base Qwen3-14B, no adapters:

PASS 1 — EXTRACTION (English -> English, per question):
  input: the English question (question_en) + the same chunks used by D3 O2
         (up to 3 per gold title, fetched by exact title)
  instruction: extract the facts from these passages that are needed to answer
         the question; output ONLY short English factual sentences, one per
         line; if a needed fact is not in the passages, do not invent it
  decoding: greedy, temperature 0, max_new_tokens 512

PASS 2 — ANSWERING (identical to D1 arm B in every respect):
  input: fixed system message + frozen Urdu instruction + facts block built
         from PASS 1's lines + question_ur
  decoding and scoring: identical to D1/D3 (AMENDMENT 5 extractor primary,
  Devanagari secondary, unparsed = incorrect, denominator 71)

Arms:
| arm | facts supplied to pass 2 |
|---|---|
| X1 | facts extracted by pass 1 |
| A, O1, O2, B | reused from D3 outputs, same 71 qids |

The extraction uses question_en, not question_ur, so pass 1 is entirely
English-side. This is deliberate: it isolates the extraction capability from
the Urdu-generation weakness documented in D1 AMENDMENT 1.

### D. Pre-declared readings

Let gap = B - A = 23.95pp on this subset.
1. X1 recovers >= 60% of gap AND X1 vs A is significant (McNemar p < 0.05):
   extraction works. The five-step method is validated end-to-end at the
   oracle-linking level. Remaining research: entity linking (step 2) and a
   corpus replacement. This is the "method found" outcome.
2. X1 vs A not significant: extraction from articles does not work either.
   Since facts work but neither pages, self-generation, nor extraction can
   produce them, the fact SOURCE itself must change (e.g. a structured
   knowledge base instead of Wikipedia prose). Page-derived pipelines close.
3. In between: report the recovered fraction with its p-value; decide
   continuation with the supervisor, not unilaterally.
4. Report alongside: extraction failure modes on 10 hand-checked rows —
   did pass 1 invent facts not present in the passages (the genread disease
   re-entering through the back door)? Any invented fact found is quoted in
   the write-up regardless of the accuracy outcome.

### E. Limitations recorded now

- n=71; only large effects are detectable (a 60% recovery of the gap is ~14pp
  and would reach significance at roughly b>=2c in discordant pairs).
- Two passes double the inference cost per question. Acceptable for a
  diagnostic; an efficiency claim is out of scope.
- The subset over-represents surviving (popular) entities, as recorded in D3.
- Extraction quality is bounded by chunk relevance: if the needed sentence is
  not in the first 3 chunks of any gold article, extraction cannot find it.
  This bound is shared with O2, so X1 vs O2 is a clean comparison of FORM
  (extracted facts vs raw pages) at equal information access.

### F. Status
Declared before execution. No D4 arm has been run.

## DIAGNOSTIC D5 — declared 2026-08-14, BEFORE any measurement

### Motivation (context, not evidence)
The D4 item-4 verification (job 62521) showed pass 1 concentrating its
6-fact budget on the first title; in the 10 inspected rows, 5 had at
least one supplied title receiving zero facts, and the single wrong
answer was the most extreme case (6-0). This is an anecdote from a
non-random sample (first 10 rows, already seen). D5 exists to test it
properly. DISCLOSURE: because those 10 rows were seen before this
declaration, the Step-0 gate below is intentionally set well under the
50% observed in them.

### STEP 0 — existence gate (CPU only, no model, no new data)
Measure on outputs/efbpt/d4/d4_extractions.jsonl, all 71 rows:
for each row, attribute each fact to the title(s) whose normalized
name or clear head-noun appears in the fact; count per row the number
of supplied titles with ZERO attributed facts. Attribution script must
be committed before running. Denominator: rows with >= 2 titles.
- GATE: if >= 30% of multi-title rows have at least one zero-fact
  title, D5 RUNS. If < 30%, D5 is CANCELLED, the result is logged, and
  work moves directly to entity linking (step 2). No re-litigating.

### ARMS (only if gate passes)
- X1: (already run, job 59085) 6 facts, free allocation. Control.
- X2: BALANCED extraction — one pass-1 call PER TITLE, exactly 2 facts
  per title, all else identical to X1 pass 1.
- X3: free allocation as X1, but total fact budget set equal to X2's
  realized total for that row (fact-count-matched control).
Pass 2 identical to D4 for all arms (frozen Urdu prompt, arm-B format).
Decoding identical to D4: thinking ON, temperature 0, greedy; pass 1
max 2048 tokens, pass 2 max 1024; dual primary/Devanagari scoring;
unparsed = incorrect; MD5-checked instructions.

### PRE-DECLARED READINGS
Primary: paired McNemar X2 vs X1 on the 71 qids, exact two-sided.
- Reading A: p < 0.05 and X2 > X1 -> balance helps; adopt X2 form.
- Reading B: p < 0.05 and X2 < X1 -> balance hurts; keep X1 form.
- Reading C: p >= 0.05 -> no detectable effect at n=71. This is the
  EXPECTED outcome given <= 7.04pp headroom (B - X1 = 83.10 - 76.06)
  and will be reported as a finding, not suppressed. X1 form is kept
  and work moves to entity linking.
Secondary (mechanism, reported alongside whichever reading fires):
- X3 vs X1 (does fact COUNT alone matter?)
- X2 vs X3 (does BALANCE, at matched count, matter?)
- Coverage (fraction of titles with >= 1 fact) per arm. For X2 this is
  a MANIPULATION CHECK ONLY (forced by construction), never a result.
Validity: unparsed spread across X1/X2/X3 must be < 5pp, else void.
No amendments to this section after any D5 number exists.
## DIAGNOSTIC D6 — END-TO-END with SELF-GENERATED entities
Declared 2026-08-14, BEFORE any D6 measurement exists.
No amendment to this section is permitted once any D6 number exists.

### A. WHY THIS IS THE DECIDING EXPERIMENT
Every result so far used GOLD article titles. D4 showed that with gold titles,
extracted facts are worth +16.91pp (76.06% vs 59.15%, p=0.0290). D5 showed the
extraction FORM is not the lever. The one untested question is whether the
pipeline works when nothing is given: the model must produce its own entities,
find its own articles, extract its own facts, and answer.

The ceiling analysis of 2026-08-14 (read-only, DEV200) found that 52.5% of gold
evidence title-instances have no lexical trace in question_en, and only 19.5%
of DEV200 rows (23.9% of the 71-subset) have ALL gold titles lexically present.
StrategyQA evidence titles are JUSTIFICATION pages, not question entities
(e.g. "Is SnapCap an example of a retail store?" requires "LendingTree").

DECLARED CONSEQUENCE OF THAT FINDING: recovering GOLD titles is NOT the target
of D6 and will not be used as a success criterion. A working system needs
titles that yield USEFUL facts, not titles that match the gold set. D6 is
therefore scored on ANSWER ACCURACY only. Gold-title overlap is reported as a
descriptive statistic and is explicitly NOT a gate.

Caveat recorded now: the ceiling classifier was purely lexical and produces
false ABSENTs ("Nepalese" not reduced to "Nepal"; "moustaches" not linked to
"Facial hair"). 52.5% is therefore an UPPER bound on unrecoverability. This
does not change D6's design, which never depends on that number.

### B. ARMS (n=71, the same qids as D3/D4/D5)
Controls already measured on these exact qids, reused, not re-run:
  A   no facts at all ................................ 59.15%  (floor)
  X1  GOLD titles, oracle linking, extracted facts ... 76.06%  (ceiling)

New arm:
  E1  SELF-LINKED end-to-end. Four stages, no gold information at any stage:
      stage 0  base Qwen3-14B reads question_ur and emits entities as
               {urdu_span, canonical_title} pairs. Base model, no adapters,
               so E1 has no dependency on the closed Plan A' branch. The
               existing dev200_C3_seed*.jsonl files are NOT used, because they
               come from fine-tuned adapters and would make E1 inconsistent
               with D4's base-model setting.
      stage 1  each generated canonical_title is normalized with norm() and
               looked up in the corpus by fetch_chunks. EXACT normalized match
               only; no fuzzy matching, no alias table, no gold fallback. A
               title that does not match yields nothing.
      stage 2  up to CHUNKS_PER_TITLE = 3 chunks per matched title, identical
               to D4 arm X1 and D3 arm O2.
      stage 3  fact extraction with D4's pass-1 prompt at the D4 budget
               (at most 6 facts), then answering with the frozen Urdu prompt.
      Rows where no generated title matches the corpus receive NO facts and
      are answered anyway. They are scored, never dropped.

Everything except stage 0 and stage 1 is IMPORTED from d4_extract_facts.py, so
decoding, chunk count, extraction prompt, facts-block format and scorers are
identical to X1 by construction. The stage-0 prompt is MD5-printed in the log.

### C. DECODING AND SCORING (identical to D4/D5)
Temperature 0, greedy, single seed. Chat-template settings imported from D4,
not restated here, so they cannot drift. Dual primary/Devanagari scoring.
Unparsed counts as INCORRECT; the denominator is always 71.

### D. VALIDITY CONDITION
Unparsed spread across A, X1 and E1 must be < 5pp. If it is >= 5pp the
accuracy comparison is VOID and only the descriptive statistics may be quoted.

### E. POWER, DECLARED BEFORE THE RUN
Computed 2026-08-14 from D4's discordant counts (b=19, c=7), assuming
correctly-linked rows behave like X1 and unlinked rows like A:

  effective linking   E1 acc   McNemar p vs A    significant?
     55%               68.5%      0.180              no
     70%               71.0%      0.096              no
     80%               72.7%      0.078              no
     90%               74.4%      0.035              YES

At n=71, significance requires roughly 90% effective linking. The subset
cannot grow: 71 is every DEV200 question whose gold titles all exist in the
defective corpus. READING 2 BELOW IS THEREFORE THE EXPECTED OUTCOME. A
non-significant positive result is a real finding here, not a failure, and
will be reported with this power table beside it.

### F. PRE-DECLARED READINGS — apply as written, invent nothing after
Primary statistic: paired exact McNemar, E1 vs A, on all 71 qids.
Let d = E1 - A in pp. Half the oracle gap is +8.46pp.

  READING 1  — WORKING METHOD.
               d > 0 and p < 0.05.
               An end-to-end Urdu multi-hop method exists and is
               statistically supported. This is the thesis contribution.

  READING 2  — WORKING BUT UNDERPOWERED.
               d >= +8.46pp and p >= 0.05.
               Self-linking recovers at least half the oracle gap. Report as
               a working method with an honest power limitation, beside the
               Section E table. Do NOT claim significance. Do NOT re-run on
               a different subset to chase a p-value.

  READING 3  — PARTIAL, INSUFFICIENT.
               0 < d < +8.46pp.
               Self-linking helps but recovers less than half the gap.
               Report as a partial result. The remaining bottleneck is named
               from the Section G diagnostics, not guessed.

  READING 4  — FAIL.
               d <= 0.
               Self-generated titles do not deliver usable knowledge. The
               retrieval-free pipeline is closed. The thesis contribution
               becomes the diagnostic chain plus the corpus and benchmark
               findings, which stand on their own.

Secondary, reported alongside whichever reading fires:
  E1 vs X1 paired McNemar — how far below the oracle ceiling E1 sits.

### G. MANDATORY DIAGNOSTICS, reported with every reading
None of these is a gate. All are descriptive and must appear in the log:
  1. entities generated per question: min / median / max.
  2. generated titles that matched the corpus: count and percent.
  3. rows receiving zero facts because nothing matched: count and percent,
     plus that subgroup's accuracy against the SAME rows in arm A (a matched
     control on the same qids, never a comparison against a global floor).
  4. overlap between generated titles and gold required_titles: descriptive
     only, explicitly NOT a success criterion (Section A).
  5. facts per question: min / median / max, and the count of empty rows.
  6. a 10-row dump, first 10 rows in file order, showing question_ur,
     generated {urdu_span, canonical_title} pairs, which titles matched,
     the extracted facts, gold and predicted answers.

### H. WHAT IS FORBIDDEN IN D6
- No gold titles, gold facts, gold spans or evidence ids at any stage of E1.
- No fuzzy or embedding-based title matching. That is a separate declared
  experiment if D6's diagnostics justify it; folding it in now would make the
  arm untestable.
- No dropping of unmatched or unparsed rows from the denominator.
- No re-running D6 on a different subset after seeing the result.
- No amendment to this section once any D6 number exists.

### I. STATUS
Declared before execution. No D6 arm has been run.

## EXPERIMENT G1 — SDFR-UR on GSM8K with a DECONTAMINATED retrieval pool
Declared 2026-08-15, BEFORE any G1 measurement exists.
No amendment to this section is permitted once any G1 number exists.

### A. WHY THIS EXPERIMENT EXISTS — the contamination audit
A read-only audit of the fair-regime SDFR outputs (2026-08-15) established
that the previously reported GSM8K result is CONTAMINATED and must not be
claimed. Facts from that audit, all reproducible from the scripts on disk:

- The evaluation set is 700 Urdu translations of items drawn from the GSM8K
  TRAIN split (`data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl`).
- The demonstration pool is the GSM8K TRAIN split, 7,473 records
  (`data/retrieval_pools/gsm8k_train_en.jsonl` -> `data/sdfr_splits/gsm8k_pool.jsonl`).
- Every one of the 700 evaluation items has its English source record present
  in the pool as an exact (question, answer) match: **700/700 = 100% overlap**.
- Neither SDFR script excludes the query item from its own retrieved
  neighbours. `retrieve()` performs a bare FAISS top-k with no id, position,
  question or answer comparison. Quoted in the audit; NO SUCH EXCLUSION EXISTS.
- Reconstruction with the same pool, index and local embedding model shows the
  evaluation item's own English source was retrieved in the top 3 for
  **632/700 = 90.29%** of items (rank 1: 580, rank 2: 36, rank 3: 16), and the
  gold answer string appears in the reconstructed demonstration block for
  **647/700 = 92.43%**.
- Model generations confirm the mechanism explicitly, e.g. GSM8K_0004:
  "The previous example had the same numbers and the answer was 14, so this
  must be correct."

The reported effect (621/700 = 88.71% baseline vs 678/700 = 96.86% SDFR,
b=63, c=6, p=4.47e-13) is therefore **answer leakage, not a method effect**.
The number is retracted here and must not appear as a result anywhere.

Note on a separate prior error: the figure "SDFR 99.38% on a clean GSM8K
subset" recorded earlier does not match the file on disk, which shows 96.86%
over all 700. The on-disk value governs; the 99.38% figure is withdrawn.

Note on scoring fairness: the audit confirmed the baseline and SDFR arms use
byte-identical answer extractors (differing only by one source comment) and
identical decoding (temperature 0.0, max_tokens 2048, enable_thinking=True,
same model and vLLM settings). Pairing is intact: 700/700 shared qids, no
duplicates, no question or gold mismatches. The comparison machinery was
sound; the retrieval pool was not.

### B. THE QUESTION G1 ASKS
Does similarity-based demonstration retrieval help Urdu GSM8K reasoning when
the model can no longer be shown the answer to the question it is being asked?

This is the only formulation under which SDFR-UR can be claimed as a method.

### C. ARMS (n=700, identical eval items to the contaminated run)
- **B0** baseline CoT, fair regime. REUSED from
  `outputs/sdfr/cot_gsm8k_baseline_fair_qwen3_14b.jsonl`, 621/700 = 88.71%.
  Not re-run: the audit confirmed its decoding and extractor already match.
- **S1** SDFR with a DECONTAMINATED pool. Identical to
  `sdfr_gsm8k_fair.py` in every respect except pool construction.

Decontamination procedure, declared here and to be implemented exactly:
1. For each of the 700 evaluation items, locate its English source record by
   exact (question, answer) match in the 7,473-record pool. The audit proved
   all 700 match uniquely.
2. Remove those 700 records from the pool. Expected pool size after removal:
   **6,773**. The script MUST assert this count and die otherwise.
3. Additionally apply a runtime near-duplicate guard: for each query, drop any
   retrieved demonstration whose normalized question has
   `difflib.SequenceMatcher` ratio >= 0.90 against the evaluation item's
   English source question, and backfill from the next-ranked neighbour so
   every item still receives exactly TOP_K demonstrations. The count of drops
   must be logged.
4. The script MUST report, before generation: pool size, records removed, and
   the number of evaluation items whose own source is still reachable in the
   pool. That last number MUST be 0 or the run aborts.

Everything else is held fixed: same embedding model, same FAISS construction
procedure, same TOP_K, same prompt template, same extractor, same decoding
(temperature 0.0, max_tokens 2048, enable_thinking=True), same 700 eval items
in the same order.

### D. VALIDITY CONDITIONS
- Pairing: S1 must produce exactly 700 rows with the same qid set as B0. Any
  mismatch voids the run.
- The unparseable-prediction rate must be reported for both arms. B0 is
  16/700 = 2.29%. If S1's rate differs from B0's by more than 5pp, the
  accuracy comparison is VOID and only descriptive statistics may be quoted.
- Unparseable predictions count as INCORRECT. The denominator is always 700.
- If the post-removal pool size is not exactly 6,773, or if any evaluation
  item's own source remains reachable, the run is INVALID and no reading
  fires.

### E. POWER, COMPUTED BEFORE THE RUN
At n=700 with a paired exact McNemar test and B0 at 621/700 = 88.71%, the
detectable effect depends on how the discordant pairs split. Computed
2026-08-15, before the run:

| net wins (S1-B0) | as pp | p if c=b/4 | p if c=b/2 | p if c=2b/3 |
|---|---|---|---|---|
| 9  | +1.29pp | 0.035 | 0.122 | 0.233 |
| 14 | +2.00pp | 0.007 | 0.044 | 0.120 |
| 21 | +3.00pp | 0.001 | 0.011 | 0.050 |

So a gain of roughly **+1.3pp reaches significance under a clean split, and
+3pp reaches it even under a noisy one**. This is far better powered than any
experiment on the 71-question StrategyQA subset, where only very large
effects were ever detectable. G1 is therefore a fair test of the method: a
real effect of modest size CAN be found here, so a null result is
informative rather than merely underpowered.

### F. PRE-DECLARED READINGS — apply as written, invent nothing afterwards
Primary statistic: paired exact McNemar, S1 vs B0, over all 700 qids,
`scipy.stats.binomtest(min(b, c), b + c, 0.5)`. Let d = S1 - B0 in pp.

- **READING 1 — METHOD CONFIRMED.** d > 0 and p < 0.05.
  Similarity-based demonstration retrieval genuinely improves Urdu
  arithmetic reasoning. SDFR-UR may be claimed as a working method, scoped to
  arithmetic reasoning, and MUST be reported together with the contamination
  history in Section A.
- **READING 2 — NO EFFECT.** p >= 0.05, regardless of the sign of d.
  The original GSM8K gain was leakage. SDFR-UR is NOT claimed as a method on
  GSM8K. This is a substantive finding at n=700, not an underpowered null,
  and is reported as such.
- **READING 3 — METHOD HARMS.** d < 0 and p < 0.05.
  Retrieved demonstrations actively hurt once the answer is removed. Reported
  as a finding; SDFR-UR is not claimed.

### G. MANDATORY DIAGNOSTICS, reported with every reading (never gates)
1. Pool size before and after removal; number of records removed.
2. Runtime near-duplicate drops: total count, and the number of evaluation
   items affected.
3. Maximum and mean SequenceMatcher similarity between each evaluation item's
   English source question and its retrieved demonstrations, after filtering.
4. Count and percentage of evaluation items whose gold answer string still
   appears anywhere in the final prompt, WITH the reason (a different problem
   may legitimately share a numeric answer). This is descriptive; it is NOT a
   gate, because two unrelated problems can share the answer "12".
5. Unparseable-prediction counts for both arms.
6. The first 10 S1 items in file order: question, the TOP_K retrieved
   demonstration questions, gold, prediction, and the last 200 characters of
   the generation.

### H. WHAT IS FORBIDDEN
- No change to the evaluation set, the prompt template, the extractor, the
  decoding settings, or TOP_K. Only the pool changes.
- No re-running on a different evaluation subset after seeing the result.
- No claiming the contaminated 96.86% figure under any framing.
- No amendment to this section once any G1 number exists.

### I. SEPARATE, PRE-EXISTING DATA BUG (not part of G1)
The audit found 5 positions where the English and Urdu PIQA records disagree
on the label (indices 524, 631, 637, 640, 704), four of them inside the
150-item PIQA evaluation range. This is a translation-alignment defect in
`data/piqa_raw/`, independent of SDFR. It must be recorded in
`experiments.md` and corrected before any PIQA number is quoted. It does not
affect G1.

### J. PIQA STATUS, recorded here for completeness
The PIQA pool/eval split IS clean: pool = English positions 0-599, eval =
Urdu positions 600-749, exact overlap 0, no retrieved demonstration reaching
0.90 similarity to its query. But the paired test is b=27, c=19,
**p=0.302** — NOT significant. The apparent +5.33pp is not statistically
supported, and the mechanism is label-bias correction: the baseline predicts
label 1 on 109/150 items against a gold distribution of 73/77, and SDFR
shifts to 89/150, gaining on gold-0 items (36/73 -> 50/73) while losing on
gold-1 items (72/77 -> 66/77). PIQA is therefore NOT claimed as a method win.

### K. STATUS
Declared before execution. No G1 arm has been run.

## EXPERIMENT L0 — PRE-TEST for constrained entity linking
Declared 2026-08-18, BEFORE any L0 measurement exists.
No amendment to this section is permitted once any L0 number exists.
L0 measures only. It decides whether a constrained-linking method (L1) is
worth building. L0 is NOT itself a method experiment.

### A. WHY — the intervention the diagnostics prescribe
D6 established that steps 1, 3, 4 and 5 of the pipeline work and step 2
(Urdu span -> canonical English title) is the sole failure point. The failure
mode is free generation into an open vocabulary: والکری -> "Walcott",
ایبیسل میدان -> "Aylesbury Vale", بکریاں -> "Sheep". Extraction stayed correct
under bad input, so the damage is entirely upstream.

The intervention the symptom prescribes is to convert GENERATION into
SELECTION: retrieve candidate titles for the Urdu span from a closed set, and
have the model choose. A title absent from the candidate set cannot be
produced, so transliteration drift becomes structurally impossible rather
than statistically discouraged.

The ceiling of that method is the retriever's recall@k. L0 measures that
ceiling, plus an honest baseline, before any method is built.

### B. THREE FLAWS FOUND WHILE STRESS-TESTING THIS PROPOSAL
Recorded because each one changed the design.

1. **recall@k is NOT final accuracy.** The model must still pick the correct
   candidate from k. If recall@10 = 75% and pick-accuracy is 85%, delivered
   accuracy is 64%, not 75%. An earlier draft of this experiment used a flat
   "recall >= 75%" gate; that gate was wrong and is replaced by the
   baseline-relative gate in Section F.
2. **The existing ~55% linking baseline is unverified.** It appears in
   `experiments.md` (54.2 / 56.3 / 54.7% across seeds, 44% seed-stable) with
   NO producing script; a repository-wide search found none. It must not be
   used as a gate. L0 re-measures the baseline on the exact evaluation pairs.
3. **n=289 is contaminated by unlinkable and trivial pairs.** The gold files
   contain coreference spans ('اس کے دادا' -> Genghis Khan, "his grandfather")
   which no linker can resolve from the span, and clean transliterations
   ('میامی' -> Miami) which inflate any score. A census must classify every
   pair BEFORE the gate is applied.

### C. DATA
Gold Urdu-span -> English-title pairs, human-verified, from:
- `data/strategyqa_official/efbpt/plan_a_gold_100.jsonl` (100 rows, field
  `entities`, id field `qid`)
- `data/strategyqa_official/efbpt/blind30_gold.jsonl` (30 rows, field
  `question_entities`, id field `urbench_qid`)
Combined: 130 rows, **289 pairs**, 278 distinct canonical titles, 0 shared
qids, 2 shared exact pairs (kept, deduplicated by (qid, span)).
Neither file has a testability field; the census in Part A creates one.

Candidate title universe: the set of unique titles present in
`rag/index/wikipedia_full_meta.jsonl`, obtained by a single sequential scan.
Expected size ~6,402,346 (the `corpus_unique_titles` value recorded in
`outputs/efbpt/d2/d2_title_coverage.json`). The script must print the count
it actually observes and must NOT assert equality, because the recorded value
is from a prior scan and a mismatch is information, not an error.
`data/strategyqa_official/efbpt/title_space_cache.txt` MUST NOT be used: it is
recorded as unreliable in experiments.md (6 of 13 known-real titles absent).
The universe must come from the corpus, because a title absent from the corpus
cannot be fetched by step 3 and is therefore useless even if linked correctly.

### D. THE THREE PARTS
**Part A — span census (CPU).** Classify all 289 pairs into exactly one
bucket, by rules declared here:
- `COREF` — the span contains no proper name and functions as a reference to
  an entity named elsewhere (possessive or demonstrative phrases such as
  "his grandfather"). Operational rule: the span shares no token with the
  canonical title after transliteration-insensitive comparison, AND the span
  contains a possessive or demonstrative marker (اس, ان, یہ, وہ, کے, کی).
  Flagged for human confirmation; the script prints every COREF candidate.
- `TRANSLIT` — the span is a phonetic rendering of the title. Operational
  rule: character-level similarity between a Latin transliteration of the
  span and the title is >= 0.70.
- `SEMANTIC` — everything else (common nouns, descriptive phrases,
  translated rather than transliterated names).
`LINKABLE` = TRANSLIT + SEMANTIC. COREF pairs are EXCLUDED from the gate and
from the primary statistic, and their count is reported.
The first 30 classifications are printed for human inspection. If the human
judges the classifier unreliable, L0 stops and the rules are revised BEFORE
any Part B or Part C number is generated.

**Part B — baseline re-measurement (GPU, ~5 min).** Run D6's stage-0 prompt
(md5 4675bc6b29aaca764b72c84da246bb9a) on the 130 questions with the base
model, identical decoding to D6. For each gold pair, the baseline is scored
CORRECT if any generated `canonical_title` matches the gold title under
norm(). This is free generation, the arm the method must beat.

**Part C — retrieval ceiling (GPU, ~20 min).** Scan the corpus for unique
titles, encode them with
`paraphrase-multilingual-MiniLM-L12-v2` (the same encoder used by SDFR and by
`rag/`), build a flat inner-product index over normalized vectors, encode the
289 Urdu spans, and report **recall@1, @5, @10, @20, @50, @100**: the fraction
of pairs whose gold title appears among the top-k retrieved titles.

### E. VALIDITY CONDITIONS
- Every number must be reported three ways: over ALL pairs, over LINKABLE
  pairs, and over SEMANTIC pairs only. A headline computed only over TRANSLIT
  pairs is not a result.
- If the observed unique-title count differs from 6,402,346 by more than 5%,
  the discrepancy must be printed prominently; the run continues but the
  difference is reported.
- If any gold title is absent from the corpus title universe, recall for that
  pair is 0 by construction. The count of such pairs MUST be reported
  separately, because it is a corpus-coverage failure, not a retriever
  failure, and it caps recall independently of the encoder.
- Part B and Part C must be scored over the identical pair set.

### F. PRE-DECLARED GATE — apply as written
Let **B** = Part-B baseline accuracy over LINKABLE pairs.
Let **R10** = Part-C recall@10 over LINKABLE pairs.
Assume a pick-accuracy of 0.85 (an optimistic-but-not-absurd estimate for a
14B model choosing among 10 candidates; it is an assumption, declared here,
not a measurement).

- **GATE PASS — BUILD L1.** `R10 * 0.85 >= B + 6pp`.
  The ceiling clears the baseline by enough that a detectable improvement is
  possible. Proceed to declare and build constrained selection.
- **GATE FAIL — DO NOT BUILD.** `R10 * 0.85 < B`.
  The ceiling is at or below the baseline. Constrained linking cannot help.
  Report as a finding: multilingual sentence embeddings do not retrieve
  entity titles from Urdu spans well enough to support constrained linking.
  Do NOT build L1. Do NOT retry with a different encoder inside the remaining
  time budget.
- **MARGINAL — DECIDE JOINTLY.** `B <= R10 * 0.85 < B + 6pp`.
  Report the full recall@k curve. The decision is made with the supervisor
  and the student, in writing, before any further work. No unilateral build.

The 6pp margin is not arbitrary: at n≈180 LINKABLE pairs, +5pp is the
smallest paired-McNemar-detectable improvement under a moderate breakage
ratio (c/b ≈ 0.4). Building toward a target below that margin would repeat
the D5 and D6 error of running an experiment that cannot reach significance.

### G. POWER, COMPUTED BEFORE THE RUN
For a later L1 comparing constrained selection against free generation,
paired exact McNemar:

| n (LINKABLE pairs) | smallest detectable gain |
|---|---|
| 289 | +3pp |
| 215 | +4pp |
| 180 | +5pp |
| 130 | +7pp |
| 65  | +14pp |

Sensitivity to how often constraining BREAKS a previously correct link
(c/b ratio), at n=215: c/b=0.2 -> +4pp; c/b=0.4 -> +5pp; c/b=0.6 -> +8pp;
c/b=0.8 -> +17pp. **A high breakage ratio is the main threat to L1's power.**
If L1 is built, it must therefore be designed as a HYBRID — fall back to free
generation when retrieval returns nothing above a confidence threshold —
rather than as an unconditional replacement. That design choice is declared
here, before any L0 number exists, so it cannot be introduced afterwards to
rescue a result.

### H. WHAT IS FORBIDDEN
- No use of `title_space_cache.txt` as the candidate universe.
- No gate applied to numbers computed over TRANSLIT pairs alone.
- No substitution of a different encoder after seeing a failing recall.
- No use of the unverified ~55% figure anywhere in the gate.
- No building of L1 if the gate fails.
- No amendment to this section once any L0 number exists.

### I. WHAT L0 CANNOT TELL US
Recorded so it is not overclaimed later. L0 measures the retrieval ceiling
and the free-generation baseline. It does NOT measure pick-accuracy, which is
assumed at 0.85 and could be materially lower. It does not establish that
better linking improves ANSWER accuracy — the ceiling analysis of 2026-08-14
showed 52.5% of gold evidence titles have no lexical trace in the question, so
answer-level gains remain capped by the benchmark regardless of linking
quality. L1, if built, is claimed at the LINKING level, not the answer level,
unless a separate answer-level experiment is declared and run.

### J. STATUS
Declared before execution. No L0 part has been run.

## L0 AMENDMENT 1 — COREF rule replaced by a hand-confirmed list
Declared 2026-08-18, BEFORE any Part B or Part C number exists. Part A is a
census, not a measurement; no gate has been applied. Amending is therefore
permitted under freeze discipline. After Parts B/C run, it would not be.

REASON. The declared automatic COREF rule (similarity < 0.50 AND the span
contains a possessive/demonstrative marker) is unsound. In Urdu, کے/کی are
ordinary genitive particles inside descriptive noun phrases, not markers of
reference. String similarity cannot separate a TRANSLATION from a REFERENCE,
because both have near-zero character overlap with the title:
  یورپ کی تاریخ -> History of Europe   is a translation, fully linkable
  اس کے دادا    -> Genghis Khan        is a reference, not linkable
The rule excluded 17 pairs, of which ~15 are ordinary translations. Excluding
them would have discarded linkable pairs and shrunk n for no reason.

REPLACEMENT RULE. COREF is no longer detected automatically. It is an
explicit, hand-confirmed list, reviewed by a native Urdu speaker against the
printed census. A pair is COREF only if the span points at an entity named or
implied elsewhere rather than naming or describing it directly.

The confirmed COREF list is exactly two pairs:
  اس کے دادا        -> Genghis Khan    ("his grandfather")
  نسان کے سی ای او  -> Carlos Ghosn    ("CEO of Nissan", a role, title is a person)

Adjudicated and NOT COREF (recorded so the decision is not revisited):
  امریکہ کے صدر     -> President of the United States  (direct translation)
  ڈایناسورز کا دور  -> Mesozoic  (describes/identifies the era; not a pronoun
                                  or definite reference)
  یورپ کی تاریخ, مغربی شہد کی مکھی, احتیاطی صحت کی دیکھ بھال, 1700 کی دہائی,
  سونے کے وقت, شہد کی موم, پتے کے چھلانگ مارنے والے, نیا سال کی شام,
  بجلی کے بند ہونے, کالج کی ڈگری, ریاستہائے متحدہ کا محکمہ,
  وہ لڑکا جس نے بھیڑیا ہونے کا شور مچایا  -- all translations, all LINKABLE.

RESULTING COUNTS. TRANSLIT 126, SEMANTIC 161, COREF 2, LINKABLE 287 of 289.
Power at n=287 is unchanged from the freeze section G table: about +4pp
detectable. The TRANSLIT threshold of 0.70 is NOT changed.

SUSPECTED BAD GOLD ANNOTATION, logged not corrected:
  شہد کے بچھڑے -> Honey badger. The span reads "honey's calves". Standard Urdu
  renderings are شہد بیجر, ہنی بیجر or ریٹل (verified against Urdu dictionary
  sources 2026-08-18). The pair is retained in LINKABLE so the evaluation set
  is not altered after declaration, but any linker failure on this pair must
  be attributed to the annotation, not to the method.
  