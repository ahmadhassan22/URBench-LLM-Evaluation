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
- Stratify by: hop count, entity count, yes/no label, reasoning pattern.
- Fixed seed. Save the qid list to a file.

**Nesting is mandatory:** 100 ⊂ 250 ⊂ 500.

The 250 sampler must **load the saved 100-qid file** and add 150 new ones.
It must not resample from scratch. Same for 500 loading 250. If this is broken,
the 100→250 trend compares two different populations and means nothing.

Files:

- `data/strategyqa_official/efbpt/plan_a_qids_100.txt`
- `data/strategyqa_official/efbpt/plan_a_qids_250.txt`
- `data/strategyqa_official/efbpt/plan_a_qids_500.txt`

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

(none yet — append below with date and reason, do not edit above)
