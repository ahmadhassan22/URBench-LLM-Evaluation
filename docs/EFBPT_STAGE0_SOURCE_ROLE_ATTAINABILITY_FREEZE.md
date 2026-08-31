# EFBPT Stage-0 Source-Role and Exact-Title-Attainability Freeze

**Date:** 2026-08-22

> **PRE-EXPERIMENT FREEZE**  
> **STAGE 0 ONLY**  
> **NO METHOD RESULT**  
> **NO MODEL RESULT**

## 1. Status and purpose

This document freezes a Stage-0 source-role and exact-title-attainability annotation study. Stage 0 precedes implementation of any new post-L0 method. It does not claim that a new method exists or that any model has succeeded.

Stage 0 asks whether DEV200 contains enough reliably annotated, corpus-covered, sequential bridge structure to justify the smallest controlled bridge-state falsification experiment. Its outputs are descriptive annotations, coverage statistics, reliability measurements, and a GO/NO-GO decision. Stage 0 does not build or evaluate the proposed retrieval method.

Two independent axes are recorded for every distinct required gold source instance:

1. **Source role:** `EXPLICIT`, `LATENT_BRIDGE`, or `AMBIGUOUS`.
2. **Exact-title corpus status:** `EXACT_PRESENT` or `EXACT_ABSENT`.

Neither axis may be inferred from the other.

## 2. Scientific motivation and novelty boundary

### 2.1 Verified empirical boundary

The pre-Stage-0 record is:

- D4: **76.06% versus 59.15%, +16.91 percentage points, paired p=.0290**.
- Under the stricter D4 parse treatment: **75.00% versus 61.76%, paired p=.0931**.
- Both D4 analyses must always be reported together.
- D6 automatic source discovery: **64.79% versus 59.15%, paired p=.4807**.
- L0: **B=45.64%, R@10=11.85%, delivered ceiling=10.07%, threshold=51.64%**.
- L0 decision: **GATE FAIL — DO NOT BUILD L1**.
- **52.52%** of gold title instances lacked the narrow automatic English lexical trace. That rule was not human validated and is not an `EXPLICIT`/`LATENT_BRIDGE` label.
- DEV200 exact-title coverage comprises **636** question-title instances: **445 `EXACT_PRESENT`**, **191 `EXACT_ABSENT`**, **71/200** fully exact-title-coverable rows, **113/200** partially coverable rows, and **16/200** zero-cover rows.

These figures motivate an annotation audit; they do not establish a source-role mechanism.

### 2.2 Post-2025 novelty boundary

Generic evidence-conditioned iterative retrieval, query refinement, reranking, stop/continue policies, bridge-entity completion, compact evidence states, and multilingual multi-turn retrieval have close prior art. The relevant neighborhood includes SMR, Orion, AutoRefine, EfficientRAG, ChainRAG, GRITHopper, LcRL, and CroSearch-R1.

Stage 0 therefore does **not** prepare a claim that iterative retrieval is new. It also does not treat an Urdu setting, by itself, as method novelty.

The remaining hypothesis worth testing is narrower: failure-aware Urdu-to-English source recovery may have useful, reproducible source types, and verified evidence may later improve discovery of sources that are genuinely latent from the Urdu question. Stage 0 tests whether the data can support that later falsification test; it does not test the mechanism itself.

## 3. Authoritative data and dataset status

### 3.1 Dataset

The Stage-0 dataset is:

`data/strategyqa_official/dev200_seed4242.jsonl`

Verified cardinalities are:

- 200 rows;
- 200 unique `urbench_qid` values;
- 636 distinct `(urbench_qid, required gold title)` source instances.

The authoritative exact-title coverage inputs and outputs are:

- `outputs/efbpt/d2/d2_title_coverage.json`;
- `outputs/efbpt/d2/d2_recall_split.json`;
- `eval/error_analysis_tests/efbpt/d2_title_coverage.py`;
- frozen title universe: `rag/index/wikipedia_full_meta.jsonl`.

The authoritative D4 implementation and saved extraction data are:

- `eval/error_analysis_tests/efbpt/d4_extract_facts.py`;
- `outputs/efbpt/d4/d4_extractions.jsonl`.

Official StrategyQA evidence must be read from the official mapped records and their evidence integration, including:

- `data/strategyqa_official/dev.json`;
- `data/strategyqa_official/train.json`;
- `data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl`;
- `data/strategyqa_official/strategyqa_train_paragraphs.json`;
- the repository scripts that map official StrategyQA qids, decomposition steps, evidence groups, paragraph IDs, and titles into URBench records.

### 3.2 Development-data declaration

**DEV200 is spent development data.** It may be used for diagnosis, annotation, and mechanism development. It must not be described as an untouched test, a final sealed evaluation, or a confirmatory final thesis test.

No currently named repository split is honestly untouched. This applies to the named development, audit, blind, Plan-A, manifest, and prior evaluation resources already used or inspected in the project history. Stage-0 results on DEV200 are development findings.

A genuinely fresh confirmation set must later be created from unused StrategyQA rows under a separate predeclared protocol, before its gold evidence is inspected. This document does not select, define, sample, or inspect that future set.

## 4. Annotation data model

Stage 0 produces exactly two analytical tables plus one append-only human annotation log. The logical tables are frozen here; this task does not create them.

### 4.1 Table 1 — Source Instance Master

Unit: one final row per distinct pair:

```text
(urbench_qid, distinct gold title)
```

Expected DEV200 size: **636 rows**.

Required fields:

- `source_instance_id`: stable identifier unique to the `(urbench_qid, distinct gold title)` pair;
- `urbench_qid`;
- `official_qid`;
- `question_ur`;
- `gold_title`;
- `normalized_gold_title`;
- `lexical_trace_en`;
- `exact_corpus_status`;
- `final_source_role`;
- `final_explicit_relation_type`;
- `final_role_confidence`;
- `final_role_rationale`;
- `dependency_status`;
- `parent_source_titles`;
- `official_step_indices`;
- `disagreement_flag`;
- `adjudication_required`;
- `adjudicated_role`;
- `adjudication_rationale`.

`source_instance_id` must be deterministic, and generation must include a collision check. `official_step_indices` and `parent_source_titles` are list-valued because a title may occur in more than one official step or admit more than one possible parent. `final_explicit_relation_type` is null for non-`EXPLICIT` final roles. A final label is entered only after the applicable independent annotation and adjudication process is complete.

The answer must **not** be stored as an annotation input field in this table.

### 4.2 Table 2 — Official Evidence Link Table

This table preserves official evidence structure without flattening it into a universal hop sequence.

Unit: one row per raw official evidence occurrence:

```text
qid
× official evidence annotator
× decomposition step
× evidence group
× paragraph_id
× gold title
```

Required fields include at least:

- `urbench_qid`;
- `official_qid`;
- `official_evidence_annotator_index`;
- `decomposition_step_index`;
- `decomposition_text`;
- `evidence_group_index`;
- `paragraph_id`;
- `gold_title`;
- `paragraph_title`;
- `paragraph_index` and/or `section`, where available.

The implementation must also retain enough raw-path or marker information to distinguish paragraph occurrences from official non-paragraph markers such as operation/no-evidence cells. Nullability must be explicit. Alternative annotators, groups, repeated titles, and repeated paragraph IDs must remain distinguishable.

This table preserves official decomposition order. It does **not** assert that decomposition order is a universal retrieval-source order.

### 4.3 Table 3 — Human Annotation Log

Unit: one append-only row per:

```text
source_instance_id × Stage-0 human annotator
```

Each annotator's record is stored independently with:

- `source_instance_id`;
- `annotator_id`;
- `blind_discovery_candidates`;
- `source_role`;
- `explicit_relation_type`;
- `urdu_span_if_explicit`;
- `confidence`;
- `rationale`;
- `dependency_status`;
- `proposed_parent_source_titles`;
- `annotation_timestamp` and/or `round`.

Pass-specific decisions must remain recoverable rather than being overwritten by later passes. The final Source Instance Master is materialized only after agreement/adjudication. Raw annotator records are never overwritten by the final decision.

## 5. Primary question view

**PRIMARY VIEW = Urdu question only.**

The research task is Urdu-to-English source recovery. English question text may be used later only as a secondary diagnostic. It must not influence the primary human `EXPLICIT`/`LATENT_BRIDGE` decision.

The historical English lexical-trace feature is likewise diagnostic only. It is never shown as role evidence and never initializes, overrides, or resolves a human role label.

## 6. Source-role labels

Source role is independent of exact-title corpus status. The only primary values are:

1. `EXPLICIT`;
2. `LATENT_BRIDGE`;
3. `AMBIGUOUS`.

### 6.1 `EXPLICIT`

Definition:

> The specific gold source identity can reasonably be recovered from the original Urdu question alone through a direct linguistic mapping, without requiring an unstated intermediate fact, retrieved evidence, decomposition answer, or external reasoning step.

Allowed `explicit_relation_type` values are:

- `DIRECT_MENTION`;
- `TRANSLITERATION`;
- `COMMON_ALIAS_OR_ABBREVIATION`;
- `MORPHOLOGICAL_OR_DEMONYM`;
- `DIRECT_SPECIFIC_CONCEPT`.

`DIRECT_SPECIFIC_CONCEPT` is deliberately narrow. It means that the Urdu question directly expresses the same specific concept represented by the page. It does not mean:

- a generally related topic;
- broad semantic similarity;
- a useful justification page;
- a hypernym or superordinate association requiring world knowledge;
- an entity inferred from another fact;
- a page that merely helps answer the question.

There is no generic `SEMANTIC` catch-all. If a relationship is only broadly semantic and cannot be defended under an allowed type, it is not `EXPLICIT`; the annotator proceeds to bridge validation or uses `AMBIGUOUS`.

### 6.2 `LATENT_BRIDGE`

Definition:

> The specific gold source identity is not recoverable from the original Urdu question under the `EXPLICIT` rules, but becomes identifiable after using at least one intermediate fact, subanswer, relation, or prior evidence step.

A `LATENT_BRIDGE` annotation must record:

- the concrete intermediate information that reveals the source;
- where that information is supported in the official decomposition/evidence structure;
- a short human rationale;
- its dependency status and, when defensible, its parent source title or titles.

A page is not `LATENT_BRIDGE` merely because it is relevant, appears in gold evidence, lacks a lexical title match, or occurs at a later decomposition index. A defensible intermediate dependency is required.

### 6.3 `AMBIGUOUS`

Use `AMBIGUOUS` when any of the following applies:

- the source cannot be clearly classified as explicit;
- no clear intermediate dependency establishes latent-bridge status;
- several interpretations are equally plausible;
- evidence annotations imply parallel or alternative source use;
- page granularity or disambiguation makes identity unclear;
- the available official evidence is insufficient for a reliable decision.

Annotators must never force `EXPLICIT` or `LATENT_BRIDGE` merely to avoid ambiguity. Confidence does not replace `AMBIGUOUS`.

### 6.4 Explicit-relation boundaries

The following boundary examples govern annotation:

- An Urdu transliteration of “Martin Luther” pointing to the `Martin Luther` page is potentially `EXPLICIT` through `TRANSLITERATION`.
- A standard demonym is potentially `EXPLICIT` through `MORPHOLOGICAL_OR_DEMONYM` only when it conventionally and directly identifies the relevant country/entity.
- A question about SnapCap that requires `LendingTree` evidence does not make `LendingTree` explicit merely because the company is topically related. It requires a direct mapping or a verified intermediate dependency.
- Broad pages such as `Hand`, `Month`, `Retail`, or `Facial hair` are not `EXPLICIT` unless the specific source concept itself is directly expressed in the Urdu question.
- Parenthetical or disambiguated titles require identity-level justification. A matching base token alone is insufficient if the question does not identify the page sense.
- A title implied only by answering a decomposition step is not explicit from the original question; it is a latent candidate only if the intermediate dependency can be verified.

Annotators judge recoverability of the **specific source identity**, not whether a page is relevant to the answer.

## 7. Exact-title corpus status: a separate axis

Allowed values are:

- `EXACT_PRESENT`;
- `EXACT_ABSENT`.

The authoritative D2 normalization is:

```python
" ".join(str(title).replace("_", " ").strip().lower().split())
```

`EXACT_PRESENT` means the normalized gold title occurs in the title universe of the frozen `rag/index/wikipedia_full_meta.jsonl` corpus.

`EXACT_ABSENT` means the normalized gold title does not occur in that title universe.

For DEV200, the frozen exact-title counts are:

- 636 source instances;
- 445 `EXACT_PRESENT`;
- 191 `EXACT_ABSENT`;
- 71 fully exact-title-coverable questions;
- 113 partially exact-title-coverable questions;
- 16 zero-cover questions.

The status is exact normalized-title membership only. Current resources do not resolve redirects, aliases, Wikidata identity, interlanguage equivalence, or semantic equivalents, and D2 does not supply a page-ID identity resolution layer for absent titles.

Therefore, `EXACT_ABSENT` does **not** mean:

- the knowledge is unavailable;
- no equivalent page exists;
- no redirect exists;
- no alias exists;
- the question is impossible.

Stage 0 must not rename this field `CORPUS_ABSENT`. It must not derive a training `ABSTAIN` target from `EXACT_ABSENT`.

## 8. Annotation visibility procedure

The staged procedure is mandatory. It reduces hindsight bias while still allowing official evidence to validate a true intermediate dependency.

### 8.1 Pass 1 — Blind discovery

Show only:

- the Urdu question.

Hide:

- the answer;
- the English question;
- all gold titles;
- decomposition;
- facts;
- evidence IDs;
- page text;
- corpus status;
- D4 facts;
- other annotations.

The annotator records any English source/entity identities they believe can be hypothesized directly from the Urdu question. This record is `blind_discovery_candidates`.

Pass 1 is not scored as retrieval performance. It is a hindsight-control record used to interpret the later per-title judgment.

Pass 1 must be completed for all 200 DEV200 questions before any Pass-2 gold-title exposure begins. All Pass-1 records are then frozen. After that freeze, Pass-2 source instances must be presented in a deterministic randomized order using seed **20260822**. Where possible, two source instances from the same qid must not be presented consecutively. The future annotation-preparation script must record the ordering procedure and resulting order. Pass-2 ordering must not use source role, corpus status, lexical trace, or any human label.

### 8.2 Pass 2 — Per-title explicitness

For one gold title at a time, show only:

- the Urdu question;
- that one gold English title;
- the annotator's own Pass-1 record.

Continue to hide:

- the answer;
- the English question;
- all other gold titles;
- corpus status;
- decomposition;
- official facts;
- evidence paragraph IDs;
- page text;
- D4 facts;
- other annotators' labels.

The annotator decides:

- `EXPLICIT`; or
- `NOT_YET_EXPLICIT`.

If `EXPLICIT`, record the allowed relation type, Urdu span when applicable, rationale, and confidence. `NOT_YET_EXPLICIT` is a pass-level state, not a final source-role label and not an automatic `LATENT_BRIDGE` decision.

Titles from the same question must be presented independently so that another gold title does not act as a hint.

### 8.3 Pass 3 — Bridge validation

Only for titles not classified `EXPLICIT`, reveal as needed:

- official decomposition;
- official step-linked evidence structure for the target title;
- evidence paragraph identity;
- supporting text only when necessary to verify page sense.

Do not reveal:

- corpus status before the role decision is finalized;
- other human annotators' labels;
- existing D4 facts.

The annotator decides `LATENT_BRIDGE` or `AMBIGUOUS`. A `LATENT_BRIDGE` decision must identify the concrete intermediate dependency and its support in official structure.

The gold answer remains hidden. If a specific adjudication case cannot be resolved without it, the adjudicator may inspect it only as a logged exception. The log must identify the source instance, reason, viewer, time/round, and whether the answer changed the decision.

## 9. Official evidence annotations and dependency structure

All three official StrategyQA evidence annotations must be preserved. They must not be collapsed into one sequence.

Official decomposition order is available, but universal source order is not. A gold title may:

- appear in several decomposition steps;
- be linked to several paragraph IDs;
- occur under alternative official annotators or evidence groups;
- be parallel rather than sequential;
- support multiple steps or share a step with another page.

Consequently, Stage 0 does not assign a numeric retrieval hop to every source and does not choose a single official annotator as the universal truth.

The secondary `dependency_status` field has exactly these values:

- `CLEAR_DEPENDENCY`;
- `MULTIPLE_PLAUSIBLE_PARENTS`;
- `PARALLEL_OR_UNORDERED`;
- `UNRESOLVED`;
- `NOT_APPLICABLE`.

`parent_source_titles` may be filled only when defensible from the official decomposition/evidence structure and the recorded intermediate relation. It may contain multiple titles only with `MULTIPLE_PLAUSIBLE_PARENTS`; it remains empty for unresolved, unordered, or non-applicable cases.

Only `CLEAR_DEPENDENCY` cases are automatically eligible for the later smallest bridge-state falsification experiment. Decomposition index alone cannot establish `CLEAR_DEPENDENCY`.

## 10. Automatic features

The following fields are automatic/non-human labels:

- `urbench_qid`;
- `official_qid`;
- gold title;
- normalized gold title;
- exact corpus status;
- official decomposition-step/evidence links;
- historical English lexical trace.

The historical lexical-trace rule is a narrow automatic diagnostic over the English question and gold title. Its observed absence rate was 52.52% of the 636 instances. It was not human reviewed, so it cannot initialize, override, or adjudicate a source-role label. In particular:

- lexical trace `YES` does not prove `EXPLICIT` under the Urdu-primary definition;
- lexical trace `NO` does not prove `LATENT_BRIDGE`;
- the feature says nothing by itself about intermediate dependency;
- it is not a proxy for title attainability or corpus presence.

Before Stage-0 results are published, the historical lexical-trace logic must be recreated as a tracked, versioned script and its per-instance output saved reproducibly. The implementation must freeze tokenization/normalization, the matching rule, input fields, and output schema. That script and output are not created by this protocol-writing task.

The 160-instance reliability sample may be drawn only after these automatic fields have been generated reproducibly.

## 11. D4 fact safety rule

> **EXISTING D4 FACTS MUST NOT BE USED AS NEXT-SOURCE STATE.**

The original D4 extractor was shown the first three chunks from **all required gold titles jointly** before it produced a row-level fact list. A saved D4 fact may therefore already contain information from the source that a later experiment would call the future or next source.

In addition, existing D4 facts retain no per-fact:

- source title;
- page ID;
- source chunk;
- supporting span;
- provenance;
- extraction-pass provenance tied to a particular page.

The saved D4 rows contain row-level titles and aggregate extraction metadata, but those fields do not establish which page supports each atomic fact. Some generated facts may synthesize information exposed jointly from more than one page. The saved data therefore cannot safely construct a `question + allowed-current-source D4 facts -> next-source retrieval` state without regeneration.

Using these aggregate D4 facts for bridge retrieval would create future-source leakage.

If Stage 0 passes its GO criteria, the later bridge-state experiment must generate **new sequential, source-specific D4-style facts**:

- only from the allowed current/first-hop source;
- before the next source is exposed;
- with source title, page/chunk identity, supporting text/span, provenance, and extraction pass retained per fact.

That future fact-generation protocol must be frozen separately before generation. Stage 0 neither regenerates facts nor runs that experiment.

## 12. Human annotation and reliability

### 12.1 Primary annotation

One primary annotator labels all **636** source instances under Passes 1–3. Pass 1 is performed at question level and retained with each applicable source-instance log record or by a lossless reference to the question-level record.

The primary annotator must finish a question's required blinded stages without seeing corpus status, old D4 facts, or another annotator's decisions.

### 12.2 Independent reliability subset

An independent second annotator labels a predeclared **160-instance** reliability subset. The subset is sampled only after the automatic fields are reproducibly available.

Sampling requirements:

- fixed random seed: **20260822**;
- sampling unit: source instance, not question;
- proportional stratification across the four cells formed by:
  - lexical trace `YES` / `NO`; and
  - `EXACT_PRESENT` / `EXACT_ABSENT`;
- the sampling manifest is fixed before the second annotator sees any item or primary label;
- the second annotator follows the same staged visibility procedure;
- the second annotator does not see the primary annotator's decisions or rationale.

For reproducibility, stratum quotas must be computed from finalized automatic-field counts as `160 × stratum_count / 636`, assigned by largest-remainder rounding to sum to 160, with deterministic lexical ordering used to break equal remainders. Source instances must first be sorted by `source_instance_id`; a seeded pseudorandom sample is then taken independently within each stratum. The eventual implementation must record the pseudorandom library/version and exact stratum order alongside the manifest.

### 12.3 Reliability reporting

Reliability is calculated from the two independent, pre-adjudication labels on the 160 instances. Report:

- raw three-class role agreement;
- Cohen's kappa for `EXPLICIT` / `LATENT_BRIDGE` / `AMBIGUOUS`;
- the 3×3 confusion matrix;
- raw agreement and class counts by automatic stratum.

The reliability thresholds are project design thresholds, not literature facts.

GO reliability requires both:

```text
raw agreement >= 80%
AND
Cohen's kappa >= 0.60
```

If either threshold fails, Stage 0 is **NO-GO** and modeling must not continue. Disagreements must be reviewed, definitions amended transparently, and a newly sampled reliability subset annotated independently under the amended protocol. The amendment must predeclare the new sampling details; a results-motivated threshold change is prohibited.

## 13. Adjudication

Adjudication is required for:

- every disagreement in the 160-instance reliability sample;
- every primary `AMBIGUOUS` case considered potentially important to bridge eligibility;
- every `LOW`-confidence primary annotation proposed for a later experiment.

The adjudicator may inspect both annotators' rationales and the evidence permitted in Pass 3. Corpus status remains irrelevant to the role decision. Answer inspection is permitted only under the logged exception in Section 8.3.

Final data must preserve:

- annotator A's original label and rationale;
- annotator B's original label and rationale, where available;
- the adjudicated label;
- the adjudication rationale;
- whether adjudication changed bridge eligibility.

Raw annotation history is append-only and is never overwritten by adjudication.

## 14. Confidence

Allowed confidence values are:

- `HIGH`;
- `MEDIUM`;
- `LOW`.

Confidence describes certainty in the selected label; it does not replace `AMBIGUOUS`. A low-confidence forced label should normally be adjudicated before it can enter a later experimental eligibility set.

## 15. Stage-0 outputs and metrics

Stage 0 is descriptive and diagnostic. Its primary outputs are:

1. Counts and proportions of `EXPLICIT`, `LATENT_BRIDGE`, and `AMBIGUOUS` source instances.
2. The source-role cross-tabulation by `EXACT_PRESENT` and `EXACT_ABSENT`.
3. Counts of unique questions containing:
   - explicit sources only;
   - at least one latent bridge;
   - at least one exact-absent title;
   - mixed source roles.
4. Exact-title coverage by human source role.
5. The number of clearly sequential bridge cases.
6. The number of unique bridge-eligible qids for which:
   - the parent source is `EXPLICIT`;
   - the parent source is `EXACT_PRESENT`;
   - the child source is `LATENT_BRIDGE`;
   - the child source is `EXACT_PRESENT`;
   - `dependency_status = CLEAR_DEPENDENCY`.
7. The annotation reliability statistics specified in Section 12.3.

For question-level summaries, “explicit sources only” means every distinct required gold title for that qid has final role `EXPLICIT`. “Mixed source roles” means at least two different final role values occur for the qid. Counts must identify whether they use pre-adjudication or final adjudicated labels; GO/NO-GO counts use final labels.

Secondary diagnostics are:

- historical English lexical trace versus the human Urdu-primary `EXPLICIT` label;
- lexical-trace precision and recall as analytical diagnostics only, treating human `EXPLICIT` as the diagnostic reference class;
- source-role distributions by official decomposition structure.

Stage 0 must not report final QA accuracy as an outcome. Final answer correctness is not source-recovery evidence, and it cannot substitute for source metrics.

## 16. GO/NO-GO to the bridge-state falsification experiment

Stage 0 passes only if **both** conditions hold.

### Condition 1 — Label reliability

On the independent 160-instance subset before adjudication:

```text
raw agreement >= 80%
AND
Cohen's kappa >= 0.60
```

### Condition 2 — Sufficient clean bridge cases

At least **30 unique DEV200 qids** must contain at least one bridge pair satisfying all of:

Parent:

- final role `EXPLICIT`;
- `EXACT_PRESENT`.

Child:

- final role `LATENT_BRIDGE`;
- `EXACT_PRESENT`.

Relationship:

- `dependency_status = CLEAR_DEPENDENCY`;
- the eligible parent is recorded in `parent_source_titles` for that child.

The count is over unique qids, not title instances or possible parent-child pair combinations. Ambiguous or unresolved cases do not qualify. The threshold of 30 is a project feasibility threshold for a paired pilot, not a literature-derived performance guarantee.

The ≥30 eligible-qid rule is only a Stage-0 feasibility gate. It does not establish adequate statistical power for a method claim and does not guarantee that the later bridge-state experiment can confirm an effect. The separate future bridge-state freeze must predeclare:

- the primary effect size of interest;
- the statistical test;
- the uncertainty/confidence interval;
- the minimum practically meaningful gain;
- any sample-size or power limitation.

If the available eligible sample cannot support a claim-bearing analysis, the later experiment must be described as diagnostic/pilot even if Stage 0 passes.

If fewer than 30 such qids exist, the result is **NO-GO** for a claim-bearing bridge-state experiment on DEV200. Cases must not be relabeled or dependency-forced to meet the threshold.

If Stage 0 passes, the next study is only the smallest controlled bridge-state falsification experiment. Passing does not authorize immediate router training or an end-to-end system.

## 17. Boundary for the next experiment

This section records a boundary only. The future experiment is not part of Stage 0 and must be frozen separately before it is run.

Under matched retrieval budgets, it must compare next-source retrieval using:

A. Urdu question only.  
B. Urdu question plus the first-hop whole page.  
C. Urdu question plus a subanswer/entity state.  
D. Urdu question plus **new sequential source-specific D4-style atomic facts**.  
E. Urdu question plus gold atomic facts as an oracle ceiling.

The primary future outcome is:

- next latent gold-source Recall@10.

Secondary future outcomes are:

- Recall@1 and Recall@5;
- row-level all-required-source coverage.

The future protocol must use:

- the same candidate universe;
- the same retriever;
- the same top-k;
- the same reranker budget;
- no future-source leakage.

Oracle first-hop and predicted first-hop conditions must be separate experiments. The separately frozen protocol must define the eligible-pair sampling unit, context budgets, provenance checks, statistical test, effect-size criterion, and a meaningful-advantage stop rule before results are inspected.

## 18. Leakage rules

DEV200 is development data. No Stage-0 result is an untouched-test result.

During source-role annotation:

- hide the gold answer;
- hide the English question for the primary label;
- hide corpus status until role is finalized;
- hide other gold titles during the per-title decision;
- hide decomposition and evidence until Pass 3;
- hide existing D4 facts completely;
- prevent annotators from seeing each other's labels before independent annotation.

For any future method:

- no same-row gold titles may appear in prompts or training inputs;
- no same-row decomposition/evidence may be used as retrieval targets unless the experiment explicitly declares an oracle condition;
- no candidate-budget inflation is allowed across methods;
- oracle first-hop and predicted first-hop results must remain separate;
- source metrics must improve before final-QA gains are interpreted as evidence for source recovery;
- facts derived from a future source may not enter the state used to retrieve that source;
- annotation labels and adjudication rationales may define analysis sets, but may not leak target source identity into non-oracle retrieval inputs.

## 19. No abstention claim yet

Because current corpus status is exact-title-only and redirect/alias resolution is unavailable, Stage 0 must not define:

```text
EXACT_ABSENT -> ABSTAIN
```

as a gold training rule or method target.

Corpus-aware abstention remains deferred. Before it becomes a target, a separate redirect/alias-aware attainability audit must determine whether an apparently absent gold title has a valid equivalent in the frozen local corpus. That audit must define identity resolution and acceptable equivalence before observing abstention performance.

## Reproducibility metadata

Every Stage-0 generated artifact must record the following metadata, either in-file or in a companion manifest:

- Stage-0 freeze document Git commit hash;
- Stage-0 preparation/annotation script Git commit hash;
- SHA-256 hash of `data/strategyqa_official/dev200_seed4242.jsonl`;
- SHA-256 hash of `outputs/efbpt/d2/d2_title_coverage.json`;
- SHA-256 hash of `outputs/efbpt/d2/d2_recall_split.json`;
- frozen corpus metadata path: `rag/index/wikipedia_full_meta.jsonl`;
- frozen unique normalized title count: **6,402,346**;
- random seed: **20260822**;
- generation timestamp;
- software/Python version used for deterministic sampling.

Hashes must not be computed or inserted while editing this freeze. The future preparation script must compute and record them when Stage-0 artifacts are generated.

## 20. Revision and amendment policy

This freeze becomes active when committed to Git. This document-creation task does not commit it.

After activation, non-material corrections—spelling, formatting, or a path clarification that does not change meaning—may be made with a note.

Material changes include:

- source-role definitions;
- label set;
- corpus-status definition;
- annotation visibility;
- dataset;
- annotation unit;
- agreement thresholds;
- reliability sample design;
- eligibility rules;
- primary Stage-0 metrics;
- GO/NO-GO criteria;
- leakage rules.

Every material change requires an appended section titled `AMENDMENT N` containing:

- date;
- exact change;
- reason;
- whether any annotations or results had already been viewed;
- affected records;
- whether reannotation is required.

The frozen rule must never be silently rewritten after results are observed. A results-motivated threshold change is prohibited.

## 21. What Stage 0 must not do

Stage 0 explicitly prohibits:

- building L1;
- swapping encoders as the contribution;
- training a router;
- training a bridge retriever;
- implementing GraphRAG;
- presenting alias/hybrid retrieval engineering as a novelty claim;
- using old D4 aggregate facts as bridge state;
- forcing a gold source sequence;
- calling lexical absence latent;
- calling exact-title absence true knowledge absence;
- claiming that Urdu alone creates method novelty;
- evaluating novelty through final QA accuracy alone;
- running the future bridge-state comparison before its separate freeze;
- treating DEV200 or any currently named repository split as an untouched final test.

## 22. Downstream decision tree

```text
If Stage 0 FAILS reliability:
    revise annotation definitions; no modeling.

If Stage 0 is reliable but has <30 eligible bridge qids:
    do not build the bridge method on DEV200; reconsider dataset/task scope.

If Stage 0 PASSES:
    freeze and run only the smallest bridge-state falsification experiment.

If that later experiment shows no meaningful D4-state advantage:
    abandon the routing/bridge method claim.

Only if that later mechanism succeeds:
    design the full failure-aware source-recovery method.
```

## AMENDMENT 1 — Durable sensitive-view provenance events

**Date:** 2026-08-31

**Exact change:** Before supporting paragraph text is displayed, the annotation runner must validate that the reveal is permitted for the current pass and source instance, append a `SUPPORTING_TEXT_VIEW` event to the human annotation log, flush and `fsync` that event successfully, and only then display the text. The event records the source instance, role-qualified viewer identity, timestamp, round/pass, and permitted support-path reference or references. It must not store supporting paragraph text.

Before the gold answer is displayed through the adjudication-only exception workflow, the runner must first require an inspection reason and validate that the reveal is permitted for the current adjudication case. It must then append an `ANSWER_INSPECTION` event, flush and `fsync` that event successfully, and only afterward display the answer. The event records the source instance, adjudicator identity, reason, timestamp, round, provisional pre-inspection role, and `changed_decision`. It must not store the gold answer value.

At inspection time, whether seeing the answer changes the final decision is unknown. The immediate `ANSWER_INSPECTION` event therefore records:

```text
changed_decision = null
```

Here null means **not yet determined**; it must not be interpreted as false. The later completed adjudication record must contain the final Boolean indicating whether answer inspection changed the decision.

Sensitive-view events are append-only. They must survive quit, crash, restart, or failure to complete the later annotation/adjudication decision. No prior event may be rewritten or erased. Repeated deliberate reveals require corresponding durable events and must not silently discard earlier inspection history. A reveal must not occur if its durable append or `fsync` fails.

**Reason:** Before any human annotation began, final review found that sensitive views were represented only inside later completed records. A quit, restart, or crash after a reveal but before the final save could therefore leave the reveal without a durable provenance record.

**Annotations or results already viewed:** None. The canonical human annotation log did not exist when this defect was identified and this amendment was made.

**Affected records:** The human annotation log schema and future `SUPPORTING_TEXT_VIEW`, `ANSWER_INSPECTION`, and completed adjudication records only. Existing generated Stage-0 artifacts are not regenerated by this amendment task.

**Reannotation required:** No. Human annotation had not started.

This is solely a pre-annotation provenance and data-model amendment. It does not change source-role definitions, dataset populations, visibility permissions, automatic features, the reliability sample or thresholds, seed **20260822**, the **30 unique eligible bridge-qid** feasibility gate, or any scientific interpretation.
