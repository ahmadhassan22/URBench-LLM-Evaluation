# URBench: Urdu Reasoning Benchmark for Large Language Models

> **Master's Thesis Research** · Harbin Institute of Technology, Shenzhen
> A systematic evaluation of LLM reasoning in Urdu — a low-resource language spoken by 230M+ people.

---

## Overview

URBench provides the most comprehensive cross-architecture comparison of Urdu reasoning performance to date. It evaluates **9 general-purpose multilingual LLMs** and **2 Urdu-specialized models** across **5 reasoning datasets** using **3 prompting strategies**, plus method work on retrieval-augmented generation (RAG), cross-lingual prompting (XLT), and **SDFR-UR** (Similarity-Based Dynamic Few-Shot Retrieval for Urdu Reasoning).

The benchmark addresses a critical gap: while Urdu has a large global speaker base, structured reasoning evaluation in Urdu remains severely underexplored compared to high-resource languages.

**A note on methodology.** This project underwent a systematic audit that identified experimental confounds in several early method comparisons. Results affected by those confounds have been **re-run under a controlled fair-regime protocol and corrected in this document**, with the original (confounded) numbers retained for transparency. The audit, the corrections, and two formally rejected method branches are documented below. Negative results are reported as first-class findings.

---

## Datasets

All datasets are translated from English into Urdu using GPT-4.1 with normalization handling Urdu-specific script variations (diacritics, Eastern/Western numerals, code-switching).

| Dataset | Task Type | Examples | Answer Format |
|---|---|---|---|
| **BoolQ** | Binary reading comprehension | 1,550 | ہاں / نہیں |
| **CSQA** | Commonsense MCQ (5-way) | 1,500 | A / B / C / D / E |
| **PIQA** | Physical commonsense | 750 | 0 / 1 |
| **StrategyQA** | Multi-hop binary reasoning | 2,290 | ہاں / نہیں |
| **GSM8K** | Arithmetic word problems | 700 | Numeric |

Official StrategyQA annotations (gold decompositions, per-step evidence paragraphs) have been integrated and mapped to all 2,290 rows, enabling evidence-grounded diagnostics. See *Data Integration* below.

---

## Models Evaluated

### General Multilingual LLMs

| Model | Family | Size | Quantization |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 1.5B | — |
| Qwen2.5-7B-Instruct | Qwen2.5 | 7B | — |
| Qwen2.5-14B-Instruct | Qwen2.5 | 14B | AWQ |
| Qwen3-8B | Qwen3 | 8B | — |
| Qwen3-14B | Qwen3 | 14B | — |
| Meta-LLaMA-3.1-8B-Instruct | LLaMA | 8B | — |
| Meta-LLaMA-3.1-70B-Instruct | LLaMA | 70B | AWQ INT4 |
| DeepSeek-R1-Distill-Qwen-7B | DeepSeek | 7B | — |
| Gemma-2-9B-IT | Gemma | 9B | — |

### Urdu-Specialized LLMs

| Model | Base | Urdu Training | Method |
|---|---|---|---|
| Alif-1.0-8B-Instruct | LLaMA-3.1-8B | Instruction tuning on Urdu-Instruct | LoRA + SFT |
| Qalb-1.0-8B-Instruct | LLaMA-3.1-8B | 1.97B token continued pre-training + SFT | Full pre-training |

---

## Prompting Strategies

| Strategy | Description |
|---|---|
| **Zero-shot** | Direct question with no examples |
| **3-shot** | Three in-context demonstrations |
| **Chain-of-Thought (CoT)** | Step-by-step reasoning before final answer |

For Qwen3 models, CoT uses `enable_thinking=True` to activate internal reasoning via `<think>` blocks. For all other models, CoT uses explicit step-by-step prompt instructions.

---

## Results — General Multilingual LLMs

### BoolQ

| Model | Zero-shot | 3-shot | CoT |
|---|---|---|---|
| Qwen2.5-1.5B | 62.69% | 55.24% | 47.00% |
| Qwen2.5-7B | 66.73% | 71.41% | 62.26% |
| Qwen2.5-14B-AWQ | 76.26% | 72.24% | 76.13% |
| Qwen3-8B | 79.68% | 81.52% | 80.84% |
| Qwen3-14B | 84.51% | 84.89% | 84.47% |
| Meta-LLaMA-3.1-8B | 78.49% | 66.82% | 71.40% |
| **Meta-LLaMA-3.1-70B-AWQ** | 87.51% | **88.49%** | 87.45% |
| DeepSeek-R1-Distill-Qwen-7B | 56.76% | 62.07% | 61.31% |
| Gemma-2-9B-IT | 86.30% | 87.20% | 86.10% |

### CSQA

| Model | Zero-shot | 3-shot | CoT |
|---|---|---|---|
| Qwen2.5-1.5B | 6.07% | 9.53% | 25.20% |
| Qwen2.5-7B | 45.40% | 44.40% | 45.13% |
| Qwen2.5-14B-AWQ | 50.93% | 46.53% | 49.73% |
| Qwen3-8B | 54.80% | 52.20% | 55.33% |
| Qwen3-14B | 58.73% | 56.60% | **63.00%** |
| Meta-LLaMA-3.1-8B | 49.20% | 48.13% | 46.47% |
| **Meta-LLaMA-3.1-70B-AWQ** | **65.60%** | 64.87% | 58.07% |
| DeepSeek-R1-Distill-Qwen-7B | 26.33% | 22.87% | 23.33% |
| Gemma-2-9B-IT | 59.87% | 59.20% | 51.87% |

### PIQA

| Model | Zero-shot | 3-shot | CoT |
|---|---|---|---|
| Qwen2.5-1.5B | 33.73% | 51.07% | 51.07% |
| Qwen2.5-7B | 59.20% | 62.00% | 44.93% |
| Qwen2.5-14B-AWQ | 60.67% | 65.33% | 58.93% |
| Qwen3-8B | 56.53% | 55.87% | 55.20% |
| Qwen3-14B | 59.33% | 63.73% | **65.73%** |
| Meta-LLaMA-3.1-8B | 35.20% | 55.87% | 49.47% |
| **Meta-LLaMA-3.1-70B-AWQ** | 62.93% | **76.40%** | 31.73% |
| DeepSeek-R1-Distill-Qwen-7B | 50.67% | 53.47% | 52.93% |
| Gemma-2-9B-IT | **68.67%** | 53.60% | 50.00% |

### StrategyQA

| Model | Zero-shot | 3-shot | CoT |
|---|---|---|---|
| Qwen2.5-1.5B | 53.28% | 48.38% | 49.91% |
| Qwen2.5-7B | 55.63% | 70.17% | 56.11% |
| Qwen2.5-14B-AWQ | 73.19% | 75.11% | 76.72% |
| Qwen3-8B | 79.96% | 80.74% | 79.74% |
| Qwen3-14B | 83.89% | 83.97% | 83.89% |
| Meta-LLaMA-3.1-8B | 57.55% | 78.43% | 78.17% |
| **Meta-LLaMA-3.1-70B-AWQ** | 88.69% | **90.09%** | 49.91%* |
| DeepSeek-R1-Distill-Qwen-7B | 55.24% | 51.66% | 47.51% |
| Gemma-2-9B-IT | 82.79% | 80.70% | 80.61% |

*CoT accuracy collapse due to answer extraction failure at 60.83% coverage.

> **Important:** these StrategyQA scores were obtained with gold supporting facts present in the prompt (the dataset's default configuration). They are **not** comparable to the no-facts fair-regime baseline used for method development (65.50%). See *Fair-Regime Protocol*.

### GSM8K

| Model | Zero-shot | 3-shot | CoT |
|---|---|---|---|
| Qwen2.5-1.5B | 11.29% | 5.43% | 3.71% |
| Qwen2.5-7B | 17.14% | 5.00% | 28.29% |
| Qwen2.5-14B-AWQ | 25.14% | 8.86% | 39.43% |
| Qwen3-8B | 55.43% | 22.86% | 79.86% |
| **Qwen3-14B** | 39.14% | 32.86% | **83.71%** |
| Meta-LLaMA-3.1-8B | 17.29% | 12.57% | 11.00% |
| Meta-LLaMA-3.1-70B-AWQ | 32.71% | 34.43% | 55.00% |
| DeepSeek-R1-Distill-Qwen-7B | 28.71% | 9.14% | 33.43% |
| Gemma-2-9B-IT | 16.86% | 16.43% | 71.57% |

---

## Results — Urdu-Specialized LLMs (CoT only)

Both Urdu-specialized models are evaluated using CoT only, compared against their shared base model (LLaMA-3.1-8B-Instruct CoT) to isolate the effect of Urdu-specific training.

| Dataset | LLaMA-3.1-8B (base) | Alif-1.0-8B | Qalb-1.0-8B |
|---|---|---|---|
| BoolQ | 71.40% | 71.57% | 55.40% |
| CSQA | 46.47% | 46.60% | 31.20% |
| PIQA | 49.47% | 44.93% | 51.07% |
| StrategyQA | 78.17% | 66.94% | 72.14% |
| GSM8K | 11.00% | **55.86%** | 38.29% |

**Key finding:** Urdu-specific instruction tuning (Alif) and continued pre-training (Qalb) help significantly on arithmetic reasoning (GSM8K, +44.86pp for Alif) but do not consistently improve — and sometimes substantially hurt — multi-hop and commonsense reasoning (StrategyQA: −11.23pp for Alif, −6.03pp for Qalb).

> **Urdu language specialization is not the same as Urdu reasoning specialization.** This distinction directly motivates the current research direction.

---

## Fair-Regime Protocol

Mid-project, a systematic audit revealed that several early method-vs-baseline comparisons were **not controlled**: the method and the baseline differed in `enable_thinking`, `max_tokens`, gold-fact availability, prompt wording, and evaluation-set size — all at once. Apparent effects were therefore uninterpretable.

A **fair-regime protocol** was adopted for all subsequent comparisons. Method and baseline must match on:

- model and quantization
- thinking mode (`enable_thinking`)
- token budget (`max_tokens`)
- prompt wording, except the component under test
- answer parser
- evaluation set and size
- gold-fact availability

Additional acceptance rules:

- A positive accuracy delta is **necessary but not sufficient**. Truncation, parse failures, label balance, and label-copy rate are inspected before any result is claimed.
- **Mechanism verification is mandatory**: the gain must be shown to come from the intended mechanism.
- Small-sample probes (n≈12–50) are treated as directional gates, never as bankable results.

Re-running the method comparisons under this protocol changed several headline conclusions. The corrected results follow.

---

## Method: SDFR-UR

SDFR-UR replaces fixed in-context examples with **dynamically retrieved cross-lingual demonstrations**: for each Urdu test question, the top-3 most semantically similar *English* examples are retrieved from an English pool using multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and FAISS `IndexFlatIP`. Retrieved English examples serve as few-shot context for the Urdu question.

### Retrieval Pool Configuration

| Dataset | Pool Source | Pool Size | Eval Size |
|---|---|---|---|
| GSM8K | English train split (ModelScope) | 7,473 | 700 |
| BoolQ | google/boolq via ModelScope (passage-based) | 7,877 | 310 |
| CSQA | Full English train (ModelScope, deduplicated) | 9,441 | 300 |
| PIQA | First 80% of English file | 600 | 150 |
| StrategyQA | First 80% of English file | 1,832 | 458 |

### Results — Fair Regime (Qwen3-14B, all parameters matched)

| Dataset | Fair Baseline | SDFR-UR | Δ | Verdict |
|---|---|---|---|---|
| **GSM8K** | 88.71% | 87.14% | −1.57pp | ❌ **RETRACTED** — earlier +8.15pp was answer leakage; clean re-run p=0.1690 |
| GSM8K *(decontaminated pool, n=700)* | 88.71% | 87.14% | −1.57pp | ❌ No effect (McNemar b=21, c=32, p=0.1690) |
| PIQA | 72.00% | 77.33% | +5.33pp | ⚠️ **Not significant** (McNemar b=27, c=19, p=0.302); mechanism: label-bias correction |
| BoolQ | 84.89% | 85.48% | +0.59pp | ➡️ Parity (both thinking-OFF) |
| CSQA | 63.33% | 61.00% | −2.33pp | ❌ Slight loss |
| StrategyQA *(no-facts)* | 65.50% | 69.43% | +3.93pp | ❌ **Not a reasoning win** — see below |

**GSM8K is RETRACTED (2026-08-18).** The +8.15pp was answer leakage: the 700 evaluation items are Urdu translations of GSM8K *train*, the retrieval pool *is* GSM8K train, overlap was 700/700, and `retrieve()` had no self-exclusion. The item's own source problem was retrieved for 90.3% of items and the gold answer sat in the prompt for 92.4%. Re-run with a decontaminated pool (700 records removed, 6,773 remaining, 829 leaked neighbours filtered across 644/700 items): **88.71% baseline vs 87.14% SDFR, d = −1.57pp, McNemar p=0.1690 — no effect.** See EXPERIMENT G1 in `experiments.md`.

**PIQA — not statistically significant.** The pool/eval split is genuinely clean (zero overlap, no near-duplicates), but the paired test gives b=27, c=19, **p=0.302**. Both arms over-predict label `1` (gold 77/73; baseline predicts 109×`1`, SDFR 89×`1`), and the shift gains on gold-0 items (36/73 → 50/73) while losing on gold-1 (72/77 → 66/77). The mechanism is label-bias correction, and the gain is not supported at n=150. **PIQA is not claimed as a method win.**

**StrategyQA is a confirmed method weakness, not a win.** The +3.93pp gain did **not** survive mechanism verification:

- demonstration label-copy rate: **65.5%**
- self-verbalized use of retrieved content: 21.6%
- similarity gap between correct and incorrect retrievals: **near zero** (retrieval quality is effectively random)

The delta is **label-following on weak retrieval**, not reasoning. It is reported as a genuine SDFR weakness on fact-dependent reasoning.

### Corrections to Previously Reported Results

The originally published SDFR-UR numbers were produced under **confounded conditions** and are corrected here:

| Dataset | Originally reported | Fair-regime result | What was wrong |
|---|---|---|---|
| GSM8K | +6.00pp | **+8.15pp** | SDFR ran *handicapped* (thinking off, lower token budget); the fair win is **larger** |
| StrategyQA | **−21.96pp** | +3.93pp (not a reasoning win) | Baseline had gold facts injected; method did not. The "catastrophic failure" was an **experimental artifact** |
| CSQA | −5.33pp | −2.33pp | Mismatched thinking mode / token budget |
| PIQA | +4.94pp | +5.33pp | Minor; eval-set and token-budget alignment |

> **The single most important methodological lesson of this project:** most apparent sign-flips traced to experimental confounds, not to model or method behaviour. The dramatic "retrieval destroys multi-hop reasoning" conclusion did not survive controlled re-testing.

### Retrieval Quality as a Selection Criterion

| Dataset | AvgTopSim | Fair-regime outcome |
|---|---|---|
| GSM8K | 0.777 | Strong win |
| CSQA | 0.694 | Slight loss |
| PIQA | 0.617 | Win |
| BoolQ | 0.611 | Parity |
| StrategyQA | 0.567 | No genuine gain |

Cross-lingual retrieval similarity remains a useful *ex ante* predictor: tasks whose reasoning structure transfers across languages (arithmetic, physical goals) benefit; tasks requiring external factual knowledge do not.

### SDFR-UR on Urdu-Specialized Models

Evaluated on the three datasets where SDFR-UR showed wins on Qwen3-14B, using a context-hints format.

**Alif-1.0-8B-Instruct**

| Dataset | Alif CoT Baseline | SDFR-UR | Δ |
|---|---|---|---|
| GSM8K | 55.86% | **64.29%** | **+8.43pp** |
| PIQA | 44.93% | 45.33% | +0.40pp |
| BoolQ | 71.57% | 67.74% | −3.83pp |

**Qalb-1.0-8B-Instruct**

| Dataset | Qalb CoT Baseline | SDFR-UR | Δ |
|---|---|---|---|
| GSM8K | 38.29% | **43.86%** | **+5.57pp** |
| PIQA | 51.07% | **52.00%** | +0.93pp |
| BoolQ | 55.40% | **65.48%** | **+10.08pp** |

GSM8K benefits consistently across all three models (+5.57 to +8.43pp), reinforcing that structured arithmetic reasoning transfers well through cross-lingual demonstrations.

### Data-Leakage Detection

An initial CSQA run using the unfiltered ModelScope pool (9,741 examples) achieved an implausible 96.00%, traced to **100% ID overlap** between the retrieval pool and the evaluation set. All 300 overlapping IDs were removed before the reported run. Leakage auditing is now a standing pre-flight check on every retrieval pool.

---

## Method: RAG on StrategyQA — Diagnosis and Re-Analysis

### The original result (superseded)

| Setting | Accuracy |
|---|---|
| CoT baseline (gold facts present) | 83.89% |
| RAG with retrieved context | 56.24% |
| Δ | −27.65pp |

This was originally reported as evidence that *RAG degrades reasoning in low-resource settings*. **That conclusion does not hold as stated.**

### What was actually wrong

Error analysis of the failing cases produced a corrected diagnosis:

| Failure mode | Share of failures |
|---|---|
| **Coverage gap** — required article absent from the index | **~48%** |
| **Semantic drift** — right topic retrieved, wrong passage | **~38%** |
| Other (multi-hop partial retrieval, entity ambiguity) | ~14% |

The retrieval index used for that experiment was built from a **filtered subset** (~53k vectors, 79 MB), not the full corpus, and additionally suffered an embedding **prefix mismatch** between index and query time. The −27.65pp figure therefore measures a *broken retrieval implementation*, not the viability of RAG.

### Corrective work

A correct full-Wikipedia dense index was rebuilt and verified:

| Property | Value |
|---|---|
| Vectors | 23,963,971 |
| Dimension | 384 (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Index type | FAISS `IndexFlatIP` (cosine on normalized vectors) |
| Size | ~35 GB |
| Metadata access | cached byte-offset seek (never fully loaded into memory) |

> **Correction (2026-07-15): the rebuilt index does NOT fully fix the coverage gap.** While preparing EFBPT audit data, the index was found to be **incomplete**: across a 30-question audit sample, 28 of 86 official evidence pages (~33%) were absent from the index title space, including major head articles (Woodrow Wilson, Vietnam War, Bucharest, Vlad the Impaler, and others). Root cause: `rag/build_chunks.py` globs whatever parquet shards happen to sit in the ModelScope download cache; ModelScope throttling (~60 KB/s) makes a partial download likely, so an unknown fraction of Wikipedia shards were never chunked. The chunker itself is not at fault. The index title space holds 6,407,814 unique titles — near-complete but missing head articles in a non-random pattern consistent with missing shards.
>
> **Consequence for the entity-first branch below:** its 41.7% union coverage measured **index gaps mixed with entity-corruption failures**, not entity grounding in isolation. With ~33% of evidence pages absent from the index, the maximum reachable coverage was already capped well below the 80% gate. The **entity-corruption taxonomy still stands** (it was observed directly), but the 41.7% figure must not be read as a pure measure of entity grounding. Rebuilding the index is deferred — the current EFBPT direction constructs training data from official evidence paragraphs and does not depend on this index.

---

## Method: Entity-First Dual-View Retrieval — Rejected by Gate

A leakage-safe, inference-time method was designed to address the residual retrieval failures:

1. typed self-decomposition of the Urdu question into `RETRIEVE` / `REASON` steps;
2. **dual-view entity canonicalization** — candidates generated independently from the Urdu question (direct) and from an English translation, then grounded against Wikipedia titles;
3. per-step retrieval and cross-encoder reranking (`bge-reranker-v2-m3`);
4. evidence verification with an exact-substring supporting-span check;
5. deterministic fallback to the no-facts answer when evidence is insufficient.

Rationale: Urdu proper nouns drop short vowels, making transliterations systematically ambiguous. Probes showed the two views corrupt **disjoint** entity sets, suggesting their union could recover the correct entity.

### Result: the branch failed its predeclared gate

A 12-item manual oracle probe (DEV50, leakage-safe) produced:

| View | Official evidence page in retrieved pool |
|---|---|
| Direct (Urdu → English entity) | 25.0% |
| Translate-first | 25.0% |
| **Union** | **41.7%** |
| **Required gate** | **≥80%** |

The dual views are genuinely **complementary** (union > either single view), but absolute quality is far below the deployable threshold. **The branch was rejected and not promoted to full evaluation.**

> **Caveat (2026-07-15):** this probe ran against the incomplete index described above. Because ~33% of official evidence pages were absent from the index, the 41.7% union figure conflates missing-page coverage gaps with genuine entity-corruption failures. The rejection verdict stands (union was far below 80% under any reading), but the number is a *lower bound mixed with an index artifact*, not a clean entity-grounding measurement. The corruption taxonomy below was observed directly and is unaffected by the index gap.

### Why it failed — a reusable failure taxonomy

Cross-lingual entity corruption occurs *before retrieval can help*. No amount of reranking or gating recovers an entity that has already changed identity:

| Intended entity | Corrupted to |
|---|---|
| Bandy (the sport) | band (music) |
| Audi R8 V10 Plus | Audi A8 / "AR TV-10 Plus" |
| Darth Vader | Darryl Ward / Dartmouth |
| L. Ron Hubbard | Elton John / "El Rancho Hacienda" |
| Latino (identity) | Latin (language) |
| sound barrier | audio barrier / noise cancellation |

A second finding: **cross-encoder reranker scores are not a reliable confidence signal.** Passages scoring ≈1.00 were repeatedly found to contain *no* supporting evidence for the query. Any evidence gate must verify entity consistency and an explicit supporting span — not trust the reranker score.

> This is reported as a **negative result with a verified mechanism**. It is the empirical basis for the current research direction.

---

## Data Integration & Leakage Controls

Official StrategyQA annotations were integrated to enable evidence-grounded diagnostics. The official release and the URBench export use **different qid namespaces**, so the join was performed conservatively on normalized English question text.

| Validation | Result |
|---|---|
| Official rows (train + dev) | 2,290 |
| Mapped official ↔ URBench | **2,290 / 2,290** |
| Duplicate-question groups resolved by secondary signature | 1 / 1 |
| Answer mismatches | 0 |
| Evidence paragraph references | 16,002 (9,251 unique) |
| Missing paragraph IDs | 0 |
| Evidence ↔ decomposition structural mismatches | 0 |

**Split discipline:**

- **eval458** — final evaluation set; untouched during method development.
- **DEV50** (seed 42) — method-development set, drawn from non-evaluation rows with real paragraph evidence; **zero intersection with eval458**.
- Gold `term`, `description`, `facts`, `decomposition`, and `evidence` are **never** method inputs. They are scorer-only fields used for offline diagnostics.

---

## XLT (Cross-Lingual Thought Prompting)

| Version | Coverage | Accuracy |
|---|---|---|
| XLT v1 (initial, broken) | 74.72% | 42.31% |
| XLT v2 (fixed prompt + tokens) | 96.42% | 58.43% |
| CoT baseline | 100% | 83.89% |

XLT consistently underperformed CoT. Two failure modes: token truncation cutting off the answer, and factual hallucination during the English restatement step on implicit multi-hop questions. Discontinued.

---

## Key Findings

**1. Experimental confounds, not model behaviour, drove most early "findings."**
Mismatched thinking modes, token budgets, gold-fact availability, and evaluation-set sizes produced dramatic but spurious effects — including an apparent −21.96pp "retrieval catastrophe" on StrategyQA that vanished under controlled conditions. Controlled re-testing is now a precondition for any claim.

**2. SDFR-UR is NOT a confirmed win. Both apparent wins failed verification.**
GSM8K's +8.15pp was answer leakage from a contaminated retrieval pool; re-measured cleanly at n=700 the effect disappears (−1.57pp, p=0.1690). PIQA's +5.33pp is not statistically significant (p=0.302) and is attributable to label-bias correction. Demonstration retrieval does not improve Urdu reasoning on tasks whose knowledge is already internal to the model.

**3. SDFR-UR does not genuinely help multi-hop factual reasoning.**
The nominal StrategyQA gain is label-copying, not reasoning — confirmed by mechanism inspection. Positive deltas are not accepted without mechanism verification.

**4. Retrieval failure in Urdu is an entity-grounding problem, not a ranking problem.**
Cross-lingual entity corruption (bandy→band, Darth Vader→Darryl Ward) occurs upstream of retrieval. Reranking, larger top-k, and confidence gating cannot repair an entity that has already changed identity.

**5. Reranker scores are not evidence.**
Cross-encoder scores near 1.00 routinely accompanied passages containing no supporting fact. Evidence gates require entity-consistency and verified supporting spans.

**6. Urdu language specialization ≠ Urdu reasoning specialization.**
Alif and Qalb improve Urdu arithmetic substantially (up to +44.86pp on GSM8K) while degrading multi-hop reasoning (Alif −11.23pp on StrategyQA). More Urdu text alone does not produce better Urdu reasoning.

**7. General multilingual models outperform Urdu-specialized models on most reasoning tasks.**
Qwen3-14B and LLaMA-3.1-70B consistently outperform Alif and Qalb despite no Urdu-specific training.

**8. Scale matters for Urdu reasoning.**
LLaMA-3.1-70B-AWQ sets highs on BoolQ (88.49%), CSQA zero-shot (65.60%), PIQA 3-shot (76.40%), and StrategyQA 3-shot (90.09%).

**9. Qwen3 leads on mathematical reasoning.**
Qwen3-14B reaches 83.71% GSM8K CoT, substantially exceeding all other architectures including LLaMA-70B (55.00%).

**10. CoT is essential for math but can be harmful for binary tasks.**
CoT hurts binary-task performance for LLaMA and Gemma architectures due to answer-extraction failures after long reasoning chains.

**11. Reasoning distillation does not transfer to low-resource settings.**
DeepSeek-R1-Distill-Qwen-7B underperforms similarly-sized instruction-tuned models on every Urdu task.

**12. Knowledge, not structure and not language, is the bottleneck in Urdu multi-hop reasoning.**
Supplying clean English facts to Qwen3-14B is worth **+20.50pp** on Urdu StrategyQA (78.00% vs 57.50% on DEV200; 83.10% vs 59.15% on the matched 71-question subset, McNemar p=0.0023). Every structural intervention tested across the full EFBPT programme moved accuracy 0–7pp. The gap is in what the model knows, not how the reasoning is organised.

**13. Correct entity labels are read but knowledge-inert.**
A causal counterfactual corrupted the entity block in the fine-tuning format: 22% of answers changed and truncation quadrupled (1.2% → 7.9%), proving the block is read. Yet on rows where both arms produced an answer, accuracy was statistically identical (63.01% vs 62.27%, p=0.71). Supplying the right name does not supply the knowledge behind the name.

**14. Fact delivery fails at every automatic route except extraction.**
On the matched 71-question subset: embedding retrieval recovers 23.15% of gold titles among pages that exist; whole correct Wikipedia pages as context are not significantly better than no facts (69.01% vs 59.15%, p=0.2295); the model's own generated facts add exactly zero (70.0% = 70.0%). Only two-pass extraction works — the model reading a correct article and writing short atomic English facts reaches **76.06% vs 59.15% (p=0.0290)**, recovering 71% of the achievable gap.

**15. StrategyQA evidence titles are largely unrecoverable from the question text.**
52.5% of gold evidence title-instances (334/636) have no lexical trace in the English question, and only 19.5% of DEV200 rows have all their gold titles lexically present. StrategyQA evidence pages are *justification* sources, not question entities — "Is SnapCap an example of a retail store?" requires the page *LendingTree*. This caps any retrieval- or linking-based method evaluated against gold evidence titles on this benchmark, independent of language. (The classifier used was lexical and produces false absents, so 52.5% is an upper bound.)

---

## Current Direction

Two method branches have been designed, tested, and **formally rejected against predeclared gates** (inference-time RAG; entity-first dual-view retrieval). Both rejections were mechanism-verified and are documented above.

The evidence points to a specific missing capability: **stable bilingual entity and relation alignment**. The failures occur before retrieval, are systematic rather than random, and cannot be repaired by additional inference-time retrieval stages. Prompting a frozen model — with schema constraints, dual views, reranking, or verification — does not fix them.

**The target failure mode is confirmed (2026-07-17).** A direct probe fed 28 multi-hop Urdu questions to Qwen3-14B with no retrieval and inspected its own reasoning traces for entity faithfulness. Three independent readers scored the traces and agreed that the model corrupts named entities in **~18–26% of testable (transliteration-ambiguous) entities** and in **39% of rows** — resolving Urdu entity mentions to wrong or fabricated identities (e.g. ŽRK Kumanovo → a pop song, Alexander Nevsky → a fabricated archbishop, llama → a Tibetan Buddhist "Lama"). Corruption concentrates on rare, transliteration-hard entities; famous entities resolve correctly. This confirms EFBPT's target failure mode exists at the model level, independent of retrieval.

**A manual-repair feasibility check (AUDIT30) showed the failure needs narrower scope than first assumed.** 30 model-drafted entity-bridge plans were hand-audited against a predeclared gate (≥24/30 clean-or-fixable). Result: 20/30 — gate failed. But the 10 broken rows failed for reasons *outside* EFBPT's scope (world-knowledge-quantity questions, single-step decompositions, translation errors), not entity disambiguation. Within entity-bridge-scoped rows, yield was 20/22 — promising, but logged as a hypothesis to test, not a proven result.

**A fully-automated training-data pipeline was then attempted and rejected (2026-07-21).** Plan: auto-draft an entity plan, independently re-verify it with a second pass, and auto-accept only rows where both passes agree — skipping per-row human review so the method could scale past ~1,000 rows without a manual bottleneck. Tested end-to-end on a fresh 30-row blind sample against predeclared gates (≥60% accepted coverage, ≥95% accepted-row precision): result was 33% coverage, 70% precision — both gates failed. Root cause, confirmed via full manual gold audit: the two agreeing passes share the same base model and the same blind spots on rare secondary entities, so agreement is not a reliable proxy for correctness. This is a structural ceiling, not a threshold that further rule-tuning can fix.

**Plan A′ was executed and closed (2026-08-01).** Model-drafted, human-verified, staged data curation produced 9 QLoRA adapters (3 conditions × 3 seeds). The branch **failed its predeclared gate**, and a causal counterfactual established the reason: correct entity labels are read by the model but are knowledge-inert (Key Finding 13). The design rationale still holds — task-specific curated data beats generic instruction data at small N ("Revisiting Superficial Alignment Hypothesis," arXiv:2410.03717), and tuning on how a model organizes facts it already knows is safer than injecting new facts (Gekhman et al., EMNLP 2024) — but the failure was not in the data, it was in the assumption that entity structure alone delivers knowledge.

**Work then moved from model structure to knowledge delivery,** producing diagnostics D1–D6 (see *Method: EFBPT and the Knowledge-Delivery Diagnostics* above). Two-pass fact extraction is the positive result: **+16.91pp at the oracle-linking level (p=0.0290)**, verified free of invented facts. End-to-end with no gold information reaches 64.79% against a 59.15% floor and a 76.06% ceiling, recovering a third of the gap but not significant at n=71. The remaining blocker is step 2, Urdu span → canonical English title, where failure is wrong-entity selection rather than string matching.

---

## Method: EFBPT and the Knowledge-Delivery Diagnostics (D1–D6)

Entity-Faithful Bilingual Plan Tuning (EFBPT) was built to repair the entity-corruption failure confirmed by the c-probe. Plan A′ (QLoRA, 9 adapters, 3 conditions × 3 seeds) **failed its predeclared gate**. Rather than abandon the branch, a causal counterfactual was run to establish *why* — the result is Key Finding 13 above.

That result redirected the work from model structure to knowledge delivery. Six diagnostics followed, each declared in `docs/EFBPT_PLAN_A_FREEZE.md` — arms, decoding, and how every possible outcome would be read — **before** execution. No section was amended after a number existed.

| | Question | Result | Verdict |
|---|---|---|---|
| **D1** | Do clean English facts help? | 83.10% vs 59.15%, p=0.0023 | Facts work |
| **D2** | Can embedding retrieval find them? | 23.15% title recall among pages that exist | Retrieval fails |
| **D3** | Do whole correct Wikipedia pages work? | 69.01% vs 59.15%, p=0.2295 | Raw pages fail |
| — | Can the model generate the facts itself? | 70.0% = 70.0% no-facts | Self-generation fails |
| **D4** | Can the model **extract** atomic facts from a correct article? | 76.06% vs 59.15%, p=0.0290 | **Extraction works** |
| **D5** | Does *balanced* extraction beat free allocation? | 67.61% vs 76.06%, p=0.2632 | Balance is not the lever |
| **D6** | Does the pipeline work end-to-end with **no gold information**? | 64.79% vs 59.15%, p=0.4807 | Partial, insufficient |

**D4 (extraction) is the positive result.** It recovers 71% of the gap between the no-facts floor (59.15%) and the clean-gold-fact ceiling (83.10%), with the paired test significant. A mandatory hand-check of 60 extracted facts against their source passages found **no invented facts**: 54/60 had zero content words absent from the source, and every named entity, number and date was present in the supplied text. Reported limitation: dropping the three unparsed baseline rows instead of scoring them incorrect shifts p to 0.0931 — the effect size survives (+13.24pp) but significance does not, so both p-values are reported.

**D5 rules out extraction *form* as a lever.** Forcing 2 facts per title cut rows with a zero-fact title from 39.7% to 15.5% and raised slot coverage from 85.0% to 95.2% — the manipulation demonstrably worked — yet accuracy fell 8.45pp. Against a fact-count-matched control the difference was exactly nil (b=7, c=7, p=1.0000). This also killed an observational signal that had looked real before the controlled test (rows with a zero-fact title were wrong 39.1% vs 17.1%, Fisher p=0.0742), most plausibly confounded by question difficulty.

**D6 measures the pipeline with no gold at any stage.** The model names its own entities from the Urdu question, matches them to the corpus by exact title lookup, extracts its own facts, and answers. Result: **64.79%**, between the 59.15% floor and the 76.06% oracle-linking ceiling, recovering 33% of the gap, not significant at n=71. The bottleneck is entity generation, not string matching: 57.3% of generated titles exist in the corpus and only 21.0% overlap the gold set, with failures splitting into transliteration drift (والکری → "Walcott"; ابیسل میدان → "Aylesbury Vale"), wrong-entity selection (goats → "Sheep"), and non-canonical surface forms ("Mustache" for *Moustache*). Fact extraction stayed correct even under bad input, once writing "Goats are not mentioned in the passages."

A fuzzy-title-matching follow-up was **considered and rejected on arithmetic before running**: it could address only the 5 rows where nothing matched, and even a perfect fuzzy matcher reaches p=0.0931 at n=71. The subset cannot grow — 71 is every DEV200 question whose gold titles all exist in the corpus.

**Component status.** Steps 1, 3 and 5 are proven; step 4 is validated at the oracle-linking level; step 2 (Urdu span → canonical English title) remains the open problem.

| Step | Operation | Status |
|---|---|---|
| 1 | Urdu question → Urdu entity spans | Proven — 97.89% verbatim (418/427) |
| 2 | spans → canonical English titles | **Open** — 57.3% corpus-matchable, 21.0% gold overlap |
| 3 | title → article | Proven — deterministic lookup |
| 4 | article → atomic English facts | Validated by D4; robust in D5; correct under bad input in D6 |
| 5 | facts + Urdu question → answer | Proven — p=0.0023 |

**Corpus defect.** The `AI-ModelScope/wikipedia 20231101.en` mirror used throughout has all 41 shards present and 6,407,814 unique page ids, yet central articles are absent in both title and text — "Douglas Noel Adams" appears 0 times across 23,963,971 chunks. Gold-evidence title coverage is 70.47%, which structurally caps every retrieval-based variant. `title_space_cache.txt` derived from the same source is likewise unreliable and is not used for validation.

## Repository Structure

```
URBench/
├── eval/
│   ├── eval_*_zero_shot_vllm.py       # Zero-shot evaluation
│   ├── eval_*_three_shot_vllm.py      # 3-shot evaluation
│   ├── eval_*_cot_vllm.py             # CoT evaluation
│   ├── eval_*_cot_alif.py / _qalb.py  # Urdu-specialized models
│   ├── eval_*_cot_p2/p3_*.py          # Prompt-sensitivity variants
│   ├── sdfr_*.py                      # SDFR-UR method
│   ├── error_analysis_tests/          # Fair-regime runs, probes, diagnostics
│   │   ├── cot_*_baseline_fair.py     # Fair-regime baselines
│   │   ├── sdfr_*_fair.py             # Fair-regime method runs
│   │   ├── integrate_official_strategyqa_v3.py
│   │   ├── phase_r_generate_dev50_v3.py
│   │   └── entity_first_ground_audit_v2.py
│   ├── xlt_exploratory/               # XLT experiments
│   └── sbatch_scripts/                # SLURM submission
├── rag/
│   ├── build_index_full.py            # Full-Wikipedia FAISS index build
│   ├── merge_shards.py                # Shard merge
│   └── retrieve.py                    # Retrieval helper (byte-offset meta seek)
├── prompts/                           # Per-dataset prompt variants (P1/P2/P3)
├── data/                              # Datasets, splits, official annotations
├── configs/ · scripts/                # Configs and utilities
├── experiments.md                     # Full experiment log
└── README.md
```

---

## Reproducing Results

```bash
pip install -r requirements.txt

# Baseline evaluation
python eval/eval_boolq_cot_vllm.py

# SDFR-UR
python eval/sdfr_prepare_splits.py     # 1. prepare retrieval pools
python eval/sdfr_build_indexes.py      # 2. build FAISS indexes (CPU, ~10 min)
sbatch eval/sbatch_scripts/sdfr_gsm8k.sh

# Fair-regime comparison (matched parameters)
python eval/error_analysis_tests/cot_gsm8k_baseline_fair.py
python eval/error_analysis_tests/sdfr_gsm8k_fair.py
```

Update `MODEL_NAME` and `OUTPUT_DIR` at the top of each script for your target model. All scripts use vLLM for batch inference.

### Hardware Requirements

| Workload | Min VRAM | Notes |
|---|---|---|
| 1.5B – 9B inference | 20 GB | A4090 / A6000 |
| 14B (AWQ) | 24 GB | A4090 / L20 |
| 14B (full) / 70B (AWQ INT4) | 40 GB | L20 |
| Full-Wikipedia retrieval | — | ~60 GB system RAM (index held on CPU) |

---

## Experiment Log

All runs — including failed and rejected branches — are logged in [`experiments.md`](experiments.md) with per-model, per-dataset, per-prompt entries covering accuracy, coverage, anomalies, confound audits, and research notes.

---

## Author

**Hassan Ahmad**
Master's Student — Computer Science & Technology (AI/NLP)
Harbin Institute of Technology, Shenzhen (HITSZ)
Student ID: 24SF51030
Supervisor: Prof. Bai XueFeng

---

## Citation

```bibtex
@mastersthesis{ahmad2026urbench,
  title     = {URBench: Benchmarking and Improving LLM Reasoning in Urdu},
  author    = {Hassan Ahmad},
  school    = {Harbin Institute of Technology, Shenzhen},
  year      = {2026}
}
```

---

*Active research — results and experiments updated continuously.*