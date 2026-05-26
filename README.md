# URBench: Urdu Reasoning Benchmark for Large Language Models

> **Master's Thesis Research** · Harbin Institute of Technology, Shenzhen  
> A systematic evaluation of LLM reasoning in Urdu — a low-resource language spoken by 230M+ people.

---

## Overview

URBench provides the most comprehensive cross-architecture comparison of Urdu reasoning performance to date. It evaluates **9 general-purpose multilingual LLMs** and **2 Urdu-specialized models** across **5 reasoning datasets** using **3 prompting strategies**, with additional experiments on retrieval-augmented generation (RAG) and cross-lingual prompting (XLT).

The benchmark addresses a critical gap: while Urdu has a large global speaker base, structured reasoning evaluation in Urdu remains severely underexplored compared to high-resource languages.

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

**Key finding:** Urdu-specific instruction tuning (Alif) and continued pre-training (Qalb) help significantly on arithmetic reasoning (GSM8K) but do not consistently improve — and sometimes hurt — multi-hop and commonsense reasoning compared to the general-purpose base model.

---

## Additional Experiments

### RAG (Retrieval-Augmented Generation)

Evaluated on StrategyQA using Qwen3-14B with a FAISS dense retrieval index over English Wikipedia (Nov 2023, 6.4M articles, 23.9M chunks).

| Setting | Accuracy |
|---|---|
| Zero-shot baseline (no RAG) | 83.89% |
| RAG with retrieved context | 56.24% |
| **Δ** | **−27.65pp** |

RAG reduced accuracy by 27.65 percentage points. Error analysis identified three failure categories: entity disambiguation failures (e.g. "The Police" band vs. law enforcement), partial retrieval for multi-hop questions, and coverage gaps in the filtered corpus.

### XLT (Cross-Lingual Thought Prompting)

Evaluated on StrategyQA using Qwen3-14B. XLT instructs the model to restate the question in English, reason step-by-step in English, then answer in Urdu.

| Version | Coverage | Accuracy |
|---|---|---|
| XLT v1 (initial, broken) | 74.72% | 42.31% |
| XLT v2 (fixed prompt + tokens) | 96.42% | 58.43% |
| CoT baseline | 100% | 83.89% |

XLT consistently underperformed CoT. Analysis revealed two failure modes: (1) token truncation cutting off the answer before generation, and (2) factual hallucination during the English restatement step on implicit multi-hop questions. XLT was discontinued in favor of baseline CoT for all subsequent evaluations.

---

## Key Findings

**1. General multilingual models outperform Urdu-specialized models on most tasks.**  
Qwen3-14B and LLaMA-3.1-70B consistently outperform Alif and Qalb despite having no Urdu-specific training, suggesting that strong general instruction tuning transfers more effectively to Urdu reasoning than targeted Urdu fine-tuning.

**2. CoT is essential for math, but harmful for binary tasks in some architectures.**  
GSM8K CoT dramatically outperforms zero-shot and 3-shot across all models. However, CoT hurts binary task performance (BoolQ, PIQA, StrategyQA) for LLaMA and Gemma architectures due to answer extraction failures after long reasoning chains.

**3. Scale matters for Urdu reasoning.**  
LLaMA-3.1-70B-AWQ sets highs on BoolQ (88.49%), CSQA zero-shot (65.60%), PIQA 3-shot (76.40%), and StrategyQA 3-shot (90.09%) — the only model to break 90% on any dataset.

**4. Qwen3 leads on mathematical reasoning.**  
Qwen3-14B achieves 83.71% GSM8K CoT — the highest math score. Qwen3-8B achieves 79.86%. Both substantially exceed all other architectures, including LLaMA-70B (55.00%).

**5. RAG degrades reasoning in low-resource settings.**  
Retrieval-augmented generation reduced StrategyQA accuracy by 27.65 points. The failure is primarily retrieval quality, not model reasoning — confirming that Urdu NLP infrastructure (entity disambiguation, dense retrieval) is a critical bottleneck.

**6. PIQA is highly prompt-sensitive.**  
PIQA shows the highest variance across prompt types of any dataset, with individual models swinging 15–20 percentage points between zero-shot and CoT. This confirms physical commonsense reasoning in Urdu is particularly sensitive to prompt format.

**7. DeepSeek-R1-Distill underperforms its size category.**  
Despite being a reasoning-distilled model, DeepSeek-R1-Distill-Qwen-7B underperforms similarly-sized instruction-tuned models on all Urdu tasks, suggesting reasoning distillation does not transfer effectively to low-resource language settings.

---

## Ongoing Work

The benchmark evaluation phase is complete. Current research focus is on **method design** — proposing and evaluating a novel approach to improve Urdu reasoning performance beyond the baselines established here.

Directions under investigation:
- Analysis of what makes general multilingual models outperform Urdu-specialized models
- Identifying the conditions under which prompting strategies (CoT, 3-shot) consistently help vs. hurt across model architectures
- Proposal of a method that leverages insights from the prompt sensitivity analysis and RAG failure analysis to improve reasoning accuracy

Results and analysis will be documented in [`experiments.md`](experiments.md) as work progresses.

---

## Repository Structure

```
URBench/
├── eval/
│   ├── eval_*_zero_shot_vllm.py      # Zero-shot evaluation scripts
│   ├── eval_*_three_shot_vllm.py     # 3-shot evaluation scripts
│   ├── eval_*_cot_vllm.py            # CoT evaluation scripts
│   ├── eval_*_cot_alif.py            # Alif model evaluation
│   ├── eval_*_cot_qalb.py            # Qalb model evaluation
│   ├── xlt_exploratory/              # XLT experiment scripts
│   ├── sbatch_scripts/               # SLURM job submission scripts
│   └── logs_archive/                 # Run logs
├── prompts/
│   ├── boolq/                        # cot_p1.txt, zero_shot_p1.txt, etc.
│   ├── csqa/
│   ├── piqa/
│   ├── strategyqa/
│   └── gsm8k/
├── rag/                              # RAG pipeline scripts and index
├── configs/                          # Model and evaluation configs
├── scripts/                          # Utility scripts
├── experiments.md                    # Full experiment log
├── requirements.txt
└── README.md
```

---

## Reproducing Results

```bash
pip install -r requirements.txt
cd eval/
python eval_boolq_cot_vllm.py
```

Update `MODEL_NAME` and `OUTPUT_DIR` at the top of each script for your target model. All scripts use vLLM for efficient batch inference.

### Hardware Requirements

| Model Size | Min VRAM | Recommended |
|---|---|---|
| 1.5B – 9B | 20 GB | A4090 / A6000 |
| 14B (AWQ) | 24 GB | A4090 / L20 |
| 14B (full) | 40 GB | L20 |
| 70B (AWQ INT4) | 40 GB | L20 |

---

## Experiment Log

All results are logged in [`experiments.md`](experiments.md) with per-model, per-dataset, per-prompt-type entries including accuracy, coverage, anomalies, and research notes.

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