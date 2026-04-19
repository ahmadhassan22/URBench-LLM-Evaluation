# URBench: An Urdu Reasoning Benchmark for Large Language Models

> **Master's Thesis Research** | Harbin Institute of Technology, Shenzhen  
> Evaluating LLM reasoning capabilities in Urdu across multiple tasks and prompting strategies.

---

## Overview

URBench is a systematic evaluation benchmark for assessing the reasoning capabilities of Large Language Models (LLMs) in **Urdu** — a low-resource language spoken by over 200 million people worldwide.

The benchmark evaluates **9 open-source LLMs** across **5 reasoning datasets** using **3 prompting strategies**, providing the most comprehensive cross-architecture comparison of Urdu reasoning performance to date.

---

## Datasets

| Dataset | Task Type | Examples | Format |
|---------|-----------|----------|--------|
| **BoolQ** | Binary Reading Comprehension | 1,550 | ہاں / نہیں |
| **CSQA** | Commonsense MCQ (5-way) | 1,500 | A / B / C / D / E |
| **PIQA** | Physical Commonsense | 750 | 0 / 1 |
| **StrategyQA** | Multi-hop Binary Reasoning | 2,290 | ہاں / نہیں |
| **GSM8K** | Arithmetic Word Problems | 700 | Numeric |

All datasets are translated and adapted into Urdu with normalization scripts handling Urdu-specific script variations (diacritics, Eastern/Western numerals, code-switching).

---

## Models Evaluated

| Model | Family | Size | Quantization |
|-------|--------|------|-------------|
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 1.5B | — |
| Qwen2.5-7B-Instruct | Qwen2.5 | 7B | — |
| Qwen2.5-14B-Instruct | Qwen2.5 | 14B | AWQ |
| Qwen3-8B | Qwen3 | 8B | — |
| Qwen3-14B | Qwen3 | 14B | — |
| Meta-LLaMA-3.1-8B-Instruct | LLaMA | 8B | — |
| Meta-LLaMA-3.1-70B-Instruct | LLaMA | 70B | AWQ INT4 |
| DeepSeek-R1-Distill-Qwen-7B | DeepSeek | 7B | — |
| Gemma-2-9B-IT | Gemma | 9B | — |

---

## Prompting Strategies

- **Zero-shot** — Direct question answering with no examples
- **3-shot** — Three in-context demonstrations
- **Chain-of-Thought (CoT)** — Step-by-step reasoning before final answer

---

## Results

### BoolQ (Binary Reading Comprehension)

| Model | Zero-shot | 3-shot | CoT |
|-------|-----------|--------|-----|
| Qwen2.5-1.5B | 62.69% | 55.24% | 47.00% |
| Qwen2.5-7B | 66.73% | 71.41% | 62.26% |
| Qwen2.5-14B-AWQ | 76.26% | 72.24% | 76.13% |
| Qwen3-8B | 79.68% | 81.52% | 80.84% |
| Qwen3-14B | 84.51% | 84.89% | 84.47% |
| Meta-LLaMA-3.1-8B | 78.49% | 66.82% | 71.40% |
| **Meta-LLaMA-3.1-70B-AWQ** | 87.51% | **88.49%** | 87.45% |
| DeepSeek-R1-Distill-Qwen-7B | 56.76% | 62.07% | 61.31% |
| Gemma-2-9B-IT | 86.30% | 87.20% | 86.10% |

### CSQA (Commonsense QA)

| Model | Zero-shot | 3-shot | CoT |
|-------|-----------|--------|-----|
| Qwen2.5-1.5B | 6.07% | 9.53% | 25.20% |
| Qwen2.5-7B | 45.40% | 44.40% | 45.13% |
| Qwen2.5-14B-AWQ | 50.93% | 46.53% | 49.73% |
| Qwen3-8B | 54.80% | 52.20% | 55.33% |
| Qwen3-14B | 58.73% | 56.60% | **63.00%** |
| Meta-LLaMA-3.1-8B | 49.20% | 48.13% | 46.47% |
| **Meta-LLaMA-3.1-70B-AWQ** | **65.60%** | **64.87%** | 58.07% |
| DeepSeek-R1-Distill-Qwen-7B | 26.33% | 22.87% | 23.33% |
| Gemma-2-9B-IT | 59.87% | 59.20% | 51.87% |

### PIQA (Physical Commonsense)

| Model | Zero-shot | 3-shot | CoT |
|-------|-----------|--------|-----|
| Qwen2.5-1.5B | 33.73% | 51.07% | 51.07% |
| Qwen2.5-7B | 59.20% | 62.00% | 44.93% |
| Qwen2.5-14B-AWQ | 60.67% | 65.33% | 58.93% |
| Qwen3-8B | 56.53% | 55.87% | 55.20% |
| Qwen3-14B | 59.33% | 63.73% | 65.73% |
| Meta-LLaMA-3.1-8B | 35.20% | 55.87% | 49.47% |
| **Meta-LLaMA-3.1-70B-AWQ** | 62.93% | **76.40%** | 31.73% |
| DeepSeek-R1-Distill-Qwen-7B | 50.67% | 53.47% | 52.93% |
| Gemma-2-9B-IT | **68.67%** | 53.60% | 50.00% |

### StrategyQA (Multi-hop Reasoning)

| Model | Zero-shot | 3-shot | CoT |
|-------|-----------|--------|-----|
| Qwen2.5-1.5B | 53.28% | 48.38% | 49.91% |
| Qwen2.5-7B | 55.63% | 70.17% | 56.11% |
| Qwen2.5-14B-AWQ | 73.19% | 75.11% | 76.72% |
| Qwen3-8B | 79.96% | 80.74% | 79.74% |
| Qwen3-14B | 83.89% | 83.97% | 83.89% |
| Meta-LLaMA-3.1-8B | 57.55% | 78.43% | 78.17% |
| **Meta-LLaMA-3.1-70B-AWQ** | 88.69% | **90.09%** | 49.91% |
| DeepSeek-R1-Distill-Qwen-7B | 55.24% | 51.66% | 47.51% |
| Gemma-2-9B-IT | 82.79% | 80.70% | 80.61% |

### GSM8K (Arithmetic Reasoning)

| Model | Zero-shot | 3-shot | CoT |
|-------|-----------|--------|-----|
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

## Key Findings

**1. Scale matters for Urdu reasoning.** Meta-LLaMA-3.1-70B-AWQ sets new highs on BoolQ (88.49%), CSQA zero-shot (65.60%), PIQA 3-shot (76.40%), and StrategyQA (90.09% — first model to break 90%).

**2. CoT is essential for math but harmful for binary tasks.** Across all models, GSM8K CoT dramatically outperforms zero-shot and 3-shot. However, CoT consistently hurts performance on binary tasks (BoolQ, PIQA, StrategyQA) for LLaMA and Gemma architectures due to answer extraction failures.

**3. Qwen3 excels at mathematical reasoning.** Qwen3-14B achieves 83.71% GSM8K CoT — the highest math score across all models — despite being outperformed by LLaMA-70B on other tasks.

**4. Architecture matters more than size for some tasks.** Gemma-2-9B-IT achieves 68.67% PIQA zero-shot — the highest across all models including 70B — but collapses to 50% under CoT, revealing strong prompt sensitivity.

**5. PIQA is an outlier dataset.** Every model shows highly inconsistent performance across prompt types on PIQA-Urdu, suggesting physical commonsense reasoning is particularly sensitive to prompt format in low-resource settings.

**6. DeepSeek-R1-Distill underperforms expectations.** Despite being a reasoning-specialized model, DeepSeek-R1-Distill-Qwen-7B consistently underperforms similarly-sized instruction-tuned models on Urdu tasks, suggesting reasoning distillation does not transfer well to low-resource languages.

---

## Repository Structure

```
URBench/
├── eval/                    # Evaluation scripts (15 scripts, one per dataset × prompt type)
│   ├── eval_boolq_zero_shot_vllm.py
│   ├── eval_boolq_three_shot_vllm.py
│   ├── eval_boolq_cot_vllm.py
│   ├── eval_csqa_*.py
│   ├── eval_piqa_*.py
│   ├── eval_strategyqa_*.py
│   └── eval_gsm8k_*.py
├── prompts/                 # Prompt templates per dataset and strategy
│   ├── boolq/
│   ├── csqa/
│   ├── piqa/
│   ├── strategyqa/
│   └── gsm8k/
├── configs/                 # Example evaluation configs
├── scripts/                 # Helper scripts
├── assets/                  # Result visualizations
├── experiments.md           # Full experiment log with all results
├── requirements.txt         # Dependencies
└── README.md
```

---

## Reproducing Results

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `vllm`, `transformers`, `torch`

### Running Evaluation

```bash
cd eval/
python eval_boolq_zero_shot_vllm.py
```

Each script contains a `CONFIG` section at the top. Update `MODEL_NAME` and `OUTPUT_DIR` for your target model. All scripts use `vllm` for efficient inference.

### Hardware Requirements

| Model Size | Minimum VRAM | Recommended Node |
|------------|-------------|-----------------|
| 1.5B – 9B | 20GB | A4090 / A6000 |
| 14B (AWQ) | 24GB | A4090 |
| 70B (AWQ INT4) | 40GB | L20 / A6000 |

---

## Experiment Tracking

All results are logged in [`experiments.md`](experiments.md) with per-model, per-dataset, per-prompt-type entries including accuracy, coverage, and notes on anomalies.

---

## Author

**Hassan Ahmad**  
Master's Student — Computer Science & Technology (AI/NLP)  
Harbin Institute of Technology, Shenzhen (HITSZ)  
Student ID: 24SF51030  
Supervisor: Prof. Bai XueFeng

---

## Citation

If you use URBench in your research, please cite:

```bibtex
@mastersthesis{ahmad2026urbench,
  title     = {Benchmarking and Improving LLMs for Reasoning in Urdu using Chain-of-Thought and Retrieval-Augmented Generation},
  author    = {Hassan Ahmad},
  school    = {Harbin Institute of Technology, Shenzhen},
  year      = {2026}
}
```

---

*This repository is part of an active Master's thesis. Results are updated as new models are evaluated.*