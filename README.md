# URBench: An Urdu Reasoning Benchmark for Large Language Models

## 1. Overview

URBench is a research benchmark designed to evaluate the reasoning capabilities of Large Language Models (LLMs) in the Urdu language.  

It measures model performance across multiple reasoning tasks translated and adapted into Urdu, enabling systematic evaluation of multilingual reasoning ability.

The benchmark currently includes:

- BoolQ (Yes/No Question Answering)
- CSQA (Commonsense Reasoning)
- PIQA (Physical Commonsense Reasoning)
- StrategyQA (Multi-hop Reasoning)
- GSM8K (Mathematical Word Problems)

URBench supports:
- Zero-shot prompting
- Few-shot prompting
- Chain-of-Thought (CoT) prompting


## 2. Repository Structure
URBench/
│
├── data/ # Raw and processed Urdu reasoning datasets
├── prompts/ # Prompt templates (zero-shot, few-shot, CoT)
├── scripts/ # Evaluation and data preparation scripts
├── eval/ # Metrics and evaluation utilities
├── outputs/ # Model generations and evaluation results
├── notebooks/ # Analysis and exploration notebooks
└── experiments.md # Experiment tracking

## 3. Objective

The goal of URBench is to:

- Evaluate reasoning performance of LLMs in Urdu.
- Compare zero-shot, few-shot, and CoT prompting strategies.
- Analyse weaknesses in multilingual reasoning.
- Provide structured and reproducible evaluation pipelines.


## 4. Evaluation Pipeline

The evaluation pipeline:

1. Loads the selected Urdu reasoning dataset.
2. Applies the chosen prompting strategy.
3. Generates model outputs.
4. Computes evaluation metrics (e.g., Accuracy, Exact Match).
5. Stores results in a structured format under `outputs/`.


## 5. Running Experiments

Example command:

```bash
python scripts/run_eval.py --config configs/example.yaml

## 6. Reproducibility

Experiments are tracked using:
Structured output directories
Config snapshots
Version-controlled scripts
This ensures consistent and reproducible evaluation across models.

7. Author

Ahmad Hassan
Master’s Student – AI & NLP
Harbin Institute of Technology, Shenzhen

Thesis Repository (Private)
