#!/bin/bash
#SBATCH --job-name=entityfirst12
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/entityfirst12_%j.out
#SBATCH --error=/mnt/home/user41/URBench/logs/entityfirst12_%j.err

set -euo pipefail
cd /mnt/home/user41/URBench
export PYTHONUNBUFFERED=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate urbench_eval

echo "node: $(hostname)  date: $(date)"
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python eval/error_analysis_tests/entity_first_probe12_v1.py --stage self-test

# Process 1: Qwen candidate generation. It exits before the full index loads.
python eval/error_analysis_tests/entity_first_probe12_v1.py \
  --stage generate \
  --batch-size 6 \
  --gpu-memory-utilization 0.82

# Process 2: full-index title discovery. Qwen GPU memory is already released.
# The MiniLM query embedder remains on CPU; the 35 GB FAISS index stays in RAM.
python eval/error_analysis_tests/entity_first_probe12_v1.py \
  --stage ground \
  --device-embed cpu

# Process 3: offline audit; official fields are scorer-only.
python eval/error_analysis_tests/entity_first_probe12_v1.py --stage audit

echo "entity-first fixed-12 probe completed: $(date)"
