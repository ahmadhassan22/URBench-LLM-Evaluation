#!/bin/bash
#SBATCH --job-name=phaseR_ret_v2
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/phaseR_ret_v2_%j.out
#SBATCH --error=/mnt/home/user41/URBench/logs/phaseR_ret_v2_%j.err

set -euo pipefail
cd /mnt/home/user41/URBench
export PYTHONUNBUFFERED=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate urbench_eval

echo "node: $(hostname)  date: $(date)"
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# This job is submitted only after generation_summary.json is reviewed.
# Qwen is not loaded here; GPU memory is reserved for the reranker.
python eval/error_analysis_tests/phase_r_dev50_v2.py \
  --stage retrieve \
  --device-embed cpu \
  --ground-min 0.60

python eval/error_analysis_tests/phase_r_dev50_v2.py --stage score

echo "retrieval and offline scoring completed: $(date)"
