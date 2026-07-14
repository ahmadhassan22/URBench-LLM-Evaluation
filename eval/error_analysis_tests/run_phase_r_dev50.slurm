#!/bin/bash
#SBATCH --job-name=phaseR_dev50
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/phaseR_dev50_%j.out
#SBATCH --error=/mnt/home/user41/URBench/logs/phaseR_dev50_%j.err

# Phase-R DEV50 retrieval diagnostic (R1-R4). DEV50 only. No answer generation.
# 120G RAM: holds the 37G flat index in CPU RAM + headroom. Embedder on CPU,
# vLLM (0.72) + reranker share the GPU.

set -euo pipefail
cd /mnt/home/user41/URBench

# Initialize conda for a non-interactive batch shell (`conda activate` fails
# otherwise: "Run 'conda init' before 'conda activate'"). Source conda.sh directly.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate urbench_eval

echo "node: $(hostname)  date: $(date)"
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python eval/error_analysis_tests/phase_r_dev50.py --device_embed cpu

echo "exit: $?  date: $(date)"
