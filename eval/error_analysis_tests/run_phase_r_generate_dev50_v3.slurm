#!/bin/bash
#SBATCH --job-name=phaseR_gen_v3
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/phaseR_gen_v3_%j.out
#SBATCH --error=/mnt/home/user41/URBench/logs/phaseR_gen_v3_%j.err

set -euo pipefail
cd /mnt/home/user41/URBench
export PYTHONUNBUFFERED=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate urbench_eval

echo "node: $(hostname)  date: $(date)"
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python eval/error_analysis_tests/phase_r_generate_dev50_v3.py --stage self-test

# Live generation: question-only input. Thinking is enabled for planning/review.
python eval/error_analysis_tests/phase_r_generate_dev50_v3.py \
  --stage generate \
  --batch-size 5 \
  --gpu-memory-utilization 0.82

# Separate CPU-only process: official annotations are used only for audit.
# The Qwen/vLLM process has exited before this command begins.
python eval/error_analysis_tests/phase_r_generate_dev50_v3.py --stage audit

echo "v3 generation and offline audit completed: $(date)"
