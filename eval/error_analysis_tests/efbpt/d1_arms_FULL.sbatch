#!/bin/bash
#SBATCH --job-name=d1_arms_FULL
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/d1_arms_FULL_%j.log
#SBATCH --error=logs/d1_arms_FULL_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== start: $(date) ==="

# 6 arms x 200 rows, base model only, no adapters.
python eval/error_analysis_tests/efbpt/d1_eval_arms.py --batch-size 16

echo "=== end: $(date) ==="
