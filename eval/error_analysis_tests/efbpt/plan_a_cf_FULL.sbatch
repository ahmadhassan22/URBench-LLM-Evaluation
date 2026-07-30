#!/bin/bash
#SBATCH --job-name=efbpt_cf_FULL
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/efbpt_cf_FULL_%j.log
#SBATCH --error=logs/efbpt_cf_FULL_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== start: $(date) ==="

# 3 seeds x 2 arms x ~197 rows. Arm A already exists from the main evaluation.
python eval/error_analysis_tests/efbpt/efbpt_counterfactual_entities.py \
    --batch-size 16

echo "=== end: $(date) ==="
