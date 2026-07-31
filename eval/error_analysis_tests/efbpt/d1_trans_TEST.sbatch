#!/bin/bash
#SBATCH --job-name=d1_trans_TEST
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/d1_trans_TEST_%j.log
#SBATCH --error=logs/d1_trans_TEST_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d1_translate_facts.py \
    --test --test-items 12 --batch-size 12

echo "=== end: $(date) ==="
