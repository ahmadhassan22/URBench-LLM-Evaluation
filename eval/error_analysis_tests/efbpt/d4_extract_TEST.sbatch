#!/bin/bash
#SBATCH --job-name=d4_extract_TEST
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/d4_extract_TEST_%j.log
#SBATCH --error=logs/d4_extract_TEST_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d4_extract_facts.py \
    --test --test-rows 8 --batch-size 8

echo "=== end: $(date) ==="
