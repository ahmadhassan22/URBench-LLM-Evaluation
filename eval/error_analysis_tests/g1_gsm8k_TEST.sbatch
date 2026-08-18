#!/bin/bash
#SBATCH --job-name=g1_gsm8k_TEST
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/g1_gsm8k_TEST_%j.log
#SBATCH --error=logs/g1_gsm8k_TEST_%j.log

set -euo pipefail
source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval
cd /mnt/home/user41/URBench
echo "=== start: $(date) ==="
python -u eval/error_analysis_tests/g1_sdfr_gsm8k_clean.py --test
echo "=== end: $(date) ==="
