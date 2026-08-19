#!/bin/bash
#SBATCH --job-name=l0_diag
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/l0_diag_%j.log
#SBATCH --error=logs/l0_diag_%j.log

set -euo pipefail
source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval
cd /mnt/home/user41/URBench
echo "=== start: $(date) ==="
python -u eval/error_analysis_tests/efbpt/l0_diag_encoder.py
echo "=== end: $(date) ==="
