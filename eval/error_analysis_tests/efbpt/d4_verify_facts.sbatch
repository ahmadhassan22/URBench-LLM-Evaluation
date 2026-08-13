#!/bin/bash
#SBATCH --job-name=d4_verify
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d4_verify_%j.log
#SBATCH --error=logs/d4_verify_%j.log

set -euo pipefail
source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval
cd /mnt/home/user41/URBench
echo "=== start: $(date) ==="
python -u eval/error_analysis_tests/efbpt/d4_verify_facts.py
echo "=== end: $(date) ==="
