#!/bin/bash
#SBATCH --job-name=d1_trans_FULL
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/d1_trans_FULL_%j.log
#SBATCH --error=logs/d1_trans_FULL_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d1_translate_facts.py --batch-size 16

echo "=== end: $(date) ==="
