#!/bin/bash
#SBATCH --job-name=d2_recall_TEST
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d2_recall_TEST_%j.log
#SBATCH --error=logs/d2_recall_TEST_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
free -h
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d2_retrieve_and_recall.py --test --test-rows 20

echo "=== end: $(date) ==="
