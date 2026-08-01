#!/bin/bash
#SBATCH --job-name=d2_coverage
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d2_coverage_%j.log
#SBATCH --error=logs/d2_coverage_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
echo "=== start: $(date) ==="

# Pure file scan. No index, no embeddings, no model.
python eval/error_analysis_tests/efbpt/d2_title_coverage.py

echo "=== end: $(date) ==="
