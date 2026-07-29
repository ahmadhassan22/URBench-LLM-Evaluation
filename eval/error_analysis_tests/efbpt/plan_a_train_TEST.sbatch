#!/bin/bash
#SBATCH --job-name=efbpt_train_TEST
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/efbpt_train_TEST_%j.log
#SBATCH --error=logs/efbpt_train_TEST_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/efbpt_train_qlora.py \
    --condition C3 \
    --seed 13 \
    --test \
    --test-steps 6

echo "=== end: $(date) ==="
