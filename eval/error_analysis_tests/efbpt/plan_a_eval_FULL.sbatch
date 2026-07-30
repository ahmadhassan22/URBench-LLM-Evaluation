#!/bin/bash
#SBATCH --job-name=efbpt_eval_FULL
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/efbpt_eval_FULL_%j.log
#SBATCH --error=logs/efbpt_eval_FULL_%j.log

set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== start: $(date) ==="

# Full DEV200: C0 + C1/C2/C3 x seeds 13/42/2026 = 10 configurations x 200 rows.
# batch-size 16: 4-bit weights are ~9GB of the 44GB card, so there is ample room.
python eval/error_analysis_tests/efbpt/efbpt_eval_dev200.py --batch-size 16

echo "=== end: $(date) ==="
