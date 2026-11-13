#!/bin/bash
#SBATCH --job-name=d13_retrieve
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/d13_retrieve_%j.log
#SBATCH --error=logs/d13_retrieve_%j.log
#
# D13 Group 2 retrieval step (docs/EFBPT_D13_BASELINES_TRANSFER_FREEZE.md §2, §7.3).
# rag/retrieve.py unchanged: the flat index needs about 37 GB of RAM, the embedder runs on the GPU.
#
# Usage (from ~/URBench):
#   sbatch eval/error_analysis_tests/efbpt/d13_retrieve.sbatch TEST
#   sbatch eval/error_analysis_tests/efbpt/d13_retrieve.sbatch FULL [dev200,eval458]
set -euo pipefail
MODE="${1:-}"
SETS="${2:-dev200,eval458}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST | FULL [sets]" >&2; exit 2 ;; esac
cd /mnt/home/user41/URBench

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   sets: $SETS   job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
free -g | head -2
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d13_retrieve.py --mode "$MODE" --sets "$SETS"

echo "=== end: $(date) ==="
