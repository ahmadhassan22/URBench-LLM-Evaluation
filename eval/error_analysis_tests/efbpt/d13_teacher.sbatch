#!/bin/bash
#SBATCH --job-name=d13_teacher
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d13_teacher_%j.log
#SBATCH --error=logs/d13_teacher_%j.log
#
# D13 Group 1 teacher arms (docs/EFBPT_D13_BASELINES_TRANSFER_FREEZE.md §2, §7.1).
# Llama-3.1-70B-Instruct-AWQ-INT4, vLLM 0.13.0 in env urbench_eval, engine as D10 §2.
#
# Usage (from ~/URBench):
#   sbatch [--time=..] eval/error_analysis_tests/efbpt/d13_teacher.sbatch TEST
#   sbatch [--time=..] eval/error_analysis_tests/efbpt/d13_teacher.sbatch FULL dev200|eval458 [ARM[,ARM]]
# TEST = DEV200 rows 0-4 of the three teacher arms under outputs/efbpt/d13/_test/ (never a result).
set -euo pipefail
MODE="${1:-}"
SET="${2:-dev200}"
ARMS="${3:-}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST | FULL dev200|eval458 [arms]" >&2; exit 2 ;; esac
case "$SET" in dev200|eval458) ;; *) echo "set must be dev200 or eval458" >&2; exit 2 ;; esac
cd /mnt/home/user41/URBench

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   set: $SET   arms: ${ARMS:-default}   job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
echo "=== start: $(date) ==="

EXTRA=()
if [ -n "$ARMS" ]; then EXTRA=(--arms "$ARMS"); fi
python eval/error_analysis_tests/efbpt/d13_teacher.py --mode "$MODE" --set "$SET" "${EXTRA[@]}"

echo "=== end: $(date) ==="
