#!/bin/bash
#SBATCH --job-name=d10_teacher
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/d10_teacher_%j.log
#SBATCH --error=logs/d10_teacher_%j.log
#
# D10 Job T — teacher facts + teacher direct (docs/EFBPT_D10_TEACHER_FREEZE.md §3).
# Partition, GPU, CPU and memory copied from d8_reachable_ceiling.sbatch.
# vLLM 0.13.0 lives in env urbench_eval (docs/handoff/TEACHER_INVENTORY.md §5).
#
# Usage (from ~/URBench), one argument, TEST or FULL:
#   bash eval/error_analysis_tests/efbpt/d10_teacher.sbatch TEST
#   bash eval/error_analysis_tests/efbpt/d10_teacher.sbatch FULL
#
# Run outside SLURM, the script submits itself with the mode in the job name
# and the log at logs/d10_teacher_{TEST,FULL}_%j.log (TEST 01:00:00,
# FULL 04:00:00, freeze §8).
#
# TEST = --limit 5 (writes only under outputs/efbpt/d10/_test/teacher/).
# TEST-5 must pass before FULL is submitted (freeze §8).

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  TEST|FULL) ;;
  *) echo "usage: $0 TEST|FULL" >&2; exit 2 ;;
esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=01:00:00; else LIMIT=04:00:00; fi
  exec sbatch --job-name="d10_teacher_${MODE}" --time="$LIMIT" \
       --output="logs/d10_teacher_${MODE}_%j.log" \
       --error="logs/d10_teacher_${MODE}_%j.log" \
       "$0" "$MODE"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "TEST" ]; then
  python eval/error_analysis_tests/efbpt/d10_teacher.py --limit 5
else
  python eval/error_analysis_tests/efbpt/d10_teacher.py
fi

echo "=== end: $(date) ==="
