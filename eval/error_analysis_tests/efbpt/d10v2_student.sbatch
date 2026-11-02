#!/bin/bash
#SBATCH --job-name=d10v2_student
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d10v2_student_%j.log
#SBATCH --error=logs/d10v2_student_%j.log
#
# D10 v2 Job S — student arms A_r2 / TF / TF_wrong / TF_noq, batch size 1 (Amendment 3)
# (docs/EFBPT_D10_TEACHER_FREEZE.md §4, docs/EFBPT_D10_AMENDMENT_3.md A3.2). Requires the sealed v2 Job T outputs
# of the same mode (outputs/efbpt/d10v2/{_test/,}teacher/T_SEAL.json) and the sealed v1
# t_direct.jsonl / t_direct_f.jsonl / derangement.json under outputs/efbpt/d10/teacher/ (pinned MD5s).
# Partition, GPU, CPU and memory copied from d8_reachable_ceiling.sbatch.
#
# Usage (from ~/URBench), one argument, TEST or FULL:
#   bash eval/error_analysis_tests/efbpt/d10v2_student.sbatch TEST
#   bash eval/error_analysis_tests/efbpt/d10v2_student.sbatch FULL
#
# Run outside SLURM, the script submits itself with the mode in the job name
# and the log at logs/d10v2_student_{TEST,FULL}_%j.log (TEST 02:00:00,
# FULL 12:00:00, Amendment 2 A2.2 / Amendment 3 A3.2).
#
# TEST = --limit 5 (writes only under outputs/efbpt/d10v2/_test/student/).
# TEST-5 must pass before FULL is submitted (freeze §8).

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  TEST|FULL) ;;
  *) echo "usage: $0 TEST|FULL" >&2; exit 2 ;;
esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=02:00:00; else LIMIT=12:00:00; fi
  exec sbatch --job-name="d10v2_student_${MODE}" --time="$LIMIT" \
       --output="logs/d10v2_student_${MODE}_%j.log" \
       --error="logs/d10v2_student_${MODE}_%j.log" \
       "$0" "$MODE"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "TEST" ]; then
  python eval/error_analysis_tests/efbpt/d10v2_student.py --limit 5
else
  python eval/error_analysis_tests/efbpt/d10v2_student.py
fi

echo "=== end: $(date) ==="
