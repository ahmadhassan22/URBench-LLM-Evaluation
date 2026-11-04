#!/bin/bash
#SBATCH --job-name=d11o_student
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/d11o_student_%j.log
#SBATCH --error=logs/d11o_student_%j.log
#
# D11-O Job S — student arms GF / TO1 / TO2 at batch size 1, sealed A_r2 reused after V2'
# (docs/EFBPT_D11O_ORACLE_READING_FREEZE.md §2, §4, §5). Requires the sealed D11-O Job T
# outputs of both versions and the same mode (outputs/efbpt/d11o/{_test/,}teacher_v{1,2}/T_SEAL.json)
# and the sealed D10 / D10v2 student files (pinned MD5s).
# Partition, GPU, CPU and memory as d10v2_student.sbatch.
#
# Usage (from ~/URBench), one argument, TEST or FULL:
#   bash eval/error_analysis_tests/efbpt/d11o_student.sbatch TEST
#   bash eval/error_analysis_tests/efbpt/d11o_student.sbatch FULL
# Optional: SBATCH_DEP=<jobid> adds --dependency=afterok:<jobid> (freeze §7 chain).
#
# Run outside SLURM, the script submits itself with the mode in the job name
# and the log at logs/d11o_student_{TEST,FULL}_%j.log (TEST 02:00:00,
# FULL 08:00:00, freeze §7).
#
# TEST = --limit 5 (writes only under outputs/efbpt/d11o/_test/student/).

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  TEST|FULL) ;;
  *) echo "usage: $0 TEST|FULL" >&2; exit 2 ;;
esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=02:00:00; else LIMIT=08:00:00; fi
  DEP=()
  if [ -n "${SBATCH_DEP:-}" ]; then DEP=(--dependency="afterok:${SBATCH_DEP}"); fi
  exec sbatch --job-name="d11o_student_${MODE}" --time="$LIMIT" \
       --output="logs/d11o_student_${MODE}_%j.log" \
       --error="logs/d11o_student_${MODE}_%j.log" \
       "${DEP[@]}" "$0" "$MODE"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "TEST" ]; then
  python eval/error_analysis_tests/efbpt/d11o_student.py --limit 5
else
  python eval/error_analysis_tests/efbpt/d11o_student.py
fi

echo "=== end: $(date) ==="
