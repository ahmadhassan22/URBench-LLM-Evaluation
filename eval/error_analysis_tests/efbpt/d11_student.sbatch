#!/bin/bash
#SBATCH --job-name=d11_student
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/d11_student_%j.log
#SBATCH --error=logs/d11_student_%j.log
#
# D11 RTG Job S — student arms (docs/EFBPT_D11_RTG_FREEZE.md §4, §5, §7, §9; Amendment 1).
# DEV200: A_r2 (sealed, V2'), TR, TR2, TR_L1, TR_wrong, TR_noq at batch size 1.
# eval458: needs docs/D11_GO_EVAL458.md and EVAL_ARM=TR|TR2|TR_L1 (A1.5).
# Requires the sealed D11 teacher outputs of the same mode and set for R1, R2, R3.
#
# Usage (from ~/URBench):  bash eval/error_analysis_tests/efbpt/d11_student.sbatch TEST|FULL [dev200|eval458]
# Optional: SBATCH_DEP=<jobid> adds --dependency=afterok:<jobid>; EVAL_ARM=<arm> for eval458.
# Run outside SLURM, the script submits itself (TEST 02:00:00, FULL 16:00:00) with the log at
# logs/d11_student_{MODE}_{SET}_%j.log. TEST = first 5 rows under outputs/efbpt/d11/_test/.

set -euo pipefail

MODE="${1:-}"
SET="${2:-dev200}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST|FULL [dev200|eval458]" >&2; exit 2 ;; esac
case "$SET" in dev200|eval458) ;; *) echo "usage: $0 TEST|FULL [dev200|eval458]" >&2; exit 2 ;; esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=02:00:00; else LIMIT=16:00:00; fi
  DEP=()
  if [ -n "${SBATCH_DEP:-}" ]; then DEP=(--dependency="afterok:${SBATCH_DEP}"); fi
  exec sbatch --job-name="d11_student_${MODE}_${SET}" --time="$LIMIT" \
       --output="logs/d11_student_${MODE}_${SET}_%j.log" \
       --error="logs/d11_student_${MODE}_${SET}_%j.log" \
       --export=ALL,EVAL_ARM="${EVAL_ARM:-}" \
       "${DEP[@]}" "$0" "$MODE" "$SET"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   set: $SET   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

EXTRA=()
if [ "$SET" = "eval458" ]; then EXTRA=(--eval-arm "${EVAL_ARM:?EVAL_ARM=TR|TR2|TR_L1 is required for eval458}"); fi
python eval/error_analysis_tests/efbpt/d11_student.py --set "$SET" --mode "$MODE" "${EXTRA[@]}"

echo "=== end: $(date) ==="
