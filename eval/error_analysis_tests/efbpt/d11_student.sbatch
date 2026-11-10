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
# eval458 (Amendment 2 A2.3-A2.5): one job per arm, ARMS=<arm> and EVAL_ARM=TR required, FULL only;
#   per-arm seal, resumable, nothing scored (the merge is run by hand: d11_student.py --merge).
# REGRESS (Amendment 2 A2.3): DEV200, arm TR, rows 0-4 through the per-arm path, byte compare with
#   the sealed file; exits non-zero on mismatch; every eval458 job depends afterok on it.
#
# Usage (from ~/URBench):  bash eval/error_analysis_tests/efbpt/d11_student.sbatch TEST|FULL [dev200|eval458]
#                          bash eval/error_analysis_tests/efbpt/d11_student.sbatch REGRESS
# Optional: SBATCH_DEP=<jobid> adds --dependency=afterok:<jobid>; EVAL_ARM=<arm> for eval458;
#           ARMS=<arm> (eval458); D11_TIME=<hh:mm:ss> overrides the time limit; D11_NICE=<n> adds --nice=<n>.
# Run outside SLURM, the script submits itself (TEST 02:00:00, FULL 16:00:00) with the log at
# logs/d11_student_{MODE}_{SET}_%j.log. TEST = first 5 rows under outputs/efbpt/d11/_test/.

set -euo pipefail

MODE="${1:-}"
SET="${2:-dev200}"
case "$MODE" in TEST|FULL|REGRESS) ;; *) echo "usage: $0 TEST|FULL [dev200|eval458]  |  $0 REGRESS" >&2; exit 2 ;; esac
case "$SET" in dev200|eval458) ;; *) echo "usage: $0 TEST|FULL [dev200|eval458]  |  $0 REGRESS" >&2; exit 2 ;; esac

cd /mnt/home/user41/URBench

ARMS="${ARMS:-}"
EVAL_ARM="${EVAL_ARM:-}"
if [ "$MODE" = "REGRESS" ] && [ "$SET" != "dev200" ]; then echo "REGRESS is DEV200 only (Amendment 2 A2.3)" >&2; exit 2; fi
if [ "$SET" = "eval458" ]; then
  if [ "$MODE" != "FULL" ]; then echo "no TEST on eval458 rows (Amendment 2 A2.4)" >&2; exit 2; fi
  case "$ARMS" in A_r2_eval|TR|TR_wrong_eval|TR_L1_eval) ;; *) echo "eval458: ARMS=A_r2_eval|TR|TR_wrong_eval|TR_L1_eval is required, one arm per job (Amendment 2 A2.3)" >&2; exit 2 ;; esac
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=02:00:00; elif [ "$MODE" = "REGRESS" ]; then LIMIT=01:00:00; elif [ "$SET" = "eval458" ]; then LIMIT=08:00:00; else LIMIT=16:00:00; fi
  LIMIT="${D11_TIME:-$LIMIT}"
  DEP=()
  if [ -n "${SBATCH_DEP:-}" ]; then DEP=(--dependency="afterok:${SBATCH_DEP}"); fi
  NICE=()
  if [ -n "${D11_NICE:-}" ]; then NICE=(--nice="${D11_NICE}"); fi
  TAG="${MODE}_${SET}${ARMS:+_${ARMS}}"
  export ARMS EVAL_ARM
  exec sbatch --job-name="d11_student_${TAG}" --time="$LIMIT" \
       --output="logs/d11_student_${TAG}_%j.log" \
       --error="logs/d11_student_${TAG}_%j.log" \
       --export=ALL \
       "${DEP[@]}" "${NICE[@]}" "$0" "$MODE" "$SET"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   set: $SET   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "REGRESS" ]; then
  python eval/error_analysis_tests/efbpt/d11_student.py --regress
else
  EXTRA=()
  if [ "$SET" = "eval458" ]; then EXTRA=(--eval-arm "${EVAL_ARM:?EVAL_ARM=TR is required for eval458}" --arms "$ARMS"); fi
  python eval/error_analysis_tests/efbpt/d11_student.py --set "$SET" --mode "$MODE" "${EXTRA[@]}"
fi

echo "=== end: $(date) ==="
