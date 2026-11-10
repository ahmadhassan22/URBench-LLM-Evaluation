#!/bin/bash
#SBATCH --job-name=d11_teacher
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/d11_teacher_%j.log
#SBATCH --error=logs/d11_teacher_%j.log
#
# D11 RTG Job T — v1 teacher facts from the sealed D11-R readings
# (docs/EFBPT_D11_RTG_FREEZE.md §3, §9; Amendment 1 A1.3). One job runs the three readers
# R1, R2, R3 in sequence, each sealed under outputs/efbpt/d11/teacher/<set>/<reader>/.
# Partition, GPU, CPU, memory as d11o_teacher.sbatch; vLLM 0.13.0 in env urbench_eval.
#
# eval458 (Amendment 2 A2.3/A2.4): one job per reader, READER=R1|R3 required, FULL only (no TEST
# on eval458 rows); job name and log carry the reader.
#
# Usage (from ~/URBench):  bash eval/error_analysis_tests/efbpt/d11_teacher.sbatch TEST|FULL [dev200|eval458]
# Optional: SBATCH_DEP=<jobid> adds --dependency=afterok:<jobid> (freeze §9 chain);
#           READER=R1|R2|R3 runs that reader only; D11_NICE=<n> adds --nice=<n>.
# Run outside SLURM, the script submits itself (TEST 01:00:00, FULL 03:00:00) with the log at
# logs/d11_teacher_{MODE}_{SET}_%j.log. TEST = first 5 rows under outputs/efbpt/d11/_test/.

set -euo pipefail

MODE="${1:-}"
SET="${2:-dev200}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST|FULL [dev200|eval458]" >&2; exit 2 ;; esac
case "$SET" in dev200|eval458) ;; *) echo "usage: $0 TEST|FULL [dev200|eval458]" >&2; exit 2 ;; esac

cd /mnt/home/user41/URBench

READER="${READER:-}"
case "$READER" in ""|R1|R2|R3) ;; *) echo "READER must be R1|R2|R3" >&2; exit 2 ;; esac
if [ "$SET" = "eval458" ]; then
  if [ "$MODE" != "FULL" ]; then echo "no TEST on eval458 rows (Amendment 2 A2.4)" >&2; exit 2; fi
  case "$READER" in R1|R3) ;; *) echo "eval458: READER=R1|R3 is required, one job per reader (Amendment 2 A2.3)" >&2; exit 2 ;; esac
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=01:00:00; else LIMIT=03:00:00; fi
  DEP=()
  if [ -n "${SBATCH_DEP:-}" ]; then DEP=(--dependency="afterok:${SBATCH_DEP}"); fi
  NICE=()
  if [ -n "${D11_NICE:-}" ]; then NICE=(--nice="${D11_NICE}"); fi
  TAG="${MODE}_${SET}${READER:+_${READER}}"
  export READER
  exec sbatch --job-name="d11_teacher_${TAG}" --time="$LIMIT" \
       --output="logs/d11_teacher_${TAG}_%j.log" \
       --error="logs/d11_teacher_${TAG}_%j.log" \
       --export=ALL \
       "${DEP[@]}" "${NICE[@]}" "$0" "$MODE" "$SET"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   set: $SET   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

for R in ${READER:-R1 R2 R3}; do
  echo "=== $(date +%T) teacher reader $R ==="
  python eval/error_analysis_tests/efbpt/d11_teacher.py --reader "$R" --set "$SET" --mode "$MODE"
done

echo "=== end: $(date) ==="
