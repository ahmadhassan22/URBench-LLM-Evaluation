#!/bin/bash
#SBATCH --job-name=d11o_teacher
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d11o_teacher_%j.log
#SBATCH --error=logs/d11o_teacher_%j.log
#
# D11-O Job T — oracle-reading teacher facts (docs/EFBPT_D11O_ORACLE_READING_FREEZE.md §3).
# Two arguments: VERSION (v1 = D10 T1 prompt pair, v2 = D10v2 T1-v2 prompt pair) and MODE (TEST|FULL).
# Partition, GPU, CPU and memory as d10v2_teacher.sbatch; vLLM 0.13.0 in env urbench_eval.
#
# Usage (from ~/URBench):
#   bash eval/error_analysis_tests/efbpt/d11o_teacher.sbatch v1 TEST
#   bash eval/error_analysis_tests/efbpt/d11o_teacher.sbatch v2 FULL
# Optional: SBATCH_DEP=<jobid> adds --dependency=afterok:<jobid> (freeze §7 chain).
#
# Run outside SLURM, the script submits itself with version and mode in the job
# name and the log at logs/d11o_teacher_{v1,v2}_{TEST,FULL}_%j.log
# (TEST 01:00:00, FULL 02:00:00, freeze §7).
#
# TEST = --limit 5 (writes only under outputs/efbpt/d11o/_test/teacher_<version>/).

set -euo pipefail

VERSION="${1:-}"
MODE="${2:-}"
case "$VERSION" in
  v1|v2) ;;
  *) echo "usage: $0 v1|v2 TEST|FULL" >&2; exit 2 ;;
esac
case "$MODE" in
  TEST|FULL) ;;
  *) echo "usage: $0 v1|v2 TEST|FULL" >&2; exit 2 ;;
esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=01:00:00; else LIMIT=02:00:00; fi
  DEP=()
  if [ -n "${SBATCH_DEP:-}" ]; then DEP=(--dependency="afterok:${SBATCH_DEP}"); fi
  exec sbatch --job-name="d11o_teacher_${VERSION}_${MODE}" --time="$LIMIT" \
       --output="logs/d11o_teacher_${VERSION}_${MODE}_%j.log" \
       --error="logs/d11o_teacher_${VERSION}_${MODE}_%j.log" \
       "${DEP[@]}" "$0" "$VERSION" "$MODE"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== version: $VERSION   mode: $MODE   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "TEST" ]; then
  python eval/error_analysis_tests/efbpt/d11o_teacher.py --version "$VERSION" --limit 5
else
  python eval/error_analysis_tests/efbpt/d11o_teacher.py --version "$VERSION"
fi

echo "=== end: $(date) ==="
