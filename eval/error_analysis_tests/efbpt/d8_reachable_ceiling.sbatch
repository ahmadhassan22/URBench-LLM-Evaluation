#!/bin/bash
#SBATCH --job-name=d8_reachable
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/d8_reachable_%j.log
#SBATCH --error=logs/d8_reachable_%j.log
#
# D8 — reachable-page ceiling (docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md §9).
# Partition, GPU, CPU and memory copied from d4_extract_FULL.sbatch.
#
# Usage (from ~/URBench), one argument, TEST or FULL:
#   bash eval/error_analysis_tests/efbpt/d8_reachable_ceiling.sbatch TEST
#   bash eval/error_analysis_tests/efbpt/d8_reachable_ceiling.sbatch FULL
#
# Run outside SLURM, the script submits itself with the mode in the job name
# and the log at logs/d8_reachable_{TEST,FULL}_%j.log (TEST 01:30:00,
# FULL 08:00:00, Amendment 1 A4). Submitted directly with `sbatch <file> MODE`
# it still runs,
# but under the fallback log name above.
#
# TEST = --limit 5 --dry-run (writes only *_TEST files). TEST-5 must pass
# before FULL is submitted (freeze §9).

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  TEST|FULL) ;;
  *) echo "usage: $0 TEST|FULL" >&2; exit 2 ;;
esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=01:30:00; else LIMIT=08:00:00; fi
  exec sbatch --job-name="d8_reachable_${MODE}" --time="$LIMIT" \
       --output="logs/d8_reachable_${MODE}_%j.log" \
       --error="logs/d8_reachable_${MODE}_%j.log" \
       "$0" "$MODE"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "TEST" ]; then
  python eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py \
      --limit 5 --dry-run --batch-size 8
else
  python eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py --batch-size 8
fi

echo "=== end: $(date) ==="
