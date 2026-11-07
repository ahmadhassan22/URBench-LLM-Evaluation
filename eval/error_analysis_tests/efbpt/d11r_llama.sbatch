#!/bin/bash
#SBATCH --job-name=d11r_llama
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d11r_llama_%j.log
#SBATCH --error=logs/d11r_llama_%j.log
#
# D11-R local reader L1 — Llama-3.1-70B-AWQ via vLLM 0.13.0 (env urbench_eval), D10 §2 engine
# args, greedy, max_tokens 128, on rdev300 + dev200 + eval458
# (docs/EFBPT_D11R_READER_FREEZE.md §2, §4). Partition, GPU, CPU and memory as d11o_teacher.sbatch.
#
# Usage (from ~/URBench), one argument, TEST or FULL:
#   bash eval/error_analysis_tests/efbpt/d11r_llama.sbatch TEST    # --limit 3, writes under outputs/efbpt/d11r/_test/
#   bash eval/error_analysis_tests/efbpt/d11r_llama.sbatch FULL
# Optional: SBATCH_DEP=<jobid> adds --dependency=afterok:<jobid>.
# Run outside SLURM, the script submits itself (TEST 00:45:00, FULL 02:00:00).
# `sbatch eval/error_analysis_tests/efbpt/d11r_llama.sbatch` (no argument) runs FULL under the
# #SBATCH header above (log logs/d11r_llama_<jobid>.log).

set -euo pipefail

MODE="${1:-FULL}"
case "$MODE" in
  TEST|FULL) ;;
  *) echo "usage: $0 TEST|FULL" >&2; exit 2 ;;
esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=00:45:00; else LIMIT=02:00:00; fi
  DEP=()
  if [ -n "${SBATCH_DEP:-}" ]; then DEP=(--dependency="afterok:${SBATCH_DEP}"); fi
  exec sbatch --job-name="d11r_llama_${MODE}" --time="$LIMIT" \
       --output="logs/d11r_llama_${MODE}_%j.log" \
       --error="logs/d11r_llama_${MODE}_%j.log" \
       "${DEP[@]}" "$0" "$MODE"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "TEST" ]; then
  python eval/error_analysis_tests/efbpt/d11r_local_readers.py --readers llama --limit 3
else
  python eval/error_analysis_tests/efbpt/d11r_local_readers.py --readers llama
fi

echo "=== end: $(date) ==="
