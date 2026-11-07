#!/bin/bash
#SBATCH --job-name=d11r_hf
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/d11r_hf_%j.log
#SBATCH --error=logs/d11r_hf_%j.log
#
# D11-R local readers L2 + L3 + L4 in one job, sequentially, one process (env urbench_eval):
#   gemma    Gemma-2-27B-it NF4 (QEA HfBnbBackend load), greedy, max_new 128
#   qwen14b  Qwen3-14B nf4 (D10 student load), greedy, enable_thinking=False, max_new 128
#   nllb     NLLB-200-3.3B fp16, urd_Arab -> eng_Latn, beam 4, max_length 128
# on rdev300 + dev200 + eval458 (docs/EFBPT_D11R_READER_FREEZE.md §2, §4).
# mem 48G: the NLLB .bin shards are loaded through CPU RAM (17.6 GB).
#
# Usage (from ~/URBench), one argument, TEST or FULL:
#   bash eval/error_analysis_tests/efbpt/d11r_hf.sbatch TEST    # --limit 3, writes under outputs/efbpt/d11r/_test/
#   bash eval/error_analysis_tests/efbpt/d11r_hf.sbatch FULL
# Optional: SBATCH_DEP=<jobid> adds --dependency=afterok:<jobid>.
# Run outside SLURM, the script submits itself (TEST 01:30:00, FULL 06:00:00).

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  TEST|FULL) ;;
  *) echo "usage: $0 TEST|FULL" >&2; exit 2 ;;
esac

cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$MODE" = "TEST" ]; then LIMIT=01:30:00; else LIMIT=06:00:00; fi
  DEP=()
  if [ -n "${SBATCH_DEP:-}" ]; then DEP=(--dependency="afterok:${SBATCH_DEP}"); fi
  exec sbatch --job-name="d11r_hf_${MODE}" --time="$LIMIT" \
       --output="logs/d11r_hf_${MODE}_%j.log" \
       --error="logs/d11r_hf_${MODE}_%j.log" \
       "${DEP[@]}" "$0" "$MODE"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

if [ "$MODE" = "TEST" ]; then
  python eval/error_analysis_tests/efbpt/d11r_local_readers.py --readers gemma qwen14b nllb --limit 3
else
  python eval/error_analysis_tests/efbpt/d11r_local_readers.py --readers gemma qwen14b nllb
fi

echo "=== end: $(date) ==="
