#!/bin/bash
#SBATCH --job-name=d13_student
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d13_student_%j.log
#SBATCH --error=logs/d13_student_%j.log
#
# D13 student arms (docs/EFBPT_D13_BASELINES_TRANSFER_FREEZE.md §2, §6, §7.3, §7.4). Batch size 1.
#
# Usage (from ~/URBench):
#   sbatch eval/error_analysis_tests/efbpt/d13_student.sbatch qwen14 REGRESS
#       freeze C1: DEV200, arm TR, rows 0-4, byte-identical to the sealed file; exits non-zero otherwise.
#   sbatch eval/error_analysis_tests/efbpt/d13_student.sbatch qwen14 TEST
#       the C1 regression, then TEST-5 of RAG_ur, RAG_en, RO_eval (needs the TEST retrieval files).
#   sbatch eval/error_analysis_tests/efbpt/d13_student.sbatch others TEST
#       freeze C2 for llama8, gemma9, qwen8 in turn (DEV200 rows 0-4, both layouts, generated twice).
#   sbatch eval/error_analysis_tests/efbpt/d13_student.sbatch llama8|gemma9|qwen8 TEST
#   sbatch [--dependency=afterok:<regress job>] eval/error_analysis_tests/efbpt/d13_student.sbatch qwen14 FULL dev200|eval458 ARM[,ARM]
#   sbatch eval/error_analysis_tests/efbpt/d13_student.sbatch llama8|gemma9|qwen8 FULL eval458 [ARM[,ARM]]
# TEST = DEV200 rows 0-4 under outputs/efbpt/d13/_test/ (never a result).
set -euo pipefail
MODEL="${1:-}"
MODE="${2:-}"
SET="${3:-dev200}"
ARMS="${4:-}"
case "$MODEL" in qwen14|llama8|gemma9|qwen8|others) ;; *) echo "model must be qwen14|llama8|gemma9|qwen8|others" >&2; exit 2 ;; esac
case "$MODE" in TEST|FULL|REGRESS) ;; *) echo "mode must be TEST|FULL|REGRESS" >&2; exit 2 ;; esac
case "$SET" in dev200|eval458) ;; *) echo "set must be dev200 or eval458" >&2; exit 2 ;; esac
if [ "$MODEL" = "others" ] && [ "$MODE" != "TEST" ]; then echo "'others' is for TEST only; a FULL run takes one model per job" >&2; exit 2; fi
if [ "$MODE" = "REGRESS" ] && [ "$MODEL" != "qwen14" ]; then echo "REGRESS is the Qwen3-14B regression (freeze C1)" >&2; exit 2; fi
cd /mnt/home/user41/URBench

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== model: $MODEL   mode: $MODE   set: $SET   arms: ${ARMS:-default}   job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
echo "=== start: $(date) ==="

PY=eval/error_analysis_tests/efbpt/d13_student.py
EXTRA=()
if [ -n "$ARMS" ]; then EXTRA=(--arms "$ARMS"); fi
if [ "$MODEL" = "others" ]; then
  for M in llama8 gemma9 qwen8; do
    echo "=== others TEST: $M  $(date) ==="
    python "$PY" --model "$M" --mode TEST
  done
elif [ "$MODEL" = "qwen14" ] && [ "$MODE" = "TEST" ]; then
  python "$PY" --model qwen14 --mode REGRESS
  python "$PY" --model qwen14 --mode TEST "${EXTRA[@]}"
elif [ "$MODE" = "REGRESS" ]; then
  python "$PY" --model qwen14 --mode REGRESS
else
  python "$PY" --model "$MODEL" --mode "$MODE" --set "$SET" "${EXTRA[@]}"
fi

echo "=== end: $(date) ==="
