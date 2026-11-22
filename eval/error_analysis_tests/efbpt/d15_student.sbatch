#!/bin/bash
#SBATCH --job-name=d15_student
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=L20004,L20007
#SBATCH --output=logs/d15_student_%j.log
#SBATCH --error=logs/d15_student_%j.log
#
# D15 student job (docs/EFBPT_D15_NATIVE_URDU_FREEZE.md N3, N4, 11.5, 11.7).
# Qwen3-14B, vLLM 0.13.0 in env urbench_eval, engine of the CSQA fair script.
#   sl          Qwen3-14B translating by itself (thinking off, max_tokens 512)
#   arms        S_ur, S_ro, S_self, S_tr (thinking on, max_tokens 1024)
#
# Usage (from ~/URBench):
#   sbatch eval/error_analysis_tests/efbpt/d15_student.sbatch TEST sl
#   sbatch eval/error_analysis_tests/efbpt/d15_student.sbatch FULL arms S_ur,S_ro,S_self,S_tr
# TEST = the first 5 sampled rows under outputs/efbpt/d15/_test/ (never a result).
set -euo pipefail
MODE="${1:-}"
WHAT="${2:-}"
ARMS="${3:-S_ur,S_ro,S_self,S_tr}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST|FULL sl|arms|sl+arms [arms]" >&2; exit 2 ;; esac
case "$WHAT" in sl|arms|sl+arms) ;; *) echo "usage: $0 TEST|FULL sl|arms|sl+arms [arms]" >&2; exit 2 ;; esac
cd /mnt/home/user41/URBench

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   what: $WHAT   arms: $ARMS   job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d15_student.py --mode "$MODE" --what "$WHAT" --arms "$ARMS"

echo "=== end: $(date) ==="
