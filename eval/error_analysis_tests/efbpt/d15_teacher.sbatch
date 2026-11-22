#!/bin/bash
#SBATCH --job-name=d15_teacher
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=L20004,L20007
#SBATCH --output=logs/d15_teacher_%j.log
#SBATCH --error=logs/d15_teacher_%j.log
#
# D15 teacher job (docs/EFBPT_D15_NATIVE_URDU_FREEZE.md N3, N4, 11.5-11.8).
# Llama-3.1-70B-Instruct-AWQ-INT4, vLLM 0.13.0 in env urbench_eval.
#   tl          the D10 engine (max_num_seqs 1), the teacher translating by itself
#   facts|arms|facts+arms   the D14 engine (D14 Amendment 1 arguments)
#
# Usage (from ~/URBench):
#   sbatch eval/error_analysis_tests/efbpt/d15_teacher.sbatch TEST tl
#   sbatch eval/error_analysis_tests/efbpt/d15_teacher.sbatch FULL facts+arms T_ur,T_ue,T_r1
# TEST = the first 5 sampled rows under outputs/efbpt/d15/_test/ (never a result).
set -euo pipefail
MODE="${1:-}"
WHAT="${2:-}"
ARMS="${3:-T_ur,T_ue,T_r1,T_self}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST|FULL tl|facts|arms|facts+arms [arms]" >&2; exit 2 ;; esac
case "$WHAT" in tl|facts|arms|facts+arms) ;; *) echo "usage: $0 TEST|FULL tl|facts|arms|facts+arms [arms]" >&2; exit 2 ;; esac
cd /mnt/home/user41/URBench

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   what: $WHAT   arms: $ARMS   job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d15_teacher.py --mode "$MODE" --what "$WHAT" --arms "$ARMS"

echo "=== end: $(date) ==="
