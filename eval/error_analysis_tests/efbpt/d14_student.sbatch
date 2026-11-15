#!/bin/bash
#SBATCH --job-name=d14_student
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=L20004,L20007
#SBATCH --output=logs/d14_student_%j.log
#SBATCH --error=logs/d14_student_%j.log
#
# D14 student job (docs/EFBPT_D14_MORE_TASKS_FREEZE.md F4, 9.6).
# Qwen3-14B, vLLM 0.13.0 in env urbench_eval, engine and sampling of the task's fair script.
#
# Usage (from ~/URBench):
#   sbatch eval/error_analysis_tests/efbpt/d14_student.sbatch TEST csqa,piqa,gsm8k
#   sbatch eval/error_analysis_tests/efbpt/d14_student.sbatch FULL csqa
# TEST = rows 0-4 under outputs/efbpt/d14/_test/ (never a result).
set -euo pipefail
MODE="${1:-}"
TASKS="${2:-csqa,piqa,gsm8k}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST|FULL tasks" >&2; exit 2 ;; esac
cd /mnt/home/user41/URBench

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   tasks: $TASKS   job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d14_student.py --mode "$MODE" --tasks "$TASKS"

echo "=== end: $(date) ==="
