#!/bin/bash
#SBATCH --job-name=d14_teacher
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=L20004,L20007
#SBATCH --output=logs/d14_teacher_%j.log
#SBATCH --error=logs/d14_teacher_%j.log
#
# D14 teacher job (docs/EFBPT_D14_MORE_TASKS_FREEZE.md F4, 9.7, 9.8).
# Llama-3.1-70B-Instruct-AWQ-INT4, vLLM 0.13.0 in env urbench_eval, D10 engine with max_num_seqs 16.
#
# Usage (from ~/URBench):
#   sbatch eval/error_analysis_tests/efbpt/d14_teacher.sbatch TEST all csqa,piqa,gsm8k
#   (TEST only: a 4th argument 'probe' tries smaller max_num_seqs values if the engine does not start)
#   sbatch eval/error_analysis_tests/efbpt/d14_teacher.sbatch FULL facts csqa,piqa
#   sbatch eval/error_analysis_tests/efbpt/d14_teacher.sbatch FULL arms csqa
# TEST = rows 0-4 under outputs/efbpt/d14/_test/ (never a result).
set -euo pipefail
MODE="${1:-}"
WHAT="${2:-}"
TASKS="${3:-csqa,piqa,gsm8k}"
case "$MODE" in TEST|FULL) ;; *) echo "usage: $0 TEST|FULL facts|arms|all tasks" >&2; exit 2 ;; esac
case "$WHAT" in facts|arms|all) ;; *) echo "usage: $0 TEST|FULL facts|arms|all tasks" >&2; exit 2 ;; esac
cd /mnt/home/user41/URBench

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== mode: $MODE   what: $WHAT   tasks: $TASKS   job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
echo "=== start: $(date) ==="

EXTRA=()
if [ "${4:-}" = "probe" ]; then EXTRA=(--probe-seqs); fi
python eval/error_analysis_tests/efbpt/d14_teacher.py --mode "$MODE" --what "$WHAT" --tasks "$TASKS" "${EXTRA[@]}"

echo "=== end: $(date) ==="
