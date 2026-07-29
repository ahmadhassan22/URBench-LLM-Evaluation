#!/bin/bash
#SBATCH --job-name=efbpt_train_ALL9
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/efbpt_train_ALL9_%j.log
#SBATCH --error=logs/efbpt_train_ALL9_%j.log

# Stop on the FIRST failure. Do not burn hours producing junk after a break.
set -euo pipefail

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== job start: $(date) ==="

SCRIPT=eval/error_analysis_tests/efbpt/efbpt_train_qlora.py

# C3 first on purpose: longest, most complex target. If anything breaks,
# it breaks in the first 15 minutes, not at run nine.
RUN=0
for COND in C3 C2 C1; do
  for SEED in 13 42 2026; do
    RUN=$((RUN + 1))
    echo ""
    echo "############################################################"
    echo "### RUN ${RUN}/9  condition=${COND}  seed=${SEED}"
    echo "### start: $(date)"
    echo "############################################################"

    python "${SCRIPT}" --condition "${COND}" --seed "${SEED}"

    echo "### RUN ${RUN}/9 done: $(date)"
  done
done

echo ""
echo "=== ALL 9 RUNS COMPLETE: $(date) ==="
ls -la outputs/efbpt/plan_a/adapters/
