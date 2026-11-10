#!/bin/bash
#SBATCH --job-name=d11_ablate_ro_dev200
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/d11_ablate_ro_dev200_%j.log
#SBATCH --error=logs/d11_ablate_ro_dev200_%j.log
#
# D11 RTG Amendment 3 A3.6 — DEV200 baseline ablation RO then RO_gold (descriptive, not gated,
# no eval458 use). One job, batch size 1, per-arm seal, resumable; V2' probe fatal.
# Partition, GPU, CPU, memory as d11_student.sbatch.
#
# Usage (from ~/URBench):
#   D11_DEP=afterany:<jobid>:<jobid> D11_NICE=200 bash eval/error_analysis_tests/efbpt/d11_ablate_ro.sbatch
# D11_DEP is passed as --dependency=<D11_DEP>; D11_NICE as --nice=<n>; D11_TIME overrides 06:00:00.
# Run outside SLURM, the script submits itself.

set -euo pipefail
cd /mnt/home/user41/URBench

if [ -z "${SLURM_JOB_ID:-}" ]; then
  DEP=()
  if [ -n "${D11_DEP:-}" ]; then DEP=(--dependency="${D11_DEP}"); fi
  NICE=()
  if [ -n "${D11_NICE:-}" ]; then NICE=(--nice="${D11_NICE}"); fi
  exec sbatch --time="${D11_TIME:-06:00:00}" "${DEP[@]}" "${NICE[@]}" "$0"
fi

source /mnt/home/user41/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval

echo "=== node: $(hostname) ==="
echo "=== job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== start: $(date) ==="

python eval/error_analysis_tests/efbpt/d11_ablate_ro.py

echo "=== end: $(date) ==="
