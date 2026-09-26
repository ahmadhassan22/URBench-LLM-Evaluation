#!/bin/bash
#SBATCH --job-name=efbpt_target_reachability_v1
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/user41/URBench/logs/efbpt_target_reachability_v1_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/efbpt_target_reachability_v1_%j.err
# CPU-only diagnostic. No submission is authorized merely by installing this file.
set -euo pipefail
umask 077

: "${REACH_PROBE_SHA256:?Supply the audited probe SHA-256}"
: "${REACH_SBATCH_SHA256:?Supply the audited sbatch SHA-256}"
[[ "$REACH_PROBE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$REACH_SBATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]

# Verify the executing Slurm copy too, not just the repository's wrapper.
reach_wrapper_line=$(sha256sum -- "${BASH_SOURCE[0]}")
[[ "${reach_wrapper_line%% *}" == "$REACH_SBATCH_SHA256" ]]

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GIT_OPTIONAL_LOCKS=0
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=20260904
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

cd /mnt/home/user41/URBench
exec /mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B -s \
  eval/error_analysis_tests/efbpt/efbpt_target_reachability_probe.py \
  --run --repo /mnt/home/user41/URBench \
  --probe-sha256 "$REACH_PROBE_SHA256" \
  --wrapper-sha256 "$REACH_SBATCH_SHA256"
