#!/bin/bash
# Stage 2 of 4: isolated oracle-E queries. Separate process, launched only
# after the parent-stage seal exists. Reads the oracle-E export and the sealed
# A queries for the empty-output fallback; never parent-only content or targets.
#SBATCH --job-name=bridge_n25_oracle_e_v1
#SBATCH --partition=q_intel_share_L20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/bridge_n25_oracle_e_v1_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/bridge_n25_oracle_e_v1_%j.err

set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
test -d /mnt/home/user41/URBench/logs
cd /mnt/home/user41/URBench
# The reviewed activation identity is RECEIVED, never defaulted. Both variables
# must be exported by the reviewed submission; there is no fallback value and
# the runner independently re-verifies this job file against the manifest.
test -n "${BRIDGE_ACTIVATION:-}"
test -n "${BRIDGE_ACTIVATION_SHA256:-}"
test -f "${BRIDGE_ACTIVATION}"
printf '%s' "${BRIDGE_ACTIVATION_SHA256}" | grep -Eq '^[0-9a-f]{64}$'
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260904
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B \
  /mnt/home/user41/URBench/eval/error_analysis_tests/efbpt/bridge_pilot_run.py \
  --stage oracle_e --tag "${BRIDGE_TAG:-v1}" --parent-tag "${BRIDGE_PARENT_TAG:-v1}" \
  --activation "${BRIDGE_ACTIVATION}" \
  --activation-sha256 "${BRIDGE_ACTIVATION_SHA256}"
