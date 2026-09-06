#!/bin/bash
# Stage 4 of 4: CPU-only scorer. Opens the sealed targets only after the
# activation manifest, preparation seal, preflight seal, parent seal, oracle-E
# seal and predictions seal all authenticate. No GPU, no model, no index.
#SBATCH --job-name=bridge_n25_score_v1
#SBATCH --partition=q_intel_share_L20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/bridge_n25_score_v1_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/bridge_n25_score_v1_%j.err

set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
test -d /mnt/home/user41/URBench/logs
cd /mnt/home/user41/URBench
# The reviewed activation identity is RECEIVED, never defaulted.
test -n "${BRIDGE_ACTIVATION:-}"
test -n "${BRIDGE_ACTIVATION_SHA256:-}"
test -n "${BRIDGE_PREDICTIONS_SEAL:-}"
test -n "${BRIDGE_PREDICTIONS_SEAL_SHA256:-}"
test -n "${BRIDGE_TARGETS_SHA256:-}"
test -n "${BRIDGE_SCORE_OUT:-}"
test -f "${BRIDGE_ACTIVATION}"
test -f "${BRIDGE_PREDICTIONS_SEAL}"
for h in "${BRIDGE_ACTIVATION_SHA256}" "${BRIDGE_PREDICTIONS_SEAL_SHA256}" "${BRIDGE_TARGETS_SHA256}"; do
  printf '%s' "${h}" | grep -Eq '^[0-9a-f]{64}$'
done
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=20260904

/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B \
  /mnt/home/user41/URBench/eval/error_analysis_tests/efbpt/bridge_pilot_score.py \
  --activation "${BRIDGE_ACTIVATION}" \
  --activation-sha256 "${BRIDGE_ACTIVATION_SHA256}" \
  --predictions-seal "${BRIDGE_PREDICTIONS_SEAL}" \
  --seal-sha256 "${BRIDGE_PREDICTIONS_SEAL_SHA256}" \
  --targets-sha256 "${BRIDGE_TARGETS_SHA256}" \
  --parent-tag "${BRIDGE_PARENT_TAG:-v1}" --oracle-e-tag "${BRIDGE_ORACLE_E_TAG:-v1}" \
  --out "${BRIDGE_SCORE_OUT}"
