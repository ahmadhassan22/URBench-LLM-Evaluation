#!/bin/bash
# Outcome-free preflight v2. Preserves failed job 81798 and all preflight_v1 files.
# Corrects only the FAISS post-deserialization concrete-class assertion.
# No Qwen generation, retrieval searches, targets or pilot scoring.
#SBATCH --job-name=bridge_n25_preflight_v2
#SBATCH --partition=q_intel_share_L20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/bridge_n25_preflight_v2_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/bridge_n25_preflight_v2_%j.err

set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
test -d /mnt/home/user41/URBench/logs
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
  /mnt/home/user41/URBench/eval/error_analysis_tests/efbpt/bridge_pilot_preflight.py \
  --preflight --repo /mnt/home/user41/URBench
