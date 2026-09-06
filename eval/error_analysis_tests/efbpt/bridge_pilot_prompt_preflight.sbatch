#!/bin/bash
# Tokenizer-only prompt preflight. CPU only: no GPU directive, no weights,
# no encoder, no FAISS index, no metadata/offsets, no scoring targets.
# Measures the exact Qwen token length of the 150 known pilot prompts and
# seals the result. Generates nothing, retrieves nothing, scores nothing.
#SBATCH --job-name=bridge_n25_prompt_preflight_v1
#SBATCH --partition=q_intel_share_L20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/bridge_n25_prompt_preflight_v1_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/bridge_n25_prompt_preflight_v1_%j.err

set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
test -d /mnt/home/user41/URBench/logs
cd /mnt/home/user41/URBench
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=20260904

/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B \
  /mnt/home/user41/URBench/eval/error_analysis_tests/efbpt/bridge_pilot_prompt_preflight.py \
  --preflight --repo /mnt/home/user41/URBench
