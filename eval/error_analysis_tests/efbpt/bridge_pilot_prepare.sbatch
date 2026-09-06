#!/bin/bash
# Preparation revision 0.3. Fresh destinations preserve jobs 80439 and 80472.
# Same CPU-only resources reviewed and used by job 80439. No GPU requested.
#SBATCH --job-name=bridge_n25_prep_r3
#SBATCH --partition=q_intel_share_L20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/bridge_n25_prepare_r3_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/bridge_n25_prepare_r3_%j.err

set -euo pipefail
umask 077

test -n "${SLURM_JOB_ID:-}"
test -d /mnt/home/user41/URBench/logs
test -x /mnt/home/user41/miniconda3/envs/urbench_eval/bin/python

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B \
  /mnt/home/user41/URBench/eval/error_analysis_tests/efbpt/bridge_pilot_prepare.py \
  --prepare \
  --repo /mnt/home/user41/URBench \
  --cache /mnt/home/user41/.cache/modelscope/hub/datasets/downloads \
  --archive /mnt/home/user41/URBench_pilot_archives/bridge_pilot_n25_preparation_v1_r3
