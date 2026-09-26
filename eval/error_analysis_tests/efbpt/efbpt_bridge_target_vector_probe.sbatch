#!/bin/bash
#SBATCH --job-name=efbpt_bridge_target_vector_v1
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/efbpt_bridge_target_vector_v1_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/efbpt_bridge_target_vector_v1_%j.err
# GPU diagnostic. The 34.3 GiB IndexFlatIP is loaded resident, hence 80G, matching
# the frozen retrieval and preflight jobs. Installing this file authorizes nothing;
# submission requires separate explicit approval.
# This wrapper never creates the diagnostic output directory. The probe's own
# output-absence guard must run first and create it.
set -euo pipefail
umask 077

: "${TVP_PROBE_SHA256:?Supply the audited probe SHA-256}"
: "${TVP_SBATCH_SHA256:?Supply the audited sbatch SHA-256}"
[[ "$TVP_PROBE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$TVP_SBATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]

# Git provenance. The compute nodes of this cluster have no git binary, so the
# submitting login-node shell captures HEAD, the full porcelain status and a UTC
# timestamp, and exports them here. The probe records this as SUBMISSION_TIME
# provenance and never claims it was runtime-verified. Status is base64 so that
# newlines survive the environment intact.
: "${TVP_GIT_HEAD:?Supply the login-node HEAD captured at submission}"
: "${TVP_GIT_STATUS_B64:?Supply base64 of git status --porcelain captured at submission}"
: "${TVP_GIT_SNAPSHOT_UTC:?Supply the UTC timestamp of the git snapshot}"
[[ "$TVP_GIT_HEAD" =~ ^[0-9a-f]{40}$ ]]
[[ "$TVP_GIT_STATUS_B64" =~ ^[A-Za-z0-9+/=]*$ ]]

# Verify the executing Slurm copy too, not just the repository's wrapper.
tvp_wrapper_line=$(sha256sum -- "${BASH_SOURCE[0]}")
[[ "${tvp_wrapper_line%% *}" == "$TVP_SBATCH_SHA256" ]]

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GIT_OPTIONAL_LOCKS=0
export PYTHONHASHSEED=20260904
# Match the frozen retrieval threading (settings.retrieval.cpu_threads = 8).
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME

cd /mnt/home/user41/URBench
exec /mnt/home/user41/miniconda3/envs/urbench_eval/bin/python -B -s \
  eval/error_analysis_tests/efbpt/efbpt_bridge_target_vector_probe.py \
  --run --repo /mnt/home/user41/URBench \
  --probe-sha256 "$TVP_PROBE_SHA256" \
  --wrapper-sha256 "$TVP_SBATCH_SHA256"
