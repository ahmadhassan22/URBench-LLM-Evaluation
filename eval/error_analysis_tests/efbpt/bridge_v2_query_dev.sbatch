#!/bin/bash
#SBATCH --job-name=bridge_v2_query_dev
#SBATCH --partition=q_intel_share_L20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/bridge_v2_query_dev_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/bridge_v2_query_dev_%j.err

# EFBPT bridge V2 query-prompt development run, N25, arms A'/P'/B'/D'.
#
# POST-OUTCOME DEVELOPMENT EVIDENCE on an already-exposed cohort. No gate, no
# significance test, no p-value. The frozen N=25 pilot is read-only input and
# is re-verified byte-identical before and after the run.
#
# Single job, five phases in one process:
#   S0 verify code, environment, inputs, Git, model files
#   S1 generate 100 queries (Qwen3-14B)
#   S2 emit input-only evidence-relevance sheet, then blinded query sheet
#   S3 encode + FAISS search, top-100
#   S4 open gold, descriptive recall   S5 re-verify frozen inputs, seal
#
# Writes only under outputs/efbpt/bridge_v2_dev/queries_dev1/, which must not
# already exist. No resume and no retry: a failed run leaves its partial
# directory in place for review and must not be relaunched without one.
#
# Submit from the repository root. The snapshot is used only if git is
# unavailable on the compute node, and is then labelled
# SUBMISSION_TIME_SNAPSHOT_NOT_RUNTIME_VERIFIED:
#
#   SNAP=$PWD/logs/bridge_v2_query_dev_submit_$(date -u +%Y%m%dT%H%M%SZ).txt
#   { git rev-parse HEAD; git status --porcelain=v1; } > "$SNAP"
#   sbatch --export=ALL,BRIDGE_V2_SUBMIT_SNAPSHOT="$SNAP",BRIDGE_V2_SUBMIT_SNAPSHOT_SHA256=$(sha256sum "$SNAP" | cut -d' ' -f1) \
#       eval/error_analysis_tests/efbpt/bridge_v2_query_dev.sbatch

set -euo pipefail

REPO=/mnt/home/user41/URBench
PY=/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python
RUNNER="${REPO}/eval/error_analysis_tests/efbpt/bridge_v2_query_dev.py"
TESTS="${REPO}/eval/error_analysis_tests/efbpt/bridge_v2_query_dev_test.py"
OUT="${REPO}/outputs/efbpt/bridge_v2_dev/queries_dev1"

# Approved development code. The wrapper pins the runner and tests; it cannot
# pin itself, so the executed copy is hashed below and recorded by the runner.
RUNNER_SHA256=388512ecc2d0c8bbd3d6f156621dc07ae1412c5fbcff0fdee614736f5f919f4f
TEST_SHA256=878e55463642468901a4e5f6bf9d166605d82151b8747d122ea741d605f57175

stopped() {
  echo "STOPPED phase=$1 exit=$2 output_root=${OUT} partial_artifacts_preserved=true" >&2
  exit 2
}

export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BRIDGE_V2_RUNNER_SHA256="${RUNNER_SHA256}"
export BRIDGE_V2_TEST_SHA256="${TEST_SHA256}"
BRIDGE_V2_WRAPPER_EXECUTED_SHA256="$(sha256sum "$0" | cut -d' ' -f1)"
export BRIDGE_V2_WRAPPER_EXECUTED_SHA256

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) utc=$(date -u +%FT%TZ)"
echo "head=$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo UNAVAILABLE)"
echo "wrapper_executed_sha256=${BRIDGE_V2_WRAPPER_EXECUTED_SHA256}"

test -x "${PY}" || stopped frozen_interpreter_missing 1
if [ -e "${OUT}" ]; then
  echo "REFUSING: output root already exists: ${OUT}" >&2
  exit 2
fi

# Approved code before any Python runs.
printf '%s  %s\n%s  %s\n' "${RUNNER_SHA256}" "${RUNNER}" "${TEST_SHA256}" "${TESTS}" \
  | sha256sum --check --strict --quiet || stopped code_identity $?

# Offline tests, then the read-only input check, both under the frozen
# interpreter. Either failing aborts before any model is loaded.
"${PY}" -B "${TESTS}" || stopped offline_tests $?
"${PY}" -B "${RUNNER}" --check-inputs || stopped check_inputs $?

# The runner prints its own stage-labelled STOPPED record on failure.
"${PY}" -B "${RUNNER}" --run || stopped run $?

echo "complete utc=$(date -u +%FT%TZ)"
