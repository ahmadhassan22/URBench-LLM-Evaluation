#!/bin/bash
#SBATCH --job-name=efbpt_qea_triage_gpu
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/efbpt_qea_triage_gpu_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/efbpt_qea_triage_gpu_%j.err
#
# QEA development triage, GPU stages. One stage per job; nothing chains automatically.
#   smoke-llama | smoke-gemma : bounded load + response-format check on fixed, already-exposed
#                               DEV200 items; outputs are SMOKE_NOT_ANNOTATION.
#   smoke-llama-so            : the same six Llama requests with opt-in structured outputs
#                               (xgrammar), written to outputs/efbpt/qea_triage_v1_smoke_v2/llama/.
#   t1 | t2                   : Llama-3.1-70B AWQ annotator (vLLM) under the runner's LLAMA_EXECUTION
#                               (amendment 1: 0.95, max_num_seqs 4, per-request xgrammar schemas; no options).
#   r1                        : Gemma-2-27B-it 4-bit reviewer R1 (own backend; no schemas, no memory fraction).
#   calib-gemma               : amendment-3 paired review calibration, 51 cases x 2 prompt versions = 102
#                               Gemma calls; reads outputs/efbpt/qea_review_calibration_v1/packet/ only.
#   ident-gemma               : amendment-4 isolated title-identification calibration, one Gemma call per item;
#                               reads outputs/efbpt/qea_identification_calibration_v1/packet/ only.
# The offline test suite must pass before any stage starts. No git, no network, no retries.
set -euo pipefail

STAGE="${1:-}"
case "${STAGE}" in
  smoke-llama|smoke-llama-so|smoke-gemma|t1|t2|r1|calib-gemma|ident-gemma) ;;
  *) echo "REFUSING: unknown or missing stage '${STAGE}'" >&2; exit 2 ;;
esac

PY=/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python
DIR=eval/error_analysis_tests/efbpt
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false

echo "stage=${STAGE} job=${SLURM_JOB_ID:-none} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Code-identity guard: fail closed before GPU queries, tests or any model load. Expected hashes are
# captured at submission and exported (QEA_EXPECT_RUNNER_SHA256, QEA_EXPECT_TEST_SHA256,
# QEA_EXPECT_WRAPPER_SHA256); none is embedded in any file. The wrapper hash is taken from the script
# actually executed ($0, Slurm's spool copy), not only from the repository copy.
for VAR in QEA_EXPECT_RUNNER_SHA256 QEA_EXPECT_TEST_SHA256 QEA_EXPECT_WRAPPER_SHA256; do
  if [[ ! "${!VAR:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "REFUSING: ${VAR} missing or malformed" >&2; exit 3
  fi
done
QEA_OBSERVED_RUNNER_SHA256="$(sha256sum "${DIR}/efbpt_qea_triage_v1.py" | cut -d' ' -f1)"
QEA_OBSERVED_TEST_SHA256="$(sha256sum "${DIR}/efbpt_qea_triage_v1_test.py" | cut -d' ' -f1)"
QEA_EXECUTED_WRAPPER_PATH="$(readlink -f "$0")"
QEA_OBSERVED_EXECUTED_WRAPPER_SHA256="$(sha256sum "${QEA_EXECUTED_WRAPPER_PATH}" | cut -d' ' -f1)"
QEA_OBSERVED_REPO_WRAPPER_SHA256="$(sha256sum "${DIR}/efbpt_qea_triage_gpu_v1.sbatch" | cut -d' ' -f1)"
echo "identity runner=${QEA_OBSERVED_RUNNER_SHA256} tests=${QEA_OBSERVED_TEST_SHA256}"
echo "identity executed_wrapper=${QEA_OBSERVED_EXECUTED_WRAPPER_SHA256} path=${QEA_EXECUTED_WRAPPER_PATH}"
echo "identity repository_wrapper=${QEA_OBSERVED_REPO_WRAPPER_SHA256} (recorded; the executed copy is authoritative)"
[ "${QEA_OBSERVED_RUNNER_SHA256}" = "${QEA_EXPECT_RUNNER_SHA256}" ] || { echo "REFUSING: runner hash mismatch" >&2; exit 3; }
[ "${QEA_OBSERVED_TEST_SHA256}" = "${QEA_EXPECT_TEST_SHA256}" ] || { echo "REFUSING: test hash mismatch" >&2; exit 3; }
[ "${QEA_OBSERVED_EXECUTED_WRAPPER_SHA256}" = "${QEA_EXPECT_WRAPPER_SHA256}" ] || { echo "REFUSING: executed wrapper hash mismatch" >&2; exit 3; }
export QEA_OBSERVED_RUNNER_SHA256 QEA_OBSERVED_TEST_SHA256 QEA_EXECUTED_WRAPPER_PATH
export QEA_OBSERVED_EXECUTED_WRAPPER_SHA256 QEA_OBSERVED_REPO_WRAPPER_SHA256
echo "identity guard: PASS"
# GPU identity and memory on the allocated node, recorded before any model load.
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version --format=csv || true
case "${STAGE}" in
  smoke-*|calib-gemma|ident-gemma)
    # Monitoring only: sampled usage every 2 s, not a measured peak. Ends with the job.
    GPUMEM="logs/efbpt_qea_triage_gpu_${SLURM_JOB_ID:-none}_gpumem.csv"
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -lms 2000 > "${GPUMEM}" 2>&1 &
    echo "gpu memory sampler every 2 s -> ${GPUMEM}"
    ;;
esac

"${PY}" -B -s "${DIR}/efbpt_qea_triage_v1_test.py"

case "${STAGE}" in
  smoke-llama) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" smoke --backend llama ;;
  smoke-llama-so) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" smoke --backend llama --constrained ;;
  smoke-gemma) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" smoke --backend gemma ;;
  t1)          exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" t1 ;;
  t2)          exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" t2 ;;
  r1)          exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" r1 --batch-size 4 ;;
  calib-gemma) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" calib-run ;;
  ident-gemma) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" ident-run ;;
esac
