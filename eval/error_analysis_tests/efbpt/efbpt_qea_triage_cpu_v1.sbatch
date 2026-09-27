#!/bin/bash
#SBATCH --job-name=efbpt_qea_triage_cpu
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/mnt/home/user41/URBench
#SBATCH --output=/mnt/home/user41/URBench/logs/efbpt_qea_triage_cpu_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/efbpt_qea_triage_cpu_%j.err
#
# QEA development triage, CPU stages. One stage per job; nothing chains automatically.
#   t0 : exact-title corpus status (streams and hashes the 25.9 GB frozen metadata once).
#   t3 | t4 | t5 : deterministic join, review packets, acceptance (seconds).
# The offline test suite must pass before any stage starts. No git, no network, no model.
set -euo pipefail

STAGE="${1:-}"
case "${STAGE}" in
  t0|t3|t4|t5) ;;
  *) echo "REFUSING: unknown or missing stage '${STAGE}'" >&2; exit 2 ;;
esac

PY=/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python
DIR=eval/error_analysis_tests/efbpt
export PYTHONDONTWRITEBYTECODE=1

echo "stage=${STAGE} job=${SLURM_JOB_ID:-none} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Code-identity guard (same contract as the GPU wrapper): fail closed before tests or any stage input.
# Expected hashes are captured at submission and exported (QEA_EXPECT_RUNNER_SHA256,
# QEA_EXPECT_TEST_SHA256, QEA_EXPECT_WRAPPER_SHA256); none is embedded in any file. The wrapper hash is
# taken from the script actually executed ($0, Slurm's spool copy), not only from the repository copy.
# The runner re-checks all three and records them in the stage seal.
for VAR in QEA_EXPECT_RUNNER_SHA256 QEA_EXPECT_TEST_SHA256 QEA_EXPECT_WRAPPER_SHA256; do
  if [[ ! "${!VAR:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "REFUSING: ${VAR} missing or malformed" >&2; exit 3
  fi
done
QEA_OBSERVED_RUNNER_SHA256="$(sha256sum "${DIR}/efbpt_qea_triage_v1.py" | cut -d' ' -f1)"
QEA_OBSERVED_TEST_SHA256="$(sha256sum "${DIR}/efbpt_qea_triage_v1_test.py" | cut -d' ' -f1)"
QEA_EXECUTED_WRAPPER_PATH="$(readlink -f "$0")"
QEA_OBSERVED_EXECUTED_WRAPPER_SHA256="$(sha256sum "${QEA_EXECUTED_WRAPPER_PATH}" | cut -d' ' -f1)"
QEA_OBSERVED_REPO_WRAPPER_SHA256="$(sha256sum "${DIR}/efbpt_qea_triage_cpu_v1.sbatch" | cut -d' ' -f1)"
echo "identity runner=${QEA_OBSERVED_RUNNER_SHA256} tests=${QEA_OBSERVED_TEST_SHA256}"
echo "identity executed_wrapper=${QEA_OBSERVED_EXECUTED_WRAPPER_SHA256} path=${QEA_EXECUTED_WRAPPER_PATH}"
echo "identity repository_wrapper=${QEA_OBSERVED_REPO_WRAPPER_SHA256} (recorded; the executed copy is authoritative)"
[ "${QEA_OBSERVED_RUNNER_SHA256}" = "${QEA_EXPECT_RUNNER_SHA256}" ] || { echo "REFUSING: runner hash mismatch" >&2; exit 3; }
[ "${QEA_OBSERVED_TEST_SHA256}" = "${QEA_EXPECT_TEST_SHA256}" ] || { echo "REFUSING: test hash mismatch" >&2; exit 3; }
[ "${QEA_OBSERVED_EXECUTED_WRAPPER_SHA256}" = "${QEA_EXPECT_WRAPPER_SHA256}" ] || { echo "REFUSING: executed wrapper hash mismatch" >&2; exit 3; }
export QEA_OBSERVED_RUNNER_SHA256 QEA_OBSERVED_TEST_SHA256 QEA_EXECUTED_WRAPPER_PATH
export QEA_OBSERVED_EXECUTED_WRAPPER_SHA256 QEA_OBSERVED_REPO_WRAPPER_SHA256
echo "identity guard: PASS"

"${PY}" -B -s "${DIR}/efbpt_qea_triage_v1_test.py"

exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" "${STAGE}"
