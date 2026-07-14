#!/bin/bash
#SBATCH --job-name=entityground12v2
#SBATCH --partition=q_intel_share_L20
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/entityground12_v2_%j.out
#SBATCH --error=/mnt/home/user41/URBench/logs/entityground12_v2_%j.err

set -euo pipefail
cd /mnt/home/user41/URBench
export PYTHONUNBUFFERED=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate urbench_eval

echo "node: $(hostname)  date: $(date)"
echo "python: $(which python)"
echo "stage: CPU grounding and offline audit only; no model generation"

python eval/error_analysis_tests/entity_first_ground_audit_v2.py --stage self-test
python eval/error_analysis_tests/entity_first_ground_audit_v2.py \
  --stage run \
  --device-embed cpu

echo "entity-first ground/audit v2 completed: $(date)"
