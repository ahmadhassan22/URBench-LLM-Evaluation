#!/bin/bash
#SBATCH --job-name=phaseR_gen_v2
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/phaseR_gen_v2_%j.out
#SBATCH --error=/mnt/home/user41/URBench/logs/phaseR_gen_v2_%j.err

set -euo pipefail
cd /mnt/home/user41/URBench
export PYTHONUNBUFFERED=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate urbench_eval

echo "node: $(hostname)  date: $(date)"
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python eval/error_analysis_tests/phase_r_dev50_v2.py --stage self-test
python eval/error_analysis_tests/phase_r_dev50_v2.py \
  --stage generate \
  --batch-size 8 \
  --gpu-memory-utilization 0.80 \
  --max-parse-failures 5

echo "generation stage completed: $(date)"
