#!/bin/bash
#SBATCH --job-name=c_probe
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=c_probe_%j.log

source $(conda info --base)/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench
python eval/error_analysis_tests/c_probe_run_full.py