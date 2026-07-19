#!/bin/bash
#SBATCH --job-name=s2_pilot
#SBATCH --partition=q_intel_share_L20,q_intel_share_A6000
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=s2_pilot_%j.log

source $(conda info --base)/etc/profile.d/conda.sh
conda activate urbench_eval

cd /mnt/home/user41/URBench
python eval/error_analysis_tests/efbpt_stage2_pilot.py
