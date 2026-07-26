#!/bin/bash
#SBATCH --job-name=pa_draft
#SBATCH --partition=q_intel_share_L20,q_intel_share_A6000
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=pa_draft_%j.log

source $(conda info --base)/etc/profile.d/conda.sh
conda activate urbench_eval
cd /mnt/home/user41/URBench

# TEST-5 first. For the full 100, remove "--limit 5".
python eval/error_analysis_tests/efbpt/efbpt_plan_a_draft.py
