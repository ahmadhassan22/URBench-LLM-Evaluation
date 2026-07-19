#!/bin/bash
#SBATCH --job-name=b30_annot
#SBATCH --partition=q_intel_share_L20,q_intel_share_A6000
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=b30_annot_%j.log
source $(conda info --base)/etc/profile.d/conda.sh
conda activate urbench_eval
cd /mnt/home/user41/URBench
python eval/error_analysis_tests/efbpt_blind30_annotate.py