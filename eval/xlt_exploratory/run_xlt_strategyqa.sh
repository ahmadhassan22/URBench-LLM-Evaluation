#!/bin/bash
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --job-name=xlt_stra_v2
#SBATCH --output=/mnt/home/user41/URBench/eval/xlt_strategyqa_v2_slurm.log
#SBATCH --error=/mnt/home/user41/URBench/eval/xlt_strategyqa_v2_slurm.err

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval
cd /mnt/home/user41/URBench/eval
python strategyqa_xlt_qwen3_14b.py
