#!/bin/bash
#SBATCH --job-name=wiki_index
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/index_full_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/index_full_%j.err

cd /mnt/home/user41/URBench
conda run -n urbench_eval python rag/build_index_full.py
