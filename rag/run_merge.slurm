#!/bin/bash
#SBATCH --job-name=wiki_merge
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/home/user41/URBench/logs/merge_%j.log
#SBATCH --error=/mnt/home/user41/URBench/logs/merge_%j.err

cd /mnt/home/user41/URBench
conda run -n urbench_eval python rag/merge_shards.py
