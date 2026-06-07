#!/bin/bash
#SBATCH --job-name=boolq_idx
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/user41/URBench/eval/logs/boolq_large_index_%j.log

conda run -n urbench_eval python /mnt/home/user41/URBench/eval/sdfr_build_boolq_large_index.py
