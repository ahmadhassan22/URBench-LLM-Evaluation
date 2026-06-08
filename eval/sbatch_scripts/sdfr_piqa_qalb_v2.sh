#!/bin/bash
#SBATCH --job-name=qalb_piqa_v2
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/user41/URBench/eval/logs/sdfr_piqa_qalb_v2_%j.log

conda run -n urbench_eval python /mnt/home/user41/URBench/eval/sdfr_piqa_qalb_v2.py
