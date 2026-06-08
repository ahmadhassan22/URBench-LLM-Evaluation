#!/bin/bash
#SBATCH --job-name=sdfr_gsm8k_qalb
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/user41/URBench/eval/logs/sdfr_gsm8k_qalb_%j.log

conda run -n urbench_eval python /mnt/home/user41/URBench/eval/sdfr_gsm8k_qalb.py
