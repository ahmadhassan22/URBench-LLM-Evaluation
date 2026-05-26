#!/bin/bash
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --job-name=qalb_all
#SBATCH --output=/mnt/home/user41/URBench/eval/qalb_all.log
#SBATCH --error=/mnt/home/user41/URBench/eval/qalb_all.err

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate urbench_eval
cd /mnt/home/user41/URBench/eval

echo "=== Starting StrategyQA ===" && python eval_strategyqa_cot_qalb.py
echo "=== Starting BoolQ ===" && python eval_boolq_cot_qalb.py
echo "=== Starting CSQA ===" && python eval_csqa_cot_qalb.py
echo "=== Starting PIQA ===" && python eval_piqa_cot_qalb.py
echo "=== Starting GSM8K ===" && python eval_gsm8k_cot_qalb.py
echo "=== ALL DONE ==="
