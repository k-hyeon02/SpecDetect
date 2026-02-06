#!/bin/bash
#SBATCH --job-name=specdetect_test
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/test_%j.out
#SBATCH --error=slurm_logs/test_%j.err

# Quick test: gpt2_xl + xsum with statistic_detect
echo "$(date), Test run starting ..."
set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate specdetect

cd /home/s2021102349/specDetect-main

mkdir -p slurm_logs
mkdir -p experiment_results/statistic_detection_results/white

echo "$(date), Running statistic_detect with gpt2_xl on xsum ..."
python py_scripts/baselines/statistic_detect.py \
  --dataset_file datasets/human_llm_data_for_experiment/xsum_gpt2_xl \
  --output_file experiment_results/statistic_detection_results/white \
  --scoring_model_name gpt2_xl

echo "$(date), Test completed successfully!"
