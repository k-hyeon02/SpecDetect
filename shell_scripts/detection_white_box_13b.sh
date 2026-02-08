#!/bin/bash
#SBATCH --job-name=specdetect_white_13b
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/white_box_13b_%j.out
#SBATCH --error=slurm_logs/white_box_13b_%j.err

# Setup environment
echo "$(date), Setup the environment (13B models, 2 GPUs) ..."
set -e  # Fail on setup errors

source $(conda info --base)/etc/profile.d/conda.sh
conda activate specdetect

cd /home/s2021102349/specDetect-main

mkdir -p slurm_logs
mkdir -p experiment_results/statistic_detection_results/white
mkdir -p experiment_results/fast_detectgpt_detection_results/white
mkdir -p experiment_results/lastde_doubleplus_detection_results/white
mkdir -p experiment_results/specdetect_detection_results/white

# Disable set -e for experiments so individual failures don't kill the whole job
set +e

datasets_path=datasets/human_llm_data_for_experiment
statistic_detection_results_path=experiment_results/statistic_detection_results
fast_detectgpt_detection_results_path=experiment_results/fast_detectgpt_detection_results
lastde_doubleplus_detection_results_path=experiment_results/lastde_doubleplus_detection_results
specdetect_doubleplus_detection_results_path=experiment_results/specdetect_detection_results

datasets="xsum squad writing"
source_models_2gpu="llama1_13b llama2_13b opt_13b"
scenarios="white"

# Statistic methods
for D in $datasets; do
  for M in $source_models_2gpu; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), Statistic detect ${D}_${M} ..."
      python py_scripts/baselines/statistic_detect.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${statistic_detection_results_path}/${scenarios} \
        --scoring_model_name ${M}
    fi
  done
done

# Sampling methods
for D in $datasets; do
  for M in $source_models_2gpu; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), Fast-DetectGPT ${D}_${M} ..."
      python py_scripts/baselines/fast_detect_gpt.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${fast_detectgpt_detection_results_path}/${scenarios} \
        --reference_model_name ${M} --scoring_model_name ${M}
    fi
  done
done

for D in $datasets; do
  for M in $source_models_2gpu; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), Lastde++ ${D}_${M} ..."
      python py_scripts/baselines/lastde_doubleplus.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${lastde_doubleplus_detection_results_path}/${scenarios} \
        --reference_model_name ${M} --scoring_model_name ${M} \
        --embed_size 4 --epsilon 8 --tau_prime 15
    fi
  done
done

for D in $datasets; do
  for M in $source_models_2gpu; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), SpecDetect++ ${D}_${M} ..."
      python py_scripts/baselines/specdetect_doubleplus.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${specdetect_doubleplus_detection_results_path}/${scenarios} \
        --reference_model_name ${M} --scoring_model_name ${M}
    fi
  done
done

echo "$(date), White-box 13B experiments completed!"
