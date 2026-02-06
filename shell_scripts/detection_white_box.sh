#!/bin/bash
#SBATCH --job-name=specdetect_white
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/white_box_%j.out
#SBATCH --error=slurm_logs/white_box_%j.err

# Setup environment
echo "$(date), Setup the environment ..."
set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate specdetect

# Change to project directory
cd /home/s2021102349/specDetect-main

# Create output directories
mkdir -p slurm_logs
mkdir -p experiment_results/statistic_detection_results/white
mkdir -p experiment_results/detectgpt_detection_results/white
mkdir -p experiment_results/npr_detection_results/white
mkdir -p experiment_results/dna_gpt_detection_results/white
mkdir -p experiment_results/fast_detectgpt_detection_results/white
mkdir -p experiment_results/lastde_doubleplus_detection_results/white
mkdir -p experiment_results/specdetect_detection_results/white

# Paths
datasets_path=datasets/human_llm_data_for_experiment
perturbation_datasets_path=datasets/perturbation_data_detectgpt_npr
regeneration_datasets_path=datasets/regeneration_data_dnagpt

# Results folders
statistic_detection_results_path=experiment_results/statistic_detection_results
detectgpt_results_path=experiment_results/detectgpt_detection_results
npr_results_path=experiment_results/npr_detection_results
dnagpt_results_path=experiment_results/dna_gpt_detection_results
fast_detectgpt_detection_results_path=experiment_results/fast_detectgpt_detection_results
lastde_doubleplus_detection_results_path=experiment_results/lastde_doubleplus_detection_results
specdetect_doubleplus_detection_results_path=experiment_results/specdetect_detection_results

# White-box: source model = scoring model (open-source models only)
# Models with datasets available (1 GPU models)
datasets="xsum squad writing"
source_models_1gpu="gpt2_xl gptneo_2.7b opt_2.7b gptj_6b bloom_7b falcon_7b gemma_7b phi2 llama3_8b"
scenarios="white"

# ============================================================
# Statistic methods: likelihood, logrank, entropy, lrr, lastde
# ============================================================
for D in $datasets; do
  for M in $source_models_1gpu; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), Statistic detect ${D}_${M} ..."
      python py_scripts/baselines/statistic_detect.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${statistic_detection_results_path}/${scenarios} \
        --scoring_model_name ${M}
    fi
  done
done

# ============================================================
# Perturbation methods: detectgpt (only run if perturbation data exists)
# ============================================================
for D in $datasets; do
  for M in $source_models_1gpu; do
    if [ -f "${perturbation_datasets_path}/${D}_${M}_perturbation_100.raw_data.json" ]; then
      echo "$(date), DetectGPT ${D}_${M} ..."
      python py_scripts/baselines/detect_gpt.py \
        --dataset_file ${perturbation_datasets_path}/${D}_${M} \
        --output_file ${detectgpt_results_path}/${scenarios} \
        --main_results --scoring_model_name ${M}
    fi
  done
done

# NPR (only run if perturbation data exists)
for D in $datasets; do
  for M in $source_models_1gpu; do
    if [ -f "${perturbation_datasets_path}/${D}_${M}_perturbation_100.raw_data.json" ]; then
      echo "$(date), NPR ${D}_${M} ..."
      python py_scripts/baselines/detect_npr.py \
        --dataset_file ${perturbation_datasets_path}/${D}_${M} \
        --output_file ${npr_results_path}/${scenarios} \
        --main_results --scoring_model_name ${M}
    fi
  done
done

# DNA-GPT (only run if regeneration data exists)
for D in $datasets; do
  for M in $source_models_1gpu; do
    if [ -f "${regeneration_datasets_path}/${D}_${M}_regeneration_10_${scenarios}.raw_data.json" ]; then
      echo "$(date), DNA-GPT ${D}_${M} ..."
      python py_scripts/baselines/dna_gpt.py \
        --dataset_file ${regeneration_datasets_path}/${D}_${M} \
        --output_file ${dnagpt_results_path}/${scenarios} \
        --scoring_model_name ${M} --scenario ${scenarios}
    fi
  done
done

# ============================================================
# Sampling methods: fast_detectgpt, lastde++, specdetect++
# ============================================================
for D in $datasets; do
  for M in $source_models_1gpu; do
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
  for M in $source_models_1gpu; do
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
  for M in $source_models_1gpu; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), SpecDetect++ ${D}_${M} ..."
      python py_scripts/baselines/specdetect_doubleplus.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${specdetect_doubleplus_detection_results_path}/${scenarios} \
        --reference_model_name ${M} --scoring_model_name ${M}
    fi
  done
done

echo "$(date), White-box experiments completed!"
