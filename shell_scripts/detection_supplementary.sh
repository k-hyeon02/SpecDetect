#!/bin/bash
#SBATCH --job-name=specdetect_supp
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/supplementary_%j.out
#SBATCH --error=slurm_logs/supplementary_%j.err

echo "$(date), Setup the environment ..."
set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate specdetect

cd /home/s2021102349/specDetect-main

mkdir -p slurm_logs
mkdir -p experiment_results/statistic_detection_results/supplementary
mkdir -p experiment_results/detectgpt_detection_results/supplementary
mkdir -p experiment_results/npr_detection_results/supplementary
mkdir -p experiment_results/fast_detectgpt_detection_results/supplementary
mkdir -p experiment_results/lastde_doubleplus_detection_results/supplementary
mkdir -p experiment_results/specdetect_detection_results/supplementary

set +e

# Results folders
statistic_results=experiment_results/statistic_detection_results/supplementary
detectgpt_results=experiment_results/detectgpt_detection_results/supplementary
npr_results=experiment_results/npr_detection_results/supplementary
fast_detectgpt_results=experiment_results/fast_detectgpt_detection_results/supplementary
lastde_results=experiment_results/lastde_doubleplus_detection_results/supplementary
specdetect_results=experiment_results/specdetect_detection_results/supplementary

# ============================================================
# 1. Paraphrasing Attack Robustness (black-box, proxy: gptj_6b)
# Dataset: xsum_gpt4turbo with PAWS paraphrasing attack
# ============================================================
paraphrase_data=datasets/paraphrasing_attack_data/xsum_gpt4turbo_paws_paraphrasing_attack
proxy=gptj_6b

if [ -f "${paraphrase_data}.raw_data.json" ]; then
  echo "$(date), === Paraphrasing Attack Evaluation ==="

  echo "$(date), Statistic detect (paraphrasing) ..."
  python py_scripts/baselines/statistic_detect.py \
    --dataset_file ${paraphrase_data} \
    --output_file ${statistic_results} \
    --scoring_model_name ${proxy}

  echo "$(date), Fast-DetectGPT (paraphrasing) ..."
  python py_scripts/baselines/fast_detect_gpt.py \
    --dataset_file ${paraphrase_data} \
    --output_file ${fast_detectgpt_results} \
    --reference_model_name ${proxy} --scoring_model_name ${proxy}

  echo "$(date), Lastde++ (paraphrasing) ..."
  python py_scripts/baselines/lastde_doubleplus.py \
    --dataset_file ${paraphrase_data} \
    --output_file ${lastde_results} \
    --reference_model_name ${proxy} --scoring_model_name ${proxy} \
    --embed_size 4 --epsilon 8 --tau_prime 15

  echo "$(date), SpecDetect++ (paraphrasing) ..."
  python py_scripts/baselines/specdetect_doubleplus.py \
    --dataset_file ${paraphrase_data} \
    --output_file ${specdetect_results} \
    --reference_model_name ${proxy} --scoring_model_name ${proxy}
fi

# ============================================================
# 2. Text Length Variation (black-box, proxy: gptj_6b)
# Dataset: xsum_gpt4turbo truncated to 160 words
# ============================================================
length_data=datasets/response_lengths_data/xsum_gpt4turbo_length_160

if [ -f "${length_data}.raw_data.json" ]; then
  echo "$(date), === Text Length Variation (160 words) ==="

  echo "$(date), Statistic detect (length_160) ..."
  python py_scripts/baselines/statistic_detect.py \
    --dataset_file ${length_data} \
    --output_file ${statistic_results} \
    --scoring_model_name ${proxy}

  echo "$(date), Fast-DetectGPT (length_160) ..."
  python py_scripts/baselines/fast_detect_gpt.py \
    --dataset_file ${length_data} \
    --output_file ${fast_detectgpt_results} \
    --reference_model_name ${proxy} --scoring_model_name ${proxy}

  echo "$(date), Lastde++ (length_160) ..."
  python py_scripts/baselines/lastde_doubleplus.py \
    --dataset_file ${length_data} \
    --output_file ${lastde_results} \
    --reference_model_name ${proxy} --scoring_model_name ${proxy} \
    --embed_size 4 --epsilon 8 --tau_prime 15

  echo "$(date), SpecDetect++ (length_160) ..."
  python py_scripts/baselines/specdetect_doubleplus.py \
    --dataset_file ${length_data} \
    --output_file ${specdetect_results} \
    --reference_model_name ${proxy} --scoring_model_name ${proxy}
fi

# ============================================================
# 3. Number of Contrast Samples Variation (white-box)
# Dataset: xsum_gemma_7b with 10 perturbations (vs standard 100)
# ============================================================
sample_num_base=datasets/sample_numbers_compare_dataset_detectgpt_npr_dnagpt/xsum_gemma_7b
sample_num_data=${sample_num_base}_perturbation_10_white

if [ -f "${sample_num_data}.raw_data.json" ]; then
  echo "$(date), === Number of Contrast Samples (n=10) ==="

  echo "$(date), DetectGPT (n_perturbations=10) ..."
  python py_scripts/baselines/detect_gpt.py \
    --dataset_file ${sample_num_base} \
    --output_file ${detectgpt_results} \
    --scoring_model_name ${proxy} --n_perturbations 10 --scenario white

  echo "$(date), NPR (n_perturbations=10) ..."
  python py_scripts/baselines/detect_npr.py \
    --dataset_file ${sample_num_base} \
    --output_file ${npr_results} \
    --scoring_model_name ${proxy} --n_perturbations 10 --scenario white
fi

echo "$(date), Supplementary experiments completed!"
