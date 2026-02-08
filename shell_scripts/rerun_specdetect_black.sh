#!/bin/bash
#SBATCH --job-name=specdetect_rerun_b
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/rerun_specdetect_black_%j.out
#SBATCH --error=slurm_logs/rerun_specdetect_black_%j.err

echo "$(date), Re-running SpecDetect++ (black-box) ..."
set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate specdetect

cd /home/s2021102349/specDetect-main
mkdir -p experiment_results/specdetect_detection_results/black

set +e

datasets_path=datasets/human_llm_data_for_experiment
specdetect_path=experiment_results/specdetect_detection_results

datasets="xsum writing reddit"
source_models="gpt2_xl gptneo_2.7b opt_2.7b llama1_13b llama2_13b llama3_8b opt_13b bloom_7b gptj_6b falcon_7b gemma_7b phi2 claude3haiku gpt4turbo gpt4o"
proxy_model="gptj_6b"

for D in $datasets; do
  for M in $source_models; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), SpecDetect++ ${D}_${M} ..."
      python py_scripts/baselines/specdetect_doubleplus.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${specdetect_path}/black \
        --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}
    fi
  done
done

echo "$(date), SpecDetect++ black-box rerun completed!"
