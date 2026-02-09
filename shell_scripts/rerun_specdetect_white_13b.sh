#!/bin/bash
#SBATCH --job-name=specdetect_rerun_13b
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/rerun_specdetect_white_13b_%j.out
#SBATCH --error=slurm_logs/rerun_specdetect_white_13b_%j.err

echo "$(date), Re-running SpecDetect++ (white-box, 13B models, 2 GPUs) ..."
set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate specdetect

cd /home/s2021102349/specDetect-main

set +e

datasets_path=datasets/human_llm_data_for_experiment
specdetect_results_path=experiment_results/specdetect_detection_results

datasets="xsum squad writing"
source_models_2gpu="llama1_13b llama2_13b opt_13b"

for D in $datasets; do
  for M in $source_models_2gpu; do
    if [ -f "${datasets_path}/${D}_${M}.raw_data.json" ]; then
      echo "$(date), SpecDetect++ ${D}_${M} ..."
      python py_scripts/baselines/specdetect_doubleplus.py \
        --dataset_file ${datasets_path}/${D}_${M} \
        --output_file ${specdetect_results_path}/white \
        --reference_model_name ${M} --scoring_model_name ${M}
    fi
  done
done

echo "$(date), SpecDetect++ white-box 13B rerun completed!"
