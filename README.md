# SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis

**AAAI 2026 Main Track Oral**

SpecDetect is a training-free method for detecting LLM-generated text based on spectral analysis. It applies FFT on token-level log-likelihood sequences to extract frequency-domain features, enabling robust discrimination between human-written and machine-generated text. Built on the [Lastde](https://github.com/TrustMedia-zju/Lastde_Detector) framework.

## Project Structure

```
specDetect-main/
├── py_scripts/
│   ├── baselines/                        # Detection method implementations
│   │   ├── model_config.py               # Centralized model config (HF Hub IDs, GPU requirements)
│   │   ├── specdetect_doubleplus.py      # SpecDetect++ (main contribution)
│   │   ├── statistic_detect.py           # Statistical methods (likelihood, logrank, entropy, LRR, LastDE)
│   │   ├── fast_detect_gpt.py            # Fast-DetectGPT
│   │   ├── lastde_doubleplus.py          # LastDE++
│   │   ├── detect_gpt.py                 # DetectGPT (perturbation-based)
│   │   ├── detect_npr.py                 # NPR (perturbation-based)
│   │   ├── dna_gpt.py                    # DNA-GPT (regeneration-based)
│   │   ├── binoculars.py                 # Binoculars
│   │   ├── scoring_methods/
│   │   │   ├── based_scoring_baselines.py  # Shared scoring utilities
│   │   │   ├── fastMDE.py                  # Fast Multi-scale Density Entropy
│   │   │   └── bart_score.py               # BART-based scoring
│   │   └── utils/
│   │       └── metrics.py                # ROC AUC, PR AUC evaluation
│   ├── data_generations/                 # LLM text generation scripts
│   │   ├── data_generation_opensource.py
│   │   ├── data_generation_perturbation.py
│   │   ├── data_generation_perturbation_parallel.py
│   │   ├── data_generation_paraphrasing.py
│   │   ├── data_generation_nonenglish.py
│   │   ├── data_generation_response_length.py
│   │   └── merge_results.py
│   └── visualize_specdetect.py           # Standalone plot regeneration script
├── shell_scripts/                        # SLURM job scripts
│   ├── detection_white_box.sh            # White-box experiments (1 GPU, ≤8B models)
│   ├── detection_white_box_13b.sh        # White-box experiments (2 GPUs, 13B models)
│   ├── detection_black_box.sh            # Black-box experiments (proxy model: gptj_6b)
│   ├── rerun_specdetect_white.sh         # Re-run SpecDetect++ only (white-box)
│   ├── rerun_specdetect_black.sh         # Re-run SpecDetect++ only (black-box)
│   └── test_single_model.sh             # Quick single-model test
├── datasets/
│   ├── human_llm_data_for_experiment/    # Main datasets (55 JSON files)
│   ├── perturbation_data_detectgpt_npr/  # Perturbation data (xsum_llama3_8b only)
│   ├── regeneration_data_dnagpt/         # Regeneration data (xsum_llama3_8b only)
│   ├── human_original_data/
│   ├── decoding_strategies_data/
│   ├── paraphrasing_attack_data/
│   ├── multi_language_data/
│   ├── response_lengths_data/
│   └── sample_numbers_compare_dataset_detectgpt_npr_dnagpt/
├── experiment_results/
│   ├── specdetect_detection_results/     # SpecDetect++ results + visualizations
│   ├── statistic_detection_results/      # Statistical method results
│   ├── fast_detectgpt_detection_results/
│   ├── lastde_doubleplus_detection_results/
│   ├── detectgpt_detection_results/
│   ├── npr_detection_results/
│   └── dna_gpt_detection_results/
├── requirements.txt
└── README.md
```

## Environment Setup

### Prerequisites
- SLURM cluster with NVIDIA GPUs (tested on RTX 3090 24GB)
- Conda package manager

### Installation

```bash
# Create conda environment
conda create -n specdetect python=3.12 -y
conda activate specdetect

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
cd specDetect-main
pip install -r requirements.txt
```

### Verified Environment
- Python 3.12, PyTorch 2.6.0+cu124, Transformers 5.x
- SLURM cluster: `sichpc` partition, nodes with 2x RTX 3090 (24GB each)

## Supported Models

Models are configured in `py_scripts/baselines/model_config.py` with HuggingFace Hub IDs:

| Model | HuggingFace ID | GPUs |
|-------|---------------|------|
| gpt2_xl | `openai-community/gpt2-xl` | 1 |
| gptneo_2.7b | `EleutherAI/gpt-neo-2.7B` | 1 |
| opt_2.7b | `facebook/opt-2.7b` | 1 |
| gptj_6b | `EleutherAI/gpt-j-6b` | 1 |
| bloom_7b | `bigscience/bloom-7b1` | 1 |
| falcon_7b | `tiiuae/falcon-7b` | 1 |
| gemma_7b | `google/gemma-7b` | 1 |
| phi2 | `microsoft/phi-2` | 1 |
| llama3_8b | `meta-llama/Meta-Llama-3-8B` | 1 |
| llama1_13b | `huggyllama/llama-13b` | 2 |
| llama2_13b | `TheBloke/Llama-2-13B-fp16` | 2 |
| opt_13b | `facebook/opt-13b` | 2 |

Closed-source models (gpt4turbo, gpt4o, claude3haiku) have pre-generated datasets but no scoring models -- they are evaluated only in **black-box** mode using a proxy model.

## Detection Methods

### SpecDetect++ (Main Contribution)
Applies FFT on token-level log-likelihood sequences to extract frequency-domain energy features. Uses a sampling discrepancy framework (z-score) to distinguish human vs. LLM text.

### Baselines
| Category | Method | Script |
|----------|--------|--------|
| Statistical | Likelihood, LogRank, Entropy, LRR, LastDE | `py_scripts/baselines/statistic_detect.py` |
| Sampling | Fast-DetectGPT | `py_scripts/baselines/fast_detect_gpt.py` |
| Sampling | LastDE++ | `py_scripts/baselines/lastde_doubleplus.py` |
| Perturbation | DetectGPT | `py_scripts/baselines/detect_gpt.py` |
| Perturbation | NPR | `py_scripts/baselines/detect_npr.py` |
| Regeneration | DNA-GPT | `py_scripts/baselines/dna_gpt.py` |
| Perplexity | Binoculars | `py_scripts/baselines/binoculars.py` |

## Running Experiments

### White-box Detection
The source model is used as the scoring model. Open-source models only.

```bash
# Models ≤ 8B (1 GPU)
sbatch shell_scripts/detection_white_box.sh

# 13B models (2 GPUs)
sbatch shell_scripts/detection_white_box_13b.sh
```

### Black-box Detection
A fixed proxy model (`gptj_6b`) is used as the scoring model for all datasets, including closed-source model outputs.

```bash
sbatch shell_scripts/detection_black_box.sh
```

### Run a Single Method / Model

```bash
conda activate specdetect
cd specDetect-main

# SpecDetect++ (white-box, xsum + gpt2_xl)
python py_scripts/baselines/specdetect_doubleplus.py \
  --dataset_file datasets/human_llm_data_for_experiment/xsum_gpt2_xl \
  --output_file experiment_results/specdetect_detection_results/white \
  --reference_model_name gpt2_xl --scoring_model_name gpt2_xl

# Statistical methods (black-box, reddit + gpt4turbo, proxy: gptj_6b)
python py_scripts/baselines/statistic_detect.py \
  --dataset_file datasets/human_llm_data_for_experiment/reddit_gpt4turbo \
  --output_file experiment_results/statistic_detection_results/black \
  --scoring_model_name gptj_6b
```

### Datasets

Main experiment data: `datasets/human_llm_data_for_experiment/{domain}_{model}.raw_data.json`

- **Domains**: xsum, squad, writing, reddit
- **Models**: 15 models (12 open-source + 3 closed-source)
- **Format**: JSON with `original` (human) and `sampled` (LLM) text arrays

Perturbation/regeneration data exists only for `xsum_llama3_8b`.

## Results & Visualization

### Output Structure
Results are saved as CSV files in `experiment_results/{method}_detection_results/{white|black}/`.

SpecDetect++ additionally saves:
- Per-sample prediction JSONs in `experiment_results/specdetect_detection_results/visualizations/`
- Sampling discrepancy distribution plots (histogram + box plot) as PNG files

### Regenerate Plots (Without Model Inference)

```bash
# Regenerate all plots from saved prediction JSONs
python py_scripts/visualize_specdetect.py

# Filter by specific dataset/model
python py_scripts/visualize_specdetect.py --filter xsum_gpt2_xl

# Custom input/output directories
python py_scripts/visualize_specdetect.py \
  --input_dir experiment_results/specdetect_detection_results/visualizations \
  --output_dir my_custom_plots/
```

### Monitor SLURM Jobs

```bash
squeue -u $USER                          # Check running jobs
tail -f slurm_logs/white_box_<JOBID>.out # Follow job output
```

## Architecture

```
Text → Tokenization → Log-likelihood extraction (scoring model)
  → FFT spectral analysis → Power spectrum → Total energy
  → Sampling discrepancy (z-score) → ROC/PR AUC metrics
```

1. **Scoring**: Each token's log-likelihood is computed using a causal LM
2. **FFT**: The log-likelihood sequence is transformed to frequency domain via FFT
3. **Energy**: Total power spectrum energy is computed as a scalar feature
4. **Sampling discrepancy**: A z-score is computed by comparing the original energy against resampled versions under the model distribution
5. **Classification**: Threshold on the z-score to separate human vs. LLM text

## Acknowledgements

We are grateful to the authors of [Lastde](https://github.com/TrustMedia-zju/Lastde_Detector) for their open-source code and insightful research. Their work laid the foundation for the implementation of SpecDetect.

## Citation

```bibtex
@article{luo2025specdetect,
  title={SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis},
  author={Luo, Haitong and Zhang, Weiyao and Wang, Suhang and Zou, Wenji and Lin, Chungang and Meng, Xuying and Zhang, Yujun},
  journal={arXiv preprint arXiv:2508.11343},
  year={2025}
}
```
