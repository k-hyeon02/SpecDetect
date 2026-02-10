import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
import time
import os
from scoring_methods import fastMDE
from utils.metrics import get_roc_metrics, get_precision_recall_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTJConfig, GPTJForCausalLM
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import warnings
import csv
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.signal import stft
import csv
import pandas as pd
warnings.filterwarnings('ignore')

# os.chdir("......") # cache_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
from model_config import model_fullnames, no_remote_code

def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in ['gptj_6b', 'llama1_13b', 'llama2_13b', 'llama3_8b', 'falcon_7b', 'bloom_7b', 'opt_13b', 'gemma_7b', 'qwen1.5_7b', 'yi1.5_6b']:
        model_kwargs.update(dict(dtype=torch.float16))
    if 'gptj' in model_name:
        model_kwargs.update(dict(revision='float16'))
    if 'falcon3-3b' in model_name:
        model_kwargs.update(dict(dtype=torch.float32))

    model = AutoModelForCausalLM.from_pretrained(model_fullname, **model_kwargs, device_map="auto", trust_remote_code=model_name not in no_remote_code)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]

    optional_tok_kwargs = {}
    if "opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'

    base_tokenizer = AutoTokenizer.from_pretrained(model_fullname, **optional_tok_kwargs, trust_remote_code=model_name not in no_remote_code)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def load_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    path_prefix = "."
    data_file = f"{input_file}.raw_data.json"
    # data_file = path_prefix + data_file
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def get_samples(logits, labels, args):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = args.n_samples
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    # lprobs = torch.softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood

def get_lastde(log_likelihood, args):
    embed_size = args.embed_size
    epsilon = int(args.epsilon * log_likelihood.shape[1])
    tau_prime = args.tau_prime

    templl = log_likelihood.mean(dim=1)
    aggmde = fastMDE.get_tau_multiscale_DE(ori_data = log_likelihood, embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime)
    lastde = templl
    return lastde

def get_sampling_discrepancy(logits_ref, logits_score, labels, args):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels, args)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)


    # lastde
    lastde_x = get_lastde(log_likelihood_x, args)
    sampled_lastde = get_lastde(log_likelihood_x_tilde, args)

    miu_tilde = sampled_lastde.mean()
    sigma_tilde = sampled_lastde.std()
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy.cpu().item()

def get_sampling_discrepancy_frequency(logits_ref, logits_score, labels, args):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels, args)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)

    # lastde
    lastde_x = get_freq_features(log_likelihood_x.cpu().numpy())
    sampled_lastde = get_freq_features(log_likelihood_x_tilde.cpu().numpy())

    miu_tilde = sampled_lastde.mean(axis=0)
    sigma_tilde = sampled_lastde.std(axis=0)
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy.item()

def get_log_likelihood(logits_score, labels):
    # assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels, args)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)


    # lastde
    lastde_x = get_lastde(log_likelihood_x, args)
    sampled_lastde = get_lastde(log_likelihood_x_tilde, args)

    miu_tilde = sampled_lastde.mean()
    sigma_tilde = sampled_lastde.std()
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy.cpu().item()


def get_freq_features(log_likelihood):

    log_likelihood = np.swapaxes(log_likelihood, 0, 2)  # 先交换第0和第3维
    log_likelihood = np.squeeze(log_likelihood, axis=-1)

    log_likelihood = log_likelihood - np.mean(log_likelihood, axis=1, keepdims=True)

    N = log_likelihood.shape[1]

    # fft analysis
    fft_y_orig = fft(log_likelihood)

    power_half_y = (((np.abs(fft_y_orig))/N) ** 2)[:,:(int(N/2))]

    energy_total = np.sum(power_half_y, axis=1)

    return -energy_total


def visualize_fft_stft(log_likelihood_human, log_likelihood_llm, save_dir, dataset_name, model_name, sample_idx=0):
    """Visualize FFT and STFT results for a single human vs LLM sample pair and save as PNG."""
    os.makedirs(save_dir, exist_ok=True)

    # Prepare 1D signals (take the first sample from the batch)
    human_signal = log_likelihood_human.squeeze()
    llm_signal = log_likelihood_llm.squeeze()

    # Zero-mean
    human_signal = human_signal - np.mean(human_signal)
    llm_signal = llm_signal - np.mean(llm_signal)

    N_h = len(human_signal)
    N_l = len(llm_signal)

    # --- FFT ---
    fft_human = fft(human_signal)
    fft_llm = fft(llm_signal)

    power_human = (np.abs(fft_human[:N_h // 2]) / N_h) ** 2
    power_llm = (np.abs(fft_llm[:N_l // 2]) / N_l) ** 2

    freq_human = np.arange(N_h // 2) / N_h
    freq_llm = np.arange(N_l // 2) / N_l

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'FFT & STFT Analysis: {dataset_name} / {model_name} (sample #{sample_idx})', fontsize=14)

    # FFT power spectrum (shared Y-axis scale)
    fft_ymax = max(power_human.max(), power_llm.max()) * 1.1

    axes[0, 0].plot(freq_human, power_human, color='blue', alpha=0.7, label='Human')
    axes[0, 0].set_title('FFT Power Spectrum - Human')
    axes[0, 0].set_xlabel('Normalized Frequency')
    axes[0, 0].set_ylabel('Power')
    axes[0, 0].set_ylim(0, fft_ymax)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freq_llm, power_llm, color='red', alpha=0.7, label='LLM')
    axes[0, 1].set_title('FFT Power Spectrum - LLM')
    axes[0, 1].set_xlabel('Normalized Frequency')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_ylim(0, fft_ymax)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- STFT (shared color scale) ---
    nperseg = min(32, N_h // 2, N_l // 2)
    if nperseg >= 4:
        f_h, t_h, Zxx_h = stft(human_signal, nperseg=nperseg, noverlap=nperseg // 2)
        f_l, t_l, Zxx_l = stft(llm_signal, nperseg=nperseg, noverlap=nperseg // 2)

        vmin = 0
        vmax = max(np.abs(Zxx_h).max(), np.abs(Zxx_l).max())

        im_h = axes[1, 0].pcolormesh(t_h, f_h, np.abs(Zxx_h), shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('STFT Spectrogram - Human')
        axes[1, 0].set_xlabel('Time (token index)')
        axes[1, 0].set_ylabel('Frequency')
        fig.colorbar(im_h, ax=axes[1, 0])

        im_l = axes[1, 1].pcolormesh(t_l, f_l, np.abs(Zxx_l), shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('STFT Spectrogram - LLM')
        axes[1, 1].set_xlabel('Time (token index)')
        axes[1, 1].set_ylabel('Frequency')
        fig.colorbar(im_l, ax=axes[1, 1])
    else:
        axes[1, 0].text(0.5, 0.5, 'Sequence too short for STFT', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'Sequence too short for STFT', ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    png_path = os.path.join(save_dir, f'fft_stft_{dataset_name}_{model_name}_sample{sample_idx}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'FFT/STFT visualization saved to {png_path}')
    return png_path


def visualize_fft_overlay(all_human_energies, all_llm_energies, save_dir, dataset_name, model_name):
    """Visualize overlaid FFT power spectra distributions for all human vs LLM samples."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'FFT Energy Distribution: {dataset_name} / {model_name}', fontsize=14)

    # Histogram of total FFT energy
    axes[0].hist(all_human_energies, bins=30, alpha=0.6, color='blue', label='Human', density=True)
    axes[0].hist(all_llm_energies, bins=30, alpha=0.6, color='red', label='LLM', density=True)
    axes[0].set_title('Sampling Discrepancy Distribution')
    axes[0].set_xlabel('Sampling Discrepancy (z-score)')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    axes[1].boxplot([all_human_energies, all_llm_energies], tick_labels=['Human', 'LLM'],
                    patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[1].set_title('Sampling Discrepancy Box Plot')
    axes[1].set_ylabel('Sampling Discrepancy (z-score)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(save_dir, f'fft_energy_dist_{dataset_name}_{model_name}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'FFT energy distribution saved to {png_path}')
    return png_path

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()

    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name)
        reference_model = load_model(args.reference_model_name)
        reference_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    # evaluate criterion
    name = "specdetect++"
    criterion_fn = get_sampling_discrepancy_frequency

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    time1 = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            original_crit = criterion_fn(logits_ref, logits_score, labels, args)
            # original_likelihood = get_log_likelihood(logits_score, labels)
            # tokenized_ori = tokenized
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)  
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, args)

        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    time2 = time.time()
    time_per_sample_ms = ((time2 - time1) / n_samples) * 1000
    print(f"Time per sample for {name}: {time_per_sample_ms:.2f} ms")

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]
                   }
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # --- FFT/STFT Visualization ---
    dataset_name = args.dataset_file.split("/")[-1]
    viz_dir = os.path.join(os.path.dirname(args.output_file) if args.output_file else '.', 'visualizations')
    # Visualize first 3 samples as individual FFT/STFT plots
    viz_count = min(3, len(data["original"]))
    for viz_idx in range(viz_count):
        orig_text = data["original"][viz_idx]
        samp_text = data["sampled"][viz_idx]
        tokenized_orig = scoring_tokenizer(orig_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        tokenized_samp = scoring_tokenizer(samp_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        with torch.no_grad():
            logits_orig = scoring_model(**tokenized_orig).logits[:, :-1]
            logits_samp = scoring_model(**tokenized_samp).logits[:, :-1]
            labels_orig = tokenized_orig.input_ids[:, 1:]
            labels_samp = tokenized_samp.input_ids[:, 1:]
            ll_orig = get_likelihood(logits_orig, labels_orig).cpu().numpy().squeeze()
            ll_samp = get_likelihood(logits_samp, labels_samp).cpu().numpy().squeeze()
        visualize_fft_stft(ll_orig, ll_samp, viz_dir, dataset_name, args.scoring_model_name, sample_idx=viz_idx)

    # Overlay distribution plot using criterion scores
    visualize_fft_overlay(predictions['real'], predictions['samples'], viz_dir, dataset_name, args.scoring_model_name)

    results = {'dataset':dataset_name,
        'model':args.reference_model_name,
        'feature index': name,
        'auc':round(roc_auc,4),
        'prauc':round(pr_auc,4),
        'n_samples': args.n_samples}

    results_file = f'{args.output_file}/{name}.csv'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'a', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=results.keys())
        if fout.tell() == 0:
            writer.writeheader()
        writer.writerow(results)
        print(f'Results appended to {results_file}')

    # Save per-sample predictions to JSON for standalone visualization
    json_file = os.path.join(viz_dir, f'predictions_{dataset_name}_{args.scoring_model_name}.json')
    with open(json_file, 'w') as f:
        json.dump({'dataset': dataset_name, 'model': args.scoring_model_name,
                   'real': predictions['real'], 'samples': predictions['samples'],
                   'roc_auc': roc_auc, 'pr_auc': pr_auc}, f)
        print(f'Predictions saved to {json_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="experiment_results/spec/xsum_gpt4turbo")
    parser.add_argument('--dataset_file', type=str, default="datasets/human_llm_data_for_experiment/xsum_gpt4turbo")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--reference_model_name', type=str, default="gptj_6b")
    parser.add_argument('--scoring_model_name', type=str, default="gptj_6b")
    parser.add_argument('--embed_size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=8)
    parser.add_argument('--tau_prime', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)