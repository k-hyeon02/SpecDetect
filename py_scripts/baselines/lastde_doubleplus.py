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
import warnings
import csv
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
    data_file = f"{input_file}.raw_data.json"
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
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood

def get_lastde(log_likelihood, args):
    embed_size = args.embed_size
    epsilon = int(args.epsilon * log_likelihood.shape[1])
    tau_prime = args.tau_prime

    templl = log_likelihood.mean(dim=1)

    aggmde = fastMDE.get_tau_multiscale_DE(ori_data = log_likelihood, embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime)
    lastde = templl / aggmde 
    # print("templ:", templl, "aggmde:", aggmde, "lastde:", lastde)
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
    name = "lastde_doubleplus"
    criterion_fn = get_sampling_discrepancy

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
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    
    # results
    # results_file = os.getcwd() + f'{args.output_file}.{name}.json'
    # results_file = f'{args.output_file}.{name}.json'
    # results = { 'name': f'{name}_threshold',
    #             'info': {'n_samples': n_samples},
    #             # 'predictions': predictions,
    #             # 'raw_results': results,
    #             # 'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
    #             # 'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
    #             'roc_auc': roc_auc,
    #             'pr_auc': pr_auc,
    #             # 'loss': 1 - pr_auc
    #             }
    # with open(results_file, 'w') as fout:
    #     json.dump(results, fout)
    #     print(f'Results written into {results_file}')
    results = {'dataset':args.dataset_file.split("/")[-1],
        'model':args.reference_model_name,
        'feature index': name,
        'auc':round(roc_auc,4),
        'prauc':round(pr_auc,4),
        'n_samples': args.n_samples}

    results_file = f'{args.output_file}/{name}.csv'
    with open(results_file, 'a', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=results.keys())
        
        # 若文件为空，先写入表头
        if fout.tell() == 0:
            writer.writeheader()
        
        # 写入数据行
        writer.writerow(results)
        print(f'Results appended to {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="experiment_results/lastde_doubleplus_detection_results/black")
    parser.add_argument('--dataset_file', type=str, default="datasets/human_llm_data_for_experiment/writing_phi-4")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--reference_model_name', type=str, default="gptj_6b")
    parser.add_argument('--scoring_model_name', type=str, default="gptj_6b")
    parser.add_argument('--embed_size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=8)
    parser.add_argument('--tau_prime', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)
