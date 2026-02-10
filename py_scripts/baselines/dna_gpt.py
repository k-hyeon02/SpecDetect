from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import  torch
from utils.metrics import get_roc_metrics, get_precision_recall_metrics
import numpy as np
from tqdm import tqdm
import json
import time
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

def load_data(args):
    # data_file = f"{dataset_file}_regeneration_{args.n_regenerations}_{args.scenario}.raw_data.json"
    data_file = f"{args.dataset_file}_regeneration_{args.n_regenerations}_{args.scenario}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()


def get_dna_gpt(args, scoring_model, scoring_tokenizer, text, rewrote_texts):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
        labels = tokenized.input_ids[:, 1:]
        logits = scoring_model(**tokenized).logits[:, :-1]
        ori_likelihood = get_likelihood(logits, labels)
        regeneration_likelihood = []
        for rewrote_text in rewrote_texts:
            tokenized = scoring_tokenizer(rewrote_text, return_tensors="pt", return_token_type_ids=False).to(device)
            labels = tokenized.input_ids[:, 1:]
            logits = scoring_model(**tokenized).logits[:, :-1]
            regeneration_likelihood.append(get_likelihood(logits, labels))
        return ori_likelihood - np.mean(regeneration_likelihood)

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()

    # load data
    data = load_data(args)
    n_samples = len(data)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    criterion_fn = get_dna_gpt
    name = 'dna_gpt'
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    eval_results = []
    time1 = time.time()
    for idx in tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data[idx]["original"]
        sampled_text = data[idx]["sampled"]
        rewrote_original = data[idx]["rewrote_original"]
        rewrote_sampled = data[idx]["rewrote_sampled"]
        original_crit = criterion_fn(args, scoring_model, scoring_tokenizer, original_text, rewrote_original)
        sampled_crit = criterion_fn(args, scoring_model, scoring_tokenizer, sampled_text, rewrote_sampled)
        # result
        eval_results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
    time2 = time.time()
    time_per_sample_ms = (time2 - time1) / n_samples * 1000
    print(f"Time per sample for {name}: {time_per_sample_ms:.2f} ms")

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in eval_results],
                    'samples': [x["sampled_crit"] for x in eval_results]}
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # log results
    # results_file = os.getcwd() + f'{args.output_file}.{name}.json'
    # results_file = f'{args.output_file}.{name}.json'
    # results = { 'name': f'{name}_threshold',
    #             'info': {'n_samples': n_samples, 'n_regenerations': args.n_regenerations},
    #             'predictions': predictions,
    #             'raw_results': eval_results,
    #             'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
    #             'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
    #             'loss': 1 - pr_auc}
    
    # with open(results_file, 'w') as fout:
    #     json.dump(results, fout)
    #     print(f'Results written into {results_file}')
    results = {'dataset':args.dataset_file.split("/")[-1],
        'model':args.scoring_model_name,
        'feature index': name,
        'auc':round(roc_auc,4),
        'prauc':round(pr_auc,4),
        'n_regenerations': args.n_regenerations,
        'time_per_sample_ms':round(time_per_sample_ms,2)}

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
    parser.add_argument('--n_regenerations', type=int, default=10)
    parser.add_argument('--output_file', type=str, default="experiment_results/dna_gpt_detection_results/xsum_gpt4turbo")
    parser.add_argument('--dataset_file', type=str, default="datasets/regeneration_data_dnagpt/xsum_gpt4turbo")
    parser.add_argument('--scoring_model_name', type=str, default="gptj_6b")
    parser.add_argument('--scenario', type=str, default='black', choices=["black", "white"])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)

