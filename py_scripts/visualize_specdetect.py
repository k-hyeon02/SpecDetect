"""
Standalone visualization script for SpecDetect++ results.
Reads per-sample prediction JSONs and regenerates plots without model inference.

Usage:
    python py_scripts/visualize_specdetect.py                          # all JSONs
    python py_scripts/visualize_specdetect.py --filter xsum_gpt2_xl    # specific dataset
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def visualize_fft_overlay(real_scores, sample_scores, save_dir, dataset_name, model_name):
    """Histogram + box plot of sampling discrepancy scores."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Sampling Discrepancy Distribution: {dataset_name} / {model_name}', fontsize=14)

    axes[0].hist(real_scores, bins=30, alpha=0.6, color='blue', label='Human', density=True)
    axes[0].hist(sample_scores, bins=30, alpha=0.6, color='red', label='LLM', density=True)
    axes[0].set_title('Sampling Discrepancy Distribution')
    axes[0].set_xlabel('Sampling Discrepancy (z-score)')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([real_scores, sample_scores], tick_labels=['Human', 'LLM'],
                    patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[1].set_title('Sampling Discrepancy Box Plot')
    axes[1].set_ylabel('Sampling Discrepancy (z-score)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(save_dir, f'fft_energy_dist_{dataset_name}_{model_name}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {png_path}')
    return png_path


def main():
    parser = argparse.ArgumentParser(description='Regenerate SpecDetect++ visualization from saved predictions')
    parser.add_argument('--input_dir', type=str,
                        default='experiment_results/specdetect_detection_results/visualizations',
                        help='Directory containing prediction JSON files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as input_dir)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only process JSONs matching this substring (e.g., xsum_gpt2_xl)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir

    json_files = sorted(glob(os.path.join(args.input_dir, 'predictions_*.json')))
    if args.filter:
        json_files = [f for f in json_files if args.filter in f]

    if not json_files:
        print(f'No prediction JSON files found in {args.input_dir}')
        print('Run specdetect_doubleplus.py first to generate prediction data.')
        return

    print(f'Found {len(json_files)} prediction files')
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        print(f'  {data["dataset"]} / {data["model"]} (AUC={data["roc_auc"]:.4f})')
        visualize_fft_overlay(data['real'], data['samples'],
                              output_dir, data['dataset'], data['model'])

    print(f'\nDone. {len(json_files)} plots regenerated in {output_dir}')


if __name__ == '__main__':
    main()
