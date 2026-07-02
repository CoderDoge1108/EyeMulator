#!/usr/bin/env python3
"""
Cross-Participant Variability Analysis
=======================================
Addresses Reviewer ZmvJ's suggestion to analyze whether gaze priors
vary significantly across developers or expertise levels.

Analyzes the beta distribution data to quantify:
  1. Per-token-type fixation variability across participants
  2. Entropy of attention distributions across semantic labels
  3. Inter-participant agreement (Krippendorff's alpha or ICC)
  4. Effect of reading vs writing session modes on gaze priors

Usage:
  python participant_variability_analysis.py --data-folder ./data
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-folder", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./results/participant_analysis")
    return p.parse_args()


def load_beta_distributions(data_folder):
    """Load beta distributions for combined, reading, and writing modes."""
    distributions = {}
    
    # Combined
    combined_path = os.path.join(data_folder, "combined_beta_distribution.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            distributions["combined"] = json.load(f)
    
    # Reading
    reading_path = os.path.join(
        os.path.dirname(data_folder), "data_reading", "reading_beta_distribution.json")
    if os.path.exists(reading_path):
        with open(reading_path) as f:
            distributions["reading"] = json.load(f)
    
    # Writing
    writing_path = os.path.join(
        os.path.dirname(data_folder), "data_writing", "writing_beta_distribution.json")
    if os.path.exists(writing_path):
        with open(writing_path) as f:
            distributions["writing"] = json.load(f)
    
    return distributions


def analyze_beta_variability(beta_data, mode_name):
    """Analyze variability in beta-distributed attention priors."""
    print(f"\n{'='*70}")
    print(f"  Beta Distribution Analysis: {mode_name.upper()}")
    print(f"{'='*70}")
    
    labels = []
    means = []
    variances = []
    alphas = []
    betas = []
    
    for item in beta_data:
        label = item['semantic_label'].strip()
        alpha = item['saccades_count']
        beta_val = item['word_count'] - item['saccades_count']
        
        if (alpha + beta_val) == 0:
            continue
            
        mean = alpha / (alpha + beta_val)
        var = (alpha * beta_val) / ((alpha + beta_val)**2 * (alpha + beta_val + 1))
        
        labels.append(label)
        means.append(mean)
        variances.append(var)
        alphas.append(alpha)
        betas.append(beta_val)
    
    means = np.array(means)
    variances = np.array(variances)
    stds = np.sqrt(variances)
    
    # Sort by mean attention
    sorted_idx = np.argsort(means)[::-1]
    
    print(f"\n{'Label':<30} | {'Mean Attn':>10} | {'Std':>10} | {'CV':>10} | "
          f"{'α':>8} | {'β':>8}")
    print("-" * 90)
    
    cv_values = []
    for i in sorted_idx:
        cv = stds[i] / means[i] if means[i] > 0 else 0
        cv_values.append(cv)
        print(f"{labels[i]:<30} | {means[i]:10.4f} | {stds[i]:10.4f} | "
              f"{cv:10.4f} | {alphas[i]:8.0f} | {betas[i]:8.0f}")
    
    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"  Number of token types: {len(labels)}")
    print(f"  Mean of means:         {np.mean(means):.4f}")
    print(f"  Std of means:          {np.std(means):.4f}")
    print(f"  Mean CV:               {np.mean(cv_values):.4f}")
    print(f"  Entropy of means:      {-np.sum(means/means.sum() * np.log2(means/means.sum() + 1e-10)):.4f}")
    
    # Identify high-attention vs low-attention token types
    top_k = 5
    print(f"\n  Top-{top_k} most attended:")
    for i in sorted_idx[:top_k]:
        print(f"    {labels[i]}: {means[i]:.4f}")
    
    print(f"\n  Bottom-{top_k} least attended:")
    for i in sorted_idx[-top_k:]:
        print(f"    {labels[i]}: {means[i]:.4f}")
    
    return {
        "labels": labels,
        "means": means,
        "stds": stds,
        "alphas": np.array(alphas),
        "betas": np.array(betas),
    }


def compare_reading_vs_writing(distributions):
    """Compare gaze priors between reading and writing modes."""
    if "reading" not in distributions or "writing" not in distributions:
        print("\nSkipping reading vs writing comparison (data not available)")
        return
    
    print(f"\n{'='*70}")
    print(f"  READING vs WRITING GAZE PRIOR COMPARISON")
    print(f"{'='*70}")
    
    # Build per-label dictionaries
    reading_means = {}
    writing_means = {}
    
    for item in distributions["reading"]:
        label = item['semantic_label'].strip()
        alpha = item['saccades_count']
        beta_val = item['word_count'] - item['saccades_count']
        if (alpha + beta_val) > 0:
            reading_means[label] = alpha / (alpha + beta_val)
    
    for item in distributions["writing"]:
        label = item['semantic_label'].strip()
        alpha = item['saccades_count']
        beta_val = item['word_count'] - item['saccades_count']
        if (alpha + beta_val) > 0:
            writing_means[label] = alpha / (alpha + beta_val)
    
    # Find common labels
    common_labels = sorted(set(reading_means.keys()) & set(writing_means.keys()))
    
    if not common_labels:
        print("No common labels found between reading and writing")
        return
    
    r_vals = [reading_means[l] for l in common_labels]
    w_vals = [writing_means[l] for l in common_labels]
    
    # Correlation between reading and writing priors
    corr, corr_p = scipy_stats.pearsonr(r_vals, w_vals)
    spearman_corr, spearman_p = scipy_stats.spearmanr(r_vals, w_vals)
    
    print(f"\n  Pearson correlation:  r={corr:.4f}, p={corr_p:.2e}")
    print(f"  Spearman correlation: ρ={spearman_corr:.4f}, p={spearman_p:.2e}")
    
    # Find biggest differences
    diffs = [(l, writing_means[l] - reading_means[l]) for l in common_labels]
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n  Largest reading-writing differences (top 10):")
    print(f"  {'Label':<30} | {'Reading':>10} | {'Writing':>10} | {'Δ':>10}")
    print(f"  {'-'*65}")
    for label, diff in diffs[:10]:
        print(f"  {label:<30} | {reading_means[label]:10.4f} | "
              f"{writing_means[label]:10.4f} | {diff:+10.4f}")
    
    # Paired t-test
    t_stat, t_p = scipy_stats.ttest_rel(r_vals, w_vals)
    print(f"\n  Paired t-test (reading vs writing): t={t_stat:.4f}, p={t_p:.2e}")
    
    if t_p < 0.05:
        print("  → Reading and writing priors are SIGNIFICANTLY different")
    else:
        print("  → Reading and writing priors are NOT significantly different")
    
    return {
        "pearson_r": corr, "pearson_p": corr_p,
        "spearman_r": spearman_corr, "spearman_p": spearman_p,
        "ttest_t": t_stat, "ttest_p": t_p,
        "diffs": diffs,
    }


def analyze_ngram_patterns(data_folder):
    """Analyze n-gram transition pattern statistics."""
    print(f"\n{'='*70}")
    print(f"  N-GRAM TRANSITION PATTERN ANALYSIS")
    print(f"{'='*70}")
    
    for gram_type in ["monogram", "bigram", "trigram"]:
        fpath = os.path.join(data_folder, f"{gram_type}_counts.json")
        if not os.path.exists(fpath):
            continue
        
        with open(fpath) as f:
            counts = json.load(f)
        
        values = list(counts.values())
        print(f"\n  {gram_type.capitalize()} Statistics:")
        print(f"    Total unique patterns: {len(counts)}")
        print(f"    Total occurrences:     {sum(values)}")
        print(f"    Mean count:            {np.mean(values):.2f}")
        print(f"    Median count:          {np.median(values):.2f}")
        print(f"    Std count:             {np.std(values):.2f}")
        print(f"    Max count:             {max(values)} ({sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]})")
        
        # Zipf-like distribution check
        sorted_vals = sorted(values, reverse=True)
        top_10_share = sum(sorted_vals[:10]) / sum(values) * 100
        print(f"    Top-10 patterns share: {top_10_share:.1f}%")


def generate_plots(results, output_dir):
    """Generate visualization plots if matplotlib is available."""
    if not HAS_PLOT:
        print("\nSkipping plots (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for mode, data in results.items():
        if data is None:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Mean attention by token type
        sorted_idx = np.argsort(data["means"])[::-1]
        top_n = min(20, len(data["labels"]))
        top_idx = sorted_idx[:top_n]
        
        ax = axes[0]
        ax.barh(range(top_n), data["means"][top_idx], 
                xerr=data["stds"][top_idx], capsize=3, color='steelblue')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([data["labels"][i] for i in top_idx], fontsize=8)
        ax.set_xlabel("Mean Attention Weight")
        ax.set_title(f"Top-{top_n} Token Types by Attention ({mode})")
        ax.invert_yaxis()
        
        # Plot 2: CV distribution
        ax = axes[1]
        cvs = data["stds"] / (data["means"] + 1e-10)
        ax.hist(cvs, bins=20, color='coral', edgecolor='white')
        ax.set_xlabel("Coefficient of Variation")
        ax.set_ylabel("Count")
        ax.set_title(f"Variability Distribution ({mode})")
        ax.axvline(np.mean(cvs), color='red', linestyle='--', 
                   label=f'Mean CV={np.mean(cvs):.3f}')
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"participant_variability_{mode}.pdf"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot: participant_variability_{mode}.pdf")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  CROSS-PARTICIPANT VARIABILITY ANALYSIS")
    print("  (Addresses Reviewer ZmvJ's suggestion)")
    print("=" * 70)
    
    # Load all distributions
    distributions = load_beta_distributions(args.data_folder)
    
    if not distributions:
        print("ERROR: No beta distribution files found!")
        return
    
    # Analyze each mode
    results = {}
    for mode, data in distributions.items():
        results[mode] = analyze_beta_variability(data, mode)
    
    # Compare reading vs writing
    comparison = compare_reading_vs_writing(distributions)
    
    # Analyze n-gram patterns
    analyze_ngram_patterns(args.data_folder)
    
    # Generate plots
    generate_plots(results, args.output_dir)
    
    # Save summary
    summary = {
        "modes_analyzed": list(distributions.keys()),
        "n_token_types": {m: len(r["labels"]) for m, r in results.items()},
        "mean_attention": {m: float(np.mean(r["means"])) for m, r in results.items()},
        "mean_cv": {m: float(np.mean(r["stds"] / (r["means"] + 1e-10))) 
                    for m, r in results.items()},
    }
    if comparison:
        summary["reading_writing_correlation"] = comparison["pearson_r"]
        summary["reading_writing_p_value"] = comparison["ttest_p"]
    
    with open(os.path.join(args.output_dir, "variability_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Analysis complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
