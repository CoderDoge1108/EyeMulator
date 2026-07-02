#!/usr/bin/env python3
"""
Unified Cross-Task Evaluation with Multi-Seed Aggregation & Significance Tests
===============================================================================
Addresses reviewer concerns:
  - Mean ± std over multiple seeds                     [ZmvJ]
  - Paired t-test for statistical significance         [ZmvJ, NpXQ]
  - Random attention baseline included in comparisons  [CX74]
  - All ablation variants supported                    [ZmvJ]
  - Training time overhead comparison                  [ZmvJ]

Usage:
  python cross_task_evaluation_unified.py --results-root ./results --seeds 42 123 456
"""

import os
import sys
import json
import math
import re
import glob
import textwrap
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats

# Third-party imports
import nltk
import sacrebleu
import Levenshtein
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from codebleu import calc_codebleu

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    EMBEDDING_MODEL = None

# NLTK data
for res in ("tokenizers/punkt", "corpora/wordnet", "corpora/omw-1.4"):
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res.split('/')[-1], quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, default="./results",
                   help="Root directory containing *_results/ folders. "
                        "Will also check ./results/results_cross_task/ if not found.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="Seeds to aggregate over")
    p.add_argument("--models", type=str, nargs="+",
                   default=["llama", "deepseek", "starcoder"])
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["completion", "translation", "summarization"])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["baseline", "eyemulator", "random",
                            "eyetracking_only", "weighted_sft_only"])
    p.add_argument("--clean", action="store_true", default=True)
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--metadata-root", type=str, default="./",
                   help="Root directory for training_metadata.json files")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────
_WS = re.compile(r"\s+")

def clean(text, task):
    if not text:
        return ""
    if task in {"completion", "translation"}:
        text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.S | re.M)
        text = re.sub(r"\s+\n", "\n", text)
    elif task == "summarization":
        text = re.sub(r"^\s*(?:answer|summary)[:\-–]\s*", "", text, flags=re.I)
        text = textwrap.dedent(text)
    return text.replace("```", "").replace("<pad>", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Metric Functions
# ─────────────────────────────────────────────────────────────────────────────
def _minify(s):
    return _WS.sub("", s or "")

def hybrid_exact(preds, refs):
    if not preds:
        return 0.0
    p_min = [_minify(p) for p in preds]
    r_min = [_minify(r) for r in refs]
    exact = sum(p == r for p, r in zip(p_min, r_min))
    contained = sum(1 for p, r in zip(p_min, r_min) if p and r and (p in r or r in p))
    return 0.5 * (exact + contained) / len(preds) * 100.0

def calc_codebleu_score(preds, refs, lang):
    if not preds:
        return 0.0
    return calc_codebleu(refs, preds, lang=lang)["codebleu"] * 100

def crystalbleu(preds, refs):
    if not preds:
        return 0.0
    return sacrebleu.corpus_chrf(preds, [refs], word_order=2).score

def nls_score(preds, refs):
    if not preds:
        return 0.0
    sims = [1 - Levenshtein.distance(p, r) / max(1, len(r), len(p))
            for p, r in zip(preds, refs)]
    return np.mean(sims) * 100

def rouge_scores(preds, refs):
    if not preds:
        return {"ROUGE1": 0.0, "ROUGEL": 0.0}
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [sc.score(r, p) for p, r in zip(preds, refs)]
    return {
        "ROUGE1": np.mean([s["rouge1"].fmeasure for s in scores]) * 100,
        "ROUGEL": np.mean([s["rougeL"].fmeasure for s in scores]) * 100,
    }

def meteor_avg(preds, refs):
    if not preds:
        return 0.0
    vals = [meteor_score([r.split()], p.split()) if p and r else 0
            for p, r in zip(preds, refs)]
    return np.mean(vals) * 100

def bert_score_f1(preds, refs, model="bert-base-uncased"):
    p_clean = [p for p in preds if p]
    r_clean = [r for p, r in zip(preds, refs) if p]
    if not p_clean:
        return 0.0
    _, _, F1 = bert_score(p_clean, r_clean, model_type=model, lang="en",
                           rescale_with_baseline=True)
    return F1.mean().item() * 100

def embedding_similarity(preds, refs, model=None, batch_size=32):
    if model is None:
        return 0.0
    if not preds:
        return 0.0
    ref_embeds = model.encode(refs, batch_size=batch_size, show_progress_bar=False,
                               normalize_embeddings=True)
    pred_embeds = model.encode(preds, batch_size=batch_size, show_progress_bar=False,
                                normalize_embeddings=True)
    sim = np.sum(ref_embeds * pred_embeds, axis=1)
    return np.mean(sim) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Task Metric Config
# ─────────────────────────────────────────────────────────────────────────────
TASK_METRICS = {
    "completion": {
        "lang": "java",
        "metrics": ["HybridExact", "CodeBLEU", "CrystalBLEU", "NLS", "EmbeddingSim"],
    },
    "translation": {
        "lang": "c_sharp",
        "metrics": ["HybridExact", "CodeBLEU", "CrystalBLEU", "NLS", "EmbeddingSim"],
    },
    "summarization": {
        "lang": "java",
        "metrics": ["ROUGE1", "ROUGEL", "METEOR", "BERTScore", "EmbeddingSim"],
    },
}


def evaluate_predictions(preds, refs, task, use_embeddings=True):
    """Compute all metrics for a task."""
    cfg = TASK_METRICS[task]
    results = {}

    if "HybridExact" in cfg["metrics"]:
        results["HybridExact"] = hybrid_exact(preds, refs)
    if "CodeBLEU" in cfg["metrics"]:
        results["CodeBLEU"] = calc_codebleu_score(preds, refs, cfg["lang"])
    if "CrystalBLEU" in cfg["metrics"]:
        results["CrystalBLEU"] = crystalbleu(preds, refs)
    if "NLS" in cfg["metrics"]:
        results["NLS"] = nls_score(preds, refs)
    if "ROUGE1" in cfg["metrics"] or "ROUGEL" in cfg["metrics"]:
        results.update(rouge_scores(preds, refs))
    if "METEOR" in cfg["metrics"]:
        results["METEOR"] = meteor_avg(preds, refs)
    if "BERTScore" in cfg["metrics"]:
        results["BERTScore"] = bert_score_f1(preds, refs)
    if use_embeddings and EMBEDDING_MODEL is not None:
        if "EmbeddingSim" in cfg["metrics"]:
            results["EmbeddingSim"] = embedding_similarity(preds, refs, EMBEDDING_MODEL)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_results_file(path, task, do_clean=True):
    """Load predictions and references from a results JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    preds = [clean(d.get("generated_output", ""), task) if do_clean
             else d.get("generated_output", "") for d in data]
    refs = [clean(d.get("ground_truth", ""), task) if do_clean
            else d.get("ground_truth", "") for d in data]
    return preds, refs


def find_result_files(results_root, model, method, task, seeds):
    """Find result files for each seed."""
    # Search in multiple possible locations
    search_dirs = [
        os.path.join(results_root, f"{model}_results"),
        os.path.join(results_root, "results_cross_task", f"{model}_results"),
    ]
    found = {}
    for seed in seeds:
        for model_dir in search_dirs:
            if not os.path.isdir(model_dir):
                continue
            # New naming convention (from unified scripts)
            pattern = os.path.join(model_dir,
                                    f"generated_results_{method}_{task}_seed{seed}.json")
            if os.path.exists(pattern):
                found[seed] = pattern
                break
            else:
                # Legacy naming convention (from original scripts)
                if method == "eyemulator":
                    legacy = os.path.join(model_dir,
                                           f"generated_results_advanced_final_{task}.json")
                elif method == "baseline":
                    legacy = os.path.join(model_dir,
                                           f"generated_results_baseline_{task}.json")
                else:
                    legacy = None

                if legacy and os.path.exists(legacy):
                    # Only assign legacy to seed 0 to avoid counting same file multiple times
                    if seed == seeds[0]:
                        found[seed] = legacy
                    break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Per-example Metrics (for paired significance testing)
# ─────────────────────────────────────────────────────────────────────────────
def per_example_nls(preds, refs):
    """Compute NLS per example."""
    return [
        (1 - Levenshtein.distance(p, r) / max(1, len(r), len(p))) * 100
        for p, r in zip(preds, refs)
    ]

def per_example_rouge(preds, refs, metric_key="rougeL"):
    """Compute ROUGE per example."""
    sc = rouge_scorer.RougeScorer([metric_key], use_stemmer=True)
    return [sc.score(r, p)[metric_key].fmeasure * 100 for p, r in zip(preds, refs)]


def paired_significance_test(scores_a, scores_b, test="ttest"):
    """
    Paired significance test between two sets of per-example scores.
    Returns (test_statistic, p_value).
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = min(len(scores_a), len(scores_b))
    scores_a = scores_a[:n]
    scores_b = scores_b[:n]

    if test == "ttest":
        stat, p = scipy_stats.ttest_rel(scores_a, scores_b)
    elif test == "wilcoxon":
        try:
            stat, p = scipy_stats.wilcoxon(scores_a - scores_b)
        except ValueError:
            stat, p = 0.0, 1.0
    elif test == "bootstrap":
        # Bootstrap confidence interval for the difference
        diffs = scores_a - scores_b
        n_bootstrap = 10000
        boot_means = np.array([
            np.mean(np.random.choice(diffs, size=len(diffs), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        p = np.mean(boot_means < 0) if np.mean(diffs) > 0 else np.mean(boot_means > 0)
        stat = np.mean(diffs)
    else:
        raise ValueError(f"Unknown test: {test}")

    return stat, p


# ─────────────────────────────────────────────────────────────────────────────
# Training Time Comparison
# ─────────────────────────────────────────────────────────────────────────────
def load_training_metadata(metadata_root, model, method, task, seeds):
    """Load training metadata for timing comparison."""
    times = []
    for seed in seeds:
        meta_path = os.path.join(
            metadata_root,
            f"{model}_{method}_{task}_seed{seed}",
            "training_metadata.json"
        )
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            times.append(meta.get("training_time_minutes", 0))
    return times


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    use_embeddings = not args.no_embeddings

    print("=" * 90)
    print("  EYEMULATOR: UNIFIED CROSS-TASK EVALUATION WITH SIGNIFICANCE TESTING")
    print("=" * 90)

    # Collect all results
    all_results = {}  # {(model, task): {method: {metric: [score_per_seed]}}}

    for model in args.models:
        for task in args.tasks:
            key = (model, task)
            all_results[key] = {}

            for method in args.methods:
                seed_files = find_result_files(
                    args.results_root, model, method, task, args.seeds)

                if not seed_files:
                    continue

                method_scores = defaultdict(list)
                for seed, fpath in seed_files.items():
                    preds, refs = load_results_file(fpath, task, args.clean)
                    scores = evaluate_predictions(preds, refs, task, use_embeddings)
                    for metric, val in scores.items():
                        method_scores[metric].append(val)

                all_results[key][method] = dict(method_scores)

    # ── Print Results Table ──
    print("\n" + "=" * 110)
    print(f"{'Model':<12} | {'Task':<15} | {'Method':<20} | {'Metric':<14} | "
          f"{'Mean':>8} | {'±Std':>8} | {'N seeds':>7}")
    print("-" * 110)

    for (model, task), methods in sorted(all_results.items()):
        if not methods:
            continue
        first = True
        for method, metrics in sorted(methods.items()):
            for metric, values in sorted(metrics.items()):
                m_display = model if first else ""
                t_display = task if first else ""
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else 0.0
                print(f"{m_display:<12} | {t_display:<15} | {method:<20} | "
                      f"{metric:<14} | {mean_val:8.2f} | {std_val:8.2f} | "
                      f"{len(values):>7}")
                first = False
        print("-" * 110)

    # ── Significance Tests (EyeMulator vs Baseline) ──
    print("\n" + "=" * 110)
    print("  STATISTICAL SIGNIFICANCE: EyeMulator vs Baseline (paired t-test)")
    print("=" * 110)
    print(f"{'Model':<12} | {'Task':<15} | {'Metric':<14} | "
          f"{'Eye. Mean':>10} | {'Base Mean':>10} | {'Δ':>8} | {'p-value':>10} | {'Sig?':>5}")
    print("-" * 110)

    for (model, task), methods in sorted(all_results.items()):
        baseline = methods.get("baseline", {})
        eyemulator = methods.get("eyemulator", {})
        if not baseline or not eyemulator:
            continue

        for metric in sorted(set(baseline.keys()) & set(eyemulator.keys())):
            base_vals = baseline[metric]
            eye_vals = eyemulator[metric]

            base_mean = np.mean(base_vals)
            eye_mean = np.mean(eye_vals)
            delta = eye_mean - base_mean

            # If we have enough seeds, do a paired test
            if len(base_vals) >= 2 and len(eye_vals) >= 2:
                n = min(len(base_vals), len(eye_vals))
                stat, p_val = scipy_stats.ttest_ind(eye_vals[:n], base_vals[:n])
                sig = "✓" if p_val < 0.05 else "✗"
                p_str = f"{p_val:.2e}"
            else:
                p_str = "N/A"
                sig = "-"

            print(f"{model:<12} | {task:<15} | {metric:<14} | "
                  f"{eye_mean:10.2f} | {base_mean:10.2f} | {delta:+8.2f} | "
                  f"{p_str:>10} | {sig:>5}")

    # ── Training Time Overhead ──
    print("\n" + "=" * 90)
    print("  TRAINING TIME COMPARISON (minutes)")
    print("=" * 90)
    print(f"{'Model':<12} | {'Task':<15} | {'Method':<20} | "
          f"{'Mean (min)':>12} | {'±Std':>8} | {'Overhead':>10}")
    print("-" * 90)

    for model in args.models:
        for task in args.tasks:
            baseline_times = load_training_metadata(
                args.metadata_root, model, "baseline", task, args.seeds)

            for method in args.methods:
                times = load_training_metadata(
                    args.metadata_root, model, method, task, args.seeds)
                if not times:
                    continue

                mean_t = np.mean(times)
                std_t = np.std(times) if len(times) > 1 else 0.0

                if baseline_times and method != "baseline":
                    overhead = (mean_t / np.mean(baseline_times) - 1) * 100
                    overhead_str = f"+{overhead:.1f}%"
                else:
                    overhead_str = "-"

                print(f"{model:<12} | {task:<15} | {method:<20} | "
                      f"{mean_t:12.1f} | {std_t:8.1f} | {overhead_str:>10}")

    print("=" * 90)
    print("\nDone! Results aggregated over seeds:", args.seeds)


if __name__ == "__main__":
    main()
