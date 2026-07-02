"""
Unified metric computation for EyeMulator experiments.

Reads generation JSON files and computes all metrics from the paper:
  - Completion / Translation: HybridExact, CodeBLEU, CrystalBLEU
  - Summarization: ROUGE-1, ROUGE-L, METEOR, BERTScore

Usage:
    python compute_metrics.py --results_dir ./results --task completion
    python compute_metrics.py --results_dir ./results --task summarization
    python compute_metrics.py --compare_dirs ./results/llama_baseline_completion_seed42 ./results/llama_eyemulator_completion_seed42 --task completion
"""

import os
import sys
import json
import re
import math
import logging
import argparse
import textwrap
from collections import defaultdict

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Lazy imports — only load heavy libraries when needed
# ═══════════════════════════════════════════════════════════════════════════════
_WS = re.compile(r"\s+")


def _ensure_nltk():
    """Ensure NLTK data is available."""
    import nltk
    for res in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res.split("/")[-1], quiet=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Text Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
def clean(text: str, task: str) -> str:
    """Clean generated text for fair comparison."""
    if not text:
        return ""
    if task in {"completion", "translation"}:
        # Remove comments and normalize whitespace for code tasks
        text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.S | re.M)
        text = re.sub(r"\s+\n", "\n", text)
    elif task == "summarization":
        text = re.sub(r"^\s*(?:answer|summary)[:\-–]\s*", "", text, flags=re.I)
        text = textwrap.dedent(text)
    return text.replace("```", "").replace("<pad>", "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Functions
# ═══════════════════════════════════════════════════════════════════════════════
def _minify(s: str) -> str:
    """Remove all whitespace."""
    return _WS.sub("", s or "")


def hybrid_exact(preds, refs):
    """H-Exact = 0.5 * ExactMatch + 0.5 * SubstringMatch (on minified strings)."""
    if not preds:
        return 0.0
    p_min = [_minify(p) for p in preds]
    r_min = [_minify(r) for r in refs]
    exact = sum(p == r for p, r in zip(p_min, r_min))
    contained = sum(
        1
        for p, r in zip(p_min, r_min)
        if p and r and (p in r or r in p)
    )
    return 0.5 * (exact + contained) / len(preds) * 100.0


def codebleu_score(preds, refs, lang):
    """CodeBLEU score (requires `codebleu` package)."""
    if not preds:
        return 0.0
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(refs, preds, lang=lang)
        return result["codebleu"] * 100
    except ImportError:
        logging.warning("codebleu not installed, skipping CodeBLEU")
        return float("nan")
    except Exception as e:
        logging.warning(f"CodeBLEU error: {e}")
        return float("nan")


def crystalbleu_score(preds, refs):
    """CrystalBLEU (chrF with word order) via sacrebleu."""
    if not preds:
        return 0.0
    try:
        import sacrebleu
        return sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    except ImportError:
        logging.warning("sacrebleu not installed, skipping CrystalBLEU")
        return float("nan")


def rouge_scores(preds, refs):
    """ROUGE-1 and ROUGE-L F1 scores."""
    if not preds:
        return {"ROUGE-1": 0.0, "ROUGE-L": 0.0}
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = [sc.score(r, p) for p, r in zip(preds, refs)]
        return {
            "ROUGE-1": np.mean([s["rouge1"].fmeasure for s in scores]) * 100,
            "ROUGE-L": np.mean([s["rougeL"].fmeasure for s in scores]) * 100,
        }
    except ImportError:
        logging.warning("rouge_score not installed, skipping ROUGE")
        return {"ROUGE-1": float("nan"), "ROUGE-L": float("nan")}


def meteor_avg(preds, refs):
    """Average METEOR score."""
    if not preds:
        return 0.0
    try:
        _ensure_nltk()
        from nltk.translate.meteor_score import meteor_score
        vals = [
            meteor_score([r.split()], p.split()) if p and r else 0.0
            for p, r in zip(preds, refs)
        ]
        return np.mean(vals) * 100
    except ImportError:
        logging.warning("nltk not installed, skipping METEOR")
        return float("nan")


def bertscore_avg(preds, refs):
    """BERTScore F1."""
    if not preds:
        return 0.0
    try:
        from bert_score import score as bert_score_fn
        p_clean = [p for p in preds if p]
        r_clean = [r for p, r in zip(preds, refs) if p]
        if not p_clean:
            return 0.0
        _, _, F1 = bert_score_fn(
            p_clean, r_clean,
            model_type="bert-base-uncased",
            lang="en",
            rescale_with_baseline=True,
        )
        return F1.mean().item() * 100
    except ImportError:
        logging.warning("bert_score not installed, skipping BERTScore")
        return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
TASK_METRICS = {
    "completion": {
        "lang": "java",
        "metrics": ["HybridExact", "CodeBLEU", "CrystalBLEU"],
    },
    "translation": {
        "lang": "c_sharp",
        "metrics": ["HybridExact", "CodeBLEU", "CrystalBLEU"],
    },
    "summarization": {
        "lang": "java",
        "metrics": ["ROUGE-1", "ROUGE-L", "METEOR", "BERTScore"],
    },
}


def load_results(path: str, task: str, do_clean: bool = True):
    """Load generation results and optionally clean them."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    preds = [
        clean(d.get("generated_output", ""), task) if do_clean
        else d.get("generated_output", "")
        for d in data
    ]
    refs = [
        clean(d.get("ground_truth", ""), task) if do_clean
        else d.get("ground_truth", "")
        for d in data
    ]
    return preds, refs


def evaluate(preds, refs, task: str) -> dict:
    """Compute all metrics for a given task."""
    cfg = TASK_METRICS[task]
    metrics = cfg["metrics"]
    lang = cfg["lang"]
    results = {}

    if "HybridExact" in metrics:
        results["HybridExact"] = hybrid_exact(preds, refs)
    if "CodeBLEU" in metrics:
        results["CodeBLEU"] = codebleu_score(preds, refs, lang)
    if "CrystalBLEU" in metrics:
        results["CrystalBLEU"] = crystalbleu_score(preds, refs)
    if "ROUGE-1" in metrics or "ROUGE-L" in metrics:
        results.update(rouge_scores(preds, refs))
    if "METEOR" in metrics:
        results["METEOR"] = meteor_avg(preds, refs)
    if "BERTScore" in metrics:
        results["BERTScore"] = bertscore_avg(preds, refs)

    return results


def print_comparison_table(all_results: dict, task: str):
    """Print a comparison table for multiple models/methods."""
    print(f"\n{'═' * 70}")
    print(f"  {task.upper()} — Metric Comparison")
    print(f"{'═' * 70}")

    # Collect all metric names
    all_metrics = []
    for name, metrics_dict in all_results.items():
        for m in metrics_dict:
            if m not in all_metrics:
                all_metrics.append(m)

    # Header
    names = list(all_results.keys())
    header = f"{'Metric':<15}"
    for name in names:
        header += f" | {name:>20}"
    print(header)
    print("-" * len(header))

    # Rows
    for m in all_metrics:
        row = f"{m:<15}"
        values = [all_results[name].get(m, float("nan")) for name in names]
        best_val = max(v for v in values if not math.isnan(v)) if any(not math.isnan(v) for v in values) else None

        for v in values:
            if math.isnan(v):
                cell = "N/A"
            elif best_val is not None and abs(v - best_val) < 0.005:
                cell = f"🏆 {v:.2f}"
            else:
                cell = f"   {v:.2f}"
            row += f" | {cell:>20}"
        print(row)

    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for EyeMulator experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True,
        choices=["completion", "translation", "summarization"],
    )
    parser.add_argument(
        "--compare_files", nargs="+",
        help="Paths to generation JSON files to compare (e.g. baseline.json eyemulator.json)",
    )
    parser.add_argument(
        "--compare_names", nargs="+",
        help="Names for each file (e.g. Baseline EyeMulator). Must match --compare_files.",
    )
    parser.add_argument(
        "--results_dir", default=None,
        help="Auto-scan this directory for generation JSON files",
    )
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--output_file", default=None, help="Save metrics JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    all_results = {}

    if args.compare_files:
        # Explicit file comparison
        names = args.compare_names or [
            os.path.splitext(os.path.basename(f))[0] for f in args.compare_files
        ]
        for name, fpath in zip(names, args.compare_files):
            if not os.path.exists(fpath):
                logging.warning(f"File not found: {fpath}, skipping")
                continue
            preds, refs = load_results(fpath, args.task, not args.no_clean)
            logging.info(f"Evaluating '{name}': {len(preds)} examples")
            all_results[name] = evaluate(preds, refs, args.task)

    elif args.results_dir:
        # Auto-scan directory
        for entry in sorted(os.listdir(args.results_dir)):
            entry_path = os.path.join(args.results_dir, entry)
            # Look for generation files in subdirectories
            if os.path.isdir(entry_path):
                for fname in os.listdir(entry_path):
                    if fname.endswith(".json") and args.task in fname:
                        fpath = os.path.join(entry_path, fname)
                        name = entry
                        preds, refs = load_results(fpath, args.task, not args.no_clean)
                        logging.info(f"Evaluating '{name}': {len(preds)} examples")
                        all_results[name] = evaluate(preds, refs, args.task)
            # Also check for JSON files directly in results_dir
            elif entry.endswith(".json") and args.task in entry:
                name = os.path.splitext(entry)[0]
                preds, refs = load_results(entry_path, args.task, not args.no_clean)
                logging.info(f"Evaluating '{name}': {len(preds)} examples")
                all_results[name] = evaluate(preds, refs, args.task)
    else:
        parser.error("Provide either --compare_files or --results_dir")

    if not all_results:
        logging.error("No results found to evaluate!")
        sys.exit(1)

    # Print comparison table
    print_comparison_table(all_results, args.task)

    # Save metrics
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Metrics saved to {args.output_file}")

    logging.info("Done! ✅")


if __name__ == "__main__":
    main()
