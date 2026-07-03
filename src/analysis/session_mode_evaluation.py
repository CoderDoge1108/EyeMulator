# Group-based evaluation for session-mode analysis.
# The script expects input result files to contain a `group_label` field.

import json, math, re, textwrap, os, sys
from collections import defaultdict
import numpy as np

# Third-party imports
import nltk, sacrebleu, Levenshtein
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from codebleu import calc_codebleu

# Attempt to import tqdm, but don't fail if not installed unless needed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        # If tqdm is not installed, just return the original iterable
        return iterable

# Import SentenceTransformer for local embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ────── ensure NLTK data ────────────────────────────────────────────────────
def _ensure_nltk_data(path: str):
    """Checks for NLTK data and downloads if missing."""
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"Downloading required NLTK resource: {path.split('/')[-1]}...")
        nltk.download(path.split('/')[-1], quiet=True)

for res in ("tokenizers/punkt", "corpora/wordnet", "corpora/omw-1.4"):
    _ensure_nltk_data(res)

# ────── cleaning helper ─────────────────────────────────────────────────────
_WS = re.compile(r"\s+")
def clean(text: str, task: str) -> str:
    """Cleans text based on the task type."""
    if not text: return ""
    task_type = "summarization" if "summarization" in task.lower() else "completion"
    if task_type == "completion":
        text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.S | re.M)
        text = re.sub(r"\s+\n", "\n", text)
    elif task_type == "summarization":
        text = re.sub(r"^\s*(?:answer|summary)[:\-–]\s*", "", text, flags=re.I)
        text = textwrap.dedent(text)
    return text.replace("```", "").replace("<pad>", "").strip()

# ────── traditional metric helpers (UNCHANGED) ──────────────────────────────
def _minify(s: str) -> str:
    """Removes all whitespace from a string."""
    return _WS.sub("", s or "")

def hybrid_exact(preds, refs):
    """Averages exact match and containment match on minified strings."""
    if not preds: return 0.0
    p_min = [_minify(p) for p in preds]
    r_min = [_minify(r[0]) for r in refs]
    exact = sum(p == r for p, r in zip(p_min, r_min))
    contained = sum(1 for p, r in zip(p_min, r_min) if p and r and (p in r or r in p))
    return 0.5 * (exact + contained) / len(preds) * 100.0

def codebleu(preds, refs, lang):
    if not preds: return 0.0
    flat_refs = [r[0] for r in refs]
    return calc_codebleu(flat_refs, preds, lang=lang)["codebleu"] * 100

def crystalbleu(preds, refs):
    if not preds: return 0.0
    single_ref_list = [r[0] for r in refs]
    return sacrebleu.corpus_chrf(preds, [single_ref_list], word_order=2).score

def nls(preds, refs):
    if not preds: return 0.0
    sims = [1 - Levenshtein.distance(p, r[0]) / max(1, len(r[0]), len(p)) for p, r in zip(preds, refs)]
    return np.mean(sims) * 100

def rouge1_l(preds, refs):
    if not preds: return {"ROUGE1": 0.0, "ROUGEL": 0.0}
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [sc.score(r[0], p) for p, r in zip(preds, refs)]
    return {
        "ROUGE1": np.mean([s["rouge1"].fmeasure for s in scores]) * 100,
        "ROUGEL": np.mean([s["rougeL"].fmeasure for s in scores]) * 100
    }

def meteor_avg(preds, refs):
    if not preds: return 0.0
    tokenized_refs = [[ref.split() for ref in r_list] for r_list in refs]
    tokenized_preds = [p.split() for p in preds]
    vals = [meteor_score(ref, pred) if pred and ref[0] else 0 for pred, ref in zip(tokenized_preds, tokenized_refs)]
    return np.mean(vals) * 100

def bscore(preds, refs, model="bert-base-uncased"):
    if not any(preds): return 0.0
    p_clean = [p for p in preds if p]
    r_clean = [r[0] for p, r in zip(preds, refs) if p]
    if not p_clean: return 0.0
    _, _, F1 = bert_score(p_clean, r_clean, model_type=model, lang="en", rescale_with_baseline=True)
    return F1.mean().item() * 100

# ────── embedding metric helper (UNCHANGED) ─────────────────────────────────
EMBEDDING_MODEL = None
if SentenceTransformer:
    try:
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Could not load SentenceTransformer model. Ensure torch is installed. Error: {e}", file=sys.stderr)

def embedding_similarity(preds, refs, model, batch_size=32):
    if model is None: return 0.0
    if not preds: return 0.0
    flat_refs = [r[0] for r in refs]
    ref_embeds = model.encode(flat_refs, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    pred_embeds = model.encode(preds, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    similarity = np.sum(ref_embeds * pred_embeds, axis=1)
    return np.mean(similarity) * 100

# ────── **REVISED** task configuration ──────────────────────────────────────
# Updated with the exact filenames you provided.
TASKS = {
    "Completion": dict(
        lang="java",
        metrics=("HybridExact", "CodeBLEU", "CrystalBLEU", "NLS", "EmbeddingSim"),
        files={
            "Baseline": "generated_results_baseline_completion_llama_results_with_groups.json",
            "Ours (Reading)": "generated_results_advanced_final_reading_completion_with_groups.json",
            "Ours (Writing)": "generated_results_advanced_final_writing_completion_with_groups.json",
            "Ours (Combined)": "generated_results_advanced_final_completion_llama_results_with_groups.json",
        }
    ),
    "Translation": dict(
        lang="c_sharp",
        metrics=("HybridExact", "CodeBLEU", "CrystalBLEU", "NLS", "EmbeddingSim"),
        files={
            "Baseline": "generated_results_baseline_translation_llama_results_with_groups.json",
            "Ours (Reading)": "generated_results_advanced_final_reading_translation_with_groups.json",
            "Ours (Writing)": "generated_results_advanced_final_writing_translation_with_groups.json",
            "Ours (Combined)": "generated_results_advanced_final_translation_llama_results_with_groups.json",
        }
    ),
    "Summarization": dict(
        lang="java",
        metrics=("ROUGE1", "ROUGEL", "METEOR", "BERTScore", "EmbeddingSim"),
        files={
            "Baseline": "generated_results_baseline_summarization_llama_results_with_groups.json",
            "Ours (Reading)": "generated_results_advanced_final_reading_summarization_with_groups.json",
            "Ours (Writing)": "generated_results_advanced_final_writing_summarization_with_groups.json",
            "Ours (Combined)": "generated_results_advanced_final_summarization_llama_results_with_groups.json",
        }
    ),
}

# ────── **SIMPLIFIED** data loading helper ──────────────────────────────────
def load_grouped_data(path: str, task: str, do_clean: bool):
    """Loads data from a JSON file and groups it by the 'group_label' field."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    grouped_data = defaultdict(lambda: {"preds": [], "refs": []})
    for item in data:
        group = item.get("group_label", "default_group")
        pred = clean(item.get("generated_output", ""), task)
        ref = [clean(item.get("ground_truth", ""), task)]
        grouped_data[group]["preds"].append(pred)
        grouped_data[group]["refs"].append(ref)
    return dict(grouped_data)

# ────── evaluation runner (UNCHANGED) ───────────────────────────────────────
def evaluate(preds, refs, lang, metrics, use_embeddings):
    """Runs a suite of evaluation metrics."""
    o={}
    if "HybridExact" in metrics: o["HybridExact"] = hybrid_exact(preds, refs)
    if "CodeBLEU"    in metrics: o["CodeBLEU"]    = codebleu(preds, refs, lang)
    if "CrystalBLEU" in metrics: o["CrystalBLEU"] = crystalbleu(preds, refs)
    if "NLS"         in metrics: o["NLS"]         = nls(preds, refs)
    if "ROUGE1" in metrics or "ROUGEL" in metrics: o.update(rouge1_l(preds, refs))
    if "METEOR"      in metrics: o["METEOR"]      = meteor_avg(preds, refs)
    if "BERTScore"   in metrics: o["BERTScore"]   = bscore(preds, refs)
    if "EmbeddingSim" in metrics and use_embeddings:
        o["EmbeddingSim"] = embedding_similarity(preds, refs, EMBEDDING_MODEL)
    return o

# ────── results presentation (UNCHANGED from previous version) ──────────────
def print_group_tables(results):
    """Prints detailed, group-based results in a single, formatted table."""
    col_order = ["Baseline", "Ours (Reading)", "Ours (Writing)", "Ours (Combined)"]
    
    header = f"{'Task':<14} | {'Group':<12} | {'Metric':<14} | "
    header += " | ".join([f"{col:>16}" for col in col_order])
    print("\n" + "=" * len(header))
    print(" DETAILED GROUP-BASED EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    all_metrics = sorted({
        metric for res in results.values() for grp in res.values() for cat in grp.values() for metric in cat
    })

    for task in sorted(results.keys()):
        task_results = results[task]
        is_first_row_for_task = True
        for group in sorted(task_results.keys()):
            group_results = task_results[group]
            is_first_row_for_group = True
            for metric in all_metrics:
                if not any(metric in group_results.get(cat, {}) for cat in col_order):
                    continue

                row_data = {cat: group_results.get(cat, {}).get(metric, float('nan')) for cat in col_order}
                valid_scores = [s for s in row_data.values() if not math.isnan(s)]
                best_score = max(valid_scores) if valid_scores else float('-inf')

                cells = []
                for cat in col_order:
                    score = row_data[cat]
                    if math.isnan(score):
                        cells.append(f"{'n/a':>16}")
                    else:
                        cell_str = f"{score:13.2f}"
                        if valid_scores and score >= best_score - 1e-9:
                            cell_str = f"🏆 {cell_str}"
                        cells.append(f"{cell_str:>16}")
                
                task_display = task if is_first_row_for_task else ""
                group_display = group if is_first_row_for_group else ""
                print(f"{task_display:<14} | {group_display:<12} | {metric:<14} | {' | '.join(cells)}")
                is_first_row_for_task = is_first_row_for_group = False
            
            if len(task_results) > 1 and group != sorted(task_results.keys())[-1]:
                 print("-" * len(header))
        if task != sorted(results.keys())[-1]:
            print("=" * len(header))


# ────── **REVISED** main execution logic ────────────────────────────────────
def run_group_evaluation(config):
    """Main group-based evaluation function."""
    if config['use_embeddings'] and EMBEDDING_MODEL is None:
        print("ERROR: 'sentence-transformers' is required for embeddings but could not be loaded.", file=sys.stderr)
        print("Please run: pip install sentence-transformers", file=sys.stderr)
        config['use_embeddings'] = False

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for task, cfg in tqdm(TASKS.items(), desc="Evaluating Tasks"):
        for tag, path in tqdm(cfg["files"].items(), desc=f"  Files for {task}", leave=False):
            if not os.path.exists(path):
                print(f"\nWarning: File not found for '{task} - {tag}' at '{path}'. Skipping.", file=sys.stderr)
                continue

            # Load and group data directly from each file
            grouped_data = load_grouped_data(path, task, config['clean'])
            
            # Calculate metrics for each group found in the file
            for group_label, data in grouped_data.items():
                scores = evaluate(data["preds"], data["refs"], cfg["lang"], cfg["metrics"], config['use_embeddings'])
                results[task][group_label][tag] = scores
                
    print_group_tables(results)

# ==================================================================================
# 🚀 RUN THE SCRIPT HERE
# ==================================================================================

if __name__ == '__main__':
    # --- Set your configuration options ---
    Config = {
        "clean": True,            # Set to True to clean code/text, False otherwise.
        "use_embeddings": True,   # Set to True to enable the embedding similarity metric.
    }

    # --- Run the evaluation ---
    run_group_evaluation(Config)