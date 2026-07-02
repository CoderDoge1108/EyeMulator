# EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention

Reproducibility artifact for **EyeMulator**. EyeMulator fine-tunes code language
models with a token-weighted loss derived from human visual-attention priors
(distilled from eye-tracking data), so the model learns to emphasize the tokens
that humans attend to most. This repository contains the code, the distilled
attention priors, the training/evaluation datasets, and the exact metric outputs
used to build the paper's tables and figures.

> This is the **code-first** layer of the artifact, focused on end-to-end
> reproduction of the experiments. The lightweight human-attention layer remains
> in `docs/`, `example/`, `priors/`, `dataset_sample/`, and `figures/`.

## Repository layout

```
EyeMulator/
├── src/
│   ├── training/                 # model training
│   │   ├── train_unified.py      # canonical CLI trainer (all backbones/tasks/variants)
│   │   └── train_<model>_*.py    # model-specific training entry points
│   ├── evaluation/
│   │   ├── eval_unified.py       # canonical CLI evaluator (generates predictions)
│   │   └── eval_<model>_*.py     # model-specific evaluation entry points
│   ├── analysis/                 # produce current result tables from predictions
│   │   ├── cross_task_evaluation_unified.py   # RQ2 cross-task table
│   │   ├── session_mode_evaluation.py         # RQ3 session-mode table
│   │   ├── session_mode_evaluation_labeling.py # RQ3 LLM-assisted labeling step
│   │   └── participant_variability_analysis.py
│   └── compute_metrics.py        # Exact Match / METEOR scoring for predictions
├── experiments/
│   ├── run_experiments.sh        # drive the full (model × task × method) grid
│   └── targeted_reruns.sh        # GPU-aware targeted low-data reruns
├── corpus/
│   ├── data/                     # main train/valid/test splits + attention priors
│   ├── data_reading/             # reading-session attention priors
│   ├── data_writing/             # writing-session attention priors
│   ├── data_session_reading/     # RQ3 session priors (large .jsonl excluded, see note)
│   └── data_session_writing/     # RQ3 session priors (large .jsonl excluded, see note)
├── paper_results/                # metric JSONs backing the paper tables/figures
├── environment_eyemulator.yml    # conda environment
├── .gitignore
└── README.md
```

`workspace/` (trained adapters) and `results/` (generated predictions) are created
on demand by the scripts and are git-ignored.

## Setup

Requires an NVIDIA GPU with CUDA and Conda/Miniconda.

```bash
conda env create -f environment_eyemulator.yml
conda activate eyemulator
```

Set `HF_TOKEN` if any backbone requires gated access (e.g. Llama). Leave it unset
otherwise — an empty token triggers an illegal `Bearer` header on some HF stacks.

## Supported backbones

`train_unified.py` / `eval_unified.py` accept these `--model` keys:

| key           | HuggingFace model                     |
|---------------|---------------------------------------|
| `llama`       | meta-llama/Llama-3.2-1B               |
| `deepseek`    | deepseek-ai/deepseek-coder-1.3b-base  |
| `starcoder`   | bigcode/starcoderbase-1b              |
| `qwen35-2b`   | Qwen/Qwen2.5-Coder-1.5B               |
| `smollm3-3b`  | HuggingFaceTB/SmolLM3-3B              |

Tasks: `completion`, `translation`, `summarization`.
Methods / variants: `baseline`, `eyemulator`, `random`, `eyetracking_only`,
`weighted_sft_only` (the last three are the gaze-signal ablation variants).

## Reproducing the main results (RQ1 / RQ2 / RQ4)

### Option A — one command per cell

```bash
# Baseline vs. EyeMulator for one cell
python src/training/train_unified.py \
    --model llama --task completion --method baseline \
    --data-folder corpus/data --output-root workspace --seed 42
python src/evaluation/eval_unified.py \
    --model llama --task completion --method baseline \
    --data-folder corpus/data --model-root workspace \
    --output-root results --use-test-set --seed 42

# ...repeat with --method eyemulator, then score:
python src/compute_metrics.py --task completion \
    --compare_files results/llama_results/generated_results_baseline_completion_seed42.json \
                    results/llama_results/generated_results_eyemulator_completion_seed42.json \
    --compare_names Baseline EyeMulator \
    --output_file results/metrics_llama_completion.json
```

### Option B — drive the whole grid

```bash
# Full-data setting across all backbones/tasks (baseline + eyemulator)
./experiments/run_experiments.sh

# Low-data setting (e.g. 200 training samples per cell)
MAX_SAMPLES=200 ./experiments/run_experiments.sh

# Subset
MODELS="qwen35-2b smollm3-3b" TASKS="completion" ./experiments/run_experiments.sh
```

Adapters land in `workspace/`, predictions in `results/`. Then score each cell
with `src/compute_metrics.py` as shown above.

### Targeted low-data reruns

`experiments/targeted_reruns.sh` waits for a free GPU, skips completed
cells, and runs selected low-data recipes. Edit the `run_job` lines at the
bottom to target other cells or hyperparameters.

## Ablation / gaze-signal variants (RQ4)

Run the same cells with `--method` set to `random`, `eyetracking_only`, or
`weighted_sft_only` to compare weighting signals against the full `eyemulator`
signal and the plain `baseline`.

## Session-mode analysis (RQ3)

1. Label predictions into reading/writing groups (needs an OpenAI API key):
   `python src/analysis/session_mode_evaluation_labeling.py`
2. Aggregate grouped metrics: `python src/analysis/session_mode_evaluation.py`

The session-mode split uses the priors in `corpus/data_session_reading` and
`corpus/data_session_writing`. The large per-session `.jsonl` splits are excluded
from version control (see `.gitignore`); regenerate or copy them into those
folders before running the session-mode pipeline.

## `paper_results/`

The JSON files here are the exact scored metrics used to build the paper's
tables/figures (e.g. `current_rq2_best_metrics_alt.json` for the main cross-task
table, `current_rq4_ablation_metrics.json` for the component ablation,
and `current_rq3_*` for session mode). Use them to cross-check regenerated
numbers. Attention-map diagnostics are excluded from this code-first layer
because they are distributed as qualitative appendix diagnostics rather than as
part of the unified training/evaluation pipeline.
