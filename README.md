# EyeMulator (Extended Version) — Artifact

![License: MIT (code) / CC-BY-4.0 (data)](https://img.shields.io/badge/license-MIT%20%2F%20CC--BY--4.0-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Artifact for the **extended version of EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention**
by Yifan Zhang, Chen Huang, Yueke Zhang, Jiahao Zhang, Toby Li, Collin McMillan, Kevin Leach, and Yu Huang.

**What EyeMulator does.** EyeMulator aligns code language models with human visual attention. Eye-tracking data is distilled into a small set of reusable priors — Beta distributions over semantic token classes plus n-gram transition counts. Pseudo-scan paths are generated from those priors over arbitrary code, and the model is trained with a weighted cross-entropy loss combined with a token-level preference loss.

**This release.** This is the extended version of the original EyeMulator paper (Zhang et al., ACL 2026): it adds an updated method implementation, more backbones, and expanded analyses. The full extended write-up ships with the repository as [`EyeMulator_Extended.pdf`](EyeMulator_Extended.pdf) — this is what "the paper" refers to below. The human visual-attention data is sourced from the EyeTrans eye-tracking study (Zhang et al., FSE'24).

**Two layers.**

- **Human-attention layer** — `priors/`, `dataset_sample/`, `docs/`, `example/`, `figures/`: inspect or integrate the distilled priors, with a reference PyTorch implementation of the method components.
- **Experiment layer** — `src/`, `experiments/`, `corpus/`, `paper_results/`: rerun the full training, evaluation, and table-generation pipelines.

## Repository layout

```
EyeMulator/
├── README.md
├── LICENSE                         MIT (code) + CC-BY-4.0 attribution (data)
├── CITATION.bib
├── EyeMulator_Extended.pdf         extended EyeMulator write-up
├── priors/
│   ├── combined/                   distilled from reading + writing sessions
│   ├── reading/                    reading-only sessions
│   └── writing/                    writing-only sessions
├── dataset_sample/                 30 examples per split per task; same schema as a full dataset
│   ├── completion_{train,valid,test}_sample.jsonl
│   ├── summarization_{train,valid,test}_sample.jsonl
│   └── translation_{train,valid,test}_sample.jsonl
├── figures/                        human-side figures from the write-up
│   ├── human_study.pdf
│   ├── eyemulator_overview.pdf
│   ├── eyemulator_pseudo_path.pdf
│   ├── combined_beta_distributions.pdf
│   ├── combined_beta_curves.pdf
│   └── category_distribution.pdf
├── docs/
│   ├── data_schema.md              field-by-field format of priors and dataset
│   ├── method_integration.md       how to wire the priors into a training loop
│   └── human_attention_analysis.md distribution analysis of the priors + figure index
├── example/
│   ├── analyze_human_attention.py  summarize Beta params and top n-grams from priors
│   ├── compute_token_weights.py    load priors and compute per-token weight w_j
│   └── weighted_sft_template.py    reference implementation of the method components
├── src/                            training, evaluation, metrics, and analysis scripts
├── experiments/                    shell drivers for full-grid runs and low-data sweeps
├── corpus/                         full task splits and session-specific gaze priors
├── paper_results/                  metric JSONs accompanying the paper tables
├── environment_eyemulator.yml      conda environment for reproduction
└── REPRODUCIBILITY.md              end-to-end experiment reproduction guide
```

## Origin of the eye-tracking data

All priors in this release are derived from the EyeTrans corpus collected by [Zhang et al., 2024, *EyeTrans: Merging Human and Machine Attention for Neural Code Summarization*](https://doi.org/10.1145/3660807), in studies conducted at the University of Notre Dame under the appropriate IRB protocols. We thank those authors and Notre Dame for making this work possible.

## Quick start

```bash
git clone https://github.com/CoderDoge1108/EyeMulator.git
cd EyeMulator

python example/compute_token_weights.py \
    --priors priors/combined \
    --jsonl  dataset_sample/completion_train_sample.jsonl \
    --limit  2
```

This prints two examples with their per-token human-attention weights `w_j`, using only the Python standard library.

## Inspecting the priors

To reproduce the distribution analysis from the paper — posterior salience per semantic label, and the most frequent monogram / bigram / trigram fixation transitions — run:

```bash
python example/analyze_human_attention.py --priors priors/combined --top 10
```

The same script accepts `--priors priors/reading` or `--priors priors/writing`, and `--plot beta.pdf` renders the Beta density curves (requires `matplotlib`). A walkthrough of what each figure shows, together with the paper's Table 1 reproduced inline, is in [`docs/human_attention_analysis.md`](docs/human_attention_analysis.md). The original PDF figures are in [`figures/`](figures).

## Using the method in a training pipeline

```bash
pip install torch transformers
```

[`docs/method_integration.md`](docs/method_integration.md) describes how to plug the priors into a training loop. The components in [`example/weighted_sft_template.py`](example/weighted_sft_template.py), named after Algorithm 1 in the paper, are:

- `sample_attention_density` — sample `ρ ~ Beta(α_agg, β_agg)`.
- `generate_pseudo_scan_path` — build a pseudo-scan path `P̃` from the priors and `ρ`.
- `token_weight` — the per-token weight `w_j = w_base + 1/log(freq(g_j)+2) + E[θ_{s_j}]`.
- `CausalLMWithWeightedLoss` — weighted causal-LM loss `L_SFT`.
- `token_level_preference_loss` — token-level preference term against a frozen reference policy.
- `EyeMulatorCompositeObjective` — the composite `L_total = L_SFT + γ · L_pref`.
- `WeightedCollator`, `build_training_example` — batching and preprocessing helpers.

The file is backbone-agnostic (swap `LlamaForCausalLM` for whichever model you use) and does not hard-code our training schedule, so it composes with an existing `Trainer`, `accelerate`, or custom loop.

## Reproducing the paper experiments

For the complete training/evaluation pipeline, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). In brief:

```bash
conda env create -f environment_eyemulator.yml
conda activate eyemulator
./experiments/run_experiments.sh
```

The scripts write trained adapters to `workspace/` and generated predictions to `results/`, both of which are intentionally git-ignored. The checked-in `paper_results/` directory contains the metric JSONs used to audit the paper tables.

## Directions worth trying

- Larger backbones (7B / 13B / 70B) on the same three tasks.
- Larger training sets, including non-Java code and more CodeXGLUE tasks.
- Parameter-efficient variants (LoRA, QLoRA) on top of the weights.
- Alternative preference objectives (IPO, KTO, SimPO, token-level DPO variants).

If you try any of these, we'd be glad to hear about it — please open an issue.

## Citing

This artifact extends EyeMulator; please cite both the EyeMulator paper and the EyeTrans dataset. BibTeX is in [`CITATION.bib`](CITATION.bib).

## License

- Code (`example/`): MIT License. See [`LICENSE`](LICENSE).
- Data and documentation (`priors/`, `dataset_sample/`, `figures/`, `docs/`): CC-BY-4.0.

The underlying eye-tracking data originates from Zhang et al., *EyeTrans* (FSE'24); please credit that source as well.

## Archival copy

An archival copy of this artifact is deposited on Zenodo for long-term citability: [https://zenodo.org/records/16134801](https://zenodo.org/records/16134801).

## Contact

For questions or issues, please open a GitHub issue, or contact the corresponding authors at the email addresses on the paper.
