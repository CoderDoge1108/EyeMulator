# EyeMulator — Human-Attention Artifact

![License: MIT (code) / CC-BY-4.0 (data)](https://img.shields.io/badge/license-MIT%20%2F%20CC--BY--4.0-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Companion artifact for the ACL 2026 paper
**"EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention"**
by Yifan Zhang, Chen Huang, Yueke Zhang, Jiahao Zhang, Toby Li, Collin McMillan, Kevin Leach, and Yu Huang.

EyeMulator aligns code language models with human visual attention by distilling eye-tracking data into reusable priors, generating pseudo-scan paths over code tokens, and training the model with a composite objective that combines a weighted supervised fine-tuning loss with a token-level preference loss. This repository provides the distilled priors, a demonstration dataset, and a reference implementation of the method components in PyTorch.

## Repository layout

```
EyeMulator/
├── README.md
├── LICENSE                        ← MIT (code) + CC-BY-4.0 attribution (data)
├── CITATION.bib                   ← BibTeX for the paper
├── priors/
│   ├── combined/                  ← priors distilled from combined reading + writing sessions
│   ├── reading/                   ← priors distilled from reading-only sessions
│   └── writing/                   ← priors distilled from writing-only sessions
├── dataset_sample/                ← 30 examples per split per task, same schema as a full-scale dataset
│   ├── completion_{train,valid,test}_sample.jsonl
│   ├── summarization_{train,valid,test}_sample.jsonl
│   └── translation_{train,valid,test}_sample.jsonl
├── figures/                       ← PDF figures from the paper for the human-attention side
│   ├── human_study.pdf
│   ├── eyemulator_overview.pdf
│   ├── eyemulator_pseudo_path.pdf
│   ├── combined_beta_distributions.pdf
│   ├── combined_beta_curves.pdf
│   └── category_distribution.pdf
├── docs/
│   ├── DATA_SCHEMA.md               ← format of priors + dataset fields
│   ├── METHOD_INTEGRATION.md        ← walkthrough of the method components
│   └── HUMAN_ATTENTION_ANALYSIS.md  ← distribution analysis of the priors + figure index
└── example/
    ├── analyze_human_attention.py ← summarize Beta params and top n-grams from the priors
    ├── compute_token_weights.py   ← runnable demo: loads priors, computes per-token weight w_j
    └── weighted_sft_template.py   ← reference implementation of the method components
```

## Origin of the human-attention data

The gaze data used to derive all priors was collected by [Zhang et al., 2024, *EyeTrans: Merging Human and Machine Attention for Neural Code Summarization*](https://doi.org/10.1145/3660807), in studies conducted at the University of Notre Dame under the appropriate IRB protocols. We gratefully acknowledge those authors and the University of Notre Dame for enabling this work.

## Quick start

```bash
git clone https://github.com/CoderDoge1108/EyeMulator.git
cd EyeMulator

python example/compute_token_weights.py \
    --priors priors/combined \
    --jsonl  dataset_sample/completion_train_sample.jsonl \
    --limit  2
```

This prints two examples with their per-token human-attention weights `w_j`. The demo uses only the Python standard library.

## Inspecting the human-attention priors

To reproduce the distribution analysis from the paper (posterior salience per semantic label, most frequent monogram / bigram / trigram fixation transitions), run:

```bash
python example/analyze_human_attention.py --priors priors/combined --top 10
```

The same script accepts `--priors priors/reading` or `--priors priors/writing` for the per-session views, and `--plot beta.pdf` to render the Beta density curves (requires `matplotlib`). A walkthrough of the priors, together with the corresponding figures from the paper, lives in [`docs/HUMAN_ATTENTION_ANALYSIS.md`](docs/HUMAN_ATTENTION_ANALYSIS.md); the original PDF figures are in [`figures/`](figures).

## Wiring the method into a training pipeline

To wire the method into a training pipeline, install the additional dependencies:

```bash
pip install torch transformers
```

Then read `docs/METHOD_INTEGRATION.md` for the full walkthrough and copy the components from `example/weighted_sft_template.py` into your own trainer. The reference file provides, in the order in which Algorithm 1 of the paper uses them:

- `sample_attention_density` — sample `ρ ~ Beta(α_agg, β_agg)`.
- `generate_pseudo_scan_path` — emit a pseudo-scan path `P̃` from the priors and `ρ`.
- `token_weight` — the per-token weight formula `w_j = w_base + 1/log(freq(g_j)+2) + E[θ_{s_j}]`.
- `CausalLMWithWeightedLoss` — weighted causal-LM loss `L_SFT = −(1/|P̃|) Σ_{j∈P̃} w_j log P_φ(x_j | x_{<j})`.
- `token_level_preference_loss` — token-level preference term comparing the policy with a frozen reference over the preferred and dispreferred token sets.
- `EyeMulatorCompositeObjective` — the composite `L_total = L_SFT + γ · L_pref`.
- `WeightedCollator`, `build_training_example` — batching and preprocessing helpers.

## Scaling up — suggested next experiments

- Apply the composite objective to **larger backbones** (7B, 13B, 70B class models) on the same three tasks.
- Evaluate on **larger training sets**, including non-Java codebases and additional CodeXGLUE tasks.
- Combine the weights with **parameter-efficient fine-tuning** (LoRA, QLoRA) to reduce compute further.
- Explore **richer weight schedules** and **alternative preference objectives** (IPO, KTO, SimPO, token-level DPO variants).

If you run any of the above, we'd love to hear about it — please open a GitHub issue.

## Citing

If you use these artifacts, please cite the EyeMulator paper and the EyeTrans dataset; BibTeX is in `CITATION.bib`.

## License

- **Code** (`example/`, any future utility scripts): MIT License. See `LICENSE`.
- **Data and documentation** (`priors/`, `dataset_sample/`, `docs/`): Creative Commons Attribution 4.0 International (CC-BY-4.0).

The underlying eye-tracking data originates from Zhang et al., *EyeTrans* (FSE'24); please credit that source as well.

## Archival copy

An archival copy of this artifact is deposited on Zenodo for long-term citability:
[https://zenodo.org/records/16134801](https://zenodo.org/records/16134801).

## Contact

For questions or issues, please open a GitHub issue on this repository, or contact the corresponding authors at the email addresses listed on the paper.
