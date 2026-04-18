# EyeMulator — Human-Attention Artifact

![License: MIT (code) / CC-BY-4.0 (data)](https://img.shields.io/badge/license-MIT%20%2F%20CC--BY--4.0-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Companion artifact for the ACL 2026 paper
**"EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention"**
by Yifan Zhang, Chen Huang, Yueke Zhang, Jiahao Zhang, Toby Li, Collin McMillan, Kevin Leach, and Yu Huang.

This repository is the **human-attention artifact release**. It is **not** a full code-reproduction package. Its purpose is to make the human-derived signals used by EyeMulator — distilled gaze priors and per-token human-attention annotations — publicly available so that other researchers can build on them.

The included example code (`example/`) is a collection of **reference snippets**, not an end-to-end training pipeline. It illustrates the key building blocks (the weighted causal-LM loss, the data collator, and the per-token weight formula) so that other researchers can integrate the EyeMulator method into their own codebases and their own datasets. This release deliberately does **not** reproduce our paper's exact training runs.

## What this release contains

```
EyeMulator-HumanArtifact/
├── README.md                     ← this file
├── LICENSE                       ← CC-BY-4.0
├── CITATION.bib                  ← BibTeX for the paper
├── priors/
│   ├── combined/                 ← priors distilled from combined reading + writing sessions
│   ├── reading/                  ← priors distilled from reading-only sessions
│   └── writing/                  ← priors distilled from writing-only sessions
├── dataset_sample/               ← small demonstration sample (30 examples per split per task)
│   ├── completion_{train,valid,test}_sample.jsonl
│   ├── summarization_{train,valid,test}_sample.jsonl
│   └── translation_{train,valid,test}_sample.jsonl
├── docs/
│   ├── DATA_SCHEMA.md            ← format of priors + dataset fields
│   └── METHOD_INTEGRATION.md     ← step-by-step description of how to use the artifacts
└── example/
    ├── compute_token_weights.py  ← runnable demo: loads priors, computes per-token weight w_j
    └── weighted_sft_template.py  ← reference snippets: weighted loss, collator, preprocessing
```

## What this release does *not* contain

This release deliberately excludes the following:

- Full train / validation / test splits for any task. The `dataset_sample/` folder only carries enough examples to illustrate the schema and exercise the demo script.
- The end-to-end training and evaluation pipeline used in the paper.
- Pre-trained model checkpoints or generation outputs from our experiments.
- The upstream CodeXGLUE source code (available separately from the CodeXGLUE benchmark).

These exclusions are intentional: the artifact is meant to let others **adopt our method on their own data**, not to reproduce our exact numerical results. The full training and evaluation code will be released on GitHub at a later date; if you need early access for review or independent replication, please contact the corresponding authors.

## Origin of the human-attention data

The gaze data used to derive all priors was collected by [Zhang et al., 2024, *EyeTrans: Merging Human and Machine Attention for Neural Code Summarization*](https://doi.org/10.1145/3660807), in studies conducted at the University of Notre Dame under the appropriate IRB protocols. We thank those authors and the University of Notre Dame for enabling this work.

## Quick start

```bash
git clone https://github.com/CoderDoge1108/EyeMulator.git
cd EyeMulator

python example/compute_token_weights.py \
    --priors priors/combined \
    --jsonl  dataset_sample/completion_train_sample.jsonl \
    --limit  2
```

The demo prints two sample examples with their per-token human-attention weights `w_j`. It uses only the Python standard library.

If you want to adapt `example/weighted_sft_template.py` into your own training code, install its additional dependencies:

```bash
pip install torch transformers
```

Then read `docs/METHOD_INTEGRATION.md` for the conceptual walk-through and copy the snippets in `example/weighted_sft_template.py` into your own training pipeline.

## Intended use

These artifacts are intended for **research** on human-attention-informed training of code language models. They are not production assets. When reporting results, please cite the paper (see `CITATION.bib`) and acknowledge the underlying EyeTrans dataset.

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
