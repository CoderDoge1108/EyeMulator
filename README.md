# EyeMulator — Human-Attention Artifact

![License: MIT (code) / CC-BY-4.0 (data)](https://img.shields.io/badge/license-MIT%20%2F%20CC--BY--4.0-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Companion artifact for the ACL 2026 paper
**"EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention"**
by Yifan Zhang, Chen Huang, Yueke Zhang, Jiahao Zhang, Toby Li, Collin McMillan, Kevin Leach, and Yu Huang.

This repository releases the human-derived signals used by EyeMulator — distilled gaze priors and per-token human-attention annotations — together with documentation and reference snippets that show how to integrate those signals into a supervised fine-tuning loop. It is designed as a **curated starter kit for the community to scale EyeMulator up to larger backbones and larger datasets than those used in our original study**.

## Why this release is scoped this way

Our original experiments in the paper were conducted on compact backbones (StarCoder, Llama-3.2-1B, DeepSeek-Coder) and on task-specific subsets of CodeXGLUE (see Section 3 of the paper). This reflected the GPU budget available to us at the time. Multiple reviewers, as well as readers of the preprint, raised the natural follow-up of evaluating the method on larger backbones and more diverse codebases. We agree this is the right next step.

To make that follow-up practical, this release provides the parts that are **hard** to reconstruct — the distilled gaze priors, which were derived from non-trivial human-subject studies — alongside lightweight reference snippets that can be dropped into any standard Hugging Face training loop. The full training pipeline and evaluation harness from our own small-scale experiments add relatively little value on top of that, and are kept in our internal research repository so that this package stays focused, easy to adopt, and backbone-agnostic. We welcome issues and pull requests from groups that scale the method up in new directions.

## What's in this repository

```
EyeMulator/
├── README.md
├── LICENSE                        ← MIT (code) + CC-BY-4.0 attribution (data)
├── CITATION.bib                   ← BibTeX for the paper
├── priors/
│   ├── combined/                  ← priors distilled from combined reading + writing sessions
│   ├── reading/                   ← priors distilled from reading-only sessions
│   └── writing/                   ← priors distilled from writing-only sessions
├── dataset_sample/                ← ready-to-run demonstration set (30 examples per split per task)
│   ├── completion_{train,valid,test}_sample.jsonl
│   ├── summarization_{train,valid,test}_sample.jsonl
│   └── translation_{train,valid,test}_sample.jsonl
├── docs/
│   ├── DATA_SCHEMA.md             ← format of priors + dataset fields
│   └── METHOD_INTEGRATION.md      ← step-by-step guide for integrating the method
└── example/
    ├── compute_token_weights.py   ← runnable demo: loads priors, computes per-token weight w_j
    └── weighted_sft_template.py   ← reference snippets for weighted-SFT integration
```

The `dataset_sample/` folder is intentionally a compact, balanced sample rather than the full split used in our paper — its purpose is to get new researchers from clone to working training loop in a few minutes. The schema is open (see `docs/DATA_SCHEMA.md`), so you can readily produce a full-scale dataset in the same format using your own data pipeline.

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

This prints two sample examples with their per-token human-attention weights `w_j`. The demo uses only the Python standard library.

To wire the method into your own training pipeline, install the additional dependencies:

```bash
pip install torch transformers
```

Then read `docs/METHOD_INTEGRATION.md` for the conceptual walk-through and copy the snippets in `example/weighted_sft_template.py` into your own trainer, pointing them at your own data and your own backbone.

## Scaling up — suggested next experiments

Some directions we think would be high-value follow-ups, and which this release is designed to support:

- Apply the weighting scheme to **larger backbones** (7B, 13B, 70B class models) on the same three tasks.
- Evaluate on **larger training sets**, including non-Java codebases and additional CodeXGLUE tasks.
- Combine the weights with **parameter-efficient fine-tuning** (LoRA, QLoRA) to reduce compute further.
- Explore **richer weight schedules** — the included snippets use a simple mean-weight aggregation for output tokens, which is just one design point among many.

If you run any of the above, we'd love to hear about it — please open a GitHub issue or reach out to the corresponding authors.

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

For questions, scale-up results, or bug reports, please open a GitHub issue on this repository, or contact the corresponding authors at the email addresses listed on the paper.
