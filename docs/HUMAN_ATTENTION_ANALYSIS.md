# Human-Attention Analysis

This document describes the distribution analysis of the human-attention
artifacts shipped with this repository. It mirrors the "RQ1: Artifact
Distillation" discussion in the EyeMulator paper and is intended to help
other researchers understand the structure of the priors before using them
downstream.

All figures referenced here live in [`../figures/`](../figures). The raw
counts and Beta parameters they summarize are in [`../priors/`](../priors).
You can regenerate the underlying statistics on your own machine with:

```bash
python example/analyze_human_attention.py --priors priors/combined --top 10
```

## 1. Eye-tracking data source

![](../figures/human_study.pdf)

**`figures/human_study.pdf`** gives an overview of the eye-tracking study
whose data was distilled into these priors. The study was conducted by
[Zhang et al., 2024 (EyeTrans, FSE'24)](https://doi.org/10.1145/3660807) at
the University of Notre Dame under the appropriate IRB protocols. Twenty-seven
programmers read and wrote Java code while their gaze was recorded at 120 Hz.
A dispersion-threshold (I-DT) algorithm extracted fixations (>= 100 ms), which
were then spatially mapped to AST leaf tokens. The resulting 1,565 scan paths
across reading and writing sessions are the source for every distilled artifact
in this repository.

## 2. Method overview

![](../figures/eyemulator_overview.pdf)

**`figures/eyemulator_overview.pdf`** is the high-level pipeline: raw gaze is
turned into (i) semantic salience priors (Beta parameters per token class) and
(ii) n-gram transition models, and those are used to synthesize pseudo-scan
paths over arbitrary code inputs. The paths are then consumed by the weighted
SFT + token-level preference objective in training. The reference
implementation of each stage lives in
[`../example/weighted_sft_template.py`](../example/weighted_sft_template.py).

## 3. Pseudo-scan path construction

![](../figures/eyemulator_pseudo_path.pdf)

**`figures/eyemulator_pseudo_path.pdf`** visualizes how a pseudo-scan path is
assembled for a new code sequence. Given an input of length `n`, the procedure
(1) samples an attention density `rho ~ Beta(alpha_agg, beta_agg)` aggregated
across all labels, (2) allocates the resulting token budget `m = floor(rho * n)`
across semantic classes proportional to their posterior means, and (3) greedily
stitches masked tokens into valid trigram / bigram / monogram transitions
subject to a line-span constraint.

## 4. Semantic salience priors (Beta parameters)

![](../figures/combined_beta_distributions.pdf)
![](../figures/combined_beta_curves.pdf)

For every semantic label `s`, the distillation step fits
`Beta(alpha_s, beta_s)` with `alpha_s = c1(s) + 1` (fixations) and
`beta_s = n_tok(s) - c1(s) + 1` (misses). The posterior mean
`E[theta_s] = alpha_s / (alpha_s + beta_s)` gives a smoothed salience
estimate. `figures/combined_beta_distributions.pdf` plots `alpha_s` against
`beta_s` for each label; `figures/combined_beta_curves.pdf` shows the full
density. Labels like `function declaration` are sharply peaked at high
attention probability, indicating consistent inspection, while `loop` and
`conditional statement` exhibit broader, sometimes bimodal densities,
reflecting context-dependent reading.

The data behind these plots is [`../priors/combined/beta_distribution.json`](../priors/combined/beta_distribution.json).
Equivalent files live under `reading/` and `writing/` for the per-session
views.

## 5. Semantic category distribution across tasks

![](../figures/category_distribution.pdf)

**`figures/category_distribution.pdf`** shows how semantic categories are
distributed differently across code completion, translation, and summarization
corpora. Completion is dominated by structural boilerplate, while summarization
skews toward API contract tokens. This motivates offering separate
reading-derived and writing-derived priors so practitioners can pick the
variant best matched to their task.

## 6. N-gram fixation patterns (Table 1 in the paper)

Expert attention is organized rather than linear. The table below reproduces
the representative transitions reported in the paper; you can generate the
full top-k list from any of the three prior folders with
`python example/analyze_human_attention.py`.

| Type     | Sequence                                                  | Count |
| :------- | :-------------------------------------------------------- | ----: |
| Mono     | variable declaration                                      | 18665 |
| Mono     | conditional statement                                     | 13222 |
| Bigram   | function declaration &rarr; variable declaration          |  8399 |
| Bigram   | conditional statement &rarr; loop                         |  6026 |
| Trigram  | function declaration &rarr; parameter &rarr; variable decl|  1634 |
| Trigram  | conditional statement &rarr; function decl &rarr; parameter|  1199 |

The underlying counts are in
[`../priors/combined/monogram_counts.json`](../priors/combined/monogram_counts.json),
[`bigram_counts.json`](../priors/combined/bigram_counts.json), and
[`trigram_counts.json`](../priors/combined/trigram_counts.json). The
`monogram_counts.json` file contains a few raw character entries alongside
semantic-label entries; `analyze_human_attention.py` filters the monogram
table against the Beta label vocabulary so the reported counts correspond to
semantic classes only.

## 7. Reproducing the analysis

```bash
python example/analyze_human_attention.py --priors priors/combined --top 10
python example/analyze_human_attention.py --priors priors/reading  --top 10
python example/analyze_human_attention.py --priors priors/writing  --top 10
```

To render your own Beta density curves (requires `matplotlib`):

```bash
pip install matplotlib
python example/analyze_human_attention.py --priors priors/combined --plot beta_combined.pdf
```

The three prior folders share the same schema; the differences between
`combined/`, `reading/`, and `writing/` are discussed in
[`DATA_SCHEMA.md`](DATA_SCHEMA.md).

## Attribution

The raw gaze data these priors summarize was collected by Zhang et al. for
EyeTrans (FSE'24). If you use any of the priors, please cite both that work
and the EyeMulator paper; BibTeX is provided in
[`../CITATION.bib`](../CITATION.bib).
