# Human-attention analysis

This note is a short walkthrough of what the shipped priors look like when you unpack them. It mirrors the "RQ1: Artifact Distillation" discussion in the paper and is mainly intended to orient a new reader before they pick the priors up.

Figures referenced here are in [`../figures/`](../figures); the underlying counts and Beta parameters are in [`../priors/`](../priors). You can regenerate the statistics yourself with

```bash
python example/analyze_human_attention.py --priors priors/combined --top 10
```

## 1. Eye-tracking data source

![](../figures/human_study.pdf)

`figures/human_study.pdf` gives an overview of the eye-tracking study whose data feeds every prior here. The study was run by [Zhang et al., 2024 (EyeTrans, FSE'24)](https://doi.org/10.1145/3660807) at the University of Notre Dame under the appropriate IRB protocols. Twenty-seven programmers read and wrote Java code while their gaze was recorded at 120 Hz. A dispersion-threshold (I-DT) algorithm extracted fixations of at least 100 ms, which were then spatially mapped to AST leaf tokens. The resulting 1,565 scan paths across reading and writing sessions are the source for every distilled artifact in this repository.

## 2. Method overview

![](../figures/eyemulator_overview.pdf)

`figures/eyemulator_overview.pdf` shows the high-level pipeline. Raw gaze is turned into (i) semantic salience priors, expressed as Beta parameters per token class, and (ii) n-gram transition models. The two are used to synthesize pseudo-scan paths over arbitrary code inputs, which drive the weighted SFT + token-level preference objective at training time. The reference implementation of each stage lives in [`../example/weighted_sft_template.py`](../example/weighted_sft_template.py).

## 3. Pseudo-scan path construction

![](../figures/eyemulator_pseudo_path.pdf)

`figures/eyemulator_pseudo_path.pdf` illustrates how a pseudo-scan path is assembled for a new code sequence. For an input of length `n`, the procedure

1. samples an attention density `ρ ~ Beta(α_agg, β_agg)` aggregated across all labels,
2. allocates a token budget `m = floor(ρ · n)` over semantic classes in proportion to their posterior means, and
3. greedily stitches masked tokens into valid trigram / bigram / monogram transitions subject to a line-span constraint.

## 4. Semantic salience priors (Beta parameters)

![](../figures/combined_beta_distributions.pdf)
![](../figures/combined_beta_curves.pdf)

For every semantic label `s` we fit `Beta(α_s, β_s)` with `α_s = c1(s) + 1` (fixations) and `β_s = n_tok(s) − c1(s) + 1` (misses). The posterior mean `E[θ_s] = α_s / (α_s + β_s)` is a smoothed salience estimate. `figures/combined_beta_distributions.pdf` plots `α_s` against `β_s` for each label; `figures/combined_beta_curves.pdf` shows the corresponding densities. Labels like `function declaration` peak sharply at high attention probability — developers look at them consistently — while `loop` and `conditional statement` have broader, sometimes bimodal densities, reflecting context-dependent reading.

The underlying data is [`../priors/combined/beta_distribution.json`](../priors/combined/beta_distribution.json). Equivalent files exist under `reading/` and `writing/` for the per-session views.

## 5. Semantic category distribution across tasks

![](../figures/category_distribution.pdf)

`figures/category_distribution.pdf` shows how the semantic category mix differs across the completion, translation, and summarization corpora. Completion is dominated by structural boilerplate; summarization skews toward API-contract tokens. This is part of why we ship separate reading-derived and writing-derived priors: you can pick the variant best matched to the task.

## 6. N-gram fixation patterns (Table 1 in the paper)

Expert attention is organized rather than linear. The table below reproduces the representative transitions reported in the paper; the full top-k list for any of the three prior folders is available from `analyze_human_attention.py`.

| Type     | Sequence                                                  | Count |
| :------- | :-------------------------------------------------------- | ----: |
| Mono     | variable declaration                                      | 18665 |
| Mono     | conditional statement                                     | 13222 |
| Bigram   | function declaration &rarr; variable declaration          |  8399 |
| Bigram   | conditional statement &rarr; loop                         |  6026 |
| Trigram  | function declaration &rarr; parameter &rarr; variable decl|  1634 |
| Trigram  | conditional statement &rarr; function decl &rarr; parameter|  1199 |

The underlying counts are in [`../priors/combined/monogram_counts.json`](../priors/combined/monogram_counts.json), [`bigram_counts.json`](../priors/combined/bigram_counts.json), and [`trigram_counts.json`](../priors/combined/trigram_counts.json). `monogram_counts.json` carries a few raw character entries alongside semantic-label entries; `analyze_human_attention.py` filters the monogram table against the Beta label vocabulary, so the reported counts correspond to semantic classes only.

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

The three prior folders share the same schema; see [`data_schema.md`](data_schema.md) for the field-level format.

## Attribution

The raw gaze data behind these priors was collected by Zhang et al. for EyeTrans (FSE'24). Please cite both that work and the EyeMulator paper when using any of these artifacts. BibTeX is in [`../CITATION.bib`](../CITATION.bib).
