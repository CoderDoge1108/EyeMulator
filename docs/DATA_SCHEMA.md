# Data schema

This document describes every file in this release.

## 1. Priors (`priors/{combined,reading,writing}/`)

The priors are the distilled, task-agnostic summary of human visual attention. Three condition-specific variants are provided:

- `combined/`  — priors distilled from all eye-tracking sessions (reading + writing).
- `reading/`   — priors distilled from sessions in which participants were asked to read code without editing it.
- `writing/`   — priors distilled from sessions in which participants were asked to write or modify code.

Each folder contains the same filenames (the condition is encoded in the folder, not the filenames):

### 1.1 `indexed_ngrams.json`

A flat `{str → int}` map assigning a unique integer index to every semantic-label n-gram observed in the eye-tracking data. Unigrams are plain strings (e.g. `"variable"`, `"conditional"`); bigrams and trigrams are stringified tuples (e.g. `"('argument', 'function call')"`).

The integer values in the dataset field `ngram_indices` (see §2) look up into this map.

Only provided in the `combined/` folder (the same index space is reused across conditions).

### 1.2 `beta_distribution.json`

A JSON list; each element is one semantic label and records the Beta distribution fitted to its fixation probability:

```json
{
  "semantic_label": "variable declaration",
  "saccades_count": 4773,
  "word_count":     10067,
  "alpha":          4774,
  "beta":           5295
}
```

Derived quantities you will use:

- `E[θ_s] = α / (α + β)` — expected fixation probability for semantic label `s`.
- `Beta(α, β)` — full distribution, if you want to sample instead of using the mean.

### 1.3 `monogram_counts.json`

`{semantic_label → int}`. Raw counts of how often each semantic label was fixated. Used to derive rarity bonuses and to sanity-check prior coverage.

### 1.4 `bigram_counts.json`

`{"('label_a', 'label_b')" → int}`. Counts of semantic-label bigrams observed in scan paths.

### 1.5 `trigram_counts.json`

`{"('label_a', 'label_b', 'label_c')" → int}`. Counts of semantic-label trigrams observed in scan paths.

The monogram / bigram / trigram counts jointly define a rarity signal: rarer n-grams get a larger boost in the per-token weight `w_j` (see `METHOD_INTEGRATION.md`).

## 2. Dataset sample (`dataset_sample/*.jsonl`)

Each `.jsonl` file is a **small demonstration sample** — one example per line. Files are named `<task>_<split>_sample.jsonl` for:

- `task ∈ {completion, summarization, translation}`
- `split ∈ {train, valid, test}`

These are not the full training / validation / test splits used in the paper. They contain enough examples to illustrate the schema and to exercise the demo script (`example/compute_token_weights.py`). To apply the EyeMulator method at full scale, users should supply their own data in the same schema — the per-token human-attention signals (`mask`, `ngram_indices`, `semantic_token_sequence`) can be computed from the priors using the process described in `METHOD_INTEGRATION.md`.

Each line is a JSON object with the following fields:

| Field                      | Type         | Description                                                                                                   |
|----------------------------|--------------|---------------------------------------------------------------------------------------------------------------|
| `code_content_hash`        | `str`        | 16-character hash, for deduplication.                                                                         |
| `flag`                     | `str`        | One of `train`, `valid`, `test`.                                                                              |
| `code`                     | `str`        | The raw input Java source code.                                                                               |
| `content`                  | `str`        | Ground-truth output (C# code for translation; natural-language summary for summarization; next-token span for completion). |
| `code_tokens`              | `List[str]`  | Length-`N` token sequence for `code`.                                                                         |
| `line_numbers`             | `List[int]`  | Length-`N`; source-file line number of each token.                                                            |
| `semantic_token_sequence`  | `List[int]`  | Length-`N`; integer semantic-label id of each token (see `indexed_ngrams.json`).                              |
| `mask`                     | `List[int]`  | Length-`N`; 0/1 — whether the token carries a distilled human-attention signal.                               |
| `ngram_indices`            | `List[int]`  | Length-`N`; integer index into `indexed_ngrams.json`. `0` means "no ngram at this position."                  |
| `order_sequence`           | `List[int]`  | Length-`N`; position of the token within its ngram window.                                                    |
| `ngrams`                   | `List[str]`  | Length-`N`; human-readable semantic label for this position (can be `null` where `mask[j] == 0`).             |
| `code_occurrence`          | `List[int]`  | Length-`N`; occurrence index for repeated tokens.                                                             |

The fields `mask`, `ngram_indices`, and `semantic_token_sequence` are the **per-token human-attention signals** — together with the priors, they are sufficient to compute the token-level weight `w_j` used by EyeMulator (see `docs/METHOD_INTEGRATION.md` and `example/compute_token_weights.py`).
