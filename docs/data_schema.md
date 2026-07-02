# Data schema

Every file in this release is described below.

## 1. Priors (`priors/{combined,reading,writing}/`)

The priors summarize human visual attention as it was recorded in the EyeTrans eye-tracking study. Three condition-specific variants are shipped:

- `combined/`  — all sessions (reading + writing).
- `reading/`   — sessions in which participants read code without editing.
- `writing/`   — sessions in which participants wrote or modified code.

Each folder contains the same filenames; the condition is encoded in the folder name, not the filenames.

### 1.1 `indexed_ngrams.json`

A flat `{str → int}` map that assigns a unique integer index to every semantic-label n-gram observed in the data. Unigrams are plain strings (e.g. `"variable"`, `"conditional"`); bigrams and trigrams are stringified tuples (e.g. `"('argument', 'function call')"`).

The `ngram_indices` field in the dataset (see §2) indexes into this map.

Shipped only in `combined/`; the same index space is reused across conditions.

### 1.2 `beta_distribution.json`

A JSON list. Each element is one semantic label with the Beta distribution fitted to its fixation probability:

```json
{
  "semantic_label": "variable declaration",
  "saccades_count": 4773,
  "word_count":     10067,
  "alpha":          4774,
  "beta":           5295
}
```

The two quantities you will typically use:

- Posterior mean `E[θ_s] = α / (α + β)`, a smoothed estimate of fixation probability for label `s`.
- The full `Beta(α, β)` distribution, if you want to sample instead of using the mean.

### 1.3 `monogram_counts.json`

`{semantic_label → int}`. Raw counts of how often each semantic label was fixated. Used to derive the rarity bonus and to sanity-check prior coverage.

### 1.4 `bigram_counts.json`

`{"('label_a', 'label_b')" → int}`. Counts of semantic-label bigrams in the observed scan paths.

### 1.5 `trigram_counts.json`

`{"('label_a', 'label_b', 'label_c')" → int}`. Counts of semantic-label trigrams in the observed scan paths.

Together, the monogram / bigram / trigram counts define the rarity signal: rarer n-grams receive a larger contribution in the per-token weight `w_j` (see [`method_integration.md`](method_integration.md)).

## 2. Dataset sample (`dataset_sample/*.jsonl`)

Each `.jsonl` file is one example per line. Files are named `<task>_<split>_sample.jsonl`, where

- `task ∈ {completion, summarization, translation}`
- `split ∈ {train, valid, test}`

The sample contains 30 examples per split per task and uses exactly the same schema as a full dataset. It is meant to exercise the pipeline end to end on a laptop. Applying the weight computation in `method_integration.md` to your own tokenized corpus (together with the shipped priors) produces the full-scale equivalent.

Each line is a JSON object with the fields below.

| Field                      | Type         | Description |
|----------------------------|--------------|-------------|
| `code_content_hash`        | `str`        | 16-character hash, for deduplication. |
| `flag`                     | `str`        | One of `train`, `valid`, `test`. |
| `code`                     | `str`        | Raw input Java source code. |
| `content`                  | `str`        | Ground-truth output (C# for translation, natural-language summary for summarization, next-token span for completion). |
| `code_tokens`              | `List[str]`  | Length-`N` token sequence for `code`. |
| `line_numbers`             | `List[int]`  | Length-`N`; line number of each token. |
| `semantic_token_sequence`  | `List[int]`  | Length-`N`; integer semantic-label id (see `indexed_ngrams.json`). |
| `mask`                     | `List[int]`  | Length-`N`; 0/1, whether the token carries a distilled human-attention signal. |
| `ngram_indices`            | `List[int]`  | Length-`N`; index into `indexed_ngrams.json`. `0` means "no ngram at this position". |
| `order_sequence`           | `List[int]`  | Length-`N`; position of the token within its n-gram window. |
| `ngrams`                   | `List[str]`  | Length-`N`; human-readable semantic label at this position (can be `null` where `mask[j] == 0`). |
| `code_occurrence`          | `List[int]`  | Length-`N`; occurrence index for repeated tokens. |

Together, `mask`, `ngram_indices`, and `semantic_token_sequence` form the per-token human-attention signal. Combined with the priors, they are sufficient to compute `w_j` (see [`method_integration.md`](method_integration.md) and [`../example/compute_token_weights.py`](../example/compute_token_weights.py)).
