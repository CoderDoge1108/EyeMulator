# Integrating the artifacts into your own training loop

This document describes, at a conceptual level, how to use the artifacts in this release to produce per-token weights `w_j` for a supervised fine-tuning (SFT) loss that emphasizes tokens humans tend to fixate on. The scripts under `example/` are reference snippets that realize what is described here; they are meant to be copied into your own codebase, pointed at your own dataset, and adapted to your own backbone.

This document does not reproduce our paper's training runs. Implementation details — batching, distributed training, mixed precision, evaluation, schedule, checkpoint selection — are standard and are intentionally left to the user. The artifact is a data release, and the snippets are a method sketch.

## Overview

At a high level, integration has three steps:

1. **Load the priors** once at start-up.
2. **For each training example**, compute a per-token weight vector `w` using the fields already present on the example (`mask`, `ngram_indices`, `semantic_token_sequence`) together with the priors.
3. **During the loss** computation, multiply the per-token cross-entropy loss by `w` before averaging.

## Step 1 — Load the priors

The priors are tiny (~300 KB total) and can be kept in memory.

```python
import json, math, os

def load_priors(prior_dir):
    # indexed_ngrams.json lives only in combined/ and is shared across conditions.
    idx_path = os.path.join(prior_dir, "indexed_ngrams.json")
    if not os.path.exists(idx_path):
        idx_path = os.path.join(os.path.dirname(prior_dir), "combined", "indexed_ngrams.json")
    with open(idx_path) as f:
        ngram_to_index = json.load(f)
    index_to_ngram = {int(v): k for k, v in ngram_to_index.items()}
    semantic_id_to_label = {
        int(v): k for k, v in ngram_to_index.items() if not k.startswith("(")
    }

    ngram_counts = {}
    for n in ("monogram", "bigram", "trigram"):
        with open(os.path.join(prior_dir, f"{n}_counts.json")) as f:
            ngram_counts.update(json.load(f))

    with open(os.path.join(prior_dir, "beta_distribution.json")) as f:
        beta = json.load(f)
    semantic_label_to_mean_attn = {
        item["semantic_label"].strip(): item["alpha"] / (item["alpha"] + item["beta"])
        for item in beta
        if (item["alpha"] + item["beta"]) > 0
    }
    return {
        "index_to_ngram":               index_to_ngram,
        "semantic_id_to_label":         semantic_id_to_label,
        "ngram_counts":                 ngram_counts,
        "semantic_label_to_mean_attn":  semantic_label_to_mean_attn,
    }
```

`prior_dir` can be `priors/combined`, `priors/reading`, or `priors/writing`. All three have the same filenames; only the folder name changes.

## Step 2 — Compute the per-token weight w_j

For each token position `j`, the weight is

```
w_j = w_base + rarity_bonus(ngram_j) + semantic_attn(semantic_label_j)       if mask_j == 1
w_j = 1                                                                       if mask_j == 0
```

where

- `w_base` is a scalar hyperparameter (we used `w_base = 3.0`).
- `rarity_bonus(g) = 1 / log(count(g) + 2)` where `count(g)` is taken from the monogram / bigram / trigram counts.
- `semantic_attn(s) = E[θ_s] = α_s / (α_s + β_s)` from the Beta distribution for label `s`.

Tokens outside the human-attention mask (i.e. `mask_j == 0`) and tokens belonging to the *prompt* portion of the sequence (instruction, input header, input code) receive weight `1.0` by default — only the **output** portion (what you are actually training the model to generate) is weighted.

See `example/compute_token_weights.py` for a runnable, commented implementation of this step.

## Step 3 — Plug w into the loss

The training objective is a standard causal-LM cross-entropy loss, multiplied element-wise by `w` and normalized by the number of active (non-masked) tokens:

```
L_SFT(φ) = (1 / |active|) * Σ_{j ∈ active}  w_j · ( − log P_φ(x_j | x_{<j}) )
```

where `active` is the set of output-side, non-padded positions (i.e. positions with `label ≠ -100`). The snippets in `example/weighted_sft_template.py` show this as a subclass of Hugging Face's `LlamaForCausalLM` (`CausalLMWithWeightedLoss`), together with a data collator (`WeightedCollator`) and a preprocessing helper (`build_training_example`). The pattern transfers cleanly to any causal LM — swap the base class to `GPT2LMHeadModel`, `GPTBigCodeForCausalLM`, or whatever your backbone requires.

A few implementation notes:

- Shift weights identically to how you shift labels for next-token prediction (`weights[..., 1:]`).
- Apply the active-token mask to weights before aggregating.
- If you want a strict average across tokens (rather than a weight-scaled sum), divide by `Σ w_j` instead of `|active|`; in our experiments the difference was small.
- The scheme above is the per-token weighting described in the paper. Simpler aggregations (e.g. a single scalar weight per example) are easier to implement but discard most of the signal — they are **not** recommended.

## Choosing a condition (combined / reading / writing)

As a default, use `priors/combined/` — this is the variant that performed best overall in our experiments. Use `priors/reading/` if your target task is comprehension-heavy (e.g. code summarization, code search) and `priors/writing/` if your target task is generation-heavy (e.g. code completion, code generation from scratch).

## A note on scope

The formula above is a clean, per-token realization of the weighting scheme described in Section 2.4 of the paper, and the snippets in `example/weighted_sft_template.py` realize exactly this formulation. The artifact is intended as a starting point for others to apply the EyeMulator method to their own models and datasets; it is not a reproduction of our paper's specific training runs.
