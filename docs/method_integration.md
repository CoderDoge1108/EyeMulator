# Integrating the priors into a training loop

This note describes how to use the shipped priors to produce per-token weights `w_j` for a supervised fine-tuning loss that emphasizes tokens humans tend to fixate on, and how to add the token-level preference term on top. The scripts under `example/` are reference snippets of what is described here; copy them into your own codebase and adapt them to your backbone.

The guide is backbone- and framework-agnostic. Batching, distributed training, mixed precision, scheduling, evaluation, and checkpoint selection are left to the implementer.

## Overview

Integration has three steps:

1. Load the priors once at start-up.
2. For each training example, compute a per-token weight vector `w` from the fields already on the example (`mask`, `ngram_indices`, `semantic_token_sequence`) and the priors.
3. During loss computation, multiply the per-token cross-entropy by `w` before averaging.

## Step 1 — Load the priors

The priors are tiny (about 300 KB total) and fit in memory.

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
        "index_to_ngram":              index_to_ngram,
        "semantic_id_to_label":        semantic_id_to_label,
        "ngram_counts":                ngram_counts,
        "semantic_label_to_mean_attn": semantic_label_to_mean_attn,
    }
```

`prior_dir` can be `priors/combined`, `priors/reading`, or `priors/writing`. All three have the same filenames; only the folder name changes.

## Step 2 — Per-token weight `w_j`

For each token position `j`,

```
w_j = w_base + rarity_bonus(ngram_j) + semantic_attn(semantic_label_j)    if mask_j == 1
w_j = 1                                                                    if mask_j == 0
```

where

- `w_base` is a scalar hyperparameter (we used `3.0`).
- `rarity_bonus(g) = 1 / log(count(g) + 2)`, with `count(g)` from the monogram / bigram / trigram counts.
- `semantic_attn(s) = E[θ_s] = α_s / (α_s + β_s)` from the Beta distribution for label `s`.

Tokens outside the human-attention mask (`mask_j == 0`), and tokens belonging to the prompt portion of the sequence (instruction, input header, input code), receive weight `1.0`. Only the output portion — what the model is being trained to generate — is weighted.

See [`../example/compute_token_weights.py`](../example/compute_token_weights.py) for a runnable implementation.

## Step 3 — Composite objective

The training objective is a composite of a weighted SFT term and a token-level preference term, controlled by a scalar `γ`:

```
L_total(φ) = L_SFT(φ) + γ · L_pref(φ)
```

### 3a. Weighted SFT

A standard causal-LM cross-entropy, scaled element-wise by `w` and normalized by the number of active positions on the pseudo-scan path:

```
L_SFT(φ) = −(1 / |P̃|) · Σ_{j ∈ P̃}  w_j · log P_φ(x_j | x_{<j})
```

`P̃` is the pseudo-scan path for the example; active positions satisfy `label_j ≠ -100`. In [`../example/weighted_sft_template.py`](../example/weighted_sft_template.py) this is realized by `CausalLMWithWeightedLoss`, a subclass of `LlamaForCausalLM` whose `forward` consumes an extra `weights` tensor. The same pattern transfers to any causal LM by swapping the base class (`GPT2LMHeadModel`, `GPTBigCodeForCausalLM`, etc.).

### 3b. Token-level preference term

The preference term adapts DPO (Rafailov et al., 2023) to the token level: the tokens on the pseudo-scan path `P̃` form the preferred set, and the complement `x \ P̃` forms the dispreferred set. With `π_φ` the trainable policy and `π_ref` a frozen copy of the initial policy, define the per-token log-ratio

```
r_j = log π_φ(x_j | x_{<j})  −  log π_ref(x_j | x_{<j})
```

and aggregate over preferred and dispreferred positions:

```
L_pref(φ) = − log σ( β · ( mean_{j ∈ P̃} r_j  −  mean_{j ∈ x\P̃} r_j ) )
```

where `β` controls the sigmoid-margin strength. `weighted_sft_template.py` provides `token_level_preference_loss(...)` as a minimal reference implementation. IPO, KTO, SimPO, or newer token-level DPO variants can be swapped in without touching the rest of the pipeline.

### 3c. Composite wrapper

`EyeMulatorCompositeObjective` combines `CausalLMWithWeightedLoss` and `token_level_preference_loss` behind a single callable. Setting `reference=None` or `gamma=0` collapses it to pure weighted SFT, which is a useful diagnostic configuration when you first bring up the training loop on a new backbone.

## Dynamic vs. precomputed scan paths

`build_training_example` accepts a `dynamic_path` flag. With `dynamic_path=True`, a fresh pseudo-scan path is sampled per call from the priors (Algorithm 1 as written in the paper); with `dynamic_path=False`, the precomputed `mask` field in each JSONL example is reused. The precomputed path is faster for small-scale runs; dynamic regeneration can reduce dependence on a single sampled path when evaluating new hyperparameters or scaling the dataset up.

A few implementation notes:

- Shift the weights the same way you shift labels for next-token prediction (`weights[..., 1:]`).
- Apply the active-token mask to the weights before aggregating.
- For a strict average across tokens rather than a weight-scaled sum, divide by `Σ w_j` instead of `|active|`. The difference was small in our experiments.
- The scheme above is the per-token weighting from the paper. A single scalar weight per example is easier to implement but discards most of the signal and is not recommended.

## Choosing a condition (combined / reading / writing)

For the paper's default configuration, use `priors/combined/`. Use `priors/reading/` for comprehension-heavy tasks (code summarization, code search) and `priors/writing/` for generation-heavy tasks (completion, open-ended code generation).

## Scope

This note realizes the weighting scheme in Section 2.4 of the paper at the per-token granularity. The artifact is a starting point for applying the method at larger scales — bigger backbones, larger datasets, or new tasks — on top of the human-attention signals that are expensive to collect from scratch.
