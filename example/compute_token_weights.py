"""Demo: load the EyeMulator priors and compute the per-token weight vector w_j
for examples in a tokenized dataset file.

Usage:
    python example/compute_token_weights.py \\
        --priors priors/combined \\
        --jsonl  dataset/completion_train_final.jsonl \\
        --limit  2

This script is intentionally small and dependency-free (standard library only).
It is meant to document the weight formula described in docs/METHOD_INTEGRATION.md
and to let users sanity-check the artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List


W_BASE = 3.0


def load_priors(prior_dir: str) -> Dict[str, Any]:
    """Load indexed n-grams, n-gram counts, and the Beta distributions for each
    semantic label from a priors subfolder (combined/, reading/, or writing/)."""
    # indexed_ngrams.json lives only in combined/ and is shared across conditions.
    indexed = os.path.join(prior_dir, "indexed_ngrams.json")
    if not os.path.exists(indexed):
        indexed = os.path.join(os.path.dirname(prior_dir), "combined", "indexed_ngrams.json")
    with open(indexed) as f:
        ngram_to_index = json.load(f)
    index_to_ngram = {int(v): k for k, v in ngram_to_index.items()}
    semantic_id_to_label = {
        int(v): k for k, v in ngram_to_index.items() if not k.startswith("(")
    }

    ngram_counts: Dict[str, int] = {}
    for n in ("monogram", "bigram", "trigram"):
        with open(os.path.join(prior_dir, f"{n}_counts.json")) as f:
            ngram_counts.update(json.load(f))

    with open(os.path.join(prior_dir, "beta_distribution.json")) as f:
        beta = json.load(f)
    label_to_mean_attn: Dict[str, float] = {}
    for item in beta:
        a = item["alpha"]
        b = item["beta"]
        if a + b > 0:
            label_to_mean_attn[item["semantic_label"].strip()] = a / (a + b)

    return {
        "index_to_ngram":       index_to_ngram,
        "semantic_id_to_label": semantic_id_to_label,
        "ngram_counts":         ngram_counts,
        "label_to_mean_attn":   label_to_mean_attn,
        "_raw_beta":            beta,
        "_prior_dir":           prior_dir,
    }


def token_weight(mask_j: int, ngram_idx_j: int, semantic_id_j: int, priors: Dict[str, Any]) -> float:
    """The per-token weight w_j as defined in docs/METHOD_INTEGRATION.md."""
    if mask_j == 0:
        return 1.0

    rarity_bonus = 0.0
    ngram = priors["index_to_ngram"].get(int(ngram_idx_j))
    if ngram is not None:
        count = priors["ngram_counts"].get(ngram, 0)
        if count > 0:
            rarity_bonus = 1.0 / math.log(count + 2)

    semantic_bonus = 0.0
    label = priors["semantic_id_to_label"].get(int(semantic_id_j))
    if label is not None:
        semantic_bonus = priors["label_to_mean_attn"].get(label, 0.0)

    return W_BASE + rarity_bonus + semantic_bonus


def weights_for_example(example: Dict[str, Any], priors: Dict[str, Any]) -> List[float]:
    n = len(example["code_tokens"])
    return [
        token_weight(
            example["mask"][j],
            example["ngram_indices"][j],
            example["semantic_token_sequence"][j],
            priors,
        )
        for j in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", required=True, help="Path to priors/combined, priors/reading, or priors/writing.")
    parser.add_argument("--jsonl",  required=True, help="Path to a dataset/*.jsonl file.")
    parser.add_argument("--limit",  type=int, default=1, help="Number of examples to display.")
    args = parser.parse_args()

    priors = load_priors(args.priors)

    with open(args.jsonl) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            ex = json.loads(line)
            w = weights_for_example(ex, priors)

            print(f"=== Example {i} ({ex.get('flag', '?')})  {len(w)} tokens ===")
            for j in range(min(20, len(w))):
                tok = ex["code_tokens"][j]
                lbl_id = ex["semantic_token_sequence"][j]
                lbl = priors["semantic_id_to_label"].get(int(lbl_id), "-")
                print(f"  j={j:3d}  tok={tok[:20]:<20s}  mask={ex['mask'][j]}  label={lbl[:25]:<25s}  w={w[j]:.3f}")
            if len(w) > 20:
                print(f"  ... {len(w) - 20} more tokens omitted ...")
            print()


if __name__ == "__main__":
    main()
