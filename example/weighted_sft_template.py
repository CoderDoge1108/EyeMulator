"""Reference snippets: how to integrate the EyeMulator human-attention weights
into a supervised fine-tuning loop.

This file is a documented collection of the key building blocks --- a
weighted causal-LM loss, a weighted data collator, and a weighted-
preprocessing function --- designed to be copied into your own training
stack and adapted to your backbone of choice (Llama, StarCoder, DeepSeek-
Coder, or any larger model you wish to scale up to).

The snippets are deliberately framework-agnostic: the choice of batching,
learning-rate schedule, evaluation strategy, and distributed setup is
highly project-specific, so those decisions are left to the implementer. See
`docs/METHOD_INTEGRATION.md` for the conceptual walk-through and
`example/compute_token_weights.py` for a runnable demo that computes w_j
from the priors without any heavyweight dependencies.

Dependencies, if you want to import these classes into your own project:
    pip install torch transformers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Snippet 1: a weighted causal-LM loss.
#
# Subclass your backbone's `*ForCausalLM` and override `forward` so the
# cross-entropy is scaled element-wise by a `weights` tensor shaped like
# `labels`. Labels of -100 (padding / prompt-side tokens) are ignored as
# usual. For other backbones, swap the base class: e.g. GPT2LMHeadModel,
# StarCoder's GPTBigCodeForCausalLM, or DeepseekForCausalLM.
# ---------------------------------------------------------------------------

class CausalLMWithWeightedLoss(LlamaForCausalLM):
    def forward(
        self,
        input_ids:      torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.LongTensor] = None,
        weights:        Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_per_token = CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            active = shift_labels.view(-1) != -100

            if weights is not None:
                shift_weights = weights[..., 1:].contiguous().view(-1)
                w = shift_weights[active]
                loss = (loss_per_token[active] * w).sum() / (active.sum() + 1e-9)
            else:
                loss = loss_per_token[active].mean()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


# ---------------------------------------------------------------------------
# Snippet 2: a data collator that pads input_ids, labels, and weights
# consistently. Adjust MAX_LENGTH to your compute budget.
# ---------------------------------------------------------------------------

@dataclass
class WeightedCollator:
    tokenizer: AutoTokenizer
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        for key in features[0]:
            for f in features:
                if len(f[key]) > self.max_length:
                    f[key] = f[key][: self.max_length]
            if key == "weights":
                pad, dtype = 1.0, torch.float
            elif key == "labels":
                pad, dtype = -100, torch.long
            else:
                pad, dtype = self.tokenizer.pad_token_id, torch.long
            batch[key] = torch.tensor(
                [f[key] + [pad] * (self.max_length - len(f[key])) for f in features],
                dtype=dtype,
            )
        return batch


# ---------------------------------------------------------------------------
# Snippet 3: tokenize one raw example into (input_ids, labels, weights).
#
# `example` is assumed to already carry the per-token human-attention signals
# described in `docs/DATA_SCHEMA.md` (the fields `mask`, `ngram_indices`,
# `semantic_token_sequence`). The output is ready to be batched by
# `WeightedCollator` and consumed by `CausalLMWithWeightedLoss`.
#
# `priors` is the dict returned by `compute_token_weights.load_priors(...)`.
# `token_weight` is imported from the same module.
#
# `task_prompt`, `input_header`, and `output_header` are the three string
# fragments that bracket the input/output portions of the prompt; change them
# to match your own instruction format.
# ---------------------------------------------------------------------------

def build_training_example(
    example:       Dict[str, Any],
    tokenizer:     AutoTokenizer,
    priors:        Dict[str, Any],
    token_weight,  # callable: (mask_j, ngram_idx_j, sem_id_j, priors) -> float
    task_prompt:   str = "### Instruction:\nComplete the following code snippet.\n\n",
    input_header:  str = "### Input Code:\n",
    output_header: str = "\n\n### Output:\n",
    max_length:    int = 1024,
) -> Dict[str, List]:
    # Per-token weights on the input side (drive where the model should attend).
    w_code = [
        token_weight(
            example["mask"][j],
            example["ngram_indices"][j],
            example["semantic_token_sequence"][j],
            priors,
        )
        for j in range(len(example["code_tokens"]))
    ]

    prompt = f"{task_prompt}{input_header}{example['code']}{output_header}"
    text   = f"{prompt}{example['content']}{tokenizer.eos_token}"

    tok_full   = tokenizer(text,   max_length=max_length, truncation=True)
    tok_prompt = tokenizer(prompt, max_length=max_length, truncation=True)
    input_ids  = tok_full["input_ids"]
    labels     = list(input_ids)
    labels[: len(tok_prompt["input_ids"])] = [-100] * len(tok_prompt["input_ids"])

    # Default weight schedule: prompt tokens get w=1.0; output tokens inherit
    # the mean input-side weight. Consider replacing this with a per-output-
    # token alignment scheme if your task warrants it.
    mean_w   = sum(w_code) / len(w_code) if w_code else 1.0
    n_prompt = len(tok_prompt["input_ids"])
    n_out    = len(input_ids) - n_prompt
    weights  = [1.0] * n_prompt + [float(mean_w)] * n_out

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
        "weights":        weights,
    }


# ---------------------------------------------------------------------------
# Typical wiring sketch (pseudocode, NOT executed by this file):
#
#     from compute_token_weights import load_priors, token_weight
#     priors = load_priors("priors/combined")
#
#     tokenizer = AutoTokenizer.from_pretrained(MY_BACKBONE)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     model = CausalLMWithWeightedLoss.from_pretrained(MY_BACKBONE)
#
#     features = [build_training_example(ex, tokenizer, priors, token_weight)
#                 for ex in my_dataset]           # or use Dataset.map(...)
#
#     trainer = Trainer(
#         model=model,
#         args=TrainingArguments(...),            # your own hyperparameters
#         train_dataset=features,
#         data_collator=WeightedCollator(tokenizer),
#     )
#     trainer.train()
# ---------------------------------------------------------------------------
