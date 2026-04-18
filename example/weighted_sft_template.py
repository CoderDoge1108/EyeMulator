"""Reference implementation of the EyeMulator method components.

Mirrors Section 2 and Algorithm 1 of the paper at the granularity of
reusable building blocks. The file is PyTorch-level (not tied to
`transformers.Trainer`), backbone-agnostic (no model id hard-coded), and
leaves every hyperparameter exposed as a named argument, so it can be
copy-pasted into a larger-scale training loop without inheriting the
experimental defaults from our small-model runs.

Components, in the order Algorithm 1 uses them:

    1. sample_attention_density       ρ ~ Beta(α_agg, β_agg)
    2. generate_pseudo_scan_path      P̃ from priors + ρ (Algorithm 1, line 9)
    3. token_weight                   w_j formula (line 12; re-exported from
                                      compute_token_weights.py)
    4. CausalLMWithWeightedLoss       L_SFT = −Σ_{j∈P̃} w_j log P_ϕ(x_j | x_<j)
    5. token_level_preference_loss    token-level preference term from §2.4
    6. EyeMulatorCompositeObjective   L = L_SFT + γ · L_pref
    7. WeightedCollator               batching with per-token weights + preferred masks
    8. build_training_example         preprocessing: raw example -> tensors

A short pseudocode wiring sketch at the bottom shows how the pieces fit
together in a training loop. The file does not call any trainer entry-point
and does not ship a `main()`: batch size, schedule, mixed precision,
sharding, and evaluation are backbone- and infra-specific.

Dependencies:
    pip install torch transformers
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, os.path.dirname(__file__))
from compute_token_weights import load_priors, token_weight  # noqa: E402


# ============================================================================
# 1. Attention-density sampling -- Algorithm 1, line 8
# ============================================================================
#
# Each pseudo-scan path is generated at a sampled "attention density" rho that
# controls what fraction of code tokens the hypothetical reader fixates on.
# The paper models this density with a Beta(alpha_agg, beta_agg) distribution
# whose parameters are aggregated from the per-semantic-label Betas in
# priors/<condition>/beta_distribution.json.
# ----------------------------------------------------------------------------

def aggregate_beta_params(priors: Dict[str, Any]) -> Tuple[float, float]:
    """Collapse per-label Beta(alpha_s, beta_s) parameters into a single
    corpus-level Beta(alpha_agg, beta_agg) by summing counts. Swap this
    out for a frequency-weighted or hierarchical-Bayesian aggregation if
    you want something richer."""
    raw = priors.get("_raw_beta")
    if not raw:
        return 1.0, 1.0  # uninformative fallback
    alpha_agg = float(sum(item["alpha"] for item in raw))
    beta_agg  = float(sum(item["beta"]  for item in raw))
    return alpha_agg, beta_agg


def sample_attention_density(alpha_agg: float, beta_agg: float,
                             generator: Optional[torch.Generator] = None) -> float:
    """Return one sample ρ ~ Beta(alpha_agg, beta_agg), interpreted as the
    expected fraction of tokens to include in the pseudo-scan path."""
    # torch has no direct Beta; sample via two Gammas.
    a = torch._standard_gamma(torch.tensor([alpha_agg]), generator=generator)
    b = torch._standard_gamma(torch.tensor([beta_agg]),  generator=generator)
    return float(a / (a + b))


# ============================================================================
# 2. Pseudo-scan-path generation -- Algorithm 1, line 9
# ============================================================================
#
# Given an example's token-level semantic labels and the condition priors,
# emit a 0/1 mask over tokens indicating which tokens belong to the
# synthetic scan path. This replaces the per-example `mask` field when you
# want to regenerate scan paths dynamically at training time (recommended
# when you scale the dataset up) rather than reuse the precomputed one.
# ----------------------------------------------------------------------------

def generate_pseudo_scan_path(
    semantic_token_sequence: Sequence[int],
    priors:                  Dict[str, Any],
    rho:                     float,
    transition_probs:        Optional[Dict[Tuple[str, str], float]] = None,
    generator:               Optional[torch.Generator] = None,
) -> List[int]:
    """Produce a binary mask `path` with `path[j] == 1` iff token j is on
    the pseudo-scan path. Two signals are combined:

      (a) a per-token salience prior `E[theta_{s_j}]` from the Beta posterior
          for token j's semantic label, and
      (b) an optional transition probability P_trans that biases consecutive
          path memberships toward observed reading transitions.

    The `rho` parameter (sampled once per example) rescales the whole path so
    that its expected density matches a realistic reader's attention budget.

    This is one realization of Algorithm 1's GeneratePath. Bernoulli-HMM
    path sampling, beam search over transitions, or purely prior-driven
    thresholding are all reasonable alternatives; the version below keeps
    the arithmetic transparent.
    """
    sem_to_label = priors["semantic_id_to_label"]
    label_mean   = priors["label_to_mean_attn"]
    n = len(semantic_token_sequence)
    path: List[int] = [0] * n

    prev_label: Optional[str] = None
    for j, sem_id in enumerate(semantic_token_sequence):
        label = sem_to_label.get(int(sem_id))
        if label is None:
            continue
        p = label_mean.get(label, 0.0) * rho

        if transition_probs is not None and prev_label is not None and path[j - 1] == 1:
            p = 0.5 * (p + transition_probs.get((prev_label, label), 0.0))

        u = torch.rand(1, generator=generator).item()
        path[j]    = 1 if u < p else 0
        prev_label = label
    return path


# ============================================================================
# 3. Per-token weight -- Algorithm 1, line 12
# ============================================================================
#
# `token_weight` is re-exported from compute_token_weights.py so that the
# demo script and this reference share a single implementation of the
# formula w_j = w_base + 1/log(freq(g_j)+2) + E[theta_{s_j}].
# ----------------------------------------------------------------------------

__all_exports_from_compute__ = [load_priors, token_weight]


# ============================================================================
# 4. Weighted SFT loss -- Algorithm 1, line 15
# ============================================================================
#
# L_SFT(phi) = -(1/|P̃|) * sum_{j in P̃}  w_j  log P_phi(x_j | x_<j)
#
# Subclass your backbone's *ForCausalLM and override `forward` so the cross-
# entropy is scaled element-wise by a `weights` tensor shaped like `labels`.
# Tokens with label == -100 are ignored as usual. The example below uses
# LlamaForCausalLM for illustration; for other backbones, swap the base
# class (GPT2LMHeadModel, GPTBigCodeForCausalLM, DeepseekForCausalLM, ...).
# ----------------------------------------------------------------------------

from transformers import LlamaForCausalLM  # noqa: E402

class CausalLMWithWeightedLoss(LlamaForCausalLM):
    """Causal LM whose loss is the weighted, mask-aware cross-entropy."""

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
            per_tok = CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            active = shift_labels.view(-1) != -100

            if weights is not None:
                w = weights[..., 1:].contiguous().view(-1)[active]
                loss = (per_tok[active] * w).sum() / (active.sum() + 1e-9)
            else:
                loss = per_tok[active].mean()

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
        )


# ============================================================================
# 5. Token-level preference loss -- Algorithm 1, line 16 (reference version)
# ============================================================================
#
# L_DPO in the paper treats the pseudo-scan-path tokens as the "winning"
# trajectory and the complement as the "losing" trajectory, adapting DPO
# (Rafailov et al., 2023) to the token level. The implementation below is
# a minimal margin-based version that:
#
#   * treats each output position as an independent preference comparison,
#   * uses a frozen copy of the initial policy as the reference pi_ref,
#   * computes per-token log-ratios r_j = log pi_phi(x_j|x_<j) - log pi_ref(x_j|x_<j),
#   * aggregates preferred (P̃) vs dispreferred (complement) r_j into a
#     sigmoid-margin loss of strength beta, masked to output-side tokens.
#
# It keeps the shape of the paper's objective while staying dependency-free
# (no trl) and easy to replace with IPO, KTO, SimPO, or a newer token-level
# DPO variant.
# ----------------------------------------------------------------------------

def _log_probs_of_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather log P(label_j | context) for each position; shape (B, T-1)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    logp = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp_min(0)  # replace -100 with 0 for the gather
    gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered.masked_fill(shift_labels == -100, 0.0)
    return gathered


def token_level_preference_loss(
    policy_logits:    torch.Tensor,      # (B, T, V)
    reference_logits: torch.Tensor,      # (B, T, V) from a frozen copy of the initial policy
    labels:           torch.Tensor,      # (B, T)   with -100 on prompt/pad positions
    preferred_mask:   torch.Tensor,      # (B, T)   1 where token is on pseudo-scan path, else 0
    beta:             float = 0.1,       # KL-regularization strength (tunable; paper used gamma*beta-like scaling)
    eps:              float = 1e-9,
) -> torch.Tensor:
    """Reference token-level preference loss. Returns a scalar.

    For every output-side position j with label_j != -100, we compute the
    per-token log-ratio r_j = log pi_phi(label_j | x_<j) - log pi_ref(label_j | x_<j).
    We then aggregate r_j separately over the preferred mask (j in P̃) and
    the dispreferred mask (j in output \\ P̃) and push the preferred mean
    to be larger than the dispreferred mean by a sigmoid margin of strength
    beta.  The resulting loss has the familiar DPO shape
        L = -log sigma( beta * ( mean_pref(r) - mean_disp(r) ) ).
    """
    log_pi   = _log_probs_of_labels(policy_logits,    labels)
    log_ref  = _log_probs_of_labels(reference_logits, labels)
    log_ratio = log_pi - log_ref                              # (B, T-1)

    mask     = (labels[..., 1:] != -100).float()
    pref     = preferred_mask[..., 1:].float() * mask
    disp     = (1.0 - preferred_mask[..., 1:].float()) * mask

    mean_pref = (log_ratio * pref).sum() / (pref.sum() + eps)
    mean_disp = (log_ratio * disp).sum() / (disp.sum() + eps)
    return -F.logsigmoid(beta * (mean_pref - mean_disp))


# ============================================================================
# 6. Composite objective -- Algorithm 1, line 17
# ============================================================================
#
# L_total = L_SFT + gamma * L_pref
# ----------------------------------------------------------------------------

@dataclass
class EyeMulatorCompositeObjective:
    """Convenience wrapper. `policy` is the trainable model (an instance of
    CausalLMWithWeightedLoss or a subclass); `reference` is a frozen copy of
    the *initial* policy used only for the preference term. Passing
    `reference=None` collapses this to pure weighted SFT."""

    policy:    nn.Module
    reference: Optional[nn.Module] = None
    gamma:     float = 0.1          # paper used a small weight on the preference term
    beta:      float = 0.1          # sigmoid-margin strength inside the preference loss

    def __call__(
        self,
        input_ids:       torch.LongTensor,
        attention_mask:  torch.Tensor,
        labels:          torch.LongTensor,
        weights:         torch.FloatTensor,
        preferred_mask:  torch.LongTensor,
    ) -> torch.Tensor:
        out = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            weights=weights,
        )
        loss = out.loss

        if self.reference is not None and self.gamma > 0:
            with torch.no_grad():
                ref_logits = self.reference(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            loss = loss + self.gamma * token_level_preference_loss(
                policy_logits    = out.logits,
                reference_logits = ref_logits,
                labels           = labels,
                preferred_mask   = preferred_mask,
                beta             = self.beta,
            )
        return loss


# ============================================================================
# 7. Data collator
# ============================================================================

@dataclass
class WeightedCollator:
    tokenizer:  PreTrainedTokenizerBase
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        keys = list(features[0].keys())
        for key in keys:
            for f in features:
                if len(f[key]) > self.max_length:
                    f[key] = f[key][: self.max_length]
            if key == "weights":
                pad, dtype = 1.0, torch.float
            elif key in ("labels",):
                pad, dtype = -100, torch.long
            elif key in ("preferred_mask",):
                pad, dtype = 0, torch.long
            else:
                pad, dtype = self.tokenizer.pad_token_id, torch.long
            batch[key] = torch.tensor(
                [f[key] + [pad] * (self.max_length - len(f[key])) for f in features],
                dtype=dtype,
            )
        return batch


# ============================================================================
# 8. Preprocessing: raw example -> training-ready tensors
# ============================================================================
#
# Take one raw JSONL example (with the per-token human-attention signals
# documented in docs/data_schema.md), tokenize it against the target
# tokenizer, compute per-token weights w_j from the priors, and produce
# aligned `input_ids`, `attention_mask`, `labels`, `weights`, and
# `preferred_mask` lists ready for the collator above.
#
# `dynamic_path=True` regenerates a fresh pseudo-scan path each call
# (closer to Algorithm 1 as written); `dynamic_path=False` reuses the
# precomputed mask in the example, which is faster and what our small-scale
# runs did.
# ----------------------------------------------------------------------------

def build_training_example(
    example:       Dict[str, Any],
    tokenizer:     PreTrainedTokenizerBase,
    priors:        Dict[str, Any],
    task_prompt:   str = "### Instruction:\nComplete the following code snippet.\n\n",
    input_header:  str = "### Input Code:\n",
    output_header: str = "\n\n### Output:\n",
    max_length:    int = 1024,
    dynamic_path:  bool = False,
    rho:           Optional[float] = None,
) -> Dict[str, List]:
    mask: Sequence[int]
    if dynamic_path:
        if rho is None:
            alpha_agg, beta_agg = aggregate_beta_params(priors)
            rho = sample_attention_density(alpha_agg, beta_agg)
        mask = generate_pseudo_scan_path(
            example["semantic_token_sequence"], priors, rho,
        )
        ngram_idx = example["ngram_indices"]        # keep for rarity bonus
        sem_seq   = example["semantic_token_sequence"]
    else:
        mask      = example["mask"]
        ngram_idx = example["ngram_indices"]
        sem_seq   = example["semantic_token_sequence"]

    w_code = [
        token_weight(mask[j], ngram_idx[j], sem_seq[j], priors)
        for j in range(len(example["code_tokens"]))
    ]

    prompt = f"{task_prompt}{input_header}{example['code']}{output_header}"
    text   = f"{prompt}{example['content']}{tokenizer.eos_token}"

    tok_full   = tokenizer(text,   max_length=max_length, truncation=True)
    tok_prompt = tokenizer(prompt, max_length=max_length, truncation=True)
    input_ids  = tok_full["input_ids"]
    n_prompt   = len(tok_prompt["input_ids"])
    n_out      = len(input_ids) - n_prompt

    labels = list(input_ids)
    labels[:n_prompt] = [-100] * n_prompt

    # Weight schedule for the output portion: the mean input-side weight is a
    # simple model-free aggregate. Cross-attention alignment or retrieval-
    # guided mapping would be natural drop-in replacements.
    mean_w  = sum(w_code) / len(w_code) if w_code else 1.0
    weights = [1.0] * n_prompt + [float(mean_w)] * n_out

    # Preferred-mask schedule for the preference loss: mark each output-side
    # token as "preferred" at a rate proportional to example-level salience.
    # A token-aligned scheme (projecting `mask` through tokenizer offsets) is
    # a stronger option once the pipeline is stable.
    preferred_rate = sum(mask) / max(1, len(mask))
    preferred_mask = [0] * n_prompt + [
        int(torch.rand(1).item() < preferred_rate) for _ in range(n_out)
    ]

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
        "weights":        weights,
        "preferred_mask": preferred_mask,
    }


# ============================================================================
# 9. Wiring sketch  (pseudocode --- NOT executed by this file)
# ============================================================================
#
#     from compute_token_weights import load_priors, token_weight
#     priors = load_priors("priors/combined")
#
#     tokenizer = AutoTokenizer.from_pretrained(YOUR_BACKBONE)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#
#     policy    = CausalLMWithWeightedLoss.from_pretrained(YOUR_BACKBONE)
#     reference = CausalLMWithWeightedLoss.from_pretrained(YOUR_BACKBONE).eval()
#     for p in reference.parameters():
#         p.requires_grad_(False)
#
#     objective = EyeMulatorCompositeObjective(policy=policy, reference=reference,
#                                              gamma=0.1, beta=0.1)
#
#     features = [
#         build_training_example(ex, tokenizer, priors, dynamic_path=True)
#         for ex in your_dataset
#     ]
#     collate = WeightedCollator(tokenizer=tokenizer, max_length=YOUR_MAX_LEN)
#     loader  = torch.utils.data.DataLoader(features, batch_size=YOUR_BS,
#                                           collate_fn=collate, shuffle=True)
#
#     optim = torch.optim.AdamW(policy.parameters(), lr=YOUR_LR)
#     for epoch in range(YOUR_EPOCHS):
#         for batch in loader:
#             loss = objective(**batch)
#             optim.zero_grad(); loss.backward(); optim.step()
#
# Replace `YOUR_BACKBONE`, `YOUR_MAX_LEN`, `YOUR_BS`, `YOUR_LR`, and
# `YOUR_EPOCHS` with values appropriate to your compute budget. Distributed
# training, mixed precision, and evaluation scaffolding are left out; they
# are infrastructure-specific.
# ============================================================================
