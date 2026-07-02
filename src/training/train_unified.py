#!/usr/bin/env python3
"""
Unified EyeMulator Training Script (Rebuttal Version)
=====================================================
Addresses reviewer concerns:
  - Multi-seed support (mean±std across seeds)   [ZmvJ, NpXQ]
  - Statistical significance via reproducible seeds [ZmvJ]
  - Ablation variants (weighted-SFT-only, random-attention, eyetracking-only) [ZmvJ, CX74]
  - LoRA (r=16, alpha=32) for speed & memory     [ZmvJ scalability]
  - bf16 / flash-attention-2 / gradient-checkpointing for speed [user request]
  - Training time / overhead reporting            [ZmvJ]
  - Configurable via CLI args (no manual edits)

Usage examples:
  # EyeMulator (full) – Llama, completion, seed 42
  python train_unified.py --model llama --task completion --method eyemulator --seed 42

  # Baseline – DeepSeek, translation
  python train_unified.py --model deepseek --task translation --method baseline --seed 42

  # Ablation: random weights
  python train_unified.py --model llama --task completion --method random --seed 42

  # Ablation: eyetracking-only (no ngram rarity bonus)
  python train_unified.py --model llama --task completion --method eyetracking_only --seed 42

  # Ablation: weighted SFT only (no semantic attention, just ngram)
  python train_unified.py --model llama --task completion --method weighted_sft_only --seed 42

  # Session-mode (reading/writing priors)
  python train_unified.py --model llama --task completion --method eyemulator --data-category reading --seed 42
"""

import os
import sys
import gc
import json
import math
import time
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    BitsAndBytesConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# Try to import LoRA / PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not installed. Running without LoRA. Install via: pip install peft")

# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Unified EyeMulator Training Script")
    p.add_argument("--model", type=str, required=True,
                   choices=["llama", "deepseek", "starcoder",
                            "qwen35-2b", "smollm3-3b"],
                   help="Model backbone to fine-tune")
    p.add_argument("--task", type=str, required=True,
                   choices=["completion", "translation", "summarization"],
                   help="Downstream task")
    p.add_argument("--method", type=str, default="eyemulator",
                   choices=["baseline", "eyemulator", "random",
                            "eyetracking_only", "weighted_sft_only"],
                   help="Training method / ablation variant")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--data-category", type=str, default=None,
                   choices=["reading", "writing"],
                   help="Session-mode data category (optional)")
    p.add_argument("--data-folder", type=str, default="./data",
                   help="Path to data folder")
    p.add_argument("--output-root", type=str, default="./",
                   help="Root directory for model output")
    p.add_argument("--epochs", type=int, default=2,
                   help="Number of training epochs (default 2 for rebuttal speed)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Maximum sequence length")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA alpha")
    p.add_argument("--no-lora", action="store_true",
                   help="Disable LoRA (full fine-tuning)")
    p.add_argument("--no-bf16", action="store_true",
                   help="Disable bf16 (use fp32)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max training samples (subsample for speed). None = use all.")
    p.add_argument("--no-eval-during-training", action="store_true",
                   help="Skip epoch-level validation to reduce peak GPU memory.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="CUDA device")
    p.add_argument("--auth-token", type=str,
                   default=os.environ.get("HF_TOKEN", None),
                   help="HuggingFace auth token (default: $HF_TOKEN env var)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama": "meta-llama/Llama-3.2-1B",
    "deepseek": "deepseek-ai/deepseek-coder-1.3b-base",
    "starcoder": "bigcode/starcoderbase-1b",
    "qwen35-2b": "Qwen/Qwen2.5-Coder-1.5B",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
}

TASK_PROMPTS = {
    "translation": {
        "instruction": "### Instruction:\nTranslate the following Java code to C#.\n\n",
        "input_header": "### Input Code:\n",
        "output_header": "\n\n### Output:\n",
    },
    "summarization": {
        "instruction": "### Instruction:\nSummarize the following code.\n\n",
        "input_header": "### Input Code:\n",
        "output_header": "\n\n### Output:\n",
    },
    "completion": {
        "instruction": "### Instruction:\nComplete the following code snippet.\n\n",
        "input_header": "### Input Code:\n",
        "output_header": "\n\n### Output:\n",
    },
}

# Baseline scripts used a DIFFERENT prompt format — we keep the above
# (eyemulator format) for all new unified runs. The existing baseline/eyemulator
# results were already generated with the original scripts.


# ─────────────────────────────────────────────────────────────────────────────
# Custom Model: Weighted Loss
# ─────────────────────────────────────────────────────────────────────────────
def weighted_loss_forward(model_self, input_ids=None, attention_mask=None,
                          labels=None, weights=None, **kwargs):
    """Unified weighted-loss forward pass, works for any CausalLM."""
    # Remove keys we set explicitly to avoid duplicates
    kwargs.pop("output_attentions", None)
    kwargs.pop("output_hidden_states", None)
    kwargs.pop("labels", None)
    outputs = model_self.__class__.__mro__[1].forward(
        model_self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    )
    logits = outputs.logits
    final_loss = None

    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        flat_logits = shift_logits.view(-1, model_self.config.vocab_size)
        flat_labels = shift_labels.view(-1)
        loss_per_token = loss_fct(flat_logits, flat_labels)
        active_loss_mask = flat_labels != -100
        active_losses = loss_per_token[active_loss_mask]

        if weights is not None:
            shift_weights = weights[..., 1:].contiguous().view(-1)
            active_weights = shift_weights[active_loss_mask]
            scaled_loss = (active_losses * active_weights).sum()
            num_active_tokens = active_loss_mask.sum()
            final_loss = scaled_loss / (num_active_tokens + 1e-9)
        else:
            final_loss = active_losses.mean()

    return CausalLMOutputWithPast(
        loss=final_loss, logits=logits, past_key_values=outputs.past_key_values,
    )


class LlamaForCausalLMWithWeightedLoss(LlamaForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                weights=None, **kwargs):
        return weighted_loss_forward(
            self, input_ids, attention_mask, labels, weights, **kwargs)


def create_weighted_model_class(model_id, auth_token):
    """Dynamically create a weighted-loss model class for non-Llama architectures."""
    base_model_tmp = AutoModelForCausalLM.from_pretrained(
        model_id, token=auth_token, trust_remote_code=True)
    base_class = base_model_tmp.__class__
    del base_model_tmp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    class DynamicWeightedLossModel(base_class):
        def forward(self, input_ids=None, attention_mask=None, labels=None,
                    weights=None, **kwargs):
            return weighted_loss_forward(
                self, input_ids, attention_mask, labels, weights, **kwargs)

    return DynamicWeightedLossModel


# ─────────────────────────────────────────────────────────────────────────────
# Custom Data Collator (for weighted methods)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WeightedDataCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        max_len = self.max_length
        for key in features[0].keys():
            for i in range(len(features)):
                if len(features[i][key]) > max_len:
                    features[i][key] = features[i][key][:max_len]
            if key == "weights":
                pad_value, dtype = 1.0, torch.float
            elif key == "labels":
                pad_value, dtype = -100, torch.long
            else:
                pad_value, dtype = self.tokenizer.pad_token_id, torch.long
            padded = [f[key] + [pad_value] * (max_len - len(f[key])) for f in features]
            batch[key] = torch.tensor(padded, dtype=dtype)
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading: Eye-tracking artifacts
# ─────────────────────────────────────────────────────────────────────────────
def load_eyetracking_artifacts(data_folder, data_category=None):
    """Load n-gram indices, counts, and beta distributions."""
    index_to_ngram = {}
    ngram_counts = {}
    semantic_label_attention = {}
    semantic_id_to_label = {}

    indexed_ngrams_file = os.path.join(data_folder, 'indexed_ngrams.json')

    if data_category:
        cat_folder = os.path.join(os.path.dirname(data_folder), f"data_{data_category}")
        mono_file = os.path.join(cat_folder, f'{data_category}_monogram_counts.json')
        bi_file = os.path.join(cat_folder, f'{data_category}_bigram_counts.json')
        tri_file = os.path.join(cat_folder, f'{data_category}_trigram_counts.json')
        beta_file = os.path.join(cat_folder, f'{data_category}_beta_distribution.json')
    else:
        mono_file = os.path.join(data_folder, 'monogram_counts.json')
        bi_file = os.path.join(data_folder, 'bigram_counts.json')
        tri_file = os.path.join(data_folder, 'trigram_counts.json')
        beta_file = os.path.join(data_folder, 'combined_beta_distribution.json')

    logging.info(f"Loading indexed n-grams from: {indexed_ngrams_file}")
    with open(indexed_ngrams_file, 'r') as f:
        ngram_to_index = json.load(f)
        for k, v in ngram_to_index.items():
            index_to_ngram[int(v)] = k
            if not k.startswith('('):
                semantic_id_to_label[int(v)] = k

    logging.info(f"Loading n-gram counts...")
    for fp in [mono_file, bi_file, tri_file]:
        with open(fp, 'r') as f:
            ngram_counts.update(json.load(f))

    logging.info(f"Loading beta distribution from: {beta_file}")
    with open(beta_file, 'r') as f:
        beta_data = json.load(f)
        for item in beta_data:
            label = item['semantic_label'].strip()
            alpha = item['saccades_count']
            beta_val = item['word_count'] - item['saccades_count']
            mean_attention = alpha / (alpha + beta_val) if (alpha + beta_val) > 0 else 0
            semantic_label_attention[label] = mean_attention

    return index_to_ngram, ngram_counts, semantic_label_attention, semantic_id_to_label


# ─────────────────────────────────────────────────────────────────────────────
# Weight Calculation (supports ablation variants)
# ─────────────────────────────────────────────────────────────────────────────
def make_weight_calculator(method, index_to_ngram, ngram_counts,
                            semantic_label_attention, semantic_id_to_label):
    """Returns a weight calculation function based on the method."""
    def calculate_weight(mask, ngram_idx, semantic_token_id):
        if mask == 0:
            return 1.0

        if method == "random":
            return random.uniform(1.0, 5.0)

        base_weight = 3.0

        # Ngram rarity bonus (absent in eyetracking_only ablation)
        ngram_rarity_bonus = 0.0
        if method != "eyetracking_only":
            ngram_pattern = index_to_ngram.get(int(ngram_idx))
            if ngram_pattern:
                ngram_count = ngram_counts.get(ngram_pattern, 0)
                if ngram_count > 0:
                    ngram_rarity_bonus = 1.0 / math.log(ngram_count + 2)

        # Semantic attention bonus (absent in weighted_sft_only ablation)
        semantic_attention_bonus = 0.0
        if method != "weighted_sft_only":
            semantic_label = semantic_id_to_label.get(int(semantic_token_id))
            if semantic_label:
                semantic_attention_bonus = semantic_label_attention.get(semantic_label, 0.0)

        return base_weight + ngram_rarity_bonus + semantic_attention_bonus

    return calculate_weight


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading and Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate_data(file_path, require_eyetracking=True):
    """Load JSONL data with optional eyetracking field validation."""
    data = []
    if require_eyetracking:
        required_keys = ['code_tokens', 'mask', 'ngram_indices',
                        'semantic_token_sequence', 'code', 'content']
    else:
        required_keys = ['code', 'content']

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                if not all(k in record and record[k] is not None for k in required_keys):
                    continue
                if require_eyetracking:
                    token_len = len(record['code_tokens'])
                    if not all(len(record[key]) == token_len
                              for key in ['mask', 'ngram_indices', 'semantic_token_sequence']):
                        continue
                data.append(record)
            except Exception as e:
                logging.warning(f"Skipping malformed line {i+1}: {e}")
    return data


def create_baseline_dataset(tokenizer, args):
    """Create dataset for standard SFT baseline (no weights)."""
    train_data = load_and_validate_data(
        os.path.join(args.data_folder, f"{args.task}_train_final.jsonl"),
        require_eyetracking=False)
    valid_data = load_and_validate_data(
        os.path.join(args.data_folder, f"{args.task}_valid_final.jsonl"),
        require_eyetracking=False)

    if not train_data or not valid_data:
        logging.error("Data is empty!"); sys.exit(1)

    # Subsample if requested (for speed during rebuttal)
    if args.max_samples and len(train_data) > args.max_samples:
        random.seed(args.seed)  # deterministic subsampling
        train_data = random.sample(train_data, args.max_samples)
        logging.info(f"Subsampled training data to {args.max_samples} samples")

    prompts = TASK_PROMPTS[args.task]

    def preprocess(example):
        code_str = example['code']
        content_str = example['content']
        full_prompt = f"{prompts['instruction']}{prompts['input_header']}{code_str}{prompts['output_header']}"
        full_text = f"{full_prompt}{content_str}{tokenizer.eos_token}"
        tokenized = tokenizer(full_text, max_length=args.max_length, truncation=True)
        return tokenized

    raw = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(valid_data),
    })
    tokenized = raw.map(preprocess, num_proc=os.cpu_count(),
                        remove_columns=raw["train"].column_names)
    if "token_type_ids" in tokenized["train"].column_names:
        tokenized = tokenized.remove_columns(["token_type_ids"])
    return tokenized


def create_weighted_dataset(tokenizer, args, calc_weight_fn):
    """Create dataset with per-token weights (for EyeMulator and ablations)."""
    train_data = load_and_validate_data(
        os.path.join(args.data_folder, f"{args.task}_train_final.jsonl"),
        require_eyetracking=True)
    valid_data = load_and_validate_data(
        os.path.join(args.data_folder, f"{args.task}_valid_final.jsonl"),
        require_eyetracking=True)

    if not train_data or not valid_data:
        logging.error("Data is empty!"); sys.exit(1)

    # Subsample if requested (for speed during rebuttal)
    if args.max_samples and len(train_data) > args.max_samples:
        random.seed(args.seed)  # deterministic subsampling
        train_data = random.sample(train_data, args.max_samples)
        logging.info(f"Subsampled training data to {args.max_samples} samples")

    raw = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(valid_data),
    })
    prompts = TASK_PROMPTS[args.task]

    def preprocess_and_weigh(example):
        code_str = example['code']
        content_str = example['content']

        weights_per_code_token = [
            calc_weight_fn(
                example['mask'][j],
                example['ngram_indices'][j],
                example['semantic_token_sequence'][j]
            ) for j in range(len(example['code_tokens']))
        ]
        aggregate_weight = max(weights_per_code_token) if weights_per_code_token else 1.0

        full_prompt = f"{prompts['instruction']}{prompts['input_header']}{code_str}{prompts['output_header']}"
        full_text = f"{full_prompt}{content_str}{tokenizer.eos_token}"

        tokenized_full = tokenizer(full_text, max_length=args.max_length, truncation=True)
        tokenized_prompt = tokenizer(full_prompt, max_length=args.max_length, truncation=True)

        input_ids = tokenized_full['input_ids']
        attention_mask = [1] * len(input_ids)
        labels = list(input_ids)
        prompt_len = len(tokenized_prompt['input_ids'])
        labels[:prompt_len] = [-100] * prompt_len
        weights = [1.0] * prompt_len + [aggregate_weight] * (len(input_ids) - prompt_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weights": weights,
        }

    return raw.map(preprocess_and_weigh, num_proc=os.cpu_count(),
                   remove_columns=raw["train"].column_names)


# ─────────────────────────────────────────────────────────────────────────────
# Seed Setting
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For full determinism (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup logging
    method_tag = args.method
    if args.data_category:
        method_tag = f"{args.method}_{args.data_category}"

    output_dir = os.path.join(
        args.output_root,
        f"{args.model}_{method_tag}_{args.task}_seed{args.seed}"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log") if os.path.exists(output_dir) else f"train_{args.model}_{method_tag}_{args.task}_seed{args.seed}.log"),
        ]
    )

    logging.info("=" * 70)
    logging.info(f"EyeMulator Unified Training")
    logging.info(f"  Model:    {args.model} ({MODEL_REGISTRY[args.model]})")
    logging.info(f"  Task:     {args.task}")
    logging.info(f"  Method:   {args.method}")
    logging.info(f"  Category: {args.data_category or 'combined'}")
    logging.info(f"  Seed:     {args.seed}")
    logging.info(f"  LoRA:     {'disabled' if args.no_lora else f'r={args.lora_r}, alpha={args.lora_alpha}'}")
    logging.info(f"  bf16:     {not args.no_bf16}")
    logging.info(f"  Output:   {output_dir}")
    logging.info("=" * 70)

    # ── Load tokenizer ──
    model_id = MODEL_REGISTRY[args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=args.auth_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

    # ── Decide whether we need weighted training ──
    is_weighted = args.method != "baseline"

    # ── Load model ──
    model_kwargs = {
        "token": args.auth_token,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if not args.no_bf16 else torch.float32,
    }

    # Use flash attention 2 if available (requires flash_attn package)
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logging.info("Using Flash Attention 2")
    except ImportError:
        logging.info("flash_attn not installed, using default attention")

    if is_weighted:
        # Need weighted-loss model
        if args.model == "llama" or args.model == "deepseek":
            model = LlamaForCausalLMWithWeightedLoss.from_pretrained(
                model_id, device_map="auto", **model_kwargs)
        else:
            # StarCoder or other architectures
            WeightedClass = create_weighted_model_class(model_id, args.auth_token)
            model = WeightedClass.from_pretrained(
                model_id, device_map="auto", **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", **model_kwargs)

    # ── Enable gradient checkpointing for memory efficiency ──
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # ── Apply LoRA ──
    if not args.no_lora and PEFT_AVAILABLE:
        logging.info(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        # Find target modules dynamically
        target_modules = []
        for name, _ in model.named_modules():
            if any(t in name for t in ["q_proj", "v_proj", "k_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj",
                                        "c_attn", "c_proj", "c_fc"]):
                # Get the leaf module name
                leaf = name.split(".")[-1]
                if leaf not in target_modules:
                    target_modules.append(leaf)

        if not target_modules:
            target_modules = ["q_proj", "v_proj"]  # fallback

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif args.no_lora:
        logging.info("LoRA disabled – full fine-tuning")
    else:
        logging.warning("PEFT not available. Falling back to full fine-tuning.")

    # ── Prepare dataset ──
    if is_weighted:
        artifacts = load_eyetracking_artifacts(args.data_folder, args.data_category)
        calc_weight_fn = make_weight_calculator(args.method, *artifacts)
        tokenized_datasets = create_weighted_dataset(tokenizer, args, calc_weight_fn)
        data_collator = WeightedDataCollator(tokenizer=tokenizer, max_length=args.max_length)
    else:
        tokenized_datasets = create_baseline_dataset(tokenizer, args)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    logging.info(f"Train size: {len(tokenized_datasets['train'])}, "
                 f"Valid size: {len(tokenized_datasets['validation'])}")

    # ── Training arguments ──
    use_bf16 = not args.no_bf16 and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="no" if args.no_eval_during_training else "epoch",
        eval_strategy="no" if args.no_eval_during_training else "epoch",
        load_best_model_at_end=not args.no_eval_during_training,
        metric_for_best_model=None if args.no_eval_during_training else "eval_loss",
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        dataloader_num_workers=4,
        prediction_loss_only=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False if is_weighted else True,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        # Speed optimizations
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=None if args.no_eval_during_training else tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # ── Train ──
    logging.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    # ── Save training metadata ──
    metadata = {
        "model": args.model,
        "model_id": model_id,
        "task": args.task,
        "method": args.method,
        "data_category": args.data_category,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "lora_r": args.lora_r if not args.no_lora else None,
        "lora_alpha": args.lora_alpha if not args.no_lora else None,
        "bf16": use_bf16,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_samples": len(tokenized_datasets["train"]),
        "valid_samples": len(tokenized_datasets["validation"]),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(os.path.join(output_dir, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # ── Report ──
    logging.info("=" * 70)
    logging.info("Training Complete!")
    logging.info(f"  Total time: {training_time:.1f}s ({training_time/60:.1f} min)")
    logging.info(f"  Final train loss: {train_result.metrics.get('train_loss', 'N/A')}")
    logging.info(f"  Model saved to: {output_dir}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
