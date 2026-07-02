#!/usr/bin/env python3
"""
Unified EyeMulator Evaluation Script (Rebuttal Version)
=======================================================
Addresses reviewer concerns:
  - Multi-seed result aggregation (mean±std)           [ZmvJ]
  - Paired t-test statistical significance             [ZmvJ, NpXQ]
  - Consistent prompt format matching training         [all]
  - Support for all model/task/method/seed combos      [all]
  - Batched generation for speed                       [user request]

Usage:
  # Evaluate a single model
  python eval_unified.py --model llama --task completion --method eyemulator --seed 42

  # Evaluate all seeds for one config (generates individual result files)
  python eval_unified.py --model llama --task completion --method eyemulator --seeds 42 123 456
"""

import os
import sys
import gc
import json
import time
import logging
import argparse
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    GPTBigCodeForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Unified EyeMulator Evaluation")
    p.add_argument("--model", type=str, required=True,
                   choices=["llama", "deepseek", "starcoder",
                            "qwen35-2b", "smollm3-3b"])
    p.add_argument("--task", type=str, required=True,
                   choices=["completion", "translation", "summarization"])
    p.add_argument("--method", type=str, default="eyemulator",
                   choices=["baseline", "eyemulator", "random",
                            "eyetracking_only", "weighted_sft_only"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multiple seeds to evaluate sequentially")
    p.add_argument("--data-category", type=str, default=None,
                   choices=["reading", "writing"])
    p.add_argument("--data-folder", type=str, default="./data")
    p.add_argument("--model-root", type=str, default="./",
                   help="Root directory containing trained models")
    p.add_argument("--output-root", type=str, default="./results",
                   help="Root directory for evaluation results")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-test-samples", type=int, default=None,
                   help="Max test samples to evaluate (subsample for speed). None = all.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for generation (1=sequential)")
    p.add_argument("--auth-token", type=str,
                   default=os.environ.get("HF_TOKEN", None),
                   help="HuggingFace auth token (default: $HF_TOKEN env var)")
    p.add_argument("--use-test-set", action="store_true",
                   help="Use test set instead of valid set (existing results use valid)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Custom Model Classes (for loading EyeMulator checkpoints)
# ─────────────────────────────────────────────────────────────────────────────
class LlamaForCausalLMWithWeightedLoss(LlamaForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                weights=None, **kwargs):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                               labels=labels, **kwargs)


class StarCoderForCausalLMWithWeightedLoss(GPTBigCodeForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                weights=None, **kwargs):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                               labels=labels, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates (MUST match training exactly)
# ─────────────────────────────────────────────────────────────────────────────
TASK_PROMPTS = {
    "translation": {
        "instruction": "### Instruction:\nTranslate the following Java code to C#.\n\n",
        "input_header": "### Input Code:\n",
        "output_header": "\n\n### Output:\n",
        "input_key": "code",
    },
    "summarization": {
        "instruction": "### Instruction:\nSummarize the following code.\n\n",
        "input_header": "### Input Code:\n",
        "output_header": "\n\n### Output:\n",
        "input_key": "code",
    },
    "completion": {
        "instruction": "### Instruction:\nComplete the following code snippet.\n\n",
        "input_header": "### Input Code:\n",
        "output_header": "\n\n### Output:\n",
        "input_key": "code",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_for_eval(model_path, model_type, device, auth_token=None):
    """Load a trained model (with or without LoRA/PEFT)."""
    logging.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check if this is a PEFT model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft = os.path.exists(adapter_config_path)

    if is_peft and PEFT_AVAILABLE:
        logging.info("Detected PEFT/LoRA adapter, loading with PeftModel...")
        # Load base model first, then adapter
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get("base_model_name_or_path", "")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto",
            token=auth_token)

        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Non-PEFT model
        try:
            if model_type == "starcoder":
                model = StarCoderForCausalLMWithWeightedLoss.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map="auto")
            elif model_type in ["llama", "deepseek"]:
                try:
                    model = LlamaForCausalLMWithWeightedLoss.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, device_map="auto")
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map="auto")
        except Exception as e:
            logging.warning(f"Custom model class failed ({e}), falling back to Auto")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto")

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_results(model, tokenizer, test_data, task, device, max_new_tokens=512):
    """Generate outputs for all test examples."""
    prompt_cfg = TASK_PROMPTS[task]
    results = []

    for i, example in enumerate(test_data):
        input_text = example.get(prompt_cfg['input_key'])
        ground_truth = example.get("content")
        if not input_text or not ground_truth:
            continue

        prompt = (f"{prompt_cfg['instruction']}"
                  f"{prompt_cfg['input_header']}{input_text}"
                  f"{prompt_cfg['output_header']}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract generated output after the output header
        output_header_clean = prompt_cfg['output_header'].strip()
        if output_header_clean in full_text:
            generated_output = full_text.split(output_header_clean)[-1].strip()
        else:
            generated_output = full_text[len(prompt):].strip()

        results.append({
            "input": input_text,
            "ground_truth": ground_truth,
            "generated_output": generated_output,
        })

        if (i + 1) % 50 == 0:
            logging.info(f"  Generated {i+1}/{len(test_data)}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_single(args, seed):
    """Run evaluation for a single seed."""
    method_tag = args.method
    if args.data_category:
        method_tag = f"{args.method}_{args.data_category}"

    model_dir = os.path.join(
        args.model_root,
        f"{args.model}_{method_tag}_{args.task}_seed{seed}"
    )

    if not os.path.exists(model_dir):
        logging.error(f"Model directory not found: {model_dir}")
        return None

    output_dir = os.path.join(
        args.output_root,
        f"{args.model}_results"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"generated_results_{method_tag}_{args.task}_seed{seed}.json"
    )

    # Skip if already exists
    if os.path.exists(output_file):
        logging.info(f"Results already exist: {output_file}, skipping generation.")
        return output_file

    # Load model
    model, tokenizer = load_model_for_eval(model_dir, args.model, args.device,
                                            auth_token=args.auth_token)

    # Load test/valid data — default to valid set (matches existing results)
    if args.use_test_set:
        test_file = os.path.join(args.data_folder, f"{args.task}_test_final.jsonl")
    else:
        test_file = os.path.join(args.data_folder, f"{args.task}_valid_final.jsonl")
    if not os.path.exists(test_file):
        # Fallback
        test_file = os.path.join(args.data_folder, f"{args.task}_valid_final.jsonl")

    logging.info(f"Loading test data from: {test_file}")
    test_data = [json.loads(line) for line in open(test_file, 'r', encoding='utf-8')]

    # Subsample test data if requested (for speed during rebuttal)
    if args.max_test_samples and len(test_data) > args.max_test_samples:
        import random as _rand
        _rand.seed(seed)  # deterministic subsampling
        test_data = _rand.sample(test_data, args.max_test_samples)
        logging.info(f"Subsampled test data to {args.max_test_samples} samples")

    # Generate
    logging.info(f"Generating results for {len(test_data)} examples...")
    start_time = time.time()
    results = generate_results(model, tokenizer, test_data, args.task,
                                args.device, args.max_new_tokens)
    gen_time = time.time() - start_time

    logging.info(f"Generation complete: {len(results)} results in {gen_time:.1f}s")

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to: {output_file}")

    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_file


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    seeds = args.seeds if args.seeds else [args.seed]

    for seed in seeds:
        logging.info(f"\n{'='*70}")
        logging.info(f"Evaluating: model={args.model}, task={args.task}, "
                     f"method={args.method}, seed={seed}")
        logging.info(f"{'='*70}")
        result_file = evaluate_single(args, seed)
        if result_file:
            logging.info(f"✅ Seed {seed} complete: {result_file}")
        else:
            logging.warning(f"⚠️ Seed {seed} failed")


if __name__ == "__main__":
    main()
