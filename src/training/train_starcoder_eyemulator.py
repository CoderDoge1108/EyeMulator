import os
import torch
import json
import logging
import numpy as np
import time
import math
import transformers
import gc
import sys
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

# --- Task Configuration ---
# Choose from completion, translation and summarization
TASK = "completion"

# --- Main Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_ID = "bigcode/starcoderbase-1b"
MODEL_NAME = "starcoder"
AUTH_TOKEN = "PLACEHOLDER" # Replace with your token

DATA_FOLDER = "./data"
OUTPUT_DIR = f"./{MODEL_NAME}_advanced_final_{TASK}"
TRAIN_FILE = os.path.join(DATA_FOLDER, f"{TASK}_train_final.jsonl")
VALID_FILE = os.path.join(DATA_FOLDER, f"{TASK}_valid_final.jsonl")
MAX_LENGTH = 1024

# --- Global Variables for Loaded Data ---
INDEX_TO_NGRAM = {}
NGRAM_COUNTS = {}
SEMANTIC_LABEL_ATTENTION = {}
SEMANTIC_ID_TO_LABEL = {}

# --- Custom Data Collator ---
@dataclass
class CustomDataCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        max_len = MAX_LENGTH
        for key in features[0].keys():
            for i in range(len(features)):
                if len(features[i][key]) > max_len:
                    features[i][key] = features[i][key][:max_len]

            if key == "weights": pad_value, dtype = 1.0, torch.float
            elif key == "labels": pad_value, dtype = -100, torch.long
            else: pad_value, dtype = self.tokenizer.pad_token_id, torch.long

            padded_list = [f[key] + [pad_value] * (max_len - len(f[key])) for f in features]
            batch[key] = torch.tensor(padded_list, dtype=dtype)
        return batch

# --- Data-Driven Weight Calculation ---
def calculate_weight(mask, ngram_idx, semantic_token_id):
    if mask == 0:
        return 1.0
    base_weight = 3.0
    
    ngram_rarity_bonus = 0.0
    # Use int() for lookup to prevent data type mismatches
    ngram_pattern = INDEX_TO_NGRAM.get(int(ngram_idx))
    if ngram_pattern:
        ngram_count = NGRAM_COUNTS.get(ngram_pattern, 0)
        if ngram_count > 0:
            ngram_rarity_bonus = 1.0 / math.log(ngram_count + 2)
            
    semantic_attention_bonus = 0.0
    # Use int() for lookup to prevent data type mismatches
    semantic_label = SEMANTIC_ID_TO_LABEL.get(int(semantic_token_id))
    if semantic_label:
        semantic_attention_bonus = SEMANTIC_LABEL_ATTENTION.get(semantic_label, 0.0)
        
    final_weight = base_weight + ngram_rarity_bonus + semantic_attention_bonus
    return final_weight

# --- Data Loading and Preprocessing ---
def load_and_validate_data(file_path: str) -> List[Dict]:
    data = []
    required_keys = ['code_tokens', 'mask', 'ngram_indices', 'semantic_token_sequence', 'code', 'content']
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if not all(k in record and record[k] is not None for k in required_keys): continue
                token_len = len(record['code_tokens'])
                if not all(len(record[key]) == token_len for key in required_keys if key not in ['code', 'content']): continue
                data.append(record)
            except Exception: continue
    return data

def create_and_process_data(tokenizer):
    logging.info("Loading and validating datasets...")
    train_data = load_and_validate_data(TRAIN_FILE)
    valid_data = load_and_validate_data(VALID_FILE)

    if not train_data or not valid_data:
        logging.error("Training or validation data is empty. Please check file paths and content integrity.")
        sys.exit(1)
        
    raw_datasets = DatasetDict({"train": Dataset.from_list(train_data), "validation": Dataset.from_list(valid_data)})
    
    task_prompts = {
        "translation": "### Instruction:\nTranslate the following Java code to C#.\n\n",
        "summarization": "### Instruction:\nSummarize the following code.\n\n",
        "completion": "### Instruction:\nComplete the following code snippet.\n\n"
    }
    
    def preprocess_and_weigh(example):
        code_str, content_str = example['code'], example['content']
        # Use task-specific prompts, falling back to a generic one
        instruction_prompt = task_prompts.get(TASK, "### Instruction:\nProcess the following code.\n\n")
        input_header = "### Input Code:\n"
        output_header = f"\n\n### Output:\n"
        
        weights_per_code_token = [
            calculate_weight(
                example['mask'][j],
                example['ngram_indices'][j],
                example['semantic_token_sequence'][j]
            ) for j in range(len(example['code_tokens']))
        ]
        # Use MAX aggregation to amplify the statistical signal
        aggregate_weight = max(weights_per_code_token) if weights_per_code_token else 1.0

        full_prompt = f"{instruction_prompt}{input_header}{code_str}{output_header}"
        full_text = f"{full_prompt}{content_str}{tokenizer.eos_token}"
        
        tokenized_full = tokenizer(full_text, max_length=MAX_LENGTH, truncation=True)
        tokenized_prompt = tokenizer(full_prompt, max_length=MAX_LENGTH, truncation=True)
        
        input_ids = tokenized_full['input_ids']
        attention_mask = [1] * len(input_ids)
        labels = list(input_ids)
        labels[:len(tokenized_prompt['input_ids'])] = [-100] * len(tokenized_prompt['input_ids'])
        
        weights = [1.0] * len(tokenized_prompt['input_ids']) + [aggregate_weight] * (len(input_ids) - len(tokenized_prompt['input_ids']))
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}

    logging.info("Applying data-driven preprocessing with MAX weight aggregation...")
    return raw_datasets.map(preprocess_and_weigh, num_proc=os.cpu_count(), remove_columns=raw_datasets["train"].column_names)

def main():
    # --- Robust Data Loading ---
    global INDEX_TO_NGRAM, NGRAM_COUNTS, SEMANTIC_LABEL_ATTENTION, SEMANTIC_ID_TO_LABEL
    
    # Clear global dictionaries to ensure a clean state
    INDEX_TO_NGRAM.clear()
    NGRAM_COUNTS.clear()
    SEMANTIC_LABEL_ATTENTION.clear()
    SEMANTIC_ID_TO_LABEL.clear()
    
    logging.info("Loading statistical and semantic data files for weighting...")
    
    # Define file paths for the single data source
    indexed_ngrams_file = os.path.join(DATA_FOLDER, 'indexed_ngrams.json')
    monogram_counts_file = os.path.join(DATA_FOLDER, 'monogram_counts.json')
    bigram_counts_file = os.path.join(DATA_FOLDER, 'bigram_counts.json')
    trigram_counts_file = os.path.join(DATA_FOLDER, 'trigram_counts.json')
    beta_dist_file = os.path.join(DATA_FOLDER, 'combined_beta_distribution.json')

    # Load and build all dictionaries, ensuring correct data types
    logging.info(f"Loading indexed n-grams from: {indexed_ngrams_file}")
    with open(indexed_ngrams_file, 'r') as f:
        ngram_to_index = json.load(f)
        for k, v in ngram_to_index.items():
            INDEX_TO_NGRAM[int(v)] = k
            if not k.startswith('('):
                SEMANTIC_ID_TO_LABEL[int(v)] = k

    logging.info(f"Loading n-gram counts from: {DATA_FOLDER}")
    with open(monogram_counts_file, 'r') as f: NGRAM_COUNTS.update(json.load(f))
    with open(bigram_counts_file, 'r') as f: NGRAM_COUNTS.update(json.load(f))
    with open(trigram_counts_file, 'r') as f: NGRAM_COUNTS.update(json.load(f))

    logging.info(f"Loading beta distribution from: {beta_dist_file}")
    with open(beta_dist_file, 'r') as f:
        beta_data = json.load(f)
        for item in beta_data:
            label = item['semantic_label'].strip()
            alpha = item['saccades_count']
            beta = item['word_count'] - item['saccades_count']
            mean_attention = alpha / (alpha + beta) if (alpha + beta) > 0 else 0
            SEMANTIC_LABEL_ATTENTION[label] = mean_attention
            
    logging.info("Data files loaded and processed successfully.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=AUTH_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id; tokenizer.padding_side = "right"
    
    # --- DYNAMIC MODEL CLASS IMPLEMENTATION ---
    logging.info(f"Dynamically creating custom model class for {MODEL_ID}")
    base_model_for_class = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=AUTH_TOKEN, trust_remote_code=True)
    model_class = base_model_for_class.__class__
    del base_model_for_class
    gc.collect()
    torch.cuda.empty_cache()

    class CustomModelForCausalLMWithWeightedLoss(model_class):
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            weights: Optional[torch.FloatTensor] = None,
            **kwargs,
        ):
            # 1. Get raw logits from the base model by passing `labels=None`.
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                output_attentions=False,
                output_hidden_states=False,
                **kwargs,
            )
            logits = outputs.logits

            # 2. Compute our custom loss only if labels are provided.
            final_loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = CrossEntropyLoss(reduction="none")
                flat_logits = shift_logits.view(-1, self.config.vocab_size)
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
                loss=final_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
            )

    # Load the model using the newly defined custom class
    model = CustomModelForCausalLMWithWeightedLoss.from_pretrained(
        MODEL_ID, use_auth_token=AUTH_TOKEN, device_map="auto", trust_remote_code=True)

    tokenized_datasets = create_and_process_data(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-5, num_train_epochs=3, report_to="tensorboard",
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, dataloader_num_workers=0
    )
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer=tokenizer)
    )

    logging.info("Starting data-driven advanced training...")
    trainer.train()
    
    logging.info("***** Training Complete *****")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info(f"Final model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()