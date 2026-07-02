import os
import torch
import json
import logging
import time
import math
import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EvalPrediction,
)

# --- 1. SET THE TASK HERE ---
# Choose from completion, translation and summarization
TASK = "completion"

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model and Tokenizer Configuration ---
MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "llama"
AUTH_TOKEN = "PLACEHOLDER"

# --- Data and Output Paths (Task-specific) ---
DATA_FOLDER = "./data"
OUTPUT_DIR = f"./{MODEL_NAME}_baseline_{TASK}"
TRAIN_FILE = os.path.join(DATA_FOLDER, f"{TASK}_train_final.jsonl")
VALID_FILE = os.path.join(DATA_FOLDER, f"{TASK}_valid_final.jsonl")
MAX_LENGTH = 1024

# --- Prompts for Different Tasks ---
TASK_PROMPTS = {
    "translation": {"instruction": "Translate the following Java code to C#.", "input_header": "### Java Code:", "output_header": "### C# Code:", "input_key": "code", "output_key": "content"},
    "summarization": {"instruction": "Summarize the following Java code.", "input_header": "### Java Code:", "output_header": "### Summary:", "input_key": "code", "output_key": "content"},
    "completion": {"instruction": "Complete the following Java code.", "input_header": "### Java Code:", "output_header": "### Completion:", "input_key": "code", "output_key": "content"}
}
PROMPT_CONFIG = TASK_PROMPTS[TASK]

def load_baseline_data(file_path: str) -> list:
    """Manually loads a .jsonl file, checking only for 'code' and 'content' keys."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get(PROMPT_CONFIG['input_key']) and record.get(PROMPT_CONFIG['output_key']):
                    data.append(record)
            except Exception: continue
    return data

def create_baseline_datasets(tokenizer):
    """Loads data and prepares it for standard fine-tuning."""
    train_data = load_baseline_data(TRAIN_FILE)
    valid_data = load_baseline_data(VALID_FILE)
    raw_datasets = DatasetDict({"train": Dataset.from_list(train_data), "validation": Dataset.from_list(valid_data)})
    
    def preprocess_function(examples):
        inputs = []
        for code, content in zip(examples[PROMPT_CONFIG['input_key']], examples[PROMPT_CONFIG['output_key']]):
            instruction = f"### Instruction:\n{PROMPT_CONFIG['instruction']}\n\n"
            input_header = f"{PROMPT_CONFIG['input_header']}\n"
            output_header = f"\n\n{PROMPT_CONFIG['output_header']}\n"
            prompt = f"{instruction}{input_header}{code}{output_header}"
            full_text = f"{prompt}{content}{tokenizer.eos_token}"
            inputs.append(full_text)
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, num_proc=os.cpu_count())
    if "token_type_ids" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns(["token_type_ids"])
    return tokenized_datasets

def compute_metrics(p: EvalPrediction):
    """Computes perplexity from evaluation loss."""
    try:
        loss = p.loss if isinstance(p.loss, float) else p.loss.mean().item()
        perplexity = math.exp(loss)
        return {"perplexity": perplexity}
    except:
        return {}

def log_system_info():
    """Logs system and library information for reproducibility."""
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Transformers version: {transformers.__version__}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("GPU not available, using CPU.")

def main():
    log_system_info()
    logging.info(f"Starting task: {TASK}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=AUTH_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id; tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, use_auth_token=AUTH_TOKEN, device_map="auto", trust_remote_code=True)
    
    tokenized_datasets = create_baseline_datasets(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=False,
        report_to="tensorboard",
        dataloader_num_workers=0,
        # ** THE FIX **: Tell the trainer to only compute the loss during evaluation
        # and not to gather the full logit tensors, which saves significant memory.
        prediction_loss_only=True,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting baseline model training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    logging.info("***** Training Complete *****")
    training_time_seconds = end_time - start_time
    logging.info(f"Total training time: {training_time_seconds:.2f} seconds ({training_time_seconds/60:.2f} minutes)")
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logging.info(f"Saving the final model to {OUTPUT_DIR}...")
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logging.info("All logs, including loss and perplexity history, can be found in the TensorBoard logs.")

if __name__ == "__main__":
    main()