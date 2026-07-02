import os
import torch
import json
import logging
import argparse
from transformers import (
    AutoTokenizer,
    GPTBigCodeForCausalLM, # Changed from LlamaForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional

# --- Custom Model Definition (Adapted for StarCoder) ---
# This class is necessary if the saved model was trained using a custom class structure.
class StarCoderForCausalLMWithWeightedLoss(GPTBigCodeForCausalLM): # Changed from LlamaForCausalLM
    def forward(
        self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None, weights: Optional[torch.FloatTensor] = None, **kwargs,
    ):
        # The custom loss logic is only needed for training. The standard forward pass is sufficient for inference.
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

# --- Prompts for Different Tasks (Must match training) ---
# This structure is assumed to be consistent with how the StarCoder model was fine-tuned.
TASK_PROMPTS = {
    "translation": {"instruction": "Translate the following Java code to C#.", "input_header": "### Java Code:", "output_header": "### C# Code:", "input_key": "code"},
    "summarization": {"instruction": "Summarize the following Java code.", "input_header": "### Java Code:", "output_header": "### Summary:", "input_key": "code"},
    "completion": {"instruction": "Complete the following Java code.", "input_header": "### Java Code:", "output_header": "### Completion:", "input_key": "code"}
}

def generate_results(model_path, task, test_data_file, output_file, device):
    """Loads a trained model and generates outputs for the test data."""
    logging.info(f"--- Starting Evaluation ---")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Task: {task}, Device: {device}")

    PROMPT_CONFIG = TASK_PROMPTS[task]
    
    try:
        # The tokenizer name is inferred from the model path.
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure pad token is set for batch generation, a common practice for StarCoder
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model = StarCoderForCausalLMWithWeightedLoss.from_pretrained(model_path) # Changed to use StarCoder class
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading the model or tokenizer: {e}"); return

    logging.info(f"Loading test data from {test_data_file}...")
    test_data = [json.loads(line) for line in open(test_data_file, 'r', encoding='utf-8')]
    
    results = []
    logging.info(f"Generating results for {len(test_data)} examples...")
    for i, example in enumerate(test_data):
        input_text = example.get(PROMPT_CONFIG['input_key'])
        ground_truth = example.get("content")
        if not input_text or not ground_truth: continue

        prompt = (f"### Instruction:\n{PROMPT_CONFIG['instruction']}\n\n"
                  f"{PROMPT_CONFIG['input_header']}\n{input_text}\n\n"
                  f"{PROMPT_CONFIG['output_header']}\n")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, do_sample=False)
        
        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_output = full_generated_text.split(PROMPT_CONFIG['output_header'])[-1].strip()

        results.append({"input": input_text, "ground_truth": ground_truth, "generated_output": generated_output})
        if (i + 1) % 50 == 0: logging.info(f"Processed example {i+1}/{len(test_data)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logging.info(f"Saving generation results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logging.info("Successfully generated and saved all results. 🎉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned StarCoder model.")
    parser.add_argument("--task", type=str, default="completion", choices=["translation", "summarization", "completion"], help="The task to evaluate.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run evaluation on (e.g., 'cuda:0', 'cpu').")
    args = parser.parse_args()

    # --- Paths for the StarCoder model version ---
    MODEL_PATH = f"./starcoder_advanced_final_{args.task}" # Changed path to reflect starcoder model
    DATA_FILE = f"./data/{args.task}_valid_final.jsonl"
    OUTPUT_FILE = f"./starcoder_results/generated_results_advanced_final_{args.task}.json" # Changed output folder

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generate_results(MODEL_PATH, args.task, DATA_FILE, OUTPUT_FILE, args.device)