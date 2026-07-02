import os
import torch
import json
import logging
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Optional

# --- Prompts for Different Tasks (Must match training) ---
# This structure is assumed to be consistent with how the StarCoder model was fine-tuned.
TASK_PROMPTS = {
    "translation": {"instruction": "Translate the following Java code to C#.", "input_header": "### Java Code:", "output_header": "### C# Code:", "input_key": "code"},
    "summarization": {"instruction": "Summarize the following Java code.", "input_header": "### Java Code:", "output_header": "### Summary:", "input_key": "code"},
    "completion": {"instruction": "Complete the following Java code.", "input_header": "### Java Code:", "output_header": "### Completion:", "input_key": "code"}
}

def generate_results(model_path, task, test_data_file, output_file, device):
    """Loads a trained baseline model and generates outputs for the test data."""
    logging.info(f"--- Starting Baseline Model Evaluation ---")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Task: {task}, Device: {device}")

    PROMPT_CONFIG = TASK_PROMPTS[task]
    
    try:
        logging.info("Loading baseline model using AutoModelForCausalLM...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use the standard AutoModelForCausalLM for the baseline model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading the baseline model or tokenizer: {e}")
        return

    # Ensure the tokenizer has a pad token, which is crucial for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info("Tokenizer pad_token_id set to eos_token_id.")


    logging.info(f"Loading test data from {test_data_file}...")
    test_data = [json.loads(line) for line in open(test_data_file, 'r', encoding='utf-8')]
    
    results = []
    logging.info(f"Generating results for {len(test_data)} examples...")
    for i, example in enumerate(test_data):
        input_text = example.get(PROMPT_CONFIG['input_key'])
        ground_truth = example.get("content")
        if not input_text or not ground_truth:
            continue

        prompt = (f"### Instruction:\n{PROMPT_CONFIG['instruction']}\n\n"
                  f"{PROMPT_CONFIG['input_header']}\n{input_text}\n\n"
                  f"{PROMPT_CONFIG['output_header']}\n")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            # These generation settings are fine for StarCoder as well
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.15
            )
        
        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Safely split the generated text to extract only the generated part
        if PROMPT_CONFIG['output_header'] in full_generated_text:
             generated_output = full_generated_text.split(PROMPT_CONFIG['output_header'])[-1].strip()
        else:
            # As a fallback, take the part of the string after the prompt
            generated_output = full_generated_text[len(prompt):].strip()


        results.append({"input": input_text, "ground_truth": ground_truth, "generated_output": generated_output})
        if (i + 1) % 50 == 0:
            logging.info(f"Processed example {i+1}/{len(test_data)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logging.info(f"Saving generation results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logging.info("Successfully generated and saved all results for the baseline model. 🎉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the baseline fine-tuned StarCoder model.") # Updated description
    parser.add_argument("--task", type=str, default="completion", choices=["translation", "summarization", "completion"], help="The task to evaluate.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run evaluation on (e.g., 'cuda:1', 'cpu').")
    args = parser.parse_args()

    # --- Paths for the BASELINE StarCoder model ---
    MODEL_PATH = f"./starcoder_baseline_{args.task}" # Changed path to reflect starcoder model
    DATA_FILE = f"./data/{args.task}_valid_final.jsonl"
    OUTPUT_FILE = f"./starcoder_results/generated_results_baseline_{args.task}.json" # Changed output folder

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generate_results(MODEL_PATH, args.task, DATA_FILE, OUTPUT_FILE, args.device)