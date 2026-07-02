#!/usr/bin/env python3
"""Assign task-subcategory labels for the RQ3 session-mode analysis."""

import os
import json
import asyncio
import logging
import backoff
from pathlib import Path
from typing import Dict, List

from openai import (
    AsyncOpenAI,
    RateLimitError,
    APITimeoutError,
    APIError,
    APIConnectionError,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# OpenAI authentication. Set OPENAI_API_KEY in the environment before running.
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or API_KEY.startswith("sk-") is False:
    raise RuntimeError(
        "Provide your key via environment variable OPENAI_API_KEY."
    )

client = AsyncOpenAI(api_key=API_KEY)

# Global constants
MODEL_NAME   = "gpt-4o"
TEMPERATURE  = 0.0
REQ_TIMEOUT  = 30         # seconds
MAX_PARALLEL = 8          # simultaneous HTTP calls
BATCH_SIZE   = 50         # records per wave
CACHE_FILE   = ".llm_label_cache.json"

_SEM = asyncio.Semaphore(MAX_PARALLEL)

# Label definitions and prompt
LABELS: Dict[str, Dict[str, str]] = {
    "summarization": {
        "S-API":   "API contract, single concern",
        "S-PARAM": "Parameter-driven docs",
        "S-RET/EXC": "Return-or-Exception semantics",
        "S-CONC":  "Concurrency / async behaviour",
        "S-META":  "Annotations / lifecycle info",
    },
    "translation": {
        "T-PRIM": "Trivial primitive / operator",
        "T-GEN":  "Generics & Collections",
        "T-OVR":  "Inheritance / override",
        "T-EXC":  "Exception / error handling",
        "T-CTRL": "Multi-statement control flow",
    },
    "completion": {
        "C-DECL": "Structural boilerplate",
        "C-SIMP": "Linear method body",
        "C-CTRL": "Complex control flow",
        "C-ASY":  "Concurrency / lambda",
        "C-TEST": "JUnit / test fixture",
    },
}

PROMPT_TMPL = """You are an expert software-engineering researcher.
Task = {task}.

We defined the following mutually-exclusive labels:
{table}

Given the JSON record based on its input and ground truth:

{record}

Return **exactly ONE** label ID from the table (e.g., "{example}").
If no label is suitable, return "UNKNOWN"."""

def make_prompt(task: str, rec: dict) -> str:
    table = "\n".join(f"- **{k}**: {v}" for k, v in LABELS[task].items())
    eg_id = next(iter(LABELS[task]))
    return PROMPT_TMPL.format(
        task=task.upper(),
        table=table,
        record=json.dumps(rec, ensure_ascii=False, indent=2),
        example=eg_id,
    )

# Resilient OpenAI call with exponential backoff and jitter
TRANSIENT_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIError,
    APIConnectionError,
)

def _log_backoff(details):
    logging.warning(
        "API error %s; retry in %.1fs (attempt %d)",
        details["exception"].__class__.__name__,
        details["wait"],
        details["tries"],
    )

@backoff.on_exception(
    backoff.expo,                 # base delay
    TRANSIENT_ERRORS,
    max_time=300,
    jitter=backoff.full_jitter,   # adds randomness
    on_backoff=_log_backoff,
)
async def label_async(task: str, rec: dict) -> str:
    async with _SEM:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[{
                "role": "user",
                "content": make_prompt(task, {
                    "input": rec["input"],
                    "ground_truth": rec.get("ground_truth"),
                }),
            }],
            timeout=REQ_TIMEOUT,
        )
    return response.choices[0].message.content.strip()

# Cache utilities
def _load_cache() -> Dict[str, List[str]]:
    if Path(CACHE_FILE).exists():
        return json.loads(Path(CACHE_FILE).read_text())
    return {}

def _save_cache(cache: dict):
    Path(CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False, indent=2))

LABELS_CACHE = _load_cache()

# Phase 1: generate labels
async def generate_labels_for_task(task: str, file_path: Path):
    if task in LABELS_CACHE:
        logging.info("Using cached labels for '%s' (%d)", task, len(LABELS_CACHE[task]))
        return

    logging.info("Generating labels for '%s' from %s", task, file_path.name)
    records = json.loads(file_path.read_text())
    labels: List[str] = []

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        labels.extend(await asyncio.gather(*(label_async(task, r) for r in batch)))
        _save_cache({**LABELS_CACHE, task: labels})
        logging.info("  %d/%d done", len(labels), len(records))

    LABELS_CACHE[task] = labels
    _save_cache(LABELS_CACHE)
    logging.info("Cached %d labels for '%s'", len(labels), task)

# Phase 2: apply labels to result files
def apply_cached_labels(path: Path, task: str, model_tag: str):
    if task not in LABELS_CACHE:
        logging.error("No cache for task '%s'; skip %s", task, path.name)
        return
    labels = LABELS_CACHE[task]
    data   = json.loads(path.read_text())

    if len(data) != len(labels):
        logging.warning("Length mismatch (%s); skipped", path.name)
        return

    for rec, lab in zip(data, labels):
        rec["group_label"] = lab

    out = path.with_name(f"{path.stem}_{model_tag}_with_groups.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logging.info("%s -> %s", path.name, out.name)

# Main driver
TASK_FROM_STEM = {
    "_summarization": "summarization",
    "_translation":   "translation",
    "_completion":    "completion",
}
MODEL_FOLDERS = ["deepseek_results", "llama_results", "starcoder_results"]

async def main(root: str = "."):
    root_path = Path(root)
    logging.info("Session-mode label assignment starting")

    # Use the first available result folder to generate one canonical label set
    # per task, then apply those labels to all model result files.
    canonical_folder = next(
        (root_path / f for f in MODEL_FOLDERS if (root_path / f).exists()),
        None,
    )
    if not canonical_folder:
        logging.error("No model result folders found under %s; abort", root)
        return

    # Phase 1
    jobs = []
    for stem, task in TASK_FROM_STEM.items():
        canon_file = next(canonical_folder.glob(f"generated_results_*{stem}*json"), None)
        if canon_file:
            jobs.append(generate_labels_for_task(task, canon_file))
        else:
            logging.warning("No canonical file for task '%s' in %s", task, canonical_folder)
    if jobs:
        await asyncio.gather(*jobs)

    # Phase 2
    logging.info("Applying cached labels")
    for model_tag in MODEL_FOLDERS:
        folder = root_path / model_tag
        if not folder.exists():
            continue
        for fp in folder.glob("generated_results_*json"):
            hit = next((k for k in TASK_FROM_STEM if k in fp.stem), None)
            if hit:
                apply_cached_labels(fp, TASK_FROM_STEM[hit], model_tag)

    logging.info("All done")

if __name__ == "__main__":
    asyncio.run(main("./"))
