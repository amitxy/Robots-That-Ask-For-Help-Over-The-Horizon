
import os
import logging
from datetime import datetime
import importlib
import sys
import pandas as pd
import torch
import re 
from typing import Dict, Any, Optional, Tuple

SLURM_PATH = '/home/yandex/MLWG2025/amitr5'
CACHE_DIR = f'{SLURM_PATH}/tmp/hf_cache' 


_LABEL_RE = re.compile(r"^\s*([A-F])\.", re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(CLICK|SELECT|TYPE)", re.IGNORECASE)
_VALUE_RE  = re.compile(r"Value:\s*(.*)$", re.IGNORECASE | re.MULTILINE)




def setup_slurm_env():
    if SLURM_PATH in os.getcwd():
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.environ["PIP_PATH"] = f"{SLURM_PATH}/BaryGNN/anaconda3/envs/conf/bin/pip"
        os.environ["TEMP_DIR"] = CACHE_DIR
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
        # Hugging Face uses HUGGINGFACE_HUB_CACHE (HF_HUB_CACHE is ignored)
        os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
        os.environ["TMPDIR"] = CACHE_DIR
        os.environ["XDG_CACHE_HOME"] = CACHE_DIR
        # os.environ["TOKENIZERS_PARALLELISM"] = "false"

def reload(*module_names):
    for name in module_names:
        if name in sys.modules:
             importlib.reload(sys.modules[name])
             print(f"{name}- reloaded")
        else:
            print(f"{name}- NOT FOUND!")

def setup_logger(name, level=logging.INFO):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger(name)
    
    # # Check if handler already exists
    # if logger.hasHandlers():
    #     return logger

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    date_str = datetime.now().strftime("%d-%m-%Y")
    handler = logging.FileHandler(f"logs/{name}-{date_str}.log")
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    
    #Prevent logs from bubbling up to the console/root logger
    logger.propagate = False
    return logger

def log_prompt(anonaction_id, action_id, prompt, response, model="flan-xl"):
    entry = {
        "type": "prompt",
        "anonaction_id": anonaction_id,
        "action_id": action_id,
        "prompt": prompt,
        "response": response,
        "model": model,
    }
    logger.info(entry)

def log_response(annotation_id, action_id, response):
    logger.info({"type": "response", "annotation_id": annotation_id, "action_uid":action_id, "response": response})

# Initialize loggers
logger = setup_logger(name="logger")


def add_eval_columns(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Add derived evaluation columns.

    Expects columns:
      - pred_label: predicted letter (e.g., 'A')
      - label: true letter
      - choices_probs: dict mapping letter -> probability

    Adds:
      - correct, true_prob, pred_prob
      - pred_set, pred_set_size if threshold is provided (include labels where 1 - prob <= threshold)
    """
    df = df.copy()
    
    for col, suffix in (("target_text", "target"), ("output_text","pred")):
        parsed = df[col].apply(lambda t: parse_output(t))
        df[f"{suffix}_label"]= parsed.apply(lambda x: x[0])
        df[f"{suffix}_action"] = parsed.apply(lambda x: x[1])
        df[f"{suffix}_value"] = parsed.apply(lambda x: x[2])

    df["correct"] = df["pred_label"] == df["target_label"]
    # df["true_prob"] = df.apply(lambda r: r["choices_probs"].get(r["target_label"], 0.0),axis=1)
    # df["pred_prob"] = df.apply(lambda r: r["choices_probs"].get(r["pred_label"], 0.0), axis=1)


    if threshold is not None:
        df["pred_set"] = df['choices_probs'].apply(lambda row: [ label for label, prob in row.items() if 1 - prob <= threshold])
        df["pred_set_size"] = df["pred_set"].apply(len)

    return df


def softmax_with_temperature_from_probs(choices_probs: dict, temperature: float = 6.0) -> dict:
    """
    Given a dict of choice->probabilities, convert to logits (log-prob) and
    re-apply softmax with the provided temperature. Returns a new dict of probs.
    """
    if not isinstance(choices_probs, dict) or len(choices_probs) == 0:
        return {}
    # convert to a fixed order for stability
    items = sorted(choices_probs.items())
    labels, probs = zip(*items)
    probs = torch.tensor(probs, dtype=torch.float32)
    logits = torch.log(probs + 1e-12)  # avoid log(0)
    scaled = logits / temperature
    new_probs = torch.softmax(scaled, dim=-1).cpu().tolist()
    return dict(zip(labels, new_probs))

def choices_logits(logits, choice2token_id) -> dict:
    """
    Extract logits for the provided choice->token_id mapping.
    """
    choices_ids = torch.tensor(list(choice2token_id.values()), device='cpu', dtype=torch.long)
    choice2logit = dict(zip(choice2token_id.keys(), logits[choices_ids].cpu().tolist()))
    return choice2logit

def choices_softmax(label_logits: dict, temperature: float = 1.0) -> dict:
    """
    Given a dict of label -> logit, return a dict of label -> softmax prob.
    Supports optional temperature scaling.
    """
    if not isinstance(label_logits, dict) or len(label_logits) == 0:
        return {}
    items = sorted(label_logits.items())
    labels, vals = zip(*items)
    t = torch.tensor(vals, dtype=torch.float32)
    probs = torch.softmax(t / float(temperature), dim=-1).tolist()
    return dict(zip(labels, probs))

def parse_output(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    letter_match = _LABEL_RE.search(text)
    letter = letter_match.group(1).upper() if letter_match else None

    action_match = _ACTION_RE.search(text)
    action = action_match.group(1).upper() if action_match else None

    value_match = _VALUE_RE.search(text)
    value = value_match.group(1).strip() if value_match else None
    if value == "":
        value = None

    return letter, action, value
