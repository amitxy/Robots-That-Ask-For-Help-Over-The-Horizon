
import os
import logging
from datetime import datetime
import importlib
import sys

SLURM_PATH = '/home/yandex/MLWG2025/amitr5'
CACHE_DIR = f'{SLURM_PATH}/tmp/hf_cache' 


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
