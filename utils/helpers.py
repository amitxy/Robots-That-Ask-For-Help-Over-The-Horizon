
import os
import logging
from datetime import datetime
import importlib
import sys
from pathlib import Path
import pandas as pd
import torch
import re 
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

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

def _logits_to_probs(logits_dict, logits_temp: float = 1) -> dict:
        if not isinstance(logits_dict, dict) or len(logits_dict) == 0:
            return {}
        items = sorted(logits_dict.items())
        labels, vals = zip(*items)
        t = torch.tensor(vals, dtype=torch.float32)
        probs = torch.softmax(t / float(logits_temp), dim=-1).tolist()
        return dict(zip(labels, probs))

def add_eval_columns(df: pd.DataFrame, threshold: float = None, logits_temp: float = 1, to_save: bool = False, save_path: str = None, add_noise=False) -> pd.DataFrame:
    """
    Add derived evaluation columns.

    Expects columns:
      - pred_label: predicted letter (e.g., 'A')
      - label: true letter
      - choices_probs: dict mapping letter -> probability

    Adds:
      - correct, true_prob, pred_prob
      - pred_set, pred_set_size if threshold is provided (include labels where 1 - prob <= threshold)
      - choices_probs from choices_logits if logits_temp is provided (softmax with temperature)
    """
    df = df.copy()

    df['step_idx'] = df.groupby('annotation_id').cumcount()
    df["choices_probs"] = df["choices_logits"].apply(lambda x: _logits_to_probs(x, logits_temp=logits_temp))

    col = "output_text" if "output_text" in df.columns else "pred_text"
    for col, suffix in (("target_text", "target"), (col,"pred")):
        parsed = df[col].apply(lambda t: parse_output(t))
        df[f"{suffix}_label"]= parsed.apply(lambda x: x[0])
        df[f"{suffix}_action"] = parsed.apply(lambda x: x[1])
        df[f"{suffix}_value"] = parsed.apply(lambda x: x[2])

    df["correct"] = df["pred_label"] == df["target_label"]
    df["target_prob"] = df.apply(lambda r: r["choices_probs"].get(r["target_label"], 0.0),axis=1)
    df["pred_prob"] = df.apply(lambda r: r["choices_probs"].get(r["pred_label"], 0.0), axis=1)

    
    if threshold is not None:
        epsilon = 1e-6
        noise = (
            # np.random.default_rng(42).uniform(0, epsilon, size=len(df))
            np.random.uniform(0, epsilon, size=len(df))
            if add_noise
            else np.zeros(len(df))
        )
        df["pred_set"] = df.apply(
            lambda r: [
                label
                for label, prob in r["choices_probs"].items()
                if 1 - prob < threshold
            ],
            axis=1,
        )
        df["pred_set_size"] = df["pred_set"].apply(len)

    if to_save and save_path:
        full_path = Path("results") / save_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(full_path)
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


def filter_to_calibration_actions(
    full_df,
    cal_dict,
):
    """
    Keep only rows in full_df whose (action_uid)
    appear in cal_dict.
    """
    filtered_dfs = []
    for split in cal_dict.keys():
        anotations = cal_dict[split].action_uids()
        df = full_df[full_df['action_uid'].isin(anotations)]
        filtered_dfs.append(df)

    all_filtered = pd.concat(filtered_dfs)
    return all_filtered

@dataclass
class LambdaResult:
    lam: float
    risk_hat: float         # empirical risky-A fraction of risky annotations
    acc: float              # overall accuracy (for tie-breaking)
    recall: float = 0.0     # recall for class A
   
def tune_lambda_group_risk(
    cal_df: pd.DataFrame,
    y_cal: pd.Series,
    label_A: int = 0,
    alpha: float = 0.01,
    lambda_grid: np.ndarray | None = None,
    shrinkage_type: str = 'linear'
) -> tuple[float, Dict[float, LambdaResult]]:
    """
    Tune lambda so that the per-annotation probability of ever predicting A
    on a non-A instance is controlled.

    For each annotation g, we define a boolean event:
        risky_A_per_ann[g] = 1  iff  ∃ i in g : y_pred_i = A and y_i != A

    We then build an upper confidence bound on the fraction of risky annotations
    using the simple pseudo-count rule
        risk_ucb = (sum_g risky_A_per_ann[g] + B) / (G + 1),  B = 1

    and choose the smallest lambda whose UCB is <= alpha.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 5000)

    lambda_grid = np.asarray(lambda_grid, dtype=float)

    # 1) Precompute logits (N, K) and labels (N,)
    class_order = ["A", "B", "C", "D", "E", "F"]
    logits_list = cal_df["choices_logits"].apply(
        lambda d: [d[c] for c in class_order]
    )
    logits = np.asarray(logits_list.tolist(), dtype=float)       # (N, K)
    y_cal_int = y_cal.to_numpy(dtype=int)                        # (N,)
    n_cal, K = logits.shape

    # 2) Group membership by annotation_id
    ann_ids, ann_inverse = np.unique(
        cal_df["annotation_id"].to_numpy(), return_inverse=True
    )
    G = len(ann_ids)                                             # #annotations
    group_masks = [(ann_inverse == g) for g in range(G)]

    # # 3) Convert logits to probabilities (temperature 6 as before)
    # probs = torch.softmax(torch.tensor(logits) / 6.0, dim=1).numpy()

    results: Dict[float, LambdaResult] = {}
    results_cov: Dict[float, LambdaResult] = {}

    for lam in lambda_grid:
        # 4) Apply per-lambda penalty only to A (column 0)
        lamb_dot = np.ones_like(logits)
        # lamb_dot[:, label_A] = 1.0 - lam
        
        # penalized = logits * lamb_dot  
        penalized = logits.copy()
        if shrinkage_type == 'linear':                      # (N, K)
            penalized[:, label_A] = logits[:, label_A] - np.abs(logits[:, label_A]) * lam
        
        elif shrinkage_type == 'constant':
            penalized[:, label_A] = logits[:, label_A] - lam

        elif shrinkage_type == 'CRC':
            pass
        else:
            raise ValueError(f"Unknown shrinkage_type: {shrinkage_type}")
        
        # 5) Point predictions
        y_pred = np.argmax(penalized, axis=1)                     # (N,)

        # 6) Per-annotation risky event
        risky_A_per_ann = np.zeros(G, dtype=bool)
        # risky_coverage = np.zeros(G, dtype=bool)
        for g, mask in enumerate(group_masks):
            if not mask.any():
                # Should not happen, but be robust
                continue
            yp_g = y_pred[mask]
            yt_g = y_cal_int[mask]
            # risky_A_per_ann[g] = np.any((yp_g == label_A) & (yt_g != label_A))
            
            if shrinkage_type != 'CRC':
                risky_A_per_ann[g] = np.sum((yp_g == label_A) & (yt_g != label_A))/ max(np.sum(yt_g != label_A),1)
                

            else:
                a_logits = penalized[mask, label_A]          # (n_actions_in_ann,)
                a_in_set = a_logits > lam                      # bool mask
                neg_mask = yt_g != label_A                                  # bool mask for negatives
                fp = (a_in_set & neg_mask).sum()
                denom = max(neg_mask.sum(), 1)
                risky_A_per_ann[g] = fp / denom




            # action_idx_with_min_prob = np.argmin(penalized[mask, yt_g]) # The index of the action that contains the smallest true prob in the annotation
            # min_prob_idx = np.where(mask)[0][action_idx_with_min_prob]
            # row = penalized_probs[min_prob_idx]             #  penalized_probs: shape [N_actions, n_classes]
            # pred_set = np.flatnonzero(1 - row < 1- 1/(1 + lam))       # indices of labels above threshold
            # y_min_t = row.argmin()                            # label with smallest prob on that row
            # risky_coverage[g] = y_min_t not in pred_set
          
     
            

        risk_hat = risky_A_per_ann.mean()
        # risk_cov_hat = risky_coverage.mean()

         # 6) Global mean FPR for "A": P( predict A | true != A )
        # is_false_A = (y_pred == label_A) & (y_cal_int != label_A)
        # denom = np.sum(y_cal_int != label_A)
        # risk_hat = float(is_false_A.sum() / max(denom, 1))

        
        
        # 7) Accuracy over samples (for reference / tie-breaking)
        acc = float((y_pred == y_cal_int).mean())
        recall_A = float(((y_pred == label_A) & (y_cal_int == label_A)).sum() / max(np.sum(y_cal_int == label_A),1))

        # 8) Simple UCB with pseudo-count B = 1
        B = 1.0

        results[float(lam)] = LambdaResult(
            lam=float(lam),
            risk_hat=risk_hat,
            avg_set_size=acc,
            # recall = recall_A
        )

        # results_cov[float(lam)] = LambdaResult(
        #     lam=float(lam),
        #     risk_hat=risk_cov_hat,
        #     acc=acc,
        #     recall = recall_A
        # )

    # 9) Feasible lambdas with UCB <= alpha
    # feasible = [res for res in results.values() if res.risk_ucb <= alpha]
    feasible = [res for res in results.values() if res.risk_hat <= (alpha*(G+1) - B)/G]

    # # feasible_cov = [res for res in results_cov.values() if res.risk_hat <= (0.2*(G+1) - B)/G]

    # if feasible and feasible_cov:
    #     best_cov = min(feasible_cov, key=lambda r: r.lam)
    #     best = min(feasible, key=lambda r: r.lam)
    #     print(
    #         f"cov_lambda={best_cov.lam:.4f} (cov_risk={best_cov.risk_hat:.4f})"
    #         f"FPR lambda={best.lam:.4f} (FPR_risk={best.risk_hat:.4f})"
    #     )
    #     return max(best.lam, best_cov.lam), results
    # else:
    #     feasible = None

   
    if not feasible:
        best_candidate = min(results.values(), key=lambda r: r.risk_hat)
        print(
            "No lambda in lambda_grid satisfies the risk constraint "
            f"(UCB <= alpha). Best candidate (lambd={best_candidate.lam:.4f}) had UCB="
            f"{best_candidate.risk_hat:.4f}>{(alpha*(G+1) - B)/G}"
        )
        return 1.0, results

    # Smallest feasible lambda
    best = min(feasible, key=lambda r: r.lam)
    # print(
    #     f"Chosen lambda={best.lam:.4f} with risk_hat={best.risk_hat:.4f} "
    #     f"and accuracy={best.acc:.4f}"
    # )
    return best.lam, results


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """
    Finite-sample conformal quantile with the (n+1) correction.
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        raise ValueError("Empty values for conformal quantile.")
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(1.0, max(0.0, q_level))
    return np.quantile(vals, q=q_level, method='higher')


def step_and_global_thresholds(
    cal_df: pd.DataFrame,
    alpha: float = 0.1,
    score_col: str = "raw_score",
    ann_col: str = "annotation_id",
    step_col: str = "step_idx",
) -> tuple[float, dict[int, float]]:
    """
    Compute:
      - global threshold from per-episode max scores
      - step-wise thresholds from scores at each step index
    """
    df = cal_df.copy()
    if step_col not in df.columns:
        df[step_col] = df.groupby(ann_col).cumcount()

    # Global: max score per episode
    global_scores = df.groupby(ann_col)[score_col].max().to_numpy(dtype=float)
    global_thr = conformal_quantile(global_scores, alpha=alpha)

    # Step-wise: quantile per step index across episodes
    step_thresholds = {}
    for step, grp in df.groupby(step_col):
        step_scores = grp[score_col].to_numpy(dtype=float)
        step_thresholds[int(step)] = conformal_quantile(step_scores, alpha=alpha)

    return global_thr, step_thresholds


def tune_lambda_weighted_crc_CHAT(
    cal_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    alpha: float = 0.1,
    lambda_grid: np.ndarray | None = None,
    score_col: str = "raw_score",
    ann_col: str = "annotation_id",
    step_col: str = "step_idx",
    logits_temp: float = 1.0,
) -> tuple[float, dict]:
    """
    Tune λ by computing conformal thresholds on cal_df and selecting the λ
    that minimizes mean prediction set size on val_df (if provided).
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 101)

    # Ensure step indices and probabilities exist
    cal_df_proc = add_eval_columns(cal_df, threshold=None, logits_temp=logits_temp, to_save=False, add_noise=False)
    global_thr, step_thresholds = step_and_global_thresholds(
        cal_df_proc, alpha=alpha, score_col=score_col, ann_col=ann_col, step_col=step_col
    )

    val_df_proc = None
    if val_df is not None:
        val_df_proc = add_eval_columns(val_df, threshold=None, logits_temp=logits_temp, to_save=False, add_noise=False)
        if step_col not in val_df_proc.columns:
            val_df_proc[step_col] = val_df_proc.groupby(ann_col).cumcount()

    results = {}
    best_lam = None
    best_metric = float("inf")

    for lam in lambda_grid:
        combined_step_thresholds = {
            step: (1.0 - lam) * thr + lam * global_thr
            for step, thr in step_thresholds.items()
        }
        metric = None
        if val_df_proc is not None:
            def _row_pred_set_size(row):
                step_thr = combined_step_thresholds.get(int(row[step_col]), global_thr)
                return sum(1 for prob in row["choices_probs"].values() if 1.0 - prob < step_thr)

            # val_df['coverd'] = val_df.apply(lambda)
            metric = float(val_df_proc.apply(_row_pred_set_size, axis=1).mean())
            if metric < best_metric:
                best_metric = metric
                best_lam = float(lam)
        results[float(lam)] = {
            "global_threshold": float(global_thr),
            "step_thresholds": combined_step_thresholds,
            "metric": metric,
        }

    if best_lam is None:
        best_lam = float(lambda_grid[0])
    return best_lam, results


def best_lambda_from_df(
    cal_df: dict,
    alpha: float = 0.1,
    lambda_grid: np.ndarray | None = None,
    label_map: dict | None = None,
    shrinkage_type: str = 'linear'
) -> tuple[float, dict]:
    """
    Filter df to calibration actions and compute best lambda (lambda_hat)
    using tune_lambda_group_risk.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 1000)

    if label_map is None:
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    # 2) make sure evaluation columns exist (choices_logits, target_label, etc.)
    cal_df_proc = add_eval_columns(cal_df)

    # 3) build y_cal
    y_cal = cal_df_proc["target_label"].map(label_map)

    # 4) run lambda tuning
    best_lambda, results = tune_lambda_group_risk(
        cal_df=cal_df_proc,
        y_cal=y_cal,
        alpha=alpha,
        lambda_grid=lambda_grid,
        shrinkage_type=shrinkage_type
    )
    return best_lambda, results

import numpy as np
import pandas as pd
from typing import Dict, Tuple

class LambdaResult:
    def __init__(self, lam, risk_hat, avg_set_size):
        self.lam = lam
        self.risk_hat = risk_hat
        self.avg_set_size = avg_set_size

def tune_lambda_weighted_crc(
    est_df: pd.DataFrame,   # Used to LEARN the q_step and q_global
    cal_df: pd.DataFrame,   # Used to TUNE lambda (risk control)
    alpha: float = 0.1,
    lambda_grid: np.ndarray | None = None,
    score_col: str = "raw_score",
    ann_col: str = "annotation_id",
    step_col: str = "step_idx"
) -> Tuple[float, Dict[float, LambdaResult]]:
    """
    Tune lambda for Single Risk Control (Episode-level validity) using CRC UCB.
    
    Logic:
    1. Learn step-wise and global thresholds from est_df.
    2. For each lambda, combine them: T_t = (1-λ)*q_t + λ*q_global.
    3. On cal_df, calculate Episode Risk: 1 if ANY step in episode is miscovered.
    4. Select smallest lambda where UCB(Risk) <= alpha.
    """
    
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 101)
    
    # --- 1. Learn the "Shape" from Estimation Set (est_df) ---
    # We calculate the raw quantiles here. 
    # Note: We use alpha to find the quantile, but the final validity depends on lambda.
   
    # Group by step to find q_step
    # step_quantiles = est_df.groupby(step_col)[score_col].quantile(1 - alpha).to_dict()
    # Calculate global max score per episode to find q_global
    # episode_max_scores = est_df.groupby(ann_col)[score_col].max()
    # global_quantile = episode_max_scores.quantile(1 - alpha)

    global_quantile , step_quantiles = step_and_global_thresholds(est_df, alpha, score_col, ann_col, step_col)
    # --- 2. Prepare Calibration Data (cal_df) ---
    # We pre-compute the relevant columns to speed up the loop
    # We need: (annotation_id, step_idx, score)
    # We group by annotation_id to handle episode-level loss
    ann_ids = cal_df[ann_col].unique()
    G = len(ann_ids)  # Number of episodes
    
    # Create a mapping of episode_id -> list of (step, score)
    # This avoids repeated pandas filtering inside the loop
    cal_data = []
    for ann in ann_ids:
        rows = cal_df[cal_df[ann_col] == ann]
        # Store as list of tuples: [(step, score), (step, score)...]
        cal_data.append(list(zip(rows[step_col].values, rows[score_col].values)))

    results: Dict[float, LambdaResult] = {}
    
    # --- 3. Iterate Grid ---
    for lam in lambda_grid:
        lam = float(lam)
        
        # Calculate the specific threshold map for this lambda
        # Optimization: Pre-calculate map for known steps
        current_thresholds = {
            s: (1.0 - lam) * q + lam * global_quantile 
            for s, q in step_quantiles.items()
        }


        episode_losses = np.zeros(G)
        total_set_size_proxy = 0.0 # Just for tracking, not optimization
        
        # Calculate Loss per Episode
        for i, episode_steps in enumerate(cal_data):
            # episode_steps is a list of (step, score)
            miscovered = False
            
            for step, score in episode_steps:
                # Get threshold (fallback to global if step is new/unknown)
                thresh = current_thresholds.get(step, global_quantile)
                thresh = min(thresh, global_quantile)
                
                # Check Miscoverage
                if score > thresh:
                    miscovered = True
                    # We can break early for Risk calculation, 
                    # but if you want to track avg_set_size, you might need to continue.
                    # For speed, we break here.
                    break 
            
            if miscovered:
                episode_losses[i] = 1.0
            else:
                episode_losses[i] = 0.0
                
        # --- 4. CRC Upper Bound Calculation ---
        # Risk = (Sum(Losses) + B) / (n + 1)
        risk_hat = episode_losses.mean()
        B = 1.0
        risk_ucb = (episode_losses.sum() + B) / (G + 1)
        
        results[lam] = LambdaResult(
            lam=lam, 
            risk_hat=risk_ucb,  # We store the UCB to compare with alpha
            avg_set_size=0.0    # Placeholder or implement set size logic if needed
        )

    # --- 5. Selection ---
    # Find all lambdas where Risk UCB <= alpha
    feasible = [res for res in results.values() if res.risk_hat <= alpha]
    
    if not feasible:
        best_candidate = min(results.values(), key=lambda r: r.risk_hat)
        print(f"Warning: No lambda satisfied risk. Best risk: {best_candidate.risk_hat:.4f}")
        return 1.0, results # Default to most conservative

    # We generally want the SMALLEST lambda that is valid 
    # (closest to local step accuracy, assuming it is more efficient)
    best = min(feasible, key=lambda r: r.lam)
    
    return best.lam, results


def fwer_step_global(
    df: pd.DataFrame,
    risk_level: float,
    thresh_alpha: float,
    seed: int,
    temp: float = 1,
    lambda_grid: np.ndarray | None = None,
    shrinkage_type: str = "constant",
    test_df: pd.DataFrame | None = None,
    cal_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Wrap the step/global threshold procedure with the same signature as fwer.
    """
    df = df.copy()
    df["raw_score"] = 1.0 - df["target_prob"]
    df = df.sort_values(["annotation_id", "step_idx"])
    df["s_running_max"] = df.groupby("annotation_id")["raw_score"].cummax()

    def _split_to_cal_penalty_stratified(data: pd.DataFrame, frac: float, seed: int, stratify_col: str):
        ids = data["annotation_id"].dropna().unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(ids)
        cut = int(len(ids) * frac)
        cal_ids = set(ids[:cut])
        pen_ids = set(ids[cut:])
        cal_split = data[data["annotation_id"].isin(cal_ids)].reset_index(drop=True)
        pen_split = data[data["annotation_id"].isin(pen_ids)].reset_index(drop=True)
        return cal_split, pen_split

    if test_df is None or cal_df is None:
        test_df, cal_df = _split_to_cal_penalty_stratified(df, frac=0.5, seed=seed, stratify_col="step_idx")
    est_df, cal_df = _split_to_cal_penalty_stratified(df, frac=0.5, seed=seed, stratify_col="step_idx")

    x = tune_lambda_weighted_crc(est_df, cal_df, score_col="raw_score")
    lamb = x[0]
    glob_q, step_q = step_and_global_thresholds(est_df)

    temp_df = []
    running_max = {}

    for step in sorted(test_df["step_idx"].unique()):
        sub_df = test_df[test_df["step_idx"] == step]
        threshold = step_q[step] * (1 - lamb) + glob_q * lamb
        sub_result_df = sub_df.copy()
        sub_result_df["threshold"] = threshold
        sub_result_df["pred_set"] = sub_result_df.apply(
            lambda r: [
                label
                for label, prob in r["choices_probs"].items()
                if max(running_max.get(r["annotation_id"], -np.inf), (1 - prob)) < threshold
            ],
            axis=1,
        )
        sub_result_df["pred_set_size"] = sub_result_df["pred_set"].apply(len)
        temp_df.append(sub_result_df)

    result_df = pd.concat(temp_df)
    result_df["lambda"] = lamb
    result_df["covered"] = result_df.apply(lambda row: row["target_label"] in row["pred_set"], axis=1)
    result_df["correct"] = result_df.apply(lambda row: row["pred_label"] == row["target_label"], axis=1)
    all_covered = result_df.groupby("annotation_id")["covered"].all().mean()

    print(
        result_df["covered"].mean(),
        all_covered,
        result_df["pred_set_size"].mean(),
        (result_df["pred_set_size"] > 1).mean(),
    )
    return result_df
