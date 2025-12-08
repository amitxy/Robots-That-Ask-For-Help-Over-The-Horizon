import pandas as pd
from utils.helpers import _ACTION_RE, _VALUE_RE
from typing import Dict, Any, Optional, Tuple

def _postprocess_action(text: str) -> Tuple[str, str]:
    """Replicates the paper's postprocess"""
    text = text.strip() if isinstance(text, str) else ""
    selected_option = text[0] if text else "A"
    action = _ACTION_RE.search(text)
    action = action.group(1) if action is not None else ""
    value = _VALUE_RE.search(text)
    value = value.group(1) if value is not None else ""
    return selected_option, f"{action.strip()} {value.strip()}".strip()


def _calculate_f1(pred: str, label: str) -> float:
    """Replicates the paper's f1 calculation"""
    pred_set = set(pred.strip().split())
    label_set = set(label.strip().split())
    if len(pred_set) == 0 and len(label_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(label_set) == 0:
        return 0.0
    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_mind2web_metrics(
    df: pd.DataFrame,
    split_value: Optional[int] = None,
    split_col: str = "test_split",
    as_percent: bool = True,
) -> Dict[str, float]:
    """
    Compute metrics in the same style as ActionEvaluatorMultiChoice:
      - Macro over tasks (annotation_id) for Ele. Acc, Op. F1, Step SR
      - SR = fraction of tasks with all steps correct
    Expects columns: annotation_id, pred_label, pred_action, pred_value, label_text.
    """
    data = df.copy()
    if split_value is not None and split_col in data.columns:
        data = data[data[split_col] == split_value].copy()
    if data.empty:
        raise ValueError("No rows left after filtering; cannot compute metrics.")

    # Gold element/action from label_text
    gold_elem_action = data["target_text"].apply(_postprocess_action)
    gold_labels = gold_elem_action.apply(lambda x: x[0])
    gold_actions = gold_elem_action.apply(lambda x: x[1])

    # Pred element/action/value
    data["pred_label"] = data["pred_label"].fillna("A")
    pred_action_value = (
        data["pred_action"].fillna("").astype(str).str.strip() + " " +
        data["pred_value"].fillna("").astype(str).str.strip()
    ).str.strip()

    data["element_correct"] = data["pred_label"].astype(str) == gold_labels.astype(str)
    data["action_f1"] = [
        _calculate_f1(p, g) for p, g in zip(pred_action_value, gold_actions)
    ]
    data["step_correct"] = data["element_correct"] & (data["action_f1"] == 1.0)

    def maybe_pct(x: float) -> float:
        return 100.0 * x if as_percent else x

    # Macro over tasks
    element_macro = data.groupby("annotation_id")["element_correct"].mean().mean()
    action_macro = data.groupby("annotation_id")["action_f1"].mean().mean()
    step_macro = data.groupby("annotation_id")["step_correct"].mean().mean()
    task_success = data.groupby("annotation_id")["step_correct"].all().mean()

    return {
        "Ele. Acc": round(maybe_pct(element_macro), 2),
        "Op. F1": round(maybe_pct(action_macro), 2),
        "Step SR": round(maybe_pct(step_macro), 2),
        "SR": round(maybe_pct(task_success), 2),
    }
