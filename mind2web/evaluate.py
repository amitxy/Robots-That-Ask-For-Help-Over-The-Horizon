import collections
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    # Prefer package-style import if mind2web is installed as a module.
    from mind2web.dataloader import format_input_multichoice
except ImportError:  # pragma: no cover - fallback for script usage
    from dataloader import format_input_multichoice

logger = logging.getLogger(__name__)


def _postprocess_action(text: str) -> Tuple[str, str]:
    """Parse the letter choice and action string from model output."""
    text = text.strip()
    selected_option = text[0] if text else "A"
    action_match = re.search(r"Action: (CLICK|SELECT|TYPE)", text)
    action = action_match.group(1) if action_match is not None else ""
    value_match = re.search(r"Value: (.*)$", text, re.MULTILINE)
    value = value_match.group(1) if value_match is not None else ""
    return selected_option, f"{action.strip()} {value.strip()}".strip()


def _calculate_f1(pred: str, label: str) -> float:
    """Token-level F1 used by the original evaluator."""
    pred_set = set(pred.strip().split())
    label_set = set(label.strip().split())
    if len(pred_set) == 0 and len(label_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(label_set) == 0:
        return 0.0
    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _build_inputs(
    tokenizer,
    seq_context: str,
    seq_in: str,
    max_context_len: int,
) -> Dict[str, List[int]]:
    """Tokenize context and input once and return concatenated tensors."""
    ctx = tokenizer(
        seq_context,
        truncation=True,
        max_length=max_context_len,
        add_special_tokens=False,
    )
    inp = tokenizer(
        seq_in,
        add_special_tokens=True,
        truncation=True,
        max_length=max_context_len,
    )
    return {
        "input_ids": ctx["input_ids"] + inp["input_ids"],
        "attention_mask": ctx["attention_mask"] + inp["attention_mask"],
    }


class FastActionEvaluatorMultiChoice:
    """
    A more efficient reimplementation of ActionEvaluatorMultiChoice.evaluate_dataset.
    Key changes:
      - Batch generation across candidate groups.
      - Tokenize once per group and pad on device.
      - Avoid recomputing aggregate metrics inside the per-sample loop.
    """

    def __init__(self, tokenizer, max_context_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset,
        model,
        batch_size: int = 8,
        top_k: int = 50,
        output_path: Optional[str] = None,
        name: str = "default",
        template: Optional[Tuple[str, str]] = None,
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device

        all_element_acc: List[Tuple[int, str]] = []
        all_action_f1: List[Tuple[float, str]] = []
        all_step_acc: List[Tuple[int, str]] = []
        sample_to_website: Dict[str, str] = {}
        all_final_predictions: List[List[str]] = []
        all_outputs: List[List[Any]] = []

        # Log candidate recall stats once up front.
        for k in [5, 10, 20, 50]:
            recall_at_k = np.mean(
                [
                    1 if any([c["rank"] < k for c in sample["pos_candidates"]]) else 0
                    for sample in dataset.data
                ]
            )
            logger.info("Recall Cap @ %s: %s", k, recall_at_k)
        acc = np.mean(
            [
                1 if any([c["rank"] == 0 for c in sample["pos_candidates"]]) else 0
                for sample in dataset.data
            ]
        )
        logger.info("Candidate generator acc: %s", acc)

        progress = tqdm(total=len(dataset.data))
        for sample in dataset.data:
            annotation_id = sample["annotation_id"]
            sample_to_website[annotation_id] = sample["website"]

            pos_candidates = [
                c for c in sample["pos_candidates"] if c.get("rank", top_k + 1) < top_k
            ]
            pos_ids = [c["backend_node_id"] for c in pos_candidates]
            if len(pos_ids) == 0:
                all_element_acc.append([0, annotation_id])
                all_action_f1.append([0.0, annotation_id])
                all_step_acc.append([0, annotation_id])
                sample_key = f"{sample['annotation_id']}_{sample['action_uid']}"
                all_final_predictions.append([sample_key, "", ""])
                all_outputs.append([sample_key, []])
                progress.update()
                continue

            _, _, target_out, _ = format_input_multichoice(
                sample, pos_ids[:1], pos_ids[0]
            )
            _, target_action = _postprocess_action(target_out)

            neg_candidates = [
                c for c in sample["neg_candidates"] if c.get("rank", top_k + 1) < top_k
            ]
            neg_ids = [c["backend_node_id"] for c in neg_candidates]
            all_candidates: List[str] = pos_ids + neg_ids
            random.shuffle(all_candidates)

            final_prediction: Optional[Tuple[str, str]] = None
            outputs: List[Any] = []

            # Process candidate groups in batches.
            while len(all_candidates) > 1:
                group_specs = []
                group_choices = []
                while len(group_specs) < batch_size and len(all_candidates) > 1:
                    candidate_ids = all_candidates[:5]
                    all_candidates = all_candidates[5:]
                    seq_context, seq_in, _, choices = format_input_multichoice(
                        sample, candidate_ids, -1
                    )
                    if template is not None:
                        seq_context = template[0] + seq_context
                        seq_in = seq_in + template[1]
                    model_inputs = _build_inputs(
                        self.tokenizer,
                        seq_context,
                        seq_in,
                        self.max_context_len,
                    )
                    group_specs.append(model_inputs)
                    group_choices.append((candidate_ids, choices, [seq_context, seq_in]))

                if not group_specs:
                    break

                padded = self.tokenizer.pad(
                    group_specs, padding=True, return_tensors="pt"
                ).to(device)

                generated = model.generate(
                    **padded,
                    eos_token_id=model.config.eos_token_id,
                    max_new_tokens=max_new_tokens,
                )
                decoded_batch = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )

                for decoded, meta in zip(decoded_batch, group_choices):
                    candidate_ids, choices, texts = meta
                    outputs.append([candidate_ids, texts, decoded])
                    pred_element, pred_action = _postprocess_action(decoded)
                    if pred_element and pred_element[0] != "A":
                        idx = ord(pred_element[0]) - ord("B")
                        if 0 <= idx < len(choices):
                            pred_id = choices[idx][0]
                            all_candidates.append(pred_id)
                            final_prediction = (pred_id, pred_action)

            sample_key = f"{sample['annotation_id']}_{sample['action_uid']}"
            all_outputs.append([sample_key, outputs])
            if len(all_candidates) == 0 or final_prediction is None:
                all_element_acc.append([0, annotation_id])
                all_action_f1.append([0.0, annotation_id])
                all_step_acc.append([0, annotation_id])
                all_final_predictions.append([sample_key, "", ""])
            else:
                all_element_acc.append(
                    [1 if final_prediction[0] in pos_ids else 0, annotation_id]
                )
                f1 = _calculate_f1(final_prediction[1], target_action)
                all_action_f1.append([f1, annotation_id])
                all_step_acc.append(
                    [1 if f1 == 1 and all_element_acc[-1][0] == 1 else 0, annotation_id]
                )
                all_final_predictions.append(
                    [sample_key, final_prediction[0], final_prediction[1]]
                )

            progress.set_postfix(
                element_acc=np.mean([x[0] for x in all_element_acc]),
                action_f1=np.mean([x[0] for x in all_action_f1]),
            )
            progress.update()

        progress.close()

        macro_element = collections.defaultdict(list)
        macro_action = collections.defaultdict(list)
        macro_step = collections.defaultdict(list)
        for val, ann in all_element_acc:
            macro_element[ann].append(val)
        for val, ann in all_action_f1:
            macro_action[ann].append(val)
        for val, ann in all_step_acc:
            macro_step[ann].append(val)

        error_ratio = collections.defaultdict(int)
        acc_per_website = collections.defaultdict(list)
        for annotation_id, vals in macro_step.items():
            acc_per_website[sample_to_website[annotation_id]].append(np.mean(vals))
            error_count = len([v for v in vals if v == 0])
            if error_count <= 3:
                error_ratio[error_count] += 1
            else:
                error_ratio[">3"] += 1
        acc_per_website = {
            k: (np.mean(v), len(v)) for k, v in acc_per_website.items()
        }
        error_ratio = {k: v / len(macro_element) for k, v in error_ratio.items()}

        result = {
            "element_acc": np.mean([x[0] for x in all_element_acc]),
            "action_f1": np.mean([x[0] for x in all_action_f1]),
            "step_acc": np.mean([x[0] for x in all_step_acc]),
            "marco_element_acc": np.mean([np.mean(x) for x in macro_element.values()]),
            "marco_action_f1": np.mean([np.mean(x) for x in macro_action.values()]),
            "marco_step_acc": np.mean([np.mean(x) for x in macro_step.values()]),
            "error_ratio": error_ratio,
            "acc_per_website": acc_per_website,
        }

        if output_path is not None:
            with open(f"{output_path}/{name}_predictions_top{top_k}.json", "w") as f:
                json.dump(all_final_predictions, f)
            with open(f"{output_path}/{name}_results_top{top_k}.json", "w") as f:
                json.dump(result, f, indent=4)
            with open(f"{output_path}/{name}_outputs_top{top_k}.json", "w") as f:
                json.dump(all_outputs, f)

        return result
