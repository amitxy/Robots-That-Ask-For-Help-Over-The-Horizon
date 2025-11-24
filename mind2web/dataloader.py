# import copy
# import glob

# import pdb
# import pickle
import logging
import json
import pathlib
import random
# import re
import sys
import pandas as pd

import lxml
import numpy as np
from datasets import load_dataset
from lxml import etree
from torch.utils.data import Dataset


from utils import log_prompt
from .data_utils.dom_utils import get_tree_repr, prune_tree


def _parse_operation(operation_field):
    """Safely parse operation field which can be dict, JSON string, or plain string.
    Returns (op, value) tuple.
    """
    if operation_field is None:
        return None, None
    
    # Already a dict
    if isinstance(operation_field, dict):
        return operation_field.get("op"), operation_field.get("value")
    
    # Try to parse as JSON string
    if isinstance(operation_field, str):
        try:
            parsed = json.loads(operation_field)
            if isinstance(parsed, dict):
                return parsed.get("op"), parsed.get("value")
        except (json.JSONDecodeError, ValueError):
            pass
        # If JSON parsing fails, treat as plain operation string
        return operation_field, None
    
    return None, None

def _extract_candidate_ids(json_txt: str):
    data = json.loads(json_txt)
    candidate_id = data.get("backend_node_id")
    return candidate_id

def format_input_generation(
    sample, candidate_ids, gt=-1, keep_html_brackets=True
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    gt = id_mapping.get(gt, -1)
    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )
    previous_k = int(sample.get("target_action_index", 0))
    if len(sample["action_reprs"]) > 0:
        for action in sample["action_reprs"][:previous_k]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"
    seq_input += (
        "What should be the next action?"
        "Please select the element to interact with, and the action to perform along with the value to type in or select. "
        "If the task cannot be completed, output None."
    )

    if gt == -1:
        seq_target = "None"
    else:
        current_action_op, current_action_value = _parse_operation(sample.get("operation"))
        seq_target = f"Element: {choices[gt][1]}\n"
        seq_target += f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices


def format_input_multichoice(
    sample, candidate_ids, gt=-1, keep_html_brackets=True
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )

    gt = id_mapping.get(str(gt), -1)
    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )
  
    fallback_chr = 'A' #chr(65 + len(candidate_ids))
    prev_actions = ''
    if len(sample["action_reprs"]) > 0:
        previous_k = int(sample.get("target_action_index", 0))
        for action in sample["action_reprs"][:previous_k]:
            prev_actions += f"{action}\n"
    else:
        prev_actions += "None\n"
    seq_input += prev_actions
    seq_input += (
        f"What should be the next action? Please select from the following choices "
        f"(If the correct action is not in the page above, please select {fallback_chr}. 'None of the above'):\n\n"
        
    )
    
    choices_str = f"A. None of the above\n"
    for idx, choice in enumerate(choices):
        # convert to ascii A, B, C, D, ...
        choices_str += f"{chr(66 + idx)}. {choice[1]}\n"  
    # seq_input += f"{fallback_chr}. None of the above\n"
    seq_input += choices_str

    if gt == -1:
        seq_target = f"{fallback_chr}."
    else:
        # gt += 1
        current_action_op, current_action_value = _parse_operation(sample.get("operation"))
        seq_target = f"{chr(66 + gt)}.\n" f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, prev_actions, choices_str




class PromptView:
    """Helper that returns a dict of the raw prompt"""
    def __init__(self, parent):
        self._parent = parent
    
    def __getitem__(self, idx:int) -> dict:

        return self._parent._get_prompt_element(idx)


class MultiChoiceDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        num_candidates=5,
        max_context_len=512,
        mode="multichoice",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_context_len = max_context_len
        self.mode = mode

        self.prompt_view = PromptView(self)

    def __len__(self):
        return len(self.data)
    
    def choices_token_ids_mapping(self):
        """Return a mapping from the candidate choices to thier token ids."""
        options = [chr(65 + i)  for i in range(self.num_candidates + 1)]

        tokens = self.tokenizer(options, add_special_tokens=False)
        option_ids = [token_id[0] for token_id in  tokens.input_ids]
        return dict(zip(options, option_ids))
    
    def _get_prompt_element(self, idx):
        """Return the raw prompt elements before tokenization
        Returns: seq_context, seq_in, seq_out
        """
        sample = self.data[idx]
        
        all_cands = sample["pos_candidates"] + sample["neg_candidates"]
        
        all_cands_sorted = sorted(
            all_cands, key=lambda c: c.get("rank", float("inf"))
        )[: self.num_candidates]

        # Ground-truth is the best-ranked positive within the top-k (if any).
        top_pos = [c for c in all_cands_sorted if c in sample["pos_candidates"]]
        if top_pos:
            # Pick the lowest-rank positive (ties broken randomly for stability).
            best_rank = min(c.get("rank", float("inf")) for c in top_pos)
            best_pos = [c for c in top_pos if c.get("rank", float("inf")) == best_rank]
            pos_candidate = random.choice(best_pos)           
            gt = _extract_candidate_ids(pos_candidate["backend_node_id"])

        else:
            gt = -1

        candidate_ids = [
            _extract_candidate_ids(c["backend_node_id"]) for c in all_cands_sorted
        ]

        if self.mode == "multichoice":
                seq_context, seq_in, seq_out, prev_actions, choices_str = format_input_multichoice(
                    sample, candidate_ids, gt
                )
                
        else:
                seq_context, seq_in, seq_out, _ = format_input_generation(
                    sample, candidate_ids, gt
                )
       
        return seq_context, seq_in, seq_out, prev_actions, choices_str
      
    def __getitem__(self, idx):
        sample = self.data[idx]
        seq_context, seq_in, seq_out, *_ = self._get_prompt_element(idx)
     
        log_prompt(sample["annotation_id"], sample["action_uid"], seq_context + seq_in, seq_out, model="flan-xl")
        
        seq_context = self.tokenizer(
            seq_context,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        )
        seq_in = self.tokenizer(
            seq_in,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_context_len,
        )
        model_input = {
            "input_ids": seq_context["input_ids"] + seq_in["input_ids"],
            "attention_mask": seq_context["attention_mask"] + seq_in["attention_mask"],
        }
        
        # seq_out = self.tokenizer(seq_out)
        model_input["labels"] = seq_out#["input_ids"]
 
        return model_input


def get_data_split(data_dir, split_file, candidate_results=None, is_train=False, cache_dir=None):
    def flatten_actions(samples):
        outputs = {
            "website": [],
            "confirmed_task": [],
            "annotation_id": [],
            "previous_actions": [],
            "action_uid": [],
            "operation": [],
            "pos_candidates": [],
            "neg_candidates": [],
            "cleaned_html": [],
        }
        num_actions = [len(actions) for actions in samples["action_reprs"]]
        for key in ["website", "confirmed_task", "annotation_id"]:
            for idx, value in enumerate(samples[key]):
                outputs[key] += [value] * num_actions[idx]
        for actions, action_reprs in zip(samples["actions"], samples["action_reprs"]):
            for a_idx, action in enumerate(actions):
                outputs["previous_actions"].append(action_reprs[:a_idx])
                for key in [
                    "action_uid",
                    "operation",
                    "pos_candidates",
                    "neg_candidates",
                    "cleaned_html",
                ]:
                    outputs[key].append(action[key])
        return outputs

    dataset = None
    if cache_dir is not None:
        dataset = load_dataset(data_dir, split=split_file, cache_dir=cache_dir)
    else:
        dataset = load_dataset(data_dir, data_files=split_file, split="all")
    
    
    # flatten_dataset = dataset.map(
    #     flatten_actions,
    #     batched=True,
    #     remove_columns=dataset.column_names,
    #     batch_size=10,
    #     num_proc=4,
    # )
    flatten_dataset = dataset
    if candidate_results is not None:
        candidate_scores = candidate_results["scores"]
        candidate_ranks = candidate_results["ranks"]

        def get_score(sample):
            sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
            for which in ["pos_candidates", "neg_candidates"]:
                candidates = sample.get(which, [])
                updated_candidates = []
                for candidate in candidates:
                    # Candidate may be a dict or a backend_node_id string
                    if isinstance(candidate, dict):
                        candidate_id = candidate.get("backend_node_id")
                        # Copy to avoid mutating potential shared references
                        candidate_copy = dict(candidate)
                    else:
                        # If it's a string or other scalar, treat as id
                        candidate_id = candidate
                        candidate_copy = {"backend_node_id": candidate_id}

                    # Safely get score and rank if available
                    score = None
                    rank = None
                    try:
                        score = candidate_scores.get(sample_id, {}).get(candidate_id)
                    except Exception:
                        score = None
                    try:
                        rank = candidate_ranks.get(sample_id, {}).get(candidate_id)
                    except Exception:
                        rank = None

                    candidate_copy["score"] = score
                    candidate_copy["rank"] = rank
                    updated_candidates.append(candidate_copy)

                # Replace the sample's candidate list with updated list
                sample[which] = updated_candidates

            return {
                "pos_candidates": sample["pos_candidates"],
                "neg_candidates": sample["neg_candidates"],
            }

        flatten_dataset = flatten_dataset.map(get_score)
    if is_train:
        flatten_dataset = flatten_dataset.filter(lambda x: len(x["pos_candidates"]) > 0)

    return flatten_dataset


def subsample_by_annotation(
    dataset,
    num_annotations=None,
    frac=None,
    seed=0,
    annotation_field="annotation_id",
):
    """
    Subsample a dataset by annotation_id, keeping all actions for each selected id.

    Args:
        dataset: Dataset with an annotation id column.
        num_annotations: Number of annotation groups to keep.
        frac: Fraction of annotation groups to keep (0-1]. Ignored if num_annotations is set.
        seed: RNG seed for deterministic sampling.
        annotation_field: Column name that identifies an annotation/task.
    """
    if num_annotations is None and frac is None:
        raise ValueError("Must provide num_annotations or frac for subsampling.")

    # Preserve original order of unique annotation ids (must be a sequence for random.sample)
    unique_annotations = list(dict.fromkeys(dataset[annotation_field]))
    total_ann = len(unique_annotations)

    if frac is not None and num_annotations is None:
        if not (0 < frac <= 1):
            raise ValueError("frac must be in (0, 1].")
        num_annotations = max(1, int(total_ann * frac))

    if num_annotations >= total_ann:
        return dataset

    rng = random.Random(seed)
    keep_ids = set(rng.sample(unique_annotations, num_annotations))

    # Filtering keeps all rows for the selected annotation ids and preserves intra-id ordering
    cal_split = dataset.filter(lambda x: x[annotation_field] in keep_ids)
    test_split = dataset.filter(lambda x: x[annotation_field] not in keep_ids)
    return cal_split, test_split

def build_split_datasets(
    split_files,
    tokenizer,
    seed=42,
    frac=0.1,
    num_candidates=5,
    max_context_len=512,
    data_dir="osunlp/Multimodal-Mind2Web",
    cache_dir=None,
    candidates_dir="candidates",
):
    """Return (cal_dict, test_dict) keyed by split name."""
    cal_dict, test_dict = {}, {}
    for split_file in split_files:
        cand_path = f"{candidates_dir}/scores_{split_file}.pkl"
        candidate_results = pd.read_pickle(cand_path)

        flattened = get_data_split(
            data_dir=data_dir,
            split_file=split_file,
            candidate_results=candidate_results,
            cache_dir=cache_dir,
        )

        cal_set, test_set = subsample_by_annotation(flattened, frac=frac, seed=seed)
        cal_dict[split_file] = MultiChoiceDataset(
            cal_set, tokenizer, num_candidates=num_candidates, max_context_len=max_context_len
        )
        test_dict[split_file] = MultiChoiceDataset(
            test_set, tokenizer, num_candidates=num_candidates, max_context_len=max_context_len
        )
    return cal_dict, test_dict
