# import copy
# import glob

# import pdb
# import pickle
import logging
import json
import pathlib
import random
from string import Template
# import re
import sys
import pandas as pd
from PIL import Image
import lxml
import numpy as np
from datasets import load_dataset
from lxml import etree
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import log_prompt
from .data_utils.dom_utils import get_tree_repr, prune_tree

choice2token_id = {'A': 71, 'B': 272, 'C': 205, 'D': 309, 'E': 262, 'F': 377}

test_splits = ['test_task', 'test_domain', 'test_website']

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

def format_input_elements(
    sample, candidate_ids, gt=-1, keep_html_brackets=True
):
    """
    Wrapper around format_input_multichoice that returns a dict of fields.
    """
    tree_repr, _, target_text, prev_actions, choices_str = format_input_multichoice(
        sample, candidate_ids, gt, keep_html_brackets=keep_html_brackets
    )
    return {
        "annotation_id": sample.get("annotation_id"),
        "action_uid": sample.get("action_uid"),
        "html_context": tree_repr,
        "task": sample.get("confirmed_task"),
        "previous_actions": prev_actions,
        "choices_str": choices_str,
        "target_text": target_text,
    }

def choice2token_id_mapping(tokenizer, num_choices):
        """Return a mapping from the candidate choices to thier token ids."""
        options = [chr(65 + i)  for i in range(num_choices + 1)]

        tokens = tokenizer(options, add_special_tokens=False)
        mapping = {char: token_id_list[0] for char, token_id_list in zip(options, tokens.input_ids)}
        return mapping

def _resize_image(image: Image.Image, max_image_edge=None) -> Image.Image:
    """Resize so the longer edge is at most max_image_edge."""
    if image is None or max_image_edge is None:
        return image
    w, h = image.size
    longer = max(w, h)
    if longer <= max_image_edge:
        return image
    scale = max_image_edge / float(longer)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)








class PromptView:
    """Helper that returns a dict of the raw prompt"""
    def __init__(self, parent):
        self._parent = parent
    
    def __getitem__(self, idx:int) -> dict:

        return self._parent._get_prompt_element(idx)

class MultiChoiceDataset(Dataset):
    """
    Mind2Web multi-choice dataset wrapper.

    Each item builds a textual context (HTML context + previous actions) and a
    multiple-choice prompt over candidate elements, returning tokenized inputs
    and labels suitable for seq2seq generation.
    The multiple-choices are deterministic here (always take the top num_candidates)

    Args:
        data: HF Dataset or list of samples with pos/neg candidates and HTML.
        tokenizer: Hugging Face tokenizer used for encoding.
        num_candidates: Max candidates to include per sample (excludes the
            fallback \"None of the above\").
        max_context_len: Max tokens for context/prompt segments.
        mode: \"multichoice\" or \"generation\" formatting.
        cache_prompt: Cache raw prompt pieces per sample (process-local shared dict).
        cache_tokenized: Cache tokenized model inputs per sample (process-local shared dict).

    __getitem__ returns a dict with:
        - input_ids: list[int], concatenated context + prompt token ids
        - attention_mask: list[int], same length as input_ids
        - labels: sequence for training (string or token ids depending on formatting)
        - annotation_id, action_uid, idx: metadata for tracking
    """
    # Shared caches across instances
    id2split = {0: "test_task", 1: "test_domain", 2: "test_website"}
    split2id = {v: k for k, v in id2split.items()}
    choice2token_id = {}
    # caches
    _prompt_cache = {}
    _token_cache = {}


    def __init__(
        self,
        data,
        tokenizer,
        num_candidates=5,
        max_context_len=512,
        mode="multichoice",
        cache_prompt=False,
        cache_tokenized=False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.num_candidates = int(num_candidates)
        self.max_context_len = max_context_len
        self.mode = mode
        self.cache_prompt = cache_prompt
        self.cache_tokenized = cache_tokenized

        self.prompt_view = PromptView(self)

        # Generate choice2token_id once
        # if not MultiChoiceDataset.choice2token_id and self.tokenizer is not None:
            # MultiChoiceDataset.choice2token_id = choice2token_id_mapping(self.tokenizer, self.num_candidates)

    def __len__(self):
        return len(self.data)
    
    def action_uids(self):
        """Return a list of all action_uid values in the dataset."""
        return [sample.get("action_uid") for sample in self.data]
    
    def _cache_key(self, idx, sample=None):
        """Stable cache key shared across instances."""
        sample = sample or self.data[idx]
        act_uid = sample.get("action_uid")
        if not act_uid:
            raise ValueError("Missing action_uid in sample")
        return act_uid
    
    def _build_prompt_element(self, idx):
        """Return the raw prompt elements before tokenization."""
        sample = self.data[idx]

        all_cands = sample["pos_candidates"] + sample["neg_candidates"]

        all_cands_sorted = sorted(
            all_cands, key=lambda c: c.get("rank", float("inf"))
        )[: self.num_candidates]

 
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
            prev_actions, choices_str = "", ""

        return seq_context, seq_in, seq_out, prev_actions, choices_str

    def _get_prompt_element(self, idx):
        """Return cached or freshly built prompt elements (raw strings)."""
        if not self.cache_prompt:
            return self._build_prompt_element(idx)
        key = self._cache_key(idx)
        cached = self._prompt_cache.get(key)
        if not cached:
            cached = self._build_prompt_element(idx)
            self._prompt_cache[key] = cached
        return cached

    def _tokenize_prompt(self, seq_context, seq_in, seq_out):
        seq_context_tok = self.tokenizer(
            seq_context,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        )
        seq_in_tok = self.tokenizer(
            seq_in,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=True,
        )

        seq_out_tok = self.tokenizer(
            seq_out,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_context_len,
        )

        model_input = {
            "input_ids": seq_context_tok["input_ids"] + seq_in_tok["input_ids"],
            "attention_mask": seq_context_tok["attention_mask"] + seq_in_tok["attention_mask"],
            "labels": seq_out_tok['input_ids']
        }


        return model_input
      
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.cache_tokenized:
            key = self._cache_key(idx, sample)
            cached = self._token_cache.get(key)
            if cached:
                return cached

        seq_context, seq_in, seq_out, *_ = self._get_prompt_element(idx)
        
        log_prompt(sample["annotation_id"], sample["action_uid"], seq_context + seq_in, seq_out, model="flan-xl")

        model_input = self._tokenize_prompt(seq_context, seq_in, seq_out)
        
        # Add metadata
        model_input["annotation_id"] = sample["annotation_id"]
        model_input["action_uid"] = sample["action_uid"]
        model_input["idx"] = idx

        if self.cache_tokenized:
            self._token_cache[key] = model_input
        return model_input

class MultiChoiceDatasetRandom(MultiChoiceDataset):

    def __init__(
        self,
        data,
        tokenizer,
        num_candidates=5,
        max_context_len=512,
        mode="multichoice",
        neg_ratio=0.05, #0.2,
        top_k=-1,
        seed=42,
    ):
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            num_candidates=num_candidates,
            max_context_len=max_context_len,
            mode=mode,
        )
        self.neg_ratio = neg_ratio
        self.top_k = top_k
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def _rng_for_idx(self, idx: int) -> random.Random:
        return random.Random(self.seed + idx)

    def sample_difficulty(self, idx: int) -> dict:
        """Return simple difficulty signals for a base sample index."""
        sample = self.data[idx]
        pos = sample.get("pos_candidates", []) or []
        neg = sample.get("neg_candidates", []) or []
        pos_ranks = [c.get("rank") for c in pos if c.get("rank") is not None]
        # Use -1 as a sentinel instead of None to keep collate happy.
        best_pos_rank = min(pos_ranks) if pos_ranks else -1
        return {
            "num_pos": len(pos),
            "num_neg": len(neg),
            "has_pos": len(pos) > 0,
            "best_pos_rank": best_pos_rank,
        }

    def __getitem__(self, idx):
        base_idx = idx
        sample = self.data[base_idx]
        rng = self._rng_for_idx(idx)

        pos_candidates = sample.get("pos_candidates", [])
        neg_candidates = sample.get("neg_candidates", [])

        # Optional top-k filtering for negatives
        if self.top_k > 0:
            top_negatives = [
                c for c in neg_candidates if c.get("rank", float("inf")) < self.top_k
            ]
            other_negatives = [
                c for c in neg_candidates if c.get("rank", float("inf")) >= self.top_k
            ]
        else:
            top_negatives = []
            other_negatives = neg_candidates

        neg_pool = top_negatives if (rng.random() < 0.8 and top_negatives) else other_negatives

        # Decide whether to include a positive candidate
        if pos_candidates and (rng.random() > self.neg_ratio or not neg_pool): #  neg_ratio=0.2
            pos_candidate = rng.choice(pos_candidates)
            neg_sample = rng.sample(
                neg_pool, min(len(neg_pool), self.num_candidates - 1)
            )
            gt = _extract_candidate_ids(pos_candidate["backend_node_id"])
            candidate_ids = [gt] + [
                _extract_candidate_ids(c["backend_node_id"]) for c in neg_sample
            ]
        else:
            neg_sample = rng.sample(neg_pool, min(len(neg_pool), self.num_candidates))
            gt = -1
            candidate_ids = [
                _extract_candidate_ids(c["backend_node_id"]) for c in neg_sample
            ]

        if self.mode == "multichoice":
            seq_context, seq_in, seq_out, prev_actions, choices_str = format_input_multichoice(
                sample, candidate_ids, gt
            )
        else:
            seq_context, seq_in, seq_out, _ = format_input_generation(
                sample, candidate_ids, gt
            )


        log_prompt(
            sample["annotation_id"],
            sample["action_uid"],
            seq_context + seq_in,
            seq_out,
            model="flan-xl",
        )

        model_input = self._tokenize_prompt(seq_context, seq_in, seq_out)
        # model_input["difficulty"] = self.sample_difficulty(base_idx)

        model_input["idx"] = base_idx
        model_input["annotation_id"] = sample["annotation_id"]
        model_input["action_uid"] = sample["action_uid"]
        
        return model_input

class MultiChoiceDatasetPrompt(Dataset):
    """
    Similar to MultiChoiceDataset but builds a single prompt string using a
    provided template and format_input_elements. Returns raw prompt components
    (no tokenization) for downstream handling.
    """

    def __init__(
        self,
        data,
        prompt_template: Template,
        split_name: str = "test",
        num_candidates: int = 5,
        add_html_context: bool = True,
        add_screenshot: bool = False,
        max_image_edge: int = 720,

    ):
        self.data = data
        self.prompt_template = prompt_template
        self.num_candidates = int(num_candidates)
        self.split_name = split_name
        self.add_html_context = add_html_context
        self.add_screenshot = add_screenshot

        # Downscale large images to save VRAM; keep aspect ratio.
        self.max_image_edge = max_image_edge

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        all_cands = sample["pos_candidates"] + sample["neg_candidates"]
        all_cands_sorted = sorted(
            all_cands, key=lambda c: c.get("rank", float("inf"))
        )[: self.num_candidates]

        top_pos = [c for c in all_cands_sorted if c in sample["pos_candidates"]]
        
        gt = -1
        if top_pos:
            best_rank = min(c.get("rank", float("inf")) for c in top_pos)
            best_pos = [c for c in top_pos if c.get("rank", float("inf")) == best_rank]
            pos_candidate = random.choice(best_pos)
            gt = _extract_candidate_ids(pos_candidate["backend_node_id"])
            
        candidate_ids = [ _extract_candidate_ids(c["backend_node_id"]) for c in all_cands_sorted]

        elements = format_input_elements(sample, candidate_ids, gt)

        prompt = self.prompt_template.safe_substitute(
            html=elements["html_context"],
            task=elements["task"],
            prev_actions=elements["previous_actions"],
            choices=elements["choices_str"],
        )

        model_input = {
            "prompt": prompt,
            "task": elements["task"],
            "previous_actions": elements["previous_actions"],
            "choices": elements["choices_str"],
            "annotation_id": sample.get("annotation_id"),
            "action_uid": sample.get("action_uid"),
            "target_text": elements["target_text"],
        }

        if self.add_html_context:
            model_input["html_context"] = elements["html_context"]
        
        if self.add_screenshot:
            screenshot = sample.get("screenshot_image")
            if screenshot is not None:
                image = Image.open(screenshot).convert("RGB")
                image = _resize_image(image, self.max_image_edge)
            else:
                image = None

            model_input["screenshot_image"] = image
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
    dataset: Dataset,
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

def build_datasets_dict(
    data_dir="osunlp/Multimodal-Mind2Web",
    cache_dir=None,
    candidates_dir="candidates",
) -> dict[str, Dataset]:
    """Return a dict of Datasets keyed by split name."""
    ds_map = {}
    base_cand_dir = pathlib.Path(candidates_dir)
    if not base_cand_dir.is_absolute():
        base_cand_dir = pathlib.Path(__file__).resolve().parent.parent / base_cand_dir

    for split_file in test_splits:
        cand_path = base_cand_dir / f"scores_{split_file}.pkl"
        candidate_results = pd.read_pickle(cand_path)

        flattened = get_data_split(
            data_dir=data_dir,
            split_file=split_file,
            candidate_results=candidate_results,
            cache_dir=cache_dir,
        )
        ds_map[split_file] = flattened
       
    return ds_map

def multichoice_collate_fn(batch, device='cuda', token_pad_id=0, label_pad_id=-100):
    """
    Convert the model_input dict returned by MultiChoiceDataset.__getitem__
    (lists of ints) or a collated batch (tensors) into tensors appropriate for model generate.
    """
    input_ids = [torch.tensor(item["input_ids"]).to(device) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]).to(device) for item in batch]
    labels  = [torch.tensor(item["labels"]) for item in batch]

    # Add padding
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=token_pad_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=label_pad_id)
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Add addional information
    out["ids"] = [ex["idx"] for ex in batch]
    out["annotation_ids"] = [ex["annotation_id"] for ex in batch]
    out["action_uids"] = [ex["action_uid"] for ex in batch]
    return out

def raw_data_collate_fn(batch: list):
    """
    Converts a list of dicts (rows) into a dict of lists (columns).
    Example:
    Input:
      [
        {'id': 1, 'prompt': 'Task A', 'image': img1}, 
        {'id': 2, 'prompt': 'Task B', 'image': img2}
      ]
      
    Output:
      {
        'id': [1, 2],
        'prompt': ['Task A', 'Task B'],
        'image': [img1, img2]
      }
    """
    keys = batch[0].keys()
    
    return {
        key: [item[key] for item in batch] 
        for key in keys
    }


def subsample_df_by_annotation(df: pd.DataFrame, seed: int = 0, frac: float = 0.5):
    """
    Split df by unique annotation_id into two DataFrames (df1 [frac] , df2 [1-frac]) deterministically.

    Args:
        df: DataFrame with an 'annotation_id' column.
        seed: RNG seed for reproducibility.
        frac: Fraction of unique annotation_ids to put in the first split.

    Returns:
        cal_split_df, pen_split_df
    """
    ids = df["annotation_id"].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * frac)
    cal_ids = set(ids[:cut])
    pen_ids = set(ids[cut:])
    cal_split = df[df["annotation_id"].isin(cal_ids)].reset_index(drop=True)
    pen_split = df[df["annotation_id"].isin(pen_ids)].reset_index(drop=True)
    return cal_split, pen_split
