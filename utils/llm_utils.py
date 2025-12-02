import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from string import Template
import pandas as pd
from tqdm.auto import tqdm
import re 
from utils.prompts import oracle_prompt_template, re_eval_prompt_template, standard_prompt_template
from PIL import Image
from utils import log_response

from typing import Dict, Any, Optional, Tuple
import os
from google import genai
import gc

_LABEL_RE = re.compile(r"^\s*([A-F])\.", re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(CLICK|SELECT|TYPE)", re.IGNORECASE)
_VALUE_RE  = re.compile(r"Value:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


import logging

# 1. Create logger
logger = logging.getLogger("experiment_logger")
logger.setLevel(logging.INFO)

# 2. Create file handler
handler = logging.FileHandler("prompt_eval.log", mode='w') # 'w' overwrites, 'a' appends
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)




def parse_output(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    letter_match = _LABEL_RE.search(text)
    letter = letter_match.group(1).upper() if letter_match else None

    action_match = _ACTION_RE.search(text)
    action = action_match.group(1).upper() if action_match else None

    value_match = _VALUE_RE.search(text)
    value = value_match.group(1).strip() if value_match else None
    if value == "":
        value = None

    return letter, action, value,

def tensorize_item(item: Dict[str, Any], device: str):
    """
    Convert the model_input dict returned by MultiChoiceDataset.__getitem__
    (lists of ints) or a collated batch (tensors) into tensors appropriate for model generate.
    """
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.long, device=device)

    input_ids = to_tensor(item["input_ids"])
    attention_mask = to_tensor(item["attention_mask"])
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def choices_probabilities(choices_to_token_ids:dict, logits, pred_set:list=None) -> dict:
    """
    Calculate the probabilities of specific choice tokens from model logits.

    This function extracts the probabilities of tokens corresponding to specific choices
    (e.g., 'A', 'B', 'C') from the model's output logits. It can optionally filter
    to a subset of choices.

    :param choices_to_token_ids: A dictionary mapping choice labels (str) to their tokenizer token IDs (int).
    :type choices_to_token_ids: dict
    :param logits: The raw output logits from the model. Expected shape (batch_size, vocab_size).
    :type logits: torch.Tensor
    :param pred_set: An optional list of choice labels to filter the calculation. 
                     If provided, only probabilities for these choices are returned.
    :type pred_set: list, optional
    :return: A dictionary mapping choice labels to their probabilities (0.0 to 1.0).
    :rtype: dict
    """
    choices = choices_to_token_ids
    
    if pred_set:
        # remap to filtered choices (A,B,...)
        mapping = {lab:chr(65 +i) for i, lab in enumerate(pred_set)}
        choices = {lab:choices_to_token_ids[mapping[lab]] for lab in pred_set }
        # choices = {lab:choices_to_token_ids[lab] for lab in pred_set }
        
    # else:
    #     mapping = dict(zip(choices_to_token_ids.keys(), choices_to_token_ids.keys()))

    choices_idx = torch.tensor(list(choices.values()), device='cpu')

    probs = F.softmax(logits[:, choices_idx], dim=-1)[0]
    #F.softmax(logits, dim=-1)[:, choices_idx][0]
    
    choices_probs = dict(zip(choices.keys(), probs.cpu().tolist()))
    # remap to original labels
    # if pred_set:
    #     inv_mapping = {v:k for k,v in mapping.items()}
    #     choices_probs = {inv_mapping[k]:v for k,v in choices_probs.items()}
    return choices_probs

def run_evaluation(data_sets:dict,model, tokenizer, max_iter=None, max_new_tokens:int=15, device='cuda') -> pd.DataFrame:
    outputs = []
    # Get choice (A,B,...,F) to token id mapping
    choices_to_token_ids = list(data_sets.values())[0].choices_token_ids_mapping()
    choices_idx = torch.tensor(list(choices_to_token_ids.values()), device=device)

    for split_idx, (ds_split, ds) in enumerate(data_sets.items()):
        relative_idx = 0
        for i, item in enumerate(tqdm(ds, desc="Generating...")):
            if max_iter is not None and i > max_iter:
                break
            model_input = tensorize_item(item, device)
 
            with torch.inference_mode():
                    out = model.generate(
                        **model_input,
                        eos_token_id=model.config.eos_token_id,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
     
            decoded = tokenizer.decode(out["sequences"][0], skip_special_tokens=True)
            pred_label, pred_action, pred_value = parse_output(decoded)
            labels_tokens = item.get("labels")
            
            # Calculate choice probabilities
            logits = out["scores"][0]
            probs = F.softmax(logits, dim=-1)[:, choices_idx][0]
            choices_probs = dict(zip(choices_to_token_ids.keys(), probs.cpu().tolist()))
            labels = item.get("labels").strip()
            outputs.append(
                    [
                        relative_idx,
                        ds.data[relative_idx]["annotation_id"],
                        ds.data[relative_idx]["action_uid"],
                        pred_label, pred_action, pred_value,
                        labels.split('.')[0],
                        labels, 
                        choices_probs,
                        choices_probs.get(pred_label, 0),
                        split_idx,
                        labels_tokens

                    ]    
                )
            log_response(ds.data[relative_idx]["annotation_id"], ds.data[relative_idx]["action_uid"], decoded)
            relative_idx += 1

            # Tight memory management
            del out 
            torch.cuda.empty_cache()

    cols = ["relative_idx", "annotation_id", "action_uid", "pred_label", "pred_action", "pred_value",
             "label",'label_text', "choices_probs", "prob", "test_split", "labels_tokens"]
    results_df = pd.DataFrame(outputs, columns=cols)

    return results_df

class GeminiBase:
    """An interface class"""
    def __init__(self, model_name: str, api_key: str = None):
        # Try to find API key in env or file if not provided
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key is None and os.path.exists("api.txt"):
                try:
                    api_key = open("api.txt", "r").read().strip()
                except:
                    pass
        
        if not api_key:
            print("Warning: GEMINI_API_KEY not found. Oracle calls will fail.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        


class QwenBase:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = None,
        device: str = None,
        dtype=torch.float16,
        max_image_edge: int = 720,
        max_tokens: int = 25,
        prompt_template: str = standard_prompt_template,
    ):
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=dtype,
            device_map="auto" if device is None else None,
        )
        if device is not None:
            self.model.to(device)
        self.device = device or 'cuda'
        # Downscale large images to save VRAM; keep aspect ratio.
        self.max_image_edge = max_image_edge
        self.max_tokens = max_tokens
        self.prompt_template = Template(prompt_template)

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        letters = list("ABCDEF")
        self.letter_ids = {c: self.tokenizer(c, add_special_tokens=False).input_ids[0] for c in letters}
        self.token_ids = list(self.letter_ids.values())

    def _maybe_resize(self, image: Image.Image) -> Image.Image:
        """Resize so the longer edge is at most max_image_edge."""
        if image is None or self.max_image_edge is None:
            return image
        w, h = image.size
        longer = max(w, h)
        if longer <= self.max_image_edge:
            return image
        scale = self.max_image_edge / float(longer)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.LANCZOS)

    def __call__(self,**kwargs) -> str:
            # task:str,
            # prev_actions:str,
            # html_context: str,
            # choices:str,
            # candidates: str,
            # image: Image.Image) -> str:
        """
        Returns the chosen candidate id (or empty string if None selected).
        """
        messages = []
        if 'history' in kwargs:
            messages = list(kwargs.pop('history'))
        prompt = self.prompt_template.safe_substitute(**kwargs)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        })
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image = kwargs.get('image', None)
        # if image is not None:
        #     image = self._maybe_resize(image)
        #     inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        # else:
        #     inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        logits = outputs.scores[0]
        
        probs = F.softmax(logits[:, self.token_ids], dim=-1)[0]
        #F.softmax(logits, dim=-1)[:, choices_idx][0]
        
        choices_probs = dict(zip(self.letter_ids.keys(), probs.cpu().tolist()))
            
        # Strip the prompt tokens
        prompt_len = inputs["input_ids"].shape[-1]
        reply_ids = outputs.sequences[0][prompt_len:]
        reply_text = self.processor.decode(reply_ids, skip_special_tokens=True).strip()
        return reply_text, choices_probs


class Oracle:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = None,
        device: str = None,
        dtype=torch.float16,
        max_image_edge: int = 720,
        max_tokens: int = 25,
    ):
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=dtype,
            device_map="auto" if device is None else None,
        )
        if device is not None:
            self.model.to(device)
        self.device = device or 'cuda'
        # Downscale large images to save VRAM; keep aspect ratio.
        self.max_image_edge = max_image_edge
        self.max_tokens = max_tokens

    def _build_prompt(self,
                      task:str,
                      prev_actions:str,
                      html_context: str,
                      choices:str,
                      candidates: str,
                      ) -> str:
        prompt_template = Template(oracle_prompt_template)
        prompt = prompt_template.safe_substitute(
            task=task,
            previous_actions=prev_actions,
            html=html_context,
            options=choices,
            prediction_set_options=candidates,
        )
        
        return prompt

    def _maybe_resize(self, image: Image.Image) -> Image.Image:
        """Resize so the longer edge is at most max_image_edge."""
        if image is None or self.max_image_edge is None:
            return image
        w, h = image.size
        longer = max(w, h)
        if longer <= self.max_image_edge:
            return image
        scale = self.max_image_edge / float(longer)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.LANCZOS)

    def ask(self,
            task:str,
            prev_actions:str,
            html_context: str,
            choices:str,
            candidates: str,
            image: Image.Image) -> str:
        """
        Returns the chosen candidate id (or empty string if None selected).
        """
   
        prompt = self._build_prompt(task, prev_actions, html_context, choices, candidates)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},  # placeholder; processor will insert the image tokens
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if image is not None:
            image = self._maybe_resize(image)
            inputs = self.processor(images=[image], text=[text], return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
            
        # Strip the prompt tokens
        prompt_len = inputs["input_ids"].shape[-1]
        reply_ids = output_ids[0, prompt_len:]
        reply_text = self.processor.decode(reply_ids, skip_special_tokens=True).strip()
        return reply_text



def filter_choices(choices_str: str, pred_set: list[str]):
    """Filter the choices string to only include those in pred_set.
    X-s IMPORTANT: if 'A' (reserved label) is not in the pred_set (unlikely)
      another label will be mapped to 'A'"""
    mask = list(ord(label) - 65 for label in pred_set)
    choices_strs = choices_str.splitlines()
    # filtered_choices_str = '\n'.join(chr(65 + i) + choices_strs[i][1:] for i in mask)
    filtered_choices_str = '\n'.join(chr(65 + shift) + choices_strs[i][1:] for shift, i in enumerate(mask))
    
    return filtered_choices_str


def re_evaluate_with_oracle(test_dict:pd.DataFrame, df:pd.DataFrame, answer_df:pd.DataFrame,model, tokenizer, device='cuda') -> pd.DataFrame:
    prompt_template = Template(re_eval_prompt_template)
    idx_split_map = {i:split_name for i, split_name in enumerate(test_dict.keys())}
    # Align answers to the filtered df order
    answers = answer_df.reindex(df.index)['oracle_answer'].values
    outputs = []

    new_correct = 0
    for pos, (_, row) in enumerate(tqdm(df.iterrows(), desc="Re-evaluating with oracle...", total=len(df))):
      

        i =  30
        if row['action_uid'] !='9e9d839b-645c-4dc2-b821-14098c653005' or row['annotation_id']!='9223f1b4-43ad-4636-9541-99ff9e6ad918':continue
        # if pos < i:
        #     continue
        # if pos > i:
        #     break

        relative_idx = row['relative_idx']
        test_set = test_dict[idx_split_map[row['test_split']]]
        html_context, seq_in, seq_out, prev_actions, choices_str = test_set.prompt_view[relative_idx]
        task = test_set.data[relative_idx]['confirmed_task']
        
        filtered_choices_str = filter_choices(choices_str, row['pred_set'])
        prompt = prompt_template.safe_substitute(task=task, html=html_context,
                                                 prev_actions=prev_actions,
                                                  choices=filtered_choices_str,
                                                #   choices = choices_str,
                                                  help=answers[pos])
                
        seq_context = tokenizer(html_context,truncation=True,max_length=512,add_special_tokens=False,)
        seq_in = tokenizer(prompt,add_special_tokens=True,truncation=True,max_length=512)
        model_input = {
                    "input_ids": seq_context["input_ids"] + seq_in["input_ids"],
                    "attention_mask": seq_context["attention_mask"] + seq_in["attention_mask"],
                }
        model_input = tensorize_item(model_input, device=device)
        with torch.inference_mode():
                    out = model.generate(
                        **model_input,
                        eos_token_id=model.config.eos_token_id,
                        max_new_tokens=25,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
        decoded = tokenizer.decode(out["sequences"][0], skip_special_tokens=True)
        # choices_probs = choices_probabilities(test_set.choices_token_ids_mapping(),
                                            #  out["scores"][0], pred_set=None)#row['pred_set']
        choices_probs = choices_probabilities(test_set.choices_token_ids_mapping(),
                                             out["scores"][0], pred_set=row['pred_set'])
        pred_label, pred_action, pred_value = parse_output(decoded)
        # print(f"Choices probs (full): {choices_probs0}")
        # print(f"Choices probs (filtered): {choices_probs}")
        if pos > 0:
            print(html_context)
            print(prompt)
            print(f"Decoded: {decoded}, pred: {pred_label}, label: {row['label']}, Original pred: {row['pred_label']}")
            print(f'pred set: {row["pred_set"]}, choices_probs: {choices_probs}')
            print(f"-------------------------")
            print(row['choices_probs'])
            return
        new_correct += int(pred_label == row['label'])
        if pos % 100 == 0:
            tqdm.write(f"{new_correct}/{pos+1}({new_correct/(pos+1):.2f})")
        

        # return
        
        outputs.append(
                [
                    relative_idx,
                    row['annotation_id'],
                    row['action_uid'],
                    pred_label, pred_action, pred_value,
                    row['label'],
                    row['label_text'],
                    choices_probs,
                    choices_probs.get(pred_label, 0),
                    row['test_split'],
                    decoded
                ]  
        )  

        # Tight memory management
        del out
        torch.cuda.empty_cache()
        gc.collect()
    
        # log_response(row['annotation_id'], row['action_uid'], decoded)
        logger.info({"re_eval": {
            "annotation_id": row['annotation_id'],
            "action_uid": row['action_uid'],
            "prompt": prompt,
            "response": decoded,
        }, 'name':'re_eval_2.1'})
        
        
        # tqdm.write(f"{pred_label} V.S. {row['pred_label']} rate: {new_correct}/{pos+1}({new_correct/(pos+1):.2f})")    
    cols = ["relative_idx", "annotation_id", "action_uid", "pred_label", "pred_action", "pred_value",
             "label",'label_text', "choices_probs", "prob", "test_split", "text_output"]
    results_df = pd.DataFrame(outputs, columns=cols)
    return results_df


def re_evaluate_with_oracle_batch(
    test_dict: pd.DataFrame,
    df: pd.DataFrame,
    answer_df: pd.DataFrame,
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 3,
    max_context_len: int = 512,
    max_new_tokens: int = 25,
) -> pd.DataFrame:
    """
    Batched variant of re_evaluate_with_oracle to speed up generation.
    """
    prompt_template = Template(re_eval_prompt_template)
    idx_split_map = {i: split_name for i, split_name in enumerate(test_dict.keys())}
    answers = answer_df.reindex(df.index)["oracle_answer"].values
    outputs = []
    new_correct = 0

    def pad_sequences(seqs, pad_id: int):
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        return out

    rows = list(df.iterrows())
    for start in tqdm(range(0, len(rows), batch_size), desc="Re-evaluating with oracle (batched)..."):
        batch_rows = rows[start : start + batch_size]
        if not batch_rows:
            continue

        contexts, prompts, meta = [], [], []
        for b_pos, (idx, row) in enumerate(batch_rows):
            relative_idx = row["relative_idx"]
            test_set = test_dict[idx_split_map[row["test_split"]]]
            html_context, _, _, prev_actions, choices_str = test_set.prompt_view[relative_idx]
            filtered_choices_str = filter_choices(choices_str, row["pred_set"])
            prompt = prompt_template.safe_substitute(
                prev_actions=prev_actions,
                choices=filtered_choices_str,
                help=answers[start + b_pos],
            )
            contexts.append(html_context)
            prompts.append(prompt)
            meta.append(
                (
                    relative_idx,
                    row,
                    test_set,
                    test_set.choices_token_ids_mapping(),
                )
            )

        ctx_tok = tokenizer(
            contexts,
            truncation=True,
            max_length=max_context_len,
            add_special_tokens=False,
        )
        prm_tok = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_context_len,
        )

        combined_ids, combined_masks = [], []
        for ctx_ids, ctx_mask, prm_ids, prm_mask in zip(
            ctx_tok["input_ids"],
            ctx_tok["attention_mask"],
            prm_tok["input_ids"],
            prm_tok["attention_mask"],
        ):
            combined_ids.append(ctx_ids + prm_ids)
            combined_masks.append(ctx_mask + prm_mask)

        input_ids = pad_sequences(combined_ids, pad_id=tokenizer.pad_token_id or 0)
        attention_mask = pad_sequences(combined_masks, pad_id=0)
        model_input = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.inference_mode():
            out = model.generate(
                **model_input,
                eos_token_id=model.config.eos_token_id,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        for b_idx, (relative_idx, row, test_set, choice_map) in enumerate(meta):
            decoded = tokenizer.decode(out["sequences"][b_idx], skip_special_tokens=True)
            logits_first = out["scores"][0][b_idx].unsqueeze(0)
            choices_probs = choices_probabilities(
                choice_map, logits_first, pred_set=row["pred_set"]
            )
            pred_label, pred_action, pred_value = parse_output(decoded)
            new_correct += int(pred_label == row["label"])
            tqdm.write(
                f"{pred_label} V.S. {row['pred_label']} "
                f"rate: {new_correct}/{start + b_idx + 1}({new_correct/(start + b_idx + 1):.2f})"
            )
            
            outputs.append(
                [
                    relative_idx,
                    row["annotation_id"],
                    row["action_uid"],
                    pred_label,
                    pred_action,
                    pred_value,
                    row["label"],
                    row["label_text"],
                    choices_probs,
                    choices_probs.get(pred_label, 0),
                    row["test_split"],
                    decoded,
                ]
            )

            # del out
            torch.cuda.empty_cache()
            gc.collect()

    cols = [
        "relative_idx",
        "annotation_id",
        "action_uid",
        "pred_label",
        "pred_action",
        "pred_value",
        "label",
        "label_text",
        "choices_probs",
        "prob",
        "test_split",
        "text_output",
    ]
    return pd.DataFrame(outputs, columns=cols)
