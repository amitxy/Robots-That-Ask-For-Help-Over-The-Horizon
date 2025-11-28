import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from string import Template
import pandas as pd
from tqdm.auto import tqdm
import re 
from utils.prompts import oracle_prompt_template, human_prompt_template, re_eval_prompt_template
from PIL import Image
from utils import log_response
from typing import Dict, Any, Optional, Tuple
import os
from google import genai

_LABEL_RE = re.compile(r"^\s*([A-F])\.", re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(CLICK|SELECT|TYPE)", re.IGNORECASE)
_VALUE_RE  = re.compile(r"Value:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

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
    (lists of ints) into tensors appropriate for model generate.
    """
    input_ids = torch.LongTensor(item["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.LongTensor(item["attention_mask"]).unsqueeze(0).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def choices_probabilities(choices_to_token_ids:dict, logits, pred_set:list=None, filtered_choices_mapping:dict=None) -> dict:
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
    :param filtered_choices_mapping: dict - mapping of the filtered choices labels to the original labels
    :rtype: dict
    """
    if pred_set is not None:
        choices_to_token_ids = {k:v for k,v in choices_to_token_ids.items() if k in pred_set}
    choices_idx = torch.tensor(list(choices_to_token_ids.values()), device='cpu')
    probs = F.softmax(logits, dim=-1)[:, choices_idx][0]
    choices_probs = dict(zip(choices_to_token_ids.keys(), probs.cpu().tolist()))
    # remap to original labels
    if filtered_choices_mapping:
        choices_probs = {filtered_choices_mapping[k]:v for k,v in choices_probs.items()}
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
     IMPORTANT: if 'A' (reserved label) is not in the pred_set (unlikely)
      another label will be mapped to 'A'"""
    mapping = {chr(65 + i): label for i, label in enumerate(pred_set)}
    mask = list(ord(label) - 65 for label in pred_set)
    choices_strs = choices_str.splitlines()
    filtered_choices_str = '\n'.join(chr(65 + i) + choices_strs[i][1:] for i in mask)
    return filtered_choices_str, mapping


def re_evaluate_with_oracle(test_dict:pd.DataFrame, df:pd.DataFrame, answer_df:pd.DataFrame,model, tokenizer, device='cuda') -> pd.DataFrame:
    prompt_template = Template(re_eval_prompt_template)
    idx_split_map = {i:split_name for i, split_name in enumerate(test_dict.keys())}
    # Align answers to the filtered df order
    answers = answer_df.reindex(df.index)['oracle_answer'].values
    outputs = []
    for pos, (_, row) in enumerate(tqdm(df.iterrows(), desc="Re-evaluating with oracle...", total=len(df))):
        relative_idx = row['relative_idx']
        test_set = test_dict[idx_split_map[row['test_split']]]
        html_context, seq_in, seq_out, prev_actions, choices_str = test_set.prompt_view[relative_idx]
        task = test_set.data[relative_idx]['confirmed_task']
        
        filtered_choices_str, choice_mapping = filter_choices(choices_str, row['pred_set'])
        prompt = prompt_template.safe_substitute(task=task,
                                                 prev_actions=prev_actions,
                                                  choices=filtered_choices_str,
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
                        max_new_tokens=20,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
        decoded = tokenizer.decode(out["sequences"][0], skip_special_tokens=True)
        choices_probs = choices_probabilities(test_set.choices_token_ids_mapping(),
                                             out["scores"][0], pred_set=row['pred_set'],
                                             filtered_choices_mapping=choice_mapping)
        pred_label, pred_action, pred_value = parse_output(decoded)
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
        break
        log_response(row['annotation_id'], row['action_uid'], decoded)
    cols = ["relative_idx", "annotation_id", "action_uid", "pred_label", "pred_action", "pred_value",
             "label",'label_text', "choices_probs", "prob", "test_split", "text_output"]
    results_df = pd.DataFrame(outputs, columns=cols)
    return results_df
