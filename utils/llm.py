from mind2web.dataloader import MultiChoiceDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, LogitsProcessor, LogitsProcessorList, AutoModelForSeq2SeqLM
from string import Template
import pandas as pd
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

from utils.prompts import oracle_prompt_template, re_eval_prompt_template, standard_prompt_template
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List, Union

import os
from google import genai
import gc


import utils
from utils import log_response
from utils.helpers import parse_output  # for parse_output
from mind2web.dataloader import MultiChoiceDataset, multichoice_collate_fn, raw_data_collate_fn, choice2token_id

import json, copy
from pathlib import Path

import logging

# 1. Create logger
logger = logging.getLogger("experiment_logger")
logger.setLevel(logging.INFO)

# 2. Create file handler
handler = logging.FileHandler("prompt_eval.log", mode='w') # 'w' overwrites, 'a' appends
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

DEFAULT_CANDIDATES = ['A', 'B', 'C', 'D', 'E', 'F']


def tensorize_item(item: Dict[str, Any], device: str):
    """
    Convert the model_input dict returned by MultiChoiceDataset.__getitem__
    (lists of ints) or a collated batch (tensors) into tensors appropriate for model generate.
    """
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.long, device='cpu')

    input_ids = to_tensor(item["input_ids"])
    attention_mask = to_tensor(item["attention_mask"])
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def choices_probabilities(logits, pred_set:list=None, temperature=1) -> dict:
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
    if temperature < 0:
            raise ValueError("Temperature must be non-negative")
    
    choices = choice2token_id
    
    if pred_set:
        # remap to filtered choices (A,B,...)
        # mapping = {lab:chr(65 +i) for i, lab in enumerate(pred_set)}
        # choices = {lab:choices[mapping[lab]] for lab in pred_set }
        choices = {lab:choice2token_id[lab] for lab in pred_set }
        
    # else:
    #     mapping = dict(zip(choice2token_id.keys(), choice2token_id.keys()))

    choices_idx = torch.tensor(list(choices.values()), device='cpu', dtype=torch.long)
    
    probs = F.softmax(logits[:,choices_idx] / temperature, dim=-1)[0]
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
    choices_to_token_ids = MultiChoiceDataset.choice2token_id
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
            labels_tokens = item.get("labels")[0]
            
            # Calculate choice probabilities
            logits = out["scores"][0]
            probs = F.softmax(logits, dim=-1)[:, choices_idx][0]
            choices_probs = dict(zip(choices_to_token_ids.keys(), probs.cpu().tolist()))
            labels = tokenizer.decode(labels_tokens, skip_special_tokens=True)
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


# This represents: {'pred_text': "[Model Output]", 'choices_logits': {'A': 5.2, 'B': 1.1, ..., 'F': 0.0}}
PredictionOutput = Dict[str, Union[str, Dict[str, float]]]

class BaseWrapper(ABC):
    @abstractmethod
    def generate(
        self, 
        prompts: List[str], 
        images: Optional[List[Image.Image]] = None, 
        **kwargs
    ) -> List[PredictionOutput]:
        """
        Unified interface. 
        Returns a list of dicts: {'pred_text': ..., 'choices_logits': ...}
        """
        pass

    @property
    def n_candidates(self) -> int:
        return self._n_candidates

    @property
    def choice2token_id(self) -> Dict[str, int]:
        """Map each choice letter ('A', 'B', ...) to its tokenizer id"""
        if self._choice2token_id:
            return self._choice2token_id
        
        mapping = {}
        for i in range(self.n_candidates + 1):
            cand = chr(65 + i)  # 'A', 'B', ...
            # Handle standard tokenizer vs processor tokenizer
            if hasattr(self.tokenizer, "encode"):
                token_ids = self.tokenizer.encode(cand, add_special_tokens=False)
            else:
                token_ids = self.tokenizer(cand, add_special_tokens=False)["input_ids"]

            mapping[cand] = token_ids[-1]
            
        return mapping


class MultimodalCausalWrapper(BaseWrapper):
    def __init__(self,
                 model_name: str ='Qwen/Qwen3-VL-8B-Instruct', 
                 device: str = "cuda",
                 cache_dir: str = None,
                 n_candidates: int = 5, 
                 max_new_tokens: int = 10):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=False, cache_dir=cache_dir)
        self.processor.tokenizer.padding_side = "left"
        # Expose tokenizer for BaseWrapper utilities (choice2token_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, 
            device_map="auto", 
            dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        self._n_candidates = n_candidates
        self._choice2token_id = {}
        self.max_new_tokens = max_new_tokens
        
       
    def generate(
        self, 
        prompts: List[str], 
        images: Optional[List[Image.Image]] = None,
        **kwargs
    ) -> List[PredictionOutput]:
        
        use_few_shots = kwargs.pop('use_few_shots', False)
         
        #Prepare Inputs
        few_shots = []
        if use_few_shots:
            base_dir = Path(__file__).resolve().parent.parent / "mind2web" / "llm_prompt.json"       
            few_shots = json.load(open(base_dir, "r"))
        formatted_inputs = []
        for i, prompt in enumerate(prompts):
            messages = copy.deepcopy(few_shots) if use_few_shots else []
            content = [{"type": "text", "text": prompt}]
            if images and images[i]:
                content.insert(0, {"type": "image", "image": images[i]})
            messages.append({"role": "user", "content": content})
            formatted_inputs.append(messages)


        text_prompts = self.processor.apply_chat_template(formatted_inputs, tokenize=False, add_generation_prompt=True)
        image_inputs = None
        if images:
            if isinstance(images, list):
                if any(img is not None for img in images):
                    image_inputs = images
            else:
                image_inputs = [images]
        inputs = self.processor(
            text=text_prompts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,         
                max_new_tokens=self.max_new_tokens,
                **kwargs
            )

        
        # Slice off the prompt portion
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs.sequences[:, input_len:]
        decoded_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

       
        first_token_logits = outputs.scores[0] 
        results = []
        for i, text in enumerate(decoded_texts):
            choices_logits = {
                cand_str: first_token_logits[i, cand_id].item() 
                for cand_str, cand_id in self.choice2token_id.items()
                }
            
            results.append({"pred_text": text.strip(), "choices_logits": choices_logits})

        return results

class TextSeq2SeqWrapper(BaseWrapper):
    def __init__(self, model_name: str = "google/flan-t5-xl", device: str = "auto", cache_dir: str = None, n_candidates: int = 5):
        self.device = device
        self._n_candidates = n_candidates
        self._choice2token_id = {}
        self.max_context_len = 512

        if model_name == "osunlp/MindAct_ActionPrediction_flan-t5-xl":
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", cache_dir=cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if device == "auto":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        
    def generate(self, prompts: List[str], html_contexts: List[str], images: Optional[List] = None, **kwargs):
        # Generic check: This architecture physically cannot see images
        if images and any(img is not None for img in images):
            print(f"Warning: {self.__class__.__name__} ignores images.")

        if len(prompts) != len(html_contexts):
            raise ValueError("prompts and html_contexts must have the same length.")

        tokenized_contexts = self.tokenizer(
            html_contexts,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
        )
        tokenized_prompts = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=True,
            return_tensors="pt",
            padding=False,
        )

        merged_ids = [torch.cat([c, p]) for c, p in zip(tokenized_contexts["input_ids"], tokenized_prompts["input_ids"])]
        merged_masks = [torch.cat([c, p]) for c, p in zip(tokenized_contexts["attention_mask"], tokenized_prompts["attention_mask"])]

        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = pad_sequence(merged_ids, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(merged_masks, batch_first=True, padding_value=0)

        if self.device == "auto" or self.device == "cuda":
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            
        # Seq2Seq models return ONLY the new tokens. No slicing needed.
        # outputs.scores is a tuple (one per generated token). Use first step for choice logits.
        decoded_texts = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        first_token_logits = output.scores[0] if output.scores else None
        results = []
        for i, text in enumerate(decoded_texts):
            choices_logits = None
            if first_token_logits is not None:
                choices_logits = {
                    cand_str: first_token_logits[i, cand_id].item()
                    for cand_str, cand_id in self.choice2token_id.items()
                }
            results.append({"pred_text": text.strip(), "choices_logits": choices_logits})

        return results

def get_wrapper(model_name: str, cache_dir: str = None, **kwargs) -> BaseWrapper:

    torch.cuda.empty_cache()

    # Logic to auto-detect architecture
    if model_name.lower() == 'finetuned':
        return TextSeq2SeqWrapper(model_name="google/flan-t5-xl", cache_dir=cache_dir)

    elif model_name.lower() == "qwen":
        return MultimodalCausalWrapper(model_name="Qwen/Qwen3-VL-8B-Instruct", cache_dir=cache_dir, **kwargs)
    
    else:
        return MultimodalCausalWrapper(model_name, cache_dir=cache_dir, **kwargs)
    
def filter_choices(choices_str: str, pred_set: list[str]):
    """Filter the choices string to only include those in pred_set.
    X-s IMPORTANT: if 'A' (reserved label) is not in the pred_set (unlikely)
      another label will be mapped to 'A'"""
    mask = list(ord(label) - 65 for label in pred_set)
    choices_strs = choices_str.splitlines()
    filtered_choices_str = '\n'.join(chr(65 + i) + choices_strs[i][1:] for i in mask)
    # filtered_choices_str = '\n'.join(chr(65 + shift) + choices_strs[i][1:] for shift, i in enumerate(mask))
    
    return filtered_choices_str


def re_evaluate_with_oracle(test_dict:Dict[str, MultiChoiceDataset], df:pd.DataFrame, answer_df:pd.DataFrame,model, tokenizer, shrinkage=0, device='cuda') -> pd.DataFrame:
    prompt_template = Template(re_eval_prompt_template)
    idx_split_map = {i:split_name for i, split_name in enumerate(test_dict.keys())}
    # Align answers to the filtered df order
    answers = answer_df.reindex(df.index)['oracle_answer'].values
    outputs = []

    num_improved = 0
    num_worse = 0
    new_correct = 0
    for pos, (_, row) in enumerate(tqdm(df.iterrows(), desc="Re-evaluating with oracle...", total=len(df))):
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
        processors = LogitsProcessorList([ShrinkageTokenProcessor([choice2token_id['A']], shrinkage=shrinkage)])
        with torch.inference_mode():
                    out = model.generate(
                        **model_input,
                        eos_token_id=model.config.eos_token_id,
                        max_new_tokens=25,
                        return_dict_in_generate=True,
                        output_scores=True,
                        logits_processor=processors,
                    )
        decoded = tokenizer.decode(out["sequences"][0], skip_special_tokens=True)
        choices_probs = choices_probabilities(out["scores"][0], pred_set=row['pred_set'])
        pred_label, pred_action, pred_value = parse_output(decoded)
        # print(f"Choices probs (full): {choices_probs0}")
        # print(f"Choices probs (filtered): {choices_probs}")
        # if pos >= 0:
        #     # print(html_context)
        #     # print(prompt)
        #     print(f"Decoded: {decoded}, pred: {pred_label}, label: {row['target_label']}, Original pred: {row['pred_label']}")
        #     print(f'pred set: {row["pred_set"]}, choices_probs: {choices_probs}')
        #     print(f"-------------------------")
        #     print(row['choices_probs'])
        #     return


        correct = int(pred_label == row['target_label'])
        new_correct += correct
        num_improved += correct and (row['pred_label'] != row['target_label'])
        num_worse += (not correct) and (row['pred_label'] == row['target_label'])

        if pos % 100 == 0:
            tqdm.write(f"{new_correct}/{pos+1}({new_correct/(pos+1):.2f})| Improved: {num_improved} | Worse: {num_worse}")

        # return
        
        outputs.append(
                [
                    relative_idx,
                    row['annotation_id'],
                    row['action_uid'],
                    pred_label, pred_action, pred_value,
                    row['target_label'],
                    row['target_text'],
                    choices_probs,
                    choices_probs.get(pred_label, 0),
                    row['test_split'],
                    decoded
                ]  
        )  

        # Tight memory management
        del out
        # torch.cuda.empty_cache()
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
                    MultiChoiceDataset.choice2token_id,
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
                logits_first, pred_set=row["pred_set"]
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

class ShrinkageTokenProcessor(LogitsProcessor):
    def __init__(self, token_ids, shrinkage=0, device=None):
        self.shrinkage = shrinkage # Shrinkage penalty
        ids = sorted(set(token_ids)) # Makes sure that shrinkage applied only once
        self.shrink_ids = torch.tensor(ids, dtype=torch.long, device=device)

        # Indicator for making sure that shrinkage is applied only on the first token generation
        self.is_first_token = True

    def __call__(self, input_ids, scores):
        if not self.is_first_token:
            return scores
        
        
        # min_vals = scores.min(dim=1, keepdim=True).values
        # scores = scores - min_vals + 1e-6
        
        scores[:, self.shrink_ids] -= self.shrinkage
        # scores[:, self.shrink_ids] = scores[:, self.shrink_ids] * (1.0 - self.shrinkage)
        self.is_first_token = False
        return scores


def generate_batch_records(model: BaseWrapper, batch: dict, temperature=1, shrinkage=0, use_few_shots: bool = False, **kwargs) -> list[dict]:
    prompts = batch["prompt"]
    images = batch.get("screenshot_image", None)
    outputs = model.generate(prompts, images=images, use_few_shots=use_few_shots)

    records = []
    for i, out in enumerate(outputs):
        records.append(
            {
                "annotation_id": batch["annotation_id"][i],
                "action_uid": batch["action_uid"][i],
                "pred_text": out['pred_text'],
                "target_text": batch['target_text'][i],
                "choices_logits": out.get('choices_logits', None),
            }
        )
    
    
    return records


def collect_prompt_predictions(
    model: BaseWrapper,
    datasets: dict,
    batch_size: int = 3,
    backup: bool = False,
    backup_path: str = "prompt_predictions_backup.pkl",
    **kwargs,
) -> pd.DataFrame:
    """
    Iterate over prompt datasets (e.g., MultiChoiceDatasetPrompt per split),
    generate model outputs, and return a single DataFrame of records.
    """
    records: list[dict] = []
    i = 0
    first = False
    for split_name, dataset in datasets.items():
        print(f"Processing split: {split_name} ({len(dataset)} samples)")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=raw_data_collate_fn,
        )
        for batch in tqdm(loader):
            i += len(batch)
            batch_records = generate_batch_records(model, batch, **kwargs)
            for rec in batch_records:
                rec["test_split"] = split_name
            records.extend(batch_records)
            if backup and (i % 100 == 0) or (i % 10 == 0 and not first):
                first = True
                pd.DataFrame(records).to_pickle(backup_path)
    
    return pd.DataFrame(records)
    

def batch_generate(model, tokenizer, loader, split_name = None, temperature=1, shrinkage=0):
    records = []
    test_split_idx = MultiChoiceDataset.split2id[split_name] if split_name else -1
    for batch in loader:
        processors = LogitsProcessorList([ShrinkageTokenProcessor([choice2token_id['A']], shrinkage=shrinkage)])
        with torch.inference_mode():
            out = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=25,
                eos_token_id=model.config.eos_token_id,
                do_sample=False,
                output_scores = True,
                return_dict_in_generate=True,
                logits_processor=processors,
            )
            out_texts = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)

            # Labels may contain ignore_index (-100); replace with pad id before decoding
            labels = batch["labels"]
            pad_id = tokenizer.pad_token_id or 0
            labels_to_decode = labels.clone()
            labels_to_decode[labels_to_decode < 0] = pad_id
            target_texts = tokenizer.batch_decode(labels_to_decode, skip_special_tokens=True)
            
            # in_text = tokenizer.batch_decode(batch['input_ids'],skip_special_tokens=True)
            
            for i in range(len(out_texts)):
                records.append(
                    {
                        "relative_idx": batch["ids"][i],
                        "annotation_id": batch["annotation_ids"][i],
                        "action_uid": batch["action_uids"][i],
                        "output_text": out_texts[i],
                        "target_text": target_texts[i],
                        "choices_logits": utils.helpers.choices_logits(out['scores'][0][i], choice2token_id),
                        "test_split" : test_split_idx,
                    }
                )

    return records

def run_split_random(split_name, ds, model, tokenizer, batch_size=7, num_iterations=1, temperature=1, shrinkage=0):
    all_records  = []
    for rand_idx in range(num_iterations):
        # Clear GPU memory before next iteration
        torch.cuda.empty_cache()
        gc.collect()
        # ds.seed = rand_idx #####
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=multichoice_collate_fn)
        results = batch_generate(model, tokenizer, loader, split_name=split_name, temperature=temperature, shrinkage=shrinkage)
        records = pd.DataFrame(results)
        records['rand_idx'] = rand_idx
        all_records.append(records)

    final_df = pd.concat(all_records, ignore_index=True)
    return final_df

def evaluate_splits(data_sets:dict, model, tokenizer, batch_size=7, num_iterations=1, shrinkage=0):
    all_splits = []
    for split_name, ds in data_sets.items():
        print(f"Evaluating split: {split_name}")
        split_df = run_split_random(split_name, ds, model, tokenizer, batch_size=batch_size, num_iterations=num_iterations, shrinkage=shrinkage)
        all_splits.append(split_df)
    
    final_df = pd.concat(all_splits, ignore_index=True)
    return final_df
