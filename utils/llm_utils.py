import torch
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor
from string import Template
import pandas as pd
from tqdm.auto import tqdm
import re 
from utils.prompts import oracle_prompt_template
from PIL import Image
from utils import log_response
from typing import Dict, Any, Optional, Tuple


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

    return letter, action, value

def tensorize_item(item: Dict[str, Any], device: str):
    """
    Convert the model_input dict returned by MultiChoiceDataset.__getitem__
    (lists of ints) into tensors appropriate for model generate.
    """
    input_ids = torch.LongTensor(item["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.LongTensor(item["attention_mask"]).unsqueeze(0).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# utils.reload('utils')



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


class Oracle:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = None,
        device: str = None,
        dtype=torch.float16,
    ):
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=dtype,
            device_map="auto" if device is None else None,
        )
        if device is not None:
            self.model.to(device)
        self.device = device or 'cuda'

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
        image =  image.convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},  # placeholder; processor will insert the image tokens
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=[image.convert("RGB")], text=[text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )
            

        text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return text.strip()