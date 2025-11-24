import torch
import re 
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
