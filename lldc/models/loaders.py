# lldc/models/loaders.py

from __future__ import annotations
from typing import Tuple, Literal
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

Arch = Literal["mlm", "ar"]


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    return tok


def load_model(arch: Arch, name: str, dtype: str = "fp16") -> torch.nn.Module:
    if arch == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(name)
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
    if torch.cuda.is_available():
        if dtype == "bf16":
            model.to(torch.bfloat16)
        elif dtype == "fp16":
            model.to(torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model
