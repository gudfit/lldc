# lldc/oracles/measuring_tokenizer.py

from __future__ import annotations
from transformers import AutoTokenizer


def get_measurement_tokenizer(name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    return tok
