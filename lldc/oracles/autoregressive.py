# lldc/oracles/autoregressive.py

from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


class AROracle:
    def __init__(
        self, name: str = "gpt2-large", device: str | None = None, dtype: str = "fp16"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(dev)
        if dtype == "bf16" and torch.cuda.is_available():
            self.model.to(torch.bfloat16)
        elif dtype == "fp16" and torch.cuda.is_available():
            self.model.to(torch.float16)
        self.model.eval()

    @torch.no_grad()
    def categorical_probs(self, input_ids: torch.LongTensor) -> List[List[float]]:
        logits = self.model(input_ids=input_ids.unsqueeze(0)).logits[0]
        probs = F.softmax(logits, dim=-1)
        return probs.tolist()

    @torch.no_grad()
    def surprisal_bits_for_text(self, text: str) -> List[float]:
        toks = self.tokenizer(text, return_tensors="pt")
        input_ids = toks["input_ids"].to(self.model.device)
        logits = self.model(input_ids=input_ids).logits[0]
        logp = F.log_softmax(logits[:-1], dim=-1)
        true_next = input_ids[0, 1:].unsqueeze(-1)
        nats = -logp.gather(-1, true_next).squeeze(-1)
        bits = (nats / math.log(2.0)).tolist()
        return [float(x) for x in bits]
