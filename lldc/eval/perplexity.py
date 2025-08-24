# lldc/eval/perplexity.py

from __future__ import annotations
from typing import List, Dict

import math
import torch
import numpy as np
import torch.nn.functional as F


def ppl_causal(model, tok, texts: List[str], device="cuda") -> float:
    model.to(device).eval()
    losses = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt").to(device)
            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())
    H_bits = float(np.mean(losses)) / math.log(2)
    return float(2**H_bits)


def pppl_mlm(model, tok, texts: List[str], device="cuda") -> float:
    model.to(device).eval()
    log2ps = []
    mask_id = tok.mask_token_id
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").to(device)["input_ids"][0]
            L = ids.size(0)
            sum_log2p = 0.0
            for i in range(L):
                masked = ids.clone()
                masked[i] = mask_id
                logits = model(input_ids=masked.unsqueeze(0)).logits[0, i]
                p = torch.softmax(logits, dim=-1)[ids[i]].item()
                sum_log2p += math.log(p + 1e-12, 2)
            log2ps.append(sum_log2p / max(1, L))
    H_bits = -float(np.mean(log2ps))
    return float(2**H_bits)


def compute_ppl(model, tok, texts: List[str], device="cuda") -> Dict:
    if hasattr(model, "generate"):
        return {"ppl": ppl_causal(model, tok, texts, device)}
    else:
        return {"pppl": pppl_mlm(model, tok, texts, device)}


def ar_bpc(model, tok, texts: List[str], device="cuda") -> float:
    model.to(device).eval()
    total_bits = 0.0
    total_chars = 0
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt").to(device)
            ids = enc["input_ids"][0]
            if ids.numel() < 2:
                total_chars += len(t)
                continue
            logits = model(input_ids=ids.unsqueeze(0)).logits[0]
            logp = F.log_softmax(logits[:-1], dim=-1)
            true_next = ids[1:].unsqueeze(-1)
            nats = -logp.gather(-1, true_next).squeeze(-1).sum().item()
            bits = nats / math.log(2.0)
            total_bits += bits
            total_chars += len(t)
    return total_bits / max(1, total_chars)
