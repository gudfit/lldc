# lldc/compression/reconstruction.py

from __future__ import annotations
from typing import Any
import torch


@torch.no_grad()
def reconstruct_mlm_text(
    tok: Any,
    mlm: Any,
    input_ids: torch.LongTensor,
    keep_flags: torch.Tensor,
) -> str:
    if input_ids.numel() == 0:
        return ""
    device = next(mlm.parameters()).device
    masked = input_ids.clone().to(device)
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("Tokenizer must have a [MASK] token for MLM reconstruction.")
    masked[~keep_flags] = int(mask_id)
    outputs = mlm(input_ids=masked.unsqueeze(0))
    logits = outputs.logits[0]  # [T, V]
    preds = torch.argmax(logits, dim=-1)
    recon = input_ids.clone().to(device)
    recon[~keep_flags] = preds[~keep_flags]
    return tok.decode(recon, skip_special_tokens=True)
