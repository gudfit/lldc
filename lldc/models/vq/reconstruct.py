# lldc/models/vq/reconstruct.py
from __future__ import annotations
from typing import Optional
import torch
from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper

def reconstruct_tokens_from_indices(model: VQBottleneckWrapper, indices: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.LongTensor:
    return model.decode_from_indices(indices, attention_mask=attention_mask)

