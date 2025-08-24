# lldc/decompression/vq_decompress.py

from __future__ import annotations
from typing import List, Optional
import torch
from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper


@torch.no_grad()
def reconstruct_tokens_from_indices(
    model: VQBottleneckWrapper,
    indices: List[int],
    start_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_new_tokens: int = 0,
    margin_factor: float = 1.05,
) -> List[int]:
    device = next(model.parameters()).device
    idx = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    toks = model.decode_from_indices(idx)[0].tolist()
    return [int(t) for t in toks]
