# lldc/compression/masking_policies.py

from __future__ import annotations
from typing import Literal, Dict
import torch
from .predictive_masking import topk_global_mask, entropy_equalisation_mask

PolicyType = Literal["topk_global", "entropy_equalisation"]


def choose_mask(
    policy: PolicyType, surprisal_bits: torch.Tensor, keep_fraction: float
) -> torch.Tensor:
    if policy == "topk_global":
        return topk_global_mask(surprisal_bits, keep_fraction)
    elif policy == "entropy_equalisation":
        return entropy_equalisation_mask(surprisal_bits, keep_fraction)
    else:
        raise ValueError(f"Unknown policy={policy}")
