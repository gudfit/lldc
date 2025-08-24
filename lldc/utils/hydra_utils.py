# lldc/utils/hydra_utils.py

from __future__ import annotations
from typing import Any
import torch
import math


def auto_dtype(hardware_name: str | None = None) -> str:
    try:
        if not torch.cuda.is_available():
            return "fp32"
        cap = torch.cuda.get_device_capability()
        return "bf16" if cap[0] >= 9 else "fp16"
    except Exception:
        return "fp16"


def auto_scheduler(num_steps: int | None) -> str:
    if not num_steps:
        return "cosine"
    return "linear" if num_steps <= 2000 else "cosine"


def auto_batch_size(seq_len: int, dtype: str, gpu_mem_gb: float = 80.0) -> int:
    bytes_per = 2 if dtype in ("fp16", "bf16") else 4
    est = int((gpu_mem_gb * (1024**3)) / (seq_len * bytes_per * 40_000))
    return max(1, min(64, est))


def resolve_auto(cfg: Any) -> Any:
    c = cfg
    if getattr(c.compute.hardware, "dtype", None) in (None, "auto"):
        c.compute.hardware.dtype = auto_dtype(
            c.compute.name if hasattr(c, "compute") else None
        )
    if getattr(c.model.specialise, "scheduler", "auto") == "auto":
        c.model.specialise.scheduler = auto_scheduler(None)
    if getattr(c.model.specialise, "warmup_ratio", "auto") == "auto":
        c.model.specialise.warmup_ratio = (
            0.06 if c.model.specialise.scheduler == "cosine" else 0.1
        )
    if getattr(c.data.processing, "max_length", None) and c.data.loader.batch_size in (
        None,
        "auto",
    ):
        c.data.loader.batch_size = auto_batch_size(
            c.data.processing.max_length, c.compute.hardware.dtype
        )
    return c
