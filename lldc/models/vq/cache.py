# lldc/models/vq/cache.py

from __future__ import annotations
from pathlib import Path


def cache_dir_for(model_name: str, K: int) -> Path:
    safe = model_name.replace("/", "-")
    return Path("artifacts/checkpoints/vq") / safe / f"K{int(K)}"


def cached_checkpoint(model_name: str, K: int) -> Path | None:
    d = cache_dir_for(model_name, K)
    ck = d / "model.pt"
    return d if ck.exists() else None
