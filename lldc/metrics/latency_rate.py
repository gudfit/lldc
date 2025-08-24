# lldc/metrics/latency_rate.py

from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path
import json
import math
import torch


def aggregate_cpu_decode_ms_from_payloads(payload_root: Path) -> Dict[str, float]:
    def _scan_one(p: Path) -> List[float]:
        vals: List[float] = []
        if not p.exists():
            return vals
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    j = json.loads(ln)
                except Exception:
                    continue
                v = j.get("cpu_decode_ms", None)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
        return vals

    all_vals: List[float] = []
    pm_vals: List[float] = []
    vq_vals: List[float] = []
    for rp in payload_root.glob("*/recons.jsonl"):
        vals = _scan_one(rp)
        all_vals.extend(vals)
        if rp.parent.name.lower().startswith("pm_"):
            pm_vals.extend(vals)
        elif rp.parent.name.lower().startswith("vq_"):
            vq_vals.extend(vals)

    def _stats(x: List[float]) -> Tuple[float, float]:
        if not x:
            return (0.0, 0.0)
        m = sum(x) / len(x)
        v = sum((a - m) ** 2 for a in x) / max(1, len(x) - 1)
        return float(m), float(math.sqrt(v))

    mean_all, std_all = _stats(all_vals)
    mean_pm, std_pm = _stats(pm_vals)
    mean_vq, std_vq = _stats(vq_vals)

    return {
        "all_mean_ms": mean_all,
        "all_std_ms": std_all,
        "pm_mean_ms": mean_pm,
        "pm_std_ms": std_pm,
        "vq_mean_ms": mean_vq,
        "vq_std_ms": std_vq,
        "num_pm": len(pm_vals),
        "num_vq": len(vq_vals),
        "num_all": len(all_vals),
    }


def _get_transformer_dims(model) -> Tuple[int, int, int, int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return (12, 768, 3072, 12)
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or 768
    n_layer = (
        getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or 12
    )
    n_head = (
        getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None) or 12
    )
    d_ff = getattr(cfg, "intermediate_size", None)
    if d_ff is None:
        d_ff = getattr(cfg, "n_inner", None) or (4 * d_model)
    return int(n_layer), int(d_model), int(d_ff), int(n_head)


def _default_seq_len(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return 1024
    return int(
        getattr(cfg, "n_positions", None)
        or getattr(cfg, "max_position_embeddings", None)
        or 1024
    )


def estimate_transformer_flops_per_token(model, seq_len: int | None = None) -> float:
    n_layer, d_model, d_ff, _ = _get_transformer_dims(model)
    L = int(seq_len or _default_seq_len(model))
    proj_flops = 4.0 * d_model * d_model
    attn_mix = 2.0 * L * d_model
    mlp_flops = 2.0 * d_model * d_ff
    flops_per_token_per_layer = proj_flops + attn_mix + mlp_flops
    total = float(n_layer) * flops_per_token_per_layer
    return float(total)


def estimate_transformer_flops_per_seq(model, seq_len: int) -> float:
    n_layer, d_model, d_ff, _ = _get_transformer_dims(model)
    L = int(seq_len)
    proj_seq = 4.0 * L * d_model * d_model
    attn_seq = 2.0 * (L**2) * d_model
    mlp_seq = 2.0 * L * d_model * d_ff
    per_layer = proj_seq + attn_seq + mlp_seq
    return float(n_layer) * per_layer
