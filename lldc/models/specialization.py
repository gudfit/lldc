# lldc/models/specialization.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal
import torch
import torch.nn as nn

Saliency = Literal["magnitude"]


def _collect_ffn_channels(module: nn.Module) -> torch.Tensor:
    inter = getattr(module, "intermediate", None)
    out = getattr(module, "output", None)
    if inter is None or out is None:
        return torch.empty(0)
    dense_i = getattr(inter, "dense", None)
    dense_o = getattr(out, "dense", None)
    if dense_i is None or dense_o is None:
        return torch.empty(0)
    w = dense_i.weight.detach()
    sal = w.abs().mean(dim=1)
    return sal


def _collect_gpt2_mlp_channels(mlp: nn.Module) -> torch.Tensor:
    c_fc = getattr(mlp, "c_fc", None)
    c_proj = getattr(mlp, "c_proj", None)
    if c_fc is None or c_proj is None:
        return torch.empty(0)
    W_in = c_fc.weight.detach()
    W_out = c_proj.weight.detach()
    sal_in = W_in.abs().mean(dim=0)
    sal_out = W_out.abs().mean(dim=1)
    return sal_in + sal_out


def prune_ffn_block(module: nn.Module, keep_mask: torch.Tensor):
    inter = module.intermediate.dense
    out = module.output.dense
    with torch.no_grad():
        inter.weight = nn.Parameter(inter.weight[keep_mask, :].clone())
        inter.bias = nn.Parameter(inter.bias[keep_mask].clone())
        out.weight = nn.Parameter(out.weight[:, keep_mask].clone())
    if hasattr(module, "config"):
        module.config.intermediate_size = int(keep_mask.sum().item())


def prune_gpt2_mlp_block(mlp: nn.Module, keep_mask: torch.Tensor):
    c_fc = getattr(mlp, "c_fc", None)
    c_proj = getattr(mlp, "c_proj", None)
    if c_fc is None or c_proj is None:
        return
    with torch.no_grad():
        c_fc.weight = nn.Parameter(c_fc.weight[:, keep_mask].clone())
        if getattr(c_fc, "bias", None) is not None:
            c_fc.bias = nn.Parameter(c_fc.bias[keep_mask].clone())
        c_proj.weight = nn.Parameter(c_proj.weight[keep_mask, :].clone())
    new_nf = int(keep_mask.sum().item())
    if hasattr(c_fc, "nf"):
        c_fc.nf = new_nf
    cfg = getattr(mlp, "config", getattr(getattr(mlp, "parent", None), "config", None))
    if cfg is not None and hasattr(cfg, "n_inner") and isinstance(cfg.n_inner, int):
        cfg.n_inner = new_nf


def _num_heads_and_size(attn_module: nn.Module) -> Tuple[int, int]:
    cfg = getattr(
        attn_module,
        "config",
        getattr(attn_module, "self", getattr(attn_module, "num_heads", None)),
    )
    try:
        n_heads = attn_module.num_attention_heads
        head_dim = attn_module.attention_head_size
    except Exception:
        n_heads = attn_module.num_heads
        head_dim = attn_module.head_dim
    return int(n_heads), int(head_dim)


def head_saliency(attn_module: nn.Module) -> torch.Tensor:
    if hasattr(attn_module, "num_heads"):
        H = attn_module.num_heads
    elif hasattr(attn_module, "self") and hasattr(
        attn_module.self, "num_attention_heads"
    ):
        H = attn_module.self.num_attention_heads
    elif hasattr(attn_module, "num_attention_heads"):
        H = attn_module.num_attention_heads
    else:
        raise AttributeError(
            f"Could not determine number of attention heads for module of type {type(attn_module).__name__}"
        )
    if hasattr(attn_module, "c_attn"):
        W = attn_module.c_attn.weight.data
        if W.size(1) % 3 == 0:
            d = W.size(1) // 3
            q, k, v = W[:, :d], W[:, d : 2 * d], W[:, 2 * d :]
            D = d // H
            sal = [
                (
                    q[:, h * D : (h + 1) * D].abs().mean()
                    + k[:, h * D : (h + 1) * D].abs().mean()
                    + v[:, h * D : (h + 1) * D].abs().mean()
                ).item()
                for h in range(H)
            ]
            return torch.tensor(sal)
        else:
            d = W.size(0) // 3
            q, k, v = W[:d, :], W[d : 2 * d, :], W[2 * d :, :]
            D = q.size(0) // H
            sal = [
                (
                    q[h * D : (h + 1) * D, :].abs().mean()
                    + k[h * D : (h + 1) * D, :].abs().mean()
                    + v[h * D : (h + 1) * D, :].abs().mean()
                ).item()
                for h in range(H)
            ]
            return torch.tensor(sal)
    else:
        q = attn_module.self.query.weight.data
        k = attn_module.self.key.weight.data
        v = attn_module.self.value.weight.data
        D = q.size(0) // H
        sal = [
            (
                q[h * D : (h + 1) * D, :].abs().mean()
                + k[h * D : (h + 1) * D, :].abs().mean()
                + v[h * D : (h + 1) * D, :].abs().mean()
            ).item()
            for h in range(H)
        ]
        return torch.tensor(sal)


def get_base_model(model: nn.Module):
    return getattr(
        model, "bert", getattr(model, "roberta", getattr(model, "distilbert", None))
    )


def structured_prune(
    model: nn.Module, level: float, drop_heads: bool = True, drop_ffn: bool = True
) -> Dict[str, int]:
    dropped = {"heads": 0, "ffn_channels": 0}
    base_model = get_base_model(model)
    is_bert_like = base_model is not None and hasattr(base_model, "encoder")
    is_gpt2_like = hasattr(model, "transformer") and hasattr(model.transformer, "h")
    if not (is_bert_like or is_gpt2_like):
        return dropped
    blocks = (
        list(base_model.encoder.layer) if is_bert_like else list(model.transformer.h)
    )
    if drop_heads:
        if is_bert_like and hasattr(model, "prune_heads"):
            heads_to_prune: Dict[int, List[int]] = {}
            embed_dim = int(getattr(model.config, "hidden_size", 0) or 0)
            for li, blk in enumerate(blocks):
                attn = getattr(blk, "attention", None)
                if attn is None:
                    continue
                sal = head_saliency(attn)
                H = sal.numel()
                if H <= 0:
                    continue
                k_drop = int(level * H)
                if k_drop <= 0 or k_drop >= H:
                    continue
                valid_keeps = (
                    [d for d in range(1, H + 1) if embed_dim % d == 0]
                    if embed_dim > 0
                    else [H]
                )
                target_keep = H - k_drop
                k_keep = min(valid_keeps, key=lambda x: abs(x - target_keep))
                new_k_drop = H - k_keep
                if new_k_drop <= 0 or new_k_drop >= H:
                    continue
                to_drop = torch.topk(sal, new_k_drop, largest=False).indices.tolist()
                if to_drop:
                    heads_to_prune[li] = to_drop
                    dropped["heads"] += len(to_drop)
            if heads_to_prune:
                model.prune_heads(heads_to_prune)
        elif is_gpt2_like:
            for li, blk in enumerate(blocks):
                attn = getattr(blk, "attn", None)
                if attn is None or not hasattr(attn, "prune_heads"):
                    continue
                sal = head_saliency(attn)
                H = attn.num_heads
                k_drop = int(level * H)
                if k_drop <= 0 or k_drop >= H:
                    continue
                embed_dim = attn.embed_dim
                target_k_keep = H - k_drop
                divisors = [i for i in range(1, H + 1) if embed_dim % i == 0]
                if not divisors:
                    continue
                k_keep = min(divisors, key=lambda x: abs(x - target_k_keep))
                new_k_drop = H - k_keep
                if new_k_drop <= 0:
                    continue
                to_drop = torch.topk(sal, new_k_drop, largest=False).indices.tolist()
                if not to_drop:
                    continue
                attn.prune_heads(set(to_drop))
                dropped["heads"] += len(to_drop)
    if drop_ffn:
        for blk in blocks:
            if is_bert_like and hasattr(blk, "intermediate"):
                sal = _collect_ffn_channels(blk)
                if sal.numel() > 0:
                    k_drop = int(level * sal.numel())
                    if 0 < k_drop < sal.numel():
                        thr = torch.topk(sal, k_drop, largest=False).values.max()
                        keep = sal > thr
                        prune_ffn_block(blk, keep)
                        dropped["ffn_channels"] += int((~keep).sum().item())
            elif is_gpt2_like and hasattr(blk, "mlp"):
                mlp = blk.mlp
                if hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
                    sal = _collect_gpt2_mlp_channels(mlp)
                    if sal.numel() > 0:
                        k_drop = int(level * sal.numel())
                        if 0 < k_drop < sal.numel():
                            thr = torch.topk(sal, k_drop, largest=False).values.max()
                            keep = sal > thr
                            prune_gpt2_mlp_block(mlp, keep)
                            dropped["ffn_channels"] += int((~keep).sum().item())
    return dropped
