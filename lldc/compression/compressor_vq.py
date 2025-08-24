# lldc/compression/compressor_vq.py
from __future__ import annotations
from typing import Any, List, Dict
import torch
from datasets import load_dataset
from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
)


def compress_dataset_with_vq(cfg: Any) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = (
        str(getattr(cfg, "model_ckpt", "")).strip()
        or str(getattr(getattr(cfg, "model", None), "ckpt", "")).strip()
        or str(cfg.model.pretrained_name)
    )
    model, tok = train_vq_joint(
        base_model_name=base_model,
        dataset_name=cfg.data.source.hf_dataset,
        dataset_config=cfg.data.source.hf_config,
        text_field=cfg.data.processing.text_field,
        max_length=cfg.data.processing.max_length,
        layer_after=cfg.stage2.vq.bottleneck.layer_after,
        codebook_size=int(cfg.stage2.vq.codebook_sizes[0]),
        lr=5e-5,
        epochs=int(getattr(cfg.stage2.vq.index_lm, "epochs", 2) or 2),
        beta=cfg.stage2.vq.bottleneck.commitment_beta,
    )
    model.eval().to(device)
    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    split_map = cfg.data.source.split_map
    text_field = cfg.data.processing.text_field
    train_split = ds[split_map.train]
    test_split = ds[split_map.test]
    idx_train: List[List[int]] = []
    for ex in train_split.select(range(min(10000, len(train_split)))):
        txt = ex.get(text_field) or ""
        if not txt:
            continue
        ids = tok(
            txt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.data.processing.max_length,
        )["input_ids"].to(device)
        seq_idx = encode_indices(model, ids)[0].tolist()
        if seq_idx:
            idx_train.append(seq_idx)
    K = int(cfg.stage2.vq.codebook_sizes[0])
    lm = train_index_lm(
        idx_train,
        K=K,
        hidden=int(cfg.stage2.vq.index_lm.hidden_size),
        layers=int(cfg.stage2.vq.index_lm.layers),
        epochs=int(cfg.stage2.vq.index_lm.epochs),
    )
    total_bits = 0.0
    total_tokens = 0
    total_chars = 0
    for ex in test_split.select(range(min(5000, len(test_split)))):
        txt = ex.get(text_field) or ""
        if not txt:
            continue
        ids = tok(
            txt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.data.processing.max_length,
        )["input_ids"].to(device)
        seq_idx = encode_indices(model, ids)[0].tolist()
        total_bits += cross_entropy_bits_index_stream(lm, seq_idx)
        total_tokens += max(0, len(seq_idx) - 1)
        total_chars += len(txt)
    bpt = total_bits / max(1, total_tokens)
    bpc = total_bits / max(1, total_chars)
    return {
        "index_bits": float(total_bits),
        "tokens": int(total_tokens),
        "chars": int(total_chars),
        "bpt": float(bpt),
        "bpc": float(bpc),
    }
