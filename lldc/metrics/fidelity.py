# lldc/metrics/fidelity.py

from __future__ import annotations
from typing import Optional, List, Any
from Levenshtein import distance as levenshtein_distance
import evaluate
from sacrebleu.metrics import CHRF
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def character_level_fidelity(orig: str, recon: str) -> float:
    if len(orig) == len(recon):
        equal = sum(1 for a, b in zip(orig, recon) if a == b)
        return 100.0 * equal / max(1, len(orig))
    d = levenshtein_distance(orig, recon)
    denom = max(len(orig), len(recon), 1)
    return 100.0 * (1.0 - d / denom)


def chrf_score(orig: str, recon: str, order: int = 6) -> float:
    chrf = CHRF(word_order=0, char_order=order)
    return float(chrf.sentence_score(recon, [orig]).score)


def _free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, total = torch.cuda.mem_get_info()
        return free / (1024**3)
    except Exception:
        return 0.0


def bertscore_f1(
    orig: str,
    recon: str,
    model_type: str = "roberta-large",
    batch_size: Optional[int] = None,
    device: str = "cpu",
    metric_obj: Optional[Any] = None,
) -> float:
    """Returns BERTScore F1 on a 0–100 scale (NOT 0–1)."""
    metric = metric_obj if metric_obj is not None else evaluate.load("bertscore")
    prefer_cuda = (device or "").startswith("cuda") and torch.cuda.is_available()
    dev = "cpu"
    if prefer_cuda and _free_vram_gb() > (2.2 if "large" in model_type else 1.2):
        dev = "cuda:0"
    try:
        res = metric.compute(
            references=[orig],
            predictions=[recon],
            model_type=model_type,
            batch_size=batch_size or 8,
            device=dev,
        )
        return float(res["f1"][0] * 100.0)
    except RuntimeError as e:
        if "CUDA" in str(e):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            res = metric.compute(
                references=[orig],
                predictions=[recon],
                model_type=model_type,
                batch_size=1,
                device="cpu",
            )
            return float(res["f1"][0] * 100.0)
        raise


def semantic_span_fidelity(
    orig: str,
    recon: str,
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    span_chars: int = 256,
    stride_chars: int = 128,
    sbert_model_obj: Optional[SentenceTransformer] = None,
) -> float:
    """
    Mean cosine similarity mapped to 0–100 across overlapping spans.
    """

    def _spans(s: str) -> List[str]:
        if not s:
            return []
        spans, i = [], 0
        while i < len(s):
            spans.append(s[i : i + span_chars])
            if i + span_chars >= len(s):
                break
            i += stride_chars
        return spans

    A = _spans(orig)
    B = _spans(recon)
    if not A and not B:
        return 0.0
    L = max(len(A), len(B))
    if len(A) < L and A:
        A += [A[-1]] * (L - len(A))
    if len(B) < L and B:
        B += [B[-1]] * (L - len(B))

    if sbert_model_obj is not None:
        enc = sbert_model_obj
        dev = str(enc.device)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.float16} if dev == "cuda" else {}
        enc = SentenceTransformer(
            sbert_model_name, device=dev, model_kwargs=model_kwargs
        )

    bs = 32 if dev.startswith("cuda") else 8
    emA = (
        enc.encode(
            A,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=bs,
        )
        if A
        else np.zeros((L, 768))
    )
    emB = (
        enc.encode(
            B,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=bs,
        )
        if B
        else np.zeros((L, 768))
    )
    sims = (emA * emB).sum(axis=1)
    if sims.size == 0:
        return 0.0
    return float(((float(np.mean(sims)) + 1.0) / 2.0) * 100.0)
