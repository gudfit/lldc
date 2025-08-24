# lldc/metrics/pm_bpt.py

from __future__ import annotations
import math
from typing import Iterable, Tuple, List, Optional

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def bits_per_symbol(vocab_size: int, entropy_coded: bool = False) -> float:
    if vocab_size <= 1:
        return 0.0
    val = math.log2(float(vocab_size))
    return val if entropy_coded else float(math.ceil(val))


def pm_bpt_from_fraction(
    keep_fraction: float,
    vocab_size: int,
    entropy_coded: bool = False,
) -> float:
    keep_fraction = max(0.0, min(1.0, float(keep_fraction)))
    return keep_fraction * bits_per_symbol(vocab_size, entropy_coded=entropy_coded)


def _tokenize_counts(
    texts: Iterable[str], tokenizer_name: Optional[str]
) -> Tuple[int, int]:
    total_chars = 0
    total_tokens = 0
    tok = None
    if tokenizer_name and AutoTokenizer is not None:
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    for t in texts:
        total_chars += len(t)
        if tok is None:
            total_tokens += len(t.split())
        else:
            total_tokens += len(tok.encode(t))
    return total_tokens, total_chars


def pm_bpt_bpc_from_fraction(
    texts: Iterable[str],
    keep_fraction: float,
    vocab_size: int,
    tokenizer_name: Optional[str] = None,
    entropy_coded: bool = False,
) -> Tuple[float, float, float]:
    tot_tok, tot_char = _tokenize_counts(texts, tokenizer_name)
    tpc = (tot_tok / tot_char) if tot_char > 0 else 0.0
    bpt = pm_bpt_from_fraction(keep_fraction, vocab_size, entropy_coded=entropy_coded)
    bpc = bpt * tpc
    return float(bpt), float(bpc), float(tpc)
