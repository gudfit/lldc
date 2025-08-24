# lldc/metrics/entropy_mi.py

from __future__ import annotations
from typing import Iterable, Tuple, Dict
from collections import Counter
import math


def unigram_entropy_bits_per_symbol(tokens: Iterable[str] | Iterable[int]) -> float:
    cnt = Counter(tokens)
    n = sum(cnt.values())
    if n == 0:
        return 0.0
    h = 0.0
    for c in cnt.values():
        p = c / n
        h -= p * math.log2(max(p, 1e-12))
    return h


def avg_token_length_bytes(tokens: Iterable[str]) -> float:
    cnt = 0
    total = 0
    for t in tokens:
        total += len(str(t).encode("utf-8"))
        cnt += 1
    return (total / max(cnt, 1)) if cnt else 0.0


def entropy_per_byte(unigram_entropy_bits: float, avg_token_len_bytes: float) -> float:
    if avg_token_len_bytes <= 0:
        return 0.0
    return unigram_entropy_bits / avg_token_len_bytes


def mutual_information_adjacent(tokens: Iterable[str] | Iterable[int]) -> float:
    tokens = list(tokens)
    if len(tokens) < 2:
        return 0.0
    cnt_unigram = Counter(tokens)
    cnt_bigram = Counter(zip(tokens[:-1], tokens[1:]))
    n1 = sum(cnt_unigram.values())
    n2 = sum(cnt_bigram.values())
    mi = 0.0
    for (a, b), c_ab in cnt_bigram.items():
        p_ab = c_ab / n2
        p_a = cnt_unigram[a] / n1
        p_b = cnt_unigram[b] / n1
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * math.log2(p_ab / (p_a * p_b))
    return mi
