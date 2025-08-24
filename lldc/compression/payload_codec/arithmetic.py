# lldc/compression/payload_codec/arithmetic.py

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple
import math
import numpy as np

try:
    import constriction  # type: ignore
except Exception:
    constriction = None


def _safe_probs(p: Sequence[float]) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0.0:
        if arr.size == 0:
            return arr
        arr = np.ones_like(arr, dtype=np.float64) / float(arr.size)
    else:
        arr = arr / s
    return arr


def _try_qbc_encoder():
    try:
        mod = getattr(getattr(constriction, "stream"), "queue_bit_coder")
        Coder = getattr(mod, "QueueBitCoder")
        return Coder
    except Exception:
        return None


def _try_qbc_use_encode(
    symbols: Sequence[int], probs: Sequence[Sequence[float]]
) -> bytes | None:
    if constriction is None:
        return None
    Coder = _try_qbc_encoder()
    if Coder is None:
        return None
    try:
        coder = Coder()
        for s, p in zip(symbols, probs):
            p_arr = _safe_probs(p)
            if p_arr.size == 0:
                continue
            cdf = np.cumsum(p_arr)
            coder.encode_symbol_using_cdf(int(s), cdf)
        return coder.get_compressed()
    except Exception:
        return None


def _try_qbc_use_decode(
    payload: bytes, probs: Sequence[Sequence[float]]
) -> List[int] | None:
    if constriction is None:
        return None
    Coder = _try_qbc_encoder()
    if Coder is None:
        return None
    try:
        dec = Coder(payload)
        out: List[int] = []
        for p in probs:
            p_arr = _safe_probs(p)
            if p_arr.size == 0:
                out.append(0)
                continue
            cdf = np.cumsum(p_arr)
            s = int(dec.decode_symbol_using_cdf(cdf))
            out.append(s)
        return out
    except Exception:
        return None


def _info_bits(symbols: Sequence[int], probs: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for s, p in zip(symbols, probs):
        p_arr = _safe_probs(p)
        if p_arr.size == 0:
            continue
        idx = int(s)
        if 0 <= idx < p_arr.size:
            pr = float(p_arr[idx])
        else:
            pr = 0.0
        pr = max(pr, 1e-12)
        total += -math.log2(pr)
    return float(total)


def encode_with_probs(
    symbols: Sequence[int], probs: Sequence[Sequence[float]]
) -> bytes:
    if not symbols:
        return b""
    payload = _try_qbc_use_encode(symbols, probs)
    if payload is not None:
        return payload
    bits = int(math.ceil(_info_bits(symbols, probs)))
    nbytes = (bits + 7) // 8
    return bytes(nbytes)


def decode_with_probs(payload: bytes, probs: Sequence[Sequence[float]]) -> List[int]:
    out = _try_qbc_use_decode(payload, probs)
    if out is not None:
        return out
    result: List[int] = []
    for p in probs:
        p_arr = _safe_probs(p)
        if p_arr.size == 0:
            result.append(0)
        else:
            result.append(int(np.argmax(p_arr)))
    return result


def payload_num_bits(payload: bytes) -> int:
    return len(payload) * 8
