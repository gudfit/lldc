# lldc/compression/payload_codec/rle_elias.py

from __future__ import annotations
from typing import Iterable, List, Tuple


def elias_gamma_encode(n: int) -> List[int]:
    assert n >= 1
    b = bin(n)[2:]
    return [0] * (len(b) - 1) + [int(x) for x in b]


def elias_gamma_decode(bits: List[int], pos: int) -> Tuple[int, int]:
    z = 0
    while pos < len(bits) and bits[pos] == 0:
        z += 1
        pos += 1
    if pos >= len(bits):
        raise ValueError("truncated gamma code")
    val = 1
    for _ in range(z):
        pos += 1
        if pos >= len(bits):
            raise ValueError("truncated gamma payload")
    pos -= z
    pos0 = pos
    pos = pos0 + 1
    val_bits = [1]
    for _ in range(z):
        if pos >= len(bits):
            raise ValueError("truncated gamma tail")
        val_bits.append(bits[pos])
        pos += 1
    val = int("".join(str(x) for x in val_bits), 2)
    return val, pos


def rle_runs(flags: Iterable[bool]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    it = iter(flags)
    try:
        first = next(it)
    except StopIteration:
        return runs
    cur_symbol = 1 if first else 0
    cur_len = 1
    for f in it:
        s = 1 if f else 0
        if s == cur_symbol:
            cur_len += 1
        else:
            runs.append((cur_symbol, cur_len))
            cur_symbol, cur_len = s, 1
    runs.append((cur_symbol, cur_len))
    return runs


def encode_rle_elias(flags: Iterable[bool]) -> bytes:
    runs = rle_runs(flags)
    bits: List[int] = []
    if not runs:
        return bytes()
    bits.append(runs[0][0])
    for _, length in runs:
        bits.extend(elias_gamma_encode(max(1, length)))
    out = bytearray()
    byte = 0
    cnt = 0
    for bit in bits:
        if bit:
            byte |= 1 << (cnt % 8)
        cnt += 1
        if cnt % 8 == 0:
            out.append(byte)
            byte = 0
    if cnt % 8:
        out.append(byte)
    return bytes(out)


def decode_rle_elias(b: bytes, n_tokens: int) -> List[bool]:
    bits: List[int] = []
    for i in range(len(b) * 8):
        byte = b[i // 8]
        bits.append((byte >> (i % 8)) & 1)
    pos = 0
    if not bits:
        return [False] * n_tokens
    start_symbol = bits[pos]
    pos += 1
    flags: List[bool] = []
    sym = start_symbol
    while len(flags) < n_tokens and pos < len(bits):
        length, pos = elias_gamma_decode(bits, pos)
        flags.extend([bool(sym)] * length)
        sym ^= 1
    if len(flags) < n_tokens:
        flags.extend([False] * (n_tokens - len(flags)))
    return flags[:n_tokens]
