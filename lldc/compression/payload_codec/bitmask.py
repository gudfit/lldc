# lldc/compression/payload_codec/bitmask.py

from __future__ import annotations
from typing import Iterable, List
import math


def pack_bitmask(keep_flags: Iterable[bool]) -> bytes:
    out = bytearray()
    byte, count = 0, 0
    for flag in keep_flags:
        if flag:
            byte |= 1 << (count % 8)
        count += 1
        if count % 8 == 0:
            out.append(byte)
            byte = 0
    if count % 8:
        out.append(byte)
    return bytes(out)


def unpack_bitmask(b: bytes, n_tokens: int) -> List[bool]:
    if len(b) * 8 < n_tokens:
        raise ValueError(f"bitmask too short: {len(b) * 8} bits for {n_tokens} tokens")
    flags: List[bool] = []
    for i in range(n_tokens):
        byte = b[i // 8]
        bit = (byte >> (i % 8)) & 1
        flags.append(bool(bit))
    return flags


def cost_bits(n_tokens: int) -> int:
    return ((n_tokens + 7) // 8) * 8
