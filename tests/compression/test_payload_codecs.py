# tests/compression/test_payload_codecs.py

import importlib.util
import pytest

from lldc.compression.payload_codec import bitmask as bm
from lldc.compression.payload_codec import rle_elias as rle


def test_bitmask_pack_unpack_roundtrip_and_cost_bits():
    flags = [True, False, True, True, False, True, False, False, True]
    payload = bm.pack_bitmask(flags)
    rt = bm.unpack_bitmask(payload, len(flags))
    assert rt == flags
    assert bm.cost_bits(len(flags)) == len(payload) * 8
    with pytest.raises(ValueError):
        bm.unpack_bitmask(payload[:1], len(flags) + 16)


def test_rle_elias_roundtrip_various_patterns():
    cases = [
        [],
        [True] * 10,
        [False] * 10,
        [True, False] * 5,
        [True, True, False, False, False, True, True, True, False],
    ]
    for flags in cases:
        enc = rle.encode_rle_elias(flags)
        dec = rle.decode_rle_elias(enc, len(flags))
        assert dec == flags


def test_rle_elias_decode_empty_produces_false_padding():
    dec = rle.decode_rle_elias(b"", 7)
    assert dec == [False] * 7


ar_spec = importlib.util.find_spec("constriction")


@pytest.mark.skipif(ar_spec is None, reason="constriction not installed")
def test_arithmetic_encode_decode_roundtrip():
    from lldc.compression.payload_codec import arithmetic as ac

    symbols = [1, 0, 2, 1]
    probs = [
        [0.1, 0.7, 0.2],
        [0.6, 0.3, 0.1],
        [0.2, 0.2, 0.6],
        [0.3, 0.5, 0.2],
    ]
    payload = ac.encode_with_probs(symbols, probs)
    out = ac.decode_with_probs(payload, probs)
    assert out == symbols
    assert ac.payload_num_bits(payload) == len(payload) * 8
