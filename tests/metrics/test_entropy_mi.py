# tests/metrics/test_entropy_mi.py


import math
from lldc.metrics.entropy_mi import (
    unigram_entropy_bits_per_symbol,
    avg_token_length_bytes,
    entropy_per_byte,
    mutual_information_adjacent,
)


def test_unigram_entropy_zero_for_constant():
    tokens = ["x"] * 100
    assert unigram_entropy_bits_per_symbol(tokens) == 0.0


def test_unigram_entropy_about_one_bit_for_fair_binary():
    tokens = ["a", "b"] * 50
    h = unigram_entropy_bits_per_symbol(tokens)
    assert 0.99 < h < 1.01


def test_avg_token_length_bytes_handles_utf8():
    assert abs(avg_token_length_bytes(["a", "é"]) - 1.5) < 1e-9


def test_entropy_per_byte_guard_on_zero_length():
    assert entropy_per_byte(10.0, 0.0) == 0.0


def test_mutual_information_adjacent_properties():
    assert mutual_information_adjacent([]) == 0.0
    assert mutual_information_adjacent(["onlyone"]) == 0.0

    constant = ["A"] * 20
    assert mutual_information_adjacent(constant) == 0.0

    alternating = ["A", "B"] * 10
    mi_alt = mutual_information_adjacent(alternating)
    assert mi_alt > 0.5
