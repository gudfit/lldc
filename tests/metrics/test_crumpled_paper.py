# tests/metrics/test_crumpled_paper.py

import math
from lldc.metrics.crumpled_paper import tcm_pcm_from_surprisal


def test_gap_cost_single_token():
    vocab_size = 50257
    gap = math.log2(max(2, vocab_size))
    s0 = []
    s1 = [3.25]
    m = tcm_pcm_from_surprisal(s0, s1, vocab_size=vocab_size)
    assert abs(m["tcm_bits"] - gap) < 1e-6
    assert abs(m["pcm_bits"] - gap) < 1e-6


def test_alignment_mismatch_and_match_mix():
    vocab_size = 8192
    gap = math.log2(max(2, vocab_size))
    s0 = [1.0, 2.0]
    s1 = [1.0]
    m = tcm_pcm_from_surprisal(s0, s1, vocab_size=vocab_size)
    assert m["tcm_bits"] >= gap - 1e-6
    assert m["pcm_bits"] >= gap - 1e-6
