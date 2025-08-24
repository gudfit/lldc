# tests/metrics/test_fidelity_metrics.py

import pytest

from lldc.metrics.fidelity import (
    character_level_fidelity,
    chrf_score,
    bertscore_f1,
    semantic_span_fidelity,
)


def test_character_level_fidelity_perfect_match():
    s = "hello world"
    assert character_level_fidelity(s, s) == 100.0


def test_character_level_fidelity_drops_with_edits():
    a = "hello world"
    b = "hello word"
    assert character_level_fidelity(a, b) < 100.0


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sacrebleu") is None,
    reason="sacrebleu not installed",
)
def test_chrf_basic_monotonicity():
    a = "the quick brown fox"
    b_same = "the quick brown fox"
    b_noisy = "the quick brown fx"
    assert chrf_score(a, b_same) >= chrf_score(a, b_noisy)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("evaluate") is None,
    reason="evaluate / model weights not available",
)
@pytest.mark.slow
def test_bertscore_runs_and_is_reasonable():
    ref = "A cat sits on the mat."
    hyp_good = "A cat is sitting on the mat."
    hyp_bad = "Quantum entropy dances under rainbows."
    g = bertscore_f1(ref, hyp_good, model_type="roberta-large")
    b = bertscore_f1(ref, hyp_bad, model_type="roberta-large")
    assert g > b


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed",
)
@pytest.mark.slow
def test_semantic_span_fidelity_reasonable():
    ref = "This is a longish passage meant to span multiple windows. " * 4
    hyp_good = (
        "This is a slightly paraphrased passage that should still match reasonably well. "
        * 4
    )
    hyp_bad = "Completely unrelated text about penguins and glaciers. " * 4
    g = semantic_span_fidelity(ref, hyp_good)
    b = semantic_span_fidelity(ref, hyp_bad)
    assert g > b
