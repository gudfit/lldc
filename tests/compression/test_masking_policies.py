# tests/compression/test_masking_policies.py

import pytest
import torch

from lldc.compression.masking_policies import choose_mask
from lldc.compression.predictive_masking import entropy_equalisation_mask


def test_topk_global_keeps_largest_surprisal():
    s = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    keep = choose_mask("topk_global", s, keep_fraction=0.4)
    assert keep.dtype == torch.bool
    assert keep.sum().item() == 2
    assert keep.tolist() == [False, False, False, True, True]


def test_entropy_equalisation_respects_windows():
    s = torch.tensor([1.0, 9.0, 2.0, 8.0, 3.0, 7.0])
    keep = entropy_equalisation_mask(s, keep_fraction=0.5, window=2)
    assert keep.sum().item() == 3
    assert keep.tolist() == [False, True, False, True, False, True]


def test_choose_mask_unknown_policy_raises():
    with pytest.raises(ValueError):
        _ = choose_mask("does_not_exist", torch.ones(5), keep_fraction=0.5)
