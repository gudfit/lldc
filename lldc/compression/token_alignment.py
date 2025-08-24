# lldc/compression/token_alignment.py

from __future__ import annotations
from typing import Iterable, List, Tuple, Any


def kept_char_spans_from_offsets(
    mlm_tokenizer: Any,
    text: str,
    keep_flags: Iterable[bool],
) -> List[Tuple[int, int]]:
    enc = mlm_tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    offs = enc.get("offset_mapping", None)
    if offs is None:
        raise RuntimeError(
            "Fast tokenizer with offset mapping is required for alignment."
        )
    offs = list(offs)
    kf = list(keep_flags)
    if len(kf) != len(offs):
        raise ValueError(
            f"keep_flags length {len(kf)} != tokenized length {len(offs)}. "
            "Ensure add_special_tokens=False during tokenization producing keep_flags."
        )
    spans: List[Tuple[int, int]] = []
    for flag, (s, e) in zip(kf, offs):
        if flag and e > s:
            spans.append((int(s), int(e)))
    return spans


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    s1, e1 = a
    s2, e2 = b
    return max(s1, s2) < min(e1, e2)


def select_oracle_token_ids_from_spans(
    oracle_tokenizer: Any,
    text: str,
    kept_char_spans: List[Tuple[int, int]],
) -> List[int]:
    if not kept_char_spans:
        return []
    enc = oracle_tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    ids: List[int] = list(enc["input_ids"])
    offs: List[Tuple[int, int]] = list(enc.get("offset_mapping", []))
    if not offs:
        return ids
    kept_ids: List[int] = []
    for tid, span in zip(ids, offs):
        if span[1] <= span[0]:
            continue
        if any(_overlaps(span, s) for s in kept_char_spans):
            kept_ids.append(int(tid))
    return kept_ids
