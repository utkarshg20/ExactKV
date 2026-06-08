"""Token-level and text-level exactness metrics.

The primary correctness criterion for ExactKV V1 is token exact match:
    exactkv_output_ids == full_greedy_output_ids   (element-wise)

Text exact match is a secondary convenience check (it may disagree with token
exact match when the tokenizer is non-invertible, e.g. leading spaces).
"""
from __future__ import annotations

from typing import Optional

import torch


def token_exact_match(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
) -> bool:
    """Return True iff the two ID tensors are identical.

    Accepts tensors of shape [1, N] or [N].  Shape mismatch → False.
    """
    a = output_a.reshape(-1)
    b = output_b.reshape(-1)
    if a.shape != b.shape:
        return False
    return bool((a == b).all().item())


def text_exact_match(text_a: str, text_b: str) -> bool:
    """Return True iff the two decoded strings are byte-for-byte identical."""
    return text_a == text_b


def first_divergence_idx(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
) -> Optional[int]:
    """Return the position of the first token that differs, or None if identical.

    If the sequences have different lengths and all shorter-length tokens match,
    the divergence index is ``min(len_a, len_b)`` (i.e. the first position
    beyond the shorter sequence).
    """
    a = output_a.reshape(-1).tolist()
    b = output_b.reshape(-1).tolist()
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min_len
    return None
