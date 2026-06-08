from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CompressedKVState:
    """Compressed KV cache state used for drafting.

    ``data`` is the compressor-specific representation.  For NoOp it is the
    identical ``past_key_values`` object as the full state.  For INT8 it would
    be quantised tensors.

    ``logical_seq_len`` is the number of tokens this state represents, which
    must always equal ``FullKVState.seq_len`` at round boundaries (cache
    alignment invariant).

    ``metadata["next_token_id"]`` is the compressed model's greedy prediction
    for the next token.  For NoOp this equals ``FullKVState.next_token_id``.
    For lossy compressors it may differ — that difference is what generates
    drafts that diverge from full-KV and thus require correction.

    ``generated_ids`` mirrors the generated portion of the full state at the
    last commit, kept here for alignment assertions.
    """

    data: Any
    metadata: dict = field(default_factory=dict)
    compressor_name: str = "unknown"
    logical_seq_len: int = 0
    generated_ids: torch.Tensor = None   # [1, gen_len]
    device: torch.device = None

    @property
    def next_token_id(self) -> int:
        """Compressed model's prediction for the next token."""
        return int(self.metadata["next_token_id"])
