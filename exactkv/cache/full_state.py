from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class FullKVState:
    """Authoritative full-precision KV cache state.

    This is the ground-truth state that the VerificationEngine runs against.
    It must never be mutated by drafting or verification — only by explicit commit.

    ``past_key_values`` covers all committed tokens (prompt + generated).
    ``metadata["next_token_id"]`` is the full model's greedy prediction for
    the *next* token, computed during the forward pass that produced the current
    ``past_key_values``.  This lets verification start without an extra forward
    pass.

    ``generated_ids`` contains tokens generated *after* the prompt, in order,
    including EOS if generation stopped on EOS.
    """

    past_key_values: Any
    prompt_ids: torch.Tensor        # [1, prompt_len]
    generated_ids: torch.Tensor     # [1, gen_len]
    full_sequence_ids: torch.Tensor # [1, prompt_len + gen_len]
    device: torch.device
    dtype: torch.dtype
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_ids.shape[1])

    @property
    def gen_len(self) -> int:
        return int(self.generated_ids.shape[1])

    @property
    def seq_len(self) -> int:
        return int(self.full_sequence_ids.shape[1])

    @property
    def next_token_id(self) -> int:
        """Full model's greedy prediction for the next token."""
        return int(self.metadata["next_token_id"])
