"""Pure acceptance / rejection logic and result data structures.

``compute_acceptance`` is a pure function (no model, no device) that decides
which drafted tokens are committed given the draft sequence and the verifier's
predictions.  All other types in this file are data-only dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DraftResult:
    """Tokens produced by the draft model in one round."""
    token_ids: list[int]
    stopped_on_eos: bool = False


@dataclass
class AcceptanceResult:
    """Outcome of comparing draft tokens to verifier tokens.

    Semantics
    ---------
    ``accepted_tokens``:  prefix of draft_tokens that matched verifier_tokens.
    ``correction_token``: verifier token at the first mismatch (None if all matched).
    ``rejected_tokens``:  draft tokens from the mismatch position onwards
                          (includes the mismatched token itself).
    ``bonus_token``:      always None in V1; reserved for future VeriCache bonus.
    ``all_matched``:      True iff every draft token matched the corresponding
                          verifier token (correction_token is None in this case).
    ``num_rejected``:     len(rejected_tokens).
    """
    draft_tokens: list[int]
    verifier_tokens: list[int]
    accepted_tokens: list[int]
    correction_token: int | None
    rejected_tokens: list[int]
    bonus_token: None             # V1: always None
    all_matched: bool
    num_accepted: int
    num_rejected: int


@dataclass
class VerificationTrace:
    """Per-round trace entry for ExactKVResult."""
    round_idx: int
    draft_tokens: list[int]
    acceptance: AcceptanceResult
    full_seq_len_before: int
    full_seq_len_after: int
    compressed_seq_len_after: int


@dataclass
class ExactKVResult:
    """Final result returned by ExactKVGenerator.generate()."""
    prompt_ids: torch.Tensor           # [1, prompt_len]
    output_ids: torch.Tensor           # [1, gen_len]  — generated tokens only
    full_sequence_ids: torch.Tensor    # [1, prompt_len + gen_len]
    output_text: str
    stopped_on_eos: bool
    traces: list[VerificationTrace]
    total_accepted: int
    total_rejected: int
    total_corrections: int
    acceptance_rate: float
    num_rounds: int


# ---------------------------------------------------------------------------
# Pure acceptance logic
# ---------------------------------------------------------------------------

def compute_acceptance(
    draft_tokens: list[int],
    verifier_tokens: list[int],
) -> AcceptanceResult:
    """Decide which drafted tokens are accepted.

    Rules (V1, bonus disabled)
    --------------------------
    1. Walk draft_tokens and verifier_tokens in lock-step.
    2. While d_i == v_i: accumulate into accepted_tokens.
    3. On the first mismatch at position i:
       - correction_token = v_i
       - rejected_tokens  = draft_tokens[i:]   (mismatch + all remaining)
       - Stop.
    4. If all tokens match: correction_token = None, rejected_tokens = [].
    5. Empty draft: return trivially with all_matched=True.

    Args:
        draft_tokens:    Tokens produced by the compressed model.
        verifier_tokens: Tokens produced by sequential full-KV verification.
                         May be shorter than draft_tokens when a mismatch is
                         found (the engine stops generating verifier tokens
                         after the first mismatch).

    Returns:
        AcceptanceResult describing which tokens are accepted / rejected.
    """
    if not draft_tokens:
        return AcceptanceResult(
            draft_tokens=[],
            verifier_tokens=[],
            accepted_tokens=[],
            correction_token=None,
            rejected_tokens=[],
            bonus_token=None,
            all_matched=True,
            num_accepted=0,
            num_rejected=0,
        )

    accepted: list[int] = []
    correction: int | None = None
    mismatch_idx: int | None = None

    for i, (d, v) in enumerate(zip(draft_tokens, verifier_tokens)):
        if d == v:
            accepted.append(d)
        else:
            correction = v
            mismatch_idx = i
            break

    if mismatch_idx is not None:
        # draft_tokens[mismatch_idx] is the mismatched draft token;
        # all tokens from mismatch_idx onwards are rejected.
        rejected = list(draft_tokens[mismatch_idx:])
        all_matched = False
    elif len(verifier_tokens) < len(draft_tokens):
        # Prefix matched, but the verifier stopped before covering every draft token.
        # Do not treat unverified tail tokens as accepted.
        rejected = list(draft_tokens[len(accepted):])
        all_matched = False
    else:
        rejected = []
        all_matched = True

    return AcceptanceResult(
        draft_tokens=list(draft_tokens),
        verifier_tokens=list(verifier_tokens),
        accepted_tokens=accepted,
        correction_token=correction,
        rejected_tokens=rejected,
        bonus_token=None,
        all_matched=all_matched,
        num_accepted=len(accepted),
        num_rejected=len(rejected),
    )
