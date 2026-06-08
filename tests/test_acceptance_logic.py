"""Pure acceptance logic tests.  No model, no device, no torch.

Every test calls compute_acceptance with hand-crafted token lists and checks
the resulting AcceptanceResult fields exactly.  These tests must run in under
1 second total.

Test matrix
-----------
- all_match              : every draft token matches verifier → accepted all
- first_token_mismatch   : d[0] != v[0] → accepted=[], correction=v[0], rejected=all draft
- middle_mismatch        : d[1] != v[1] → accepted=[d[0]], correction=v[1], rejected=d[1:]
- eos_as_correction      : v at mismatch is EOS → correction=EOS, rejected includes d[i:]
- empty_draft            : draft=[] → trivially accepted, all_matched=True
"""
from __future__ import annotations

import pytest

from exactkv.verification.acceptance import compute_acceptance

EOS = 2  # arbitrary sentinel; not model-specific in pure tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_invariants(result, draft_tokens, verifier_tokens) -> None:
    """Structural invariants that must hold for every AcceptanceResult."""
    assert result.bonus_token is None, "bonus_token must be None in V1"
    assert result.num_accepted == len(result.accepted_tokens)
    assert result.num_rejected == len(result.rejected_tokens)
    # accepted + rejected accounts for all draft tokens past the boundary
    if result.correction_token is None:
        # All matched — no rejected tokens
        assert result.rejected_tokens == []
        assert result.all_matched is True
        assert result.accepted_tokens == list(draft_tokens[: len(result.accepted_tokens)])
    else:
        # Mismatch: accepted_tokens + [mismatched_draft] + remaining = draft_tokens
        assert result.all_matched is False
        expected_rejected_start = len(result.accepted_tokens)
        assert result.rejected_tokens == list(draft_tokens[expected_rejected_start:])


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def test_all_match() -> None:
    draft    = [10, 20, 30]
    verifier = [10, 20, 30]
    r = compute_acceptance(draft, verifier)

    assert r.all_matched is True
    assert r.accepted_tokens == [10, 20, 30]
    assert r.correction_token is None
    assert r.rejected_tokens == []
    assert r.num_accepted == 3
    assert r.num_rejected == 0
    assert r.bonus_token is None
    _check_invariants(r, draft, verifier)


def test_first_token_mismatch() -> None:
    draft    = [1, 2, 3]
    verifier = [9]        # engine stops after finding mismatch at position 0
    r = compute_acceptance(draft, verifier)

    assert r.all_matched is False
    assert r.accepted_tokens == []
    assert r.correction_token == 9
    # All three draft tokens are rejected (1 is mismatched; 2,3 are unverified)
    assert r.rejected_tokens == [1, 2, 3]
    assert r.num_accepted == 0
    assert r.num_rejected == 3
    _check_invariants(r, draft, verifier)


def test_middle_mismatch() -> None:
    draft    = [1, 2, 3]
    verifier = [1, 9]     # engine stops after mismatch at position 1
    r = compute_acceptance(draft, verifier)

    assert r.all_matched is False
    assert r.accepted_tokens == [1]
    assert r.correction_token == 9
    # Tokens from mismatch onwards are rejected: draft[1:] = [2, 3]
    assert r.rejected_tokens == [2, 3]
    assert r.num_accepted == 1
    assert r.num_rejected == 2
    _check_invariants(r, draft, verifier)


def test_eos_as_correction() -> None:
    draft    = [1, 2, 3]
    verifier = [1, 2, EOS]   # full model predicts EOS at position 2
    r = compute_acceptance(draft, verifier)

    assert r.all_matched is False
    assert r.accepted_tokens == [1, 2]
    assert r.correction_token == EOS
    # draft[2:] = [3] is rejected in favour of EOS
    assert r.rejected_tokens == [3]
    assert r.num_accepted == 2
    assert r.num_rejected == 1
    _check_invariants(r, draft, verifier)


def test_empty_draft() -> None:
    r = compute_acceptance([], [])

    assert r.all_matched is True
    assert r.accepted_tokens == []
    assert r.correction_token is None
    assert r.rejected_tokens == []
    assert r.num_accepted == 0
    assert r.num_rejected == 0
    assert r.bonus_token is None


def test_single_token_match() -> None:
    r = compute_acceptance([42], [42])

    assert r.all_matched is True
    assert r.accepted_tokens == [42]
    assert r.correction_token is None
    assert r.rejected_tokens == []


def test_single_token_mismatch() -> None:
    r = compute_acceptance([42], [99])

    assert r.all_matched is False
    assert r.accepted_tokens == []
    assert r.correction_token == 99
    assert r.rejected_tokens == [42]
    assert r.num_rejected == 1
