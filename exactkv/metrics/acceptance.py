"""Acceptance summary metrics derived from ExactKV verification traces.

Invariant (asserted by summarize_acceptance):
    total_drafted == total_accepted + total_rejected

This is the bookkeeping identity that V1 must maintain across all rounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exactkv.verification.acceptance import ExactKVResult, VerificationTrace


@dataclass
class AcceptanceSummary:
    """Aggregate acceptance statistics across all verification rounds."""
    total_rounds: int
    total_drafted: int
    total_accepted: int
    total_rejected: int
    total_corrections: int
    acceptance_rate: float          # total_accepted / max(total_drafted, 1)
    avg_accepted_per_round: float   # total_accepted / max(total_rounds, 1)
    avg_drafted_per_round: float    # total_drafted  / max(total_rounds, 1)

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_acceptance(
    traces: "list[VerificationTrace]",
) -> AcceptanceSummary:
    """Aggregate acceptance statistics from a list of per-round traces.

    The invariant ``total_drafted == total_accepted + total_rejected`` is
    asserted here so callers are alerted early if bookkeeping is broken.
    """
    if not traces:
        return AcceptanceSummary(
            total_rounds=0,
            total_drafted=0,
            total_accepted=0,
            total_rejected=0,
            total_corrections=0,
            acceptance_rate=1.0,
            avg_accepted_per_round=0.0,
            avg_drafted_per_round=0.0,
        )

    total_accepted = sum(t.acceptance.num_accepted for t in traces)
    total_rejected = sum(t.acceptance.num_rejected for t in traces)
    total_drafted = sum(len(t.draft_tokens) for t in traces)
    total_corrections = sum(
        1 for t in traces if t.acceptance.correction_token is not None
    )
    total_rounds = len(traces)

    assert total_drafted == total_accepted + total_rejected, (
        f"Bookkeeping invariant broken: drafted={total_drafted}, "
        f"accepted={total_accepted}, rejected={total_rejected}"
    )

    denom = max(total_drafted, 1)
    acceptance_rate = total_accepted / denom

    return AcceptanceSummary(
        total_rounds=total_rounds,
        total_drafted=total_drafted,
        total_accepted=total_accepted,
        total_rejected=total_rejected,
        total_corrections=total_corrections,
        acceptance_rate=acceptance_rate,
        avg_accepted_per_round=total_accepted / total_rounds,
        avg_drafted_per_round=total_drafted / total_rounds,
    )


def summarize_from_result(result: "ExactKVResult") -> AcceptanceSummary:
    """Convenience wrapper: summarize from an ExactKVResult directly."""
    return summarize_acceptance(result.traces)
