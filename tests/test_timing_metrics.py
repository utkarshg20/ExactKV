"""Tests for diagnostic timing helpers (Exp 030)."""
from __future__ import annotations

from dataclasses import dataclass

from exactkv.metrics.timing import (
    estimate_sequential_verifier_forwards,
    estimate_span_verifier_forwards,
    summarize_trials,
    timed_call,
    tokens_per_second,
)
from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace


def _acceptance(
    draft: list[int],
    verifier: list[int],
    *,
    accepted: list[int],
    correction: int | None,
    rejected: list[int],
    all_matched: bool,
) -> AcceptanceResult:
    return AcceptanceResult(
        draft_tokens=draft,
        verifier_tokens=verifier,
        accepted_tokens=accepted,
        correction_token=correction,
        rejected_tokens=rejected,
        bonus_token=None,
        all_matched=all_matched,
        num_accepted=len(accepted),
        num_rejected=len(rejected),
    )


def test_tokens_per_second_basic() -> None:
    assert tokens_per_second(32, 1.0) == 32.0
    assert tokens_per_second(0, 1.0) == 0.0
    assert tokens_per_second(10, 0.0) == 0.0


def test_summarize_trials() -> None:
    stats = summarize_trials([1.0, 2.0, 3.0], [10, 20, 30])
    assert stats["mean_wall_time_seconds"] == 2.0
    assert stats["median_wall_time_seconds"] == 2.0
    assert stats["mean_tokens_per_second"] == 10.0


def test_timed_call_cpu() -> None:
    result, elapsed = timed_call("cpu", lambda: 42)
    assert result == 42
    assert elapsed >= 0.0


def test_sequential_verifier_forwards_all_match() -> None:
    acc = _acceptance(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        accepted=[1, 2, 3, 4],
        correction=None,
        rejected=[],
        all_matched=True,
    )
    trace = VerificationTrace(
        round_idx=0,
        draft_tokens=[1, 2, 3, 4],
        acceptance=acc,
        full_seq_len_before=10,
        full_seq_len_after=14,
        compressed_seq_len_after=14,
    )
    assert estimate_sequential_verifier_forwards([trace]) == 3


def test_sequential_verifier_forwards_mismatch() -> None:
    acc = _acceptance(
        [1, 2, 9],
        [1, 2, 3],
        accepted=[1, 2],
        correction=3,
        rejected=[9],
        all_matched=False,
    )
    trace = VerificationTrace(
        round_idx=0,
        draft_tokens=[1, 2, 9],
        acceptance=acc,
        full_seq_len_before=10,
        full_seq_len_after=13,
        compressed_seq_len_after=13,
    )
    assert estimate_sequential_verifier_forwards([trace]) == 2


def test_span_verifier_forwards() -> None:
    acc = _acceptance(
        [1, 2, 3],
        [1, 2, 3],
        accepted=[1, 2, 3],
        correction=None,
        rejected=[],
        all_matched=True,
    )
    traces = [
        VerificationTrace(0, [1], acc, 10, 11, 11),
        VerificationTrace(1, [2, 3], acc, 11, 13, 13),
    ]
    assert estimate_span_verifier_forwards(traces) == 1
