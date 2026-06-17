"""Tests for opt-in live round observer instrumentation (Phase 16P)."""
from __future__ import annotations

import dataclasses

import pytest

from exactkv.attention.live_round_observer import (
    LiveRoundObserver,
    LiveRoundSnapshot,
    build_live_round_snapshot,
    compare_snapshot_to_trace,
    compare_snapshots_to_traces,
)
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace


def test_live_round_snapshot_is_immutable() -> None:
    snap = build_live_round_snapshot(
        round_index=0,
        prompt_token_ids=(1, 2, 3),
        generated_token_ids_before=(),
        generated_token_ids_after=(10,),
        draft_token_ids=[10, 11],
        acceptance=AcceptanceResult(
            draft_tokens=[10, 11],
            verifier_tokens=[10, 11],
            accepted_tokens=[10, 11],
            correction_token=None,
            rejected_tokens=[],
            bonus_token=None,
            all_matched=True,
            num_accepted=2,
            num_rejected=0,
        ),
        compressor_name="noop",
        max_new_tokens=8,
        full_seq_len_before=3,
        full_seq_len_after=4,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.round_index = 1  # type: ignore[misc]


def test_observer_return_value_ignored() -> None:
    def _cb(_snap: LiveRoundSnapshot) -> str:
        return "must_be_ignored"

    obs = LiveRoundObserver(on_round=_cb)
    snap = build_live_round_snapshot(
        round_index=0,
        prompt_token_ids=(1,),
        generated_token_ids_before=(),
        generated_token_ids_after=(9,),
        draft_token_ids=[9],
        acceptance=None,
        compressor_name="noop",
        max_new_tokens=4,
    )
    obs.observe(snap)
    assert len(obs.snapshots) == 1


def test_observer_cannot_mutate_snapshot_prefix() -> None:
    mutated: list[int] = []

    def _cb(snap: LiveRoundSnapshot) -> None:
        mutated.extend(snap.prefix_token_ids_after)

    obs = LiveRoundObserver(on_round=_cb)
    snap = build_live_round_snapshot(
        round_index=0,
        prompt_token_ids=(1, 2),
        generated_token_ids_before=(),
        generated_token_ids_after=(5,),
        draft_token_ids=[5],
        acceptance=None,
        compressor_name="noop",
        max_new_tokens=4,
    )
    obs.observe(snap)
    assert mutated == [1, 2, 5]


def test_observer_exception_captured() -> None:
    def _boom(_snap: LiveRoundSnapshot) -> None:
        raise RuntimeError("observer failed")

    obs = LiveRoundObserver(on_round=_boom)
    snap = build_live_round_snapshot(
        round_index=0,
        prompt_token_ids=(1,),
        generated_token_ids_before=(),
        generated_token_ids_after=(1,),
        draft_token_ids=[1],
        acceptance=None,
        compressor_name="noop",
        max_new_tokens=4,
    )
    obs.observe(snap)
    assert obs.exceptions
    assert "RuntimeError" in obs.exceptions[0]


def test_generator_notify_noop_when_observer_none() -> None:
    gen = ExactKVGenerator.__new__(ExactKVGenerator)
    gen.round_observer = None
    gen._notify_round_observer(
        round_idx=0,
        seq_len_before=3,
        full_state=type("FS", (), {"prompt_ids": __import__("torch").tensor([[1, 2, 3]]), "seq_len": 4})(),
        draft_tokens=[9],
        acceptance=AcceptanceResult(
            draft_tokens=[9],
            verifier_tokens=[9],
            accepted_tokens=[9],
            correction_token=None,
            rejected_tokens=[],
            bonus_token=None,
            all_matched=True,
            num_accepted=1,
            num_rejected=0,
        ),
        gen_tokens_before=(),
        gen_tokens_after=(9,),
        max_new_tokens=4,
    )


def test_compare_snapshot_to_trace_match() -> None:
    acceptance = AcceptanceResult(
        draft_tokens=[10, 11],
        verifier_tokens=[10, 11],
        accepted_tokens=[10, 11],
        correction_token=None,
        rejected_tokens=[],
        bonus_token=None,
        all_matched=True,
        num_accepted=2,
        num_rejected=0,
    )
    snap = build_live_round_snapshot(
        round_index=0,
        prompt_token_ids=(1, 2, 3),
        generated_token_ids_before=(),
        generated_token_ids_after=(10, 11),
        draft_token_ids=[10, 11],
        acceptance=acceptance,
        compressor_name="noop",
        max_new_tokens=8,
        full_seq_len_before=3,
        full_seq_len_after=5,
    )
    trace = VerificationTrace(
        round_idx=0,
        draft_tokens=[10, 11],
        acceptance=acceptance,
        full_seq_len_before=3,
        full_seq_len_after=5,
        compressed_seq_len_after=5,
    )
    ok, mismatches = compare_snapshot_to_trace(snap, trace)
    assert ok
    assert mismatches == []


def test_compare_snapshots_to_traces_count_mismatch() -> None:
    result = compare_snapshots_to_traces([], [object()])
    assert result["snapshot_vs_result_round_log_match"] is False
