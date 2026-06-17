"""Tests for guarded decode-time shadow observer (Phase 16R)."""
from __future__ import annotations

import json

from exactkv.attention.decode_time_shadow_observer import (
    GuardedDecodeTimeShadowObserver,
    _aggregate_safety_gate_summary,
    _cell_safety_gates,
    compare_decode_time_vs_posthoc_shadow,
    decode_time_shadow_cell_matches_posthoc,
    run_shadow_diagnostic_for_snapshot,
    snapshot_is_post_commit,
)
from exactkv.attention.live_round_observer import build_live_round_snapshot
from exactkv.verification.acceptance import AcceptanceResult

TOKENS = [100, 101, 102, 103]


def _acceptance() -> AcceptanceResult:
    return AcceptanceResult(
        draft_tokens=TOKENS[:2],
        verifier_tokens=TOKENS[:2],
        accepted_tokens=TOKENS[:2],
        correction_token=None,
        rejected_tokens=[],
        bonus_token=None,
        all_matched=True,
        num_accepted=2,
        num_rejected=0,
    )


def _post_commit_snapshot(round_idx: int = 0) -> object:
    return build_live_round_snapshot(
        round_index=round_idx,
        prompt_token_ids=(1, 2, 3),
        generated_token_ids_before=() if round_idx == 0 else (100, 101),
        generated_token_ids_after=(100, 101) if round_idx == 0 else (100, 101, 102, 103),
        draft_token_ids=TOKENS[round_idx : round_idx + 2],
        acceptance=_acceptance(),
        compressor_name="noop",
        max_new_tokens=8,
        full_seq_len_before=3 if round_idx == 0 else 5,
        full_seq_len_after=5 if round_idx == 0 else 7,
    )


def _pre_commit_snapshot() -> object:
    return build_live_round_snapshot(
        round_index=0,
        prompt_token_ids=(1, 2, 3),
        generated_token_ids_before=(),
        generated_token_ids_after=(100, 101),
        draft_token_ids=[100, 101],
        acceptance=_acceptance(),
        compressor_name="noop",
        max_new_tokens=8,
    )


def test_observer_stores_snapshots_and_results() -> None:
    def _shadow(**kwargs: object) -> dict:
        del kwargs
        return {
            "num_layers_replayed": 1,
            "streaming_vs_materialized_logit_metrics": {"top1_agreement": True},
            "full_vs_streaming_logit_metrics": {},
            "blockers": [],
        }

    observer = GuardedDecodeTimeShadowObserver(
        shadow_diagnostic_fn=_shadow,
        prompt_id="p0",
    )
    snap = _post_commit_snapshot()
    ret = observer.observe(snap)
    assert ret is not None
    assert len(observer.snapshots) == 1
    assert len(observer.decode_time_shadow_cells) == 1
    assert observer.decode_time_shadow_cells[0]["shadow_status"] == "shadow_complete"


def test_observer_return_value_ignored() -> None:
    observer = GuardedDecodeTimeShadowObserver(prompt_id="p0")
    ret = observer.observe(_post_commit_snapshot())
    assert ret is not None


def test_shadow_does_not_mutate_snapshot() -> None:
    def _mutating_shadow(**kwargs: object) -> dict:
        snap = kwargs.get("input_ids")
        if snap is not None:
            pass
        return {"num_layers_replayed": 1, "blockers": []}

    snap = _post_commit_snapshot()
    before_after = snap.prefix_token_ids_after
    observer = GuardedDecodeTimeShadowObserver(
        shadow_diagnostic_fn=_mutating_shadow,
        prompt_id="p0",
    )
    observer.observe(snap)
    assert snap.prefix_token_ids_after == before_after


def test_shadow_exception_captured() -> None:
    def _boom(**kwargs: object) -> dict:
        del kwargs
        raise RuntimeError("shadow failed")

    observer = GuardedDecodeTimeShadowObserver(
        shadow_diagnostic_fn=_boom,
        prompt_id="p0",
    )
    observer.observe(_post_commit_snapshot())
    assert observer.shadow_callback_exceptions
    cell = observer.decode_time_shadow_cells[0]
    assert cell["exception"] is not None
    assert cell["shadow_status"] == "shadow_blocked"


def test_post_commit_validation_passes_and_fails() -> None:
    assert snapshot_is_post_commit(_post_commit_snapshot()) is True
    assert snapshot_is_post_commit(_pre_commit_snapshot()) is False


def test_decode_time_vs_posthoc_comparison() -> None:
    dt = {
        "round_index": 0,
        "shadow_status": "shadow_complete",
        "tolerance_policy_status": "local_alignment_pass_free_running_accumulation",
        "topk_agreement_metrics": {"top1_agreement": True},
    }
    ph = dict(dt)
    assert decode_time_shadow_cell_matches_posthoc(dt, ph) is True
    summary = compare_decode_time_vs_posthoc_shadow([dt], [ph])
    assert summary["all_match"] is True
    ph["shadow_status"] = "shadow_blocked"
    assert decode_time_shadow_cell_matches_posthoc(dt, ph) is False


def test_run_shadow_diagnostic_with_fake_fn() -> None:
    def _fake(**kwargs: object) -> dict:
        del kwargs
        return {
            "num_layers_replayed": 2,
            "streaming_vs_materialized_logit_metrics": {"top1_agreement": True},
            "full_vs_streaming_logit_metrics": {},
            "blockers": [],
        }

    result = run_shadow_diagnostic_for_snapshot(
        _post_commit_snapshot(),
        prompt_id="p0",
        hf_model=None,
        shadow_diagnostic_fn=_fake,
    )
    assert result["shadow_status"] == "shadow_complete"
    assert result["topk_agreement_metrics"]["top1_agreement"] is True


def test_safety_gate_summary_aggregation() -> None:
    cells = [{"safety_gates": _cell_safety_gates()} for _ in range(3)]
    summary = _aggregate_safety_gate_summary(cells)
    assert summary["cells_all_gates_ok"] == 3
    assert summary["cells_with_gate_failure"] == 0
