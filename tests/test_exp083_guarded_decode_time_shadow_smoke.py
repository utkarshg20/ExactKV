"""Tests for Experiment 083 guarded decode-time shadow smoke (Phase 16R)."""
from __future__ import annotations

import json

from exactkv.attention.decode_time_shadow_observer import (
    EXPERIMENT_083_ID,
    GuardedDecodeTimeShadowObserver,
    run_exp083_guarded_decode_time_shadow_smoke,
    validate_exp083_report,
)
from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS
from exactkv.attention.live_round_observer import (
    LiveRoundSnapshot,
    build_live_round_snapshot,
    compare_snapshots_to_traces,
)
from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace

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


def _fake_trace(round_idx: int, before: int, after: int) -> VerificationTrace:
    return VerificationTrace(
        round_idx=round_idx,
        draft_tokens=TOKENS[round_idx : round_idx + 2],
        acceptance=_acceptance(),
        full_seq_len_before=before,
        full_seq_len_after=after,
        compressed_seq_len_after=after,
    )


def _fake_snapshots() -> list[LiveRoundSnapshot]:
    prompt = (1, 2, 3)
    snaps = []
    for rnd, (before, after) in enumerate([(3, 5), (5, 7)]):
        before_gen = tuple(TOKENS[: rnd * 2])
        after_gen = tuple(TOKENS[: rnd * 2 + 2])
        snaps.append(
            build_live_round_snapshot(
                round_index=rnd,
                prompt_token_ids=prompt,
                generated_token_ids_before=before_gen,
                generated_token_ids_after=after_gen,
                draft_token_ids=TOKENS[rnd : rnd + 2],
                acceptance=_acceptance(),
                compressor_name="noop",
                max_new_tokens=8,
                full_seq_len_before=before,
                full_seq_len_after=after,
            )
        )
    return snaps


def _baseline_fn(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": TOKENS,
        "generated_text": "out",
        "exactkv_failures": 0,
        "token_exact_match": True,
        "blockers": [],
    }


def _fake_shadow(**kwargs: object) -> dict:
    del kwargs
    return {
        "num_layers_replayed": 2,
        "streaming_vs_materialized_logit_metrics": {
            "max_abs_error": 0.001,
            "top1_agreement": True,
        },
        "full_vs_streaming_logit_metrics": {"max_abs_error": 0.001},
        "blockers": [],
    }


def _guarded_fn(**kwargs: object) -> dict:
    del kwargs
    snaps = _fake_snapshots()
    observer = GuardedDecodeTimeShadowObserver(
        shadow_diagnostic_fn=_fake_shadow,
        prompt_id="p0",
    )
    for snap in snaps:
        observer.observe(snap)
    traces = [_fake_trace(0, 3, 5), _fake_trace(1, 5, 7)]
    return {
        "generation_completed": True,
        "generated_token_ids": TOKENS,
        "generated_text": "out",
        "exactkv_failures": 0,
        "token_exact_match": True,
        "live_snapshots": snaps,
        "decode_time_shadow_cells": observer.decode_time_shadow_cells,
        "decode_time_shadow_callback_count": len(observer.decode_time_shadow_cells),
        "decode_time_shadow_successful_callbacks": sum(
            1
            for c in observer.decode_time_shadow_cells
            if c.get("shadow_status") == "shadow_complete"
        ),
        "decode_time_shadow_exception_callbacks": 0,
        "result_traces": traces,
        "snapshot_comparison": compare_snapshots_to_traces(snaps, traces),
        "blockers": [],
    }


def _run_smoke(**overrides: object) -> dict:
    defaults = {
        "model_id": "mock",
        "prompts": [("p0", "hello")],
        "compressors_requested": ["noop"],
        "baseline_generation_fn": _baseline_fn,
        "guarded_generation_fn": _guarded_fn,
        "shadow_diagnostic_fn": _fake_shadow,
    }
    defaults.update(overrides)
    return run_exp083_guarded_decode_time_shadow_smoke(**defaults)


def test_baseline_vs_guarded_parity_pass() -> None:
    report = _run_smoke()
    assert report["status"] == "diagnostic_complete"
    assert report["baseline_vs_guarded_token_match_cells"] == 1
    assert validate_exp083_report(report) == []


def test_baseline_vs_guarded_parity_failure() -> None:
    def _bad_guarded(**kwargs: object) -> dict:
        out = _guarded_fn(**kwargs)
        out["generated_token_ids"] = [999]
        return out

    report = _run_smoke(guarded_generation_fn=_bad_guarded)
    assert report["status"] == "failed"
    assert report["baseline_vs_guarded_token_match_cells"] == 0


def test_shadow_exception_does_not_affect_generation_output() -> None:
    def _boom_shadow(**kwargs: object) -> dict:
        del kwargs
        raise RuntimeError("shadow boom")

    def _guarded_with_exception(**kwargs: object) -> dict:
        out = _guarded_fn(**kwargs)
        out["decode_time_shadow_exception_callbacks"] = 1
        return out

    report = _run_smoke(
        guarded_generation_fn=_guarded_with_exception,
        shadow_diagnostic_fn=_boom_shadow,
    )
    assert report["baseline_vs_guarded_token_match_cells"] == 1
    assert report["generation_modified_by_decode_time_shadow"] is False


def test_decode_time_vs_posthoc_comparison_aggregation() -> None:
    report = _run_smoke()
    assert report["posthoc_shadow_comparison_cells"] == 1
    assert report["decode_time_vs_posthoc_shadow_match_cells"] == 1
    cell = report["cells"][0]
    assert cell["posthoc_comparison_summary"]["all_match"] is True


def test_safety_gates_enforced() -> None:
    report = _run_smoke()
    gates = report["cells"][0]["safety_gates"]
    assert gates["decode_time_shadow_used_for_token_commit"] is False
    assert gates["shadow_exception_affects_generation"] is False
    assert gates["observer_return_value_ignored"] is True
    assert report["decode_time_shadow_used_for_token_commit"] is False


def test_report_schema_validates() -> None:
    report = _run_smoke()
    assert report["experiment_id"] == EXPERIMENT_083_ID
    assert validate_exp083_report(report) == []


def test_no_forbidden_claim_fields() -> None:
    report = _run_smoke()
    assert set(SHADOW_FORBIDDEN_CLAIMS).issubset(set(report.get("forbidden_claims", [])))
    for cell in report["cells"]:
        dumped = json.dumps(cell).lower()
        for forbidden in (
            "throughput",
            "latency",
            "speedup",
            "tokens_per_second",
            "runtime_seconds",
            "active_gpu_memory_savings",
            "production_memory_savings",
        ):
            assert forbidden not in dumped
