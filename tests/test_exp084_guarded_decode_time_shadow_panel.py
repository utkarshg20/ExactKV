"""Tests for Experiment 084 expanded guarded decode-time shadow panel (Phase 16S)."""
from __future__ import annotations

import json

from exactkv.attention.decode_time_shadow_observer import (
    EXPERIMENT_084_ID,
    GuardedDecodeTimeShadowObserver,
    _cell_safety_gates,
    run_exp084_guarded_decode_time_shadow_panel,
    validate_exp084_report,
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


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "model_id": "mock",
        "prompts": [("p0", "hello"), ("p1", "world")],
        "compressors_requested": ["noop", "int8"],
        "max_new_tokens_values": [4, 8],
        "baseline_generation_fn": _baseline_fn,
        "guarded_generation_fn": _guarded_fn,
        "shadow_diagnostic_fn": _fake_shadow,
    }
    defaults.update(overrides)
    return run_exp084_guarded_decode_time_shadow_panel(**defaults)


def test_expanded_panel_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_084_ID
    assert report["total_cells"] == 8  # 2 prompts × 2 compressors × 2 max_new_tokens
    assert validate_exp084_report(report) == []


def test_multiple_prompts_compressors_max_new_tokens() -> None:
    report = _run_panel()
    max_nt = {c["max_new_tokens"] for c in report["cells"]}
    compressors = {c["compressor"] for c in report["cells"]}
    prompts = {c["prompt_id"] for c in report["cells"]}
    assert max_nt == {4, 8}
    assert compressors == {"noop", "int8"}
    assert prompts == {"p0", "p1"}


def test_baseline_vs_guarded_parity_pass() -> None:
    report = _run_panel()
    assert report["status"] == "diagnostic_complete"
    assert report["baseline_vs_guarded_token_match_cells"] == 8
    assert report["baseline_vs_guarded_text_match_cells"] == 8


def test_baseline_vs_guarded_parity_failure() -> None:
    def _bad_guarded(**kwargs: object) -> dict:
        out = _guarded_fn(**kwargs)
        out["generated_token_ids"] = [999]
        return out

    report = _run_panel(guarded_generation_fn=_bad_guarded)
    assert report["status"] == "failed"
    assert report["baseline_vs_guarded_token_mismatch_cells"] == 8


def test_decode_time_callback_success_aggregation() -> None:
    report = _run_panel()
    assert report["decode_time_shadow_callback_count"] == 16  # 8 cells × 2 rounds
    assert report["decode_time_shadow_successful_callbacks"] == 16


def test_decode_time_callback_exception_aggregation() -> None:
    def _guarded_with_exception(**kwargs: object) -> dict:
        out = _guarded_fn(**kwargs)
        out["decode_time_shadow_exception_callbacks"] = 1
        out["decode_time_shadow_successful_callbacks"] = (
            out["decode_time_shadow_successful_callbacks"] - 1
        )
        return out

    report = _run_panel(guarded_generation_fn=_guarded_with_exception)
    assert report["decode_time_shadow_exception_callbacks"] >= 1


def test_decode_time_vs_posthoc_match_aggregation() -> None:
    report = _run_panel()
    assert report["posthoc_shadow_comparison_cells"] == 8
    assert report["decode_time_vs_posthoc_shadow_match_cells"] == 8
    assert report["decode_time_vs_posthoc_shadow_mismatch_cells"] == 0


def test_safety_gate_summary() -> None:
    report = _run_panel()
    summary = report["safety_gate_summary"]
    assert summary["cells_all_gates_ok"] == 8
    assert summary["cells_with_gate_failure"] == 0
    gates = _cell_safety_gates()
    assert gates["shadow_result_exposed_to_generator"] is False


def test_report_failure_on_safety_gate() -> None:
    def _bad_gates_guarded(**kwargs: object) -> dict:
        out = _guarded_fn(**kwargs)
        out["generated_token_ids"] = [1, 2, 3, 4]
        return out

    report = _run_panel(guarded_generation_fn=_bad_gates_guarded)
    assert report["status"] == "failed"


def test_no_timing_or_performance_fields() -> None:
    report = _run_panel()
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
