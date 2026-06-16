"""Tests for Experiment 079 decode-prefix ladder shadow observer (Phase 16N)."""
from __future__ import annotations

import json

import torch

from exactkv.attention.generation_shadow_observer import (
    DEFAULT_EXP079_COMPRESSORS,
    EXPERIMENT_079_ID,
    ROUND_SOURCE_POSTHOC,
    ROUND_SOURCE_ROUND_LOG,
    SHADOW_FORBIDDEN_CLAIMS,
    GenerationOutput,
    build_decode_prefix_ladder,
    default_exp079_prompts,
    resolve_round_source,
    run_exp079_decode_prefix_ladder_panel,
    validate_exp079_report,
)


def _fake_generation(**kwargs: object) -> GenerationOutput:
    max_new = int(kwargs.get("max_new_tokens", 8))
    token_ids = list(range(100, 100 + max_new))
    return GenerationOutput(
        generation_completed=True,
        generation_output_text="generated output",
        generation_output_token_ids=token_ids,
        prompt_ids=torch.tensor([[1, 2, 3, 4, 5]]),
    )


def _fake_generation_with_traces(**kwargs: object) -> GenerationOutput:
    out = _fake_generation(**kwargs)
    out.exactkv_traces = [{"round_idx": 0, "draft_tokens": [100]}]
    return out


def _fake_shadow_replay(**kwargs: object) -> dict:
    input_ids = kwargs["input_ids"]
    k = int(input_ids.shape[-1]) - 5
    top1 = k < 3
    return {
        "num_layers_replayed": 2,
        "streaming_vs_materialized_logit_metrics": {
            "max_abs_error": 0.001 * max(1, k),
            "top1_agreement": top1,
            "top5_overlap": 5,
            "top10_overlap": 10,
        },
        "full_vs_streaming_logit_metrics": {
            "max_abs_error": 0.002 * max(1, k),
        },
        "blockers": [],
    }


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "prompts": default_exp079_prompts()[:2],
        "max_new_tokens": 4,
        "compressors_requested": ["noop"],
        "generation_fn": _fake_generation,
        "shadow_replay_fn": _fake_shadow_replay,
    }
    defaults.update(overrides)
    return run_exp079_decode_prefix_ladder_panel(**defaults)


def test_prefix_ladder_k0_to_n() -> None:
    gen = _fake_generation(max_new_tokens=4)
    ladder, blockers = build_decode_prefix_ladder(gen, ladder_stride=1)
    assert blockers == []
    ks = [k for k, _ in ladder]
    assert ks == [0, 1, 2, 3, 4]
    assert ladder[0][1].shape[-1] == 5
    assert ladder[-1][1].shape[-1] == 9


def test_ladder_stride_skips_intermediate_steps() -> None:
    gen = _fake_generation(max_new_tokens=8)
    ladder, _ = build_decode_prefix_ladder(gen, ladder_stride=2)
    ks = [k for k, _ in ladder]
    assert ks == [0, 2, 4, 6, 8]


def test_missing_generated_token_ids_blocks_safely() -> None:
    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="out",
        generation_output_token_ids=None,
        prompt_ids=torch.tensor([[1, 2, 3]]),
    )
    ladder, blockers = build_decode_prefix_ladder(gen)
    assert ladder == []
    assert "generated token IDs unavailable" in blockers[0]

    report = _run_panel(
        generation_fn=lambda **k: gen,
        max_new_tokens=4,
    )
    cell = report["generation_cells"][0]
    assert cell["round_source"] == "blocked_no_round_data"
    assert cell["prefix_shadow_cells"] == []


def test_no_default_retokenization() -> None:
    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="some text",
        generation_output_token_ids=None,
        prompt_ids=torch.tensor([[1, 2]]),
    )
    ladder, blockers = build_decode_prefix_ladder(
        gen, allow_generated_text_retokenize=False,
    )
    assert ladder == []
    assert blockers


def test_posthoc_prefix_ladder_round_source() -> None:
    gen = _fake_generation()
    assert resolve_round_source(gen) == ROUND_SOURCE_POSTHOC


def test_exactkv_round_log_round_source_when_traces_supplied() -> None:
    gen = _fake_generation_with_traces()
    assert resolve_round_source(gen) == ROUND_SOURCE_ROUND_LOG

    report = _run_panel(generation_fn=_fake_generation_with_traces)
    assert report["round_source_counts"].get(ROUND_SOURCE_ROUND_LOG, 0) >= 1


def test_safety_gates_always_false_for_shadow_commit() -> None:
    report = _run_panel()
    assert report["generation_modified_by_shadow"] is False
    assert report["shadow_used_for_token_commit"] is False
    assert report["default_runtime_changed"] is False
    for cell in report["generation_cells"]:
        gates = cell["safety_gates"]
        assert gates["generated_output_unchanged"] is True
        assert gates["shadow_used_for_token_commit"] is False
        assert gates["generation_modified_by_shadow"] is False
        assert gates["default_runtime_changed"] is False


def test_first_status_change_aggregation() -> None:
    report = _run_panel(max_new_tokens=4)
    summary = report["first_status_change_summary"]
    assert "cells_with_status_change" in summary
    assert "cells_all_stable" in summary
    assert "changes" in summary


def test_first_top1_mismatch_aggregation() -> None:
    report = _run_panel(max_new_tokens=4)
    summary = report["first_top1_mismatch_summary"]
    assert summary["cells_with_top1_mismatch"] >= 1
    assert summary["mismatches"]
    first = summary["mismatches"][0]
    assert first["first_mismatch_at_prefix_length"] == 3


def test_report_schema_validates() -> None:
    report = _run_panel(compressors_requested=DEFAULT_EXP079_COMPRESSORS[:1])
    assert report["experiment_id"] == EXPERIMENT_079_ID
    assert validate_exp079_report(report) == []
    assert report["total_prefix_shadow_cells"] == 2 * 5  # 2 prompts, k=0..4


def test_no_forbidden_claim_fields() -> None:
    report = _run_panel()
    assert set(SHADOW_FORBIDDEN_CLAIMS).issubset(set(report.get("forbidden_claims", [])))
    for cell in report["generation_cells"]:
        dumped = json.dumps(cell).lower()
        for forbidden in (
            "throughput",
            "latency",
            "speedup",
            "tokens_per_second",
            "active_gpu_memory_savings",
            "production_memory_savings",
        ):
            assert forbidden not in dumped
