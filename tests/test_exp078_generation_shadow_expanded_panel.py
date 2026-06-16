"""Tests for Experiment 078 expanded generation-shadow panel (Phase 16M)."""
from __future__ import annotations

import json

import torch

from exactkv.attention.generation_shadow_observer import (
    DEFAULT_EXP078_COMPRESSORS,
    DEFAULT_EXP078_MAX_NEW_TOKENS,
    EXPERIMENT_078_ID,
    SHADOW_FORBIDDEN_CLAIMS,
    default_exp078_prompts,
    resolve_panel_compressors,
    run_exp078_expanded_panel,
    validate_exp078_report,
)
from exactkv.attention.generation_shadow_observer import GenerationOutput


def _fake_generation(**kwargs: object) -> GenerationOutput:
    compressor = kwargs.get("compressor_name", "noop")
    max_new = int(kwargs.get("max_new_tokens", 4))
    token_ids = list(range(10, 10 + max_new))
    return GenerationOutput(
        generation_completed=True,
        generation_output_text=f"out_{compressor}_{max_new}",
        generation_output_token_ids=token_ids,
        prompt_ids=torch.tensor([[1, 2, 3]]),
        exactkv_failures=0,
        token_exact_match=True,
        compressor_name=str(compressor),
    )


def _fake_shadow_replay(**kwargs: object) -> dict:
    input_ids = kwargs["input_ids"]
    n = int(input_ids.shape[-1])
    return {
        "full_model_parity_status": "passed",
        "num_layers_replayed": 2,
        "streaming_vs_materialized_logit_metrics": {
            "max_abs_error": 1e-3,
            "top1_agreement": True,
            "top5_overlap": 5,
            "top10_overlap": 10,
        },
        "full_vs_streaming_logit_metrics": {"max_abs_error": 1e-3},
        "blockers": [],
        "sequence_length": n,
    }


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "prompts": default_exp078_prompts()[:2],
        "max_new_tokens_values": DEFAULT_EXP078_MAX_NEW_TOKENS,
        "compressors_requested": DEFAULT_EXP078_COMPRESSORS,
        "generation_fn": _fake_generation,
        "shadow_replay_fn": _fake_shadow_replay,
    }
    defaults.update(overrides)
    return run_exp078_expanded_panel(**defaults)


def test_exp078_report_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_078_ID
    assert validate_exp078_report(report) == []
    assert report["prompts_requested"] == 2
    assert set(report["max_new_tokens_values"]) == {4, 8}
    assert len(report["generation_cells"]) == 2 * 2 * len(report["compressors_run"])


def test_multiple_prompts_and_lengths() -> None:
    report = _run_panel(prompts=default_exp078_prompts()[:4])
    assert report["prompts_requested"] == 4
    prompt_ids = {c["prompt_id"] for c in report["generation_cells"]}
    assert len(prompt_ids) == 4
    max_new_set = {c["max_new_tokens"] for c in report["generation_cells"]}
    assert max_new_set == {4, 8}


def test_compressors_blocked_when_api_missing() -> None:
    requested = ["noop", "not_a_real_compressor_xyz"]
    runnable, blocked = resolve_panel_compressors(requested)
    assert "noop" in runnable
    assert any(b["compressor"] == "not_a_real_compressor_xyz" for b in blocked)
    assert any("blocked_compressor_api_missing" in b["reason"] for b in blocked)

    report = _run_panel(compressors_requested=requested)
    assert "noop" in report["compressors_run"]
    assert any(b["compressor"] == "not_a_real_compressor_xyz" for b in report["compressors_blocked"])


def test_safety_gates_always_false() -> None:
    report = _run_panel()
    assert report["generation_modified_by_shadow"] is False
    assert report["shadow_used_for_token_commit"] is False
    assert report["default_runtime_changed"] is False
    for cell in report["generation_cells"]:
        assert cell["generation_modified_by_shadow"] is False
        assert cell["shadow_used_for_token_commit"] is False


def test_prompt_plus_generated_succeeds_with_token_ids() -> None:
    report = _run_panel(prompts=default_exp078_prompts()[:1], compressors_requested=["noop"])
    cell = report["generation_cells"][0]
    assert cell["generation_output_token_ids_available"] is True
    ppg = next(
        sc for sc in cell["shadow_cells"]
        if sc["shadow_sequence_mode"] == "prompt_plus_generated_tokens"
    )
    assert ppg["shadow_status"] == "shadow_complete"
    assert ppg["shadow_sequence_length"] == 3 + 4  # prompt + max_new_tokens=4 for first cell


def test_prompt_plus_generated_blocks_without_token_ids() -> None:
    def _gen_no_ids(**kwargs: object) -> GenerationOutput:
        del kwargs
        return GenerationOutput(
            generation_completed=True,
            generation_output_text="out",
            generation_output_token_ids=None,
            prompt_ids=torch.tensor([[1, 2, 3]]),
        )

    report = _run_panel(
        prompts=default_exp078_prompts()[:1],
        compressors_requested=["noop"],
        max_new_tokens_values=[4],
        generation_fn=_gen_no_ids,
    )
    cell = report["generation_cells"][0]
    ppg = next(
        sc for sc in cell["shadow_cells"]
        if sc["shadow_sequence_mode"] in (
            "prompt_plus_generated_tokens",
            "blocked_missing_tokens",
        )
    )
    assert ppg["shadow_status"] == "shadow_blocked"
    assert report["prompt_plus_generated_blocked_cells"] >= 1


def test_exactkv_failure_summary_handles_missing_fields() -> None:
    def _gen_unknown(**kwargs: object) -> GenerationOutput:
        del kwargs
        return GenerationOutput(
            generation_completed=False,
            generation_output_text="",
            generation_output_token_ids=None,
        )

    report = _run_panel(
        prompts=default_exp078_prompts()[:1],
        compressors_requested=["noop"],
        max_new_tokens_values=[4],
        generation_fn=_gen_unknown,
    )
    summary = report["exactkv_failure_summary"]
    assert summary["cells_with_unknown_exactkv_status"] >= 1
    assert "total_generation_cells" in summary


def test_topk_agreement_supplementary_only() -> None:
    report = _run_panel()
    assert "topk_agreement_summary" in report
    assert "top1_agreement_cells" in report["topk_agreement_summary"]
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


def test_no_forbidden_claim_fields() -> None:
    report = _run_panel()
    assert set(SHADOW_FORBIDDEN_CLAIMS).issubset(set(report.get("forbidden_claims", [])))
    for forbidden in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "active_gpu_memory_savings",
        "production_memory_savings",
    ):
        assert forbidden not in report
