"""Tests for Experiment 077 prompt+generated generation-shadow panel (Phase 16L)."""
from __future__ import annotations

import torch

from scripts.research.run_exp077_generation_shadow_prompt_plus_generated_panel import (
    EXPERIMENT_077_ID,
    run_exp077_panel,
    validate_exp077_report,
)
from exactkv.attention.generation_shadow_observer import GenerationOutput


def _fake_generation(**kwargs: object) -> GenerationOutput:
    # generation runs once; returns prompt_ids and generated token ids.
    prompt = kwargs.get("prompt", "")
    del prompt
    return GenerationOutput(
        generation_completed=True,
        generation_output_text="out",
        generation_output_token_ids=[9, 10, 11, 12],
        prompt_ids=torch.tensor([[1, 2, 3]]),
    )


def _fake_shadow_replay(**kwargs: object) -> dict:
    input_ids = kwargs["input_ids"]
    # encode length into metrics so tests can assert correctness.
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
        "streaming_vs_materialized_hidden_metrics": {"max_abs_error": float(n) / 1000.0},
        "full_vs_streaming_logit_metrics": {"max_abs_error": 1e-3},
        "blockers": [],
    }


def test_exp077_report_validates() -> None:
    report = run_exp077_panel(
        model_id="mock",
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        shadow_modes=["prompt_prefix_only", "prompt_plus_generated_tokens"],
        compressors=["noop"],
        local_files_only=True,
        allow_shadow_fail=True,
        max_prompts=2,
        generation_fn=_fake_generation,
        shadow_replay_fn=_fake_shadow_replay,
    )
    assert report["experiment_id"] == EXPERIMENT_077_ID
    assert validate_exp077_report(report) == []
    assert report["generation_modified_by_shadow"] is False
    assert report["shadow_used_for_token_commit"] is False
    assert report["default_runtime_changed"] is False


def test_prompt_plus_generated_uses_generated_token_ids() -> None:
    report = run_exp077_panel(
        model_id="mock",
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        shadow_modes=["prompt_prefix_only", "prompt_plus_generated_tokens"],
        compressors=["noop"],
        local_files_only=True,
        allow_shadow_fail=True,
        max_prompts=1,
        generation_fn=_fake_generation,
        shadow_replay_fn=_fake_shadow_replay,
    )
    pr = report["prompt_results"][0]
    assert pr["generation_output_token_ids_available"] is True
    cells = pr["shadow_cells"]
    assert len(cells) == 2
    prefix = next(c for c in cells if c["shadow_sequence_mode"] == "prompt_prefix_only")
    plus = next(c for c in cells if c["shadow_sequence_mode"] == "prompt_plus_generated_tokens")
    assert prefix["shadow_sequence_length"] == 3
    assert plus["shadow_sequence_length"] == 7


def test_missing_generated_token_ids_blocks_prompt_plus_generated() -> None:
    def _gen_missing(**kwargs: object) -> GenerationOutput:
        del kwargs
        return GenerationOutput(
            generation_completed=True,
            generation_output_text="out",
            generation_output_token_ids=None,
            prompt_ids=torch.tensor([[1, 2, 3]]),
        )

    report = run_exp077_panel(
        model_id="mock",
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        shadow_modes=["prompt_plus_generated_tokens"],
        compressors=["noop"],
        local_files_only=True,
        allow_shadow_fail=True,
        max_prompts=1,
        generation_fn=_gen_missing,
        shadow_replay_fn=_fake_shadow_replay,
    )
    cell = report["prompt_results"][0]["shadow_cells"][0]
    assert cell["shadow_status"] == "shadow_blocked"
    assert "generated token IDs unavailable" in " ".join(cell["blockers"])


def test_compressor_expansion_deferred_validates() -> None:
    report = run_exp077_panel(
        model_id="mock",
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        shadow_modes=["prompt_prefix_only"],
        compressors=["noop", "int8"],
        local_files_only=True,
        allow_shadow_fail=True,
        max_prompts=1,
        generation_fn=_fake_generation,
        shadow_replay_fn=_fake_shadow_replay,
    )
    assert validate_exp077_report(report) == []
    assert report["blockers"]

