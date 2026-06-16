"""Tests for Experiment 076 generation-shadow observer smoke (Phase 16K)."""
from __future__ import annotations

import torch

from exactkv.attention.generation_shadow_observer import (
    EXPERIMENT_076_ID,
    GenerationOutput,
    GenerationShadowObserverConfig,
    build_exp076_report,
    run_generation_shadow_observer,
    validate_exp076_report,
)


def _fake_gen(**kwargs: object) -> GenerationOutput:
    del kwargs
    return GenerationOutput(
        generation_completed=True,
        generation_output_text="unchanged output",
        generation_output_token_ids=[99],
        prompt_ids=torch.tensor([[5, 6, 7]]),
    )


def _fake_shadow(**kwargs: object) -> dict:
    return {
        "full_model_parity_status": "passed",
        "num_layers_replayed": 2,
        "streaming_vs_materialized_logit_metrics": {
            "max_abs_error": 1e-3,
            "top1_agreement": True,
            "top5_overlap": 5,
            "top10_overlap": 10,
        },
        "streaming_vs_materialized_hidden_metrics": {"max_abs_error": 1e-4},
        "full_vs_streaming_logit_metrics": {"max_abs_error": 1e-3},
        "blockers": [],
    }


def test_exp076_disabled_observer_skipped() -> None:
    report = build_exp076_report(
        run_generation_shadow_observer(
            [],
            config=GenerationShadowObserverConfig(shadow_observer_enabled=False),
        )
    )
    report["status"] = "skipped"
    assert report["generation_shadow_observer_enabled"] is False
    assert validate_exp076_report(report) == []


def test_exp076_report_experiment_id() -> None:
    observer = run_generation_shadow_observer(
        [("p0", "prompt")],
        config=GenerationShadowObserverConfig(shadow_observer_enabled=True),
        generation_fn=_fake_gen,
        shadow_replay_fn=_fake_shadow,
        runtime_loader=lambda **k: (type("R", (), {"model": object(), "device": "cpu"})(), None),
    )
    report = build_exp076_report(observer)
    assert report["experiment_id"] == EXPERIMENT_076_ID
    assert report["shadow_successful_prompts"] == 1
    assert validate_exp076_report(report) == []


def test_exp076_tolerance_and_topk_in_report() -> None:
    observer = run_generation_shadow_observer(
        [("p0", "prompt")],
        config=GenerationShadowObserverConfig(shadow_observer_enabled=True),
        generation_fn=_fake_gen,
        shadow_replay_fn=_fake_shadow,
        runtime_loader=lambda **k: (type("R", (), {"model": object(), "device": "cpu"})(), None),
    )
    report = build_exp076_report(observer)
    assert report["topk_agreement_summary"]["top1_agreement_prompts"] == 1
    assert report["tolerance_policy_status_counts"]
