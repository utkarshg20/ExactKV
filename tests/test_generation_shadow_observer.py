"""Tests for external L1 generation-shadow observer (Phase 16K)."""
from __future__ import annotations

import torch

from exactkv.attention.generation_shadow_observer import (
    EXP076_CLAIM_NOTE,
    GenerationOutput,
    GenerationShadowObserverConfig,
    GenerationShadowStatus,
    build_exp076_report,
    observe_prompt,
    reconstruct_shadow_input_ids,
    run_generation_shadow_observer,
    validate_exp076_report,
)
from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS


def _fake_generation(**kwargs: object) -> GenerationOutput:
    del kwargs
    return GenerationOutput(
        generation_completed=True,
        generation_output_text="PARIS_TOKEN_OUTPUT",
        generation_output_token_ids=[10, 20, 30],
        prompt_ids=torch.tensor([[1, 2, 3, 4]]),
        full_sequence_ids=torch.tensor([[1, 2, 3, 4, 10, 20, 30]]),
    )


def _fake_shadow_replay(**kwargs: object) -> dict:
    return {
        "full_model_parity_status": "passed",
        "num_layers_replayed": 2,
        "streaming_vs_materialized_logit_metrics": {
            "max_abs_error": 0.1,
            "top1_agreement": True,
            "top5_overlap": 5,
            "top10_overlap": 10,
        },
        "streaming_vs_materialized_hidden_metrics": {"max_abs_error": 0.5},
        "full_vs_streaming_logit_metrics": {"max_abs_error": 0.12, "top1_agreement": True},
        "top1_changed_full_vs_streaming": False,
        "blockers": [],
    }


def test_generation_output_unchanged() -> None:
    cfg = GenerationShadowObserverConfig(shadow_observer_enabled=True, shadow_mode="prompt_prefix_only")
    result = observe_prompt(
        prompt_id="p0",
        prompt_text="test prompt",
        config=cfg,
        generation_fn=_fake_generation,
        shadow_replay_fn=_fake_shadow_replay,
        hf_model=object(),
    )
    assert result.generation_output_preview == "PARIS_TOKEN_OUTPUT"
    assert result.generation_output_token_ids_available is True
    assert result.generation_completed is True


def test_shadow_runs_after_generation() -> None:
    cfg = GenerationShadowObserverConfig(shadow_observer_enabled=True)
    result = observe_prompt(
        prompt_id="p0",
        prompt_text="test",
        config=cfg,
        generation_fn=_fake_generation,
        shadow_replay_fn=_fake_shadow_replay,
        hf_model=object(),
    )
    assert result.shadow_ran_after_generation is True
    assert result.shadow_status == GenerationShadowStatus.SHADOW_COMPLETE.value


def test_safety_flags_always_false() -> None:
    observer = run_generation_shadow_observer(
        [("p0", "hello")],
        config=GenerationShadowObserverConfig(shadow_observer_enabled=True),
        generation_fn=_fake_generation,
        shadow_replay_fn=_fake_shadow_replay,
        runtime_loader=lambda **k: (type("R", (), {"model": object(), "device": "cpu"})(), None),
    )
    assert observer.shadow_used_for_token_commit is False
    assert observer.generation_modified_by_shadow is False
    assert observer.default_runtime_changed is False
    report = build_exp076_report(observer)
    assert report["shadow_used_for_token_commit"] is False
    assert report["generation_modified_by_shadow"] is False
    assert report["default_runtime_changed"] is False


def test_prompt_prefix_only_mode() -> None:
    gen = _fake_generation()
    ids, mode, blockers = reconstruct_shadow_input_ids(
        gen, shadow_mode="prompt_prefix_only",
    )
    assert blockers == []
    assert mode == "prompt_prefix_only"
    assert ids is not None
    assert ids.shape[-1] == 4


def test_prompt_plus_generated_tokens_mode() -> None:
    gen = _fake_generation()
    ids, mode, blockers = reconstruct_shadow_input_ids(
        gen, shadow_mode="prompt_plus_generated_tokens",
    )
    assert blockers == []
    assert mode == "prompt_plus_generated_tokens"
    assert ids is not None
    assert ids.shape[-1] == 7


def test_blocked_missing_tokens_mode() -> None:
    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="x",
        generation_output_token_ids=None,
        prompt_ids=None,
    )
    ids, mode, blockers = reconstruct_shadow_input_ids(
        gen, shadow_mode="prompt_plus_generated_tokens",
    )
    assert ids is None
    assert mode == "blocked_missing_tokens"
    assert blockers


def test_generation_api_missing_blocked() -> None:
    cfg = GenerationShadowObserverConfig(shadow_observer_enabled=True)
    result = observe_prompt(prompt_id="p0", prompt_text="x", config=cfg)
    assert result.shadow_status == GenerationShadowStatus.GENERATION_BLOCKED.value
    assert "generation API missing" in result.blockers[0]


def test_report_schema_validates() -> None:
    observer = run_generation_shadow_observer(
        [("p0", "hello")],
        config=GenerationShadowObserverConfig(shadow_observer_enabled=True),
        generation_fn=_fake_generation,
        shadow_replay_fn=_fake_shadow_replay,
        runtime_loader=lambda **k: (type("R", (), {"model": object(), "device": "cpu"})(), None),
    )
    report = build_exp076_report(observer)
    assert validate_exp076_report(report) == []


def test_no_forbidden_claim_fields() -> None:
    blob = str({"forbidden": list(SHADOW_FORBIDDEN_CLAIMS), "note": EXP076_CLAIM_NOTE}).lower()
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "active_gpu_memory_savings",
        "production_memory_savings",
    ):
        assert term in blob


def test_default_exp078_prompts_count() -> None:
    from exactkv.attention.generation_shadow_observer import default_exp078_prompts

    assert len(default_exp078_prompts()) == 8


def test_resolve_panel_compressors_blocks_unknown() -> None:
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    runnable, blocked = resolve_panel_compressors(["noop", "missing_compressor_xyz"])
    assert "noop" in runnable
    assert any(b["reason"] == "blocked_compressor_api_missing" for b in blocked)


def test_build_decode_prefix_ladder_includes_k0() -> None:
    from exactkv.attention.generation_shadow_observer import build_decode_prefix_ladder

    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="x",
        generation_output_token_ids=[10, 11],
        prompt_ids=torch.tensor([[1, 2, 3]]),
    )
    ladder, blockers = build_decode_prefix_ladder(gen)
    assert blockers == []
    assert [k for k, _ in ladder] == [0, 1, 2]


def test_extract_round_log_entries_from_traces() -> None:
    from exactkv.attention.generation_shadow_observer import extract_round_log_entries
    from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace

    acc = AcceptanceResult(
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
    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="x",
        generation_output_token_ids=[10, 11],
        prompt_ids=torch.tensor([[1, 2, 3]]),
        exactkv_traces=[
            VerificationTrace(
                round_idx=0,
                draft_tokens=[10, 11],
                acceptance=acc,
                full_seq_len_before=3,
                full_seq_len_after=5,
                compressed_seq_len_after=5,
            ),
        ],
    )
    entries, blockers = extract_round_log_entries(gen)
    assert blockers == []
    assert entries[0]["round_index"] == 0
    assert entries[0]["accepted_token_count"] == 2
