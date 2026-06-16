"""Tests for Experiment 073 Qwen-family divergence panel (Phase 16H)."""
from __future__ import annotations

import torch
import torch.nn as nn

from exactkv.attention.hf_full_replay_probe import (
    EXPERIMENT_073_ID,
    FORBIDDEN_ATTENTION_CLAIMS,
    classify_model_panel_entry,
    run_exp073_probe,
    validate_exp073_report,
)
from exactkv.attention.hf_single_layer_probe import (
    extract_qwen_model_architecture,
    probe_qwen_architecture_support,
)
from tests.test_hf_full_replay_probe import (
    _DummyModel,
    _FakeTokenizer,
    _mock_loader,
    _mock_prompts,
)


class _BrokenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(64)
        self.post_attention_layernorm = nn.LayerNorm(64)
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))


class _UnsupportedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 64,
        })()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 64)
        self.model.layers = nn.ModuleList([_BrokenLayer()])
        self.model.norm = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 128, bias=False)

    def eval(self) -> "_UnsupportedModel":
        return self

    def to(self, device: str) -> "_UnsupportedModel":
        return self


def _panel_loader(**kwargs: object) -> tuple[_DummyModel, _FakeTokenizer]:
    del kwargs
    return _DummyModel(depth=2), _FakeTokenizer()


def _failing_loader(**kwargs: object) -> tuple[_DummyModel, _FakeTokenizer]:
    del kwargs
    raise RuntimeError("mock model load failed")


def test_classify_free_running_accumulation_confirmed() -> None:
    cls = classify_model_panel_entry(
        model_load_succeeded=True,
        architecture_supported=True,
        parity_passed=True,
        teacher_forced_max_attn=1e-6,
        teacher_forced_max_post_mlp=1e-6,
        free_running_max_post_mlp=0.5,
        free_running_root_cause_counts={"free_running_accumulation": 4},
    )
    assert cls == "free_running_accumulation_confirmed"


def test_classify_local_attention_mismatch() -> None:
    cls = classify_model_panel_entry(
        model_load_succeeded=True,
        architecture_supported=True,
        parity_passed=True,
        teacher_forced_max_attn=0.01,
        teacher_forced_max_post_mlp=0.01,
        free_running_max_post_mlp=0.5,
        free_running_root_cause_counts={"local_attention_mismatch": 4},
    )
    assert cls == "local_attention_mismatch_detected"


def test_classify_parity_failure() -> None:
    cls = classify_model_panel_entry(
        model_load_succeeded=True,
        architecture_supported=True,
        parity_passed=False,
        teacher_forced_max_attn=1e-6,
        teacher_forced_max_post_mlp=1e-6,
        free_running_max_post_mlp=0.5,
        free_running_root_cause_counts={},
    )
    assert cls == "parity_failure"


def test_classify_model_load_blocked() -> None:
    cls = classify_model_panel_entry(
        model_load_succeeded=False,
        architecture_supported=False,
        parity_passed=False,
        teacher_forced_max_attn=0.0,
        teacher_forced_max_post_mlp=0.0,
        free_running_max_post_mlp=0.0,
        free_running_root_cause_counts={},
    )
    assert cls == "model_load_blocked"


def test_probe_qwen_architecture_support_dummy() -> None:
    model = _DummyModel(depth=2)
    ok, blockers, arch = probe_qwen_architecture_support(model)
    assert ok is True
    assert blockers == []
    assert arch["num_layers"] == 2


def test_probe_qwen_architecture_unsupported() -> None:
    model = _UnsupportedModel()
    ok, blockers, _arch = probe_qwen_architecture_support(model)
    assert ok is False
    assert blockers


def test_extract_qwen_model_architecture() -> None:
    model = _DummyModel(depth=3)
    arch = extract_qwen_model_architecture(model)
    assert arch["num_layers"] == 3


def _minimal_panel_report(**overrides: object) -> dict:
    base = {
        "experiment_id": EXPERIMENT_073_ID,
        "status": "diagnostic_complete",
        "model_ids": ["mock-a", "mock-b"],
        "loaded_models": ["mock-a"],
        "blocked_models": [{"model_id": "mock-b", "classification": "model_load_blocked", "blockers": ["fail"]}],
        "device": "cpu",
        "dtype": "float32",
        "target_token_lengths": [32],
        "chunk_sizes": [16],
        "total_cells": 2,
        "successful_cells": 2,
        "blocked_cells": 0,
        "model_level_classifications": {
            "mock-a": "free_running_accumulation_confirmed",
            "mock-b": "model_load_blocked",
        },
        "teacher_forced_local_error_summary_by_model": {
            "mock-a": {"max_post_mlp_error": 1e-6, "max_attn_context_error": 1e-6, "cell_count": 1},
            "mock-b": {"max_post_mlp_error": 0.0, "max_attn_context_error": 0.0, "cell_count": 0},
        },
        "free_running_error_summary_by_model": {
            "mock-a": {"max_post_mlp_error": 0.5, "max_attn_context_error": 0.1, "cell_count": 1},
            "mock-b": {"max_post_mlp_error": 0.0, "max_attn_context_error": 0.0, "cell_count": 0},
        },
        "final_topk_agreement_summary_by_model": {
            "mock-a": {
                "max_final_hidden_error": 0.5,
                "max_final_logit_error": 0.1,
                "top1_agreement_cells": 1,
                "top5_overlap_mean": 5.0,
                "top10_overlap_mean": 10.0,
                "cell_count": 1,
            },
            "mock-b": {
                "max_final_hidden_error": 0.0,
                "max_final_logit_error": 0.0,
                "top1_agreement_cells": 0,
                "top5_overlap_mean": 0.0,
                "top10_overlap_mean": 0.0,
                "cell_count": 0,
            },
        },
        "root_cause_counts_by_model": {
            "mock-a": {"free_running_accumulation": 1},
            "mock-b": {},
        },
        "memory_accounting_summary_by_model": {
            "mock-a": {
                "best_theoretical_streaming_reduction_max": 0.6,
                "best_theoretical_streaming_reduction_mean": 0.6,
                "cell_count": 2,
            },
            "mock-b": {
                "best_theoretical_streaming_reduction_max": 0.0,
                "best_theoretical_streaming_reduction_mean": 0.0,
                "cell_count": 0,
            },
        },
        "limitations": ["test"],
        "no_performance_claims_note": "no performance claims",
        "claim_note": "test",
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "model_entries": [
            {
                "model_id": "mock-a",
                "model_load_succeeded": True,
                "architecture_supported": True,
                "classification": "free_running_accumulation_confirmed",
                "blockers": [],
                "cells": [
                    {
                        "model_id": "mock-a",
                        "prompt_id": "p0",
                        "target_token_length": 32,
                        "chunk_size": 16,
                        "trace_mode": "free_running",
                        "root_cause_classification": "free_running_accumulation",
                        "passed": False,
                        "blockers": [],
                    },
                ],
            },
            {
                "model_id": "mock-b",
                "model_load_succeeded": False,
                "architecture_supported": False,
                "classification": "model_load_blocked",
                "blockers": ["fail"],
                "cells": [],
            },
        ],
    }
    base.update(overrides)
    return base


def test_validate_exp073_mixed_panel() -> None:
    assert validate_exp073_report(_minimal_panel_report()) == []


def test_blocked_model_entry_preserves_blockers() -> None:
    report = _minimal_panel_report()
    blocked = report["blocked_models"][0]
    assert blocked["blockers"] == ["fail"]
    assert report["model_entries"][1]["blockers"] == ["fail"]


def test_no_forbidden_claim_fields() -> None:
    blob = str({"forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS)}).lower()
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "active_gpu_memory_savings",
        "production_memory_savings",
    ):
        assert term not in blob or term in FORBIDDEN_ATTENTION_CLAIMS


def test_run_exp073_mock_success() -> None:
    report = run_exp073_probe(
        model_ids=["mock/success"],
        target_token_lengths=[32],
        chunk_sizes=[16],
        max_prompts=1,
        model_loader=_panel_loader,
        prompt_provider=_mock_prompts,
    )
    assert report["experiment_id"] == EXPERIMENT_073_ID
    assert report["loaded_models"] == ["mock/success"]
    assert report["successful_cells"] > 0
    assert validate_exp073_report(report) == []


def test_run_exp073_mock_blocked_load() -> None:
    report = run_exp073_probe(
        model_ids=["mock/fail"],
        model_loader=_failing_loader,
        prompt_provider=_mock_prompts,
    )
    assert report["loaded_models"] == []
    assert report["blocked_models"][0]["classification"] == "model_load_blocked"
    assert report["model_entries"][0]["blockers"]


def test_run_exp073_mock_unsupported_architecture() -> None:
    def _unsupported_loader(**kwargs: object) -> tuple[_UnsupportedModel, _FakeTokenizer]:
        del kwargs
        return _UnsupportedModel(), _FakeTokenizer()

    report = run_exp073_probe(
        model_ids=["mock/unsupported"],
        target_token_lengths=[32],
        chunk_sizes=[16],
        max_prompts=1,
        model_loader=_unsupported_loader,
        prompt_provider=_mock_prompts,
    )
    entry = report["model_entries"][0]
    assert entry["classification"] == "unsupported_architecture"
    assert entry["architecture_supported"] is False
    assert entry["blockers"]
