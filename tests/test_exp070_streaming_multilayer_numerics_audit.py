"""Tests for Experiment 070 streaming multi-layer numerics audit (Phase 16E)."""
from __future__ import annotations

import torch
import torch.nn as nn

from exactkv.attention.hf_multilayer_probe import (
    EXPERIMENT_070_ID,
    FORBIDDEN_ATTENTION_CLAIMS,
    PHASE16D_REGRESSION_CELL,
    recommend_tolerance_policy,
    run_exp070_probe,
    validate_exp070_report,
)
from tests.test_hf_multilayer_probe import (
    _DummyDecoderLayer,
    _MockRotaryEmb,
)


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Deterministic length from text hash for long-context mock paths
        length = 128 if "128" in text or len(text) > 100 else 64
        return list(range(1, length + 1))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{i}" for i in token_ids)

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        ids = self.encode(text)
        return {"input_ids": torch.tensor([ids])}


def _mock_prompts(
    tokenizer: _FakeTokenizer,
    lengths: tuple[int, ...],
    max_prompts: int,
) -> list[tuple[str, str, int, int]]:
    del tokenizer
    out: list[tuple[str, str, int, int]] = []
    for target in lengths[:max_prompts]:
        out.append((f"long_{target}", f"mock prompt target {target}", target, target))
    return out


class _MockModel(nn.Module):
    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(5000, 64)
        self.model.layers = nn.ModuleList(
            [_DummyDecoderLayer() for _ in range(depth)]
        )
        self.model.rotary_emb = _MockRotaryEmb()

    def eval(self) -> "_MockModel":
        return self

    def to(self, device: str) -> "_MockModel":
        return self


def _mock_loader(**kwargs: object) -> tuple[_MockModel, _FakeTokenizer]:
    return _MockModel(depth=4), _FakeTokenizer()


def _minimal_report() -> dict:
    return {
        "experiment_id": EXPERIMENT_070_ID,
        "status": "pass",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "target_token_lengths": [128],
        "prefix_layer_counts": [4],
        "chunk_sizes": [32],
        "accumulator_modes": ["default", "float32", "float64"],
        "total_cells": 3,
        "successful_cells": 3,
        "blocked_cells": 0,
        "failed_cells_under_strict_tolerance": 1,
        "failed_cells_under_recommended_tolerance": 0,
        "phase16d_regression_target": PHASE16D_REGRESSION_CELL,
        "phase16d_failure_reproduced": True,
        "phase16d_failure_status_after_audit": "reproduced_failure",
        "max_error_by_accumulator_mode": {"default": 0.001, "float32": 1e-5},
        "max_error_by_prefix_depth": {"4": 0.001},
        "max_error_by_chunk_size": {"32": 0.001},
        "tolerance_policy_recommendation": {
            "policy": "keep_strict_tolerance_use_float32_accumulator",
            "rationale": "test",
            "strict_tolerance": 5e-4,
            "recommended_tolerance_formula": "strict_16d",
        },
        "algorithm_change_made": False,
        "limitations": ["test"],
        "no_performance_claims_note": "no performance claims",
        "claim_note": "test",
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "cells": [
            {
                "prompt_id": "long_128",
                "target_token_length": 128,
                "prefix_layer_count": 4,
                "chunk_size": 32,
                "accumulator_mode": "default",
                "strict_tolerance_pass": False,
                "recommended_tolerance_pass": True,
                "blockers": [],
                "phase16d_regression_target": True,
                "streaming_vs_materialized_hidden_metrics": {
                    "max_abs_error": 0.000579,
                    "mean_abs_error": 1e-6,
                    "cosine_similarity": 1.0,
                    "relative_l2_error": 1e-6,
                },
                "diagnostics": {
                    "any_layer_nan": False,
                    "any_layer_inf": False,
                    "layer_diagnostics": [],
                },
            },
            {
                "prompt_id": "long_128",
                "target_token_length": 128,
                "prefix_layer_count": 4,
                "chunk_size": 32,
                "accumulator_mode": "float32",
                "strict_tolerance_pass": True,
                "recommended_tolerance_pass": True,
                "blockers": [],
                "phase16d_regression_target": False,
                "streaming_vs_materialized_hidden_metrics": {
                    "max_abs_error": 1e-6,
                    "mean_abs_error": 1e-7,
                    "cosine_similarity": 1.0,
                    "relative_l2_error": 1e-7,
                },
                "diagnostics": {"any_layer_nan": False, "any_layer_inf": False},
            },
        ],
    }


def test_phase16d_regression_marker() -> None:
    assert PHASE16D_REGRESSION_CELL["target_token_length"] == 128
    assert PHASE16D_REGRESSION_CELL["chunk_size"] == 32
    assert PHASE16D_REGRESSION_CELL["phase16d_tolerance"] == 5e-4


def test_validate_exp070_report_schema() -> None:
    errors = validate_exp070_report(_minimal_report())
    assert errors == []


def test_strict_tolerance_failure_report_validates() -> None:
    report = _minimal_report()
    report["status"] = "failed"
    report["failed_cells_under_strict_tolerance"] = 2
    assert validate_exp070_report(report) == []


def test_recommended_tolerance_report_validates() -> None:
    report = _minimal_report()
    report["status"] = "pass_with_recommended_tolerance"
    report["failed_cells_under_recommended_tolerance"] = 0
    assert validate_exp070_report(report) == []


def test_no_forbidden_claim_fields() -> None:
    report = _minimal_report()
    blob = str(report).lower()
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "active_gpu_memory_savings",
        "production_memory_savings",
    ):
        assert term not in blob or term in report["forbidden_claims"]


def test_recommend_tolerance_policy_fp32_remedy() -> None:
    cells = [
        {
            "accumulator_mode": "default",
            "strict_tolerance_pass": False,
            "streaming_vs_materialized_hidden_metrics": {"max_abs_error": 0.0006},
        },
        {
            "accumulator_mode": "float32",
            "strict_tolerance_pass": True,
            "streaming_vs_materialized_hidden_metrics": {"max_abs_error": 1e-6},
        },
        {
            "accumulator_mode": "float64",
            "strict_tolerance_pass": True,
            "streaming_vs_materialized_hidden_metrics": {"max_abs_error": 1e-7},
        },
    ]
    policy = recommend_tolerance_policy(cells, dtype=torch.float32)
    assert policy["policy"] == "keep_strict_tolerance_use_float32_accumulator"


def test_run_exp070_mock_end_to_end() -> None:
    report = run_exp070_probe(
        model_id="mock",
        target_token_lengths=(128,),
        prefix_layer_counts=(4,),
        chunk_sizes=(32,),
        accumulator_modes=("default", "float32", "float64"),
        max_prompts=1,
        model_loader=_mock_loader,
        prompt_provider=_mock_prompts,
    )
    assert report["model_load_succeeded"] is True
    assert report["successful_cells"] > 0
    assert report["phase16d_regression_target"] == PHASE16D_REGRESSION_CELL
    assert validate_exp070_report(report) == []
    assert report["algorithm_change_made"] is False
    for cell in report["cells"]:
        if cell.get("diagnostics"):
            assert cell["diagnostics"].get("any_layer_nan") is False
            assert cell["diagnostics"].get("any_layer_inf") is False
