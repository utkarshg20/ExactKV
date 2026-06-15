"""Tests for Experiment 068 Qwen RoPE/GQA long-context attention probe."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from exactkv.attention.hf_single_layer_probe import (
    EXPERIMENT_068_ID,
    EXP068_CLAIM_NOTE,
    generate_long_prompt_text,
    long_context_prompts,
    run_exp068_probe,
    validate_exp068_report,
    extract_qkv_from_qwen2_layer,
    _try_apply_qwen2_rope,
)
from exactkv.attention.streaming_quant_attention import FORBIDDEN_ATTENTION_CLAIMS

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_068_QWEN_ROPE_LONG_CONTEXT_ATTENTION_PROBE.md"


class _FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {f"t{i}": i + 1 for i in range(5000)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        count = max(1, len(text.split()))
        return list(range(1, count + 1))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{i}" for i in token_ids)

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        ids = self.encode(text, add_special_tokens=False)
        return {"input_ids": torch.tensor([ids])}


class _DummySelfAttn(nn.Module):
    def __init__(self, hidden: int = 64, num_heads: int = 4, num_kv_heads: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden, bias=False)


class _DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _DummySelfAttn()


class _MockRotaryEmb(nn.Module):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _ = position_ids
        b, _, t, d = x.shape
        cos = torch.ones(b, t, d, device=x.device, dtype=x.dtype)
        sin = torch.zeros(b, t, d, device=x.device, dtype=x.dtype)
        return cos, sin


def _metrics() -> dict[str, float]:
    return {
        "max_abs_error": 1e-6,
        "mean_abs_error": 1e-7,
        "cosine_similarity": 1.0,
        "relative_l2_error": 1e-6,
        "top_dim_max_abs": 1e-6,
    }


def _mem(num_chunks: int = 8) -> dict[str, object]:
    return {
        "full_kv_bytes": 16384,
        "stored_quantized_kv_bytes": 8192,
        "materialized_working_kv_bytes": 16384,
        "streaming_peak_chunk_working_kv_bytes": 4096,
        "metadata_bytes": 512,
        "chunk_size": 16,
        "num_chunks": num_chunks,
        "theoretical_streaming_working_reduction_vs_materialized": 0.75,
    }


def _sample_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model_id": "mock",
        "prompt_id": "long_128",
        "prompt_preview": "hello",
        "target_token_length": 128,
        "actual_token_length": 128,
        "layer_idx": 0,
        "extraction_status": "success",
        "extraction_mode": "exact_qwen2_like",
        "rope_status": "applied",
        "gqa_status": "repeated",
        "num_kv_heads_original": 2,
        "num_kv_heads_repeated": 4,
        "q_shape": [1, 4, 128, 16],
        "k_shape": [1, 4, 128, 16],
        "v_shape": [1, 4, 128, 16],
        "chunk_size": 16,
        "streaming_vs_materialized": _metrics(),
        "full_vs_materialized": _metrics(),
        "full_vs_streaming": {**_metrics(), "max_abs_error": 0.02},
        "memory_accounting": _mem(num_chunks=8),
        "passed": True,
        "tolerance": 5e-4,
        "blockers": [],
    }
    base.update(overrides)
    return base


def _blocked_cell() -> dict[str, object]:
    return {
        "model_id": "mock",
        "prompt_id": "long_64",
        "prompt_preview": "hello",
        "target_token_length": 64,
        "actual_token_length": 64,
        "layer_idx": 0,
        "extraction_status": "blocked",
        "extraction_mode": "blocked",
        "rope_status": "unsupported",
        "gqa_status": "failed",
        "chunk_size": 16,
        "passed": False,
        "blockers": ["model load failed"],
    }


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_068_ID,
        "status": "pass",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "model_load_succeeded": True,
        "target_token_lengths": [64, 128, 256],
        "layers": [0, 12, 23],
        "chunk_sizes": [16, 32, 64],
        "total_cells": 1,
        "successful_cells": 1,
        "blocked_cells": 0,
        "extraction_mode_counts": {"exact_qwen2_like": 1},
        "rope_status_counts": {"applied": 1},
        "gqa_status_counts": {"repeated": 1},
        "streaming_vs_materialized_pass_cells": 1,
        "max_streaming_vs_materialized_error": 1e-6,
        "full_vs_streaming_drift_summary": {
            "max_abs_error": 0.02,
            "mean_abs_error": 0.01,
            "cell_count": 1,
        },
        "full_vs_materialized_drift_summary": {
            "max_abs_error": 0.03,
            "mean_abs_error": 0.01,
            "cell_count": 1,
        },
        "output_projection_drift_summary": {
            "cells_with_output_projection": 1,
            "max_full_vs_streaming_after_o_proj": 0.02,
            "mean_full_vs_streaming_after_o_proj": 0.01,
        },
        "memory_accounting_summary": {
            "best_theoretical_streaming_working_reduction": 0.875,
            "worst_theoretical_streaming_working_reduction": 0.5,
            "cells_with_reduction_gt_zero": 1,
        },
        "longest_context_tested": 128,
        "max_num_chunks": 8,
        "best_theoretical_streaming_reduction": 0.875,
        "layer_parity_attempted": False,
        "layer_parity_summary": None,
        "extraction_blockers": [],
        "cells": [_sample_cell()],
        "claim_note": EXP068_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": ["offline only"],
        "no_performance_claims_note": "no throughput claim",
        "prompt_count": 1,
    }
    base.update(overrides)
    return base


def test_rope_success_marks_exact_qwen2_like() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(1, 8, 64)
    rotary = _MockRotaryEmb()
    position_ids = torch.arange(8).unsqueeze(0)
    extracted = extract_qkv_from_qwen2_layer(
        hidden,
        layer,
        layer_idx=0,
        rotary_emb=rotary,
        position_ids=position_ids,
        allow_projection_only=True,
    )
    assert extracted.extraction_mode == "exact_qwen2_like"
    assert extracted.rope_status == "applied"


def test_rope_failure_falls_back_to_projection_only() -> None:
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    position_ids = torch.arange(8).unsqueeze(0)

    class _BadRotary(nn.Module):
        def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            raise RuntimeError("rotary boom")

    _, _, status = _try_apply_qwen2_rope(q, k, v, rotary_emb=_BadRotary(), position_ids=position_ids)
    assert status.startswith("failed:")


def test_gqa_repeat_expands_kv_heads() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(1, 10, 64)
    extracted = extract_qkv_from_qwen2_layer(hidden, layer, layer_idx=0)
    assert extracted.gqa_status == "repeated"
    assert extracted.num_kv_heads_original == 2
    assert extracted.num_kv_heads_repeated == 4
    assert extracted.k.shape[1] == extracted.q.shape[1]


def test_long_prompt_target_length_enforced() -> None:
    tok = _FakeTokenizer()
    text, actual = generate_long_prompt_text(tok, 64)
    assert actual >= 64
    assert len(tok.encode(text, add_special_tokens=False)) >= 64


def test_long_context_prompts_respect_max_prompts() -> None:
    tok = _FakeTokenizer()
    prompts = long_context_prompts(tok, [64, 128, 256, 512], max_prompts=3)
    assert len(prompts) == 3
    assert prompts[0][2] == 64
    assert prompts[2][2] == 256


def test_memory_accounting_num_chunks_gt_one() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(1, 128, 64)
    extracted = extract_qkv_from_qwen2_layer(hidden, layer, layer_idx=0)
    from exactkv.attention.hf_single_layer_probe import run_hf_attention_drift_cell

    cell = run_hf_attention_drift_cell(
        extracted,
        chunk_size=16,
        target_token_length=128,
        actual_token_length=128,
    )
    mem = cell["memory_accounting"]
    assert mem["num_chunks"] > 1
    assert mem["streaming_peak_chunk_working_kv_bytes"] < mem["materialized_working_kv_bytes"]


def test_exp068_success_report_validates() -> None:
    assert validate_exp068_report(_report()) == []


def test_exp068_blocked_report_validates() -> None:
    assert validate_exp068_report(
        _report(
            status="blocked",
            successful_cells=0,
            blocked_cells=1,
            streaming_vs_materialized_pass_cells=0,
            cells=[_blocked_cell()],
            model_load_succeeded=False,
            extraction_blockers=["model load failed"],
        )
    ) == []


def test_extraction_mode_counts_in_report() -> None:
    report = _report(
        extraction_mode_counts={"exact_qwen2_like": 2, "projection_only": 1},
        rope_status_counts={"applied": 2, "unsupported": 1},
        gqa_status_counts={"repeated": 3},
    )
    assert report["extraction_mode_counts"]["exact_qwen2_like"] == 2


def test_no_forbidden_performance_fields() -> None:
    body = {k: v for k, v in _report().items() if k not in ("forbidden_claims", "claim_note")}
    blob = json.dumps(body).lower()
    for term in ("throughput_improved", "latency_improved", "speedup_claim", "active_gpu_memory_savings"):
        assert term not in blob


def test_run_exp068_with_mock_model() -> None:
    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_DummyLayer(), _DummyLayer(), _DummyLayer()])
            self.model.rotary_emb = _MockRotaryEmb()

        def eval(self) -> "_MockModel":
            return self

        def to(self, device: str) -> "_MockModel":
            return self

        def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            b, t = input_ids.shape
            hs = [torch.randn(b, t, 64) for _ in range(4)]

            class _Out:
                hidden_states = tuple(hs)

            return _Out()

    def _loader(**kwargs: object) -> tuple[_MockModel, _FakeTokenizer]:
        return _MockModel(), _FakeTokenizer()

    report = run_exp068_probe(
        model_id="mock",
        target_token_lengths=(64, 128),
        chunk_sizes=(16,),
        max_prompts=2,
        layer_indices=[0, 1],
        model_loader=_loader,
        attempt_layer_parity=False,
    )
    assert report["model_load_succeeded"] is True
    assert report["successful_cells"] > 0
    assert report["max_num_chunks"] >= 1


def test_doc_required_wording() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "offline single-layer long-context attention probe",
        "not model generation integration",
        "not wired into exactkv generation",
        "rope/gqa",
        "no cuda",
        "no triton",
        "no vllm integration",
        "vericache",
        "theoretical memory accounting",
        "phase 16b",
    ):
        assert phrase in text, phrase
