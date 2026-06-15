"""Tests for HF single-layer attention drift probe (Phase 16B)."""
from __future__ import annotations

import torch
import torch.nn as nn

from exactkv.attention.hf_single_layer_probe import (
    compute_drift_metrics,
    extract_qkv_from_qwen2_layer,
    run_hf_attention_drift_cell,
    run_exp067_probe,
)


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


def test_extract_qkv_from_dummy_qwen_like_layer() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(1, 8, 64)
    extracted = extract_qkv_from_qwen2_layer(
        hidden,
        layer,
        layer_idx=0,
        rotary_emb=None,
        position_ids=None,
        allow_projection_only=True,
    )
    assert extracted.extraction_mode == "projection_only"
    assert extracted.grouped_query_status == "repeated"
    assert extracted.q.shape == (1, 4, 8, 16)
    assert extracted.k.shape == (1, 4, 8, 16)
    assert extracted.v.shape == (1, 4, 8, 16)


def test_blocked_extraction_without_self_attn() -> None:
    layer = nn.Module()
    hidden = torch.randn(1, 4, 32)
    extracted = extract_qkv_from_qwen2_layer(hidden, layer, layer_idx=0)
    assert extracted.extraction_mode == "blocked"
    assert extracted.blockers


def test_gqa_repeat_behavior() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(2, 5, 64)
    extracted = extract_qkv_from_qwen2_layer(hidden, layer, layer_idx=0)
    assert extracted.k.shape[1] == extracted.q.shape[1]


def test_compute_drift_metrics() -> None:
    a = torch.ones(1, 2, 3, 4)
    b = a.clone()
    m = compute_drift_metrics(a, b)
    assert m.max_abs_error == 0.0
    assert m.cosine_similarity > 0.99


def test_streaming_vs_materialized_path_on_extracted_qkv() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(1, 32, 64)
    extracted = extract_qkv_from_qwen2_layer(hidden, layer, layer_idx=0)
    cell = run_hf_attention_drift_cell(extracted, chunk_size=16, causal=True)
    assert cell["extraction_status"] == "success"
    assert cell["passed"] is True
    assert cell["streaming_vs_materialized"]["max_abs_error"] < 5e-4
    assert "output_projection" in cell
    assert cell["memory_accounting"]["full_kv_bytes"] > 0


def test_projection_only_status_recorded() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(1, 6, 64)
    extracted = extract_qkv_from_qwen2_layer(hidden, layer, layer_idx=0)
    assert extracted.rope_status in ("unsupported", "skipped")
    assert extracted.extraction_mode == "projection_only"


def test_rope_applied_with_mock_rotary() -> None:
    class _MockRotaryEmb(nn.Module):
        def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            b, _, t, d = x.shape
            cos = torch.ones(b, t, d, device=x.device, dtype=x.dtype)
            sin = torch.zeros(b, t, d, device=x.device, dtype=x.dtype)
            return cos, sin

    layer = _DummyLayer()
    hidden = torch.randn(1, 6, 64)
    position_ids = torch.arange(6).unsqueeze(0)
    extracted = extract_qkv_from_qwen2_layer(
        hidden,
        layer,
        layer_idx=0,
        rotary_emb=_MockRotaryEmb(),
        position_ids=position_ids,
    )
    assert extracted.extraction_mode == "exact_qwen2_like"
    assert extracted.rope_status == "applied"


def test_run_exp067_with_mock_model() -> None:
    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_DummyLayer(), _DummyLayer()])
            self.model.rotary_emb = None

        def eval(self) -> "_MockModel":
            return self

        def to(self, device: str) -> "_MockModel":
            return self

        def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            b, t = input_ids.shape
            hs0 = torch.randn(b, t, 64)
            hs1 = torch.randn(b, t, 64)
            hs2 = torch.randn(b, t, 64)

            class _Out:
                hidden_states = (hs0, hs1, hs2)

            return _Out()

    class _MockTokenizer:
        def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    def _loader(**kwargs: object) -> tuple[_MockModel, _MockTokenizer]:
        return _MockModel(), _MockTokenizer()

    report = run_exp067_probe(
        model_id="mock",
        chunk_sizes=(16,),
        max_prompts=2,
        layer_indices=[0, 1],
        model_loader=_loader,
    )
    assert report["model_load_succeeded"] is True
    assert report["successful_cells"] > 0
    assert report["status"] in ("pass", "failed")


def test_memory_accounting_non_negative_in_cell() -> None:
    layer = _DummyLayer()
    hidden = torch.randn(1, 64, 64)
    extracted = extract_qkv_from_qwen2_layer(hidden, layer, layer_idx=0)
    cell = run_hf_attention_drift_cell(extracted, chunk_size=32)
    mem = cell["memory_accounting"]
    assert mem["streaming_peak_chunk_working_kv_bytes"] <= mem["materialized_working_kv_bytes"]
