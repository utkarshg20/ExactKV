"""Tests for offline multi-layer drift accumulation probe (Phase 16D)."""
from __future__ import annotations

import torch
import torch.nn as nn

from exactkv.attention.hf_multilayer_probe import (
    PHASE16D_REGRESSION_CELL,
    aggregate_layer_memory,
    check_full_block_parity,
    replay_prefix_layers,
    run_multilayer_drift_cell,
    run_qwen_decoder_block,
    run_exp069_probe,
    LayerMemoryRecord,
)
from exactkv.attention.hf_single_layer_probe import extract_qkv_from_qwen2_layer


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


class _DummyMLP(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _DummyDecoderLayer(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _DummySelfAttn(hidden=hidden)
        self.post_attention_layernorm = nn.Identity()
        self.mlp = _DummyMLP(hidden=hidden)


class _MockRotaryEmb(nn.Module):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, t, d = x.shape
        cos = torch.ones(b, t, d, device=x.device, dtype=x.dtype)
        sin = torch.zeros(b, t, d, device=x.device, dtype=x.dtype)
        return cos, sin


def test_run_qwen_decoder_block_shape() -> None:
    layer = _DummyDecoderLayer()
    hidden = torch.randn(1, 32, 64)
    position_ids = torch.arange(32).unsqueeze(0)
    out, mem, _ = run_qwen_decoder_block(
        hidden,
        layer,
        layer_idx=0,
        attention_path="full",
        chunk_size=16,
        rotary_emb=_MockRotaryEmb(),
        position_ids=position_ids,
    )
    assert out.shape == (1, 32, 64)
    assert mem.full_kv_bytes > 0


def test_materialized_and_streaming_paths_shape() -> None:
    layer = _DummyDecoderLayer()
    hidden = torch.randn(1, 64, 64)
    position_ids = torch.arange(64).unsqueeze(0)
    rotary = _MockRotaryEmb()
    mat, _, _ = run_qwen_decoder_block(
        hidden, layer, layer_idx=0, attention_path="materialized_compressed",
        chunk_size=16, rotary_emb=rotary, position_ids=position_ids,
    )
    stream, _, _ = run_qwen_decoder_block(
        hidden, layer, layer_idx=0, attention_path="streaming_compressed",
        chunk_size=16, rotary_emb=rotary, position_ids=position_ids,
    )
    assert mat.shape == stream.shape == (1, 64, 64)


def test_replay_prefix_layers_multi_layer() -> None:
    layers = [_DummyDecoderLayer(), _DummyDecoderLayer()]
    hidden = torch.randn(1, 48, 64)
    position_ids = torch.arange(48).unsqueeze(0)
    out, mems, _ = replay_prefix_layers(
        hidden,
        layers,
        prefix_layer_count=2,
        attention_path="streaming_compressed",
        chunk_size=16,
        rotary_emb=_MockRotaryEmb(),
        position_ids=position_ids,
    )
    assert out.shape == (1, 48, 64)
    assert len(mems) == 2


def test_streaming_matches_materialized_multilayer() -> None:
    layers = [_DummyDecoderLayer(), _DummyDecoderLayer()]
    hidden = torch.randn(1, 64, 64)
    position_ids = torch.arange(64).unsqueeze(0)
    rotary = _MockRotaryEmb()
    mat, _, _ = replay_prefix_layers(
        hidden, layers, prefix_layer_count=2, attention_path="materialized_compressed",
        chunk_size=16, rotary_emb=rotary, position_ids=position_ids,
    )
    stream, _, _ = replay_prefix_layers(
        hidden, layers, prefix_layer_count=2, attention_path="streaming_compressed",
        chunk_size=16, rotary_emb=rotary, position_ids=position_ids,
    )
    max_err = float((mat - stream).abs().max().item())
    assert max_err < 5e-4


def test_full_block_parity_identical() -> None:
    x = torch.randn(1, 16, 64)
    info = check_full_block_parity(x, x.clone())
    assert info["full_block_parity_status"] == "passed"


def test_aggregate_layer_memory_consistent() -> None:
    records = [
        LayerMemoryRecord(0, 100, 60, 100, 40, 8, 16, 4),
        LayerMemoryRecord(1, 100, 60, 100, 30, 8, 16, 4),
    ]
    agg = aggregate_layer_memory(records)
    assert agg["aggregate_full_kv_bytes"] == 200
    assert agg["aggregate_streaming_peak_working_kv_bytes_conservative"] == 40
    assert agg["best_theoretical_streaming_reduction"] > 0


def test_causal_attention_no_future_leak() -> None:
    """Later query positions should not change when future keys are zeroed."""
    layer = _DummyDecoderLayer()
    hidden = torch.randn(1, 8, 64)
    position_ids = torch.arange(8).unsqueeze(0)
    out_full, _, _ = run_qwen_decoder_block(
        hidden, layer, layer_idx=0, attention_path="full",
        chunk_size=4, rotary_emb=None, position_ids=position_ids,
    )
    hidden2 = hidden.clone()
    hidden2[:, 4:, :] = 0.0
    out_trunc, _, _ = run_qwen_decoder_block(
        hidden2, layer, layer_idx=0, attention_path="full",
        chunk_size=4, rotary_emb=None, position_ids=position_ids,
    )
    # First 4 token outputs should be identical under causal attention
    assert torch.allclose(out_full[:, :4, :], out_trunc[:, :4, :], atol=1e-5)


def test_run_multilayer_cell_with_mock_model() -> None:
    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(5000, 64)
            self.model.layers = nn.ModuleList([_DummyDecoderLayer(), _DummyDecoderLayer()])
            self.model.rotary_emb = _MockRotaryEmb()

        def eval(self) -> "_MockModel":
            return self

    model = _MockModel()
    input_ids = torch.randint(0, 100, (1, 32))
    hidden0 = model.model.embed_tokens(input_ids)
    hf_hs = (hidden0,)
    h = hidden0
    for layer in model.model.layers:
        h, _, _ = run_qwen_decoder_block(
            h, layer, layer_idx=0, attention_path="full",
            chunk_size=16, rotary_emb=model.model.rotary_emb,
            position_ids=torch.arange(32).unsqueeze(0),
        )
        hf_hs = hf_hs + (h,)

    cell = run_multilayer_drift_cell(
        model=model,
        input_ids=input_ids,
        hf_hidden_states=hf_hs,
        prompt_id="p0",
        prompt_preview="test",
        target_token_length=32,
        actual_token_length=32,
        prefix_layer_count=2,
        chunk_size=16,
        allow_parity_fail=False,
    )
    assert cell["streaming_passed"] is True
    assert cell["full_block_parity_status"] == "passed"
    assert cell["passed"] is True
    assert cell["aggregate_memory_accounting"] is not None


def test_phase16d_regression_marker_exists() -> None:
    assert PHASE16D_REGRESSION_CELL["prompt_id"] == "long_128"
    assert PHASE16D_REGRESSION_CELL["prefix_layer_count"] == 4
    assert PHASE16D_REGRESSION_CELL["chunk_size"] == 32
    assert PHASE16D_REGRESSION_CELL["accumulator_mode"] == "default"


def test_run_exp069_mock_end_to_end() -> None:
    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(5000, 64)
            self.model.layers = nn.ModuleList([_DummyDecoderLayer(), _DummyDecoderLayer(), _DummyDecoderLayer()])
            self.model.rotary_emb = _MockRotaryEmb()

        def eval(self) -> "_MockModel":
            return self

        def to(self, device: str) -> "_MockModel":
            return self

        def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            b, t = input_ids.shape
            h0 = self.model.embed_tokens(input_ids)
            hs = [h0]
            h = h0
            pos = torch.arange(t).unsqueeze(0)
            for i, layer in enumerate(self.model.layers):
                h, _, _ = run_qwen_decoder_block(
                    h, layer, layer_idx=i, attention_path="full",
                    chunk_size=16, rotary_emb=self.model.rotary_emb, position_ids=pos,
                )
                hs.append(h)

            class _Out:
                hidden_states = tuple(hs)

            return _Out()

    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(range(1, 65))

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(f"t{i}" for i in token_ids)

        def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
            ids = self.encode(text)
            return {"input_ids": torch.tensor([ids])}

    def _loader(**kwargs: object) -> tuple[_MockModel, _FakeTokenizer]:
        return _MockModel(), _FakeTokenizer()

    report = run_exp069_probe(
        model_id="mock",
        target_token_lengths=(64,),
        prefix_layer_counts=(1, 2),
        chunk_sizes=(16,),
        max_prompts=1,
        model_loader=_loader,
    )
    assert report["model_load_succeeded"] is True
    assert report["successful_cells"] > 0
    assert report["status"] in ("pass", "failed")
