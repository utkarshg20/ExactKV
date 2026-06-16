"""Tests for offline full-prefix logit drift smoke (Phase 16F)."""
from __future__ import annotations

import torch
import torch.nn as nn

from exactkv.attention.hf_full_replay_probe import (
    EXPERIMENT_071_ID,
    FORBIDDEN_ATTENTION_CLAIMS,
    aggregate_full_stack_memory,
    check_full_model_parity,
    compute_logit_drift_metrics,
    replay_full_decoder_stack,
    run_exp071_logit_cell,
    run_exp071_probe,
    validate_exp071_report,
)
from exactkv.attention.hf_multilayer_probe import LayerMemoryRecord
from tests.test_hf_multilayer_probe import (
    _DummyDecoderLayer,
    _MockRotaryEmb,
)


class _DummyModel(nn.Module):
    def __init__(self, *, hidden: int = 64, vocab: int = 128, depth: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList(
            [_DummyDecoderLayer(hidden=hidden) for _ in range(depth)]
        )
        self.model.norm = nn.LayerNorm(hidden)
        self.model.rotary_emb = _MockRotaryEmb()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def eval(self) -> "_DummyModel":
        return self

    def to(self, device: str) -> "_DummyModel":
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> object:
        del use_cache
        b, t = input_ids.shape
        h = self.model.embed_tokens(input_ids)
        hs = [h]
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        for i, layer in enumerate(self.model.layers):
            from exactkv.attention.hf_multilayer_probe import run_qwen_decoder_block

            h, _, _ = run_qwen_decoder_block(
                h,
                layer,
                layer_idx=i,
                attention_path="full",
                chunk_size=16,
                rotary_emb=self.model.rotary_emb,
                position_ids=pos,
            )
            hs.append(h)
        normed = self.model.norm(h)
        if output_hidden_states:
            hs[-1] = normed
        logits = self.lm_head(normed)

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.hidden_states = tuple(hs) if output_hidden_states else None
        return out


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        length = 64 if "64" in text else 32
        return list(range(1, length + 1))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{i}" for i in token_ids)

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([self.encode(text)])}


def _mock_prompts(
    tokenizer: _FakeTokenizer,
    lengths: tuple[int, ...],
    max_prompts: int,
) -> list[tuple[str, str, int, int]]:
    del tokenizer
    return [
        (f"long_{target}", f"mock prompt target {target}", target, target)
        for target in lengths[:max_prompts]
    ]


def _mock_loader(**kwargs: object) -> tuple[_DummyModel, _FakeTokenizer]:
    return _DummyModel(depth=2), _FakeTokenizer()


def test_replay_full_decoder_stack_shapes() -> None:
    model = _DummyModel(depth=2)
    input_ids = torch.randint(0, 100, (1, 32))
    hidden, logits, mem = replay_full_decoder_stack(
        model, input_ids, attention_path="full", chunk_size=16,
    )
    assert hidden.shape == (1, 32, 64)
    assert logits.shape == (1, 128)
    assert len(mem) == 2


def test_compressed_paths_shapes() -> None:
    model = _DummyModel(depth=2)
    input_ids = torch.randint(0, 100, (1, 48))
    for path in ("materialized_compressed", "streaming_compressed"):
        hidden, logits, mem = replay_full_decoder_stack(
            model, input_ids, attention_path=path, chunk_size=16,  # type: ignore[arg-type]
        )
        assert hidden.shape == (1, 48, 64)
        assert logits.shape == (1, 128)
        assert len(mem) == 2


def test_compute_logit_drift_metrics() -> None:
    ref = torch.randn(1, 50)
    other = ref + 1e-4
    m = compute_logit_drift_metrics(ref, other)
    assert m.top1_agreement is True
    assert m.top5_overlap >= 4
    assert m.top1_changed is False
    d = m.to_dict()
    assert "max_abs_error" in d
    assert "top10_overlap" in d


def test_full_model_parity_identical() -> None:
    logits = torch.randn(1, 64)
    hidden = torch.randn(1, 16, 64)
    info = check_full_model_parity(logits, logits.clone(), hidden, hidden.clone())
    assert info["full_model_parity_status"] == "passed"


def test_aggregate_full_stack_memory() -> None:
    records = [
        LayerMemoryRecord(0, 100, 60, 100, 40, 8, 16, 4),
        LayerMemoryRecord(1, 100, 60, 100, 30, 8, 16, 4),
    ]
    agg = aggregate_full_stack_memory(records, context_length=64)
    assert agg["aggregate_full_kv_bytes"] == 200
    assert agg["num_layers"] == 2
    assert agg["context_length"] == 64
    assert agg["best_theoretical_streaming_reduction"] > 0


def test_streaming_matches_materialized_logit_cell() -> None:
    model = _DummyModel(depth=2)
    input_ids = torch.randint(0, 100, (1, 64))
    hf_out = model(input_ids, output_hidden_states=True)
    cell = run_exp071_logit_cell(
        model=model,
        input_ids=input_ids,
        hf_logits=hf_out.logits[:, -1, :],
        hf_hidden=hf_out.hidden_states[-1],
        prompt_id="p0",
        target_token_length=64,
        actual_token_length=64,
        chunk_size=16,
    )
    assert cell["streaming_passed"] is True
    assert cell["full_model_parity_status"] == "passed"
    assert cell["passed"] is True
    assert cell["top1_changed_full_vs_streaming"] is False


def test_top1_change_recorded_not_streaming_failure() -> None:
    """Top-1 mismatch vs full is recorded but streaming-vs-materialized can still pass."""
    ref = torch.zeros(1, 10)
    ref[0, 3] = 10.0
    stream = ref.clone()
    mat = ref.clone()
    m_fs = compute_logit_drift_metrics(ref, stream + 1e-3)
    m_sm = compute_logit_drift_metrics(mat, stream)
    assert m_sm.top1_agreement is True
    assert m_fs.top1_changed is False


def test_validate_exp071_report_pass() -> None:
    report = {
        "experiment_id": EXPERIMENT_071_ID,
        "status": "pass",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "target_token_lengths": [64],
        "chunk_sizes": [16],
        "total_cells": 1,
        "successful_cells": 1,
        "blocked_cells": 0,
        "full_model_parity_pass_cells": 1,
        "streaming_vs_materialized_pass_cells": 1,
        "compressed_top1_changed_cells": 0,
        "max_streaming_vs_materialized_logit_error": 1e-6,
        "max_streaming_vs_materialized_hidden_error": 1e-6,
        "full_vs_streaming_logit_drift_summary": {"max_abs_error": 1e-3, "mean_abs_error": 1e-4, "cell_count": 1},
        "top1_change_summary": {"cells_with_top1_change_full_vs_streaming": 0, "cell_count": 1},
        "topk_overlap_summary": {
            "streaming_vs_materialized_top5_mean": 5.0,
            "full_vs_streaming_top5_mean": 5.0,
            "streaming_vs_materialized_top10_mean": 10.0,
        },
        "memory_accounting_summary": {"best_theoretical_streaming_reduction": 0.5, "worst_theoretical_streaming_reduction": 0.5},
        "longest_context_tested": 64,
        "num_layers_replayed": 2,
        "limitations": ["test"],
        "no_performance_claims_note": "no performance claims",
        "claim_note": "test",
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "cells": [{
            "prompt_id": "p0",
            "target_token_length": 64,
            "actual_token_length": 64,
            "chunk_size": 16,
            "full_model_parity_status": "passed",
            "passed": True,
            "blockers": [],
        }],
    }
    assert validate_exp071_report(report) == []


def test_validate_exp071_parity_fail_report() -> None:
    report = {
        "experiment_id": EXPERIMENT_071_ID,
        "status": "failed",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "target_token_lengths": [64],
        "chunk_sizes": [16],
        "total_cells": 1,
        "successful_cells": 1,
        "blocked_cells": 0,
        "full_model_parity_pass_cells": 0,
        "streaming_vs_materialized_pass_cells": 1,
        "compressed_top1_changed_cells": 0,
        "max_streaming_vs_materialized_logit_error": 1e-6,
        "max_streaming_vs_materialized_hidden_error": 1e-6,
        "full_vs_streaming_logit_drift_summary": {"max_abs_error": 1.0, "mean_abs_error": 0.1, "cell_count": 1},
        "top1_change_summary": {"cells_with_top1_change_full_vs_streaming": 0, "cell_count": 1},
        "topk_overlap_summary": {
            "streaming_vs_materialized_top5_mean": 5.0,
            "full_vs_streaming_top5_mean": 3.0,
            "streaming_vs_materialized_top10_mean": 10.0,
        },
        "memory_accounting_summary": {"best_theoretical_streaming_reduction": 0.5, "worst_theoretical_streaming_reduction": 0.5},
        "longest_context_tested": 64,
        "num_layers_replayed": 2,
        "limitations": ["test"],
        "no_performance_claims_note": "no performance claims",
        "claim_note": "test",
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "cells": [{
            "prompt_id": "p0",
            "target_token_length": 64,
            "actual_token_length": 64,
            "chunk_size": 16,
            "full_model_parity_status": "failed",
            "passed": False,
            "blockers": ["parity failed"],
        }],
    }
    assert validate_exp071_report(report) == []


def test_no_forbidden_claim_fields() -> None:
    report = {
        "experiment_id": EXPERIMENT_071_ID,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
    }
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


def test_run_exp071_mock_end_to_end() -> None:
    report = run_exp071_probe(
        model_id="mock",
        target_token_lengths=(32, 64),
        chunk_sizes=(16, 32),
        max_prompts=2,
        model_loader=_mock_loader,
        prompt_provider=_mock_prompts,
    )
    assert report["model_load_succeeded"] is True
    assert report["successful_cells"] > 0
    assert report["num_layers_replayed"] == 2
    assert validate_exp071_report(report) == []
