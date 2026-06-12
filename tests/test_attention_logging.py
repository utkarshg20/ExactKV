"""Tests for attention logging helpers (Experiment 026). No model downloads."""
from __future__ import annotations

import json

import pytest
import torch

from exactkv.analysis.attention_logging import (
    FORBIDDEN_ATTENTION_FIELDS,
    assert_attention_artifact_safe,
    evaluate_feasibility,
    load_exp026_prompt_subset,
    summarize_last_token_attention,
)


def test_load_exp026_prompt_count():
    prompts = load_exp026_prompt_subset()
    assert len(prompts) == 6
    suites = {p["v10_suite"] for p in prompts}
    assert suites == {"long_context", "retrieval_copy", "tool_json"}


def test_summarize_last_token_attention_4d():
    # batch=1, heads=2, seq=8, seq=8 — peaked on last key
    attn = torch.zeros(1, 2, 8, 8)
    attn[:, :, -1, -1] = 1.0
    s = summarize_last_token_attention(attn, recent_k=2, early_k=2)
    assert s["seq_len"] == 8
    assert s["mass_to_recent_tokens"] == pytest.approx(1.0)
    assert s["mass_to_early_tokens"] == pytest.approx(0.0)
    assert s["entropy"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_uniform_entropy():
    attn = torch.ones(1, 4, 6, 6) / 6.0
    s = summarize_last_token_attention(attn)
    assert s["entropy"] is not None
    assert s["entropy"] > 1.5


def test_evaluate_feasibility_no_go():
    v = evaluate_feasibility([
        {"weights_obtained": False, "model_name": "Qwen/Qwen2.5-0.5B"},
    ])
    assert v["recommendation"] == "no_go"
    assert v["any_weights_obtained"] is False


def test_evaluate_feasibility_restricted_go_prefill():
    v = evaluate_feasibility([
        {
            "weights_obtained": True,
            "phase": "prefill",
            "model_name": "Qwen/Qwen2.5-0.5B",
        },
    ])
    assert v["recommendation"] == "restricted_go_prefill_only"
    assert v["qwen_prefill_weights"] is True


def test_assert_attention_artifact_safe():
    with pytest.raises(ValueError, match="throughput"):
        assert_attention_artifact_safe({"throughput": 1.0})
    assert_attention_artifact_safe({"verdict": {"recommendation": "no_go"}})


def test_forbidden_fields_not_in_summary_blob():
    sample = {"verdict": {"any_weights_obtained": False}}
    blob = json.dumps(sample)
    for field in FORBIDDEN_ATTENTION_FIELDS:
        assert f'"{field}"' not in blob
    assert_attention_artifact_safe(sample)


def test_no_fabricated_default_weights():
    """Failed attempts must not claim weights_obtained."""
    v = evaluate_feasibility([{"weights_obtained": False}])
    assert v["any_weights_obtained"] is False
