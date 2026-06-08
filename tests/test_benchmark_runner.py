"""Benchmark runner tests — Step 14.

Tests:
  * Prompt loader loads smoke.jsonl (>= 10 prompts, required fields present)
  * run_one emits a valid JSON-compatible dict
  * Report dict contains full, lossy, exactkv, memory, acceptance keys
  * exactkv_vs_full token exact match is True
  * aggregate exactkv_failures == 0 (2-prompt subset)
  * Works with INT8 compressor
"""
from __future__ import annotations

import json

import pytest

from exactkv.benchmarks.prompts import load_smoke_prompts
from exactkv.benchmarks.runner import RunConfig, run_one, run_suite
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

# Single deterministic prompt for fast structural checks
_TEST_PROMPT = {
    "prompt_id": "test_001",
    "category": "test",
    "prompt": "The capital of France is",
}


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def noop_config() -> RunConfig:
    return RunConfig(compressor_name="noop", draft_len=4, max_new_tokens=12)


@pytest.fixture(scope="module")
def int8_config() -> RunConfig:
    return RunConfig(compressor_name="int8", draft_len=4, max_new_tokens=12)


# ---------------------------------------------------------------------------
# Prompt loader (model-free)
# ---------------------------------------------------------------------------

def test_smoke_prompts_load() -> None:
    prompts = load_smoke_prompts()
    assert len(prompts) >= 10, f"Expected >= 10 prompts, got {len(prompts)}"


def test_smoke_prompts_required_fields() -> None:
    for p in load_smoke_prompts():
        for field in ("prompt_id", "category", "prompt"):
            assert field in p, f"Missing field {field!r} in {p}"


def test_smoke_prompts_unique_ids() -> None:
    prompts = load_smoke_prompts()
    ids = [p["prompt_id"] for p in prompts]
    assert len(ids) == len(set(ids)), "Duplicate prompt_ids in smoke.jsonl"


# ---------------------------------------------------------------------------
# run_one — structure checks
# ---------------------------------------------------------------------------

def test_run_one_emits_dict(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    assert isinstance(report, dict)


def test_run_one_top_level_keys(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    for key in ("prompt_id", "prompt", "category", "model_name",
                "compressor_name", "draft_len", "max_new_tokens",
                "full", "lossy", "exactkv", "memory", "exactkv_failure"):
        assert key in report, f"Missing key {key!r} in report"


def test_run_one_full_keys(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    assert "output_ids" in report["full"]
    assert "output_text" in report["full"]


def test_run_one_lossy_keys(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    for key in ("output_ids", "output_text", "token_exact_match", "first_divergence_idx"):
        assert key in report["lossy"], f"Missing key {key!r} in report['lossy']"


def test_run_one_exactkv_keys(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    for key in ("output_ids", "output_text", "token_exact_match", "acceptance"):
        assert key in report["exactkv"], f"Missing key {key!r} in report['exactkv']"


def test_run_one_memory_keys(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    for key in ("full_bytes", "compressed_bytes", "compression_ratio"):
        assert key in report["memory"], f"Missing key {key!r} in report['memory']"


def test_run_one_acceptance_keys(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    acc = report["exactkv"]["acceptance"]
    for key in ("total_drafted", "total_accepted", "total_rejected", "acceptance_rate"):
        assert key in acc, f"Missing key {key!r} in acceptance dict"


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def test_noop_exactkv_exact_match(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    assert report["exactkv"]["token_exact_match"] is True
    assert report["exactkv_failure"] is False


def test_int8_exactkv_exact_match(runtime: ModelRuntime, int8_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, int8_config)
    assert report["exactkv"]["token_exact_match"] is True
    assert report["exactkv_failure"] is False


def test_run_one_json_serializable(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    report = run_one(runtime, _TEST_PROMPT, noop_config)
    dumped = json.dumps(report)  # should not raise
    assert isinstance(dumped, str)


# ---------------------------------------------------------------------------
# run_suite aggregate
# ---------------------------------------------------------------------------

def test_aggregate_exactkv_failures_zero(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    """Run 2-prompt subset; no ExactKV failures expected with NoOp."""
    mini_suite = load_smoke_prompts()[:2]
    suite_report = run_suite(runtime, mini_suite, noop_config)
    assert suite_report["aggregate"]["exactkv_failures"] == 0, (
        f"ExactKV failures: {suite_report['aggregate']['exactkv_failures']}"
    )


def test_suite_aggregate_keys(runtime: ModelRuntime, noop_config: RunConfig) -> None:
    mini_suite = load_smoke_prompts()[:2]
    suite_report = run_suite(runtime, mini_suite, noop_config)
    agg = suite_report["aggregate"]
    for key in ("total_prompts", "exactkv_failures", "exactkv_pass_rate"):
        assert key in agg, f"Missing key {key!r} in aggregate"


def test_int8_aggregate_failures_zero(runtime: ModelRuntime, int8_config: RunConfig) -> None:
    """INT8 ExactKV must also produce zero failures."""
    mini_suite = load_smoke_prompts()[:2]
    suite_report = run_suite(runtime, mini_suite, int8_config)
    assert suite_report["aggregate"]["exactkv_failures"] == 0
