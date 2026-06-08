"""Tests for the unified config (ExactKVConfig and BenchmarkConfig).

Verifies:
  * ExactKVConfig still works unchanged (backward compat)
  * BenchmarkConfig validates inputs correctly
  * BenchmarkConfig.to_exactkv_config produces a valid ExactKVConfig
  * Invalid BenchmarkConfig raises ValueError
"""
from __future__ import annotations

import pytest


def test_exactkv_config_backward_compat():
    """ExactKVConfig must still construct with the same args as before."""
    from exactkv.config import ExactKVConfig

    cfg = ExactKVConfig(model_name="Qwen/Qwen2.5-0.5B")
    assert cfg.model_name == "Qwen/Qwen2.5-0.5B"
    assert cfg.compressor == "noop"
    assert cfg.draft_len == 8
    assert cfg.max_new_tokens == 128
    assert cfg.greedy is True


def test_exactkv_config_rejects_non_greedy():
    from exactkv.config import ExactKVConfig

    with pytest.raises(ValueError, match="greedy"):
        ExactKVConfig(model_name="Qwen/Qwen2.5-0.5B", greedy=False)


def test_exactkv_config_rejects_zero_draft_len():
    from exactkv.config import ExactKVConfig

    with pytest.raises(ValueError, match="draft_len"):
        ExactKVConfig(model_name="Qwen/Qwen2.5-0.5B", draft_len=0)


def test_benchmark_config_defaults():
    from exactkv.config import BenchmarkConfig

    cfg = BenchmarkConfig(model_name="Qwen/Qwen2.5-0.5B")
    assert cfg.compressors == ["int8"]
    assert cfg.draft_lens == [4]
    assert cfg.max_new_tokens == 32
    assert cfg.prompt_suite == "smoke"
    assert cfg.output_dir == "reports"
    assert cfg.report_formats == ["json"]


def test_benchmark_config_custom_values():
    from exactkv.config import BenchmarkConfig

    cfg = BenchmarkConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        compressors=["noop", "int8"],
        draft_lens=[2, 4, 8],
        max_new_tokens=64,
        prompt_suite="core",
    )
    assert "noop" in cfg.compressors
    assert "int8" in cfg.compressors
    assert cfg.draft_lens == [2, 4, 8]


def test_benchmark_config_rejects_empty_compressors():
    from exactkv.config import BenchmarkConfig

    with pytest.raises(ValueError, match="compressors"):
        BenchmarkConfig(model_name="Qwen/Qwen2.5-0.5B", compressors=[])


def test_benchmark_config_rejects_empty_draft_lens():
    from exactkv.config import BenchmarkConfig

    with pytest.raises(ValueError, match="draft_lens"):
        BenchmarkConfig(model_name="Qwen/Qwen2.5-0.5B", draft_lens=[])


def test_benchmark_config_rejects_zero_draft_len():
    from exactkv.config import BenchmarkConfig

    with pytest.raises(ValueError, match="draft_lens"):
        BenchmarkConfig(model_name="Qwen/Qwen2.5-0.5B", draft_lens=[0])


def test_benchmark_config_rejects_zero_max_new_tokens():
    from exactkv.config import BenchmarkConfig

    with pytest.raises(ValueError, match="max_new_tokens"):
        BenchmarkConfig(model_name="Qwen/Qwen2.5-0.5B", max_new_tokens=0)


def test_benchmark_config_rejects_invalid_report_format():
    from exactkv.config import BenchmarkConfig

    with pytest.raises(ValueError, match="report_formats"):
        BenchmarkConfig(model_name="Qwen/Qwen2.5-0.5B", report_formats=["toml"])


def test_benchmark_config_accepts_csv():
    from exactkv.config import BenchmarkConfig

    cfg = BenchmarkConfig(model_name="Qwen/Qwen2.5-0.5B", report_formats=["csv"])
    assert "csv" in cfg.report_formats


def test_benchmark_config_accepts_json_and_csv():
    from exactkv.config import BenchmarkConfig

    cfg = BenchmarkConfig(
        model_name="Qwen/Qwen2.5-0.5B", report_formats=["json", "csv"]
    )
    assert set(cfg.report_formats) == {"json", "csv"}


def test_to_exactkv_config_produces_valid_config():
    from exactkv.config import BenchmarkConfig, ExactKVConfig

    bench_cfg = BenchmarkConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        compressors=["int8"],
        draft_lens=[4],
        max_new_tokens=64,
        dtype="float32",
    )
    ekv_cfg = bench_cfg.to_exactkv_config(compressor="int8", draft_len=4)

    assert isinstance(ekv_cfg, ExactKVConfig)
    assert ekv_cfg.model_name == "Qwen/Qwen2.5-0.5B"
    assert ekv_cfg.compressor == "int8"
    assert ekv_cfg.draft_len == 4
    assert ekv_cfg.max_new_tokens == 64
    assert ekv_cfg.dtype == "float32"
    assert ekv_cfg.greedy is True   # always True — not sampling


def test_to_exactkv_config_inherits_device_and_seed():
    from exactkv.config import BenchmarkConfig

    bench_cfg = BenchmarkConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu",
        seed=42,
    )
    ekv_cfg = bench_cfg.to_exactkv_config(compressor="noop", draft_len=2)
    assert ekv_cfg.device == "cpu"
    assert ekv_cfg.seed == 42
