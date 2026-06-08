"""Smoke test for examples/qwen_smoke.py — Step 15.

Verifies that:
  * The script can be imported without side-effects.
  * main() runs to completion with a tiny token budget.
  * main() returns a valid result dict with the expected keys.
  * exactkv_matches_full is True.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

# Path to the example script
_EXAMPLE_PATH = Path(__file__).parent.parent / "examples" / "qwen_smoke.py"


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def qwen_smoke_module():
    """Import examples/qwen_smoke.py as a module without running __main__."""
    spec = importlib.util.spec_from_file_location("qwen_smoke", _EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_imports_cleanly(qwen_smoke_module) -> None:
    assert hasattr(qwen_smoke_module, "main"), "main() function not found in qwen_smoke.py"


def test_example_main_runs(runtime: ModelRuntime, qwen_smoke_module) -> None:
    result = qwen_smoke_module.main(
        prompt="The capital of France is",
        max_new_tokens=8,
        draft_len=2,
        compressor_name="int8",
        runtime=runtime,
    )
    assert isinstance(result, dict)


def test_example_main_result_keys(runtime: ModelRuntime, qwen_smoke_module) -> None:
    result = qwen_smoke_module.main(
        prompt="The capital of France is",
        max_new_tokens=8,
        draft_len=2,
        compressor_name="int8",
        runtime=runtime,
    )
    for key in (
        "prompt", "full_text", "lossy_text", "exactkv_text",
        "exactkv_matches_full", "lossy_matches_full", "acceptance_rate",
        "correction_count", "rejection_count",
        "first_lossy_divergence_idx", "compressor_name", "memory",
    ):
        assert key in result, f"Missing key {key!r} in main() result"


def test_example_main_exactkv_matches_full(runtime: ModelRuntime, qwen_smoke_module) -> None:
    result = qwen_smoke_module.main(
        prompt="The capital of France is",
        max_new_tokens=8,
        draft_len=2,
        compressor_name="int8",
        runtime=runtime,
    )
    assert result["exactkv_matches_full"] is True, (
        f"ExactKV did not match full in demo script: "
        f"full={result['full_text']!r} ekv={result['exactkv_text']!r}"
    )


def test_example_main_noop_matches_full(runtime: ModelRuntime, qwen_smoke_module) -> None:
    result = qwen_smoke_module.main(
        prompt="Hello",
        max_new_tokens=6,
        draft_len=2,
        compressor_name="noop",
        runtime=runtime,
    )
    assert result["exactkv_matches_full"] is True
