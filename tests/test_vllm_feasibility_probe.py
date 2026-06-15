"""Unit tests for vLLM feasibility probe helpers (no vLLM/CUDA required)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from exactkv.integrations.vllm_probe import (
    EXPERIMENT_059_ID,
    EXP059_CLAIM_NOTE,
    FORBIDDEN_CLAIMS,
    build_vllm_blocked_report,
    probe_vllm_availability,
    validate_exp059_report,
)

_PROBE_FILE = Path(__file__).resolve().parents[1] / "exactkv" / "integrations" / "vllm_probe.py"


def _base_report(**overrides: object) -> dict[str, object]:
    surfaces = {
        "model_loading_surface": "blocked",
        "generation_call_surface": "blocked",
        "sampling_greedy_config_surface": "blocked",
        "kv_cache_access_surface": "blocked",
        "scheduler_cache_api_surface": "blocked",
        "restored_full_kv_verifier_path": "blocked",
    }
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_059_ID,
        "status": "blocked",
        "python_executable": "/usr/bin/python3",
        "platform": "Linux",
        "torch_version": "2.8.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX A5000",
        "vllm_importable": False,
        "vllm_version": "",
        "import_error": "ModuleNotFoundError: No module named 'vllm'",
        "llm_class_importable": False,
        "sampling_params_importable": False,
        "generation_smoke_attempted": False,
        "generation_smoke_passed": False,
        "generation_smoke_error": "",
        "visible_integration_surfaces": surfaces,
        "kv_cache_access_status": "blocked — vLLM not importable or cache APIs not visible",
        "blockers": ["ModuleNotFoundError: No module named 'vllm'"],
        "claim_note": EXP059_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_vllm_missing_report_validates() -> None:
    assert validate_exp059_report(_base_report()) == []


def test_build_vllm_blocked_report_validates() -> None:
    report = build_vllm_blocked_report(import_error="ModuleNotFoundError: No module named 'vllm'")
    assert validate_exp059_report(report) == []
    assert report["status"] == "blocked"
    assert report["vllm_importable"] is False


def test_vllm_import_success_mocked_report_validates() -> None:
    mock_vllm = MagicMock()
    mock_vllm.__version__ = "0.6.0"
    mock_llm_instance = MagicMock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="four")]
    mock_llm_instance.generate.return_value = [mock_output]
    mock_vllm.LLM = MagicMock(return_value=mock_llm_instance)
    mock_vllm.SamplingParams = MagicMock()

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        with patch("importlib.import_module", return_value=mock_vllm):
            with patch.dict(sys.modules, {"vllm": mock_vllm}, clear=False):
                with patch(
                    "exactkv.integrations.vllm_probe._torch_environment",
                    return_value=("2.8.0", True, "GPU"),
                ):
                    result = probe_vllm_availability(run_generation_smoke=True)
    report = result.to_report_dict()
    assert report["vllm_importable"] is True
    assert report["llm_class_importable"] is True
    assert validate_exp059_report(report) == []


def test_generation_smoke_skipped_report_validates() -> None:
    report = _base_report(
        status="blocked",
        vllm_importable=False,
        generation_smoke_attempted=False,
    )
    assert validate_exp059_report(report) == []


def test_generation_smoke_failure_report_validates() -> None:
    report = _base_report(
        status="failed",
        vllm_importable=True,
        vllm_version="0.6.0",
        llm_class_importable=True,
        sampling_params_importable=True,
        generation_smoke_attempted=True,
        generation_smoke_passed=False,
        generation_smoke_error="RuntimeError: smoke failed",
        blockers=["generation_smoke: RuntimeError: smoke failed"],
        visible_integration_surfaces={
            "model_loading_surface": "accessible",
            "generation_call_surface": "accessible",
            "sampling_greedy_config_surface": "accessible",
            "kv_cache_access_surface": "unknown",
            "scheduler_cache_api_surface": "unknown",
            "restored_full_kv_verifier_path": "unknown",
        },
    )
    assert validate_exp059_report(report) == []


def test_no_module_level_vllm_import() -> None:
    tree = ast.parse(_PROBE_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "vllm"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "vllm"


def test_probe_blocked_when_vllm_missing() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        result = probe_vllm_availability(run_generation_smoke=False)
    assert result.status == "blocked"
    assert not result.vllm_importable
    assert result.generation_smoke_attempted is False
