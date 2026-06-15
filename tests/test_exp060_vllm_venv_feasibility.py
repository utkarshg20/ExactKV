"""Tests for Experiment 060 isolated vLLM venv feasibility (no vLLM/CUDA required)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from exactkv.integrations.vllm_probe import (
    EXPERIMENT_060_ID,
    EXP060_CLAIM_NOTE,
    FORBIDDEN_CLAIMS,
    PythonEnvMetadata,
    run_vllm_venv_feasibility,
    validate_exp060_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md"


def _base_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_060_ID,
        "status": "blocked",
        "system_python": "/usr/bin/python3",
        "venv_python": "/workspace/ExactKV/.venv-vllm/bin/python",
        "system_torch_version": "2.8.0+cu128",
        "venv_torch_version": "",
        "system_cuda_available": True,
        "venv_cuda_available": False,
        "gpu_name": "NVIDIA RTX A5000",
        "vllm_importable": False,
        "vllm_version": "",
        "install_attempted": False,
        "install_success": False,
        "import_error": "venv not created",
        "generation_smoke_attempted": False,
        "generation_smoke_passed": False,
        "generation_smoke_error": "",
        "stdout_tail": "",
        "stderr_tail": "",
        "blockers": ["venv python not found"],
        "claim_note": EXP060_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_missing_venv_report_validates() -> None:
    assert validate_exp060_report(_base_report()) == []


def test_install_blocked_report_validates() -> None:
    report = _base_report(
        install_attempted=True,
        install_success=False,
        import_error="ModuleNotFoundError: No module named 'vllm'",
        blockers=["ModuleNotFoundError: No module named 'vllm'"],
    )
    assert validate_exp060_report(report) == []


def test_vllm_import_success_mocked_report_validates() -> None:
    report = _base_report(
        status="pass",
        venv_torch_version="2.8.0+cu128",
        venv_cuda_available=True,
        vllm_importable=True,
        vllm_version="0.8.5",
        install_attempted=True,
        install_success=True,
        generation_smoke_attempted=True,
        generation_smoke_passed=True,
        blockers=[],
    )
    assert validate_exp060_report(report) == []


def test_generation_smoke_success_mocked_report_validates() -> None:
    report = _base_report(
        status="pass",
        venv_torch_version="2.8.0+cu128",
        venv_cuda_available=True,
        vllm_importable=True,
        install_success=True,
        install_attempted=True,
        generation_smoke_attempted=True,
        generation_smoke_passed=True,
        blockers=[],
    )
    assert validate_exp060_report(report) == []


def test_generation_smoke_failure_mocked_report_validates() -> None:
    report = _base_report(
        status="failed",
        venv_torch_version="2.8.0+cu128",
        venv_cuda_available=True,
        vllm_importable=True,
        install_attempted=True,
        install_success=True,
        generation_smoke_attempted=True,
        generation_smoke_passed=False,
        generation_smoke_error="RuntimeError: smoke failed",
        blockers=["generation_smoke: RuntimeError: smoke failed"],
    )
    assert validate_exp060_report(report) == []


def test_run_blocked_when_venv_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "bin" / "python"
    with patch(
        "exactkv.integrations.vllm_probe.collect_python_env_metadata",
        return_value=PythonEnvMetadata(
            python_executable="/usr/bin/python3",
            python_version="3.12.0",
            torch_version="2.8.0+cu128",
            cuda_available=True,
            gpu_name="GPU",
        ),
    ):
        result = run_vllm_venv_feasibility(venv_python=missing)
    assert result.status == "blocked"
    assert not result.install_attempted
    assert validate_exp060_report(result.to_report_dict()) == []


def test_run_pass_mocked(tmp_path: Path) -> None:
    venv_py = tmp_path / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_py.chmod(0o755)

    system_meta = PythonEnvMetadata(
        python_executable="/usr/bin/python3",
        python_version="3.12.0",
        torch_version="2.8.0+cu128",
        cuda_available=True,
        gpu_name="GPU",
    )
    venv_meta = PythonEnvMetadata(
        python_executable="/venv/bin/python",
        python_version="3.12.0",
        torch_version="2.8.0+cu128",
        cuda_available=True,
        gpu_name="GPU",
    )
    probe = {
        "vllm_importable": True,
        "llm_class_importable": True,
        "sampling_params_importable": True,
        "vllm_version": "0.8.5",
        "import_error": "",
        "generation_smoke_attempted": True,
        "generation_smoke_passed": True,
        "generation_smoke_error": "",
        "blockers": [],
    }
    with patch("exactkv.integrations.vllm_probe.collect_python_env_metadata", side_effect=[system_meta, venv_meta]):
        with patch("exactkv.integrations.vllm_probe.probe_vllm_in_subprocess", return_value=(probe, "ok", "")):
            result = run_vllm_venv_feasibility(venv_python=venv_py)
    assert result.status == "pass"
    assert result.install_success


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "isolated vllm environment feasibility",
        "not vllm integration",
        "not installed into system python",
        "default exactkv",
        "throughput",
        "vericache",
        "future integration work",
    ):
        assert phrase in text, phrase


def test_doc_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("exactkv is integrated with vllm", "throughput improved", "production serving works"):
        assert phrase not in text
