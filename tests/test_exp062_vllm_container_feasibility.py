"""Tests for Experiment 062 vLLM container feasibility (no vLLM/CUDA/network)."""
from __future__ import annotations

from pathlib import Path

from exactkv.integrations.vllm_probe import (
    EXPERIMENT_062_ID,
    EXP062_CLAIM_NOTE,
    FORBIDDEN_CLAIMS,
    VllmProbeResult,
    build_exp062_report_from_probe,
    validate_exp062_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md"

_SURFACES = {
    "model_loading_surface": "blocked",
    "generation_call_surface": "blocked",
    "sampling_greedy_config_surface": "blocked",
    "kv_cache_access_surface": "blocked",
    "scheduler_cache_api_surface": "blocked",
    "restored_full_kv_verifier_path": "blocked",
}


def _probe_result(**overrides: object) -> VllmProbeResult:
    base: dict[str, object] = {
        "status": "blocked",
        "python_executable": "/usr/bin/python3",
        "platform_info": "Linux",
        "torch_version": "2.8.0+cu128",
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
        "generation_smoke_text": "",
        "visible_integration_surfaces": dict(_SURFACES),
        "kv_cache_access_status": "blocked — vLLM not importable or cache APIs not visible",
        "blockers": ["ModuleNotFoundError: No module named 'vllm'"],
    }
    base.update(overrides)
    return VllmProbeResult(**base)  # type: ignore[arg-type]


def _report(**overrides: object) -> dict[str, object]:
    report = build_exp062_report_from_probe(
        _probe_result(**{k: v for k, v in overrides.items() if k in VllmProbeResult.__dataclass_fields__}),
        environment_label="test-env",
        python_version="3.12.0",
    )
    report.update({k: v for k, v in overrides.items() if k not in VllmProbeResult.__dataclass_fields__})
    return report


def test_blocked_report_validates() -> None:
    assert validate_exp062_report(_report()) == []


def test_import_pass_generation_skipped_report_validates() -> None:
    report = _report(
        status="pass",
        vllm_importable=True,
        vllm_version="0.23.0",
        import_error="",
        llm_class_importable=True,
        sampling_params_importable=True,
        visible_integration_surfaces={
            "model_loading_surface": "accessible",
            "generation_call_surface": "accessible",
            "sampling_greedy_config_surface": "accessible",
            "kv_cache_access_surface": "accessible",
            "scheduler_cache_api_surface": "accessible",
            "restored_full_kv_verifier_path": "unknown",
        },
        kv_cache_access_status="partial — vLLM cache-related symbols detected; export path not implemented",
        blockers=[],
    )
    report["generation_smoke_attempted"] = False
    report["generation_smoke_passed"] = False
    assert validate_exp062_report(report) == []


def test_generation_pass_report_validates() -> None:
    report = _report(
        status="pass",
        vllm_importable=True,
        vllm_version="0.23.0",
        import_error="",
        llm_class_importable=True,
        sampling_params_importable=True,
        generation_smoke_attempted=True,
        generation_smoke_passed=True,
        generation_smoke_text="four",
        visible_integration_surfaces={
            "model_loading_surface": "accessible",
            "generation_call_surface": "accessible",
            "sampling_greedy_config_surface": "accessible",
            "kv_cache_access_surface": "accessible",
            "scheduler_cache_api_surface": "accessible",
            "restored_full_kv_verifier_path": "unknown",
        },
        kv_cache_access_status="partial — vLLM cache-related symbols detected; export path not implemented",
        blockers=[],
    )
    report["generated_text_preview"] = "four"
    assert validate_exp062_report(report) == []


def test_generation_fail_report_validates() -> None:
    report = _report(
        status="failed",
        vllm_importable=True,
        vllm_version="0.23.0",
        llm_class_importable=True,
        sampling_params_importable=True,
        generation_smoke_attempted=True,
        generation_smoke_passed=False,
        generation_smoke_error="RuntimeError: smoke failed",
        blockers=["generation_smoke: RuntimeError: smoke failed"],
    )
    assert validate_exp062_report(report) == []


def test_integration_surface_schema_validates() -> None:
    report = _report()
    surfaces = report["visible_integration_surfaces"]
    assert isinstance(surfaces, dict)
    for key in (
        "model_loading_surface",
        "generation_call_surface",
        "sampling_greedy_config_surface",
        "kv_cache_access_surface",
        "scheduler_cache_api_surface",
        "restored_full_kv_verifier_path",
    ):
        assert key in surfaces


def test_blocker_schema_validates() -> None:
    report = _report(blockers=["import failed"])
    assert isinstance(report["blockers"], list)
    assert validate_exp062_report(report) == []


def test_claim_flags_remain_false() -> None:
    report = _report()
    assert report["claim_note"] == EXP062_CLAIM_NOTE
    assert list(report["forbidden_claims"]) == list(FORBIDDEN_CLAIMS)
    assert report["experiment_id"] == EXPERIMENT_062_ID


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "feasibility probe",
        "not exactkv",
        "not vllm integration",
        "default runtime",
        "throughput",
        "vericache",
    ):
        assert phrase in text, phrase


def test_doc_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("exactkv is integrated with vllm", "throughput improved", "production serving works"):
        assert phrase not in text
