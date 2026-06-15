"""Tests for Experiment 061 vLLM version sweep (no vLLM/CUDA/network required)."""
from __future__ import annotations

from pathlib import Path

from exactkv.integrations.vllm_probe import (
    EXPERIMENT_061_ID,
    EXP061_CLAIM_NOTE,
    FORBIDDEN_CLAIMS,
    PythonEnvMetadata,
    build_exp061_report,
    build_candidate_result_from_probe,
    build_install_failed_candidate,
    classify_candidate_result,
    validate_exp061_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_061_VLLM_VERSION_SWEEP.md"


def _candidate(
    *,
    version: str = "0.22.0",
    classification: str,
    install_success: bool = True,
    import_success: bool = False,
    llm: bool = False,
    sampling: bool = False,
    cuda: bool = True,
    gen_attempted: bool = False,
    gen_passed: bool = False,
    error: str = "error",
) -> dict[str, object]:
    return {
        "version": version,
        "venv_path": f"/workspace/ExactKV/.venv-vllm-sweep/vllm-{version.replace('.', '_')}",
        "python_version": "3.12.0",
        "install_success": install_success,
        "import_success": import_success,
        "llm_class_importable": llm,
        "sampling_params_importable": sampling,
        "venv_torch_version": "2.8.0+cu128",
        "venv_cuda_available": cuda,
        "vllm_version": version if install_success else "",
        "generation_smoke_attempted": gen_attempted,
        "generation_smoke_passed": gen_passed,
        "generation_smoke_text": "four" if gen_passed else "",
        "classification": classification,
        "error_summary": error,
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _report(*, candidates: list[dict[str, object]], **overrides: object) -> dict[str, object]:
    base = build_exp061_report(
        candidate_results=candidates,
        candidates=[str(c["version"]) for c in candidates],
        excluded_versions=["0.23.0"],
    )
    base.update(overrides)
    return base


def test_candidate_result_schema_helpers() -> None:
    install = build_install_failed_candidate(
        version="0.22.0",
        venv_path=Path("/tmp/v"),
        install_error="pip failed",
    )
    assert install["classification"] == "install_failed"
    probe = build_candidate_result_from_probe(
        version="0.21.0",
        venv_path=Path("/tmp/v"),
        install_success=True,
        env_meta=PythonEnvMetadata(
            python_executable="/tmp/v/bin/python",
            python_version="3.12.0",
            torch_version="2.8.0+cu128",
            cuda_available=True,
            gpu_name="GPU",
        ),
        probe_payload={
            "vllm_importable": True,
            "llm_class_importable": True,
            "sampling_params_importable": True,
            "generation_smoke_attempted": True,
            "generation_smoke_passed": True,
            "generation_smoke_text": "ok",
            "vllm_version": "0.21.0",
        },
    )
    assert probe["classification"] == "pass"
    assert classify_candidate_result(probe) == "pass"


def test_install_failure_candidate_validates() -> None:
    report = _report(
        candidates=[_candidate(classification="install_failed", install_success=False, error="pip failed")],
    )
    assert validate_exp061_report(report) == []


def test_import_failure_candidate_validates() -> None:
    report = _report(
        candidates=[
            _candidate(
                classification="import_failed",
                import_success=False,
                error="libcudart.so.13",
            )
        ],
    )
    assert validate_exp061_report(report) == []


def test_cuda_failure_candidate_validates() -> None:
    report = _report(
        candidates=[
            _candidate(
                classification="cuda_failed",
                cuda=False,
                error="CUDA unavailable",
            )
        ],
    )
    assert validate_exp061_report(report) == []


def test_generation_failure_candidate_validates() -> None:
    report = _report(
        candidates=[
            _candidate(
                classification="generation_failed",
                import_success=True,
                llm=True,
                sampling=True,
                gen_attempted=True,
                gen_passed=False,
                error="RuntimeError: smoke failed",
            )
        ],
    )
    assert validate_exp061_report(report) == []


def test_passing_candidate_validates() -> None:
    report = _report(
        candidates=[
            _candidate(
                classification="pass",
                import_success=True,
                llm=True,
                sampling=True,
                gen_attempted=True,
                gen_passed=True,
                error="",
            )
        ],
        any_candidate_passed=True,
        winning_candidate="0.22.0",
        generation_smoke_passed=True,
        blockers=[],
    )
    assert validate_exp061_report(report) == []


def test_no_candidate_pass_report_validates() -> None:
    report = _report(
        candidates=[
            _candidate(classification="import_failed", error="libcudart.so.13"),
            _candidate(version="0.21.0", classification="import_failed", error="libcudart.so.13"),
        ],
        any_candidate_passed=False,
        winning_candidate=None,
        generation_smoke_passed=False,
    )
    assert validate_exp061_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_061_ID
    assert report["recommended_next_step"].startswith("Separate environment phase")


def test_winning_candidate_report_validates() -> None:
    report = _report(
        candidates=[
            _candidate(classification="import_failed", error="bad"),
            _candidate(
                version="0.18.0",
                classification="pass",
                import_success=True,
                llm=True,
                sampling=True,
                gen_attempted=True,
                gen_passed=True,
                error="",
            ),
        ],
        any_candidate_passed=True,
        winning_candidate="0.18.0",
        generation_smoke_passed=True,
        blockers=[],
    )
    assert validate_exp061_report(report) == []
    assert report["winning_candidate"] == "0.18.0"


def test_claim_flags_remain_false_in_report() -> None:
    report = _report(
        candidates=[_candidate(classification="pass", import_success=True, llm=True, sampling=True, gen_passed=True, gen_attempted=True, error="")],
        any_candidate_passed=True,
        winning_candidate="0.22.0",
        generation_smoke_passed=True,
        blockers=[],
    )
    assert report["claim_note"] == EXP061_CLAIM_NOTE
    assert list(report["forbidden_claims"]) == list(FORBIDDEN_CLAIMS)
    assert report["any_candidate_passed"] is True
    assert "integration" not in report["claim_note"].lower() or "not" in report["claim_note"].lower()


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "compatibility sweep",
        "not vllm integration",
        "not installed into system python",
        "default runtime",
        "throughput",
        "vericache",
        "future integration work",
    ):
        assert phrase in text, phrase


def test_doc_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("exactkv is integrated with vllm", "throughput improved", "production serving works"):
        assert phrase not in text
