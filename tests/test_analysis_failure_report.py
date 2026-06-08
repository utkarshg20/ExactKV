"""Tests for exactkv.analysis.failure_report.

Gate: failure report gate.

Verifies:
  * status == "pass" when exactkv_failure_count == 0.
  * status == "fail" when exactkv_failure_count > 0.
  * list_exactkv_failures returns empty list when no failures.
  * list_exactkv_failures returns failing entries.
  * list_lossy_divergences returns empty list when no divergence.
  * list_lossy_divergences returns diverging entries.
  * Lossy divergence is NOT treated as ExactKV failure.
  * build_failure_report counts match list lengths.
  * write_failure_report_json writes valid JSON (creates dirs).
  * No forbidden performance fields in any output.
  * Works on both synthetic and real sweep reports.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def all_pass_report():
    return make_report(
        make_result(compressor_name="noop", exactkv_failure=False, lossy_exact=True),
        make_result(compressor_name="int8", exactkv_failure=False, lossy_exact=True),
    )


@pytest.fixture
def lossy_diverges_but_ekv_pass():
    """Lossy diverges; ExactKV still correct — normal for real compressors."""
    return make_report(
        make_result(compressor_name="int8",
                    lossy_exact=False, first_div_idx=2,
                    exactkv_failure=False),
    )


@pytest.fixture
def exactkv_failure_report():
    """Contains one ExactKV failure alongside a passing result."""
    return make_report(
        make_result(compressor_name="noop", exactkv_failure=False, lossy_exact=True),
        make_result(compressor_name="bad",  exactkv_failure=True,  lossy_exact=False,
                    first_div_idx=0),
    )


@pytest.fixture
def both_diverge_report():
    """Both lossy and ExactKV diverge in one result; separate in another."""
    return make_report(
        make_result(compressor_name="bad",  exactkv_failure=True,  lossy_exact=False, first_div_idx=1),
        make_result(compressor_name="int8", exactkv_failure=False, lossy_exact=False, first_div_idx=3),
    )


# ---------------------------------------------------------------------------
# build_failure_report — status field
# ---------------------------------------------------------------------------

def test_status_pass_when_no_failures(all_pass_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(all_pass_report)
    assert fr["status"] == "pass"


def test_status_fail_when_exactkv_failure(exactkv_failure_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(exactkv_failure_report)
    assert fr["status"] == "fail"


def test_status_pass_when_only_lossy_diverges(lossy_diverges_but_ekv_pass):
    """Lossy divergence must NOT set status to 'fail'."""
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(lossy_diverges_but_ekv_pass)
    assert fr["status"] == "pass"


# ---------------------------------------------------------------------------
# build_failure_report — counts
# ---------------------------------------------------------------------------

def test_failure_count_zero_when_all_pass(all_pass_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(all_pass_report)
    assert fr["exactkv_failure_count"] == 0


def test_failure_count_equals_list_length(exactkv_failure_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(exactkv_failure_report)
    assert fr["exactkv_failure_count"] == len(fr["exactkv_failures"])


def test_lossy_divergence_count_correct(both_diverge_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(both_diverge_report)
    assert fr["lossy_divergence_count"] == 2   # both results have lossy divergence


def test_lossy_divergence_count_equals_list_length(both_diverge_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(both_diverge_report)
    assert fr["lossy_divergence_count"] == len(fr["lossy_divergences"])


# ---------------------------------------------------------------------------
# list_exactkv_failures
# ---------------------------------------------------------------------------

def test_list_exactkv_failures_empty_when_none(all_pass_report):
    from exactkv.analysis.failure_report import list_exactkv_failures
    assert list_exactkv_failures(all_pass_report) == []


def test_list_exactkv_failures_contains_failing(exactkv_failure_report):
    from exactkv.analysis.failure_report import list_exactkv_failures
    failures = list_exactkv_failures(exactkv_failure_report)
    assert len(failures) == 1
    assert failures[0]["compressor_name"] == "bad"


def test_list_exactkv_failures_required_keys(exactkv_failure_report):
    from exactkv.analysis.failure_report import list_exactkv_failures
    required = {"prompt_id", "compressor_name", "draft_len", "category"}
    for rec in list_exactkv_failures(exactkv_failure_report):
        assert required <= set(rec.keys())


# ---------------------------------------------------------------------------
# list_lossy_divergences
# ---------------------------------------------------------------------------

def test_list_lossy_divergences_empty_when_none(all_pass_report):
    from exactkv.analysis.failure_report import list_lossy_divergences
    assert list_lossy_divergences(all_pass_report) == []


def test_list_lossy_divergences_contains_diverged(lossy_diverges_but_ekv_pass):
    from exactkv.analysis.failure_report import list_lossy_divergences
    divs = list_lossy_divergences(lossy_diverges_but_ekv_pass)
    assert len(divs) == 1
    assert divs[0]["first_divergence_idx"] == 2


def test_list_lossy_divergences_required_keys(lossy_diverges_but_ekv_pass):
    from exactkv.analysis.failure_report import list_lossy_divergences
    required = {"prompt_id", "compressor_name", "draft_len", "category", "first_divergence_idx"}
    for rec in list_lossy_divergences(lossy_diverges_but_ekv_pass):
        assert required <= set(rec.keys())


# ---------------------------------------------------------------------------
# Separation: lossy divergence ≠ ExactKV failure
# ---------------------------------------------------------------------------

def test_lossy_only_not_in_exactkv_failures(lossy_diverges_but_ekv_pass):
    """A result where only lossy diverges must NOT appear in list_exactkv_failures."""
    from exactkv.analysis.failure_report import list_exactkv_failures
    assert list_exactkv_failures(lossy_diverges_but_ekv_pass) == []


def test_exactkv_failure_separately_tracked(both_diverge_report):
    """ExactKV failure and lossy divergence must be listed in separate lists."""
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(both_diverge_report)
    # Only 1 result has exactkv_failure=True
    assert fr["exactkv_failure_count"] == 1
    # Both results have lossy divergence
    assert fr["lossy_divergence_count"] == 2


# ---------------------------------------------------------------------------
# No forbidden fields
# ---------------------------------------------------------------------------

def test_failure_report_no_forbidden_fields(exactkv_failure_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(exactkv_failure_report)

    def _check(obj, path="root"):
        if isinstance(obj, dict):
            for key in obj:
                assert key not in _FORBIDDEN, f"Forbidden field {key!r} at {path}"
                _check(obj[key], f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _check(item, f"{path}[{i}]")

    _check(fr)


# ---------------------------------------------------------------------------
# write_failure_report_json
# ---------------------------------------------------------------------------

def test_write_failure_report_json(exactkv_failure_report, tmp_path):
    from exactkv.analysis.failure_report import build_failure_report, write_failure_report_json
    fr = build_failure_report(exactkv_failure_report)
    path = tmp_path / "failures.json"
    write_failure_report_json(fr, path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "fail"
    assert loaded["exactkv_failure_count"] == 1


def test_write_failure_report_json_creates_dirs(all_pass_report, tmp_path):
    from exactkv.analysis.failure_report import build_failure_report, write_failure_report_json
    nested = tmp_path / "a" / "b" / "failures.json"
    assert not nested.parent.exists()
    write_failure_report_json(build_failure_report(all_pass_report), nested)
    assert nested.exists()


def test_write_failure_report_json_pass(all_pass_report, tmp_path):
    from exactkv.analysis.failure_report import build_failure_report, write_failure_report_json
    fr = build_failure_report(all_pass_report)
    path = tmp_path / "pass.json"
    write_failure_report_json(fr, path)
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "pass"
    assert loaded["exactkv_failure_count"] == 0


# ---------------------------------------------------------------------------
# Integration: real sweep report
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def real_sweep_report():
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime

    rt = ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")
    return run_sweep(
        runtime=rt,
        prompts=[{"prompt_id": "fr_001", "category": "nl",
                  "prompt": "The capital of Italy is"}],
        compressor_names=["noop", "int8"],
        draft_lengths=[4],
        max_new_tokens=8,
        prompt_suite="test",
    )


def test_failure_report_status_pass_on_real_sweep(real_sweep_report):
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(real_sweep_report)
    assert fr["status"] == "pass", (
        f"Expected pass but got failures: {fr['exactkv_failures']}"
    )


def test_lossy_divergence_in_real_sweep_not_failure(real_sweep_report):
    """int8 may produce lossy divergence; that is expected, not an ExactKV failure."""
    from exactkv.analysis.failure_report import build_failure_report
    fr = build_failure_report(real_sweep_report)
    # lossy_divergence_count >= 0 is fine; the key check is status == "pass"
    assert fr["exactkv_failure_count"] == 0
