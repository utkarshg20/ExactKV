"""Tests for exactkv.analysis.mismatch.

Gate: mismatch analysis gate.

Verifies:
  * first_lossy_divergences returns one record per result.
  * first_divergence_idx is None when no divergence.
  * first_divergence_idx is int when divergence occurred.
  * mismatch_position_summary counts correctly.
  * Lossy divergence is NOT counted as ExactKV failure.
  * rejection_position_summary captures total_rejected and acceptance_rate.
  * Works on synthetic and real sweep reports.
  * No forbidden performance fields in any output.
"""
from __future__ import annotations

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
def no_divergence_report():
    """All lossy outputs match full — no divergence."""
    return make_report(
        make_result(compressor_name="noop", lossy_exact=True, first_div_idx=None),
        make_result(compressor_name="noop", lossy_exact=True, first_div_idx=None),
    )


@pytest.fixture
def mixed_divergence_report():
    """Some lossy outputs diverge; no ExactKV failures."""
    return make_report(
        make_result(compressor_name="noop", lossy_exact=True, first_div_idx=None),
        make_result(compressor_name="int8", lossy_exact=False, first_div_idx=3),
        make_result(compressor_name="int8", lossy_exact=False, first_div_idx=1),
        make_result(compressor_name="debug_noise", lossy_exact=False, first_div_idx=0),
    )


@pytest.fixture
def exactkv_failure_report():
    """One ExactKV failure + one lossy divergence — must stay separate."""
    return make_report(
        make_result(compressor_name="int8",
                    lossy_exact=False, first_div_idx=2,
                    exactkv_failure=True),
        make_result(compressor_name="noop",
                    lossy_exact=True, first_div_idx=None,
                    exactkv_failure=False),
    )


# ---------------------------------------------------------------------------
# first_lossy_divergences
# ---------------------------------------------------------------------------

def test_first_lossy_divergences_count(mixed_divergence_report):
    from exactkv.analysis.mismatch import first_lossy_divergences
    records = first_lossy_divergences(mixed_divergence_report)
    assert len(records) == len(mixed_divergence_report["results"])


def test_first_lossy_divergences_none_when_no_divergence(no_divergence_report):
    from exactkv.analysis.mismatch import first_lossy_divergences
    for rec in first_lossy_divergences(no_divergence_report):
        assert rec["first_divergence_idx"] is None
        assert rec["lossy_diverged"] is False


def test_first_lossy_divergences_idx_when_diverged(mixed_divergence_report):
    from exactkv.analysis.mismatch import first_lossy_divergences
    diverged = [r for r in first_lossy_divergences(mixed_divergence_report)
                if r["lossy_diverged"]]
    assert len(diverged) == 3
    for rec in diverged:
        assert isinstance(rec["first_divergence_idx"], int)


def test_first_lossy_divergences_required_keys(mixed_divergence_report):
    from exactkv.analysis.mismatch import first_lossy_divergences
    required = {"prompt_id", "category", "compressor_name", "draft_len",
                "first_divergence_idx", "lossy_diverged"}
    for rec in first_lossy_divergences(mixed_divergence_report):
        assert required <= set(rec.keys()), f"Missing keys in record: {rec}"


def test_first_lossy_divergences_no_forbidden(mixed_divergence_report):
    from exactkv.analysis.mismatch import first_lossy_divergences
    for rec in first_lossy_divergences(mixed_divergence_report):
        assert not (set(rec.keys()) & _FORBIDDEN)


# ---------------------------------------------------------------------------
# mismatch_position_summary
# ---------------------------------------------------------------------------

def test_mismatch_summary_total_runs(mixed_divergence_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(mixed_divergence_report)
    assert summary["total_runs"] == 4


def test_mismatch_summary_divergence_count(mixed_divergence_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(mixed_divergence_report)
    assert summary["lossy_divergence_count"] == 3
    assert summary["no_divergence_count"] == 1


def test_mismatch_summary_no_divergence_report(no_divergence_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(no_divergence_report)
    assert summary["lossy_divergence_count"] == 0
    assert summary["no_divergence_count"] == 2
    assert summary["mean_first_divergence_idx"] is None
    assert summary["min_first_divergence_idx"] is None


def test_mismatch_summary_mean_idx(mixed_divergence_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(mixed_divergence_report)
    # Divergence indices are [3, 1, 0] → mean = 4/3 ≈ 1.33
    assert summary["mean_first_divergence_idx"] == pytest.approx((3 + 1 + 0) / 3)


def test_mismatch_summary_min_max_idx(mixed_divergence_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(mixed_divergence_report)
    assert summary["min_first_divergence_idx"] == 0
    assert summary["max_first_divergence_idx"] == 3


def test_mismatch_summary_by_compressor_keys(mixed_divergence_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(mixed_divergence_report)
    by_comp = summary["by_compressor"]
    assert "noop" in by_comp
    assert "int8" in by_comp
    assert "debug_noise" in by_comp


def test_mismatch_summary_no_forbidden_fields(mixed_divergence_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(mixed_divergence_report)
    assert not (set(summary.keys()) & _FORBIDDEN)


# ---------------------------------------------------------------------------
# Lossy divergence ≠ ExactKV failure (key distinction)
# ---------------------------------------------------------------------------

def test_lossy_divergence_not_exactkv_failure(exactkv_failure_report):
    """Lossy divergence count must not be confused with ExactKV failure count."""
    from exactkv.analysis.mismatch import mismatch_position_summary

    summary = mismatch_position_summary(exactkv_failure_report)
    # int8 result: lossy diverged (first_div_idx=2) AND exactkv_failure=True
    # noop result: no lossy divergence, no exactkv failure
    # mismatch summary only tracks *lossy* divergence, not exactkv failures
    assert summary["lossy_divergence_count"] == 1   # only the int8 result


def test_exactkv_failure_not_in_mismatch_summary(exactkv_failure_report):
    """mismatch_position_summary should not expose exactkv_failure counts."""
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(exactkv_failure_report)
    assert "exactkv_failure_count" not in summary
    assert "exactkv_failures" not in summary


# ---------------------------------------------------------------------------
# rejection_position_summary
# ---------------------------------------------------------------------------

def test_rejection_summary_count(mixed_divergence_report):
    from exactkv.analysis.mismatch import rejection_position_summary
    records = rejection_position_summary(mixed_divergence_report)
    assert len(records) == 4


def test_rejection_summary_required_keys(mixed_divergence_report):
    from exactkv.analysis.mismatch import rejection_position_summary
    required = {"prompt_id", "category", "compressor_name", "draft_len",
                "total_rejected", "total_corrections", "acceptance_rate"}
    for rec in rejection_position_summary(mixed_divergence_report):
        assert required <= set(rec.keys())


def test_rejection_summary_values_non_negative(mixed_divergence_report):
    from exactkv.analysis.mismatch import rejection_position_summary
    for rec in rejection_position_summary(mixed_divergence_report):
        assert rec["total_rejected"] >= 0
        assert rec["total_corrections"] >= 0
        assert 0.0 <= rec["acceptance_rate"] <= 1.0


def test_rejection_summary_no_forbidden(mixed_divergence_report):
    from exactkv.analysis.mismatch import rejection_position_summary
    for rec in rejection_position_summary(mixed_divergence_report):
        assert not (set(rec.keys()) & _FORBIDDEN)


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
        prompts=[{"prompt_id": "mm_001", "category": "nl",
                  "prompt": "The capital of Germany is"}],
        compressor_names=["noop", "debug_noise"],
        draft_lengths=[4],
        max_new_tokens=8,
        prompt_suite="test",
    )


def test_mismatch_on_real_sweep(real_sweep_report):
    from exactkv.analysis.mismatch import mismatch_position_summary
    summary = mismatch_position_summary(real_sweep_report)
    # noop: no divergence; debug_noise: likely diverges
    assert summary["total_runs"] == 2


def test_noop_no_divergence_in_real_sweep(real_sweep_report):
    from exactkv.analysis.mismatch import first_lossy_divergences
    records = first_lossy_divergences(real_sweep_report)
    noop_rec = next(r for r in records if r["compressor_name"] == "noop")
    assert noop_rec["lossy_diverged"] is False
