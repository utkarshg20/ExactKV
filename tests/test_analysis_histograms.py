"""Tests for exactkv.analysis.histograms (V3 Phase B).

Gates
-----
* Histogram-count reconciliation — bucket counts sum to the number of results.
* Bucket correctness — values land in the right bucket.
* no_divergence bucket — None idx goes to the sentinel bucket.
* Grouping — group_by compressor / draft_len / category produces one
  sub-histogram per group.
* Sweep reports — functions work on sweep-shaped dicts.
* No forbidden performance fields in any returned structure.
"""
from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total(h: dict) -> int:
    return h["total"]


def _counts(h: dict) -> dict:
    return dict(h["buckets"])


def _no_forbidden(obj, path="root"):
    """Recursively assert no forbidden fields appear."""
    if isinstance(obj, dict):
        for k in obj:
            assert k not in _FORBIDDEN, f"Forbidden field {k!r} at {path}"
            _no_forbidden(obj[k], path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_forbidden(v, path=f"{path}[{i}]")


# ---------------------------------------------------------------------------
# accepted_length_histogram
# ---------------------------------------------------------------------------

class TestAcceptedLengthHistogram:
    def test_total_equals_result_count(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(
            make_result(avg_accepted_per_round=1.5),
            make_result(avg_accepted_per_round=3.0),
            make_result(avg_accepted_per_round=6.0),
        )
        h = accepted_length_histogram(report)
        assert _total(h) == 3

    def test_bucket_0(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(make_result(avg_accepted_per_round=0.0))
        h = accepted_length_histogram(report)
        assert _counts(h)["0"] == 1

    def test_bucket_1(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(make_result(avg_accepted_per_round=1.9))
        h = accepted_length_histogram(report)
        assert _counts(h)["1"] == 1

    def test_bucket_2_3(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(
            make_result(avg_accepted_per_round=2.0),
            make_result(avg_accepted_per_round=3.9),
        )
        h = accepted_length_histogram(report)
        assert _counts(h)["2-3"] == 2

    def test_bucket_4_7(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(make_result(avg_accepted_per_round=4.5))
        h = accepted_length_histogram(report)
        assert _counts(h)["4-7"] == 1

    def test_bucket_16_plus(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(make_result(avg_accepted_per_round=20.0))
        h = accepted_length_histogram(report)
        assert _counts(h)["16+"] == 1

    def test_all_buckets_present(self):
        from exactkv.analysis.histograms import accepted_length_histogram, DEFAULT_ACCEPTED_BUCKETS
        h = accepted_length_histogram(make_report())
        expected = {label for _, _, label in DEFAULT_ACCEPTED_BUCKETS}
        assert set(h["buckets"]) == expected

    def test_empty_report_total_zero(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        h = accepted_length_histogram(make_report())
        assert _total(h) == 0

    def test_group_by_compressor(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(
            make_result(compressor_name="noop", avg_accepted_per_round=4.0),
            make_result(compressor_name="int8", avg_accepted_per_round=1.0),
            make_result(compressor_name="int8", avg_accepted_per_round=2.0),
        )
        h = accepted_length_histogram(report, group_by="compressor_name")
        assert "noop" in h
        assert "int8" in h
        assert h["noop"]["total"] == 1
        assert h["int8"]["total"] == 2

    def test_group_by_draft_len(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(
            make_result(draft_len=4, avg_accepted_per_round=2.0),
            make_result(draft_len=8, avg_accepted_per_round=4.0),
            make_result(draft_len=8, avg_accepted_per_round=6.0),
        )
        h = accepted_length_histogram(report, group_by="draft_len")
        assert h["4"]["total"] == 1
        assert h["8"]["total"] == 2

    def test_group_by_category(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        report = make_report(
            make_result(category="code", avg_accepted_per_round=3.0),
            make_result(category="json", avg_accepted_per_round=5.0),
        )
        h = accepted_length_histogram(report, group_by="category")
        assert "code" in h and "json" in h

    def test_no_forbidden_fields(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        h = accepted_length_histogram(make_report(make_result()))
        _no_forbidden(h)


# ---------------------------------------------------------------------------
# first_divergence_histogram
# ---------------------------------------------------------------------------

class TestFirstDivergenceHistogram:
    def test_total_equals_result_count(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(
            make_result(lossy_exact=True),   # no divergence
            make_result(lossy_exact=False, first_div_idx=2),
            make_result(lossy_exact=False, first_div_idx=10),
        )
        h = first_divergence_histogram(report)
        assert _total(h) == 3

    def test_no_divergence_bucket_for_none(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(make_result(lossy_exact=True))
        h = first_divergence_histogram(report)
        assert _counts(h)["no_divergence"] == 1

    def test_bucket_0_for_idx_0(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(make_result(lossy_exact=False, first_div_idx=0))
        h = first_divergence_histogram(report)
        assert _counts(h)["0"] == 1

    def test_bucket_1_4_for_idx_3(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(make_result(lossy_exact=False, first_div_idx=3))
        h = first_divergence_histogram(report)
        assert _counts(h)["1-4"] == 1

    def test_bucket_5_16_for_idx_12(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(make_result(lossy_exact=False, first_div_idx=12))
        h = first_divergence_histogram(report)
        assert _counts(h)["5-16"] == 1

    def test_bucket_33_plus(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(make_result(lossy_exact=False, first_div_idx=50))
        h = first_divergence_histogram(report)
        assert _counts(h)["33+"] == 1

    def test_all_buckets_present(self):
        from exactkv.analysis.histograms import (
            first_divergence_histogram,
            DEFAULT_DIVERGENCE_BUCKETS,
        )
        h = first_divergence_histogram(make_report())
        expected = {label for _, _, label in DEFAULT_DIVERGENCE_BUCKETS}
        assert set(h["buckets"]) == expected

    def test_mixed_report(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(
            make_result(lossy_exact=True),            # no_divergence
            make_result(lossy_exact=False, first_div_idx=1),   # 1-4
            make_result(lossy_exact=False, first_div_idx=20),  # 17-32
        )
        h = first_divergence_histogram(report)
        assert _counts(h)["no_divergence"] == 1
        assert _counts(h)["1-4"] == 1
        assert _counts(h)["17-32"] == 1
        assert _total(h) == 3

    def test_group_by_compressor(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        report = make_report(
            make_result(compressor_name="noop", lossy_exact=True),
            make_result(compressor_name="int8", lossy_exact=False, first_div_idx=5),
        )
        h = first_divergence_histogram(report, group_by="compressor_name")
        assert h["noop"]["total"] == 1
        assert h["int8"]["total"] == 1
        assert h["noop"]["buckets"]["no_divergence"] == 1
        assert h["int8"]["buckets"]["5-16"] == 1

    def test_no_forbidden_fields(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        h = first_divergence_histogram(make_report(make_result()))
        _no_forbidden(h)


# ---------------------------------------------------------------------------
# rejection_count_histogram
# ---------------------------------------------------------------------------

class TestRejectionCountHistogram:
    def test_total_equals_result_count(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        report = make_report(
            make_result(total_rejected=0),
            make_result(total_rejected=2),
            make_result(total_rejected=8),
        )
        h = rejection_count_histogram(report)
        assert _total(h) == 3

    def test_bucket_zero_rejections(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        report = make_report(make_result(total_rejected=0))
        h = rejection_count_histogram(report)
        assert _counts(h)["0"] == 1

    def test_bucket_1_2(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        report = make_report(
            make_result(total_rejected=1),
            make_result(total_rejected=2),
        )
        h = rejection_count_histogram(report)
        assert _counts(h)["1-2"] == 2

    def test_bucket_11_plus(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        report = make_report(make_result(total_rejected=20))
        h = rejection_count_histogram(report)
        assert _counts(h)["11+"] == 1

    def test_group_by_compressor(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        report = make_report(
            make_result(compressor_name="noop", total_rejected=0),
            make_result(compressor_name="int4_sim", total_rejected=10),
            make_result(compressor_name="int4_sim", total_rejected=15),
        )
        h = rejection_count_histogram(report, group_by="compressor_name")
        assert h["noop"]["buckets"]["0"] == 1
        assert h["int4_sim"]["total"] == 2

    def test_no_forbidden_fields(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        h = rejection_count_histogram(make_report(make_result()))
        _no_forbidden(h)


# ---------------------------------------------------------------------------
# Real sweep report (module-scoped to share model load)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

import json

@pytest.fixture(scope="module")
def real_sweep_report(tmp_path_factory):
    """Generate a small sweep report (1 prompt × 2 compressors × 1 draft_len)."""
    from exactkv.benchmarks.runner import RunConfig, run_one
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime

    rt = ModelRuntime(MODEL_NAME, device="auto", dtype="float32")
    prompts = [{"prompt_id": "hist_t1", "category": "test",
                "prompt": "The capital of France is"}]
    return run_sweep(
        runtime=rt,
        prompts=prompts,
        compressor_names=["noop", "int8"],
        draft_lengths=[4],
        max_new_tokens=8,
    )


def test_accepted_length_histogram_on_real_sweep(real_sweep_report):
    from exactkv.analysis.histograms import accepted_length_histogram
    h = accepted_length_histogram(real_sweep_report)
    # 1 prompt × 2 compressors × 1 draft_len = 2 results
    assert _total(h) == 2
    assert sum(_counts(h).values()) == 2


def test_first_divergence_histogram_on_real_sweep(real_sweep_report):
    from exactkv.analysis.histograms import first_divergence_histogram
    h = first_divergence_histogram(real_sweep_report)
    assert _total(h) == 2
    # noop never diverges
    assert _counts(h)["no_divergence"] >= 1


def test_rejection_count_histogram_on_real_sweep(real_sweep_report):
    from exactkv.analysis.histograms import rejection_count_histogram
    h = rejection_count_histogram(real_sweep_report)
    assert _total(h) == 2


def test_histograms_group_by_compressor_on_real_sweep(real_sweep_report):
    from exactkv.analysis.histograms import accepted_length_histogram
    h = accepted_length_histogram(real_sweep_report, group_by="compressor_name")
    assert "noop" in h
    assert "int8" in h
    assert h["noop"]["total"] == 1
    assert h["int8"]["total"] == 1


def test_real_sweep_histograms_no_forbidden_fields(real_sweep_report):
    from exactkv.analysis.histograms import (
        accepted_length_histogram,
        first_divergence_histogram,
        rejection_count_histogram,
    )
    for fn in [accepted_length_histogram, first_divergence_histogram,
               rejection_count_histogram]:
        _no_forbidden(fn(real_sweep_report))
