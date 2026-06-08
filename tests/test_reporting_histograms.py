"""Tests for exactkv.reporting.histograms."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}


def _no_forbidden(text: str):
    for f in _FORBIDDEN:
        assert f not in text, f"Forbidden field {f!r} found in rendered output"


def _basic_report():
    return make_report(
        make_result(avg_accepted_per_round=2.5, lossy_exact=True, total_rejected=0),
        make_result(avg_accepted_per_round=5.0, lossy_exact=False, first_div_idx=3,
                    total_rejected=2),
    )


# ──────────────────────────────────────────────────────────────
# render_accepted_length_table
# ──────────────────────────────────────────────────────────────

class TestRenderAcceptedLengthTable:
    def test_returns_string(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        from exactkv.reporting.histograms import render_accepted_length_table
        h = accepted_length_histogram(_basic_report())
        assert isinstance(render_accepted_length_table(h), str)

    def test_contains_bucket_labels(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        from exactkv.reporting.histograms import render_accepted_length_table
        h = accepted_length_histogram(_basic_report())
        text = render_accepted_length_table(h)
        # Default buckets include "0" and "16+"
        assert "0" in text
        assert "16+" in text

    def test_contains_count_column(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        from exactkv.reporting.histograms import render_accepted_length_table
        h = accepted_length_histogram(_basic_report())
        text = render_accepted_length_table(h)
        assert "count" in text.lower()

    def test_contains_total_row(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        from exactkv.reporting.histograms import render_accepted_length_table
        h = accepted_length_histogram(_basic_report())
        text = render_accepted_length_table(h)
        assert "Total" in text or "total" in text

    def test_grouped_histogram_renders_per_group(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        from exactkv.reporting.histograms import render_accepted_length_table
        report = make_report(
            make_result(compressor_name="noop", avg_accepted_per_round=4.0),
            make_result(compressor_name="int8", avg_accepted_per_round=2.0),
        )
        h = accepted_length_histogram(report, group_by="compressor_name")
        text = render_accepted_length_table(h, group_label="compressor")
        assert "noop" in text
        assert "int8" in text

    def test_no_forbidden_fields(self):
        from exactkv.analysis.histograms import accepted_length_histogram
        from exactkv.reporting.histograms import render_accepted_length_table
        _no_forbidden(render_accepted_length_table(accepted_length_histogram(_basic_report())))


# ──────────────────────────────────────────────────────────────
# render_first_divergence_table
# ──────────────────────────────────────────────────────────────

class TestRenderFirstDivergenceTable:
    def test_returns_string(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        from exactkv.reporting.histograms import render_first_divergence_table
        h = first_divergence_histogram(_basic_report())
        assert isinstance(render_first_divergence_table(h), str)

    def test_contains_no_divergence_bucket(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        from exactkv.reporting.histograms import render_first_divergence_table
        report = make_report(make_result(lossy_exact=True))
        h = first_divergence_histogram(report)
        text = render_first_divergence_table(h)
        assert "no_divergence" in text

    def test_contains_divergence_buckets(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        from exactkv.reporting.histograms import render_first_divergence_table
        h = first_divergence_histogram(_basic_report())
        text = render_first_divergence_table(h)
        assert "1-4" in text or "5-16" in text

    def test_no_forbidden_fields(self):
        from exactkv.analysis.histograms import first_divergence_histogram
        from exactkv.reporting.histograms import render_first_divergence_table
        _no_forbidden(render_first_divergence_table(first_divergence_histogram(_basic_report())))


# ──────────────────────────────────────────────────────────────
# render_rejection_count_table
# ──────────────────────────────────────────────────────────────

class TestRenderRejectionCountTable:
    def test_returns_string(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        from exactkv.reporting.histograms import render_rejection_count_table
        h = rejection_count_histogram(_basic_report())
        assert isinstance(render_rejection_count_table(h), str)

    def test_contains_zero_bucket(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        from exactkv.reporting.histograms import render_rejection_count_table
        report = make_report(make_result(total_rejected=0))
        h = rejection_count_histogram(report)
        text = render_rejection_count_table(h)
        assert "0" in text

    def test_no_forbidden_fields(self):
        from exactkv.analysis.histograms import rejection_count_histogram
        from exactkv.reporting.histograms import render_rejection_count_table
        _no_forbidden(render_rejection_count_table(rejection_count_histogram(_basic_report())))


# ──────────────────────────────────────────────────────────────
# Real sweep
# ──────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def real_sweep():
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime
    rt = ModelRuntime(MODEL_NAME, device="auto", dtype="float32")
    prompts = [{"prompt_id": "hist_r1", "category": "test",
                "prompt": "The capital of France is"}]
    return run_sweep(rt, prompts=prompts,
                     compressor_names=["noop", "int8"],
                     draft_lengths=[4], max_new_tokens=8)


def test_all_histogram_tables_render_on_real_sweep(real_sweep):
    from exactkv.analysis.histograms import (
        accepted_length_histogram,
        first_divergence_histogram,
        rejection_count_histogram,
    )
    from exactkv.reporting.histograms import (
        render_accepted_length_table,
        render_first_divergence_table,
        render_rejection_count_table,
    )
    for fn_h, fn_r in [
        (accepted_length_histogram, render_accepted_length_table),
        (first_divergence_histogram, render_first_divergence_table),
        (rejection_count_histogram, render_rejection_count_table),
    ]:
        text = fn_r(fn_h(real_sweep))
        assert isinstance(text, str)
        assert len(text) > 0
        _no_forbidden(text)
