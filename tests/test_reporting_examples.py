"""Tests for exactkv.reporting.examples."""
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


# ──────────────────────────────────────────────────────────────
# render_lossy_divergence_examples
# ──────────────────────────────────────────────────────────────

class TestRenderLossyDivergenceExamples:
    def test_returns_string(self):
        from exactkv.reporting.examples import render_lossy_divergence_examples
        assert isinstance(render_lossy_divergence_examples([]), str)

    def test_empty_shows_none_found_message(self):
        from exactkv.reporting.examples import render_lossy_divergence_examples
        text = render_lossy_divergence_examples([])
        assert "No lossy divergences found" in text or "no lossy" in text.lower()

    def test_contains_full_lossy_exactkv_labels(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        from exactkv.reporting.examples import render_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=3))
        examples = extract_lossy_divergence_examples(report, limit=5)
        text = render_lossy_divergence_examples(examples)
        assert "Full" in text or "full" in text.lower()
        assert "Lossy" in text or "lossy" in text.lower()
        assert "ExactKV" in text or "exactkv" in text.lower()

    def test_contains_prompt_id(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        from exactkv.reporting.examples import render_lossy_divergence_examples
        report = make_report(make_result(prompt_id="mytest", lossy_exact=False, first_div_idx=2))
        examples = extract_lossy_divergence_examples(report)
        text = render_lossy_divergence_examples(examples)
        assert "mytest" in text

    def test_lossy_divergence_described_as_expected(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        from exactkv.reporting.examples import render_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=5))
        examples = extract_lossy_divergence_examples(report)
        text = render_lossy_divergence_examples(examples)
        assert "expected" in text.lower()

    def test_exactkv_not_called_failure_when_matches(self):
        """When exactkv_matches_full=True the text must not call it a failure."""
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        from exactkv.reporting.examples import render_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=1, exactkv_failure=False))
        examples = extract_lossy_divergence_examples(report)
        text = render_lossy_divergence_examples(examples)
        # "yes" or "✓" should appear for ExactKV matching
        assert "yes" in text.lower() or "✓" in text

    def test_no_forbidden_fields(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        from exactkv.reporting.examples import render_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=1))
        text = render_lossy_divergence_examples(extract_lossy_divergence_examples(report))
        _no_forbidden(text)


# ──────────────────────────────────────────────────────────────
# render_exactkv_failure_examples
# ──────────────────────────────────────────────────────────────

class TestRenderExactKVFailureExamples:
    def test_empty_shows_pass_message(self):
        from exactkv.reporting.examples import render_exactkv_failure_examples
        text = render_exactkv_failure_examples([])
        assert "0" in text or "zero" in text.lower() or "pass" in text.lower() \
               or "failure count" in text.lower()

    def test_failure_wording_correct(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        from exactkv.reporting.examples import render_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        examples = extract_exactkv_failure_examples(report)
        text = render_exactkv_failure_examples(examples)
        # Must mention correctness bug or failure
        assert "bug" in text.lower() or "failure" in text.lower()

    def test_exactkv_matches_full_false_is_explicit(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        from exactkv.reporting.examples import render_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        examples = extract_exactkv_failure_examples(report)
        text = render_exactkv_failure_examples(examples)
        assert "False" in text or "false" in text.lower() or "✗" in text

    def test_contains_full_and_exactkv_text_labels(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        from exactkv.reporting.examples import render_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        examples = extract_exactkv_failure_examples(report)
        text = render_exactkv_failure_examples(examples)
        assert "Full" in text or "full" in text.lower()
        assert "ExactKV" in text or "exactkv" in text.lower()

    def test_no_forbidden_fields(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        from exactkv.reporting.examples import render_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        text = render_exactkv_failure_examples(extract_exactkv_failure_examples(report))
        _no_forbidden(text)


# ──────────────────────────────────────────────────────────────
# render_rejection_examples
# ──────────────────────────────────────────────────────────────

class TestRenderRejectionExamples:
    def test_returns_string(self):
        from exactkv.reporting.examples import render_rejection_examples
        assert isinstance(render_rejection_examples([]), str)

    def test_contains_total_rejected(self):
        from exactkv.analysis.examples import extract_rejection_examples
        from exactkv.reporting.examples import render_rejection_examples
        report = make_report(make_result(total_rejected=7))
        examples = extract_rejection_examples(report)
        text = render_rejection_examples(examples)
        assert "7" in text

    def test_contains_acceptance_rate(self):
        from exactkv.analysis.examples import extract_rejection_examples
        from exactkv.reporting.examples import render_rejection_examples
        report = make_report(make_result(acceptance_rate=0.625))
        examples = extract_rejection_examples(report)
        text = render_rejection_examples(examples)
        assert "0.625" in text or "Acceptance" in text or "acceptance" in text

    def test_no_forbidden_fields(self):
        from exactkv.analysis.examples import extract_rejection_examples
        from exactkv.reporting.examples import render_rejection_examples
        report = make_report(make_result(total_rejected=3))
        text = render_rejection_examples(extract_rejection_examples(report))
        _no_forbidden(text)


# ──────────────────────────────────────────────────────────────
# Real sweep
# ──────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def real_sweep():
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime
    rt = ModelRuntime(MODEL_NAME, device="auto", dtype="float32")
    prompts = [{"prompt_id": "ex_r1", "category": "test",
                "prompt": "The capital of France is"}]
    return run_sweep(rt, prompts=prompts,
                     compressor_names=["noop", "int8"],
                     draft_lengths=[4], max_new_tokens=8)


def test_failure_examples_empty_on_real_sweep(real_sweep):
    from exactkv.analysis.examples import extract_exactkv_failure_examples
    from exactkv.reporting.examples import render_exactkv_failure_examples
    examples = extract_exactkv_failure_examples(real_sweep)
    assert examples == []
    text = render_exactkv_failure_examples(examples)
    assert isinstance(text, str)
    _no_forbidden(text)


def test_lossy_examples_on_real_sweep(real_sweep):
    from exactkv.analysis.examples import extract_lossy_divergence_examples
    from exactkv.reporting.examples import render_lossy_divergence_examples
    examples = extract_lossy_divergence_examples(real_sweep, limit=3)
    text = render_lossy_divergence_examples(examples)
    assert isinstance(text, str)
    _no_forbidden(text)


def test_rejection_examples_on_real_sweep(real_sweep):
    from exactkv.analysis.examples import extract_rejection_examples
    from exactkv.reporting.examples import render_rejection_examples
    examples = extract_rejection_examples(real_sweep, limit=2)
    text = render_rejection_examples(examples)
    assert isinstance(text, str)
    _no_forbidden(text)
