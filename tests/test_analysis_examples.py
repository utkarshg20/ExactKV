"""Tests for exactkv.analysis.examples (V3 Phase B).

Gates
-----
* Lossy divergence examples include full_text, lossy_text, exactkv_text.
* Lossy divergence is NOT treated as ExactKV failure.
* exactkv_matches_full is True in every lossy divergence example (since
  ExactKV corrects the divergence).
* ExactKV failure examples are empty when exactkv_failures == 0.
* Synthetic ExactKV failure appears in failure examples.
* Rejection examples are sorted by total_rejected descending.
* No forbidden performance fields in any returned dict.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}


def _no_forbidden(obj, path="root"):
    if isinstance(obj, dict):
        for k in obj:
            assert k not in _FORBIDDEN, f"Forbidden field {k!r} at {path}"
            _no_forbidden(obj[k], path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_forbidden(v, path=f"{path}[{i}]")


# ──────────────────────────────────────────────────────────────
# extract_lossy_divergence_examples
# ──────────────────────────────────────────────────────────────

class TestLossyDivergenceExamples:
    def test_empty_when_no_divergence(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=True))
        assert extract_lossy_divergence_examples(report) == []

    def test_returns_diverged_results(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(
            make_result(prompt_id="p1", lossy_exact=True),
            make_result(prompt_id="p2", lossy_exact=False, first_div_idx=3),
        )
        examples = extract_lossy_divergence_examples(report)
        assert len(examples) == 1
        assert examples[0]["prompt_id"] == "p2"

    def test_limit_respected(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(
            *[make_result(prompt_id=f"p{i}", lossy_exact=False, first_div_idx=1)
              for i in range(10)]
        )
        examples = extract_lossy_divergence_examples(report, limit=3)
        assert len(examples) == 3

    def test_includes_full_lossy_exactkv_text(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=2))
        ex = extract_lossy_divergence_examples(report)[0]
        assert "full_text" in ex
        assert "lossy_text" in ex
        assert "exactkv_text" in ex

    def test_includes_first_divergence_idx(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=5))
        ex = extract_lossy_divergence_examples(report)[0]
        assert ex["first_divergence_idx"] == 5

    def test_exactkv_matches_full_true_when_no_failure(self):
        """Lossy divergence examples must have exactkv_matches_full=True.
        Lossy divergence != ExactKV failure."""
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(
            make_result(lossy_exact=False, first_div_idx=1, exactkv_failure=False)
        )
        ex = extract_lossy_divergence_examples(report)[0]
        assert ex["exactkv_matches_full"] is True

    def test_lossy_matches_full_false(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=2))
        ex = extract_lossy_divergence_examples(report)[0]
        assert ex["lossy_matches_full"] is False

    def test_explanation_field_present(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=1))
        ex = extract_lossy_divergence_examples(report)[0]
        assert "explanation" in ex
        assert "expected" in ex["explanation"].lower()

    def test_explanation_mentions_exackv_failure_distinction(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=1))
        ex = extract_lossy_divergence_examples(report)[0]
        # The explanation must distinguish lossy divergence from ExactKV failure
        assert "correctness" in ex["explanation"].lower() or \
               "bug" in ex["explanation"].lower() or \
               "ExactKV" in ex["explanation"]

    def test_includes_compressor_and_category(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(
            make_result(compressor_name="int4_sim", category="code",
                        lossy_exact=False, first_div_idx=1)
        )
        ex = extract_lossy_divergence_examples(report)[0]
        assert ex["compressor_name"] == "int4_sim"
        assert ex["category"] == "code"

    def test_no_forbidden_fields(self):
        from exactkv.analysis.examples import extract_lossy_divergence_examples
        report = make_report(make_result(lossy_exact=False, first_div_idx=3))
        _no_forbidden(extract_lossy_divergence_examples(report))


# ──────────────────────────────────────────────────────────────
# extract_exactkv_failure_examples
# ──────────────────────────────────────────────────────────────

class TestExactKVFailureExamples:
    def test_empty_when_no_failures(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        report = make_report(
            make_result(exactkv_failure=False),
            make_result(exactkv_failure=False),
        )
        assert extract_exactkv_failure_examples(report) == []

    def test_returns_failing_result(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        report = make_report(
            make_result(prompt_id="ok", exactkv_failure=False),
            make_result(prompt_id="bad", exactkv_failure=True),
        )
        examples = extract_exactkv_failure_examples(report)
        assert len(examples) == 1
        assert examples[0]["prompt_id"] == "bad"

    def test_exactkv_matches_full_false(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        ex = extract_exactkv_failure_examples(report)[0]
        assert ex["exactkv_matches_full"] is False

    def test_includes_full_and_exactkv_text(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        ex = extract_exactkv_failure_examples(report)[0]
        assert "full_text" in ex
        assert "exactkv_text" in ex

    def test_limit_respected(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        report = make_report(
            *[make_result(prompt_id=f"bad{i}", exactkv_failure=True) for i in range(6)]
        )
        examples = extract_exactkv_failure_examples(report, limit=2)
        assert len(examples) == 2

    def test_note_field_describes_failure(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        ex = extract_exactkv_failure_examples(report)[0]
        assert "note" in ex
        assert "bug" in ex["note"].lower() or "correctness" in ex["note"].lower()

    def test_no_forbidden_fields(self):
        from exactkv.analysis.examples import extract_exactkv_failure_examples
        report = make_report(make_result(exactkv_failure=True))
        _no_forbidden(extract_exactkv_failure_examples(report))


# ──────────────────────────────────────────────────────────────
# extract_rejection_examples
# ──────────────────────────────────────────────────────────────

class TestRejectionExamples:
    def test_sorted_by_rejected_descending(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(
            make_result(prompt_id="low",  total_rejected=1),
            make_result(prompt_id="high", total_rejected=10),
            make_result(prompt_id="mid",  total_rejected=5),
        )
        examples = extract_rejection_examples(report)
        assert examples[0]["prompt_id"] == "high"
        assert examples[1]["prompt_id"] == "mid"
        assert examples[2]["prompt_id"] == "low"

    def test_limit_respected(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(
            *[make_result(prompt_id=f"p{i}", total_rejected=i) for i in range(10)]
        )
        examples = extract_rejection_examples(report, limit=3)
        assert len(examples) == 3

    def test_includes_total_rejected(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(make_result(total_rejected=7, total_corrections=3))
        ex = extract_rejection_examples(report)[0]
        assert ex["total_rejected"] == 7

    def test_includes_total_corrections(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(make_result(total_rejected=4, total_corrections=2))
        ex = extract_rejection_examples(report)[0]
        assert ex["total_corrections"] == 2

    def test_includes_acceptance_rate(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(make_result(acceptance_rate=0.75))
        ex = extract_rejection_examples(report)[0]
        assert "acceptance_rate" in ex

    def test_note_field_present(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(make_result(total_rejected=5))
        ex = extract_rejection_examples(report)[0]
        assert "note" in ex

    def test_exactkv_matches_full_present(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(make_result(exactkv_failure=False))
        ex = extract_rejection_examples(report)[0]
        assert "exactkv_matches_full" in ex
        assert ex["exactkv_matches_full"] is True

    def test_empty_report(self):
        from exactkv.analysis.examples import extract_rejection_examples
        assert extract_rejection_examples(make_report()) == []

    def test_no_forbidden_fields(self):
        from exactkv.analysis.examples import extract_rejection_examples
        report = make_report(make_result(total_rejected=3))
        _no_forbidden(extract_rejection_examples(report))


# ──────────────────────────────────────────────────────────────
# __init__ re-exports
# ──────────────────────────────────────────────────────────────

def test_public_api_exported():
    from exactkv import analysis
    for name in (
        "extract_lossy_divergence_examples",
        "extract_exactkv_failure_examples",
        "extract_rejection_examples",
    ):
        assert hasattr(analysis, name), f"analysis.{name} not exported"


# ──────────────────────────────────────────────────────────────
# Real sweep report
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_sweep(tmp_path_factory):
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime

    rt = ModelRuntime(MODEL_NAME, device="auto", dtype="float32")
    prompts = [
        {"prompt_id": "ex_t1", "category": "test",
         "prompt": "The capital of France is"},
    ]
    return run_sweep(
        runtime=rt,
        prompts=prompts,
        compressor_names=["noop", "int8"],
        draft_lengths=[4],
        max_new_tokens=8,
    )


def test_lossy_divergence_examples_on_real_sweep(real_sweep):
    from exactkv.analysis.examples import extract_lossy_divergence_examples
    examples = extract_lossy_divergence_examples(real_sweep, limit=5)
    # May be 0 (noop never diverges, int8 may or may not)
    assert isinstance(examples, list)
    for ex in examples:
        assert ex["exactkv_matches_full"] is True  # ExactKV corrects all
        assert ex["lossy_matches_full"] is False


def test_exactkv_failure_examples_empty_on_real_sweep(real_sweep):
    from exactkv.analysis.examples import extract_exactkv_failure_examples
    # In a correct implementation there must be zero failures
    examples = extract_exactkv_failure_examples(real_sweep)
    assert examples == [], f"Expected 0 ExactKV failures, got {examples}"


def test_rejection_examples_on_real_sweep(real_sweep):
    from exactkv.analysis.examples import extract_rejection_examples
    examples = extract_rejection_examples(real_sweep, limit=3)
    assert isinstance(examples, list)
    for ex in examples:
        assert "total_rejected" in ex
        assert ex["exactkv_matches_full"] is True


def test_real_sweep_examples_no_forbidden_fields(real_sweep):
    from exactkv.analysis.examples import (
        extract_lossy_divergence_examples,
        extract_exactkv_failure_examples,
        extract_rejection_examples,
    )
    for fn in [extract_lossy_divergence_examples,
               extract_exactkv_failure_examples,
               extract_rejection_examples]:
        _no_forbidden(fn(real_sweep))
