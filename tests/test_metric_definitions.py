"""Metric definitions doc tests (Phase J)."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
METRICS_DOC = _ROOT / "docs/METRIC_DEFINITIONS.md"

REQUIRED_METRICS = (
    "acceptance_rate",
    "first_divergence_index",
    "avg_accepted_span",
    "verifier_agreement_score",
    "exactkv_failure_rate",
    "compression_ratio",
    "stored tensor byte ratio",
    "kernel_microbenchmark_speedup",
    "divergence_type",
    "kernel_consistency",
    "fallback/proxy adapter",
    "probe-first adapter",
)


def test_metric_definitions_exists() -> None:
    assert METRICS_DOC.is_file()


def test_all_metrics_documented() -> None:
    text = METRICS_DOC.read_text(encoding="utf-8").lower()
    for metric in REQUIRED_METRICS:
        assert metric.replace("_", " ") in text or metric in text, f"missing {metric}"


def test_metrics_have_limitations() -> None:
    text = METRICS_DOC.read_text(encoding="utf-8")
    assert text.lower().count("limitation") >= len(REQUIRED_METRICS)


def test_metrics_have_public_claim_status() -> None:
    text = METRICS_DOC.read_text(encoding="utf-8")
    assert "public claim status" in text.lower()
