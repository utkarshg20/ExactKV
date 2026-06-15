"""Tests for Experiment 066 streaming quant attention feasibility docs/reports."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exactkv.attention.streaming_quant_attention import (
    EXPERIMENT_066_ID,
    EXP066_CLAIM_NOTE,
    FORBIDDEN_ATTENTION_CLAIMS,
    validate_exp066_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md"
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "run_exp066_streaming_quant_attention_feasibility.py"
)


def _sample_cell(**overrides: object) -> dict[str, object]:
    mem = {
        "full_kv_bytes": 8192,
        "stored_quantized_kv_bytes": 4096,
        "materialized_working_kv_bytes": 8192,
        "streaming_peak_chunk_working_kv_bytes": 2048,
        "metadata_bytes": 256,
        "chunk_size": 16,
        "num_chunks": 8,
        "theoretical_streaming_working_reduction_vs_materialized": 0.75,
    }
    base: dict[str, object] = {
        "dtype": "float32",
        "B": 1,
        "H": 2,
        "Q": 1,
        "T": 64,
        "D": 32,
        "chunk_size": 16,
        "causal": False,
        "passed": True,
        "tolerance": 5e-4,
        "max_abs_streaming_vs_materialized": 1e-6,
        "max_abs_full_vs_materialized": 0.1,
        "max_abs_full_vs_streaming": 0.1,
        "memory_accounting": mem,
    }
    base.update(overrides)
    return base


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_066_ID,
        "status": "pass",
        "total_cells": 1,
        "pass_cells": 1,
        "failed_cells": 0,
        "max_streaming_vs_materialized_error": 1e-6,
        "max_full_vs_streaming_error": 0.1,
        "best_theoretical_streaming_working_reduction": 0.9,
        "worst_theoretical_streaming_working_reduction": 0.5,
        "cells": [_sample_cell()],
        "claim_note": EXP066_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": ["reference only"],
        "no_performance_claims_note": "no throughput claim",
    }
    base.update(overrides)
    return base


def test_exp066_report_schema_validates() -> None:
    assert validate_exp066_report(_report()) == []


def test_exp066_report_missing_key_fails() -> None:
    bad = _report()
    del bad["total_cells"]
    assert any("total_cells" in e for e in validate_exp066_report(bad))


def test_forbidden_claim_terms_listed() -> None:
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "production_memory_savings",
        "active_gpu_memory_savings",
    ):
        assert term in FORBIDDEN_ATTENTION_CLAIMS


def test_doc_required_wording() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "tensor-level feasibility probe",
        "not model inference integration",
        "not wired into exactkv generation",
        "no cuda",
        "no triton",
        "no vllm integration",
        "vericache",
        "materialization bottleneck",
        "online softmax",
    ):
        assert phrase in text, phrase


def test_doc_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("throughput improved", "latency improved", "production memory savings achieved"):
        assert phrase not in text


def test_exp066_script_generates_valid_report(tmp_path: Path) -> None:
    out = tmp_path / "experiment_066_streaming_quant_attention_feasibility.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--json-out", str(out)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT.resolve().parents[2]),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert validate_exp066_report(data) == []
    assert data["total_cells"] == 144
    assert data["pass_cells"] == data["total_cells"]
    blob = json.dumps(data).lower()
    for term in FORBIDDEN_ATTENTION_CLAIMS:
        if term in blob:
            assert term in data.get("forbidden_claims", [])
