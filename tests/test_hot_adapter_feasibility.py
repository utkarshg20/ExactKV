"""Tests for Experiment 032 hot-adapter feasibility helpers."""
from __future__ import annotations

import pytest

from exactkv.analysis.hot_adapter_feasibility import (
    FeasibilityClass,
    analyze_shardkv,
    analyze_snapkv,
    build_feasibility_artifact,
    design_snapkv_experimental_mvp,
)


def test_snapkv_classified_restricted() -> None:
    snapkv = analyze_snapkv()
    assert snapkv.classification == FeasibilityClass.B_RESTRICTED
    assert not snapkv.exactkv_generator_changes_required
    assert not snapkv.verification_engine_changes_required
    assert snapkv.attention_weights_required


def test_shardkv_classified_no_go() -> None:
    shardkv = analyze_shardkv()
    assert shardkv.classification == FeasibilityClass.C_NO_GO


def test_build_artifact_no_adapter_implemented() -> None:
    artifact = build_feasibility_artifact()
    assert artifact["adapter_implemented"] is False
    assert artifact["chosen_path"] == "snapkv_restricted_mvp_phase_5b"
    assert artifact["interpretation"]["full_kv_verifier_authoritative"] is True
    assert artifact["interpretation"]["production_snapkv_claim_allowed"] is False


def test_design_mvp_not_in_registry() -> None:
    design = design_snapkv_experimental_mvp()
    assert design["compressor_name"] == "snapkv_experimental"
    assert "NOT default registry" in design["registry_policy"]
    assert design["implementation_status"].startswith("design_only")


def test_artifact_forbidden_fields_absent() -> None:
    artifact = build_feasibility_artifact()
    forbidden = {"tokens_per_second", "throughput", "speedup"}
    text = str(artifact)
    for field in forbidden:
        assert field not in artifact
        assert f"'{field}'" not in text or field in ("interpretation",)
