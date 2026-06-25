"""Phase I novelty audit tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SYSTEMS = (
    "VeriCache",
    "KVQuant",
    "KIVI",
    "TurboQuant",
    "QuantSpec",
    "SparseSpec",
    "SpecAttn",
    "MagicDec",
    "LMCache",
    "CacheGen",
    "ShardCache (shard-kv)",
)

FORBIDDEN_CLAIMS = (
    "ExactKV is the first system like this.",
    "ExactKV reproduces VeriCache.",
    "ExactKV invented compressed-KV verification.",
    "ExactKV proves end-to-end speedups.",
    "ExactKV proves active GPU memory savings.",
    "ExactKV is production ready.",
    "ExactKV compares real SpectralQuant.",
    "ExactKV compares real Shard.",
)


@pytest.fixture(scope="module")
def novelty_report() -> dict:
    json_path = _ROOT / "reports" / "novelty_audit.json"
    if not json_path.is_file():
        subprocess.run(
            [sys.executable, "scripts/run_novelty_audit.py"],
            cwd=_ROOT,
            check=True,
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def test_novelty_audit_artifacts_exist() -> None:
    assert (_ROOT / "docs" / "NOVELTY_AUDIT.md").is_file()
    assert (_ROOT / "reports" / "novelty_audit.json").is_file()
    assert (_ROOT / "reports" / "novelty_audit_matrix.csv").is_file()


def test_run_novelty_audit_script() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_novelty_audit.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "phase_id=phaseI_novelty_audit" in proc.stdout


def test_required_prior_art_systems(novelty_report: dict) -> None:
    names = {p["system_name"] for p in novelty_report["prior_art_systems"]}
    for name in REQUIRED_SYSTEMS:
        assert name in names, f"missing prior art: {name}"


def test_vericache_closest_prior_art(novelty_report: dict) -> None:
    assert "VeriCache" in novelty_report.get("closest_prior_art", [])
    vericache = next(
        p for p in novelty_report["prior_art_systems"] if p["system_name"] == "VeriCache"
    )
    assert vericache["category"] == "verifier_mediated_inference_system"
    assert vericache.get("is_closest_conceptual_prior_art") is True


def test_shardcache_classification(novelty_report: dict) -> None:
    shard = next(
        p
        for p in novelty_report["prior_art_systems"]
        if p["system_name"] == "ShardCache (shard-kv)"
    )
    assert shard["category"] in (
        "cache_database_benchmark_system",
        "semantic_cache_benchmark_system",
    )
    assert shard["measures_token_level_divergence"] is False


def test_all_candidate_claims_present(novelty_report: dict) -> None:
    from exactkv.platform.novelty_audit import REQUIRED_CANDIDATE_CLAIMS

    claims = {c["claim"] for c in novelty_report["novelty_claims"]}
    for required in REQUIRED_CANDIDATE_CLAIMS:
        assert required in claims


def test_forbidden_claims_not_allowed(novelty_report: dict) -> None:
    by_claim = {c["claim"]: c["status"] for c in novelty_report["novelty_claims"]}
    for claim in FORBIDDEN_CLAIMS:
        assert by_claim[claim] == "forbidden"


def test_source_pending_not_used_for_uniqueness(novelty_report: dict) -> None:
    first_claim = next(
        c for c in novelty_report["novelty_claims"]
        if "first system" in c["claim"]
    )
    assert first_claim["status"] == "forbidden"
    pending = [
        p
        for p in novelty_report["prior_art_systems"]
        if p.get("evidence_status") == "source_pending"
    ]
    assert pending, "expected some source_pending systems"
    for p in pending:
        assert p.get("is_closest_conceptual_prior_art") is not True


def test_novelty_matrix_csv_has_header() -> None:
    csv_path = _ROOT / "reports" / "novelty_audit_matrix.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("system_name,")
    assert len(lines) > 5
