"""Phase K launch pack artifact tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PHRASES = (
    "first ever",
    "nothing like this exists",
    "production ready",
    "beats vericache",
    "reproduces vericache",
)

LAUNCH_POSTS = (
    "docs/launch_blog_final.md",
    "docs/launch_x_thread_final.md",
    "docs/launch_linkedin_final.md",
)


@pytest.fixture(scope="module")
def ensure_launch_pack() -> None:
    subprocess.run(
        [sys.executable, "scripts/build_launch_pack.py"],
        cwd=_ROOT,
        check=True,
    )


def test_technical_report_exists() -> None:
    assert (_ROOT / "docs/EXACTKV_TECHNICAL_REPORT.md").is_file()


def test_launch_posts_exist() -> None:
    for rel in LAUNCH_POSTS:
        assert (_ROOT / rel).is_file(), rel


def test_demo_cards_exist(ensure_launch_pack: None) -> None:
    release = _ROOT / "reports/public_release"
    assert (release / "demo_cards.json").is_file()
    assert (release / "demo_cards.md").is_file()
    cards = json.loads((release / "demo_cards.json").read_text(encoding="utf-8"))
    assert len(cards.get("demo_cards") or []) >= 5


def test_launch_manifest_exists(ensure_launch_pack: None) -> None:
    manifest = json.loads(
        (_ROOT / "reports/public_release/launch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_of_truth_artifact"] == "reports/scale_7b/raw.json"
    assert manifest["benchmark_cell_count"] == 1500
    assert manifest["exactkv_failures"] == 0
    assert "historical_artifact_inventory.json" in manifest["historical_inventory_path"]
    assert manifest.get("version_arc") == "V1-V21"
    assert manifest.get("version_lineage_path") == "docs/VERSION_LINEAGE.md"
    assert "version_lineage.json" in manifest.get("version_lineage_json", "")
    assert len(manifest.get("remaining_known_limitations") or []) >= 5


def test_technical_report_lineage_and_caveats() -> None:
    text = (_ROOT / "docs/EXACTKV_TECHNICAL_REPORT.md").read_text(encoding="utf-8").lower()
    assert "project lineage" in text
    assert "did not start at phase a" in text
    assert "vericache" in text and "does not reproduce" in text
    assert "not a production serving" in text or "not production serving" in text
    assert "kernel microbenchmark" in text
    assert "fallback" in text and "spectralquant" in text
    assert "probe" in text and "shard" in text


def test_launch_posts_caveats_and_no_forbidden() -> None:
    for rel in LAUNCH_POSTS:
        text = (_ROOT / rel).read_text(encoding="utf-8").lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in text, f"{rel} contains {phrase}"
        assert "1500" in text
        assert "exactkv_failures" in text or "exactkv failures" in text
        assert "vericache" in text
        assert "kernel microbenchmark" in text or "not end-to-end" in text
        assert "fallback" in text or "proxy" in text
        assert "probe" in text


def test_readme_links_launch_docs() -> None:
    text = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
    for needle in (
        "exactkv_technical_report.md",
        "project_lineage.md",
        "version_lineage.md",
        "historical_artifact_inventory.md",
        "novelty_audit.md",
        "claim_boundaries.md",
        "metric_definitions.md",
    ):
        assert needle in text, needle


def test_public_leaderboard_llama_mistral_numeric() -> None:
    lb = json.loads(
        (_ROOT / "reports/public_release/leaderboard_final.json").read_text(encoding="utf-8")
    )
    entries = lb.get("entries") or []
    llama = [e for e in entries if "llama" in str(e.get("model", "")).lower() and e.get("score") is not None]
    mistral = [e for e in entries if "mistral" in str(e.get("model", "")).lower() and e.get("score") is not None]
    assert llama
    assert mistral


def test_check_launch_pack_script() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/check_launch_pack.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_launch_pack_validator_module() -> None:
    from exactkv.platform.launch_pack_validator import validate_launch_pack

    report = validate_launch_pack(_ROOT)
    errors = [i for i in report.issues if i.severity == "error"]
    assert report.valid, [f"{e.check}: {e.detail}" for e in errors]


def test_release_check_includes_launch_pack() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/exactkv_repro.py", "--release-check"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
