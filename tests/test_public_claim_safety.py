"""Tests for public claim safety audit (Gate R0 + Phase I)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_FILES = (
    "README.md",
    "reports/public_release/README_PUBLIC.md",
    "reports/public_release/methodology.md",
    "docs/blog_post.md",
    "docs/x_thread.md",
    "docs/linkedin_post.md",
    "docs/paper_draft.md",
    "docs/launch_blog_final.md",
    "docs/launch_x_thread_final.md",
    "docs/launch_linkedin_final.md",
    "docs/EXACTKV_TECHNICAL_REPORT.md",
)

FORBIDDEN_PHRASES = (
    "nothing like this exists",
    "first ever",
    "production ready",
)

REQUIRED_POSITIONING_SNIPPET = "compressor-agnostic crash-test"


def test_audit_public_claims_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/audit_public_claims.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED" in proc.stdout


def test_forbidden_claim_detector() -> None:
    import tempfile

    from scripts.audit_public_claims import scan_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as fh:
        fh.write("ExactKV is production ready for serving system deployment.\n")
        tmp_path = Path(fh.name)
    try:
        hits = scan_file(tmp_path)
        assert hits
    finally:
        tmp_path.unlink(missing_ok=True)


def test_novelty_audit_artifacts() -> None:
    assert (_ROOT / "docs" / "NOVELTY_AUDIT.md").is_file()
    assert (_ROOT / "reports" / "novelty_audit.json").is_file()
    assert (_ROOT / "reports" / "novelty_audit_matrix.csv").is_file()


def test_forbidden_phrases_absent_in_public_files() -> None:
    for rel in PUBLIC_FILES:
        path = _ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in text, f"{rel} contains forbidden phrase: {phrase}"


def test_active_gpu_memory_savings_not_claimed() -> None:
    negation = re.compile(r"\b(not|no|never|without)\b", re.I)
    phrase = re.compile(r"active\s+gpu\s+memory\s+savings", re.I)
    for rel in PUBLIC_FILES:
        path = _ROOT / rel
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not phrase.search(line):
                continue
            match = phrase.search(line)
            assert match is not None
            before = line[: match.start()]
            assert negation.search(before[-80:]), (
                f"{rel} claims active GPU memory savings without negation: {line.strip()[:100]}"
            )


def test_required_positioning_in_key_public_docs() -> None:
    for rel in (
        "docs/launch_blog_final.md",
        "reports/public_release/README_PUBLIC.md",
    ):
        text = (_ROOT / rel).read_text(encoding="utf-8").lower()
        assert REQUIRED_POSITIONING_SNIPPET in text


def test_methodology_has_phase_i_caveats() -> None:
    text = (_ROOT / "reports/public_release/methodology.md").read_text(encoding="utf-8").lower()
    assert "kernel microbenchmark" in text
    assert "stored tensor byte" in text
    assert "fallback" in text
    assert "probe-first" in text
    assert "not reproduce vericache" in text


def test_novelty_json_forbidden_claims() -> None:
    report = json.loads(
        (_ROOT / "reports" / "novelty_audit.json").read_text(encoding="utf-8")
    )
    by_claim = {c["claim"]: c["status"] for c in report["novelty_claims"]}
    assert by_claim["ExactKV is the first system like this."] == "forbidden"
    assert by_claim["ExactKV reproduces VeriCache."] == "forbidden"
