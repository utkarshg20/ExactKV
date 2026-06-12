"""Tests for the ExactKV terminal-native crash-test demo (Phase 8e)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "exactkv_terminal_crash_test.py"


def _run_demo(*extra: str) -> str:
    cmd = [sys.executable, str(_SCRIPT), "--no-delay", "--plain", *extra]
    result = subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.mark.parametrize(
    "needle",
    [
        "EXACTKV CRASH TEST",
        "Everyone is racing to shrink KV caches",
        "DRIFT DETECTED",
        "draft token",
        "verifier token",
        "REJECTED",
        "COMMITTED",
        "ExactKV failures",
        "Final output match",
        "KV compression should not be trusted",
    ],
)
def test_terminal_crash_test_required_strings(needle: str) -> None:
    out = _run_demo()
    assert needle in out


def test_terminal_demo_uses_semantic_trace_by_default() -> None:
    out = _run_demo()
    assert "pharm_001" in out or "dropoff" in out.lower() or "pickup" in out.lower()
    assert "ExactKV failures          0" in out
    assert "Final output match        TRUE" in out


def test_terminal_demo_exp034_json_source() -> None:
    exp034 = _ROOT / "reports" / "experiment_034_killer_correction_demo.json"
    if not exp034.is_file():
        pytest.skip("Exp 034 JSON not present locally")
    out = _run_demo("--source-json", str(exp034))
    assert "int4_sim" in out
    assert "get_weather" in out
    assert "metric" in out


def test_terminal_demo_fallback_without_json(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    out = _run_demo("--source-json", str(missing))
    assert "EXACTKV CRASH TEST" in out
    assert "ExactKV failures          0" in out


def test_ansi_fallback_without_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "rich" or name.startswith("rich."):
            raise ImportError("rich blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    out = _run_demo()
    assert "DRIFT DETECTED" in out
