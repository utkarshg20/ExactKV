"""Tests for exactkv.cli (V2 Phase F).

Strategy:
  * Call main(argv) directly (no subprocess) to keep tests in-process and fast.
  * Capture stdout with capsys where content checks are needed.
  * Use --suite-file with a 1-prompt JSONL for bench/sweep to avoid running
    the full 16-prompt smoke suite in tests.
  * Compressor validation is tested without loading the model (validation
    happens before model load).
  * The analyze command is tested on a JSON file produced by a prior bench call.
  * All model-dependent tests use scope="module" to share a single model load.

Gates:
  list-compressors gate  — all four built-ins appear in output
  bench CLI gate         — writes valid JSON, exactkv_failures == 0
  sweep CLI gate         — writes valid JSON, rows == prompts×compressors×draft_lens
  analyze CLI gate       — reads report, writes acceptance CSV + failure JSON
  no-forbidden-fields    — no performance field in any output
"""
from __future__ import annotations

import csv
import json
import os

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}

# ---------------------------------------------------------------------------
# Tiny 1-prompt suite file (created once per module)
# ---------------------------------------------------------------------------

_MINI_PROMPT = {
    "prompt_id": "cli_001",
    "category": "test",
    "prompt": "The capital of France is",
}


@pytest.fixture(scope="module")
def mini_suite(tmp_path_factory):
    """Write a 1-prompt JSONL file; return its path."""
    p = tmp_path_factory.mktemp("cli_suite") / "mini.jsonl"
    p.write_text(json.dumps(_MINI_PROMPT) + "\n", encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# list-compressors (model-free)
# ---------------------------------------------------------------------------

def test_list_compressors_returns_zero():
    from exactkv.cli import main
    assert main(["list-compressors"]) == 0


def test_list_compressors_includes_all_builtins(capsys):
    from exactkv.cli import main
    main(["list-compressors"])
    out = capsys.readouterr().out
    for name in ("noop", "int8", "int4_sim", "debug_noise"):
        assert name in out, f"Compressor {name!r} missing from list-compressors output"


def test_list_compressors_includes_capabilities_fields(capsys):
    from exactkv.cli import main
    main(["list-compressors"])
    out = capsys.readouterr().out
    for field in ("is_simulated", "supports_real_bytes_claim",
                  "supports_quantization", "compressor_type"):
        assert field in out, f"Field {field!r} missing from list-compressors output"


def test_list_compressors_no_forbidden_fields(capsys):
    from exactkv.cli import main
    main(["list-compressors"])
    out = capsys.readouterr().out
    for f in _FORBIDDEN:
        assert f not in out, f"Forbidden field {f!r} in list-compressors output"


# ---------------------------------------------------------------------------
# bench (model-dependent, module-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bench_json(tmp_path_factory, mini_suite):
    """Run bench CLI once; return (json_path, csv_path)."""
    from exactkv.cli import main

    out_dir = tmp_path_factory.mktemp("bench_out")
    json_path = str(out_dir / "bench.json")
    csv_path = str(out_dir / "bench.csv")

    rc = main([
        "bench",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite,
        "--compressor", "noop",
        "--draft-len", "4",
        "--max-new-tokens", "8",
        "--json-out", json_path,
        "--csv-out", csv_path,
    ])
    assert rc == 0, f"bench CLI returned non-zero: {rc}"
    return json_path, csv_path


def test_bench_writes_json(bench_json):
    json_path, _ = bench_json
    assert os.path.exists(json_path), "bench did not write JSON"


def test_bench_writes_csv(bench_json):
    _, csv_path = bench_json
    assert os.path.exists(csv_path), "bench did not write CSV"


def test_bench_json_valid(bench_json):
    json_path, _ = bench_json
    data = json.loads(open(json_path).read())
    assert "results" in data


def test_bench_json_exactkv_failures_zero(bench_json):
    json_path, _ = bench_json
    data = json.loads(open(json_path).read())
    failures = sum(1 for r in data["results"] if r.get("exactkv_failure", False))
    assert failures == 0, f"Expected 0 ExactKV failures, got {failures}"


def test_bench_json_no_forbidden_fields(bench_json):
    json_path, _ = bench_json
    text = open(json_path).read()
    for field in _FORBIDDEN:
        assert f'"{field}"' not in text, f"Forbidden field {field!r} in bench JSON"


def test_bench_csv_no_forbidden_columns(bench_json):
    _, csv_path = bench_json
    with open(csv_path, encoding="utf-8") as f:
        headers = set(csv.DictReader(f).fieldnames or [])
    assert not (headers & _FORBIDDEN), f"Forbidden columns in bench CSV: {headers & _FORBIDDEN}"


# ---------------------------------------------------------------------------
# bench — invalid compressor (model-free: validation before model load)
# ---------------------------------------------------------------------------

def test_bench_invalid_compressor_returns_nonzero(mini_suite, tmp_path):
    from exactkv.cli import main
    rc = main([
        "bench",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite,
        "--compressor", "nonexistent_xyz_compressor",
        "--max-new-tokens", "4",
    ])
    assert rc != 0, "bench should return non-zero for unknown compressor"


# ---------------------------------------------------------------------------
# sweep (model-dependent, module-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep_json(tmp_path_factory, mini_suite):
    """Run sweep CLI once (1 prompt × 2 compressors × 1 draft_len = 2 cells)."""
    from exactkv.cli import main

    out_dir = tmp_path_factory.mktemp("sweep_out")
    json_path = str(out_dir / "sweep.json")
    csv_path = str(out_dir / "sweep.csv")

    rc = main([
        "sweep",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite,
        "--compressors", "noop,int8",
        "--draft-lengths", "4",
        "--max-new-tokens", "8",
        "--json-out", json_path,
        "--csv-out", csv_path,
    ])
    assert rc == 0, f"sweep CLI returned non-zero: {rc}"
    return json_path, csv_path


def test_sweep_writes_json(sweep_json):
    json_path, _ = sweep_json
    assert os.path.exists(json_path)


def test_sweep_writes_csv(sweep_json):
    _, csv_path = sweep_json
    assert os.path.exists(csv_path)


def test_sweep_json_valid(sweep_json):
    json_path, _ = sweep_json
    data = json.loads(open(json_path).read())
    assert "results" in data
    assert "aggregate" in data


def test_sweep_json_row_count(sweep_json):
    """1 prompt × 2 compressors × 1 draft_len = 2 results."""
    json_path, _ = sweep_json
    data = json.loads(open(json_path).read())
    assert len(data["results"]) == 2, (
        f"Expected 2 results, got {len(data['results'])}"
    )


def test_sweep_json_exactkv_failures_zero(sweep_json):
    json_path, _ = sweep_json
    data = json.loads(open(json_path).read())
    failures = data["aggregate"]["exactkv_failures"]
    assert failures == 0, f"Expected 0 ExactKV failures in sweep, got {failures}"


def test_sweep_csv_rows_match_results(sweep_json):
    json_path, csv_path = sweep_json
    data = json.loads(open(json_path).read())
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(data["results"])


def test_sweep_json_no_forbidden_fields(sweep_json):
    json_path, _ = sweep_json
    text = open(json_path).read()
    for field in _FORBIDDEN:
        assert f'"{field}"' not in text, f"Forbidden field {field!r} in sweep JSON"


def test_sweep_csv_no_forbidden_columns(sweep_json):
    _, csv_path = sweep_json
    with open(csv_path, encoding="utf-8") as f:
        headers = set(csv.DictReader(f).fieldnames or [])
    assert not (headers & _FORBIDDEN)


# ---------------------------------------------------------------------------
# sweep — invalid compressor (model-free)
# ---------------------------------------------------------------------------

def test_sweep_invalid_compressor_returns_nonzero(mini_suite):
    from exactkv.cli import main
    rc = main([
        "sweep",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite,
        "--compressors", "noop,bad_compressor_xyz",
        "--draft-lengths", "4",
        "--max-new-tokens", "4",
    ])
    assert rc != 0, "sweep should return non-zero for unknown compressor"


# ---------------------------------------------------------------------------
# analyze (model-free — reads the sweep JSON produced above)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyze_outputs(tmp_path_factory, sweep_json):
    """Run analyze on the sweep JSON; return (acceptance_csv, failure_json)."""
    from exactkv.cli import main

    json_path, _ = sweep_json
    out_dir = tmp_path_factory.mktemp("analyze_out")
    acc_csv = str(out_dir / "acceptance.csv")
    fail_json = str(out_dir / "failure.json")

    rc = main([
        "analyze",
        "--report", json_path,
        "--acceptance-csv", acc_csv,
        "--failure-json", fail_json,
    ])
    assert rc == 0, f"analyze CLI returned non-zero: {rc}"
    return acc_csv, fail_json


def test_analyze_writes_acceptance_csv(analyze_outputs):
    acc_csv, _ = analyze_outputs
    assert os.path.exists(acc_csv)


def test_analyze_writes_failure_json(analyze_outputs):
    _, fail_json = analyze_outputs
    assert os.path.exists(fail_json)


def test_analyze_failure_json_status_pass(analyze_outputs):
    _, fail_json = analyze_outputs
    data = json.loads(open(fail_json).read())
    assert data["status"] == "pass", f"Expected status=pass, got {data['status']}"


def test_analyze_failure_json_exactkv_count_zero(analyze_outputs):
    _, fail_json = analyze_outputs
    data = json.loads(open(fail_json).read())
    assert data["exactkv_failure_count"] == 0


def test_analyze_acceptance_csv_has_rows(analyze_outputs, sweep_json):
    """Acceptance CSV should have one row per (compressor, draft_len) group."""
    acc_csv, _ = analyze_outputs
    with open(acc_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # 2 compressors × 1 draft_len = 2 rows
    assert len(rows) == 2


def test_analyze_missing_report_returns_nonzero():
    from exactkv.cli import main
    rc = main([
        "analyze",
        "--report", "/nonexistent/path/report.json",
    ])
    assert rc != 0, "analyze should return non-zero for missing report file"


def test_analyze_failure_json_no_forbidden_fields(analyze_outputs):
    _, fail_json = analyze_outputs
    text = open(fail_json).read()
    for field in _FORBIDDEN:
        assert f'"{field}"' not in text, f"Forbidden field {field!r} in failure JSON"


# ---------------------------------------------------------------------------
# No-command prints help and returns non-zero
# ---------------------------------------------------------------------------

def test_no_command_returns_nonzero():
    from exactkv.cli import main
    rc = main([])
    assert rc != 0


# ---------------------------------------------------------------------------
# python -m exactkv smoke test (subprocess)
# ---------------------------------------------------------------------------

def test_python_m_exactkv_list_compressors():
    """Verify `python -m exactkv list-compressors` works as a subprocess."""
    import subprocess
    result = subprocess.run(
        ["python3", "-m", "exactkv", "list-compressors"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        env={**os.environ, "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    for name in ("noop", "int8", "int4_sim", "debug_noise"):
        assert name in result.stdout, f"Compressor {name!r} not in subprocess output"


from pathlib import Path


# ---------------------------------------------------------------------------
# report command (V3 Phase D)
# ---------------------------------------------------------------------------

# Required Markdown section headings (lower-cased match)
_MD_REQUIRED_SECTIONS = [
    "experiment summary",
    "correctness",
    "acceptance",
    "leaderboard",
    "histogram",
    "what this report proves",
    "what this report does not prove",
]

# Patterns that must NOT appear as data fields in the Markdown
_MD_FORBIDDEN_DATA_FIELD_PATTERNS = [
    "| tokens_per_second",
    "| throughput |",
    "| latency |",
    "| speedup |",
    "| runtime_seconds",
    "tokens_per_second:",
    "throughput:",
    "latency:",
    "speedup:",
    "runtime_seconds:",
]


@pytest.fixture(scope="module")
def report_json(tmp_path_factory, mini_suite):
    """Run a small sweep and write a JSON report for report-command tests."""
    out = tmp_path_factory.mktemp("report_cmd") / "report.json"
    from exactkv.cli import main
    rc = main([
        "sweep",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite,
        "--compressors", "noop,int8",
        "--draft-lengths", "4",
        "--max-new-tokens", "8",
        "--json-out", str(out),
    ])
    assert rc == 0, "sweep fixture failed"
    return out


class TestReportCommand:
    def test_report_writes_markdown(self, report_json, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "out.md"
        rc = main(["report", "--report", str(report_json),
                   "--markdown-out", str(out_md)])
        assert rc == 0
        assert out_md.exists()
        assert out_md.stat().st_size > 0

    def test_report_creates_parent_directories(self, report_json, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "nested" / "deep" / "report.md"
        rc = main(["report", "--report", str(report_json),
                   "--markdown-out", str(out_md)])
        assert rc == 0
        assert out_md.exists()

    def test_report_custom_title(self, report_json, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "titled.md"
        rc = main(["report", "--report", str(report_json),
                   "--markdown-out", str(out_md),
                   "--title", "My Custom Title"])
        assert rc == 0
        content = out_md.read_text(encoding="utf-8")
        assert "My Custom Title" in content

    def test_report_required_sections_present(self, report_json, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "sections.md"
        main(["report", "--report", str(report_json), "--markdown-out", str(out_md)])
        content = out_md.read_text(encoding="utf-8").lower()
        for section in _MD_REQUIRED_SECTIONS:
            assert section in content, f"Required section missing: {section!r}"

    def test_report_lossy_divergence_described_as_expected(self, report_json, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "wording.md"
        main(["report", "--report", str(report_json), "--markdown-out", str(out_md)])
        content = out_md.read_text(encoding="utf-8").lower()
        assert "lossy divergence is expected" in content

    def test_report_no_forbidden_data_fields(self, report_json, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "forbidden.md"
        main(["report", "--report", str(report_json), "--markdown-out", str(out_md)])
        content = out_md.read_text(encoding="utf-8").lower()
        for pattern in _MD_FORBIDDEN_DATA_FIELD_PATTERNS:
            assert pattern not in content, (
                f"Forbidden data-field pattern {pattern!r} in markdown report"
            )

    def test_report_nonzero_for_missing_file(self, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "out.md"
        rc = main(["report", "--report", str(tmp_path / "missing.json"),
                   "--markdown-out", str(out_md)])
        assert rc != 0

    def test_report_nonzero_for_invalid_json(self, tmp_path):
        from exactkv.cli import main
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json{{", encoding="utf-8")
        out_md = tmp_path / "out.md"
        rc = main(["report", "--report", str(bad_json),
                   "--markdown-out", str(out_md)])
        assert rc != 0

    def test_report_no_examples_flag(self, report_json, tmp_path):
        from exactkv.cli import main
        out_md = tmp_path / "no_ex.md"
        rc = main(["report", "--report", str(report_json),
                   "--markdown-out", str(out_md), "--no-examples"])
        assert rc == 0
        content = out_md.read_text(encoding="utf-8").lower()
        # Lossy divergence examples section should not appear
        assert "lossy divergence examples" not in content

    def test_report_int4_sim_disclaimer_when_int4_present(
        self, tmp_path_factory, mini_suite
    ):
        """When int4_sim is in the report the simulation disclaimer must appear."""
        from exactkv.cli import main
        sweep_json = tmp_path_factory.mktemp("int4_report") / "sweep.json"
        main([
            "sweep", "--model", MODEL_NAME,
            "--suite-file", mini_suite,
            "--compressors", "int4_sim",
            "--draft-lengths", "4",
            "--max-new-tokens", "8",
            "--json-out", str(sweep_json),
        ])
        out_md = tmp_path_factory.mktemp("int4_report") / "out.md"
        rc = main(["report", "--report", str(sweep_json),
                   "--markdown-out", str(out_md)])
        assert rc == 0
        content = out_md.read_text(encoding="utf-8").lower()
        assert "simulated" in content


# ---------------------------------------------------------------------------
# report command in CLI help text
# ---------------------------------------------------------------------------

def test_report_appears_in_help(capsys):
    from exactkv.cli import main
    try:
        main(["--help"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "report" in captured.out
