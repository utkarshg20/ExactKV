"""ExactKV reporting module (V2 JSON/CSV + V5 workspace-aware memory fields).

Provides stable JSON and CSV report writing/reading for benchmark results.
Every report includes a run manifest with provenance metadata so that results
are self-describing when shared or archived.

Design constraints (V2)
-----------------------
* No timing, latency, throughput, or speedup metrics.
* Memory statistics for simulated compressors (e.g. ``int4_sim``) must include
  a ``memory_claim_note`` and ``supports_real_bytes_claim`` so downstream
  readers cannot misinterpret byte counts as real packed-INT4 savings.
* JSON round-trips losslessly.
* CSV has one row per prompt result; all compressor metadata is flattened.

V5 additive changes
-------------------
* JSON memory sub-dict now includes workspace-aware fields when available:
  ``stored_kv_bytes``, ``materialized_working_kv_bytes``, ``metadata_bytes``,
  ``temporary_workspace_bytes``, ``total_kv_footprint_bytes``.
* CSV adds five corresponding columns (after ``memory_reduction_factor``).
* Legacy V1–V4 reports without these fields are still valid; missing values
  default to 0 in CSV rows and are absent from JSON (no mutation of old files).
* ``total_kv_footprint_bytes`` is a conservative accounting sum, NOT a
  measured peak GPU memory value.  Active GPU measurement is deferred.

Public API
----------
``build_run_manifest(...)``         → dict  manifest for provenance
``write_json_report(report, path)`` → None  write report to JSON file
``load_json_report(path)``          → dict  load and return report
``flatten_report_to_rows(report)``  → list  one flat dict per prompt result
``write_csv_report(report, path)``  → None  write CSV file
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Forbidden performance fields — these must never appear in reports.
# ---------------------------------------------------------------------------
_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})


# ---------------------------------------------------------------------------
# Memory claim notes
# ---------------------------------------------------------------------------

def _memory_claim_note(compressor_name: str, caps: dict[str, Any]) -> str:
    """Return a human-readable note explaining any limitations on byte claims.

    V5 framing: notes distinguish stored bytes, materialized working bytes,
    and accounting totals.  total_kv_footprint_bytes is a conservative
    accounting sum, NOT a measured peak GPU memory value.

    Args:
        compressor_name: The compressor's registry name.
        caps:            Compressor capabilities dict.

    Returns:
        A non-empty note when the byte numbers require qualification,
        or a brief honesty note even for real-bytes compressors.
    """
    if not caps.get("supports_real_bytes_claim", True):
        sim_note = (
            f"'{compressor_name}' is simulated: sub-INT8 values are stored in "
            "int8 containers — stored_kv_bytes reflects int8 container reality, "
            "not packed sub-INT8 savings. "
        )
    else:
        sim_note = f"'{compressor_name}' uses real storage (no simulation). "

    working_note = (
        "materialized_working_kv_bytes == full_kv_bytes for all current "
        "ExactKV compressors (dequantisation produces a full-precision working copy). "
    )
    total_note = (
        "total_kv_footprint_bytes is a conservative accounting sum "
        "(stored + materialized + metadata + temporary); "
        "it is NOT a measured peak GPU memory value. "
        "Active GPU memory measurement is deferred to a later CUDA-specific phase."
    )
    return sim_note + working_note + total_note


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def _try_git_commit() -> str | None:
    """Return the current HEAD commit SHA, or None if git is unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        return sha or None
    except Exception:
        return None


def _try_version(package: str) -> str | None:
    """Return the installed version of ``package``, or None."""
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None


def build_run_manifest(
    model_name: str,
    prompt_suite: str = "smoke",
    compressor_names: list[str] | None = None,
    draft_len: int | None = None,
    draft_lengths: list[int] | None = None,
    max_new_tokens: int = 32,
    seed: int = 0,
    dtype: str = "float32",
    device: str = "auto",
) -> dict[str, Any]:
    """Build a provenance manifest for a benchmark run.

    Args:
        model_name:       HuggingFace model identifier.
        prompt_suite:     Name of the prompt suite used (e.g. ``"smoke"``).
        compressor_names: List of compressor names used in this run.
        draft_len:        Single draft length (used when ``draft_lengths`` is None).
        draft_lengths:    List of draft lengths for sweep runs.
        max_new_tokens:   Token generation budget per prompt.
        seed:             Random seed.
        dtype:            Model dtype string (e.g. ``"float32"``).
        device:           Device string (e.g. ``"cpu"``).

    Returns:
        A dict suitable for inclusion in a JSON report under ``"manifest"``.

    Note: timing fields (``runtime_seconds``, ``tokens_per_second``, etc.) are
    deliberately excluded.  This manifest is provenance-only.
    """
    draft_info: list[int] | int | None
    if draft_lengths is not None:
        draft_info = draft_lengths
    elif draft_len is not None:
        draft_info = [draft_len]
    else:
        draft_info = None

    try:
        import torch
        torch_ver: str | None = torch.__version__
    except Exception:
        torch_ver = None

    try:
        import transformers
        transformers_ver: str | None = transformers.__version__
    except Exception:
        transformers_ver = None

    return {
        "model_name": model_name,
        "prompt_suite": prompt_suite,
        "compressor_names": compressor_names or [],
        "draft_lengths": draft_info,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "dtype": dtype,
        "device": device,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _try_git_commit(),
        "exactkv_version": _try_version("exactkv"),
        "transformers_version": transformers_ver,
        "torch_version": torch_ver,
    }


# ---------------------------------------------------------------------------
# Per-result enrichment
# ---------------------------------------------------------------------------

def _enrich_result(result: dict[str, Any]) -> dict[str, Any]:
    """Add ``compressor_capabilities`` and enriched memory fields to a result dict.

    ``result`` is the dict returned by ``runner.run_one``.  We add:
    * ``compressor_capabilities`` (copied from result if present, else minimal stub)
    * An enriched ``memory`` sub-dict with V1–V4 honesty fields and V5
      workspace-aware fields.

    V5 workspace-aware fields (all present when result comes from Phase-A runner):
        stored_kv_bytes, materialized_working_kv_bytes, metadata_bytes,
        temporary_workspace_bytes, total_kv_footprint_bytes.

    The original ``memory`` keys (``full_bytes``, ``compressed_bytes``, etc.) are
    preserved for backward compatibility.  Legacy results without V5 fields are
    handled gracefully — missing fields default to 0.

    Note: total_kv_footprint_bytes is a conservative accounting total, NOT a
    measured peak GPU memory value.
    """
    enriched = dict(result)

    caps: dict[str, Any] = enriched.get("compressor_capabilities") or {}
    compressor_name: str = enriched.get("compressor_name", "")

    # Enrich memory section
    mem = dict(enriched.get("memory", {}))

    # V1–V4 backward-compat aliases
    mem["full_kv_bytes"] = mem.get("full_kv_bytes") or mem.get("full_bytes", 0)
    mem["compressed_kv_bytes"] = (
        mem.get("compressed_kv_bytes") or mem.get("compressed_bytes", 0)
    )

    # Honesty fields: prefer values already in mem (from MemorySummary.to_dict()),
    # fall back to capabilities dict for older results that predate Phase A.
    if "supports_real_bytes_claim" not in mem:
        mem["supports_real_bytes_claim"] = caps.get("supports_real_bytes_claim", True)
    if "is_simulated" not in mem:
        mem["is_simulated"] = caps.get("is_simulated", False)

    # memory_claim_note: prefer the richer note from MemorySummary (Phase A);
    # fall back to the generated note for older results.
    if not mem.get("memory_claim_note"):
        mem["memory_claim_note"] = _memory_claim_note(compressor_name, caps)

    # V5 workspace-aware fields: pass through if present (Phase A); default to 0
    # for legacy results so CSV/JSON consumers always see a numeric value.
    for field in (
        "stored_kv_bytes",
        "materialized_working_kv_bytes",
        "metadata_bytes",
        "temporary_workspace_bytes",
        "total_kv_footprint_bytes",
    ):
        if field not in mem:
            mem[field] = 0

    enriched["memory"] = mem
    return enriched


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def write_json_report(
    report: dict[str, Any],
    path: str | Path,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Write a benchmark report to a JSON file.

    Args:
        report:   Dict returned by ``runner.run_suite`` (or ``run_one``).
                  Must contain ``"results"`` and optionally ``"aggregate"``.
        path:     Destination file path.  Parent directories must exist.
        manifest: Optional provenance manifest from ``build_run_manifest``.
                  If provided, written under ``report["manifest"]``.

    The written JSON is UTF-8, human-readable (indent=2), and contains only
    JSON-native types (no datetimes, tensors, or numpy arrays).

    Performance fields (``tokens_per_second`` etc.) are prohibited and will
    raise ``ValueError`` if found in ``report``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate no forbidden fields are present anywhere in the report
    _assert_no_forbidden_fields(report)

    results = report.get("results", [])
    enriched_results = [_enrich_result(r) for r in results]

    output: dict[str, Any] = {}
    if manifest is not None:
        output["manifest"] = manifest
    output["results"] = enriched_results
    if "aggregate" in report:
        output["aggregate"] = report["aggregate"]

    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report written by ``write_json_report``.

    Args:
        path: Path to a JSON report file.

    Returns:
        The deserialized report dict.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

# Ordered list of CSV column names.  All columns must be present in every row
# (missing values are written as empty string).
_CSV_COLUMNS: list[str] = [
    "prompt_id",
    "category",
    "model_name",
    "compressor_name",
    "compressor_type",
    "is_simulated",
    "supports_real_bytes_claim",
    "key_bit_width",
    "value_bit_width",
    "asymmetric",
    "draft_len",
    "max_new_tokens",
    "exactkv_token_exact_match",
    "lossy_token_exact_match",
    "lossy_first_divergence_idx",
    "acceptance_rate",
    "average_accepted_length",
    "total_drafted",
    "total_accepted",
    "total_rejected",
    "total_corrections",
    "full_kv_bytes",
    "compressed_kv_bytes",
    "compression_ratio",
    "memory_reduction_factor",
    # V5 workspace-aware memory fields (additive; 0 for legacy results)
    "stored_kv_bytes",
    "materialized_working_kv_bytes",
    "metadata_bytes",
    "temporary_workspace_bytes",
    "total_kv_footprint_bytes",
    "memory_claim_note",
]


def _result_to_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one enriched result dict into a single CSV row dict."""
    enriched = _enrich_result(result)
    caps: dict[str, Any] = enriched.get("compressor_capabilities") or {}
    mem: dict[str, Any] = enriched.get("memory", {})
    acc: dict[str, Any] = enriched.get("exactkv", {}).get("acceptance", {})
    lossy: dict[str, Any] = enriched.get("lossy", {})

    return {
        "prompt_id": enriched.get("prompt_id", ""),
        "category": enriched.get("category", ""),
        "model_name": enriched.get("model_name", ""),
        "compressor_name": enriched.get("compressor_name", ""),
        "compressor_type": caps.get("compressor_type", ""),
        "is_simulated": caps.get("is_simulated", ""),
        "supports_real_bytes_claim": caps.get("supports_real_bytes_claim", ""),
        "key_bit_width": caps.get("key_bit_width", ""),
        "value_bit_width": caps.get("value_bit_width", ""),
        "asymmetric": caps.get("asymmetric", ""),
        "draft_len": enriched.get("draft_len", ""),
        "max_new_tokens": enriched.get("max_new_tokens", ""),
        "exactkv_token_exact_match": enriched.get("exactkv", {}).get("token_exact_match", ""),
        "lossy_token_exact_match": lossy.get("token_exact_match", ""),
        "lossy_first_divergence_idx": lossy.get("first_divergence_idx", ""),
        "acceptance_rate": acc.get("acceptance_rate", ""),
        "average_accepted_length": acc.get("avg_accepted_per_round", ""),
        "total_drafted": acc.get("total_drafted", ""),
        "total_accepted": acc.get("total_accepted", ""),
        "total_rejected": acc.get("total_rejected", ""),
        "total_corrections": acc.get("total_corrections", ""),
        "full_kv_bytes": mem.get("full_kv_bytes", mem.get("full_bytes", "")),
        "compressed_kv_bytes": mem.get("compressed_kv_bytes", mem.get("compressed_bytes", "")),
        "compression_ratio": mem.get("compression_ratio", ""),
        "memory_reduction_factor": mem.get("memory_reduction_factor", ""),
        # V5 workspace-aware memory fields (0 for legacy results without Phase A)
        "stored_kv_bytes": mem.get("stored_kv_bytes", 0),
        "materialized_working_kv_bytes": mem.get("materialized_working_kv_bytes", 0),
        "metadata_bytes": mem.get("metadata_bytes", 0),
        "temporary_workspace_bytes": mem.get("temporary_workspace_bytes", 0),
        "total_kv_footprint_bytes": mem.get("total_kv_footprint_bytes", 0),
        "memory_claim_note": mem.get("memory_claim_note", ""),
    }


def flatten_report_to_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a benchmark report into a flat list of CSV row dicts.

    Args:
        report: Dict returned by ``runner.run_suite`` (with ``"results"`` key).

    Returns:
        One dict per prompt result, with keys matching ``_CSV_COLUMNS``.
    """
    return [_result_to_csv_row(r) for r in report.get("results", [])]


def write_csv_report(
    report: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a benchmark report to a CSV file.

    One row per prompt result.  Columns follow ``_CSV_COLUMNS``.  Missing
    values are written as empty strings.

    Performance fields (``tokens_per_second`` etc.) are absent by design.

    Args:
        report: Dict returned by ``runner.run_suite``.
        path:   Destination file path.  Parent directories must exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_forbidden_fields(report)

    rows = flatten_report_to_rows(report)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_CSV_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _assert_no_forbidden_fields(obj: Any, _path: str = "root") -> None:
    """Recursively assert that no forbidden performance field names appear in ``obj``.

    Raises:
        ValueError: If a forbidden key is found anywhere in the report.
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in _FORBIDDEN_FIELDS:
                raise ValueError(
                    f"Forbidden performance field {key!r} found at {_path}. "
                    "ExactKV V2 does not report timing or throughput metrics."
                )
            _assert_no_forbidden_fields(val, _path=f"{_path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, _path=f"{_path}[{i}]")


def validate_report(report: dict[str, Any]) -> list[str]:
    """Return a list of validation warnings for a report dict.

    Does not raise; callers can decide whether warnings are fatal.

    Checks:
    * No forbidden performance fields.
    * Each result has required top-level keys.
    * Each result's memory section has the V1–V4 honesty fields.
    * V5 workspace fields, when present, are non-negative and the total reconciles.

    Backward compatibility: V1–V4 reports without V5 workspace fields pass
    validation without warnings.  The new fields are optional.
    """
    warnings: list[str] = []
    required_result_keys = {
        "prompt_id", "compressor_name", "full", "lossy", "exactkv", "memory",
    }
    required_memory_keys = {
        "full_kv_bytes", "compressed_kv_bytes", "compression_ratio",
        "memory_reduction_factor", "memory_claim_note", "supports_real_bytes_claim",
        "is_simulated",
    }
    # V5 workspace fields: optional but validated when present.
    v5_workspace_fields = (
        "stored_kv_bytes",
        "materialized_working_kv_bytes",
        "metadata_bytes",
        "temporary_workspace_bytes",
        "total_kv_footprint_bytes",
    )

    try:
        _assert_no_forbidden_fields(report)
    except ValueError as exc:
        warnings.append(str(exc))

    for i, result in enumerate(report.get("results", [])):
        for key in required_result_keys:
            if key not in result:
                warnings.append(f"Result[{i}] missing required key {key!r}")
        mem = result.get("memory", {})
        for key in required_memory_keys:
            if key not in mem:
                warnings.append(
                    f"Result[{i}].memory missing honesty key {key!r}"
                )
        # V5 optional checks: only validate when fields are present.
        v5_present = all(f in mem for f in v5_workspace_fields)
        if v5_present:
            for f in v5_workspace_fields:
                val = mem[f]
                if not isinstance(val, (int, float)) or val < 0:
                    warnings.append(
                        f"Result[{i}].memory.{f} must be a non-negative number, "
                        f"got {val!r}"
                    )
            # Reconciliation: total == stored + materialized + metadata + temporary
            expected_total = (
                mem.get("stored_kv_bytes", 0)
                + mem.get("materialized_working_kv_bytes", 0)
                + mem.get("metadata_bytes", 0)
                + mem.get("temporary_workspace_bytes", 0)
            )
            actual_total = mem.get("total_kv_footprint_bytes", 0)
            if actual_total != expected_total:
                warnings.append(
                    f"Result[{i}].memory.total_kv_footprint_bytes ({actual_total}) "
                    f"!= stored+materialized+metadata+temporary ({expected_total})"
                )

    return warnings
