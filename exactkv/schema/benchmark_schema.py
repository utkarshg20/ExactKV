"""Canonical benchmark data schema (Phase H)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BenchmarkCell:
    """Canonical per-cell benchmark record."""

    model: str
    compressor: str
    prompt_id: str
    max_new_tokens: int
    compression_ratio: float | None
    first_divergence: int | None
    acceptance_rate: float
    avg_accepted_span: float | None
    exactkv_failure: bool
    verifier_score: float | None
    latency_ms: float | None = None
    divergence_type: str | None = None
    backend_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_phase_a_cell(cls, cell: Mapping[str, Any]) -> BenchmarkCell:
        metrics = cell.get("metrics") or {}
        acceptance = (cell.get("exactkv") or {}).get("acceptance") or {}
        total_accepted = acceptance.get("total_accepted")
        total_rounds = acceptance.get("total_rounds")
        avg_span = None
        if total_accepted is not None and total_rounds and total_rounds > 0:
            avg_span = float(total_accepted) / float(total_rounds)
        return cls(
            model=str(cell.get("model_name") or ""),
            compressor=str(cell.get("compressor_name") or ""),
            prompt_id=str(cell.get("prompt_id") or ""),
            max_new_tokens=int(cell.get("max_new_tokens") or 0),
            compression_ratio=metrics.get("compression_ratio"),
            first_divergence=metrics.get("first_divergence_index"),
            acceptance_rate=float(metrics.get("acceptance_rate") or 0.0),
            avg_accepted_span=avg_span,
            exactkv_failure=bool(cell.get("exactkv_failure") or metrics.get("exactkv_failure")),
            verifier_score=metrics.get("verifier_agreement_score"),
            latency_ms=cell.get("latency_ms"),
            divergence_type=metrics.get("divergence_type"),
            backend_tier=cell.get("backend_tier"),
        )


@dataclass
class BenchmarkConfig:
    """Hash-stable benchmark configuration."""

    models: tuple[str, ...]
    compressors: tuple[str, ...]
    prompt_ids: tuple[str, ...]
    max_new_tokens_values: tuple[int, ...]
    device: str = "cpu"
    dtype: str = "float32"
    deterministic_mode: bool = False
    draft_len: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "compressors": list(self.compressors),
            "prompt_ids": list(self.prompt_ids),
            "max_new_tokens_values": list(self.max_new_tokens_values),
            "device": self.device,
            "dtype": self.dtype,
            "deterministic_mode": self.deterministic_mode,
            "draft_len": self.draft_len,
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class BenchmarkRun:
    """One full matrix execution — hash-stable metadata."""

    run_id: str
    config: BenchmarkConfig
    config_hash: str
    git_commit: str | None
    cells: list[BenchmarkCell] = field(default_factory=list)
    status: str = "pending"
    exactkv_failure_rate: float = 0.0
    total_cells: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "status": self.status,
            "total_cells": self.total_cells,
            "exactkv_failure_rate": self.exactkv_failure_rate,
            "cells": [c.to_dict() for c in self.cells],
        }


def resolve_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def cells_from_phase_a_report(report: Mapping[str, Any]) -> list[BenchmarkCell]:
    raw = report.get("cells") or []
    return [BenchmarkCell.from_phase_a_cell(c) for c in raw]


def aggregate_failure_rate(cells: Sequence[BenchmarkCell]) -> float:
    if not cells:
        return 0.0
    return sum(1 for c in cells if c.exactkv_failure) / len(cells)
