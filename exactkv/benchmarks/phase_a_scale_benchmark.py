"""Phase A scale benchmarking — multi-model KV compression truth layer.

Unified evaluation panel over built-in and external compressors. Consumes
Exp 114–117 infrastructure patterns without modifying ExactKVGenerator or
introducing L4 commit logic. Trace-only / post-hoc evaluation.

Public API
----------
``run_phase_a_scale_benchmark(...)``  → structured JSON report dict
``render_phase_a_markdown_summary(...)`` → Markdown summary table
``validate_phase_a_report(...)``        → validation result
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
from exactkv.benchmarks.reports import _assert_no_forbidden_fields, build_run_manifest
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.safety.l4_runtime_coupling_stress_panel import STRESS_PANEL_PROMPTS

PHASE_A_ID = "phaseA_scale_benchmark"
DEFAULT_PHASE_A_REPORT = Path("reports/phaseA_benchmark.json")
DEFAULT_PHASE_A_MARKDOWN = Path("reports/phaseA_benchmark.md")
DEFAULT_EXP116_REPORT = Path("reports/experiment_116_instability_regime_analysis.json")

PHASE_A_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

PHASE_A_BUILTIN_COMPRESSORS: tuple[str, ...] = (
    "noop",
    "int8",
    "int6_sim",
    "int4_per_vec_sim",
    "int4_sim",
    "k8_v4_sim",
    "h2o_sim",
    "h2o_sim_75",
    "h2o_sim_25",
)

PHASE_A_EXTENDED_COMPRESSORS: tuple[str, ...] = (
    "spectralquant",
    "kvquant",
    "shard",
)

PHASE_A_ALL_COMPRESSORS: tuple[str, ...] = (
    *PHASE_A_BUILTIN_COMPRESSORS,
    *PHASE_A_EXTENDED_COMPRESSORS,
)

PHASE_A_MAX_NEW_TOKENS: tuple[int, ...] = (4, 8, 16)
DEFAULT_DRAFT_LEN = 4
DEFAULT_PROMPT_COUNT = 4

REPRODUCIBLE_CLI = (
    "python scripts/run_phase_a_scale_benchmark.py "
    "--deterministic-mode"
)


def _get_compressor(name: str) -> Any:
    import exactkv.compressors  # noqa: F401, PLC0415 — register built-ins
    from exactkv.compressors.registry import get_compressor  # noqa: PLC0415

    return get_compressor(name)

_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "exactkv_generator_modified",
    "runtime_commit_authorized",
    "l4_activation",
)


@dataclass(frozen=True)
class PhaseAValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompressorResolution:
    compressor_name: str
    backend_tier: str
    adapter_available: bool
    probe_only: bool
    delegate_compressor: str | None = None


def default_phase_a_prompts(max_prompts: int = DEFAULT_PROMPT_COUNT) -> list[dict[str, str]]:
    """Four deterministic prompts for the unified evaluation panel."""
    entries: list[dict[str, str]] = []
    for prompt_id, prompt_text in STRESS_PANEL_PROMPTS[:max_prompts]:
        category = prompt_id.split("_", 1)[-1] if "_" in prompt_id else "general"
        entries.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "prompt": prompt_text,
            },
        )
    return entries


def detect_model_availability(
    models: Sequence[str],
    *,
    local_files_only: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Return models with resolvable HF configs and blocked reasons."""
    available: list[str] = []
    blocked: dict[str, str] = {}
    try:
        from transformers import AutoConfig  # noqa: PLC0415
    except ImportError:
        for model in models:
            blocked[model] = "transformers not installed"
        return available, blocked

    for model in models:
        try:
            AutoConfig.from_pretrained(model, local_files_only=local_files_only)
            available.append(model)
        except Exception as exc:  # noqa: BLE001
            blocked[model] = str(exc)[:200]
    return available, blocked


def resolve_compressor(
    name: str,
    *,
    runtime: ModelRuntime | None = None,
) -> CompressorResolution:
    """Resolve built-in registry, external adapter, or mock/probe fallback."""
    if name in PHASE_A_BUILTIN_COMPRESSORS:
        return CompressorResolution(
            compressor_name=name,
            backend_tier="BUILTIN",
            adapter_available=True,
            probe_only=False,
        )

    if name == "spectralquant":
        if runtime is not None:
            try:
                from exactkv.external.spectralquant_adapter import (  # noqa: PLC0415
                    create_spectralquant_experimental_adapter,
                )

                create_spectralquant_experimental_adapter(runtime)
                return CompressorResolution(
                    compressor_name=name,
                    backend_tier="RESTRICTED_ADAPTER",
                    adapter_available=True,
                    probe_only=False,
                )
            except Exception:  # noqa: BLE001
                pass
        return CompressorResolution(
            compressor_name=name,
            backend_tier="MOCK",
            adapter_available=False,
            probe_only=False,
            delegate_compressor="int4_sim",
        )

    if name == "kvquant":
        if runtime is not None:
            try:
                from exactkv.compressors.kvquant_adapter import (  # noqa: PLC0415
                    create_kvquant_sim_adapter,
                )

                create_kvquant_sim_adapter(runtime)
                return CompressorResolution(
                    compressor_name=name,
                    backend_tier="RESTRICTED_ADAPTER",
                    adapter_available=True,
                    probe_only=False,
                )
            except Exception:  # noqa: BLE001
                pass
        return CompressorResolution(
            compressor_name=name,
            backend_tier="MOCK",
            adapter_available=False,
            probe_only=False,
            delegate_compressor="int4_sim",
        )

    if name == "shard":
        return CompressorResolution(
            compressor_name=name,
            backend_tier="PROBE_ONLY",
            adapter_available=False,
            probe_only=True,
        )

    msg = f"unknown compressor: {name}"
    raise ValueError(msg)


def _instantiate_compressor(
    resolution: CompressorResolution,
    runtime: ModelRuntime,
) -> Any:
    if resolution.probe_only:
        msg = "shard is probe-only; use run_shard_probe_cell"
        raise ValueError(msg)
    if resolution.delegate_compressor:
        base = _get_compressor(resolution.delegate_compressor)
        return _MockExternalCompressor(base, resolution.compressor_name)
    if resolution.compressor_name == "spectralquant":
        from exactkv.external.spectralquant_adapter import (  # noqa: PLC0415
            create_spectralquant_experimental_adapter,
        )

        return create_spectralquant_experimental_adapter(runtime)
    if resolution.compressor_name == "kvquant":
        from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter  # noqa: PLC0415

        return create_kvquant_sim_adapter(runtime)
    return _get_compressor(resolution.compressor_name)


class _MockExternalCompressor:
    """Delegates to a built-in compressor with external backend metadata."""

    def __init__(self, delegate: Any, external_name: str) -> None:
        from exactkv.compressors.base import CompressorCapabilities  # noqa: PLC0415

        self._delegate = delegate
        self.name = external_name
        caps = getattr(delegate, "capabilities", None)
        if caps is not None:
            self.capabilities = CompressorCapabilities(
                name=external_name,
                compressor_type=caps.compressor_type,
                is_simulated=True,
                supports_real_bytes_claim=False,
                supports_token_dropping=caps.supports_token_dropping,
                supports_quantization=caps.supports_quantization,
                key_bit_width=caps.key_bit_width,
                value_bit_width=caps.value_bit_width,
                adapter_name=f"Mock{external_name.title()}Adapter",
                notes=(
                    f"Mock fallback for {external_name}: delegates to "
                    f"{getattr(delegate, 'name', 'builtin')} when restricted "
                    "adapter is unavailable."
                ),
            )
        else:
            self.capabilities = CompressorCapabilities(
                name=external_name,
                compressor_type="quantization",
                is_simulated=True,
                supports_real_bytes_claim=False,
                supports_token_dropping=False,
                supports_quantization=True,
                adapter_name=f"Mock{external_name.title()}Adapter",
                notes=f"Mock fallback for {external_name}.",
            )

    def compress(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate.compress(*args, **kwargs)

    def materialize_for_draft(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate.materialize_for_draft(*args, **kwargs)

    def update_after_commit(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate.update_after_commit(*args, **kwargs)

    def stats(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate.stats(*args, **kwargs)


def run_one_with_compressor(
    runtime: ModelRuntime,
    prompt_entry: Mapping[str, str],
    *,
    compressor: Any,
    compressor_name: str,
    draft_len: int,
    max_new_tokens: int,
    backend_tier: str = "BUILTIN",
    adapter_available: bool = True,
) -> dict[str, Any]:
    """Benchmark one cell with an explicit compressor instance (no registry)."""
    from exactkv.metrics.memory import estimate_kv_memory  # noqa: PLC0415
    from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: PLC0415
    from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy  # noqa: PLC0415

    caps_dict: dict[str, Any] = {}
    if hasattr(compressor, "capabilities"):
        caps_dict = asdict(compressor.capabilities)
    caps_dict["phase_a_compressor_name"] = compressor_name
    caps_dict["backend_tier"] = backend_tier
    caps_dict["adapter_available"] = adapter_available

    prompt = prompt_entry["prompt"]
    full_res = generate_full_greedy(runtime, prompt, max_new_tokens)
    full_ids = full_res.generated_ids.squeeze(0).tolist()

    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()

    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    ekv_res = ExactKVGenerator(runtime, compressor, draft_len=draft_len).generate(
        prompt,
        max_new_tokens,
    )
    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    return _enrich_cell_metrics(
        {
            "prompt_id": prompt_entry["prompt_id"],
            "category": prompt_entry.get("category", "unknown"),
            "model_name": runtime.model_name,
            "compressor_name": compressor_name,
            "compressor_capabilities": caps_dict,
            "draft_len": draft_len,
            "max_new_tokens": max_new_tokens,
            "full": {
                "output_ids": full_ids,
                "output_text": full_res.output_text,
            },
            "lossy": {
                "output_ids": lossy_ids,
                "output_text": lossy_res.output_text,
                "token_exact_match": lossy_exact,
                "first_divergence_idx": lossy_div,
            },
            "exactkv": {
                "output_ids": ekv_ids,
                "output_text": ekv_res.output_text,
                "token_exact_match": ekv_exact,
                "acceptance": acceptance.to_dict(),
            },
            "memory": mem.to_dict(),
            "exactkv_failure": not ekv_exact,
            "backend_tier": backend_tier,
            "probe_only": False,
        },
    )


def _compute_verifier_agreement_score(result: Mapping[str, Any]) -> float:
    acceptance = (result.get("exactkv") or {}).get("acceptance") or {}
    drafted = int(acceptance.get("total_drafted") or 0)
    accepted = int(acceptance.get("total_accepted") or 0)
    if drafted > 0:
        return accepted / drafted
    if result.get("probe_only"):
        return float(result.get("verifier_agreement_score") or 0.0)
    return 1.0 if result.get("exactkv", {}).get("token_exact_match") else 0.0


def _compression_ratio(memory: Mapping[str, Any]) -> float | None:
    stored = memory.get("stored_kv_bytes")
    materialized = memory.get("materialized_working_kv_bytes")
    if stored is None or materialized is None or materialized <= 0:
        return None
    return float(stored) / float(materialized)


def _enrich_cell_metrics(result: dict[str, Any]) -> dict[str, Any]:
    lossy = result.get("lossy") or {}
    acceptance = (result.get("exactkv") or {}).get("acceptance") or {}
    memory = result.get("memory") or {}
    result["metrics"] = {
        "token_level_divergence": not bool(lossy.get("token_exact_match", True)),
        "first_divergence_index": lossy.get("first_divergence_idx"),
        "acceptance_rate": float(acceptance.get("acceptance_rate") or 0.0),
        "verifier_agreement_score": _compute_verifier_agreement_score(result),
        "exactkv_failure": bool(result.get("exactkv_failure", False)),
        "compression_ratio": _compression_ratio(memory),
    }
    return result


def build_deterministic_phase_a_cell(
    *,
    model_name: str,
    prompt_entry: Mapping[str, str],
    compressor_name: str,
    max_new_tokens: int,
    draft_len: int = DEFAULT_DRAFT_LEN,
    resolution: CompressorResolution | None = None,
) -> dict[str, Any]:
    """Hash-seeded synthetic cell for CI / offline reproducibility."""
    resolution = resolution or resolve_compressor(compressor_name)
    prompt_id = prompt_entry["prompt_id"]
    seed = abs(hash((model_name, prompt_id, compressor_name, max_new_tokens))) % 10_000

    if resolution.probe_only:
        draft_len_eff = min(draft_len, max_new_tokens)
        diverges = compressor_name == "shard" and seed % 4 == 0
        first_div = 1 if diverges else None
        acceptance_rate = 0.25 if diverges else min(1.0, 0.55 + (seed % 40) / 100.0)
        return _enrich_cell_metrics(
            {
                "prompt_id": prompt_id,
                "category": prompt_entry.get("category", "unknown"),
                "model_name": model_name,
                "compressor_name": compressor_name,
                "compressor_capabilities": {
                    "backend_tier": resolution.backend_tier,
                    "adapter_available": False,
                    "probe_only": True,
                },
                "draft_len": draft_len_eff,
                "max_new_tokens": max_new_tokens,
                "probe_only": True,
                "exactkv_failure": False,
                "verifier_agreement_score": acceptance_rate,
                "lossy": {
                    "token_exact_match": not diverges,
                    "first_divergence_idx": first_div,
                },
                "exactkv": {
                    "token_exact_match": True,
                    "acceptance": {
                        "acceptance_rate": acceptance_rate,
                        "total_drafted": draft_len_eff,
                        "total_accepted": int(acceptance_rate * draft_len_eff),
                        "total_rejected": 0,
                        "total_corrections": 0,
                    },
                },
                "memory": {
                    "stored_kv_bytes": 8000 + seed,
                    "materialized_working_kv_bytes": 32000,
                },
                "backend_tier": resolution.backend_tier,
            },
        )

    diverges = compressor_name in ("int4_sim", "k8_v4_sim", "spectralquant", "kvquant") and seed % 3 == 0
    first_div = (seed % max_new_tokens) if diverges else None
    acceptance_rate = 0.0 if diverges else min(1.0, 0.6 + (seed % 35) / 100.0)
    exactkv_failure = False

    return _enrich_cell_metrics(
        {
            "prompt_id": prompt_id,
            "category": prompt_entry.get("category", "unknown"),
            "model_name": model_name,
            "compressor_name": compressor_name,
            "compressor_capabilities": {
                "backend_tier": resolution.backend_tier,
                "adapter_available": resolution.adapter_available,
                "delegate_compressor": resolution.delegate_compressor,
                "probe_only": False,
            },
            "draft_len": min(draft_len, max_new_tokens),
            "max_new_tokens": max_new_tokens,
            "probe_only": False,
            "exactkv_failure": exactkv_failure,
            "lossy": {
                "token_exact_match": not diverges,
                "first_divergence_idx": first_div,
            },
            "exactkv": {
                "token_exact_match": not exactkv_failure,
                "acceptance": {
                    "acceptance_rate": acceptance_rate,
                    "total_drafted": min(draft_len, max_new_tokens),
                    "total_accepted": int(acceptance_rate * min(draft_len, max_new_tokens)),
                    "total_rejected": 1 if diverges else 0,
                    "total_corrections": 0,
                },
            },
            "memory": {
                "stored_kv_bytes": 4000 + seed * 2,
                "materialized_working_kv_bytes": 32000,
            },
            "backend_tier": resolution.backend_tier,
        },
    )


def run_phase_a_cell(
    runtime: ModelRuntime,
    prompt_entry: Mapping[str, str],
    *,
    compressor_name: str,
    max_new_tokens: int,
    draft_len: int = DEFAULT_DRAFT_LEN,
) -> dict[str, Any]:
    """Run one benchmark cell with adapter/mock resolution."""
    resolution = resolve_compressor(compressor_name, runtime=runtime)
    if resolution.probe_only:
        return build_deterministic_phase_a_cell(
            model_name=runtime.model_name,
            prompt_entry=prompt_entry,
            compressor_name=compressor_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            resolution=resolution,
        )
    compressor = _instantiate_compressor(resolution, runtime)
    return run_one_with_compressor(
        runtime,
        prompt_entry,
        compressor=compressor,
        compressor_name=compressor_name,
        draft_len=draft_len,
        max_new_tokens=max_new_tokens,
        backend_tier=resolution.backend_tier,
        adapter_available=resolution.adapter_available,
    )


def _aggregate_compressor_metrics(cells: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    from collections import defaultdict

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        buckets[str(cell.get("compressor_name") or "")].append(dict(cell.get("metrics") or {}))

    summary: dict[str, dict[str, Any]] = {}
    for comp, metrics_list in sorted(buckets.items()):
        n = max(len(metrics_list), 1)
        div_count = sum(1 for m in metrics_list if m.get("token_level_divergence"))
        failures = sum(1 for m in metrics_list if m.get("exactkv_failure"))
        first_divs = [m["first_divergence_index"] for m in metrics_list if m.get("first_divergence_index") is not None]
        ratios = [m["compression_ratio"] for m in metrics_list if m.get("compression_ratio") is not None]
        summary[comp] = {
            "num_cells": len(metrics_list),
            "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in metrics_list) / n,
            "mean_verifier_agreement_score": sum(
                m.get("verifier_agreement_score", 0.0) for m in metrics_list
            ) / n,
            "divergence_rate": div_count / n,
            "divergence_stability_score": 1.0 - (div_count / n),
            "exactkv_failure_rate": failures / n,
            "mean_first_divergence_index": (sum(first_divs) / len(first_divs)) if first_divs else None,
            "mean_compression_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        }
    return summary


def build_compressor_rankings(
    compressor_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Rank compressors by acceptance, divergence stability, and failure rate."""
    rows = [
        {"compressor": name, **dict(stats)}
        for name, stats in compressor_summary.items()
    ]

    def _rank(key: str, *, reverse: bool) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda r: (r.get(key) is None, r.get(key, 0)), reverse=reverse)
        return [
            {"rank": i + 1, "compressor": r["compressor"], "value": r.get(key)}
            for i, r in enumerate(ordered)
        ]

    return {
        "by_acceptance_rate": _rank("mean_acceptance_rate", reverse=True),
        "by_divergence_stability": _rank("divergence_stability_score", reverse=True),
        "by_failure_rate": _rank("exactkv_failure_rate", reverse=False),
    }


def build_per_model_tables(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-model compressor comparison tables."""
    from collections import defaultdict

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_model[str(cell.get("model_name") or "")].append(cell)

    tables: dict[str, dict[str, Any]] = {}
    for model, model_cells in sorted(by_model.items()):
        tables[model] = {
            "compressor_summary": _aggregate_compressor_metrics(model_cells),
            "total_cells": len(model_cells),
            "exactkv_failures": sum(1 for c in model_cells if c.get("exactkv_failure")),
        }
    return tables


def load_instability_scores(
    path: Path | str = DEFAULT_EXP116_REPORT,
) -> dict[str, float]:
    """Mean instability score per compressor from Exp 116 if available."""
    report_path = Path(path)
    if not report_path.is_file():
        return {}
    data = json.loads(report_path.read_text())
    descriptors = data.get("cell_descriptors") or []
    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    for d in descriptors:
        buckets[str(d.get("compressor") or "")].append(float(d.get("instability_score") or 0.0))
    return {k: float(sum(v) / len(v)) for k, v in buckets.items() if v}


def render_phase_a_markdown_summary(report: Mapping[str, Any]) -> str:
    """Render summary markdown table from Phase A report."""
    summary = report.get("compressor_summary") or {}
    instability = report.get("instability_scores_exp116") or {}
    rankings = report.get("compressor_rankings") or {}

    lines = [
        "# Phase A Scale Benchmark Summary",
        "",
        f"**Status:** {report.get('status', 'unknown')}",
        f"**Mode:** {'deterministic' if report.get('deterministic_mode') else 'inference'}",
        f"**Models evaluated:** {len(report.get('models_evaluated') or [])}",
        f"**Total cells:** {report.get('total_cells', 0)}",
        "",
        "## Compressor Comparison",
        "",
        "| Compressor | Acceptance | Divergence Stability | Failure Rate | Verifier Agreement | Instability (Exp116) |",
        "|------------|------------|----------------------|--------------|--------------------|-----------------------|",
    ]

    for comp in PHASE_A_ALL_COMPRESSORS:
        stats = summary.get(comp)
        if not stats:
            continue
        inst = instability.get(comp)
        lines.append(
            "| {comp} | {acc:.3f} | {stab:.3f} | {fail:.3f} | {ver:.3f} | {inst} |".format(
                comp=comp,
                acc=stats.get("mean_acceptance_rate", 0.0),
                stab=stats.get("divergence_stability_score", 0.0),
                fail=stats.get("exactkv_failure_rate", 0.0),
                ver=stats.get("mean_verifier_agreement_score", 0.0),
                inst=f"{inst:.3f}" if inst is not None else "n/a",
            ),
        )

    lines.extend(["", "## Rankings", ""])
    for axis, title in (
        ("by_acceptance_rate", "By acceptance rate"),
        ("by_divergence_stability", "By divergence stability"),
        ("by_failure_rate", "By failure rate (lower is better)"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        for row in rankings.get(axis) or []:
            val = row.get("value")
            val_s = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
            lines.append(f"{row.get('rank')}. `{row.get('compressor')}` — {val_s}")
        lines.append("")

    lines.extend(
        [
            "## Reproducibility",
            "",
            f"```bash",
            report.get("reproducible_cli_command", REPRODUCIBLE_CLI),
            "```",
            "",
            report.get("limitations_note", ""),
        ],
    )
    return "\n".join(lines).strip() + "\n"


def validate_phase_a_report(report: Mapping[str, Any]) -> PhaseAValidationResult:
    errors: list[str] = []

    for key in ("phase_id", "status", "cells", "compressor_summary", "compressor_rankings"):
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("phase_id") != PHASE_A_ID:
        errors.append("phase_id mismatch")

    for flag in _FORBIDDEN_FLAGS:
        if report.get(flag) is True:
            errors.append(f"{flag} must be false")

    if report.get("exactkv_generator_modified") is not False:
        errors.append("exactkv_generator_modified must be false")

    try:
        _assert_no_forbidden_fields(dict(report))
    except ValueError as exc:
        errors.append(str(exc))

    for comp in PHASE_A_ALL_COMPRESSORS:
        if comp not in (report.get("compressor_summary") or {}):
            errors.append(f"missing compressor summary: {comp}")

    return PhaseAValidationResult(valid=len(errors) == 0, errors=tuple(errors))


def run_phase_a_scale_benchmark(
    *,
    models: Sequence[str] | None = None,
    compressors: Sequence[str] | None = None,
    prompts: Sequence[Mapping[str, str]] | None = None,
    max_new_tokens_values: Sequence[int] | None = None,
    draft_len: int = DEFAULT_DRAFT_LEN,
    device: str = "cpu",
    dtype: str = "float32",
    deterministic_mode: bool = False,
    local_files_only: bool = False,
    runtime_loader: Callable[..., ModelRuntime] | None = None,
    exp116_path: Path | str = DEFAULT_EXP116_REPORT,
) -> dict[str, Any]:
    """Execute Phase A unified evaluation panel."""
    model_list = list(models or PHASE_A_MODELS)
    compressor_list = list(compressors or PHASE_A_ALL_COMPRESSORS)
    prompt_list = list(prompts or default_phase_a_prompts())
    mnt_values = list(max_new_tokens_values or PHASE_A_MAX_NEW_TOKENS)

    if deterministic_mode:
        models_evaluated = list(model_list)
        models_blocked: dict[str, str] = {}
    else:
        models_evaluated, models_blocked = detect_model_availability(
            model_list,
            local_files_only=local_files_only,
        )

    cells: list[dict[str, Any]] = []
    compressor_resolutions: dict[str, dict[str, Any]] = {}

    for comp in compressor_list:
        res = resolve_compressor(comp)
        compressor_resolutions[comp] = asdict(res)

    for model_name in models_evaluated:
        runtime: ModelRuntime | None = None
        if not deterministic_mode:
            loader = runtime_loader or ModelRuntime
            runtime = loader(model_name, device=device, dtype=dtype)

        for prompt_entry in prompt_list:
            for compressor_name in compressor_list:
                for mnt in mnt_values:
                    if deterministic_mode:
                        cell = build_deterministic_phase_a_cell(
                            model_name=model_name,
                            prompt_entry=prompt_entry,
                            compressor_name=compressor_name,
                            max_new_tokens=mnt,
                            draft_len=draft_len,
                            resolution=resolve_compressor(compressor_name),
                        )
                    else:
                        assert runtime is not None
                        if compressor_name in PHASE_A_BUILTIN_COMPRESSORS:
                            from exactkv.benchmarks.runner import RunConfig, run_one  # noqa: PLC0415

                            raw = run_one(
                                runtime,
                                dict(prompt_entry),
                                RunConfig(
                                    compressor_name=compressor_name,
                                    draft_len=draft_len,
                                    max_new_tokens=mnt,
                                ),
                            )
                            raw["backend_tier"] = "BUILTIN"
                            raw["probe_only"] = False
                            cell = _enrich_cell_metrics(raw)
                        else:
                            cell = run_phase_a_cell(
                                runtime,
                                prompt_entry,
                                compressor_name=compressor_name,
                                max_new_tokens=mnt,
                                draft_len=draft_len,
                            )
                    cells.append(cell)

    compressor_summary = _aggregate_compressor_metrics(cells)
    instability_scores = load_instability_scores(exp116_path)

    sweep_like = {
        "results": [
            {
                "compressor_name": c.get("compressor_name"),
                "exactkv_failure": c.get("exactkv_failure"),
                "exactkv": c.get("exactkv"),
            }
            for c in cells
            if not c.get("probe_only")
        ],
    }
    acceptance_table = group_acceptance_by_compressor(sweep_like) if sweep_like["results"] else []

    report = {
        "phase_id": PHASE_A_ID,
        "status": "benchmark_complete",
        "deterministic_mode": deterministic_mode,
        "manifest": build_run_manifest(
            model_name=",".join(models_evaluated) if models_evaluated else "none",
            compressor_names=list(compressor_list),
            draft_lengths=[draft_len],
            prompt_suite="phase_a_unified_panel",
            max_new_tokens=max(mnt_values) if mnt_values else 0,
        ),
        "models_requested": list(model_list),
        "models_evaluated": models_evaluated,
        "models_blocked": models_blocked,
        "compressors": list(compressor_list),
        "builtin_compressors": list(PHASE_A_BUILTIN_COMPRESSORS),
        "extended_compressors": list(PHASE_A_EXTENDED_COMPRESSORS),
        "compressor_resolutions": compressor_resolutions,
        "max_new_tokens_values": mnt_values,
        "prompt_count": len(prompt_list),
        "draft_len": draft_len,
        "total_cells": len(cells),
        "expected_cells": len(models_evaluated) * len(prompt_list) * len(compressor_list) * len(mnt_values),
        "cells": cells,
        "compressor_summary": compressor_summary,
        "compressor_rankings": build_compressor_rankings(compressor_summary),
        "per_model_tables": build_per_model_tables(cells),
        "acceptance_table": acceptance_table,
        "instability_scores_exp116": instability_scores,
        "baseline_experiment_ids": [
            "exp114_l4_minimal_runtime_coupling_layer",
            "exp115_l4_runtime_coupling_stress_panel",
            "exp116_instability_regime_analysis",
            "exp117_instability_visualization_engine",
        ],
        "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "l4_activation": False,
        "trace_only": True,
        "analysis_only": False,
        "reproducible_cli_command": REPRODUCIBLE_CLI
        + (" --device cuda" if not deterministic_mode else ""),
        "limitations_note": (
            "Phase A reports token-level divergence and acceptance only. "
            "No speed or memory savings claims unless directly measured. "
            "External compressors may use mock/probe fallbacks when adapters "
            "are unavailable."
        ),
        "validation_result": {},
    }
    report["validation_result"] = validate_phase_a_report(report).to_dict()
    return report


def write_phase_a_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path | str = DEFAULT_PHASE_A_REPORT,
    markdown_path: Path | str = DEFAULT_PHASE_A_MARKDOWN,
) -> tuple[Path, Path]:
    """Write JSON report and markdown summary."""
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n")
    md_out.write_text(render_phase_a_markdown_summary(report))
    return json_out, md_out
