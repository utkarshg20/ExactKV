"""Explicit CLI opt-in for experimental restored-verifier runtime (Phase 13B).

``--experimental-restored-verifier`` must be passed to enable the path.
No environment-variable activation. Default CLI behavior unchanged when flag absent.
"""
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from typing import Any

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import DEFAULT_MODEL, VERIFIER_SOURCE
from exactkv.cache.restored_verifier_runner import (
    DEFAULT_SMOKE_COMPRESSORS,
    DEFAULT_SMOKE_DRAFT_LEN,
    DEFAULT_SMOKE_MAX_NEW_TOKENS,
    DEFAULT_SMOKE_PROMPT_IDS,
)
from exactkv.runtime.experimental import (
    EXPERIMENTAL_RUNTIME_CLAIM_NOTE,
    ExperimentalRestoredVerifierConfig,
    ExperimentalRuntimeMode,
    ExperimentalRuntimeResult,
    run_experimental_restored_verifier,
    validate_experimental_config,
)

EXPERIMENT_055_ID = "exp055_experimental_restored_verifier_cli"

EXPERIMENTAL_CLI_CLAIM_NOTE = (
    "Explicit CLI opt-in for non-default experimental restored-verifier runtime "
    "(Phase 13B). Activated only with --experimental-restored-verifier; "
    "environment variables do not enable this mode. Default ExactKV generation "
    "unchanged. Not vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

EXP055_CLAIM_NOTE = EXPERIMENTAL_CLI_CLAIM_NOTE

CLI_FLAG_OPTION = "--experimental-restored-verifier"
CLI_FLAG_DEST = "experimental_restored_verifier"


@dataclass
class ExperimentalCliResolution:
    """Result of parsing CLI args into experimental runtime config."""

    cli_flag_present: bool
    config: ExperimentalRestoredVerifierConfig
    parse_errors: list[str]

    @property
    def enabled(self) -> bool:
        return self.config.enabled


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_csv_ints(value: str | None) -> list[int]:
    parts = _split_csv(value)
    return [int(part) for part in parts]


def add_experimental_restored_verifier_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register explicit experimental restored-verifier CLI flags."""
    group = parser.add_argument_group("experimental restored-verifier (explicit opt-in)")
    group.add_argument(
        CLI_FLAG_OPTION,
        action="store_true",
        dest=CLI_FLAG_DEST,
        help="Explicitly enable experimental restored-verifier runtime (non-default)",
    )
    group.add_argument("--model-id", default=DEFAULT_MODEL, help="Model id when flag enabled")
    group.add_argument("--device", default="cpu", help="Device when flag enabled")
    group.add_argument("--dtype", default="float32", help="Dtype when flag enabled")
    group.add_argument(
        "--prompt-ids",
        default=",".join(DEFAULT_SMOKE_PROMPT_IDS),
        help="Comma-separated prompt ids when flag enabled",
    )
    group.add_argument(
        "--storage-backends",
        default="in_memory_kv_storage",
        help="Comma-separated storage backends when flag enabled",
    )
    group.add_argument(
        "--compressors",
        default=",".join(DEFAULT_SMOKE_COMPRESSORS),
        help="Comma-separated compressor names when flag enabled",
    )
    group.add_argument(
        "--draft-lens",
        default=str(DEFAULT_SMOKE_DRAFT_LEN),
        help="Comma-separated draft lengths when flag enabled",
    )
    group.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_SMOKE_MAX_NEW_TOKENS,
        help="Max new tokens when flag enabled",
    )
    group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for experimental CLI report",
    )


def resolve_experimental_cli_args(args: argparse.Namespace) -> ExperimentalCliResolution:
    """Build experimental config from parsed CLI args.

    When ``--experimental-restored-verifier`` is absent, returns disabled config
    and does not require other fields.
    """
    flag_present = bool(getattr(args, CLI_FLAG_DEST, False))
    if not flag_present:
        return ExperimentalCliResolution(
            cli_flag_present=False,
            config=ExperimentalRestoredVerifierConfig.disabled(),
            parse_errors=[],
        )

    parse_errors: list[str] = []
    prompt_ids = _split_csv(getattr(args, "prompt_ids", None))
    storage_backends = _split_csv(getattr(args, "storage_backends", None))
    compressor_names = _split_csv(getattr(args, "compressors", None))
    draft_lens = _split_csv_ints(getattr(args, "draft_lens", None))
    model_id = getattr(args, "model_id", "") or ""
    device = getattr(args, "device", "") or ""
    dtype = getattr(args, "dtype", "") or ""
    max_new_tokens = int(getattr(args, "max_new_tokens", 0) or 0)

    if not prompt_ids:
        parse_errors.append("--prompt-ids required when --experimental-restored-verifier is set")
    if not storage_backends:
        parse_errors.append(
            "--storage-backends required when --experimental-restored-verifier is set"
        )
    if not compressor_names:
        parse_errors.append("--compressors required when --experimental-restored-verifier is set")
    if not draft_lens:
        parse_errors.append("--draft-lens required when --experimental-restored-verifier is set")
    if max_new_tokens <= 0:
        parse_errors.append("--max-new-tokens must be positive when flag is set")

    config = ExperimentalRestoredVerifierConfig(
        enabled=True,
        mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE,
        model_id=model_id,
        device=device,
        dtype=dtype,
        prompt_ids=prompt_ids,
        storage_backends=storage_backends,
        compressor_names=compressor_names,
        draft_lens=draft_lens,
        max_new_tokens=max_new_tokens,
        verifier_source=VERIFIER_SOURCE,
        claim_note=EXP055_CLAIM_NOTE,
        namespace_prefix="exp055",
    )

    if not parse_errors:
        parse_errors.extend(validate_experimental_config(config))

    return ExperimentalCliResolution(
        cli_flag_present=True,
        config=config,
        parse_errors=parse_errors,
    )


def run_experimental_restored_verifier_from_cli(
    args: argparse.Namespace,
    *,
    experiment_id: str = EXPERIMENT_055_ID,
) -> tuple[ExperimentalCliResolution, ExperimentalRuntimeResult]:
    """Resolve CLI args and run experimental runtime when explicitly enabled."""
    resolution = resolve_experimental_cli_args(args)
    if not resolution.cli_flag_present:
        result = run_experimental_restored_verifier(
            ExperimentalRestoredVerifierConfig.disabled(),
            experiment_id=experiment_id,
        )
        return resolution, result

    if resolution.parse_errors:
        return resolution, ExperimentalRuntimeResult(
            enabled=True,
            mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE.value,
            status="invalid",
            runner_called=False,
            validation_errors=list(resolution.parse_errors),
            message="experimental CLI config validation failed",
        )

    result = run_experimental_restored_verifier(
        resolution.config,
        experiment_id=experiment_id,
    )
    return resolution, result


def report_to_exp055_json(
    resolution: ExperimentalCliResolution,
    result: ExperimentalRuntimeResult,
) -> dict[str, Any]:
    """Serialize CLI experimental run to Exp 055 JSON schema."""
    base: dict[str, Any] = {
        "experiment_id": EXPERIMENT_055_ID,
        "status": result.status,
        "cli_flag_present": resolution.cli_flag_present,
        "runtime_mode": result.mode,
        "enabled": result.enabled,
        "runner_called": result.runner_called,
        "validation_errors": list(result.validation_errors),
        "message": result.message,
        "claim_note": EXP055_CLAIM_NOTE if resolution.cli_flag_present else EXPERIMENTAL_RUNTIME_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }

    if not resolution.cli_flag_present or result.runner_report is None:
        base.update(
            {
                "model": "",
                "device": "",
                "dtype": "",
                "prompt_count": 0,
                "storage_backends": [],
                "compressor_names": [],
                "draft_len": 0,
                "draft_lens": [],
                "max_new_tokens": 0,
                "verifier_source": VERIFIER_SOURCE,
                "total_cells": 0,
                "exactkv_failures": 0,
                "token_exact_match_count": 0,
                "mean_acceptance": 0.0,
                "draft_divergence_count": 0,
                "restore_blockers": [],
                "draft_blockers": [],
                "verification_blockers": [],
                "blockers": {
                    "restore_blockers": [],
                    "draft_blockers": [],
                    "verification_blockers": [],
                },
            }
        )
        if resolution.cli_flag_present and resolution.config.enabled:
            cfg = resolution.config
            base.update(
                {
                    "model": cfg.model_id,
                    "device": cfg.device,
                    "dtype": cfg.dtype,
                    "prompt_count": len(cfg.prompt_ids),
                    "storage_backends": list(cfg.storage_backends),
                    "compressor_names": list(cfg.compressor_names),
                    "draft_len": cfg.draft_lens[0] if cfg.draft_lens else 0,
                    "draft_lens": list(cfg.draft_lens),
                    "max_new_tokens": cfg.max_new_tokens,
                }
            )
        return base

    report = result.runner_report
    cfg = report.config
    blockers = report.blockers
    draft_lens = cfg.resolved_draft_lens()
    base.update(
        {
            "config": cfg.to_dict(),
            "model": cfg.model_id,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "prompt_count": len(cfg.prompt_ids),
            "storage_backends": sorted({c.storage_backend for c in report.cells}),
            "compressor_names": list(cfg.compressor_names),
            "draft_len": draft_lens[0] if draft_lens else 0,
            "draft_lens": draft_lens,
            "max_new_tokens": cfg.max_new_tokens,
            "verifier_source": cfg.verifier_source,
            "total_cells": report.total_cells,
            "exactkv_failures": report.exactkv_failures,
            "token_exact_match_count": report.token_exact_match_count,
            "mean_acceptance": report.mean_acceptance,
            "draft_divergence_count": report.draft_divergence_count,
            "semantic_divergence_count": report.semantic_divergence_count,
            "no_real_drift_observed": report.no_real_drift_observed,
            "first_divergences": report.first_divergences,
            "restore_blockers": blockers.get("restore_blockers", []),
            "draft_blockers": blockers.get("draft_blockers", []),
            "verification_blockers": blockers.get("verification_blockers", []),
            "blockers": blockers,
            "claim_note": report.claim_note,
        }
    )
    return base


def validate_exp055_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 055 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "cli_flag_present",
        "runtime_mode",
        "enabled",
        "runner_called",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends",
        "compressor_names",
        "draft_len",
        "max_new_tokens",
        "verifier_source",
        "exactkv_failures",
        "token_exact_match_count",
        "mean_acceptance",
        "draft_divergence_count",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_055_ID:
        errors.append("experiment_id must be exp055_experimental_restored_verifier_cli")
    if not isinstance(report.get("cli_flag_present"), bool):
        errors.append("cli_flag_present must be a bool")
    if not isinstance(report.get("enabled"), bool):
        errors.append("enabled must be a bool")
    if not isinstance(report.get("runner_called"), bool):
        errors.append("runner_called must be a bool")
    if not report.get("cli_flag_present") and report.get("enabled"):
        errors.append("enabled must be false when cli_flag_present is false")
    if not report.get("cli_flag_present") and report.get("runner_called"):
        errors.append("runner_called must be false when cli_flag_present is false")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    if report.get("cli_flag_present") and report.get("runner_called"):
        failures = int(report.get("exactkv_failures", -1))
        exact_count = int(report.get("token_exact_match_count", -1))
        total = int(report.get("total_cells", 0))
        if total > 0 and exact_count + failures != total:
            errors.append("token_exact_match_count + exactkv_failures must equal total_cells")
    draft_div = report.get("draft_divergence_count")
    if not isinstance(draft_div, int) or draft_div < 0:
        errors.append("draft_divergence_count must be a non-negative int")
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    return errors


def cli_module_has_no_env_activation() -> bool:
    """Return True when experimental_cli does not read environment variables."""
    import exactkv.runtime.experimental_cli as mod

    lines = inspect.getsource(mod).splitlines()
    body_lines: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("def cli_module_has_no_env_activation"):
            skipping = True
            continue
        if skipping:
            if line and not line[0].isspace():
                skipping = False
            else:
                continue
        body_lines.append(line)
    body = "\n".join(body_lines)
    return "os.environ" not in body and "getenv(" not in body


def format_cli_summary(
    resolution: ExperimentalCliResolution,
    result: ExperimentalRuntimeResult,
) -> str:
    """Human-readable one-line CLI summary (does not affect default CLI output)."""
    if not resolution.cli_flag_present:
        return "experimental restored-verifier: disabled (flag absent)"
    if result.runner_report is not None:
        report = result.runner_report
        return (
            f"experimental restored-verifier: {result.status} "
            f"exact={report.token_exact_match_count}/{report.total_cells} "
            f"draft_div={report.draft_divergence_count}"
        )
    return f"experimental restored-verifier: {result.status} runner_called={result.runner_called}"
