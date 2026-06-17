"""Opt-in live round observer instrumentation (Phase 16P).

Records immutable round snapshots during ExactKV generation when a
``LiveRoundObserver`` is attached. Observer output is diagnostic only and
cannot affect token commits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import torch

from exactkv.attention.generation_shadow_review import (
    PROPOSED_SHADOW_CLI_FLAG,
    SHADOW_FORBIDDEN_CLAIMS,
)
from exactkv.verification.acceptance import VerificationTrace

EXPERIMENT_081_ID = "exp081_live_round_observer_smoke"
DEFAULT_EXP081_REPORT = Path("reports/experiment_081_live_round_observer_smoke.json")
PROPOSED_LIVE_OBSERVER_CLI_FLAG = "--live-round-observer"
ROUND_SOURCE_LIVE = "live_round_observer"

EXP081_CLAIM_NOTE = (
    "Opt-in live round observer smoke (Phase 16P). Disabled-by-default "
    "ExactKVGenerator instrumentation records immutable round snapshots. "
    "Not streaming-attention generation integration, vLLM, CUDA/Triton kernels, "
    "or default runtime change. Observer output is ignored and diagnostic only."
)


@dataclass(frozen=True)
class LiveRoundSnapshot:
    """Immutable diagnostic snapshot at an ExactKV round boundary (post-commit)."""

    round_index: int
    prefix_token_ids_before: tuple[int, ...]
    prefix_token_ids_after: tuple[int, ...]
    draft_token_ids: tuple[int, ...] | None
    accepted_token_count: int | None
    rejected_or_corrected_token_count: int | None
    verifier_match_count: int | None
    compressor_name: str | None
    max_new_tokens: int
    metadata: tuple[tuple[str, Any], ...] = ()


class LiveRoundObserverCallback(Protocol):
    """Optional user callback; return value is ignored."""

    def __call__(self, snapshot: LiveRoundSnapshot) -> Any: ...


@dataclass
class LiveRoundObserver:
    """Collects live round snapshots; optional callback with captured exceptions."""

    on_round: LiveRoundObserverCallback | None = None
    snapshots: list[LiveRoundSnapshot] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)

    def observe(self, snapshot: LiveRoundSnapshot) -> None:
        """Record snapshot and invoke optional callback; never raises to caller."""
        self.snapshots.append(snapshot)
        if self.on_round is None:
            return
        try:
            _ = self.on_round(snapshot)
        except Exception as exc:  # noqa: BLE001
            self.exceptions.append(f"{type(exc).__name__}: {exc}")

    def clear(self) -> None:
        self.snapshots.clear()
        self.exceptions.clear()


def build_live_round_snapshot(
    *,
    round_index: int,
    prompt_token_ids: tuple[int, ...],
    generated_token_ids_before: tuple[int, ...],
    generated_token_ids_after: tuple[int, ...],
    draft_token_ids: Sequence[int] | None,
    acceptance: Any | None,
    compressor_name: str | None,
    max_new_tokens: int,
    full_seq_len_before: int | None = None,
    full_seq_len_after: int | None = None,
) -> LiveRoundSnapshot:
    """Build immutable snapshot from post-commit round state."""
    accepted: int | None = None
    rejected_or_corrected: int | None = None
    verifier_match: int | None = None
    if acceptance is not None:
        if isinstance(acceptance, dict):
            accepted = acceptance.get("num_accepted")
            rejected = acceptance.get("num_rejected")
            correction = acceptance.get("correction_token")
        else:
            accepted = getattr(acceptance, "num_accepted", None)
            rejected = getattr(acceptance, "num_rejected", None)
            correction = getattr(acceptance, "correction_token", None)
        if rejected is not None or correction is not None:
            rejected_or_corrected = int(rejected or 0) + (
                1 if correction is not None else 0
            )
        verifier_match = accepted

    correction_token: int | None = None
    if acceptance is not None:
        correction_token = (
            acceptance.get("correction_token")
            if isinstance(acceptance, dict)
            else getattr(acceptance, "correction_token", None)
        )

    meta: list[tuple[str, Any]] = []
    if full_seq_len_before is not None:
        meta.append(("full_seq_len_before", full_seq_len_before))
    if full_seq_len_after is not None:
        meta.append(("full_seq_len_after", full_seq_len_after))
    if correction_token is not None:
        meta.append(("correction_token", correction_token))

    draft_tuple: tuple[int, ...] | None
    if draft_token_ids is None:
        draft_tuple = None
    else:
        draft_tuple = tuple(int(t) for t in draft_token_ids)

    return LiveRoundSnapshot(
        round_index=round_index,
        prefix_token_ids_before=prompt_token_ids + generated_token_ids_before,
        prefix_token_ids_after=prompt_token_ids + generated_token_ids_after,
        draft_token_ids=draft_tuple,
        accepted_token_count=accepted,
        rejected_or_corrected_token_count=rejected_or_corrected,
        verifier_match_count=verifier_match,
        compressor_name=compressor_name,
        max_new_tokens=max_new_tokens,
        metadata=tuple(meta),
    )


def _trace_value(trace: Any, key: str, default: Any = None) -> Any:
    if isinstance(trace, dict):
        return trace.get(key, default)
    return getattr(trace, key, default)


def compare_snapshot_to_trace(
    snapshot: LiveRoundSnapshot,
    trace: Any,
) -> tuple[bool, list[str]]:
    """Compare one live snapshot to an ExactKVResult VerificationTrace."""
    mismatches: list[str] = []
    trace_round = _trace_value(trace, "round_idx")
    if trace_round is not None and snapshot.round_index != trace_round:
        mismatches.append(
            f"round_index {snapshot.round_index} != trace.round_idx {trace_round}",
        )

    draft = _trace_value(trace, "draft_tokens")
    if draft is not None and snapshot.draft_token_ids is not None:
        if snapshot.draft_token_ids != tuple(draft):
            mismatches.append("draft_token_ids mismatch")

    acceptance = _trace_value(trace, "acceptance")
    if acceptance is not None and snapshot.accepted_token_count is not None:
        acc_accepted = (
            acceptance.get("num_accepted")
            if isinstance(acceptance, dict)
            else getattr(acceptance, "num_accepted", None)
        )
        if acc_accepted is not None and int(acc_accepted) != int(snapshot.accepted_token_count):
            mismatches.append("accepted_token_count mismatch")

    full_before = _trace_value(trace, "full_seq_len_before")
    full_after = _trace_value(trace, "full_seq_len_after")
    meta = dict(snapshot.metadata)
    if full_before is not None and meta.get("full_seq_len_before") != full_before:
        mismatches.append("full_seq_len_before mismatch")
    if full_after is not None and meta.get("full_seq_len_after") != full_after:
        mismatches.append("full_seq_len_after mismatch")

    if full_before is not None and len(snapshot.prefix_token_ids_before) != int(full_before):
        mismatches.append("prefix_token_ids_before length mismatch")
    if full_after is not None and len(snapshot.prefix_token_ids_after) != int(full_after):
        mismatches.append("prefix_token_ids_after length mismatch")

    return len(mismatches) == 0, mismatches


def compare_snapshots_to_traces(
    snapshots: Sequence[LiveRoundSnapshot],
    traces: Sequence[Any],
) -> dict[str, Any]:
    """Compare live observer snapshots to ExactKVResult round logs."""
    if not snapshots and not traces:
        return {
            "live_snapshot_count": 0,
            "result_round_log_count": 0,
            "snapshot_vs_result_round_log_match": True,
            "mismatches": [],
        }
    if len(snapshots) != len(traces):
        return {
            "live_snapshot_count": len(snapshots),
            "result_round_log_count": len(traces),
            "snapshot_vs_result_round_log_match": False,
            "mismatches": [f"count mismatch: live={len(snapshots)} result={len(traces)}"],
        }

    all_match = True
    mismatches: list[dict[str, Any]] = []
    for snap, trace in zip(snapshots, traces, strict=True):
        ok, reasons = compare_snapshot_to_trace(snap, trace)
        if not ok:
            all_match = False
            mismatches.append({
                "round_index": snap.round_index,
                "reasons": reasons,
            })

    return {
        "live_snapshot_count": len(snapshots),
        "result_round_log_count": len(traces),
        "snapshot_vs_result_round_log_match": all_match,
        "mismatches": mismatches,
    }


def live_snapshots_to_trace_dicts(
    snapshots: Sequence[LiveRoundSnapshot],
) -> list[dict[str, Any]]:
    """Convert live snapshots to trace-like dicts for post-hoc shadow replay."""
    traces: list[dict[str, Any]] = []
    for snap in snapshots:
        meta = dict(snap.metadata)
        traces.append({
            "round_idx": snap.round_index,
            "draft_tokens": list(snap.draft_token_ids or ()),
            "acceptance": {
                "num_accepted": snap.accepted_token_count,
                "num_rejected": (
                    (snap.rejected_or_corrected_token_count or 0)
                    - (1 if snap.rejected_or_corrected_token_count else 0)
                )
                if snap.rejected_or_corrected_token_count is not None
                else None,
                "correction_token": None,
            },
            "full_seq_len_before": meta.get("full_seq_len_before"),
            "full_seq_len_after": meta.get("full_seq_len_after"),
        })
    return traces


def _cell_safety_gates() -> dict[str, bool]:
    return {
        "observer_used_for_token_commit": False,
        "generation_modified_by_observer": False,
        "default_runtime_changed": False,
        "observer_return_value_ignored": True,
    }


def _token_lists_match(a: list[int] | None, b: list[int] | None) -> bool:
    if a is None or b is None:
        return a == b
    return list(a) == list(b)


def run_exp081_live_round_observer_panel(
    *,
    model_id: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_new_tokens: int = 8,
    compressors_requested: Sequence[str] = ("noop", "int8", "int4_sim", "k8_v4_sim"),
    draft_len: int = 4,
    local_files_only: bool = False,
    run_posthoc_shadow: bool = False,
    allow_shadow_fail: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    observer_generation_fn: Callable[..., dict[str, Any]] | None = None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 081 baseline vs live-observer parity smoke panel."""
    from exactkv.attention.generation_shadow_observer import (
        DEFAULT_MODEL_ID,
        default_exp080_prompts,
        resolve_panel_compressors,
        run_round_log_shadow_for_generation,
    )
    from exactkv.attention.generation_shadow_observer import GenerationOutput

    if model_id == "Qwen/Qwen2.5-0.5B":
        model_id = DEFAULT_MODEL_ID

    prompt_panel = list(prompts) if prompts is not None else default_exp080_prompts()
    runnable, _blocked = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if not runnable:
        blockers.append("no compressors runnable via get_compressor API")

    runtime: Any | None = None
    if baseline_generation_fn is None or observer_generation_fn is None:
        try:
            from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
            from exactkv.runtime.model_runtime import ModelRuntime

            runtime = ModelRuntime(model_id, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    cells: list[dict[str, Any]] = []
    baseline_ok = 0
    observer_ok = 0
    token_match = 0
    text_match = 0
    live_snapshot_total = 0
    snapshot_match_cells = 0
    exception_cells = 0
    round_log_available = 0
    shadow_success = 0
    shadow_total = 0

    for prompt_id, prompt_text in prompt_panel:
        preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
        for compressor in runnable:
            cell_blockers: list[str] = []

            if baseline_generation_fn is not None:
                baseline = baseline_generation_fn(
                    prompt=prompt_text, max_new_tokens=max_new_tokens, compressor_name=compressor,
                )
            elif runtime is not None:
                baseline = _run_baseline_generation(
                    runtime, prompt_text, max_new_tokens, compressor, draft_len,
                )
            else:
                baseline = {"generation_completed": False, "blockers": list(blockers)}

            if observer_generation_fn is not None:
                observed = observer_generation_fn(
                    prompt=prompt_text, max_new_tokens=max_new_tokens, compressor_name=compressor,
                )
            elif runtime is not None:
                observed = _run_observer_generation(
                    runtime, prompt_text, max_new_tokens, compressor, draft_len,
                )
            else:
                observed = {"generation_completed": False, "blockers": list(blockers)}

            if baseline.get("generation_completed"):
                baseline_ok += 1
            if observed.get("generation_completed"):
                observer_ok += 1

            b_ids = baseline.get("generated_token_ids")
            o_ids = observed.get("generated_token_ids")
            tok_match = _token_lists_match(b_ids, o_ids)
            txt_match = baseline.get("generated_text") == observed.get("generated_text")
            if tok_match:
                token_match += 1
            if txt_match:
                text_match += 1
            if not tok_match:
                cell_blockers.append("baseline_vs_observer_token_mismatch")
            if not txt_match:
                cell_blockers.append("baseline_vs_observer_text_mismatch")

            snap_cmp = observed.get("snapshot_comparison") or compare_snapshots_to_traces(
                observed.get("live_snapshots") or [],
                observed.get("result_traces") or [],
            )
            live_snapshot_total += int(snap_cmp.get("live_snapshot_count", 0))
            if observed.get("result_traces"):
                round_log_available += 1
            if snap_cmp.get("snapshot_vs_result_round_log_match"):
                snapshot_match_cells += 1
            elif observed.get("generation_completed"):
                cell_blockers.append("snapshot_vs_result_round_log_mismatch")

            if observed.get("observer_exceptions"):
                exception_cells += 1

            posthoc_status = "skipped"
            if run_posthoc_shadow and observed.get("generation_completed"):
                gen_out = GenerationOutput(
                    generation_completed=True,
                    generation_output_text=observed.get("generated_text", ""),
                    generation_output_token_ids=o_ids,
                    prompt_ids=observed.get("prompt_ids"),
                    full_sequence_ids=observed.get("full_sequence_ids"),
                    exactkv_traces=live_snapshots_to_trace_dicts(
                        observed.get("live_snapshots") or [],
                    ),
                )
                hf_model = getattr(runtime, "model", None) if runtime else None
                round_cells, _, _, _ = run_round_log_shadow_for_generation(
                    gen_out=gen_out,
                    prompt_id=prompt_id,
                    hf_model=hf_model,
                    shadow_replay_fn=shadow_replay_fn,
                    chunk_size=16,
                    accumulator_mode="float32",
                    allow_parity_fail=True,
                    allow_shadow_fail=allow_shadow_fail,
                )
                cell_shadow_success = sum(
                    1 for rc in round_cells if rc.get("shadow_status") == "shadow_complete"
                )
                shadow_total += len(round_cells)
                shadow_success += cell_shadow_success
                posthoc_status = (
                    "complete"
                    if round_cells and cell_shadow_success == len(round_cells)
                    else ("partial" if round_cells else "blocked")
                )

            cells.append({
                "prompt_id": prompt_id,
                "prompt_preview": preview,
                "compressor": compressor,
                "baseline_generation_completed": bool(baseline.get("generation_completed")),
                "observer_generation_completed": bool(observed.get("generation_completed")),
                "baseline_generated_token_ids": b_ids,
                "observer_generated_token_ids": o_ids,
                "baseline_vs_observer_token_match": tok_match,
                "baseline_vs_observer_text_match": txt_match,
                "live_snapshot_count": snap_cmp.get("live_snapshot_count", 0),
                "result_round_log_count": snap_cmp.get("result_round_log_count", 0),
                "snapshot_vs_result_round_log_match": snap_cmp.get(
                    "snapshot_vs_result_round_log_match", False,
                ),
                "observer_exceptions": list(observed.get("observer_exceptions") or []),
                "exactkv_failures_baseline": baseline.get("exactkv_failures"),
                "exactkv_failures_observer": observed.get("exactkv_failures"),
                "token_exact_match_baseline": baseline.get("token_exact_match"),
                "token_exact_match_observer": observed.get("token_exact_match"),
                "posthoc_shadow_status": posthoc_status,
                "safety_gates": _cell_safety_gates(),
                "blockers": cell_blockers,
            })

    total = len(cells)
    parity_ok = token_match == total and text_match == total and total > 0
    if not parity_ok and total > 0:
        status = "failed"
    elif baseline_ok == 0:
        status = "blocked"
    elif parity_ok:
        status = "diagnostic_complete"
    else:
        status = "diagnostic_partial"

    return {
        "experiment_id": EXPERIMENT_081_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens": max_new_tokens,
        "total_cells": total,
        "baseline_generation_successful_cells": baseline_ok,
        "observer_generation_successful_cells": observer_ok,
        "baseline_vs_observer_token_match_cells": token_match,
        "baseline_vs_observer_text_match_cells": text_match,
        "live_snapshot_total": live_snapshot_total,
        "live_snapshot_cells": total,
        "observer_exception_cells": exception_cells,
        "result_round_log_available_cells": round_log_available,
        "snapshot_vs_result_round_log_match_cells": snapshot_match_cells,
        "posthoc_shadow_run": run_posthoc_shadow,
        "posthoc_shadow_successful_cells": shadow_success,
        "exactkv_failure_summary": {
            "baseline_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_baseline") or 0) > 0
            ),
            "observer_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_observer") or 0) > 0
            ),
        },
        "generation_modified_by_observer": False,
        "observer_used_for_token_commit": False,
        "default_runtime_changed": False,
        "cells": cells,
        "blockers": blockers,
        "limitations": [
            "Opt-in live observer instrumentation; not streaming-attention integration.",
            "Default ExactKVGenerator path unchanged when observer is None.",
            "Observer callbacks cannot affect token commits.",
            "Post-hoc shadow remains after generation when enabled.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP081_CLAIM_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "live_observer_cli_flag": PROPOSED_LIVE_OBSERVER_CLI_FLAG,
        "shadow_cli_flag": PROPOSED_SHADOW_CLI_FLAG,
    }


def _run_baseline_generation(
    runtime: Any,
    prompt: str,
    max_new_tokens: int,
    compressor_name: str,
    draft_len: int,
) -> dict[str, Any]:
    from exactkv.attention.generation_shadow_observer import run_exactkv_generation_with_baseline

    gen_out = run_exactkv_generation_with_baseline(
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        compressor_name=compressor_name,
        draft_len=draft_len,
    )
    return _generation_output_to_cell_dict(gen_out, live_snapshots=[], observer_exceptions=[])


def _run_observer_generation(
    runtime: Any,
    prompt: str,
    max_new_tokens: int,
    compressor_name: str,
    draft_len: int,
) -> dict[str, Any]:
    from exactkv.compressors import get_compressor
    from exactkv.metrics.exactness import token_exact_match
    from exactkv.runtime.exactkv_generator import ExactKVGenerator
    from exactkv.runtime.generation import generate_full_greedy

    compressor = get_compressor(compressor_name)
    observer = LiveRoundObserver()
    generator = ExactKVGenerator(
        runtime, compressor, draft_len=draft_len, round_observer=observer,
    )
    result = generator.generate(prompt, max_new_tokens)
    token_ids = result.output_ids.squeeze().tolist()
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    full_res = generate_full_greedy(runtime, prompt, max_new_tokens)
    match = bool(
        token_exact_match(full_res.generated_ids, result.output_ids),
    )

    snap_cmp = compare_snapshots_to_traces(observer.snapshots, result.traces)

    return {
        "generation_completed": True,
        "generated_token_ids": list(token_ids),
        "generated_text": result.output_text,
        "prompt_ids": result.prompt_ids,
        "full_sequence_ids": result.full_sequence_ids,
        "exactkv_failures": 0 if match else 1,
        "token_exact_match": match,
        "live_snapshots": observer.snapshots,
        "result_traces": result.traces,
        "snapshot_comparison": snap_cmp,
        "observer_exceptions": observer.exceptions,
        "blockers": [],
    }


def _generation_output_to_cell_dict(
    gen_out: Any,
    *,
    live_snapshots: list[LiveRoundSnapshot],
    observer_exceptions: list[str],
) -> dict[str, Any]:
    return {
        "generation_completed": bool(gen_out.generation_completed),
        "generated_token_ids": gen_out.generation_output_token_ids,
        "generated_text": gen_out.generation_output_text,
        "prompt_ids": gen_out.prompt_ids,
        "full_sequence_ids": gen_out.full_sequence_ids,
        "exactkv_failures": gen_out.exactkv_failures,
        "token_exact_match": gen_out.token_exact_match,
        "live_snapshots": live_snapshots,
        "result_traces": gen_out.exactkv_traces or [],
        "observer_exceptions": observer_exceptions,
        "blockers": list(gen_out.blockers),
    }


def validate_exp081_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens",
        "total_cells",
        "baseline_generation_successful_cells",
        "observer_generation_successful_cells",
        "baseline_vs_observer_token_match_cells",
        "baseline_vs_observer_text_match_cells",
        "live_snapshot_total",
        "observer_exception_cells",
        "snapshot_vs_result_round_log_match_cells",
        "generation_modified_by_observer",
        "observer_used_for_token_commit",
        "default_runtime_changed",
        "cells",
        "blockers",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_081_ID:
        errors.append("experiment_id mismatch")

    for flag in (
        "generation_modified_by_observer",
        "observer_used_for_token_commit",
        "default_runtime_changed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    for idx, cell in enumerate(report.get("cells", [])):
        gates = cell.get("safety_gates") or {}
        if gates.get("observer_used_for_token_commit") is not False:
            errors.append(f"cells[{idx}].safety_gates.observer_used_for_token_commit must be false")
        if gates.get("generation_modified_by_observer") is not False:
            errors.append(f"cells[{idx}].safety_gates.generation_modified_by_observer must be false")
        if gates.get("default_runtime_changed") is not False:
            errors.append(f"cells[{idx}].safety_gates.default_runtime_changed must be false")

    return errors
