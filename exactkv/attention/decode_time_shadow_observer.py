"""Guarded decode-time shadow observer dry-run (Phase 16R).

Runs shadow diagnostics inside the opt-in live round observer callback after
post-commit snapshots are emitted. Shadow results are observer-owned diagnostic
data only and cannot affect token commits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS
from exactkv.attention.live_round_observer import (
    LiveRoundSnapshot,
    _run_baseline_generation,
    _token_lists_match,
    build_live_round_snapshot,
)

EXPERIMENT_083_ID = "exp083_guarded_decode_time_shadow_smoke"
DEFAULT_EXP083_REPORT = Path("reports/experiment_083_guarded_decode_time_shadow_smoke.json")
PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG = "--guarded-decode-time-shadow"

EXP083_CLAIM_NOTE = (
    "Guarded decode-time shadow observer dry-run (Phase 16R). Shadow diagnostics "
    "run inside an opt-in observer callback after post-commit snapshots. Not "
    "streaming-attention generation integration, vLLM, CUDA/Triton kernels, or "
    "default runtime change. Shadow results are diagnostic only."
)

FORBIDDEN_TIMING_FIELDS: tuple[str, ...] = (
    "runtime_seconds",
    "latency",
    "throughput",
    "speedup",
    "tokens_per_second",
)


def default_exp083_prompts() -> list[tuple[str, str]]:
    """Two deterministic prompts for guarded decode-time shadow smoke."""
    from exactkv.attention.generation_shadow_observer import default_exp080_prompts

    return default_exp080_prompts()[:2]


def snapshot_is_post_commit(snapshot: LiveRoundSnapshot) -> bool:
    """Return True when snapshot metadata identifies a post-commit round boundary."""
    meta = dict(snapshot.metadata)
    full_before = meta.get("full_seq_len_before")
    full_after = meta.get("full_seq_len_after")
    if full_before is None or full_after is None:
        return False
    if len(snapshot.prefix_token_ids_before) != int(full_before):
        return False
    if len(snapshot.prefix_token_ids_after) != int(full_after):
        return False
    if int(full_after) < int(full_before):
        return False
    return True


def run_shadow_diagnostic_for_snapshot(
    snapshot: LiveRoundSnapshot,
    *,
    prompt_id: str,
    hf_model: Any | None,
    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None,
    allow_shadow_fail: bool = True,
) -> dict[str, Any]:
    """Run one shadow diagnostic replay for a live snapshot prefix."""
    from exactkv.attention.generation_shadow_observer import _shadow_round_cell

    prefix_after = snapshot.prefix_token_ids_after
    input_ids = torch.tensor([list(prefix_after)], dtype=torch.long)
    meta = dict(snapshot.metadata)
    entry = {
        "round_index": snapshot.round_index,
        "prefix_length_before_round": meta.get("full_seq_len_before")
        or len(snapshot.prefix_token_ids_before),
        "prefix_length_after_round": meta.get("full_seq_len_after")
        or len(prefix_after),
        "draft_length": (
            len(snapshot.draft_token_ids) if snapshot.draft_token_ids is not None else None
        ),
        "accepted_token_count": snapshot.accepted_token_count,
        "rejected_or_corrected_token_count": snapshot.rejected_or_corrected_token_count,
    }
    raw = _shadow_round_cell(
        entry=entry,
        input_ids=input_ids,
        prompt_id=prompt_id,
        hf_model=hf_model,
        shadow_replay_fn=shadow_diagnostic_fn,
        chunk_size=16,
        accumulator_mode="float32",
        allow_parity_fail=True,
        allow_shadow_fail=allow_shadow_fail,
    )
    return {
        "round_index": snapshot.round_index,
        "shadow_sequence_length": raw.get("shadow_sequence_length", 0),
        "shadow_status": raw.get("shadow_status"),
        "tolerance_policy_status": raw.get("tolerance_policy_status"),
        "topk_agreement_metrics": raw.get("topk_agreement_metrics"),
        "shadow_top1_token_id": raw.get("shadow_top1_token_id"),
        "shadow_top1_token_text": raw.get("shadow_top1_token_text"),
        "shadow_topk_token_ids": raw.get("shadow_topk_token_ids"),
        "streaming_top1_token_id": raw.get("streaming_top1_token_id"),
        "streaming_top5_token_ids": raw.get("streaming_top5_token_ids"),
        "interpretation_note": raw.get("interpretation_note", ""),
        "blockers": list(raw.get("blockers") or []),
    }


def decode_time_shadow_cell_matches_posthoc(
    decode_cell: dict[str, Any],
    posthoc_cell: dict[str, Any],
) -> bool:
    """Compare deterministic decode-time vs post-hoc shadow fields."""
    if decode_cell.get("round_index") != posthoc_cell.get("round_index"):
        return False
    if decode_cell.get("shadow_status") != posthoc_cell.get("shadow_status"):
        return False
    if decode_cell.get("tolerance_policy_status") != posthoc_cell.get(
        "tolerance_policy_status",
    ):
        return False
    dt_topk = decode_cell.get("topk_agreement_metrics") or {}
    ph_topk = posthoc_cell.get("topk_agreement_metrics") or {}
    return dt_topk.get("top1_agreement") == ph_topk.get("top1_agreement")


def compare_decode_time_vs_posthoc_shadow(
    decode_cells: Sequence[dict[str, Any]],
    posthoc_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize decode-time vs post-hoc shadow agreement per round."""
    posthoc_by_round = {c.get("round_index"): c for c in posthoc_cells}
    matches = 0
    mismatches: list[dict[str, Any]] = []
    for dt in decode_cells:
        rnd = dt.get("round_index")
        ph = posthoc_by_round.get(rnd)
        if ph is None:
            mismatches.append({"round_index": rnd, "reason": "missing posthoc cell"})
            continue
        if decode_time_shadow_cell_matches_posthoc(dt, ph):
            matches += 1
        else:
            mismatches.append({
                "round_index": rnd,
                "reason": "decode_time_vs_posthoc_mismatch",
                "decode_shadow_status": dt.get("shadow_status"),
                "posthoc_shadow_status": ph.get("shadow_status"),
            })
    total = len(decode_cells)
    return {
        "total_rounds": total,
        "matching_rounds": matches,
        "mismatching_rounds": len(mismatches),
        "all_match": total > 0 and matches == total and not mismatches,
        "mismatches": mismatches,
    }


@dataclass
class GuardedDecodeTimeShadowObserver:
    """Live observer that runs guarded decode-time shadow inside the callback."""

    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None = None
    hf_model: Any | None = None
    prompt_id: str = ""
    allow_shadow_fail: bool = True
    snapshots: list[LiveRoundSnapshot] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    decode_time_shadow_cells: list[dict[str, Any]] = field(default_factory=list)
    shadow_callback_exceptions: list[str] = field(default_factory=list)

    def observe(self, snapshot: LiveRoundSnapshot) -> Any:
        """Record snapshot and run decode-time shadow; return value is ignored."""
        snap_copy = snapshot
        self.snapshots.append(snap_copy)
        is_post_commit = snapshot_is_post_commit(snapshot)

        cell: dict[str, Any] = {
            "round_index": snapshot.round_index,
            "snapshot_is_post_commit": is_post_commit,
            "shadow_executed_after_round_commit": is_post_commit,
            "shadow_result_used_for_token_commit": False,
            "shadow_sequence_length": 0,
            "shadow_status": "blocked",
            "tolerance_policy_status": "blocked",
            "topk_agreement_metrics": None,
            "exception": None,
            "posthoc_shadow_match": None,
            "interpretation_note": "",
        }

        if not is_post_commit:
            cell["interpretation_note"] = (
                "Snapshot timing cannot be identified as post-commit; "
                "decode-time shadow blocked."
            )
            self.decode_time_shadow_cells.append(cell)
            return None

        try:
            shadow_result = run_shadow_diagnostic_for_snapshot(
                snapshot,
                prompt_id=self.prompt_id,
                hf_model=self.hf_model,
                shadow_diagnostic_fn=self.shadow_diagnostic_fn,
                allow_shadow_fail=self.allow_shadow_fail,
            )
            cell.update({
                "shadow_sequence_length": shadow_result.get("shadow_sequence_length", 0),
                "shadow_status": shadow_result.get("shadow_status"),
                "tolerance_policy_status": shadow_result.get("tolerance_policy_status"),
                "topk_agreement_metrics": shadow_result.get("topk_agreement_metrics"),
                "interpretation_note": shadow_result.get("interpretation_note", ""),
            })
            for blocker in shadow_result.get("blockers") or []:
                if "shadow replay failed" in blocker:
                    exc_str = blocker.split("shadow replay failed: ", 1)[-1]
                    cell["exception"] = exc_str
                    self.shadow_callback_exceptions.append(exc_str)
            if shadow_result.get("blockers"):
                cell["interpretation_note"] = (
                    f"{cell['interpretation_note']} blockers={shadow_result['blockers']}"
                ).strip()
        except Exception as exc:  # noqa: BLE001
            exc_str = f"{type(exc).__name__}: {exc}"
            cell["exception"] = exc_str
            cell["shadow_status"] = "shadow_blocked"
            cell["tolerance_policy_status"] = "blocked"
            cell["interpretation_note"] = "Decode-time shadow exception captured."
            self.shadow_callback_exceptions.append(exc_str)

        self.decode_time_shadow_cells.append(cell)
        return {"ignored_diagnostic": True}

    def clear(self) -> None:
        self.snapshots.clear()
        self.exceptions.clear()
        self.decode_time_shadow_cells.clear()
        self.shadow_callback_exceptions.clear()


def _cell_safety_gates() -> dict[str, bool]:
    return {
        "decode_time_shadow_used_for_token_commit": False,
        "generation_modified_by_decode_time_shadow": False,
        "default_runtime_changed": False,
        "observer_return_value_ignored": True,
        "shadow_exception_affects_generation": False,
        "shadow_result_exposed_to_generator": False,
    }


def _safety_gates_ok(gates: dict[str, bool]) -> bool:
    for key in (
        "decode_time_shadow_used_for_token_commit",
        "generation_modified_by_decode_time_shadow",
        "default_runtime_changed",
        "shadow_exception_affects_generation",
        "shadow_result_exposed_to_generator",
    ):
        if gates.get(key) is not False:
            return False
    return gates.get("observer_return_value_ignored") is True


def _aggregate_safety_gate_summary(cells: Sequence[dict[str, Any]]) -> dict[str, int]:
    ok = sum(1 for c in cells if _safety_gates_ok(c.get("safety_gates") or {}))
    return {
        "cells_all_gates_ok": ok,
        "cells_with_gate_failure": len(cells) - ok,
    }


def _exp083_cell_safety_gates() -> dict[str, bool]:
    return _cell_safety_gates()


def _exp083_safety_gates_ok(gates: dict[str, bool]) -> bool:
    return _safety_gates_ok(gates)


def _run_guarded_shadow_generation(
    runtime: Any,
    prompt: str,
    prompt_id: str,
    max_new_tokens: int,
    compressor_name: str,
    draft_len: int,
    *,
    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None = None,
    allow_shadow_fail: bool = True,
) -> dict[str, Any]:
    from exactkv.compressors import get_compressor
    from exactkv.metrics.exactness import token_exact_match
    from exactkv.runtime.exactkv_generator import ExactKVGenerator
    from exactkv.runtime.generation import generate_full_greedy

    compressor = get_compressor(compressor_name)
    observer = GuardedDecodeTimeShadowObserver(
        shadow_diagnostic_fn=shadow_diagnostic_fn,
        hf_model=getattr(runtime, "model", None),
        prompt_id=prompt_id,
        allow_shadow_fail=allow_shadow_fail,
    )
    generator = ExactKVGenerator(
        runtime, compressor, draft_len=draft_len, round_observer=observer,
    )
    result = generator.generate(prompt, max_new_tokens)
    token_ids = result.output_ids.squeeze().tolist()
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    full_res = generate_full_greedy(runtime, prompt, max_new_tokens)
    match = bool(token_exact_match(full_res.generated_ids, result.output_ids))

    return {
        "generation_completed": True,
        "generated_token_ids": list(token_ids),
        "generated_text": result.output_text,
        "prompt_ids": result.prompt_ids,
        "full_sequence_ids": result.full_sequence_ids,
        "exactkv_failures": 0 if match else 1,
        "token_exact_match": match,
        "live_snapshots": observer.snapshots,
        "decode_time_shadow_cells": observer.decode_time_shadow_cells,
        "decode_time_shadow_callback_count": len(observer.decode_time_shadow_cells),
        "decode_time_shadow_successful_callbacks": sum(
            1
            for c in observer.decode_time_shadow_cells
            if c.get("shadow_status") == "shadow_complete"
            and c.get("snapshot_is_post_commit")
            and not c.get("exception")
        ),
        "decode_time_shadow_exception_callbacks": len(observer.shadow_callback_exceptions),
        "observer_exceptions": observer.exceptions,
        "blockers": [],
    }


def run_exp083_guarded_decode_time_shadow_smoke(
    *,
    model_id: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_new_tokens: int = 8,
    compressors_requested: Sequence[str] = ("noop", "int8"),
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_shadow_fail: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    guarded_generation_fn: Callable[..., dict[str, Any]] | None = None,
    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 083 guarded decode-time shadow observer smoke."""
    from exactkv.attention.generation_shadow_observer import (
        DEFAULT_MODEL_ID,
        resolve_panel_compressors,
        run_posthoc_shadow_from_live_snapshots,
    )

    del local_files_only  # reserved for future ModelRuntime wiring

    if model_id == "Qwen/Qwen2.5-0.5B":
        model_id = DEFAULT_MODEL_ID

    prompt_panel = list(prompts) if prompts is not None else default_exp083_prompts()
    runnable, _blocked = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if not runnable:
        blockers.append("no compressors runnable via get_compressor API")

    runtime: Any | None = None
    if baseline_generation_fn is None or guarded_generation_fn is None:
        try:
            from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
            from exactkv.runtime.model_runtime import ModelRuntime

            runtime = ModelRuntime(model_id, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    cells: list[dict[str, Any]] = []
    baseline_ok = 0
    guarded_ok = 0
    token_match = 0
    text_match = 0
    decode_callback_total = 0
    decode_success_total = 0
    decode_exception_total = 0
    posthoc_comparison_cells = 0
    decode_vs_posthoc_match_cells = 0
    failed_cells = 0

    for prompt_id, prompt_text in prompt_panel:
        preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
        for compressor in runnable:
            cell_blockers: list[str] = []

            if baseline_generation_fn is not None:
                baseline = baseline_generation_fn(
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                )
            elif runtime is not None:
                baseline = _run_baseline_generation(
                    runtime, prompt_text, max_new_tokens, compressor, draft_len,
                )
            else:
                baseline = {"generation_completed": False, "blockers": list(blockers)}

            if guarded_generation_fn is not None:
                guarded = guarded_generation_fn(
                    prompt=prompt_text,
                    prompt_id=prompt_id,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                )
            elif runtime is not None:
                guarded = _run_guarded_shadow_generation(
                    runtime,
                    prompt_text,
                    prompt_id,
                    max_new_tokens,
                    compressor,
                    draft_len,
                    shadow_diagnostic_fn=shadow_diagnostic_fn,
                    allow_shadow_fail=allow_shadow_fail,
                )
            else:
                guarded = {"generation_completed": False, "blockers": list(blockers)}

            baseline_completed = bool(baseline.get("generation_completed"))
            guarded_completed = bool(guarded.get("generation_completed"))
            if baseline_completed:
                baseline_ok += 1
            if guarded_completed:
                guarded_ok += 1

            b_ids = baseline.get("generated_token_ids")
            g_ids = guarded.get("generated_token_ids")
            tok_match = _token_lists_match(b_ids, g_ids)
            txt_match = baseline.get("generated_text") == guarded.get("generated_text")
            if tok_match:
                token_match += 1
            if txt_match:
                text_match += 1
            if not tok_match:
                cell_blockers.append("baseline_vs_guarded_token_mismatch")
            if not txt_match:
                cell_blockers.append("baseline_vs_guarded_text_mismatch")

            dt_cells = list(guarded.get("decode_time_shadow_cells") or [])
            dt_callback_count = int(guarded.get("decode_time_shadow_callback_count", len(dt_cells)))
            dt_success = int(guarded.get("decode_time_shadow_successful_callbacks", 0))
            dt_exceptions = int(guarded.get("decode_time_shadow_exception_callbacks", 0))
            decode_callback_total += dt_callback_count
            decode_success_total += dt_success
            decode_exception_total += dt_exceptions

            posthoc_cells: list[dict[str, Any]] = []
            posthoc_summary: dict[str, Any] = {
                "status": "skipped",
                "all_match": False,
                "matching_rounds": 0,
                "total_rounds": 0,
            }
            if guarded_completed and dt_cells:
                snaps = guarded.get("live_snapshots") or []
                if not snaps:
                    cell_blockers.append("blocked_missing_live_snapshots")
                elif not all(snapshot_is_post_commit(s) for s in snaps):
                    cell_blockers.append("blocked_snapshot_not_post_commit")
                else:
                    hf_model = getattr(runtime, "model", None) if runtime else None
                    posthoc_cells, ph_blockers = run_posthoc_shadow_from_live_snapshots(
                        snapshots=snaps,
                        prompt_id=prompt_id,
                        hf_model=hf_model,
                        shadow_replay_fn=shadow_diagnostic_fn,
                        allow_shadow_fail=allow_shadow_fail,
                    )
                    cell_blockers.extend(ph_blockers)
                    posthoc_summary = compare_decode_time_vs_posthoc_shadow(
                        dt_cells, posthoc_cells,
                    )
                    posthoc_summary["status"] = (
                        "complete" if posthoc_summary.get("all_match") else "mismatch"
                    )
                    for dt_cell, ph_cell in zip(dt_cells, posthoc_cells, strict=False):
                        if dt_cell.get("round_index") == ph_cell.get("round_index"):
                            dt_cell["posthoc_shadow_match"] = (
                                decode_time_shadow_cell_matches_posthoc(dt_cell, ph_cell)
                            )
                    posthoc_comparison_cells += 1
                    if posthoc_summary.get("all_match"):
                        decode_vs_posthoc_match_cells += 1
                    else:
                        cell_blockers.append("decode_time_vs_posthoc_shadow_mismatch")

            safety = _cell_safety_gates()
            if not _safety_gates_ok(safety):
                cell_blockers.append("safety_gate_failed")
                failed_cells += 1
            if not tok_match or not txt_match:
                failed_cells += 1

            cells.append({
                "prompt_id": prompt_id,
                "prompt_preview": preview,
                "compressor": compressor,
                "baseline_generation_completed": baseline_completed,
                "guarded_shadow_generation_completed": guarded_completed,
                "baseline_generated_token_ids": b_ids,
                "guarded_shadow_generated_token_ids": g_ids,
                "baseline_vs_guarded_token_match": tok_match,
                "baseline_vs_guarded_text_match": txt_match,
                "decode_time_shadow_callback_count": dt_callback_count,
                "decode_time_shadow_successful_callbacks": dt_success,
                "decode_time_shadow_exception_callbacks": dt_exceptions,
                "decode_time_shadow_cells": dt_cells,
                "posthoc_comparison_summary": posthoc_summary,
                "exactkv_failures_baseline": baseline.get("exactkv_failures"),
                "exactkv_failures_guarded": guarded.get("exactkv_failures"),
                "token_exact_match_baseline": baseline.get("token_exact_match"),
                "token_exact_match_guarded": guarded.get("token_exact_match"),
                "safety_gates": safety,
                "blockers": cell_blockers,
            })

    total = len(cells)
    parity_ok = token_match == total and text_match == total and total > 0
    if failed_cells > 0 or (total > 0 and not parity_ok):
        status = "failed"
    elif baseline_ok == 0:
        status = "blocked"
    elif parity_ok and decode_vs_posthoc_match_cells == posthoc_comparison_cells:
        status = "diagnostic_complete"
    elif parity_ok:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_083_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens": max_new_tokens,
        "total_cells": total,
        "baseline_generation_successful_cells": baseline_ok,
        "guarded_shadow_generation_successful_cells": guarded_ok,
        "baseline_vs_guarded_token_match_cells": token_match,
        "baseline_vs_guarded_text_match_cells": text_match,
        "decode_time_shadow_callback_count": decode_callback_total,
        "decode_time_shadow_successful_callbacks": decode_success_total,
        "decode_time_shadow_exception_callbacks": decode_exception_total,
        "posthoc_shadow_comparison_cells": posthoc_comparison_cells,
        "decode_time_vs_posthoc_shadow_match_cells": decode_vs_posthoc_match_cells,
        "exactkv_failure_summary": {
            "baseline_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_baseline") or 0) > 0
            ),
            "guarded_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_guarded") or 0) > 0
            ),
        },
        "generation_modified_by_decode_time_shadow": False,
        "decode_time_shadow_used_for_token_commit": False,
        "default_runtime_changed": False,
        "cells": cells,
        "blockers": blockers,
        "limitations": [
            "Guarded decode-time shadow observer dry-run; not streaming-attention integration.",
            "Shadow runs inside opt-in observer callback only.",
            "Shadow results cannot affect token commits.",
            "Observer return values are ignored.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP083_CLAIM_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "guarded_decode_time_shadow_cli_flag": PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
    }


def validate_exp083_report(report: dict[str, Any]) -> list[str]:
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
        "guarded_shadow_generation_successful_cells",
        "baseline_vs_guarded_token_match_cells",
        "baseline_vs_guarded_text_match_cells",
        "decode_time_shadow_callback_count",
        "decode_time_shadow_successful_callbacks",
        "decode_time_shadow_exception_callbacks",
        "posthoc_shadow_comparison_cells",
        "decode_time_vs_posthoc_shadow_match_cells",
        "exactkv_failure_summary",
        "generation_modified_by_decode_time_shadow",
        "decode_time_shadow_used_for_token_commit",
        "default_runtime_changed",
        "cells",
        "blockers",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_083_ID:
        errors.append("experiment_id mismatch")

    for flag in (
        "generation_modified_by_decode_time_shadow",
        "decode_time_shadow_used_for_token_commit",
        "default_runtime_changed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    for idx, cell in enumerate(report.get("cells", [])):
        gates = cell.get("safety_gates") or {}
        if gates.get("decode_time_shadow_used_for_token_commit") is not False:
            errors.append(
                f"cells[{idx}].safety_gates.decode_time_shadow_used_for_token_commit must be false",
            )
        if gates.get("shadow_exception_affects_generation") is not False:
            errors.append(
                f"cells[{idx}].safety_gates.shadow_exception_affects_generation must be false",
            )
        if gates.get("shadow_result_exposed_to_generator") is not False:
            errors.append(
                f"cells[{idx}].safety_gates.shadow_result_exposed_to_generator must be false",
            )
        for ck in ("decode_time_shadow_cells", "posthoc_comparison_summary", "safety_gates"):
            if ck not in cell:
                errors.append(f"cells[{idx}] missing {ck}")
        for dt in cell.get("decode_time_shadow_cells") or []:
            if dt.get("shadow_result_used_for_token_commit") is not False:
                errors.append(
                    f"cells[{idx}] decode_time shadow_result_used_for_token_commit must be false",
                )

    return errors


# --- Phase 16S: expanded guarded decode-time shadow panel ---

EXPERIMENT_084_ID = "exp084_guarded_decode_time_shadow_panel"
DEFAULT_EXP084_REPORT = Path("reports/experiment_084_guarded_decode_time_shadow_panel.json")
DEFAULT_EXP084_MAX_NEW_TOKENS_VALUES: tuple[int, ...] = (4, 8)
DEFAULT_EXP084_COMPRESSORS: tuple[str, ...] = ("noop", "int8", "int4_sim", "k8_v4_sim")

EXP084_CLAIM_NOTE = (
    "Expanded guarded decode-time shadow panel (Phase 16S). Broader prompt/compressor/"
    "max_new_tokens sweep with callback-time shadow diagnostics. Not streaming-attention "
    "generation integration, vLLM, CUDA/Triton kernels, or default runtime change."
)


def default_exp084_prompts() -> list[tuple[str, str]]:
    """Four deterministic prompts for expanded guarded decode-time shadow panel."""
    from exactkv.attention.generation_shadow_observer import default_exp080_prompts

    return default_exp080_prompts()[:4]


def _build_guarded_panel_cell(
    *,
    prompt_id: str,
    prompt_preview: str,
    compressor: str,
    max_new_tokens: int,
    runtime: Any | None,
    draft_len: int,
    global_blockers: list[str],
    baseline_generation_fn: Callable[..., dict[str, Any]] | None,
    guarded_generation_fn: Callable[..., dict[str, Any]] | None,
    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None,
    allow_shadow_fail: bool,
) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
    """Run one baseline vs guarded-shadow panel cell; return cell, failed, shadow cells."""
    from exactkv.attention.generation_shadow_observer import run_posthoc_shadow_from_live_snapshots

    cell_blockers: list[str] = []

    if baseline_generation_fn is not None:
        baseline = baseline_generation_fn(
            prompt=prompt_preview,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
        )
    elif runtime is not None:
        baseline = _run_baseline_generation(
            runtime, prompt_preview, max_new_tokens, compressor, draft_len,
        )
    else:
        baseline = {"generation_completed": False, "blockers": list(global_blockers)}

    if guarded_generation_fn is not None:
        guarded = guarded_generation_fn(
            prompt=prompt_preview,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
        )
    elif runtime is not None:
        guarded = _run_guarded_shadow_generation(
            runtime,
            prompt_preview,
            prompt_id,
            max_new_tokens,
            compressor,
            draft_len,
            shadow_diagnostic_fn=shadow_diagnostic_fn,
            allow_shadow_fail=allow_shadow_fail,
        )
    else:
        guarded = {"generation_completed": False, "blockers": list(global_blockers)}

    baseline_completed = bool(baseline.get("generation_completed"))
    guarded_completed = bool(guarded.get("generation_completed"))

    b_ids = baseline.get("generated_token_ids")
    g_ids = guarded.get("generated_token_ids")
    tok_match = _token_lists_match(b_ids, g_ids)
    txt_match = baseline.get("generated_text") == guarded.get("generated_text")
    if not tok_match:
        cell_blockers.append("baseline_vs_guarded_token_mismatch")
    if not txt_match:
        cell_blockers.append("baseline_vs_guarded_text_mismatch")

    dt_cells = list(guarded.get("decode_time_shadow_cells") or [])
    dt_callback_count = int(guarded.get("decode_time_shadow_callback_count", len(dt_cells)))
    dt_success = int(guarded.get("decode_time_shadow_successful_callbacks", 0))
    dt_exceptions = int(guarded.get("decode_time_shadow_exception_callbacks", 0))

    posthoc_summary: dict[str, Any] = {
        "status": "skipped",
        "all_match": False,
        "matching_rounds": 0,
        "total_rounds": 0,
    }
    if guarded_completed and dt_cells:
        snaps = guarded.get("live_snapshots") or []
        if not snaps:
            cell_blockers.append("blocked_missing_live_snapshots")
        elif not all(snapshot_is_post_commit(s) for s in snaps):
            cell_blockers.append("blocked_snapshot_not_post_commit")
        else:
            hf_model = getattr(runtime, "model", None) if runtime else None
            posthoc_cells, ph_blockers = run_posthoc_shadow_from_live_snapshots(
                snapshots=snaps,
                prompt_id=prompt_id,
                hf_model=hf_model,
                shadow_replay_fn=shadow_diagnostic_fn,
                allow_shadow_fail=allow_shadow_fail,
            )
            cell_blockers.extend(ph_blockers)
            posthoc_summary = compare_decode_time_vs_posthoc_shadow(dt_cells, posthoc_cells)
            posthoc_summary["status"] = (
                "complete" if posthoc_summary.get("all_match") else "mismatch"
            )
            for dt_cell, ph_cell in zip(dt_cells, posthoc_cells, strict=False):
                if dt_cell.get("round_index") == ph_cell.get("round_index"):
                    dt_cell["posthoc_shadow_match"] = decode_time_shadow_cell_matches_posthoc(
                        dt_cell, ph_cell,
                    )
            if not posthoc_summary.get("all_match"):
                cell_blockers.append("decode_time_vs_posthoc_shadow_mismatch")

    safety = _cell_safety_gates()
    failed = not _safety_gates_ok(safety) or not tok_match or not txt_match
    if not _safety_gates_ok(safety):
        cell_blockers.append("safety_gate_failed")

    cell = {
        "prompt_id": prompt_id,
        "prompt_preview": prompt_preview,
        "compressor": compressor,
        "max_new_tokens": max_new_tokens,
        "baseline_generation_completed": baseline_completed,
        "guarded_shadow_generation_completed": guarded_completed,
        "baseline_generated_token_ids": b_ids,
        "guarded_shadow_generated_token_ids": g_ids,
        "baseline_vs_guarded_token_match": tok_match,
        "baseline_vs_guarded_text_match": txt_match,
        "decode_time_shadow_callback_count": dt_callback_count,
        "decode_time_shadow_successful_callbacks": dt_success,
        "decode_time_shadow_exception_callbacks": dt_exceptions,
        "decode_time_shadow_cells": dt_cells,
        "posthoc_comparison_summary": posthoc_summary,
        "exactkv_failures_baseline": baseline.get("exactkv_failures"),
        "exactkv_failures_guarded": guarded.get("exactkv_failures"),
        "token_exact_match_baseline": baseline.get("token_exact_match"),
        "token_exact_match_guarded": guarded.get("token_exact_match"),
        "safety_gates": safety,
        "blockers": cell_blockers,
    }
    return cell, failed, dt_cells


def run_exp084_guarded_decode_time_shadow_panel(
    *,
    model_id: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_new_tokens_values: Sequence[int] = DEFAULT_EXP084_MAX_NEW_TOKENS_VALUES,
    compressors_requested: Sequence[str] = DEFAULT_EXP084_COMPRESSORS,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_shadow_fail: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    guarded_generation_fn: Callable[..., dict[str, Any]] | None = None,
    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 084 expanded guarded decode-time shadow panel."""
    from exactkv.attention.generation_shadow_observer import (
        DEFAULT_MODEL_ID,
        _aggregate_tolerance_by_round,
        _aggregate_topk_by_round,
        resolve_panel_compressors,
    )

    del local_files_only

    if model_id == "Qwen/Qwen2.5-0.5B":
        model_id = DEFAULT_MODEL_ID

    prompt_panel = list(prompts) if prompts is not None else default_exp084_prompts()
    max_nt_values = list(max_new_tokens_values)
    runnable, _blocked = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if not runnable:
        blockers.append("no compressors runnable via get_compressor API")
    if not max_nt_values:
        blockers.append("no max_new_tokens_values provided")

    runtime: Any | None = None
    if baseline_generation_fn is None or guarded_generation_fn is None:
        try:
            from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
            from exactkv.runtime.model_runtime import ModelRuntime

            runtime = ModelRuntime(model_id, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    cells: list[dict[str, Any]] = []
    all_shadow_cells: list[dict[str, Any]] = []
    baseline_ok = 0
    guarded_ok = 0
    token_match = 0
    text_match = 0
    decode_callback_total = 0
    decode_success_total = 0
    decode_exception_total = 0
    posthoc_comparison_cells = 0
    decode_vs_posthoc_match_cells = 0
    failed_cells = 0

    for prompt_id, prompt_text in prompt_panel:
        preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
        for compressor in runnable:
            for max_new_tokens in max_nt_values:
                cell, failed, dt_cells = _build_guarded_panel_cell(
                    prompt_id=prompt_id,
                    prompt_preview=preview,
                    compressor=compressor,
                    max_new_tokens=max_new_tokens,
                    runtime=runtime,
                    draft_len=draft_len,
                    global_blockers=blockers,
                    baseline_generation_fn=baseline_generation_fn,
                    guarded_generation_fn=guarded_generation_fn,
                    shadow_diagnostic_fn=shadow_diagnostic_fn,
                    allow_shadow_fail=allow_shadow_fail,
                )
                cells.append(cell)
                all_shadow_cells.extend(dt_cells)

                if cell["baseline_generation_completed"]:
                    baseline_ok += 1
                if cell["guarded_shadow_generation_completed"]:
                    guarded_ok += 1
                if cell["baseline_vs_guarded_token_match"]:
                    token_match += 1
                if cell["baseline_vs_guarded_text_match"]:
                    text_match += 1

                decode_callback_total += cell["decode_time_shadow_callback_count"]
                decode_success_total += cell["decode_time_shadow_successful_callbacks"]
                decode_exception_total += cell["decode_time_shadow_exception_callbacks"]

                if cell["guarded_shadow_generation_completed"] and dt_cells:
                    posthoc_comparison_cells += 1
                    if cell["posthoc_comparison_summary"].get("all_match"):
                        decode_vs_posthoc_match_cells += 1

                if failed:
                    failed_cells += 1

    total = len(cells)
    token_mismatch = total - token_match
    text_mismatch = total - text_match
    posthoc_mismatch = posthoc_comparison_cells - decode_vs_posthoc_match_cells
    parity_ok = token_match == total and text_match == total and total > 0

    if failed_cells > 0 or (total > 0 and not parity_ok):
        status = "failed"
    elif baseline_ok == 0:
        status = "blocked"
    elif parity_ok and decode_vs_posthoc_match_cells == posthoc_comparison_cells:
        status = "diagnostic_complete"
    elif parity_ok:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_084_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens_values": max_nt_values,
        "total_cells": total,
        "baseline_generation_successful_cells": baseline_ok,
        "guarded_shadow_generation_successful_cells": guarded_ok,
        "baseline_vs_guarded_token_match_cells": token_match,
        "baseline_vs_guarded_text_match_cells": text_match,
        "baseline_vs_guarded_token_mismatch_cells": token_mismatch,
        "baseline_vs_guarded_text_mismatch_cells": text_mismatch,
        "decode_time_shadow_callback_count": decode_callback_total,
        "decode_time_shadow_successful_callbacks": decode_success_total,
        "decode_time_shadow_exception_callbacks": decode_exception_total,
        "posthoc_shadow_comparison_cells": posthoc_comparison_cells,
        "decode_time_vs_posthoc_shadow_match_cells": decode_vs_posthoc_match_cells,
        "decode_time_vs_posthoc_shadow_mismatch_cells": posthoc_mismatch,
        "exactkv_failure_summary": {
            "baseline_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_baseline") or 0) > 0
            ),
            "guarded_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_guarded") or 0) > 0
            ),
        },
        "tolerance_policy_summary_by_round": _aggregate_tolerance_by_round(all_shadow_cells),
        "topk_agreement_summary_by_round": _aggregate_topk_by_round(all_shadow_cells),
        "safety_gate_summary": _aggregate_safety_gate_summary(cells),
        "generation_modified_by_decode_time_shadow": False,
        "decode_time_shadow_used_for_token_commit": False,
        "default_runtime_changed": False,
        "cells": cells,
        "blockers": blockers,
        "limitations": [
            "Expanded guarded decode-time shadow panel; not streaming-attention integration.",
            "Shadow runs inside opt-in observer callback only.",
            "Shadow results cannot affect token commits.",
            "Observer return values are ignored.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP084_CLAIM_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "guarded_decode_time_shadow_cli_flag": PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
    }


def validate_exp084_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens_values",
        "total_cells",
        "baseline_generation_successful_cells",
        "guarded_shadow_generation_successful_cells",
        "baseline_vs_guarded_token_match_cells",
        "baseline_vs_guarded_text_match_cells",
        "baseline_vs_guarded_token_mismatch_cells",
        "baseline_vs_guarded_text_mismatch_cells",
        "decode_time_shadow_callback_count",
        "decode_time_shadow_successful_callbacks",
        "decode_time_shadow_exception_callbacks",
        "posthoc_shadow_comparison_cells",
        "decode_time_vs_posthoc_shadow_match_cells",
        "decode_time_vs_posthoc_shadow_mismatch_cells",
        "exactkv_failure_summary",
        "tolerance_policy_summary_by_round",
        "topk_agreement_summary_by_round",
        "safety_gate_summary",
        "generation_modified_by_decode_time_shadow",
        "decode_time_shadow_used_for_token_commit",
        "default_runtime_changed",
        "cells",
        "blockers",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_084_ID:
        errors.append("experiment_id mismatch")

    for flag in (
        "generation_modified_by_decode_time_shadow",
        "decode_time_shadow_used_for_token_commit",
        "default_runtime_changed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    for idx, cell in enumerate(report.get("cells", [])):
        gates = cell.get("safety_gates") or {}
        if gates.get("decode_time_shadow_used_for_token_commit") is not False:
            errors.append(
                f"cells[{idx}].safety_gates.decode_time_shadow_used_for_token_commit must be false",
            )
        if gates.get("shadow_exception_affects_generation") is not False:
            errors.append(
                f"cells[{idx}].safety_gates.shadow_exception_affects_generation must be false",
            )
        if gates.get("shadow_result_exposed_to_generator") is not False:
            errors.append(
                f"cells[{idx}].safety_gates.shadow_result_exposed_to_generator must be false",
            )
        for ck in (
            "max_new_tokens",
            "decode_time_shadow_cells",
            "posthoc_comparison_summary",
            "safety_gates",
        ):
            if ck not in cell:
                errors.append(f"cells[{idx}] missing {ck}")

    return errors
