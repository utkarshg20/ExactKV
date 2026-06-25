"""L4 minimal runtime coupling layer (Phase 21M / Exp 114).

First real inference-driven verification path: loads model outputs, extracts
round-log proposals, runs Stage 3 trace-only verifier comparison. No L4 commit.
Must not be imported by default ExactKVGenerator runtime paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_noop_opt_in_scaffold import (
    DEFAULT_COMPRESSORS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PROMPTS,
    DEFAULT_MODEL_ID,
    default_noop_panel_prompts,
    run_baseline_generation_external,
)
from exactkv.safety.l4_stage3_verifier_mediated_dry_run_scaffold import (
    TERMINAL_STATES,
    execute_stage3_dry_run,
    simulate_prefix_walk,
)
from exactkv.safety.l4_trace_only_dry_run_scaffold import (
    _trace_get,
    extract_proposal_evidence_from_round_trace,
    extract_verifier_evidence_from_round_trace,
)
from exactkv.safety.l4_verifier_evidence_trace_schema_design import TRACE_SCHEMA_VERSION
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_114_ID = "exp114_l4_minimal_runtime_coupling_layer"
DEFAULT_EXP114_REPORT = Path(
    "reports/experiment_114_l4_minimal_runtime_coupling_layer.json",
)
PHASE_21M = "21M"
STAGE = "minimal_runtime_coupling"
MODE = "inference_driven_trace_only_verification"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21M = "phase21n_l4_runtime_coupling_stress_panel"
FORBIDDEN_NEXT_PHASES_21M: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_default_runtime_modification",
    "l4_verifier_in_loop_execution",
)

PANEL_OUTCOME_COMPLETE = "runtime_coupling_panel_complete"
PANEL_OUTCOME_INCOMPLETE = "runtime_coupling_panel_incomplete"
PANEL_OUTCOME_BLOCKED = "runtime_coupling_panel_blocked"

STAGE3_DECISIONS: tuple[str, ...] = TERMINAL_STATES

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

INTERPRETATION_NOTE = (
    "Runtime coupling trace records are diagnostic only; not commit authority."
)

CREATED_BY = "l4_runtime_coupling"


@dataclass(frozen=True)
class ModelOutputRecord:
    """One real or cached inference cell output."""

    model_name: str
    prompt_id: str
    prompt_text: str
    compressor: str
    max_new_tokens: int
    generation_output: dict[str, Any]
    ingestion_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RealProposalRound:
    """Draft proposal extracted from real model round logs."""

    round_index: int
    trace_id: str
    proposal_token_ids: tuple[int, ...]
    proposal_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "trace_id": self.trace_id,
            "proposal_token_ids": list(self.proposal_token_ids),
            "proposal_source": self.proposal_source,
        }


@dataclass(frozen=True)
class VerifierComparisonResult:
    """Read-only verifier comparison outcome (no commit)."""

    decision: str
    prefix_match_length: int
    mismatch_index: int | None
    block_reason: str | None
    dry_run_decision_used_for_token_commit: bool = False
    exposed_to_generator: bool = False
    verifier_source_of_truth: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4RuntimeTraceRecord:
    """One inference-driven trace-only verification record."""

    model_output: dict[str, Any]
    proposal_tokens: tuple[int, ...]
    verifier_evidence: dict[str, Any]
    decision: str
    mismatch_index: int | None
    prefix_length: int
    round_index: int
    trace_id: str
    prompt_id: str
    compressor: str
    dry_run_decision_used_for_token_commit: bool = False
    exposed_to_generator: bool = False
    rollback_conceptual_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_output": self.model_output,
            "proposal_tokens": list(self.proposal_tokens),
            "verifier_evidence": self.verifier_evidence,
            "decision": self.decision,
            "mismatch_index": self.mismatch_index,
            "prefix_length": self.prefix_length,
            "round_index": self.round_index,
            "trace_id": self.trace_id,
            "prompt_id": self.prompt_id,
            "compressor": self.compressor,
            "dry_run_decision_used_for_token_commit": self.dry_run_decision_used_for_token_commit,
            "exposed_to_generator": self.exposed_to_generator,
            "rollback_conceptual_only": self.rollback_conceptual_only,
            "interpretation_note": INTERPRETATION_NOTE,
        }


@dataclass(frozen=True)
class L4RuntimeCouplingValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_ids(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(raw)
    return ()


def cache_key_for_cell(
    prompt_id: str,
    compressor: str,
    max_new_tokens: int,
    *,
    model_name: str = "",
) -> str:
    """Deterministic cache key for one panel cell."""
    if model_name:
        return f"{model_name}|{prompt_id}|{compressor}|{max_new_tokens}"
    return f"{prompt_id}|{compressor}|{max_new_tokens}"


def _cache_key(prompt_id: str, compressor: str, max_new_tokens: int = 0) -> str:
    return cache_key_for_cell(prompt_id, compressor, max_new_tokens)


def build_synthetic_model_output(
    *,
    prompt_id: str = "p0",
    compressor: str = "noop",
    draft_tokens: Sequence[int] = (10, 11, 12),
    verifier_tokens: Sequence[int] | None = None,
    round_index: int = 0,
) -> dict[str, Any]:
    """Deterministic generation output for unit tests (not a fake proposal source)."""
    verifier = (
        list(verifier_tokens)
        if verifier_tokens is not None
        else list(draft_tokens)
    )
    return {
        "generation_completed": True,
        "generated_token_ids": verifier[:2] if len(verifier) >= 2 else verifier,
        "generated_text": "out",
        "exactkv_failures": 0,
        "token_exact_match": True,
        "result_traces": [
            {
                "round_idx": round_index,
                "draft_tokens": list(draft_tokens),
                "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
                "acceptance": {
                    "draft_tokens": list(draft_tokens),
                    "verifier_tokens": verifier,
                    "accepted_tokens": list(draft_tokens[: len(verifier)])
                    if draft_tokens[: len(verifier)] == verifier[: len(draft_tokens)]
                    else [],
                    "correction_token": None,
                    "rejected_tokens": [],
                    "all_matched": list(draft_tokens) == list(verifier[: len(draft_tokens)]),
                    "num_accepted": min(len(draft_tokens), len(verifier)),
                    "num_rejected": 0,
                },
            },
        ],
        "blockers": [],
    }


def load_model_outputs(
    model_name: str,
    prompts: Sequence[tuple[str, str]],
    max_new_tokens: int,
    *,
    compressors: Sequence[str] = DEFAULT_COMPRESSORS,
    device: str = "cpu",
    dtype: str = "float32",
    draft_len: int = 4,
    local_files_only: bool = False,
    cached_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
    allow_model_blocked: bool = True,
) -> tuple[ModelOutputRecord, ...]:
    """Load real HF inference outputs or explicit cached outputs per prompt/compressor."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    runnable, _blocked = resolve_panel_compressors(compressors)
    records: list[ModelOutputRecord] = []

    runtime: Any | None = None
    if generation_fn is None and cached_outputs is None:
        try:
            if runtime_loader is not None:
                runtime = runtime_loader(
                    model_id=model_name,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            else:
                from exactkv.runtime.model_runtime import ModelRuntime

                runtime = ModelRuntime(
                    model_name,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
        except Exception as exc:  # noqa: BLE001
            if not allow_model_blocked:
                raise
            return tuple(
                ModelOutputRecord(
                    model_name=model_name,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    compressor=compressor,
                    max_new_tokens=max_new_tokens,
                    generation_output={
                        "generation_completed": False,
                        "blockers": [f"{type(exc).__name__}: {exc}"],
                    },
                    ingestion_source="blocked",
                )
                for prompt_id, prompt_text in prompts
                for compressor in runnable
            )

    for prompt_id, prompt_text in prompts:
        for compressor in runnable:
            cache = cached_outputs or {}
            cached = cache.get(_cache_key(prompt_id, compressor, max_new_tokens))
            if cached is not None:
                gen_out = dict(cached)
                source = "cached_outputs"
            elif generation_fn is not None:
                gen_out = generation_fn(
                    prompt=prompt_text,
                    prompt_id=prompt_id,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                    model_id=model_name,
                )
                source = "generation_fn"
            elif runtime is not None:
                gen_out = run_baseline_generation_external(
                    runtime=runtime,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                    draft_len=draft_len,
                )
                source = "huggingface"
            else:
                gen_out = {
                    "generation_completed": False,
                    "blockers": ["no runtime or cached outputs"],
                }
                source = "blocked"

            records.append(
                ModelOutputRecord(
                    model_name=model_name,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    compressor=compressor,
                    max_new_tokens=max_new_tokens,
                    generation_output=gen_out,
                    ingestion_source=source,
                ),
            )
    return tuple(records)


def extract_proposals_from_model_outputs(
    model_output: Mapping[str, Any],
    *,
    prompt_id: str = "unknown",
    compressor: str = "noop",
) -> tuple[RealProposalRound, ...]:
    """Extract real round-log draft proposals from inference output traces."""
    traces = model_output.get("result_traces") or model_output.get("exactkv_traces") or []
    proposals: list[RealProposalRound] = []

    for trace in traces:
        round_index = int(
            _trace_get(trace, "round_index", _trace_get(trace, "round_idx", len(proposals)))
            or len(proposals),
        )
        proposal_ids, proposal_source = extract_proposal_evidence_from_round_trace(trace)
        trace_id = f"{prompt_id}|{compressor}|round_{round_index}"
        proposals.append(
            RealProposalRound(
                round_index=round_index,
                trace_id=trace_id,
                proposal_token_ids=proposal_ids,
                proposal_source=proposal_source,
            ),
        )
    return tuple(proposals)


def _normalize_verifier_source(raw_source: str | None, *, available: bool) -> str:
    if not available:
        return "verifier_exception_or_block_reason"
    if raw_source in (
        "full_kv_verifier_output_tokens",
        "verifier_comparison_output_for_proposal",
        "verifier_matching_prefix_evidence",
        "verifier_mismatch_evidence",
    ):
        return raw_source
    return "full_kv_verifier_output_tokens"


def _build_verifier_evidence_block(
    *,
    proposal: Sequence[int],
    verifier: Sequence[int],
    verifier_source: str,
) -> dict[str, Any]:
    normalized_source = _normalize_verifier_source(verifier_source, available=True)
    prefix_len, mismatch_idx, status = simulate_prefix_walk(proposal, verifier)
    matching = list(proposal[:prefix_len])
    if mismatch_idx is not None:
        rejected = list(proposal[mismatch_idx:])
    elif len(proposal) <= len(verifier):
        rejected = []
    else:
        rejected = list(proposal[len(verifier) :])
        mismatch_idx = len(verifier)

    decision_status = "all_match" if status == "ACCEPT_PREFIX" and not rejected else "partial_match"
    if mismatch_idx == 0:
        decision_status = "first_token_mismatch"

    return {
        "verifier_evidence_available": True,
        "verifier_evidence_source": normalized_source,
        "verifier_evidence_token_ids": list(verifier),
        "verifier_evidence_text": None,
        "verifier_evidence_is_full_kv": True,
        "verifier_evidence_is_authoritative": True,
        "verifier_checked_proposal_token_ids": list(proposal),
        "verifier_matching_prefix_token_ids": matching,
        "verifier_rejected_suffix_token_ids": rejected,
        "verifier_first_mismatch_index": mismatch_idx,
        "verifier_decision_status": decision_status,
        "verifier_exception": None,
        "verifier_block_reason": None,
        "verifier_trace_complete": True,
    }


def build_verifier_evidence_from_round_trace(
    trace: Any,
    *,
    proposal_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Build explicit verifier evidence dict from a real round trace."""
    verifier_ids, verifier_source = extract_verifier_evidence_from_round_trace(trace)
    if not verifier_ids or verifier_source is None:
        return {
            "verifier_evidence_available": False,
            "verifier_evidence_source": "verifier_exception_or_block_reason",
            "verifier_evidence_token_ids": [],
            "verifier_evidence_text": None,
            "verifier_evidence_is_full_kv": False,
            "verifier_evidence_is_authoritative": False,
            "verifier_checked_proposal_token_ids": list(proposal_token_ids),
            "verifier_matching_prefix_token_ids": [],
            "verifier_rejected_suffix_token_ids": [],
            "verifier_first_mismatch_index": None,
            "verifier_decision_status": "blocked",
            "verifier_exception": None,
            "verifier_block_reason": "no explicit verifier evidence in trace",
            "verifier_trace_complete": True,
        }
    return _build_verifier_evidence_block(
        proposal=proposal_token_ids,
        verifier=verifier_ids,
        verifier_source=verifier_source,
    )


def run_verifier_comparison(
    proposal_tokens: Sequence[int],
    verifier_evidence: Mapping[str, Any],
) -> VerifierComparisonResult:
    """Read-only prefix comparison; never commits tokens."""
    if verifier_evidence.get("trace_schema_version") == "corrupted_v99":
        return VerifierComparisonResult(
            decision="INVALID_TRACE",
            prefix_match_length=0,
            mismatch_index=None,
            block_reason="invalid trace schema version",
        )

    if not verifier_evidence.get("verifier_evidence_available"):
        return VerifierComparisonResult(
            decision="BLOCK_MISSING_EVIDENCE",
            prefix_match_length=0,
            mismatch_index=None,
            block_reason=str(
                verifier_evidence.get("verifier_block_reason")
                or "missing_verifier_evidence",
            ),
        )

    verifier_ids = _normalize_ids(verifier_evidence.get("verifier_evidence_token_ids"))
    if not verifier_ids:
        return VerifierComparisonResult(
            decision="BLOCK_MISSING_EVIDENCE",
            prefix_match_length=0,
            mismatch_index=None,
            block_reason="missing_verifier_evidence_token_ids",
        )

    prefix_len, mismatch_idx, terminal = simulate_prefix_walk(proposal_tokens, verifier_ids)
    return VerifierComparisonResult(
        decision=terminal,
        prefix_match_length=prefix_len,
        mismatch_index=mismatch_idx,
        block_reason=(
            f"mismatch_at_index_{mismatch_idx}" if terminal == "REJECT" else None
        ),
    )


def build_schema_record_from_model_round(
    *,
    cell_id: str,
    prompt_id: str,
    compressor: str,
    round_index: int,
    trace: Any,
    proposal: RealProposalRound,
) -> dict[str, Any]:
    """Build L4 schema v1 record from real inference round trace."""
    verifier_block = build_verifier_evidence_from_round_trace(
        trace,
        proposal_token_ids=proposal.proposal_token_ids,
    )
    return {
        "cell_id": cell_id,
        "prompt_id": prompt_id,
        "compressor": compressor,
        "round_index": round_index,
        "proposal_source": proposal.proposal_source,
        "proposal_token_ids": list(proposal.proposal_token_ids),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "created_by": CREATED_BY,
        "diagnostic_only": True,
        **verifier_block,
    }


def run_stage3_decision_engine(
    schema_record: Mapping[str, Any],
    *,
    case_id: str,
) -> VerifierComparisonResult:
    """Run Stage 3 dry-run decision engine on one schema record (trace-only)."""
    execution = execute_stage3_dry_run(schema_record, case_id=case_id)
    terminal = execution.dry_run_result.decision_status
    return VerifierComparisonResult(
        decision=terminal,
        prefix_match_length=execution.decision_graph_trace.prefix_match_length,
        mismatch_index=execution.decision_graph_trace.first_mismatch_index,
        block_reason=execution.dry_run_result.block_reason,
    )


def build_runtime_trace_records(
    model_record: ModelOutputRecord,
) -> tuple[L4RuntimeTraceRecord, ...]:
    """Full coupling pipeline for one model output cell."""
    gen = model_record.generation_output
    if not gen.get("generation_completed"):
        return ()

    cell_id = (
        f"{model_record.model_name}|{model_record.prompt_id}|"
        f"{model_record.compressor}|{model_record.max_new_tokens}"
    )
    traces = gen.get("result_traces") or gen.get("exactkv_traces") or []
    proposals = extract_proposals_from_model_outputs(
        gen,
        prompt_id=model_record.prompt_id,
        compressor=model_record.compressor,
    )
    proposal_by_round = {p.round_index: p for p in proposals}

    records: list[L4RuntimeTraceRecord] = []
    for trace in traces:
        round_index = int(
            _trace_get(trace, "round_index", _trace_get(trace, "round_idx", 0)) or 0,
        )
        proposal = proposal_by_round.get(round_index)
        if proposal is None:
            continue

        schema_record = build_schema_record_from_model_round(
            cell_id=cell_id,
            prompt_id=model_record.prompt_id,
            compressor=model_record.compressor,
            round_index=round_index,
            trace=trace,
            proposal=proposal,
        )
        comparison = run_stage3_decision_engine(
            schema_record,
            case_id=proposal.trace_id,
        )
        records.append(
            L4RuntimeTraceRecord(
                model_output={
                    "prompt_id": model_record.prompt_id,
                    "compressor": model_record.compressor,
                    "generated_token_ids": gen.get("generated_token_ids"),
                    "exactkv_failures": gen.get("exactkv_failures"),
                    "ingestion_source": model_record.ingestion_source,
                },
                proposal_tokens=proposal.proposal_token_ids,
                verifier_evidence=schema_record,
                decision=comparison.decision,
                mismatch_index=comparison.mismatch_index,
                prefix_length=comparison.prefix_match_length,
                round_index=round_index,
                trace_id=proposal.trace_id,
                prompt_id=model_record.prompt_id,
                compressor=model_record.compressor,
            ),
        )
    return tuple(records)


def _default_safety_gates() -> dict[str, bool]:
    return {
        "default_runtime_changed": False,
        "exactkv_generator_modified": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "l4_runtime_commit_implemented": False,
        "dry_run_decision_used_for_token_commit": False,
        "exposed_to_generator": False,
        "verifier_read_only_comparison": True,
        "rollback_execution_performed": False,
        "rollback_conceptual_only": True,
    }


def build_panel_cell_from_model_record(model_record: ModelOutputRecord) -> dict[str, Any]:
    gen = model_record.generation_output
    trace_records = build_runtime_trace_records(model_record)
    decisions = [r.decision for r in trace_records]
    safety = _default_safety_gates()
    blockers: list[str] = list(gen.get("blockers") or [])

    if not gen.get("generation_completed"):
        blockers.append("generation_incomplete")

    return {
        "model_name": model_record.model_name,
        "prompt_id": model_record.prompt_id,
        "compressor": model_record.compressor,
        "max_new_tokens": model_record.max_new_tokens,
        "ingestion_source": model_record.ingestion_source,
        "generation_completed": bool(gen.get("generation_completed")),
        "exactkv_failures": gen.get("exactkv_failures"),
        "trace_record_count": len(trace_records),
        "decisions": decisions,
        "trace_records": [r.to_dict() for r in trace_records],
        "safety_gates": safety,
        "blockers": blockers,
    }


def validate_exp114_panel_report(report: Mapping[str, Any]) -> L4RuntimeCouplingValidationResult:
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "panel_outcome",
        "model_name",
        "trace_records_total",
        "cells",
        "runtime_commit_authorized",
        "l4_activation",
        "model_experiments_run",
        "safety_gate_summary",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_114_ID:
        errors.append("experiment_id mismatch")

    if report.get("runtime_commit_authorized") is not False:
        errors.append("runtime_commit_authorized must be false")

    if report.get("l4_activation") is not False:
        errors.append("l4_activation must be false")

    if report.get("exactkv_generator_modified") is not False:
        errors.append("exactkv_generator_modified must be false")

    if report.get("default_runtime_changed") is not False:
        errors.append("default_runtime_changed must be false")

    for cell in report.get("cells") or []:
        for rec in cell.get("trace_records") or []:
            if rec.get("dry_run_decision_used_for_token_commit") is not False:
                errors.append("dry_run_decision_used_for_token_commit must be false")
            if rec.get("exposed_to_generator") is not False:
                errors.append("exposed_to_generator must be false")

    return L4RuntimeCouplingValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
    )


def run_exp114_l4_minimal_runtime_coupling_panel(
    *,
    model_name: str = DEFAULT_MODEL_ID,
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = DEFAULT_MAX_PROMPTS,
    compressors: Sequence[str] = DEFAULT_COMPRESSORS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: str = "cpu",
    dtype: str = "float32",
    draft_len: int = 4,
    local_files_only: bool = False,
    cached_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
    allow_model_blocked: bool = True,
) -> dict[str, Any]:
    """Run Phase 21M minimal runtime coupling panel (small scale, trace-only)."""
    prompt_panel = (
        list(prompts) if prompts is not None else default_noop_panel_prompts(max_prompts)
    )

    model_outputs = load_model_outputs(
        model_name,
        prompt_panel,
        max_new_tokens,
        compressors=compressors,
        device=device,
        dtype=dtype,
        draft_len=draft_len,
        local_files_only=local_files_only,
        cached_outputs=cached_outputs,
        generation_fn=generation_fn,
        runtime_loader=runtime_loader,
        allow_model_blocked=allow_model_blocked,
    )

    cells = [build_panel_cell_from_model_record(m) for m in model_outputs]
    trace_records_total = sum(c.get("trace_record_count", 0) for c in cells)
    completed = sum(1 for c in cells if c.get("generation_completed"))
    decision_counts: dict[str, int] = {d: 0 for d in STAGE3_DECISIONS}
    for cell in cells:
        for decision in cell.get("decisions") or []:
            if decision in decision_counts:
                decision_counts[decision] += 1
            else:
                decision_counts[str(decision)] = decision_counts.get(str(decision), 0) + 1

    model_experiments_run = any(
        m.ingestion_source in ("huggingface", "generation_fn") for m in model_outputs
    )

    if not cells:
        status = "blocked"
        panel_outcome = PANEL_OUTCOME_BLOCKED
    elif completed == 0:
        status = "blocked"
        panel_outcome = PANEL_OUTCOME_BLOCKED
    elif completed == len(cells) and trace_records_total > 0:
        status = "panel_complete"
        panel_outcome = PANEL_OUTCOME_COMPLETE
    elif completed > 0:
        status = "panel_partial"
        panel_outcome = PANEL_OUTCOME_INCOMPLETE
    else:
        status = "failed"
        panel_outcome = PANEL_OUTCOME_INCOMPLETE

    report = {
        "experiment_id": EXPERIMENT_114_ID,
        "status": status,
        "panel_outcome": panel_outcome,
        "phase": PHASE_21M,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "model_name": model_name,
        "device": device,
        "dtype": dtype,
        "compressors": list(compressors),
        "max_new_tokens": max_new_tokens,
        "prompt_count": len(prompt_panel),
        "total_cells": len(cells),
        "successful_generation_cells": completed,
        "trace_records_total": trace_records_total,
        "decision_status_counts": decision_counts,
        "cells": cells,
        "safety_gate_summary": _default_safety_gates(),
        "runtime_commit_authorized": False,
        "l4_activation": False,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "model_experiments_run": model_experiments_run,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21M,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21M),
        "implementation_blockers_remaining": [
            {"blocker_id": "l4_runtime_commit", "description": text}
            for text in L4_IMPLEMENTATION_BLOCKERS
            if "commit" in text.lower() or "runtime" in text.lower()
        ][:6],
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "Minimal runtime coupling layer only; trace-only verification.",
            "ExactKVGenerator and default runtime unchanged.",
            "No L4 commit; no token commits from dry-run decisions.",
            "Rollback is conceptual flag only; no rollback execution.",
            "Small panel only; no scaling, GPU optimization, or serving integration.",
            "No speed, throughput, latency, serving, or memory claims.",
        ],
    }
    report["validation_result"] = validate_exp114_panel_report(report).to_dict()
    return report


def validate_exp114_report(report: Mapping[str, Any]) -> list[str]:
    return list(validate_exp114_panel_report(report).errors)
