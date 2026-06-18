"""L3 guarded draft-shadow no-commit scaffold (Phase 18B / Exp 091).

Draft-shadow proposals are diagnostic only and cannot affect token commits.
Not L4 verifier-mediated acceptance or streaming-attention token-commit integration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.attention.decode_time_shadow_observer import (
    _run_guarded_shadow_generation,
    _token_lists_match,
    default_exp083_prompts,
)
from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS
from exactkv.attention.live_round_observer import _run_baseline_generation
from exactkv.safety.integration_safety_spec import (
    IntegrationProposal,
    NO_PERFORMANCE_CLAIMS_NOTE,
    TOPK_INTERPRETATION_NOTE,
    validate_integration_proposal,
)

EXPERIMENT_091_ID = "exp091_guarded_draft_shadow_no_commit_scaffold"
DEFAULT_EXP091_REPORT = Path(
    "reports/experiment_091_guarded_draft_shadow_no_commit_scaffold.json",
)
PHASE_18B = "18B"
SAFETY_LEVEL = "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT"

PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG = "--guarded-draft-shadow-no-commit"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_COMPRESSORS: tuple[str, ...] = ("noop", "int8")

PROPOSAL_SOURCE_SYNTHETIC = "synthetic_shadow_provider"
PROPOSAL_SOURCE_DECODE_TOP1 = "decode_time_shadow_top1"
PROPOSAL_SOURCE_BLOCKED = "blocked_no_provider"

PROPOSAL_SOURCES: tuple[str, ...] = (
    PROPOSAL_SOURCE_SYNTHETIC,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_BLOCKED,
)

PROPOSAL_INTERPRETATION_NOTE = (
    "Proposal match rate is supplementary diagnostic only; not an exactness guarantee."
)

RECOMMENDED_NEXT_PHASE = "phase18c_guarded_draft_shadow_panel_validation"

EXPERIMENT_092_ID = "exp092_guarded_draft_shadow_panel_validation"
DEFAULT_EXP092_REPORT = Path(
    "reports/experiment_092_guarded_draft_shadow_panel_validation.json",
)
PHASE_18C = "18C"

DEFAULT_PANEL_COMPRESSORS: tuple[str, ...] = ("noop", "int8", "int4_sim", "k8_v4_sim")
DEFAULT_MAX_NEW_TOKENS_VALUES: tuple[int, ...] = (4, 8)
DEFAULT_PANEL_PROMPTS = 4

RECOMMENDED_NEXT_PHASE_18C = "phase18d_shadow_top1_extraction_hardening"

EXPERIMENT_093_ID = "exp093_shadow_top1_extraction_hardening"
DEFAULT_EXP093_REPORT = Path(
    "reports/experiment_093_shadow_top1_extraction_hardening.json",
)
PHASE_18D = "18D"
RECOMMENDED_NEXT_PHASE_18D = "phase18e_l3_shadow_proposal_provenance_audit"

EXTRACTION_SOURCES_ALLOWED: tuple[str, ...] = (
    "shadow_top1_token_id",
    "diagnostic_proposal_token_id",
    "topk_agreement_metrics.shadow_top1_token_id",
    "streaming_top1_token_id",
    "streaming_vs_materialized_logit_metrics.other_top1_token_id",
    "streaming_vs_materialized_metrics.other_top1_token_id",
    "shadow_topk_token_ids[0]",
    "streaming_top5_token_ids[0]",
)

EXTRACTION_SOURCES_FORBIDDEN: tuple[str, ...] = (
    "committed_token_id",
    "generated_token_id",
    "baseline_token_id",
    "verifier_committed_token_id",
    "full_top1_token_id",
    "materialized_top1_token_id",
    "reference_top1_token_id",
    "streaming_vs_materialized_logit_metrics.reference_top1_token_id",
    "streaming_vs_materialized_metrics.reference_top1_token_id",
    "unsafe_retokenization_token_id",
    "proposed_token_text_from_generated_output",
)

L3_PANEL_SAFETY_SPEC_PROPOSAL = IntegrationProposal(
    proposal_id="exp092_l3_panel_self_validation",
    proposed_level=SAFETY_LEVEL,
    opt_in_only=True,
    modifies_default_runtime=False,
    verifier_source_of_truth=True,
    shadow_can_commit_directly=False,
    compressed_draft_can_commit_without_verifier=False,
    fallback_to_baseline=True,
    reports_exactkv_failures=True,
    hides_token_divergence=False,
    makes_performance_claim=False,
    makes_memory_claim=False,
    makes_serving_claim=False,
    makes_vericache_claim=False,
)

L3_SCAFFOLD_SAFETY_SPEC_PROPOSAL = IntegrationProposal(
    proposal_id="exp091_l3_scaffold_self_validation",
    proposed_level=SAFETY_LEVEL,
    opt_in_only=True,
    modifies_default_runtime=False,
    verifier_source_of_truth=True,
    shadow_can_commit_directly=False,
    compressed_draft_can_commit_without_verifier=False,
    fallback_to_baseline=True,
    reports_exactkv_failures=True,
    hides_token_divergence=False,
    makes_performance_claim=False,
    makes_memory_claim=False,
    makes_serving_claim=False,
    makes_vericache_claim=False,
)


def _frozen_metadata(data: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not data:
        return ()
    return tuple((str(k), str(v)) for k, v in sorted(data.items()))


@dataclass(frozen=True)
class ShadowTop1ExtractionResult:
    extraction_status: str
    proposed_token_id: int | None
    proposed_token_text: str | None
    extraction_source_field: str | None
    extraction_confidence: str
    block_reason: str | None
    is_shadow_derived: bool
    uses_committed_token: bool
    uses_baseline_token: bool
    exception: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nested_mapping_get(
    data: Mapping[str, Any],
    path: str,
) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _topk_rank0_token_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    seq = list(value)
    if not seq:
        return None
    return int(seq[0])


def extract_shadow_top1_candidate(
    shadow_output: Mapping[str, Any],
    *,
    allow_unsafe_retokenization: bool = False,
) -> ShadowTop1ExtractionResult:
    """Safely extract a shadow-derived top-1 proposal candidate from shadow diagnostics."""
    blocked_base = {
        "proposed_token_id": None,
        "proposed_token_text": None,
        "extraction_source_field": None,
        "extraction_confidence": "blocked",
        "is_shadow_derived": False,
        "uses_committed_token": False,
        "uses_baseline_token": False,
        "exception": None,
    }

    if (
        shadow_output.get("unsafe_retokenization_token_id") is not None
        and not allow_unsafe_retokenization
    ):
        return ShadowTop1ExtractionResult(
            extraction_status="blocked",
            block_reason="unsafe retokenization disabled by default",
            **blocked_base,
        )

    if (
        shadow_output.get("proposed_token_text_from_generated_output")
        and not allow_unsafe_retokenization
    ):
        return ShadowTop1ExtractionResult(
            extraction_status="blocked",
            block_reason="unsafe retokenization disabled by default",
            **blocked_base,
        )

    explicit_source = shadow_output.get("_extraction_source_field")
    if isinstance(explicit_source, str) and explicit_source in EXTRACTION_SOURCES_FORBIDDEN:
        return ShadowTop1ExtractionResult(
            extraction_status="unsafe_rejected",
            block_reason=f"forbidden extraction source: {explicit_source}",
            **blocked_base,
        )

    safe_candidates: list[tuple[str, str, int | None]] = [
        ("shadow_top1_token_id", "explicit_field", shadow_output.get("shadow_top1_token_id")),
        (
            "diagnostic_proposal_token_id",
            "explicit_field",
            shadow_output.get("diagnostic_proposal_token_id"),
        ),
        (
            "topk_agreement_metrics.shadow_top1_token_id",
            "explicit_field",
            _nested_mapping_get(shadow_output, "topk_agreement_metrics.shadow_top1_token_id"),
        ),
        (
            "streaming_top1_token_id",
            "explicit_field",
            shadow_output.get("streaming_top1_token_id"),
        ),
        (
            "streaming_vs_materialized_logit_metrics.other_top1_token_id",
            "explicit_field",
            _nested_mapping_get(
                shadow_output, "streaming_vs_materialized_logit_metrics.other_top1_token_id",
            ),
        ),
        (
            "streaming_vs_materialized_metrics.other_top1_token_id",
            "explicit_field",
            _nested_mapping_get(
                shadow_output, "streaming_vs_materialized_metrics.other_top1_token_id",
            ),
        ),
        (
            "shadow_topk_token_ids[0]",
            "topk_rank0",
            _topk_rank0_token_id(shadow_output.get("shadow_topk_token_ids")),
        ),
        (
            "streaming_top5_token_ids[0]",
            "topk_rank0",
            _topk_rank0_token_id(shadow_output.get("streaming_top5_token_ids")),
        ),
    ]

    for source_field, confidence, raw_token in safe_candidates:
        if raw_token is None:
            continue
        token_id = int(raw_token)
        uses_committed = source_field in (
            "committed_token_id",
            "generated_token_id",
            "verifier_committed_token_id",
        )
        uses_baseline = source_field == "baseline_token_id"
        is_shadow = source_field in EXTRACTION_SOURCES_ALLOWED
        if not is_shadow or uses_committed or uses_baseline:
            return ShadowTop1ExtractionResult(
                extraction_status="unsafe_rejected",
                proposed_token_id=None,
                proposed_token_text=None,
                extraction_source_field=source_field,
                extraction_confidence="unsafe_rejected",
                block_reason="extraction provenance requirements not met",
                is_shadow_derived=False,
                uses_committed_token=uses_committed,
                uses_baseline_token=uses_baseline,
                exception=None,
            )

        text = shadow_output.get("shadow_top1_token_text")
        if source_field == "diagnostic_proposal_token_id":
            text = shadow_output.get("diagnostic_proposal_token_text") or text

        return ShadowTop1ExtractionResult(
            extraction_status="success",
            proposed_token_id=token_id,
            proposed_token_text=text if isinstance(text, str) else None,
            extraction_source_field=source_field,
            extraction_confidence=confidence,
            block_reason=None,
            is_shadow_derived=True,
            uses_committed_token=False,
            uses_baseline_token=False,
            exception=None,
        )

    if shadow_output.get("committed_token_id") is not None and not any(
        shadow_output.get(k) is not None
        for k in (
            "shadow_top1_token_id",
            "streaming_top1_token_id",
            "diagnostic_proposal_token_id",
        )
    ):
        return ShadowTop1ExtractionResult(
            extraction_status="unsafe_rejected",
            block_reason="committed token source rejected",
            **blocked_base,
        )

    if shadow_output.get("baseline_token_id") is not None and not any(
        shadow_output.get(k) is not None
        for k in (
            "shadow_top1_token_id",
            "streaming_top1_token_id",
            "diagnostic_proposal_token_id",
        )
    ):
        return ShadowTop1ExtractionResult(
            extraction_status="unsafe_rejected",
            block_reason="baseline token source rejected",
            **blocked_base,
        )

    if shadow_output.get("shadow_status") not in (None, "shadow_complete"):
        block_reason = "no safe top1 extraction from shadow output"
    else:
        block_reason = "no explicit shadow top-1 diagnostic field available"

    return ShadowTop1ExtractionResult(
        extraction_status="blocked",
        block_reason=block_reason,
        **blocked_base,
    )


def _extraction_metadata(extraction: ShadowTop1ExtractionResult) -> tuple[tuple[str, str], ...]:
    d = extraction.to_dict()
    return tuple(
        (str(k), str(v) if v is not None else "")
        for k, v in sorted(d.items())
    )


@dataclass(frozen=True)
class GuardedDraftShadowProposal:
    round_index: int
    prompt_id: str
    compressor: str
    prefix_token_ids: tuple[int, ...]
    proposed_token_ids: tuple[int, ...]
    proposed_text: str | None
    proposal_source: str
    proposal_status: str
    exception: str | None
    metadata: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metadata"] = dict(self.metadata)
        return d


@dataclass(frozen=True)
class GuardedDraftShadowDecision:
    round_index: int
    accepted_for_commit: bool
    exposed_to_generator: bool
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedDraftShadowSafetyResult:
    proposal_used_for_token_commit: bool
    proposal_exposed_to_generator: bool
    proposal_return_value_ignored: bool
    proposal_exception_affects_generation: bool
    generated_output_modified_by_proposal: bool
    default_runtime_changed: bool

    @property
    def all_gates_ok(self) -> bool:
        return (
            self.proposal_used_for_token_commit is False
            and self.proposal_exposed_to_generator is False
            and self.proposal_return_value_ignored is True
            and self.proposal_exception_affects_generation is False
            and self.generated_output_modified_by_proposal is False
            and self.default_runtime_changed is False
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def to_gates_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedDraftShadowCell:
    prompt_id: str
    prompt_preview: str
    compressor: str
    baseline_generation_completed: bool
    draft_shadow_generation_completed: bool
    baseline_vs_draft_shadow_token_match: bool
    baseline_vs_draft_shadow_text_match: bool
    proposal_source: str
    proposal_count: int
    proposals: tuple[GuardedDraftShadowProposal, ...]
    decisions: tuple[GuardedDraftShadowDecision, ...]
    safety_results: tuple[GuardedDraftShadowSafetyResult, ...]
    exactkv_failures_baseline: int | None
    exactkv_failures_draft_shadow: int | None
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_preview": self.prompt_preview,
            "compressor": self.compressor,
            "baseline_generation_completed": self.baseline_generation_completed,
            "draft_shadow_generation_completed": self.draft_shadow_generation_completed,
            "baseline_vs_draft_shadow_token_match": self.baseline_vs_draft_shadow_token_match,
            "baseline_vs_draft_shadow_text_match": self.baseline_vs_draft_shadow_text_match,
            "proposal_source": self.proposal_source,
            "proposal_count": self.proposal_count,
            "proposals": [p.to_dict() for p in self.proposals],
            "proposal_match_summary": summarize_proposal_match(
                self.proposals,
                committed_token_ids=_committed_tokens_from_proposals(self.proposals),
            ),
            "exactkv_failures_baseline": self.exactkv_failures_baseline,
            "exactkv_failures_draft_shadow": self.exactkv_failures_draft_shadow,
            "safety_gates": _aggregate_cell_safety_gates(self),
            "blockers": list(self.blockers),
        }


def _committed_tokens_from_proposals(
    proposals: Sequence[GuardedDraftShadowProposal],
) -> list[int | None]:
    out: list[int | None] = []
    for prop in proposals:
        committed: int | None = None
        for k, v in prop.metadata:
            if k == "committed_token_id":
                committed = int(v)
        out.append(committed)
    return out


def default_no_commit_safety_result() -> GuardedDraftShadowSafetyResult:
    return GuardedDraftShadowSafetyResult(
        proposal_used_for_token_commit=False,
        proposal_exposed_to_generator=False,
        proposal_return_value_ignored=True,
        proposal_exception_affects_generation=False,
        generated_output_modified_by_proposal=False,
        default_runtime_changed=False,
    )


def default_no_commit_decision(round_index: int) -> GuardedDraftShadowDecision:
    return GuardedDraftShadowDecision(
        round_index=round_index,
        accepted_for_commit=False,
        exposed_to_generator=False,
        decision_reason="L3 no-commit scaffold: proposals are diagnostic only.",
    )


def proposal_block_reason(proposal: GuardedDraftShadowProposal) -> str | None:
    if proposal.proposal_status != "blocked":
        return None
    if proposal.exception:
        return proposal.exception
    for k, v in proposal.metadata:
        if k == "reason":
            return v
    return "blocked_unknown"


def summarize_proposal_coverage(
    proposals: Sequence[GuardedDraftShadowProposal],
) -> dict[str, Any]:
    successful = 0
    blocked = 0
    block_reasons: dict[str, int] = {}
    first_success: int | None = None
    first_blocked: int | None = None
    for prop in proposals:
        if prop.proposal_status == "complete" and prop.proposed_token_ids:
            successful += 1
            if first_success is None:
                first_success = prop.round_index
        else:
            blocked += 1
            if first_blocked is None:
                first_blocked = prop.round_index
            reason = proposal_block_reason(prop) or "blocked_unknown"
            block_reasons[reason] = block_reasons.get(reason, 0) + 1
    total = len(proposals)
    return {
        "total_proposals": total,
        "successful_proposals": successful,
        "blocked_proposals": blocked,
        "proposal_block_reasons": block_reasons,
        "first_successful_proposal_round": first_success,
        "first_blocked_proposal_round": first_blocked,
        "proposal_coverage_rate": successful / total if total else 0.0,
    }


def aggregate_proposal_block_reasons(
    cells: Sequence[dict[str, Any]],
) -> dict[str, int]:
    agg: dict[str, int] = {}
    for cell in cells:
        for reason, count in (cell.get("proposal_block_reasons") or {}).items():
            agg[reason] = agg.get(reason, 0) + int(count)
    return agg


def default_panel_prompts(max_prompts: int = DEFAULT_PANEL_PROMPTS) -> list[tuple[str, str]]:
    from exactkv.attention.generation_shadow_observer import default_exp080_prompts

    return default_exp080_prompts()[:max_prompts]


def proposal_to_report_dict(
    proposal: GuardedDraftShadowProposal,
    *,
    committed_token_id: int | None,
) -> dict[str, Any]:
    matched = (
        committed_token_id is not None
        and proposal.proposed_token_ids
        and proposal.proposed_token_ids[0] == committed_token_id
    )
    safety = default_no_commit_safety_result()
    meta = dict(proposal.metadata)
    extraction_fields = {
        "extraction_status": meta.get("extraction_status"),
        "proposed_token_id": (
            int(meta["proposed_token_id"])
            if meta.get("proposed_token_id") not in (None, "")
            else None
        ),
        "proposed_token_text": meta.get("proposed_token_text") or None,
        "extraction_source_field": meta.get("extraction_source_field") or None,
        "extraction_confidence": meta.get("extraction_confidence"),
        "block_reason": meta.get("block_reason") or proposal_block_reason(proposal),
        "is_shadow_derived": meta.get("is_shadow_derived") == "True",
        "uses_committed_token": meta.get("uses_committed_token") == "True",
        "uses_baseline_token": meta.get("uses_baseline_token") == "True",
        "exception": meta.get("exception") or proposal.exception,
    }
    return {
        **proposal.to_dict(),
        **extraction_fields,
        "matched_committed_token": matched,
        "proposal_used_for_token_commit": safety.proposal_used_for_token_commit,
        "proposal_exposed_to_generator": safety.proposal_exposed_to_generator,
        "interpretation_note": PROPOSAL_INTERPRETATION_NOTE,
    }


def summarize_proposal_match(
    proposals: Sequence[GuardedDraftShadowProposal],
    *,
    committed_token_ids: Sequence[int | None],
) -> dict[str, Any]:
    matching = 0
    not_matching = 0
    blocked = 0
    first_mismatch: int | None = None
    for prop, committed in zip(proposals, committed_token_ids, strict=False):
        if prop.proposal_status != "complete" or not prop.proposed_token_ids:
            blocked += 1
            continue
        if committed is None:
            not_matching += 1
            if first_mismatch is None:
                first_mismatch = prop.round_index
            continue
        if prop.proposed_token_ids[0] == committed:
            matching += 1
        else:
            not_matching += 1
            if first_mismatch is None:
                first_mismatch = prop.round_index
    total = len(proposals)
    rate = matching / total if total else 0.0
    return {
        "proposal_count": total,
        "proposals_matching_committed_token": matching,
        "proposals_not_matching_committed_token": not_matching,
        "blocked_proposals": blocked,
        "first_proposal_mismatch_round": first_mismatch,
        "proposal_match_rate": rate,
        "interpretation_note": PROPOSAL_INTERPRETATION_NOTE,
    }


def _aggregate_cell_safety_gates(cell: GuardedDraftShadowCell) -> dict[str, bool]:
    base = default_no_commit_safety_result().to_gates_dict()
    base["baseline_generation_completed"] = cell.baseline_generation_completed
    base["draft_shadow_generation_completed"] = cell.draft_shadow_generation_completed
    base["baseline_vs_draft_shadow_token_match"] = cell.baseline_vs_draft_shadow_token_match
    base["baseline_vs_draft_shadow_text_match"] = cell.baseline_vs_draft_shadow_text_match
    return base


def _cell_safety_gates_ok(gates: dict[str, bool]) -> bool:
    safety = default_no_commit_safety_result()
    for key, expected in safety.to_gates_dict().items():
        if gates.get(key) != expected:
            return False
    for key in (
        "baseline_generation_completed",
        "draft_shadow_generation_completed",
        "baseline_vs_draft_shadow_token_match",
        "baseline_vs_draft_shadow_text_match",
    ):
        if gates.get(key) is not True:
            return False
    return True


def build_synthetic_proposals(
    *,
    prompt_id: str,
    compressor: str,
    generated_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
) -> tuple[GuardedDraftShadowProposal, ...]:
    """Test provider: mirror committed tokens as diagnostic proposals."""
    proposals: list[GuardedDraftShadowProposal] = []
    prefix = tuple(prefix_token_ids)
    for rnd, token_id in enumerate(generated_token_ids):
        proposals.append(
            GuardedDraftShadowProposal(
                round_index=rnd,
                prompt_id=prompt_id,
                compressor=compressor,
                prefix_token_ids=prefix,
                proposed_token_ids=(int(token_id),),
                proposed_text=None,
                proposal_source=PROPOSAL_SOURCE_SYNTHETIC,
                proposal_status="complete",
                exception=None,
                metadata=(
                    ("committed_token_id", str(token_id)),
                    ("provider", PROPOSAL_SOURCE_SYNTHETIC),
                ),
            ),
        )
    return tuple(proposals)


def build_blocked_proposals(
    *,
    prompt_id: str,
    compressor: str,
    reason: str,
) -> tuple[GuardedDraftShadowProposal, ...]:
    return (
        GuardedDraftShadowProposal(
            round_index=0,
            prompt_id=prompt_id,
            compressor=compressor,
            prefix_token_ids=(),
            proposed_token_ids=(),
            proposed_text=None,
            proposal_source=PROPOSAL_SOURCE_BLOCKED,
            proposal_status="blocked",
            exception=reason,
            metadata=(("reason", reason),),
        ),
    )


def build_decode_time_shadow_top1_proposals(
    *,
    prompt_id: str,
    compressor: str,
    generated_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    posthoc_shadow_cells: Sequence[dict[str, Any]],
) -> tuple[GuardedDraftShadowProposal, ...]:
    """Extract streaming shadow top-1 token IDs when safely available."""
    proposals: list[GuardedDraftShadowProposal] = []
    prefix = tuple(prefix_token_ids)
    for rnd, committed in enumerate(generated_token_ids):
        cell = posthoc_shadow_cells[rnd] if rnd < len(posthoc_shadow_cells) else {}
        extraction = extract_shadow_top1_candidate(cell)

        if extraction.extraction_status == "unsafe_rejected":
            proposals.append(
                GuardedDraftShadowProposal(
                    round_index=rnd,
                    prompt_id=prompt_id,
                    compressor=compressor,
                    prefix_token_ids=prefix,
                    proposed_token_ids=(),
                    proposed_text=None,
                    proposal_source=PROPOSAL_SOURCE_BLOCKED,
                    proposal_status="blocked",
                    exception=extraction.block_reason,
                    metadata=_extraction_metadata(extraction),
                ),
            )
            continue

        if extraction.extraction_status != "success" or extraction.proposed_token_id is None:
            block_reason = extraction.block_reason or "no safe top1 extraction from shadow output"
            proposals.append(
                GuardedDraftShadowProposal(
                    round_index=rnd,
                    prompt_id=prompt_id,
                    compressor=compressor,
                    prefix_token_ids=prefix,
                    proposed_token_ids=(),
                    proposed_text=None,
                    proposal_source=PROPOSAL_SOURCE_BLOCKED,
                    proposal_status="blocked",
                    exception=block_reason,
                    metadata=_extraction_metadata(extraction)
                    + (("committed_token_id", str(committed)),),
                ),
            )
            continue

        proposals.append(
            GuardedDraftShadowProposal(
                round_index=rnd,
                prompt_id=prompt_id,
                compressor=compressor,
                prefix_token_ids=prefix,
                proposed_token_ids=(extraction.proposed_token_id,),
                proposed_text=extraction.proposed_token_text,
                proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
                proposal_status="complete",
                exception=None,
                metadata=_extraction_metadata(extraction)
                + (
                    ("committed_token_id", str(committed)),
                    ("provider", PROPOSAL_SOURCE_DECODE_TOP1),
                ),
            ),
        )
    return tuple(proposals)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if hasattr(value, "squeeze"):
        flat = value.squeeze().tolist()
        if isinstance(flat, int):
            return [flat]
        return [int(x) for x in flat]
    return [int(x) for x in list(value)]


def extract_proposals(
    *,
    proposal_source: str,
    prompt_id: str,
    compressor: str,
    draft_shadow_out: dict[str, Any],
    allow_provider_blocked: bool = True,
) -> tuple[GuardedDraftShadowProposal, ...]:
    gen_ids = _as_int_list(draft_shadow_out.get("generated_token_ids"))
    prompt_ids = _as_int_list(draft_shadow_out.get("prompt_ids"))
    prefix = prompt_ids if prompt_ids else []

    if proposal_source == PROPOSAL_SOURCE_SYNTHETIC:
        return build_synthetic_proposals(
            prompt_id=prompt_id,
            compressor=compressor,
            generated_token_ids=gen_ids,
            prefix_token_ids=prefix,
        )

    if proposal_source == PROPOSAL_SOURCE_DECODE_TOP1:
        from exactkv.attention.generation_shadow_observer import (
            run_posthoc_shadow_from_live_snapshots,
        )

        snaps = draft_shadow_out.get("live_snapshots") or []
        hf_model = draft_shadow_out.get("_hf_model")
        if not snaps or hf_model is None:
            if allow_provider_blocked:
                return build_blocked_proposals(
                    prompt_id=prompt_id,
                    compressor=compressor,
                    reason="missing snapshots or model for decode_time_shadow_top1",
                )
            return ()
        posthoc_cells, _blockers = run_posthoc_shadow_from_live_snapshots(
            snapshots=snaps,
            prompt_id=prompt_id,
            hf_model=hf_model,
            shadow_replay_fn=draft_shadow_out.get("_shadow_diagnostic_fn"),
            allow_shadow_fail=True,
        )
        if not posthoc_cells:
            if allow_provider_blocked:
                return build_blocked_proposals(
                    prompt_id=prompt_id,
                    compressor=compressor,
                    reason="post-hoc shadow produced no cells for top1 extraction",
                )
            return ()
        return build_decode_time_shadow_top1_proposals(
            prompt_id=prompt_id,
            compressor=compressor,
            generated_token_ids=gen_ids,
            prefix_token_ids=prefix,
            posthoc_shadow_cells=posthoc_cells,
        )

    return build_blocked_proposals(
        prompt_id=prompt_id,
        compressor=compressor,
        reason=f"unknown proposal source: {proposal_source}",
    )


def _run_draft_shadow_no_commit_generation(
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
    """Run guarded shadow generation path; proposals extracted separately."""
    out = _run_guarded_shadow_generation(
        runtime,
        prompt,
        prompt_id,
        max_new_tokens,
        compressor_name,
        draft_len,
        shadow_diagnostic_fn=shadow_diagnostic_fn,
        allow_shadow_fail=allow_shadow_fail,
    )
    out["_hf_model"] = getattr(runtime, "model", None)
    out["_shadow_diagnostic_fn"] = shadow_diagnostic_fn
    return out


def _build_panel_cell(
    *,
    prompt_id: str,
    prompt_text: str,
    compressor: str,
    max_new_tokens: int,
    proposal_source: str,
    runtime: Any | None,
    draft_len: int,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None,
    draft_shadow_generation_fn: Callable[..., dict[str, Any]] | None,
    allow_provider_blocked: bool,
) -> tuple[dict[str, Any], bool]:
    preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
    blockers: list[str] = []

    if baseline_generation_fn is not None:
        baseline = baseline_generation_fn(
            prompt=prompt_text,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
        )
    elif runtime is not None:
        baseline = _run_baseline_generation(
            runtime, prompt_text, max_new_tokens, compressor, draft_len,
        )
    else:
        baseline = {"generation_completed": False, "blockers": ["no runtime"]}

    if draft_shadow_generation_fn is not None:
        draft_shadow = draft_shadow_generation_fn(
            prompt=prompt_text,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
        )
    elif runtime is not None:
        draft_shadow = _run_draft_shadow_no_commit_generation(
            runtime,
            prompt_text,
            prompt_id,
            max_new_tokens,
            compressor,
            draft_len,
        )
    else:
        draft_shadow = {"generation_completed": False, "blockers": ["no runtime"]}

    baseline_ok = bool(baseline.get("generation_completed"))
    draft_ok = bool(draft_shadow.get("generation_completed"))
    tok_match = _token_lists_match(
        baseline.get("generated_token_ids"),
        draft_shadow.get("generated_token_ids"),
    )
    txt_match = baseline.get("generated_text") == draft_shadow.get("generated_text")
    if not tok_match:
        blockers.append("baseline_vs_draft_shadow_token_mismatch")
    if not txt_match:
        blockers.append("baseline_vs_draft_shadow_text_mismatch")

    proposals = extract_proposals(
        proposal_source=proposal_source,
        prompt_id=prompt_id,
        compressor=compressor,
        draft_shadow_out=draft_shadow,
        allow_provider_blocked=allow_provider_blocked,
    )
    decisions = tuple(default_no_commit_decision(p.round_index) for p in proposals)
    safety_results = tuple(default_no_commit_safety_result() for _ in proposals)

    committed_ids = _committed_tokens_from_proposals(proposals)
    prop_match = summarize_proposal_match(proposals, committed_token_ids=committed_ids)
    coverage = summarize_proposal_coverage(proposals)

    cell_obj = GuardedDraftShadowCell(
        prompt_id=prompt_id,
        prompt_preview=preview,
        compressor=compressor,
        baseline_generation_completed=baseline_ok,
        draft_shadow_generation_completed=draft_ok,
        baseline_vs_draft_shadow_token_match=tok_match,
        baseline_vs_draft_shadow_text_match=txt_match,
        proposal_source=proposal_source,
        proposal_count=len(proposals),
        proposals=proposals,
        decisions=decisions,
        safety_results=safety_results,
        exactkv_failures_baseline=baseline.get("exactkv_failures"),
        exactkv_failures_draft_shadow=draft_shadow.get("exactkv_failures"),
        blockers=tuple(blockers),
    )
    cell = cell_obj.to_dict()
    cell["max_new_tokens"] = max_new_tokens
    cell["total_proposals"] = coverage["total_proposals"]
    cell["successful_proposals"] = coverage["successful_proposals"]
    cell["blocked_proposals"] = coverage["blocked_proposals"]
    cell["proposal_block_reasons"] = coverage["proposal_block_reasons"]
    cell["first_successful_proposal_round"] = coverage["first_successful_proposal_round"]
    cell["first_blocked_proposal_round"] = coverage["first_blocked_proposal_round"]
    cell["proposal_match_summary"] = prop_match
    cell["token_exact_match_baseline"] = baseline.get("token_exact_match")
    cell["token_exact_match_draft_shadow"] = draft_shadow.get("token_exact_match")
    cell["proposals"] = [
        proposal_to_report_dict(p, committed_token_id=cid)
        for p, cid in zip(proposals, committed_ids, strict=False)
    ]

    gates = cell["safety_gates"]
    failed = not _cell_safety_gates_ok(gates)
    if failed and "safety_gate_failed" not in blockers:
        blockers.append("safety_gate_failed")
    cell["blockers"] = blockers
    return cell, failed


def run_exp091_guarded_draft_shadow_no_commit_scaffold(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = 2,
    max_new_tokens: int = 4,
    compressors_requested: Sequence[str] = DEFAULT_COMPRESSORS,
    proposal_source: str = PROPOSAL_SOURCE_SYNTHETIC,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_provider_blocked: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    draft_shadow_generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run Experiment 091 L3 guarded draft-shadow no-commit scaffold."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    prompt_panel = list(prompts) if prompts is not None else default_exp083_prompts()[:max_prompts]
    runnable, _blocked_comp = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if proposal_source not in PROPOSAL_SOURCES:
        blockers.append(f"unknown proposal_source: {proposal_source}")

    runtime: Any | None = None
    if baseline_generation_fn is None or draft_shadow_generation_fn is None:
        try:
            if runtime_loader is not None:
                runtime = runtime_loader(
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            else:
                from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
                from exactkv.runtime.model_runtime import ModelRuntime

                runtime = ModelRuntime(model_id, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"model load failed: {type(exc).__name__}: {exc}")

    safety_spec_validation = validate_integration_proposal(L3_SCAFFOLD_SAFETY_SPEC_PROPOSAL)
    if not safety_spec_validation["pass"]:
        blockers.append("safety_spec_validation_failed")

    cells: list[dict[str, Any]] = []
    baseline_ok = 0
    draft_ok = 0
    tok_match = 0
    txt_match = 0
    total_proposals = 0
    successful_proposals = 0
    blocked_proposals = 0
    failed_cells = 0
    sg_ok = 0

    b_fn = baseline_generation_fn
    d_fn = draft_shadow_generation_fn

    if runtime is None and (b_fn is None or d_fn is None):
        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                cells.append({
                    "prompt_id": prompt_id,
                    "prompt_preview": prompt_text[:80],
                    "compressor": compressor,
                    "baseline_generation_completed": False,
                    "draft_shadow_generation_completed": False,
                    "baseline_vs_draft_shadow_token_match": False,
                    "baseline_vs_draft_shadow_text_match": False,
                    "proposal_source": proposal_source,
                    "proposal_count": 0,
                    "proposals": [],
                    "proposal_match_summary": summarize_proposal_match([], committed_token_ids=[]),
                    "safety_gates": {},
                    "blockers": list(blockers),
                })
    else:
        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                cell, failed = _build_panel_cell(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    compressor=compressor,
                    max_new_tokens=max_new_tokens,
                    proposal_source=proposal_source,
                    runtime=runtime,
                    draft_len=draft_len,
                    baseline_generation_fn=b_fn,
                    draft_shadow_generation_fn=d_fn,
                    allow_provider_blocked=allow_provider_blocked,
                )
                cells.append(cell)
                if cell["baseline_generation_completed"]:
                    baseline_ok += 1
                if cell["draft_shadow_generation_completed"]:
                    draft_ok += 1
                if cell["baseline_vs_draft_shadow_token_match"]:
                    tok_match += 1
                if cell["baseline_vs_draft_shadow_text_match"]:
                    txt_match += 1
                total_proposals += cell["proposal_count"]
                for prop in cell.get("proposals") or []:
                    if prop.get("proposal_status") == "complete":
                        successful_proposals += 1
                    elif prop.get("proposal_status") == "blocked":
                        blocked_proposals += 1
                if _cell_safety_gates_ok(cell.get("safety_gates") or {}):
                    sg_ok += 1
                if failed:
                    failed_cells += 1

    proposal_match_agg = {
        "proposal_count": total_proposals,
        "proposals_matching_committed_token": sum(
            (c.get("proposal_match_summary") or {}).get("proposals_matching_committed_token", 0)
            for c in cells
        ),
        "proposals_not_matching_committed_token": sum(
            (c.get("proposal_match_summary") or {}).get("proposals_not_matching_committed_token", 0)
            for c in cells
        ),
        "blocked_proposals": blocked_proposals,
        "interpretation_note": PROPOSAL_INTERPRETATION_NOTE,
    }
    if cells:
        rates = [
            (c.get("proposal_match_summary") or {}).get("proposal_match_rate", 0.0)
            for c in cells
            if (c.get("proposal_match_summary") or {}).get("proposal_count", 0) > 0
        ]
        proposal_match_agg["mean_proposal_match_rate"] = (
            sum(rates) / len(rates) if rates else 0.0
        )

    total = len(cells)
    if not safety_spec_validation["pass"]:
        status = "failed"
    elif failed_cells > 0:
        status = "failed"
    elif baseline_ok == total and draft_ok == total and tok_match == total and total > 0:
        status = "scaffold_complete"
    elif baseline_ok > 0:
        status = "scaffold_partial"
    else:
        status = "blocked"

    safety_top = default_no_commit_safety_result()

    return {
        "experiment_id": EXPERIMENT_091_ID,
        "status": status,
        "phase": PHASE_18B,
        "safety_level": SAFETY_LEVEL,
        "safety_spec_validation": safety_spec_validation,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "proposal_source": proposal_source,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens": max_new_tokens,
        "total_cells": total,
        "baseline_generation_successful_cells": baseline_ok,
        "draft_shadow_generation_successful_cells": draft_ok,
        "baseline_vs_draft_shadow_token_match_cells": tok_match,
        "baseline_vs_draft_shadow_text_match_cells": txt_match,
        "total_proposals": total_proposals,
        "successful_proposals": successful_proposals,
        "blocked_proposals": blocked_proposals,
        "proposal_match_summary": proposal_match_agg,
        "exactkv_failure_summary": {
            "baseline_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_baseline") or 0) > 0
            ),
            "draft_shadow_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_draft_shadow") or 0) > 0
            ),
        },
        "safety_gate_summary": {
            "cells_all_gates_ok": sg_ok,
            "cells_with_gate_failure": total - sg_ok,
        },
        "proposal_used_for_token_commit": safety_top.proposal_used_for_token_commit,
        "proposal_exposed_to_generator": safety_top.proposal_exposed_to_generator,
        "generated_output_modified_by_proposal": (
            safety_top.generated_output_modified_by_proposal
        ),
        "default_runtime_changed": safety_top.default_runtime_changed,
        "cells": cells,
        "topk_interpretation_note": TOPK_INTERPRETATION_NOTE,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE,
        "claim_note": (
            "L3 guarded draft-shadow no-commit scaffold. Proposals are diagnostic only."
        ),
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "blockers": blockers,
        "limitations": [
            "L3 scaffold only; not L4 verifier-mediated compressed draft.",
            "Draft proposals cannot affect token commits or generator decisions.",
            "Proposal match rate is supplementary; not exactness.",
            "ExactKVGenerator and default runtime unchanged.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
    }


def validate_exp091_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "safety_level",
        "safety_spec_validation",
        "model_id",
        "device",
        "dtype",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens",
        "total_cells",
        "baseline_generation_successful_cells",
        "draft_shadow_generation_successful_cells",
        "baseline_vs_draft_shadow_token_match_cells",
        "baseline_vs_draft_shadow_text_match_cells",
        "total_proposals",
        "successful_proposals",
        "blocked_proposals",
        "proposal_match_summary",
        "exactkv_failure_summary",
        "safety_gate_summary",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "generated_output_modified_by_proposal",
        "default_runtime_changed",
        "blockers",
        "limitations",
        "no_performance_claims_note",
        "cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_091_ID:
        errors.append("experiment_id mismatch")

    if report.get("safety_level") != SAFETY_LEVEL:
        errors.append("safety_level mismatch")

    if report.get("proposal_used_for_token_commit") is not False:
        errors.append("proposal_used_for_token_commit must be false")

    if report.get("proposal_exposed_to_generator") is not False:
        errors.append("proposal_exposed_to_generator must be false")

    spec_val = report.get("safety_spec_validation") or {}
    if spec_val.get("pass") is not True:
        errors.append("safety_spec_validation must pass")

    for idx, cell in enumerate(report.get("cells") or []):
        for ck in ("proposal_source", "proposal_count", "safety_gates", "proposals"):
            if ck not in cell:
                errors.append(f"cells[{idx}] missing {ck}")

    return errors


def run_exp092_guarded_draft_shadow_panel_validation(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = DEFAULT_PANEL_PROMPTS,
    max_new_tokens_values: Sequence[int] = DEFAULT_MAX_NEW_TOKENS_VALUES,
    compressors_requested: Sequence[str] = DEFAULT_PANEL_COMPRESSORS,
    proposal_source: str = PROPOSAL_SOURCE_DECODE_TOP1,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_provider_blocked: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    draft_shadow_generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
    safety_spec_proposal: IntegrationProposal | None = None,
) -> dict[str, Any]:
    """Run Experiment 092 expanded L3 panel with proposal coverage diagnostics."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    prompt_panel = list(prompts) if prompts is not None else default_panel_prompts(max_prompts)
    runnable, _blocked_comp = resolve_panel_compressors(compressors_requested)
    mnt_values = list(max_new_tokens_values)

    blockers: list[str] = []
    if proposal_source not in PROPOSAL_SOURCES:
        blockers.append(f"unknown proposal_source: {proposal_source}")
    if proposal_source == PROPOSAL_SOURCE_SYNTHETIC and baseline_generation_fn is None:
        blockers.append("synthetic_shadow_provider reserved for tests")

    runtime: Any | None = None
    if baseline_generation_fn is None or draft_shadow_generation_fn is None:
        try:
            if runtime_loader is not None:
                runtime = runtime_loader(
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            else:
                from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
                from exactkv.runtime.model_runtime import ModelRuntime

                runtime = ModelRuntime(model_id, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"model load failed: {type(exc).__name__}: {exc}")

    spec_proposal = safety_spec_proposal or L3_PANEL_SAFETY_SPEC_PROPOSAL
    safety_spec_validation = validate_integration_proposal(spec_proposal)
    if not safety_spec_validation["pass"]:
        blockers.append("safety_spec_validation_failed")

    cells: list[dict[str, Any]] = []
    baseline_ok = 0
    draft_ok = 0
    tok_match = 0
    txt_match = 0
    total_proposals = 0
    successful_proposals = 0
    blocked_proposals = 0
    failed_cells = 0
    sg_ok = 0

    b_fn = baseline_generation_fn
    d_fn = draft_shadow_generation_fn

    if runtime is None and (b_fn is None or d_fn is None):
        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                for max_new in mnt_values:
                    cells.append({
                        "prompt_id": prompt_id,
                        "prompt_preview": prompt_text[:80],
                        "compressor": compressor,
                        "max_new_tokens": max_new,
                        "baseline_generation_completed": False,
                        "draft_shadow_generation_completed": False,
                        "baseline_vs_draft_shadow_token_match": False,
                        "baseline_vs_draft_shadow_text_match": False,
                        "proposal_source": proposal_source,
                        "total_proposals": 0,
                        "successful_proposals": 0,
                        "blocked_proposals": 0,
                        "proposal_block_reasons": {},
                        "proposals": [],
                        "proposal_match_summary": summarize_proposal_match(
                            [], committed_token_ids=[],
                        ),
                        "safety_gates": {},
                        "blockers": list(blockers),
                    })
    else:
        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                for max_new in mnt_values:
                    cell, failed = _build_panel_cell(
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        compressor=compressor,
                        max_new_tokens=max_new,
                        proposal_source=proposal_source,
                        runtime=runtime,
                        draft_len=draft_len,
                        baseline_generation_fn=b_fn,
                        draft_shadow_generation_fn=d_fn,
                        allow_provider_blocked=allow_provider_blocked,
                    )
                    cells.append(cell)
                    if cell["baseline_generation_completed"]:
                        baseline_ok += 1
                    if cell["draft_shadow_generation_completed"]:
                        draft_ok += 1
                    if cell["baseline_vs_draft_shadow_token_match"]:
                        tok_match += 1
                    if cell["baseline_vs_draft_shadow_text_match"]:
                        txt_match += 1
                    total_proposals += cell.get("total_proposals", 0)
                    successful_proposals += cell.get("successful_proposals", 0)
                    blocked_proposals += cell.get("blocked_proposals", 0)
                    if _cell_safety_gates_ok(cell.get("safety_gates") or {}):
                        sg_ok += 1
                    if failed:
                        failed_cells += 1

    block_reason_summary = aggregate_proposal_block_reasons(cells)
    proposal_coverage_rate = (
        successful_proposals / total_proposals if total_proposals else 0.0
    )

    proposal_match_agg = {
        "proposal_count": total_proposals,
        "proposals_matching_committed_token": sum(
            (c.get("proposal_match_summary") or {}).get(
                "proposals_matching_committed_token", 0,
            )
            for c in cells
        ),
        "proposals_not_matching_committed_token": sum(
            (c.get("proposal_match_summary") or {}).get(
                "proposals_not_matching_committed_token", 0,
            )
            for c in cells
        ),
        "blocked_proposals": blocked_proposals,
        "interpretation_note": PROPOSAL_INTERPRETATION_NOTE,
    }
    if cells:
        rates = [
            (c.get("proposal_match_summary") or {}).get("proposal_match_rate", 0.0)
            for c in cells
            if (c.get("proposal_match_summary") or {}).get("proposal_count", 0) > 0
        ]
        proposal_match_agg["mean_proposal_match_rate"] = (
            sum(rates) / len(rates) if rates else 0.0
        )

    total = len(cells)
    if not safety_spec_validation["pass"]:
        status = "failed"
    elif failed_cells > 0:
        status = "failed"
    elif baseline_ok == total and draft_ok == total and tok_match == total and total > 0:
        status = "panel_complete"
    elif baseline_ok > 0:
        status = "panel_partial"
    else:
        status = "blocked"

    safety_top = default_no_commit_safety_result()

    return {
        "experiment_id": EXPERIMENT_092_ID,
        "status": status,
        "phase": PHASE_18C,
        "safety_level": SAFETY_LEVEL,
        "safety_spec_validation": safety_spec_validation,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "proposal_source": proposal_source,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens_values": mnt_values,
        "total_cells": total,
        "baseline_generation_successful_cells": baseline_ok,
        "draft_shadow_generation_successful_cells": draft_ok,
        "baseline_vs_draft_shadow_token_match_cells": tok_match,
        "baseline_vs_draft_shadow_text_match_cells": txt_match,
        "total_proposals": total_proposals,
        "successful_proposals": successful_proposals,
        "blocked_proposals": blocked_proposals,
        "proposal_coverage_rate": proposal_coverage_rate,
        "proposal_block_reason_summary": block_reason_summary,
        "proposal_match_summary": proposal_match_agg,
        "exactkv_failure_summary": {
            "baseline_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_baseline") or 0) > 0
            ),
            "draft_shadow_failures": sum(
                1 for c in cells if (c.get("exactkv_failures_draft_shadow") or 0) > 0
            ),
        },
        "safety_gate_summary": {
            "cells_all_gates_ok": sg_ok,
            "cells_with_gate_failure": total - sg_ok,
        },
        "proposal_used_for_token_commit": safety_top.proposal_used_for_token_commit,
        "proposal_exposed_to_generator": safety_top.proposal_exposed_to_generator,
        "generated_output_modified_by_proposal": (
            safety_top.generated_output_modified_by_proposal
        ),
        "default_runtime_changed": safety_top.default_runtime_changed,
        "cells": cells,
        "topk_interpretation_note": TOPK_INTERPRETATION_NOTE,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE_18C,
        "claim_note": (
            "L3 guarded draft-shadow panel validation. Proposals are diagnostic only."
        ),
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "blockers": blockers,
        "limitations": [
            "L3 panel validation only; not L4 verifier-mediated compressed draft.",
            "Blocked proposals reported, not fabricated.",
            "Proposal coverage and match rates supplementary; not exactness.",
            "ExactKVGenerator and default runtime unchanged.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
    }


def validate_exp092_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "safety_level",
        "safety_spec_validation",
        "model_id",
        "device",
        "dtype",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens_values",
        "proposal_source",
        "total_cells",
        "baseline_generation_successful_cells",
        "draft_shadow_generation_successful_cells",
        "baseline_vs_draft_shadow_token_match_cells",
        "baseline_vs_draft_shadow_text_match_cells",
        "total_proposals",
        "successful_proposals",
        "blocked_proposals",
        "proposal_coverage_rate",
        "proposal_block_reason_summary",
        "proposal_match_summary",
        "exactkv_failure_summary",
        "safety_gate_summary",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "generated_output_modified_by_proposal",
        "default_runtime_changed",
        "blockers",
        "limitations",
        "no_performance_claims_note",
        "cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_092_ID:
        errors.append("experiment_id mismatch")

    if report.get("safety_level") != SAFETY_LEVEL:
        errors.append("safety_level mismatch")

    if report.get("proposal_used_for_token_commit") is not False:
        errors.append("proposal_used_for_token_commit must be false")

    if report.get("proposal_exposed_to_generator") is not False:
        errors.append("proposal_exposed_to_generator must be false")

    spec_val = report.get("safety_spec_validation") or {}
    if spec_val.get("pass") is not True:
        errors.append("safety_spec_validation must pass")

    for idx, cell in enumerate(report.get("cells") or []):
        for ck in (
            "prompt_id",
            "compressor",
            "max_new_tokens",
            "proposal_source",
            "total_proposals",
            "successful_proposals",
            "blocked_proposals",
            "proposal_block_reasons",
            "proposal_match_summary",
            "safety_gates",
            "proposals",
        ):
            if ck not in cell:
                errors.append(f"cells[{idx}] missing {ck}")

    return errors


def load_exp092_previous_coverage(
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Load Exp092 proposal coverage for comparison; unknown if report missing."""
    path = report_path or DEFAULT_EXP092_REPORT
    unknown = {
        "previous_total_proposals": None,
        "previous_successful_proposals": None,
        "previous_blocked_proposals": None,
        "previous_coverage_rate": None,
        "previous_report_available": False,
    }
    if not path.is_file():
        return unknown
    try:
        import json

        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return unknown
    if data.get("experiment_id") != EXPERIMENT_092_ID:
        return unknown
    return {
        "previous_total_proposals": data.get("total_proposals"),
        "previous_successful_proposals": data.get("successful_proposals"),
        "previous_blocked_proposals": data.get("blocked_proposals"),
        "previous_coverage_rate": data.get("proposal_coverage_rate"),
        "previous_report_available": True,
    }


def aggregate_extraction_results(
    cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate hardened top-1 extraction outcomes from panel cells."""
    extractions: list[dict[str, Any]] = []
    source_summary: dict[str, int] = {}
    block_summary: dict[str, int] = {}
    successful = 0
    blocked = 0
    unsafe_rejected = 0

    for cell in cells:
        for prop in cell.get("proposals") or []:
            status = prop.get("extraction_status")
            extraction = {
                "extraction_status": status,
                "proposed_token_id": prop.get("proposed_token_id"),
                "proposed_token_text": prop.get("proposed_token_text"),
                "extraction_source_field": prop.get("extraction_source_field"),
                "extraction_confidence": prop.get("extraction_confidence"),
                "block_reason": prop.get("block_reason"),
                "is_shadow_derived": prop.get("is_shadow_derived"),
                "uses_committed_token": prop.get("uses_committed_token"),
                "uses_baseline_token": prop.get("uses_baseline_token"),
                "exception": prop.get("exception"),
            }
            extractions.append(extraction)
            if status == "success":
                successful += 1
                src = extraction.get("extraction_source_field")
                if src:
                    source_summary[src] = source_summary.get(src, 0) + 1
            elif status == "unsafe_rejected":
                unsafe_rejected += 1
                reason = extraction.get("block_reason") or "unsafe_rejected"
                block_summary[reason] = block_summary.get(reason, 0) + 1
            else:
                blocked += 1
                reason = extraction.get("block_reason") or prop.get("exception") or "blocked"
                block_summary[str(reason)] = block_summary.get(str(reason), 0) + 1

    total = len(extractions)
    coverage_rate = successful / total if total else 0.0
    return {
        "extractions": extractions,
        "total_extractions": total,
        "successful_extractions": successful,
        "blocked_extractions": blocked,
        "unsafe_extractions_rejected": unsafe_rejected,
        "extraction_source_summary": source_summary,
        "extraction_block_reason_summary": block_summary,
        "current_total_proposals": total,
        "current_successful_proposals": successful,
        "current_blocked_proposals": blocked + unsafe_rejected,
        "current_coverage_rate": coverage_rate,
    }


def compute_coverage_delta(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare current extraction coverage to Exp092 baseline when available."""
    prev_rate = previous.get("previous_coverage_rate")
    cur_rate = current.get("current_coverage_rate")
    delta: float | None = None
    if prev_rate is not None and cur_rate is not None:
        delta = float(cur_rate) - float(prev_rate)
    return {
        **previous,
        **current,
        "coverage_delta": delta,
    }


def run_exp093_shadow_top1_extraction_hardening(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = DEFAULT_PANEL_PROMPTS,
    max_new_tokens_values: Sequence[int] = DEFAULT_MAX_NEW_TOKENS_VALUES,
    compressors_requested: Sequence[str] = DEFAULT_PANEL_COMPRESSORS,
    proposal_source: str = PROPOSAL_SOURCE_DECODE_TOP1,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_provider_blocked: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    draft_shadow_generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
    exp092_report_path: Path | None = None,
) -> dict[str, Any]:
    """Run Experiment 093 L3 shadow top-1 extraction hardening panel."""
    panel = run_exp092_guarded_draft_shadow_panel_validation(
        model_id=model_id,
        device=device,
        dtype=dtype,
        prompts=prompts,
        max_prompts=max_prompts,
        max_new_tokens_values=max_new_tokens_values,
        compressors_requested=compressors_requested,
        proposal_source=proposal_source,
        draft_len=draft_len,
        local_files_only=local_files_only,
        allow_provider_blocked=allow_provider_blocked,
        baseline_generation_fn=baseline_generation_fn,
        draft_shadow_generation_fn=draft_shadow_generation_fn,
        runtime_loader=runtime_loader,
    )

    extraction_agg = aggregate_extraction_results(panel.get("cells") or [])
    previous = load_exp092_previous_coverage(exp092_report_path)
    coverage = compute_coverage_delta(previous, extraction_agg)

    safety_top = default_no_commit_safety_result()
    status = panel.get("status", "blocked")
    if status == "panel_complete":
        status = "hardening_complete"
    elif status == "panel_partial":
        status = "hardening_partial"

    return {
        "experiment_id": EXPERIMENT_093_ID,
        "status": status,
        "phase": PHASE_18D,
        "safety_level": SAFETY_LEVEL,
        "safety_spec_validation": panel.get("safety_spec_validation"),
        "model_id": panel.get("model_id"),
        "device": panel.get("device"),
        "dtype": panel.get("dtype"),
        "compressors_requested": panel.get("compressors_requested"),
        "compressors_run": panel.get("compressors_run"),
        "max_new_tokens_values": panel.get("max_new_tokens_values"),
        "proposal_source": panel.get("proposal_source"),
        "extraction_sources_allowed": list(EXTRACTION_SOURCES_ALLOWED),
        "extraction_sources_forbidden": list(EXTRACTION_SOURCES_FORBIDDEN),
        "previous_coverage": {
            k: previous.get(k)
            for k in (
                "previous_total_proposals",
                "previous_successful_proposals",
                "previous_blocked_proposals",
                "previous_coverage_rate",
                "previous_report_available",
            )
        },
        "current_coverage": {
            "current_total_proposals": coverage.get("current_total_proposals"),
            "current_successful_proposals": coverage.get("current_successful_proposals"),
            "current_blocked_proposals": coverage.get("current_blocked_proposals"),
            "current_coverage_rate": coverage.get("current_coverage_rate"),
        },
        "coverage_delta": coverage.get("coverage_delta"),
        "total_extractions": extraction_agg["total_extractions"],
        "successful_extractions": extraction_agg["successful_extractions"],
        "blocked_extractions": extraction_agg["blocked_extractions"],
        "unsafe_extractions_rejected": extraction_agg["unsafe_extractions_rejected"],
        "extraction_source_summary": extraction_agg["extraction_source_summary"],
        "extraction_block_reason_summary": extraction_agg["extraction_block_reason_summary"],
        "total_cells": panel.get("total_cells"),
        "baseline_generation_successful_cells": panel.get(
            "baseline_generation_successful_cells",
        ),
        "draft_shadow_generation_successful_cells": panel.get(
            "draft_shadow_generation_successful_cells",
        ),
        "baseline_vs_draft_shadow_token_match_cells": panel.get(
            "baseline_vs_draft_shadow_token_match_cells",
        ),
        "baseline_vs_draft_shadow_text_match_cells": panel.get(
            "baseline_vs_draft_shadow_text_match_cells",
        ),
        "total_proposals": panel.get("total_proposals"),
        "successful_proposals": panel.get("successful_proposals"),
        "blocked_proposals": panel.get("blocked_proposals"),
        "proposal_coverage_rate": panel.get("proposal_coverage_rate"),
        "proposal_block_reason_summary": panel.get("proposal_block_reason_summary"),
        "proposal_match_summary": panel.get("proposal_match_summary"),
        "exactkv_failure_summary": panel.get("exactkv_failure_summary"),
        "safety_gate_summary": panel.get("safety_gate_summary"),
        "proposal_used_for_token_commit": safety_top.proposal_used_for_token_commit,
        "proposal_exposed_to_generator": safety_top.proposal_exposed_to_generator,
        "generated_output_modified_by_proposal": (
            safety_top.generated_output_modified_by_proposal
        ),
        "default_runtime_changed": safety_top.default_runtime_changed,
        "cells": panel.get("cells"),
        "topk_interpretation_note": panel.get("topk_interpretation_note"),
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE_18D,
        "claim_note": (
            "L3 shadow top-1 extraction hardening. Proposals are diagnostic only."
        ),
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "blockers": panel.get("blockers"),
        "limitations": [
            "L3 extraction hardening only; not L4 verifier-mediated compressed draft.",
            "Extracted proposals cannot affect token commits or generator decisions.",
            "Proposal coverage is not exactness; match rate supplementary only.",
            "ExactKVGenerator and default runtime unchanged.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
    }


def validate_exp093_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "safety_level",
        "safety_spec_validation",
        "model_id",
        "device",
        "dtype",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens_values",
        "proposal_source",
        "extraction_sources_allowed",
        "extraction_sources_forbidden",
        "previous_coverage",
        "current_coverage",
        "coverage_delta",
        "total_extractions",
        "successful_extractions",
        "blocked_extractions",
        "unsafe_extractions_rejected",
        "extraction_source_summary",
        "extraction_block_reason_summary",
        "total_cells",
        "baseline_generation_successful_cells",
        "draft_shadow_generation_successful_cells",
        "baseline_vs_draft_shadow_token_match_cells",
        "baseline_vs_draft_shadow_text_match_cells",
        "exactkv_failure_summary",
        "safety_gate_summary",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "generated_output_modified_by_proposal",
        "default_runtime_changed",
        "blockers",
        "limitations",
        "no_performance_claims_note",
        "cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_093_ID:
        errors.append("experiment_id mismatch")

    if report.get("safety_level") != SAFETY_LEVEL:
        errors.append("safety_level mismatch")

    if report.get("proposal_used_for_token_commit") is not False:
        errors.append("proposal_used_for_token_commit must be false")

    if report.get("proposal_exposed_to_generator") is not False:
        errors.append("proposal_exposed_to_generator must be false")

    spec_val = report.get("safety_spec_validation") or {}
    if spec_val.get("pass") is not True:
        errors.append("safety_spec_validation must pass")

    for cell in report.get("cells") or []:
        for prop in cell.get("proposals") or []:
            for pk in (
                "extraction_status",
                "extraction_source_field",
                "extraction_confidence",
                "is_shadow_derived",
                "uses_committed_token",
                "uses_baseline_token",
            ):
                if pk not in prop:
                    errors.append(f"proposal missing extraction field: {pk}")

    return errors
