"""External L1 generation-shadow observer (Phase 16K).

Runs ExactKV generation unchanged, then post-hoc offline shadow replay/logit
diagnostics. **Does not** modify ExactKVGenerator or affect token commits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from exactkv.attention.generation_shadow_review import (
    PROPOSED_SHADOW_CLI_FLAG,
    SHADOW_FORBIDDEN_CLAIMS,
)
from exactkv.attention.streaming_quant_attention import FORBIDDEN_ATTENTION_CLAIMS

EXPERIMENT_076_ID = "exp076_generation_shadow_observer_smoke"
DEFAULT_EXP076_REPORT = Path("reports/experiment_076_generation_shadow_observer_smoke.json")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_SHADOW_MODE = "prompt_prefix_only"
DEFAULT_CHUNK_SIZE = 16

EXP076_CLAIM_NOTE = (
    "External L1 generation-shadow observer smoke (Phase 16K). Runs ExactKV "
    "generation unchanged, then offline shadow replay diagnostics. Not generation "
    "integration, vLLM, CUDA/Triton kernels, or default runtime change. Shadow "
    "logits/top-k are diagnostic only. No speed, throughput, latency, active GPU "
    "memory, or serving claim."
)

ShadowMode = str  # prompt_prefix_only | prompt_plus_generated_tokens | blocked_missing_tokens


class GenerationShadowStatus(str, Enum):
    SHADOW_COMPLETE = "shadow_complete"
    SHADOW_BLOCKED = "shadow_blocked"
    GENERATION_BLOCKED = "generation_blocked"
    SKIPPED = "skipped"


@dataclass
class GenerationShadowObserverConfig:
    """Configuration for external generation-shadow observer."""

    shadow_observer_enabled: bool = False
    shadow_mode: str = DEFAULT_SHADOW_MODE
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    dtype: str = "float32"
    max_new_tokens: int = 4
    chunk_size: int = DEFAULT_CHUNK_SIZE
    draft_len: int = 4
    compressor_name: str = "noop"
    accumulator_mode: str = "float32"
    allow_shadow_fail: bool = True
    allow_parity_fail: bool = True
    skip_generation: bool = False
    local_files_only: bool = False
    allow_generated_text_retokenize: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationOutput:
    """Captured generation result (unchanged by shadow)."""

    generation_completed: bool
    generation_output_text: str
    generation_output_token_ids: list[int] | None
    prompt_ids: torch.Tensor | None = None
    full_sequence_ids: torch.Tensor | None = None
    blockers: list[str] = field(default_factory=list)


@dataclass
class GenerationShadowMetrics:
    """Offline shadow drift metrics for one prompt."""

    full_model_parity_status: str | None = None
    streaming_vs_materialized_hidden: dict[str, Any] | None = None
    streaming_vs_materialized_logit: dict[str, Any] | None = None
    full_vs_streaming_hidden: dict[str, Any] | None = None
    full_vs_streaming_logit: dict[str, Any] | None = None
    topk_agreement_metrics: dict[str, Any] | None = None
    depth_aware_tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptShadowResult:
    """Per-prompt generation + shadow observer result."""

    prompt_id: str
    prompt_preview: str
    generation_completed: bool
    generation_output_preview: str
    generation_output_token_ids_available: bool
    generation_output_token_count: int
    shadow_observer_enabled: bool
    shadow_ran_after_generation: bool
    shadow_sequence_mode: str
    shadow_sequence_length: int
    shadow_status: str
    tolerance_policy_status: str | None
    generation_shadow_metrics: GenerationShadowMetrics | None
    streaming_vs_materialized_metrics: dict[str, Any] | None
    full_vs_streaming_metrics: dict[str, Any] | None
    topk_agreement_metrics: dict[str, Any] | None
    interpretation_note: str
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.generation_shadow_metrics is not None:
            d["generation_shadow_metrics"] = self.generation_shadow_metrics.to_dict()
        return d


@dataclass
class GenerationShadowObserverResult:
    """Aggregate result from observer run."""

    config: GenerationShadowObserverConfig
    prompt_results: list[PromptShadowResult]
    generation_modified_by_shadow: bool = False
    shadow_used_for_token_commit: bool = False
    default_runtime_changed: bool = False
    blockers: list[str] = field(default_factory=list)

    @property
    def generation_successful_prompts(self) -> int:
        return sum(1 for p in self.prompt_results if p.generation_completed)

    @property
    def shadow_successful_prompts(self) -> int:
        return sum(
            1 for p in self.prompt_results
            if p.shadow_ran_after_generation and p.shadow_status == GenerationShadowStatus.SHADOW_COMPLETE.value
        )

    @property
    def blocked_prompts(self) -> int:
        return sum(
            1 for p in self.prompt_results
            if p.shadow_status in (
                GenerationShadowStatus.SHADOW_BLOCKED.value,
                GenerationShadowStatus.GENERATION_BLOCKED.value,
            )
        )


def _preview(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def reconstruct_shadow_input_ids(
    gen_out: GenerationOutput,
    *,
    shadow_mode: str,
    tokenizer_encode: Callable[[str], torch.Tensor] | None = None,
    prompt_text: str = "",
    generated_text_retokenize_ok: bool = False,
) -> tuple[torch.Tensor | None, str, list[str]]:
    """Build fixed sequence for post-hoc shadow replay."""
    blockers: list[str] = []
    if shadow_mode == "blocked_missing_tokens":
        return None, "blocked_missing_tokens", ["shadow mode blocked_missing_tokens"]

    prompt_ids = gen_out.prompt_ids
    if prompt_ids is None and tokenizer_encode is not None and prompt_text:
        prompt_ids = tokenizer_encode(prompt_text)

    if prompt_ids is None:
        return None, "blocked_missing_tokens", ["prompt token IDs unavailable"]

    if shadow_mode == "prompt_prefix_only":
        return prompt_ids, "prompt_prefix_only", []

    if shadow_mode == "prompt_plus_generated_tokens":
        gen_token_ids = gen_out.generation_output_token_ids
        if not gen_token_ids and generated_text_retokenize_ok and tokenizer_encode is not None:
            # Conservative fallback: only when explicitly enabled.
            # Note: this is not guaranteed to reproduce the model's actual token IDs.
            if gen_out.generation_output_text:
                try:
                    gen_ids = tokenizer_encode(gen_out.generation_output_text)
                    gen_token_ids = gen_ids.squeeze(0).tolist()
                    if isinstance(gen_token_ids, int):
                        gen_token_ids = [gen_token_ids]
                except Exception:  # noqa: BLE001
                    gen_token_ids = None
        if not gen_token_ids:
            return None, "blocked_missing_tokens", ["generated token IDs unavailable"]
        gen_tensor = torch.tensor([gen_token_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
        return torch.cat([prompt_ids, gen_tensor], dim=1), "prompt_plus_generated_tokens", []

    return None, "blocked_missing_tokens", [f"unknown shadow_mode: {shadow_mode}"]


def default_exactkv_generation(
    *,
    runtime: Any,
    prompt: str,
    max_new_tokens: int,
    draft_len: int = 4,
    compressor_name: str = "noop",
) -> GenerationOutput:
    """Run ExactKVGenerator.generate without modification."""
    from exactkv.compressors import get_compressor
    from exactkv.runtime.exactkv_generator import ExactKVGenerator

    try:
        compressor = get_compressor(compressor_name)
        generator = ExactKVGenerator(runtime, compressor, draft_len=draft_len)
        result = generator.generate(prompt, max_new_tokens)
        token_ids = result.output_ids.squeeze().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return GenerationOutput(
            generation_completed=True,
            generation_output_text=result.output_text,
            generation_output_token_ids=list(token_ids),
            prompt_ids=result.prompt_ids,
            full_sequence_ids=result.full_sequence_ids,
        )
    except Exception as exc:  # noqa: BLE001
        return GenerationOutput(
            generation_completed=False,
            generation_output_text="",
            generation_output_token_ids=None,
            blockers=[f"generation failed: {type(exc).__name__}: {exc}"],
        )


def default_offline_shadow_replay(
    *,
    model: Any,
    input_ids: torch.Tensor,
    prompt_id: str,
    chunk_size: int,
    accumulator_mode: str,
    allow_parity_fail: bool,
) -> dict[str, Any]:
    """Run Phase 16F-style offline shadow replay on a fixed prefix."""
    from exactkv.attention.hf_full_replay_probe import run_exp071_logit_cell

    with torch.no_grad():
        hf_out = model(input_ids, output_hidden_states=True, use_cache=False)
    hf_logits = hf_out.logits[:, -1, :]
    hf_hidden = hf_out.hidden_states[-1] if hf_out.hidden_states else None
    if hf_hidden is None:
        return {"blockers": ["HF forward missing hidden_states"], "shadow_status": "shadow_blocked"}

    seq_len = int(input_ids.shape[-1])
    return run_exp071_logit_cell(
        model=model,
        input_ids=input_ids,
        hf_logits=hf_logits,
        hf_hidden=hf_hidden,
        prompt_id=prompt_id,
        target_token_length=seq_len,
        actual_token_length=seq_len,
        chunk_size=chunk_size,
        accumulator_mode=accumulator_mode,
        allow_parity_fail=allow_parity_fail,
    )


def apply_tolerance_policy_to_shadow_cell(
    shadow_cell: dict[str, Any],
    *,
    num_layers: int,
    policy: Any | None = None,
    teacher_forced_metrics: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Apply Phase 16I policy to shadow logit metrics."""
    from exactkv.attention.tolerance_policy import (
        AttentionTolerancePolicy,
        MetricType,
        TopKAgreementSummary,
        evaluate_offline_attention_cell,
    )

    policy = policy or AttentionTolerancePolicy()
    logit_m = shadow_cell.get("streaming_vs_materialized_logit_metrics")
    if not logit_m:
        return "blocked", "Shadow logit metrics unavailable for tolerance policy."

    topk = TopKAgreementSummary(
        top1_agreement=logit_m.get("top1_agreement"),
        top5_overlap=logit_m.get("top5_overlap"),
        top10_overlap=logit_m.get("top10_overlap"),
    )
    parity_ok = shadow_cell.get("full_model_parity_status") == "passed"
    hidden_m = shadow_cell.get("streaming_vs_materialized_hidden_metrics") or {}
    decision = evaluate_offline_attention_cell(
        policy=policy,
        metric_type=MetricType.LOGITS,
        prefix_layers=num_layers,
        max_abs_error=float(logit_m.get("max_abs_error", 0.0)),
        blockers=shadow_cell.get("blockers"),
        parity_passed=parity_ok,
        teacher_forced_max_attn_error=(teacher_forced_metrics or {}).get("max_attn"),
        teacher_forced_max_post_mlp_error=(teacher_forced_metrics or {}).get("max_post_mlp"),
        free_running_max_post_mlp_error=float(hidden_m.get("max_abs_error", 0.0)),
        root_cause_classification=shadow_cell.get("root_cause_classification"),
        topk=topk,
    )
    return decision.overall_offline_status.value, decision.interpretation_note


def _metrics_from_shadow_cell(shadow_cell: dict[str, Any]) -> GenerationShadowMetrics:
    sm_logit = shadow_cell.get("streaming_vs_materialized_logit_metrics") or {}
    return GenerationShadowMetrics(
        full_model_parity_status=shadow_cell.get("full_model_parity_status"),
        streaming_vs_materialized_hidden=shadow_cell.get("streaming_vs_materialized_hidden_metrics"),
        streaming_vs_materialized_logit=shadow_cell.get("streaming_vs_materialized_logit_metrics"),
        full_vs_streaming_hidden=shadow_cell.get("full_vs_streaming_hidden_metrics"),
        full_vs_streaming_logit=shadow_cell.get("full_vs_streaming_logit_metrics"),
        topk_agreement_metrics={
            "top1_agreement": sm_logit.get("top1_agreement"),
            "top5_overlap": sm_logit.get("top5_overlap"),
            "top10_overlap": sm_logit.get("top10_overlap"),
            "top1_changed_full_vs_streaming": shadow_cell.get("top1_changed_full_vs_streaming"),
        },
        depth_aware_tolerance=shadow_cell.get("depth_aware_tolerance"),
    )


def observe_prompt(
    *,
    prompt_id: str,
    prompt_text: str,
    config: GenerationShadowObserverConfig,
    generation_fn: Callable[..., GenerationOutput] | None = None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None = None,
    tolerance_policy_fn: Callable[..., tuple[str, str]] | None = None,
    runtime: Any | None = None,
    hf_model: Any | None = None,
    tokenizer_encode: Callable[[str], torch.Tensor] | None = None,
) -> PromptShadowResult:
    """Run generation then post-hoc shadow for one prompt."""
    if not config.shadow_observer_enabled:
        return PromptShadowResult(
            prompt_id=prompt_id,
            prompt_preview=_preview(prompt_text),
            generation_completed=False,
            generation_output_preview="",
            generation_output_token_ids_available=False,
            generation_output_token_count=0,
            shadow_observer_enabled=False,
            shadow_ran_after_generation=False,
            shadow_sequence_mode="skipped",
            shadow_sequence_length=0,
            shadow_status=GenerationShadowStatus.SKIPPED.value,
            tolerance_policy_status=None,
            generation_shadow_metrics=None,
            streaming_vs_materialized_metrics=None,
            full_vs_streaming_metrics=None,
            topk_agreement_metrics=None,
            interpretation_note="Shadow observer not enabled.",
            blockers=["shadow_observer_enabled is false"],
        )

    gen_out: GenerationOutput
    if config.skip_generation:
        gen_out = GenerationOutput(
            generation_completed=False,
            generation_output_text="",
            generation_output_token_ids=None,
            blockers=["skip_generation flag set"],
        )
    elif generation_fn is not None:
        gen_out = generation_fn(prompt=prompt_text, max_new_tokens=config.max_new_tokens)
    elif runtime is not None:
        gen_out = default_exactkv_generation(
            runtime=runtime,
            prompt=prompt_text,
            max_new_tokens=config.max_new_tokens,
            draft_len=config.draft_len,
            compressor_name=config.compressor_name,
        )
    else:
        return PromptShadowResult(
            prompt_id=prompt_id,
            prompt_preview=_preview(prompt_text),
            generation_completed=False,
            generation_output_preview="",
            generation_output_token_ids_available=False,
            generation_output_token_count=0,
            shadow_observer_enabled=True,
            shadow_ran_after_generation=False,
            shadow_sequence_mode="blocked",
            shadow_sequence_length=0,
            shadow_status=GenerationShadowStatus.GENERATION_BLOCKED.value,
            tolerance_policy_status=None,
            generation_shadow_metrics=None,
            streaming_vs_materialized_metrics=None,
            full_vs_streaming_metrics=None,
            topk_agreement_metrics=None,
            interpretation_note="Generation API unavailable; no runtime or generator injected.",
            blockers=["generation API missing: provide runtime or generation_fn"],
        )

    if not gen_out.generation_completed:
        return PromptShadowResult(
            prompt_id=prompt_id,
            prompt_preview=_preview(prompt_text),
            generation_completed=False,
            generation_output_preview="",
            generation_output_token_ids_available=False,
            generation_output_token_count=0,
            shadow_observer_enabled=True,
            shadow_ran_after_generation=False,
            shadow_sequence_mode="blocked",
            shadow_sequence_length=0,
            shadow_status=GenerationShadowStatus.GENERATION_BLOCKED.value,
            tolerance_policy_status=None,
            generation_shadow_metrics=None,
            streaming_vs_materialized_metrics=None,
            full_vs_streaming_metrics=None,
            topk_agreement_metrics=None,
            interpretation_note="Generation failed before shadow could run.",
            blockers=list(gen_out.blockers),
        )

    input_ids, seq_mode, recon_blockers = reconstruct_shadow_input_ids(
        gen_out,
        shadow_mode=config.shadow_mode,
        tokenizer_encode=tokenizer_encode,
        prompt_text=prompt_text,
        generated_text_retokenize_ok=config.allow_generated_text_retokenize,
    )
    if input_ids is None:
        return PromptShadowResult(
            prompt_id=prompt_id,
            prompt_preview=_preview(prompt_text),
            generation_completed=True,
            generation_output_preview=_preview(gen_out.generation_output_text),
            generation_output_token_ids_available=bool(gen_out.generation_output_token_ids),
            generation_output_token_count=len(gen_out.generation_output_token_ids or []),
            shadow_observer_enabled=True,
            shadow_ran_after_generation=False,
            shadow_sequence_mode=seq_mode,
            shadow_sequence_length=0,
            shadow_status=GenerationShadowStatus.SHADOW_BLOCKED.value,
            tolerance_policy_status=None,
            generation_shadow_metrics=None,
            streaming_vs_materialized_metrics=None,
            full_vs_streaming_metrics=None,
            topk_agreement_metrics=None,
            interpretation_note="Could not reconstruct shadow sequence.",
            blockers=recon_blockers,
        )

    model = hf_model
    if model is None and runtime is not None:
        model = getattr(runtime, "model", None)

    if model is None:
        return PromptShadowResult(
            prompt_id=prompt_id,
            prompt_preview=_preview(prompt_text),
            generation_completed=True,
            generation_output_preview=_preview(gen_out.generation_output_text),
            generation_output_token_ids_available=bool(gen_out.generation_output_token_ids),
            generation_output_token_count=len(gen_out.generation_output_token_ids or []),
            shadow_observer_enabled=True,
            shadow_ran_after_generation=False,
            shadow_sequence_mode=seq_mode,
            shadow_sequence_length=int(input_ids.shape[-1]),
            shadow_status=GenerationShadowStatus.SHADOW_BLOCKED.value,
            tolerance_policy_status=None,
            generation_shadow_metrics=None,
            streaming_vs_materialized_metrics=None,
            full_vs_streaming_metrics=None,
            topk_agreement_metrics=None,
            interpretation_note="HF model unavailable for offline shadow replay.",
            blockers=["hf model missing for shadow replay"],
        )

    replay = shadow_replay_fn or default_offline_shadow_replay
    try:
        shadow_cell = replay(
            model=model,
            input_ids=input_ids,
            prompt_id=prompt_id,
            chunk_size=config.chunk_size,
            accumulator_mode=config.accumulator_mode,
            allow_parity_fail=config.allow_parity_fail,
        )
    except Exception as exc:  # noqa: BLE001
        shadow_cell = {"blockers": [f"shadow replay failed: {type(exc).__name__}: {exc}"]}

    if shadow_cell.get("blockers") and not config.allow_shadow_fail:
        shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
    elif shadow_cell.get("blockers"):
        shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
    else:
        shadow_status = GenerationShadowStatus.SHADOW_COMPLETE.value

    num_layers = int(shadow_cell.get("num_layers_replayed") or 0)
    if tolerance_policy_fn is not None:
        tol_status, interp = tolerance_policy_fn(shadow_cell=shadow_cell, num_layers=num_layers)
    else:
        tol_status, interp = apply_tolerance_policy_to_shadow_cell(
            shadow_cell, num_layers=num_layers or 24,
        )

    metrics = _metrics_from_shadow_cell(shadow_cell) if shadow_cell else None
    sm = shadow_cell.get("streaming_vs_materialized_logit_metrics")
    fs = shadow_cell.get("full_vs_streaming_logit_metrics")

    return PromptShadowResult(
        prompt_id=prompt_id,
        prompt_preview=_preview(prompt_text),
        generation_completed=True,
        generation_output_preview=_preview(gen_out.generation_output_text),
        generation_output_token_ids_available=bool(gen_out.generation_output_token_ids),
        generation_output_token_count=len(gen_out.generation_output_token_ids or []),
        shadow_observer_enabled=True,
        shadow_ran_after_generation=True,
        shadow_sequence_mode=seq_mode,
        shadow_sequence_length=int(input_ids.shape[-1]),
        shadow_status=shadow_status,
        tolerance_policy_status=tol_status,
        generation_shadow_metrics=metrics,
        streaming_vs_materialized_metrics=shadow_cell.get("streaming_vs_materialized_logit_metrics"),
        full_vs_streaming_metrics=fs,
        topk_agreement_metrics=metrics.topk_agreement_metrics if metrics else None,
        interpretation_note=interp,
        blockers=list(shadow_cell.get("blockers") or []),
    )


def run_generation_shadow_observer(
    prompts: Sequence[tuple[str, str]],
    *,
    config: GenerationShadowObserverConfig,
    generation_fn: Callable[..., GenerationOutput] | None = None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None = None,
    tolerance_policy_fn: Callable[..., tuple[str, str]] | None = None,
    runtime_loader: Callable[..., tuple[Any, Any]] | None = None,
) -> GenerationShadowObserverResult:
    """Run external generation-shadow observer over a prompt panel."""
    runtime: Any | None = None
    tokenizer: Any | None = None
    blockers: list[str] = []

    if config.shadow_observer_enabled and not config.skip_generation and runtime_loader is None and generation_fn is None:
        try:
            from exactkv.runtime.model_runtime import ModelRuntime

            runtime = ModelRuntime(
                config.model_id,
                device=config.device,
                dtype=config.dtype,
            )
            tokenizer = runtime.tokenizer
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")
    elif runtime_loader is not None and config.shadow_observer_enabled and not config.skip_generation:
        try:
            runtime, tokenizer = runtime_loader(
                model_id=config.model_id,
                device=config.device,
                dtype=config.dtype,
                local_files_only=config.local_files_only,
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    def _encode(text: str) -> torch.Tensor:
        if tokenizer is not None:
            return tokenizer(text, return_tensors="pt")["input_ids"].to(
                getattr(runtime, "device", "cpu")
            )
        raise RuntimeError("tokenizer unavailable")

    prompt_results: list[PromptShadowResult] = []
    for prompt_id, prompt_text in prompts:
        if blockers and generation_fn is None:
            prompt_results.append(
                PromptShadowResult(
                    prompt_id=prompt_id,
                    prompt_preview=_preview(prompt_text),
                    generation_completed=False,
                    generation_output_preview="",
                    generation_output_token_count=0,
                    shadow_observer_enabled=config.shadow_observer_enabled,
                    shadow_ran_after_generation=False,
                    shadow_sequence_mode="blocked",
                    shadow_sequence_length=0,
                    shadow_status=GenerationShadowStatus.GENERATION_BLOCKED.value,
                    tolerance_policy_status=None,
                    generation_shadow_metrics=None,
                    streaming_vs_materialized_metrics=None,
                    full_vs_streaming_metrics=None,
                    topk_agreement_metrics=None,
                    interpretation_note="Runtime load blocked generation.",
                    blockers=list(blockers),
                )
            )
            continue

        prompt_results.append(
            observe_prompt(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                config=config,
                generation_fn=generation_fn,
                shadow_replay_fn=shadow_replay_fn,
                tolerance_policy_fn=tolerance_policy_fn,
                runtime=runtime,
                hf_model=getattr(runtime, "model", None) if runtime else None,
                tokenizer_encode=_encode if tokenizer is not None else None,
            )
        )

    return GenerationShadowObserverResult(
        config=config,
        prompt_results=prompt_results,
        generation_modified_by_shadow=False,
        shadow_used_for_token_commit=False,
        default_runtime_changed=False,
        blockers=blockers,
    )


def build_exp076_report(
    observer_result: GenerationShadowObserverResult,
) -> dict[str, Any]:
    """Build Experiment 076 JSON report from observer result."""
    cfg = observer_result.config
    prompts = observer_result.prompt_results

    top1_agree = sum(
        1 for p in prompts
        if p.topk_agreement_metrics and p.topk_agreement_metrics.get("top1_agreement")
    )
    tol_statuses: dict[str, int] = {}
    for p in prompts:
        if p.tolerance_policy_status:
            tol_statuses[p.tolerance_policy_status] = tol_statuses.get(p.tolerance_policy_status, 0) + 1

    if not cfg.shadow_observer_enabled:
        status = "skipped"
    elif observer_result.blockers and observer_result.generation_successful_prompts == 0:
        status = "blocked"
    elif observer_result.shadow_successful_prompts == len(prompts) and prompts:
        status = "diagnostic_complete"
    elif observer_result.shadow_successful_prompts > 0:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_076_ID,
        "status": status,
        "model_id": cfg.model_id,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "max_new_tokens": cfg.max_new_tokens,
        "shadow_mode": cfg.shadow_mode,
        "generation_shadow_observer_enabled": cfg.shadow_observer_enabled,
        "cli_flag": PROPOSED_SHADOW_CLI_FLAG,
        "total_prompts": len(prompts),
        "generation_successful_prompts": observer_result.generation_successful_prompts,
        "shadow_successful_prompts": observer_result.shadow_successful_prompts,
        "blocked_prompts": observer_result.blocked_prompts,
        "generation_modified_by_shadow": observer_result.generation_modified_by_shadow,
        "shadow_used_for_token_commit": observer_result.shadow_used_for_token_commit,
        "default_runtime_changed": observer_result.default_runtime_changed,
        "tolerance_policy_status_counts": tol_statuses,
        "topk_agreement_summary": {
            "top1_agreement_prompts": top1_agree,
            "prompt_count": len(prompts),
        },
        "prompt_results": [p.to_dict() for p in prompts],
        "blockers": observer_result.blockers,
        "limitations": [
            "External post-hoc observer; not generation integration.",
            "Shadow runs after generation; cannot affect token commits.",
            "Top-k agreement is supplementary; not exactness.",
            "Prefix shadow only; no per-round decode observer.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP076_CLAIM_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
    }


def validate_exp076_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "max_new_tokens",
        "shadow_mode",
        "generation_shadow_observer_enabled",
        "total_prompts",
        "generation_successful_prompts",
        "shadow_successful_prompts",
        "blocked_prompts",
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
        "prompt_results",
        "blockers",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_076_ID:
        errors.append("experiment_id mismatch")

    if report.get("shadow_used_for_token_commit") is not False:
        errors.append("shadow_used_for_token_commit must be false")
    if report.get("generation_modified_by_shadow") is not False:
        errors.append("generation_modified_by_shadow must be false")
    if report.get("default_runtime_changed") is not False:
        errors.append("default_runtime_changed must be false")

    for idx, pr in enumerate(report.get("prompt_results", [])):
        if not isinstance(pr, dict):
            errors.append(f"prompt_results[{idx}] not dict")
            continue
        for ck in (
            "prompt_id",
            "generation_completed",
            "shadow_observer_enabled",
            "shadow_ran_after_generation",
            "shadow_status",
            "interpretation_note",
            "blockers",
        ):
            if ck not in pr:
                errors.append(f"prompt_results[{idx}] missing {ck}")

    return errors
