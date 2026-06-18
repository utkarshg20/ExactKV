"""External L1 generation-shadow observer (Phase 16K).

Runs ExactKV generation unchanged, then post-hoc offline shadow replay/logit
diagnostics. **Does not** modify ExactKVGenerator or affect token commits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    exactkv_failures: int | None = None
    token_exact_match: bool | None = None
    compressor_name: str | None = None
    exactkv_traces: list[Any] | None = None
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
            exactkv_traces=list(result.traces) if getattr(result, "traces", None) else None,
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


# --- Phase 16M: expanded generation-shadow panel ---

EXPERIMENT_078_ID = "exp078_generation_shadow_expanded_panel"
DEFAULT_EXP078_REPORT = Path("reports/experiment_078_generation_shadow_expanded_panel.json")
DEFAULT_EXP078_COMPRESSORS: tuple[str, ...] = ("noop", "int8", "int4_sim", "k8_v4_sim")
DEFAULT_EXP078_MAX_NEW_TOKENS: tuple[int, ...] = (4, 8)
DEFAULT_EXP078_SHADOW_MODES: tuple[str, ...] = (
    "prompt_prefix_only",
    "prompt_plus_generated_tokens",
)

EXP078_CLAIM_NOTE = (
    "External expanded generation-shadow panel (Phase 16M). Runs ExactKV generation "
    "unchanged across prompts, max_new_tokens, and compressors when cleanly exposed, "
    "then post-hoc shadow diagnostics. Not generation integration, vLLM, CUDA/Triton "
    "kernels, or default runtime change. Compressor results reported only when APIs "
    "expose them cleanly. Shadow logits/top-k are diagnostic only."
)


def default_exp078_prompts() -> list[tuple[str, str]]:
    """Eight deterministic CPU-friendly prompts for expanded panel."""
    long_ctx = (
        "ExactKV offline generation-shadow panel long-context filler. "
        "Deterministic text for fixed-sequence post-hoc replay. " * 4
    )
    return [
        ("p0_capital_france", "The capital of France is"),
        ("p1_simple_math", "Two plus two equals"),
        ("p2_json_like", '{"name": "ExactKV", "mode": "shadow", "tokens": 4}'),
        ("p3_code_like", "def add(a, b):\n    return a + b\n\n# call:"),
        ("p4_arithmetic_text", "If you multiply 7 by 8 you get"),
        ("p5_short_story", "Write one sentence about a cat:"),
        ("p6_structured_list", "List three colors separated by commas:"),
        ("p7_long_context", long_ctx),
    ]


def resolve_panel_compressors(
    requested: Sequence[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Return compressors that can be instantiated via get_compressor."""
    try:
        # Pre-import breaks compressors ↔ cache circular import on fresh interpreters.
        from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
        from exactkv.compressors import get_compressor, list_compressors
    except ImportError:
        return [], [
            {"compressor": name, "reason": "blocked_compressor_api_missing"}
            for name in requested
        ]

    available = set(list_compressors())
    runnable: list[str] = []
    blocked: list[dict[str, str]] = []
    for name in requested:
        if name not in available:
            blocked.append({
                "compressor": name,
                "reason": "blocked_compressor_api_missing",
            })
            continue
        try:
            get_compressor(name)
            runnable.append(name)
        except Exception as exc:  # noqa: BLE001
            blocked.append({
                "compressor": name,
                "reason": f"compressor init failed: {type(exc).__name__}: {exc}",
            })
    return runnable, blocked


def run_exactkv_generation_with_baseline(
    *,
    runtime: Any,
    prompt: str,
    max_new_tokens: int,
    compressor_name: str,
    draft_len: int = 4,
    full_baseline_ids: list[int] | None = None,
) -> GenerationOutput:
    """Run ExactKVGenerator unchanged and compare to full greedy baseline."""
    from exactkv.metrics.exactness import token_exact_match
    from exactkv.runtime.generation import generate_full_greedy

    gen_out = default_exactkv_generation(
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        compressor_name=compressor_name,
    )
    gen_out.compressor_name = compressor_name
    if not gen_out.generation_completed:
        gen_out.exactkv_failures = None
        gen_out.token_exact_match = None
        return gen_out

    try:
        if full_baseline_ids is None:
            full_res = generate_full_greedy(runtime, prompt, max_new_tokens)
            full_baseline_ids = full_res.generated_ids.squeeze().tolist()
            if isinstance(full_baseline_ids, int):
                full_baseline_ids = [full_baseline_ids]
        ekv_ids = gen_out.generation_output_token_ids or []
        match = bool(
            token_exact_match(
                torch.tensor([full_baseline_ids], dtype=torch.long),
                torch.tensor([ekv_ids], dtype=torch.long),
            )
        )
        gen_out.token_exact_match = match
        gen_out.exactkv_failures = 0 if match else 1
    except Exception as exc:  # noqa: BLE001
        gen_out.exactkv_failures = None
        gen_out.token_exact_match = None
        gen_out.blockers.append(f"baseline compare failed: {type(exc).__name__}: {exc}")
    return gen_out


def run_shadow_cells_for_generation(
    *,
    gen_out: GenerationOutput,
    prompt_id: str,
    shadow_modes: Sequence[str],
    hf_model: Any | None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None,
    chunk_size: int,
    accumulator_mode: str,
    allow_parity_fail: bool,
    allow_shadow_fail: bool,
) -> list[dict[str, Any]]:
    """Run post-hoc shadow for each shadow mode on one generation output."""
    cells: list[dict[str, Any]] = []
    for mode in shadow_modes:
        input_ids, seq_mode, recon_blockers = reconstruct_shadow_input_ids(
            gen_out,
            shadow_mode=mode,
            generated_text_retokenize_ok=False,
        )
        if input_ids is None:
            tol = (
                "blocked_missing_generated_token_ids"
                if mode == "prompt_plus_generated_tokens"
                else "blocked"
            )
            cells.append({
                "shadow_sequence_mode": seq_mode,
                "shadow_mode_requested": mode,
                "shadow_sequence_length": 0,
                "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
                "tolerance_policy_status": tol,
                "streaming_vs_materialized_metrics": None,
                "full_vs_streaming_metrics": None,
                "topk_agreement_metrics": None,
                "interpretation_note": "Could not reconstruct shadow sequence.",
                "blockers": recon_blockers,
            })
            continue

        if hf_model is None and shadow_replay_fn is None:
            cells.append({
                "shadow_sequence_mode": seq_mode,
                "shadow_mode_requested": mode,
                "shadow_sequence_length": int(input_ids.shape[-1]),
                "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
                "tolerance_policy_status": "blocked",
                "streaming_vs_materialized_metrics": None,
                "full_vs_streaming_metrics": None,
                "topk_agreement_metrics": None,
                "interpretation_note": "HF model unavailable for shadow replay.",
                "blockers": ["hf model missing for shadow replay"],
            })
            continue

        replay = shadow_replay_fn or default_offline_shadow_replay
        try:
            shadow_cell = replay(
                model=hf_model,
                input_ids=input_ids,
                prompt_id=prompt_id,
                chunk_size=chunk_size,
                accumulator_mode=accumulator_mode,
                allow_parity_fail=allow_parity_fail,
            )
        except Exception as exc:  # noqa: BLE001
            shadow_cell = {"blockers": [f"shadow replay failed: {type(exc).__name__}: {exc}"]}

        if shadow_cell.get("blockers") and not allow_shadow_fail:
            shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
        elif shadow_cell.get("blockers"):
            shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
        else:
            shadow_status = GenerationShadowStatus.SHADOW_COMPLETE.value

        num_layers = int(shadow_cell.get("num_layers_replayed") or 24)
        tol_status, interp = apply_tolerance_policy_to_shadow_cell(
            shadow_cell, num_layers=num_layers,
        )
        sm = shadow_cell.get("streaming_vs_materialized_logit_metrics") or {}
        fs = shadow_cell.get("full_vs_streaming_logit_metrics") or {}
        topk = {
            "top1_agreement": sm.get("top1_agreement"),
            "top5_overlap": sm.get("top5_overlap"),
            "top10_overlap": sm.get("top10_overlap"),
        }
        cells.append({
            "shadow_sequence_mode": seq_mode,
            "shadow_mode_requested": mode,
            "shadow_sequence_length": int(input_ids.shape[-1]),
            "shadow_status": shadow_status,
            "tolerance_policy_status": tol_status,
            "streaming_vs_materialized_metrics": sm,
            "full_vs_streaming_metrics": fs,
            "topk_agreement_metrics": topk,
            "interpretation_note": interp,
            "blockers": list(shadow_cell.get("blockers") or []),
        })
    return cells


def _summarize_exactkv_failures(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failures = 0
    matches = 0
    unknown = 0
    for cell in cells:
        ef = cell.get("exactkv_failures")
        if ef is None:
            unknown += 1
        elif int(ef) > 0:
            failures += 1
        tem = cell.get("token_exact_match")
        if tem is True:
            matches += 1
    return {
        "cells_with_exactkv_failures": failures,
        "cells_with_token_exact_match": matches,
        "cells_with_unknown_exactkv_status": unknown,
        "total_generation_cells": len(cells),
    }


def run_exp078_expanded_panel(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_new_tokens_values: Sequence[int] = DEFAULT_EXP078_MAX_NEW_TOKENS,
    shadow_modes: Sequence[str] = DEFAULT_EXP078_SHADOW_MODES,
    compressors_requested: Sequence[str] = DEFAULT_EXP078_COMPRESSORS,
    draft_len: int = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    accumulator_mode: str = "float32",
    allow_shadow_fail: bool = True,
    allow_parity_fail: bool = True,
    local_files_only: bool = False,
    generation_fn: Callable[..., GenerationOutput] | None = None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 078 expanded external generation-shadow panel."""
    prompt_panel = list(prompts) if prompts is not None else default_exp078_prompts()
    runnable_compressors, blocked_compressors = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if not runnable_compressors:
        blockers.append("no compressors runnable via get_compressor API")

    runtime: Any | None = None
    if generation_fn is None:
        if runtime_loader is not None:
            try:
                runtime, _tokenizer = runtime_loader(
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")
        else:
            try:
                from exactkv.runtime.model_runtime import ModelRuntime

                runtime = ModelRuntime(model_id, device=device, dtype=dtype)
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    hf_model = getattr(runtime, "model", None) if runtime is not None else None
    baseline_cache: dict[tuple[str, int], list[int]] = {}

    generation_cells: list[dict[str, Any]] = []
    gen_success = 0
    gen_blocked = 0
    shadow_success = 0
    shadow_blocked = 0
    ppg_success = 0
    ppg_blocked = 0
    token_ids_available = 0
    tol_summary: dict[str, int] = {}
    topk_top1 = 0

    for prompt_id, prompt_text in prompt_panel:
        preview = _preview(prompt_text)
        for max_new in max_new_tokens_values:
            baseline_key = (prompt_id, max_new)
            for compressor in runnable_compressors:
                cell_blockers: list[str] = []
                if runtime is None and generation_fn is None:
                    gen_blocked += 1
                    generation_cells.append({
                        "prompt_id": prompt_id,
                        "prompt_preview": preview,
                        "compressor": compressor,
                        "max_new_tokens": max_new,
                        "generation_completed": False,
                        "exactkv_failures": None,
                        "token_exact_match": None,
                        "generation_output_preview": "",
                        "generation_output_token_ids_available": False,
                        "generation_output_token_count": 0,
                        "generation_modified_by_shadow": False,
                        "shadow_used_for_token_commit": False,
                        "shadow_cells": [],
                        "blockers": list(blockers),
                    })
                    continue

                if generation_fn is not None:
                    gen_out = generation_fn(
                        prompt=prompt_text,
                        max_new_tokens=max_new,
                        compressor_name=compressor,
                    )
                else:
                    full_ids = baseline_cache.get(baseline_key)
                    gen_out = run_exactkv_generation_with_baseline(
                        runtime=runtime,
                        prompt=prompt_text,
                        max_new_tokens=max_new,
                        compressor_name=compressor,
                        draft_len=draft_len,
                        full_baseline_ids=full_ids,
                    )
                    if full_ids is None and gen_out.generation_completed:
                        try:
                            from exactkv.runtime.generation import generate_full_greedy

                            full_res = generate_full_greedy(runtime, prompt_text, max_new)
                            full_ids = full_res.generated_ids.squeeze().tolist()
                            if isinstance(full_ids, int):
                                full_ids = [full_ids]
                            baseline_cache[baseline_key] = full_ids
                        except Exception:  # noqa: BLE001
                            pass

                if not gen_out.generation_completed:
                    gen_blocked += 1
                    cell_blockers.extend(gen_out.blockers)
                else:
                    gen_success += 1

                if gen_out.generation_output_token_ids:
                    token_ids_available += 1

                shadow_cells = []
                if gen_out.generation_completed:
                    shadow_cells = run_shadow_cells_for_generation(
                        gen_out=gen_out,
                        prompt_id=prompt_id,
                        shadow_modes=shadow_modes,
                        hf_model=hf_model,
                        shadow_replay_fn=shadow_replay_fn,
                        chunk_size=chunk_size,
                        accumulator_mode=accumulator_mode,
                        allow_parity_fail=allow_parity_fail,
                        allow_shadow_fail=allow_shadow_fail,
                    )
                elif gen_out.generation_completed and shadow_replay_fn is None and hf_model is None:
                    for mode in shadow_modes:
                        shadow_blocked += 1
                        if mode == "prompt_plus_generated_tokens":
                            ppg_blocked += 1
                        shadow_cells.append({
                            "shadow_sequence_mode": mode,
                            "shadow_mode_requested": mode,
                            "shadow_sequence_length": 0,
                            "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
                            "tolerance_policy_status": "blocked",
                            "streaming_vs_materialized_metrics": None,
                            "full_vs_streaming_metrics": None,
                            "topk_agreement_metrics": None,
                            "interpretation_note": "HF model unavailable.",
                            "blockers": ["hf model missing"],
                        })

                for sc in shadow_cells:
                    if sc.get("shadow_status") == GenerationShadowStatus.SHADOW_COMPLETE.value:
                        shadow_success += 1
                        if sc.get("shadow_mode_requested") == "prompt_plus_generated_tokens":
                            ppg_success += 1
                        tol = sc.get("tolerance_policy_status")
                        if tol:
                            tol_summary[tol] = tol_summary.get(tol, 0) + 1
                        if sc.get("topk_agreement_metrics", {}).get("top1_agreement"):
                            topk_top1 += 1
                    else:
                        shadow_blocked += 1
                        if sc.get("shadow_mode_requested") == "prompt_plus_generated_tokens":
                            ppg_blocked += 1

                generation_cells.append({
                    "prompt_id": prompt_id,
                    "prompt_preview": preview,
                    "compressor": compressor,
                    "max_new_tokens": max_new,
                    "generation_completed": bool(gen_out.generation_completed),
                    "exactkv_failures": gen_out.exactkv_failures,
                    "token_exact_match": gen_out.token_exact_match,
                    "generation_output_preview": _preview(gen_out.generation_output_text),
                    "generation_output_token_ids_available": bool(gen_out.generation_output_token_ids),
                    "generation_output_token_count": len(gen_out.generation_output_token_ids or []),
                    "generation_modified_by_shadow": False,
                    "shadow_used_for_token_commit": False,
                    "shadow_cells": shadow_cells,
                    "blockers": cell_blockers,
                })

    total_gen = len(generation_cells)
    total_shadow = shadow_success + shadow_blocked
    if gen_success == 0:
        status = "blocked"
    elif shadow_success == total_shadow and total_shadow > 0:
        status = "diagnostic_complete"
    elif shadow_success > 0:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_078_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "prompts_requested": len(prompt_panel),
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable_compressors,
        "compressors_blocked": blocked_compressors,
        "max_new_tokens_values": list(max_new_tokens_values),
        "shadow_modes": list(shadow_modes),
        "total_generation_cells": total_gen,
        "generation_successful_cells": gen_success,
        "generation_blocked_cells": gen_blocked,
        "total_shadow_cells": total_shadow,
        "shadow_successful_cells": shadow_success,
        "shadow_blocked_cells": shadow_blocked,
        "prompt_plus_generated_successful_cells": ppg_success,
        "prompt_plus_generated_blocked_cells": ppg_blocked,
        "generated_token_ids_available_cells": token_ids_available,
        "generation_modified_by_shadow": False,
        "shadow_used_for_token_commit": False,
        "default_runtime_changed": False,
        "exactkv_failure_summary": _summarize_exactkv_failures(generation_cells),
        "tolerance_policy_summary": tol_summary,
        "topk_agreement_summary": {
            "top1_agreement_cells": topk_top1,
            "cell_count": shadow_success,
        },
        "generation_cells": generation_cells,
        "blockers": blockers,
        "limitations": [
            "External expanded post-hoc observer panel; not generation integration.",
            "Compressor results only when get_compressor API exposes them cleanly.",
            "Prompt+generated replay is fixed-sequence analysis, not token generation.",
            "Top-k agreement is supplementary; not exactness.",
            "No per-round decode observer.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP078_CLAIM_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "cli_flag": PROPOSED_SHADOW_CLI_FLAG,
    }


def validate_exp078_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "prompts_requested",
        "compressors_requested",
        "compressors_run",
        "compressors_blocked",
        "max_new_tokens_values",
        "shadow_modes",
        "total_generation_cells",
        "generation_successful_cells",
        "generation_blocked_cells",
        "total_shadow_cells",
        "shadow_successful_cells",
        "shadow_blocked_cells",
        "prompt_plus_generated_successful_cells",
        "prompt_plus_generated_blocked_cells",
        "generated_token_ids_available_cells",
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
        "exactkv_failure_summary",
        "tolerance_policy_summary",
        "topk_agreement_summary",
        "generation_cells",
        "blockers",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_078_ID:
        errors.append("experiment_id mismatch")

    for flag in (
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    for idx, cell in enumerate(report.get("generation_cells", [])):
        if not isinstance(cell, dict):
            errors.append(f"generation_cells[{idx}] not dict")
            continue
        for ck in (
            "prompt_id",
            "compressor",
            "max_new_tokens",
            "generation_completed",
            "generation_output_token_ids_available",
            "generation_modified_by_shadow",
            "shadow_used_for_token_commit",
            "shadow_cells",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"generation_cells[{idx}] missing {ck}")
        if cell.get("generation_modified_by_shadow") is not False:
            errors.append(f"generation_cells[{idx}].generation_modified_by_shadow must be false")
        if cell.get("shadow_used_for_token_commit") is not False:
            errors.append(f"generation_cells[{idx}].shadow_used_for_token_commit must be false")

    return errors


# --- Phase 16N: decode-prefix ladder shadow observer ---

EXPERIMENT_079_ID = "exp079_decode_prefix_ladder_shadow_observer"
DEFAULT_EXP079_REPORT = Path("reports/experiment_079_decode_prefix_ladder_shadow_observer.json")
DEFAULT_EXP079_COMPRESSORS: tuple[str, ...] = DEFAULT_EXP078_COMPRESSORS
DEFAULT_EXP079_MAX_NEW_TOKENS = 8
ROUND_SOURCE_LIVE = "live_round_observer"
ROUND_SOURCE_POSTHOC = "posthoc_prefix_ladder"
ROUND_SOURCE_ROUND_LOG = "exactkv_round_log"
ROUND_SOURCE_BLOCKED = "blocked_no_round_data"

EXP079_CLAIM_NOTE = (
    "Post-hoc decode-prefix ladder shadow observer (Phase 16N). Runs ExactKV generation "
    "unchanged, then offline shadow diagnostics on prompt + first-k generated token "
    "prefixes. Not live decode integration, vLLM, CUDA/Triton kernels, or default "
    "runtime change. Shadow logits/top-k are diagnostic only."
)


def default_exp079_prompts() -> list[tuple[str, str]]:
    """Four deterministic prompts for decode-prefix ladder panel."""
    return default_exp078_prompts()[:4]


def resolve_round_source(gen_out: GenerationOutput) -> str:
    """Classify decode-round data source without generator hooks."""
    if gen_out.prompt_ids is None or not gen_out.generation_output_token_ids:
        return ROUND_SOURCE_BLOCKED
    if gen_out.exactkv_traces:
        return ROUND_SOURCE_ROUND_LOG
    return ROUND_SOURCE_POSTHOC


def build_decode_prefix_ladder(
    gen_out: GenerationOutput,
    *,
    ladder_stride: int = 1,
    allow_generated_text_retokenize: bool = False,
) -> tuple[list[tuple[int, torch.Tensor]], list[str]]:
    """Build k=0..N prefix ladder from prompt + generated token IDs."""
    blockers: list[str] = []
    prompt_ids = gen_out.prompt_ids
    if prompt_ids is None:
        return [], ["prompt token IDs unavailable"]

    gen_token_ids = gen_out.generation_output_token_ids
    if not gen_token_ids and allow_generated_text_retokenize:
        blockers.append("retokenization not implemented for ladder; pass token IDs")
    if not gen_token_ids:
        return [], ["generated token IDs unavailable"]

    stride = max(1, int(ladder_stride))
    n_gen = len(gen_token_ids)
    ks = list(range(0, n_gen + 1, stride))
    if ks[-1] != n_gen:
        ks.append(n_gen)

    ladder: list[tuple[int, torch.Tensor]] = []
    for k in ks:
        if k == 0:
            ladder.append((0, prompt_ids))
            continue
        prefix = gen_token_ids[:k]
        gen_tensor = torch.tensor([prefix], dtype=prompt_ids.dtype, device=prompt_ids.device)
        ladder.append((k, torch.cat([prompt_ids, gen_tensor], dim=1)))
    return ladder, blockers


def _shadow_prefix_cell(
    *,
    generated_prefix_length: int,
    input_ids: torch.Tensor,
    prompt_id: str,
    hf_model: Any | None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None,
    chunk_size: int,
    accumulator_mode: str,
    allow_parity_fail: bool,
    allow_shadow_fail: bool,
) -> dict[str, Any]:
    """Run offline shadow on one decode-prefix step."""
    if hf_model is None and shadow_replay_fn is None:
        return {
            "generated_prefix_length": generated_prefix_length,
            "shadow_sequence_length": int(input_ids.shape[-1]),
            "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
            "tolerance_policy_status": "blocked",
            "streaming_vs_materialized_metrics": None,
            "full_vs_streaming_metrics": None,
            "topk_agreement_metrics": None,
            "interpretation_note": "HF model unavailable for shadow replay.",
            "blockers": ["hf model missing for shadow replay"],
        }

    replay = shadow_replay_fn or default_offline_shadow_replay
    try:
        shadow_cell = replay(
            model=hf_model,
            input_ids=input_ids,
            prompt_id=prompt_id,
            chunk_size=chunk_size,
            accumulator_mode=accumulator_mode,
            allow_parity_fail=allow_parity_fail,
        )
    except Exception as exc:  # noqa: BLE001
        shadow_cell = {"blockers": [f"shadow replay failed: {type(exc).__name__}: {exc}"]}

    if shadow_cell.get("blockers") and not allow_shadow_fail:
        shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
    elif shadow_cell.get("blockers"):
        shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
    else:
        shadow_status = GenerationShadowStatus.SHADOW_COMPLETE.value

    num_layers = int(shadow_cell.get("num_layers_replayed") or 24)
    tol_status, interp = apply_tolerance_policy_to_shadow_cell(
        shadow_cell, num_layers=num_layers,
    )
    sm = shadow_cell.get("streaming_vs_materialized_logit_metrics") or {}
    fs = shadow_cell.get("full_vs_streaming_logit_metrics") or {}
    topk = {
        "top1_agreement": sm.get("top1_agreement"),
        "top5_overlap": sm.get("top5_overlap"),
        "top10_overlap": sm.get("top10_overlap"),
    }
    return {
        "generated_prefix_length": generated_prefix_length,
        "shadow_sequence_length": int(input_ids.shape[-1]),
        "shadow_status": shadow_status,
        "tolerance_policy_status": tol_status,
        "streaming_vs_materialized_metrics": sm,
        "full_vs_streaming_metrics": fs,
        "topk_agreement_metrics": topk,
        "interpretation_note": interp,
        "blockers": list(shadow_cell.get("blockers") or []),
    }


def run_prefix_ladder_shadow_for_generation(
    *,
    gen_out: GenerationOutput,
    prompt_id: str,
    hf_model: Any | None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None,
    chunk_size: int,
    accumulator_mode: str,
    allow_parity_fail: bool,
    allow_shadow_fail: bool,
    ladder_stride: int = 1,
    allow_generated_text_retokenize: bool = False,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Run post-hoc decode-prefix ladder shadow diagnostics."""
    ladder, blockers = build_decode_prefix_ladder(
        gen_out,
        ladder_stride=ladder_stride,
        allow_generated_text_retokenize=allow_generated_text_retokenize,
    )
    round_source = resolve_round_source(gen_out)
    if not ladder:
        return [], round_source, blockers

    prefix_cells: list[dict[str, Any]] = []
    for k, input_ids in ladder:
        prefix_cells.append(
            _shadow_prefix_cell(
                generated_prefix_length=k,
                input_ids=input_ids,
                prompt_id=prompt_id,
                hf_model=hf_model,
                shadow_replay_fn=shadow_replay_fn,
                chunk_size=chunk_size,
                accumulator_mode=accumulator_mode,
                allow_parity_fail=allow_parity_fail,
                allow_shadow_fail=allow_shadow_fail,
            )
        )
    return prefix_cells, round_source, blockers


def _generation_safety_gates(
    *,
    generation_completed: bool,
    shadow_ran: bool,
) -> dict[str, bool]:
    return {
        "generation_completed": generation_completed,
        "generated_output_unchanged": True,
        "shadow_ran_after_generation": shadow_ran,
        "shadow_used_for_token_commit": False,
        "generation_modified_by_shadow": False,
        "default_runtime_changed": False,
    }


def _aggregate_tolerance_by_prefix(
    prefix_cells: Sequence[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    by_k: dict[str, dict[str, int]] = {}
    for cell in prefix_cells:
        k = str(cell.get("generated_prefix_length", ""))
        status = cell.get("tolerance_policy_status")
        if status is None:
            continue
        by_k.setdefault(k, {})
        by_k[k][status] = by_k[k].get(status, 0) + 1
    return by_k


def _aggregate_topk_by_prefix(
    prefix_cells: Sequence[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    by_k: dict[str, dict[str, int]] = {}
    for cell in prefix_cells:
        k = str(cell.get("generated_prefix_length", ""))
        topk = cell.get("topk_agreement_metrics") or {}
        by_k.setdefault(k, {"top1_agreement_true": 0, "top1_agreement_false": 0, "top1_agreement_unknown": 0})
        agree = topk.get("top1_agreement")
        if agree is True:
            by_k[k]["top1_agreement_true"] += 1
        elif agree is False:
            by_k[k]["top1_agreement_false"] += 1
        else:
            by_k[k]["top1_agreement_unknown"] += 1
    return by_k


def _max_drift_by_prefix(
    prefix_cells: Sequence[dict[str, Any]],
    metrics_key: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for cell in prefix_cells:
        k = str(cell.get("generated_prefix_length", ""))
        metrics = cell.get(metrics_key) or {}
        err = metrics.get("max_abs_error")
        if err is None:
            continue
        out[k] = max(float(out.get(k, 0.0)), float(err))
    return out


def _first_status_change_for_cell(
    prefix_cells: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not prefix_cells:
        return None
    base = prefix_cells[0].get("tolerance_policy_status")
    for cell in prefix_cells[1:]:
        status = cell.get("tolerance_policy_status")
        if status != base:
            return {
                "first_change_at_prefix_length": cell.get("generated_prefix_length"),
                "from_status": base,
                "to_status": status,
            }
    return None


def _first_top1_mismatch_for_cell(
    prefix_cells: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    for cell in prefix_cells:
        topk = cell.get("topk_agreement_metrics") or {}
        if topk.get("top1_agreement") is False:
            return {
                "first_mismatch_at_prefix_length": cell.get("generated_prefix_length"),
            }
    return None


def _aggregate_first_status_changes(
    generation_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    stable = 0
    for gc in generation_cells:
        pcs = gc.get("prefix_shadow_cells") or []
        change = _first_status_change_for_cell(pcs)
        if change is None:
            stable += 1
        else:
            changes.append({
                "prompt_id": gc.get("prompt_id"),
                "compressor": gc.get("compressor"),
                **change,
            })
    return {
        "cells_with_status_change": len(changes),
        "cells_all_stable": stable,
        "changes": changes,
    }


def _aggregate_first_top1_mismatches(
    generation_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    all_agree = 0
    for gc in generation_cells:
        pcs = gc.get("prefix_shadow_cells") or []
        mismatch = _first_top1_mismatch_for_cell(pcs)
        if mismatch is None:
            all_agree += 1
        else:
            mismatches.append({
                "prompt_id": gc.get("prompt_id"),
                "compressor": gc.get("compressor"),
                **mismatch,
            })
    return {
        "cells_with_top1_mismatch": len(mismatches),
        "cells_all_top1_agree": all_agree,
        "mismatches": mismatches,
    }


def run_exp079_decode_prefix_ladder_panel(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_new_tokens: int = DEFAULT_EXP079_MAX_NEW_TOKENS,
    compressors_requested: Sequence[str] = DEFAULT_EXP079_COMPRESSORS,
    draft_len: int = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    accumulator_mode: str = "float32",
    ladder_stride: int = 1,
    allow_shadow_fail: bool = True,
    allow_parity_fail: bool = True,
    allow_generated_text_retokenize: bool = False,
    local_files_only: bool = False,
    generation_fn: Callable[..., GenerationOutput] | None = None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 079 post-hoc decode-prefix ladder shadow observer."""
    prompt_panel = list(prompts) if prompts is not None else default_exp079_prompts()
    runnable_compressors, _blocked_compressors = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if not runnable_compressors:
        blockers.append("no compressors runnable via get_compressor API")

    runtime: Any | None = None
    if generation_fn is None:
        if runtime_loader is not None:
            try:
                runtime, _tokenizer = runtime_loader(
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")
        else:
            try:
                from exactkv.runtime.model_runtime import ModelRuntime

                runtime = ModelRuntime(model_id, device=device, dtype=dtype)
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    hf_model = getattr(runtime, "model", None) if runtime is not None else None
    baseline_cache: dict[str, list[int]] = {}

    generation_cells: list[dict[str, Any]] = []
    all_prefix_cells: list[dict[str, Any]] = []
    gen_success = 0
    gen_blocked = 0
    prefix_success = 0
    prefix_blocked = 0
    round_source_counts: dict[str, int] = {}
    max_prefix_len = 0

    for prompt_id, prompt_text in prompt_panel:
        preview = _preview(prompt_text)
        for compressor in runnable_compressors:
            cell_blockers: list[str] = []
            if runtime is None and generation_fn is None:
                gen_blocked += 1
                generation_cells.append({
                    "prompt_id": prompt_id,
                    "prompt_preview": preview,
                    "compressor": compressor,
                    "max_new_tokens": max_new_tokens,
                    "generation_completed": False,
                    "exactkv_failures": None,
                    "token_exact_match": None,
                    "generation_output_preview": "",
                    "generation_output_token_count": 0,
                    "generation_output_token_ids_available": False,
                    "round_source": ROUND_SOURCE_BLOCKED,
                    "prefix_shadow_cells": [],
                    "safety_gates": _generation_safety_gates(
                        generation_completed=False, shadow_ran=False,
                    ),
                    "blockers": list(blockers),
                })
                continue

            if generation_fn is not None:
                gen_out = generation_fn(
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                )
            else:
                full_ids = baseline_cache.get(prompt_id)
                gen_out = run_exactkv_generation_with_baseline(
                    runtime=runtime,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                    draft_len=draft_len,
                    full_baseline_ids=full_ids,
                )
                if full_ids is None and gen_out.generation_completed:
                    try:
                        from exactkv.runtime.generation import generate_full_greedy

                        full_res = generate_full_greedy(runtime, prompt_text, max_new_tokens)
                        full_ids = full_res.generated_ids.squeeze().tolist()
                        if isinstance(full_ids, int):
                            full_ids = [full_ids]
                        baseline_cache[prompt_id] = full_ids
                    except Exception:  # noqa: BLE001
                        pass

            generation_completed = bool(gen_out.generation_completed)
            if generation_completed:
                gen_success += 1
            else:
                gen_blocked += 1
                cell_blockers.extend(gen_out.blockers)

            prefix_cells: list[dict[str, Any]] = []
            round_source = ROUND_SOURCE_BLOCKED
            ladder_blockers: list[str] = []
            shadow_ran = False

            if generation_completed:
                prefix_cells, round_source, ladder_blockers = run_prefix_ladder_shadow_for_generation(
                    gen_out=gen_out,
                    prompt_id=prompt_id,
                    hf_model=hf_model,
                    shadow_replay_fn=shadow_replay_fn,
                    chunk_size=chunk_size,
                    accumulator_mode=accumulator_mode,
                    allow_parity_fail=allow_parity_fail,
                    allow_shadow_fail=allow_shadow_fail,
                    ladder_stride=ladder_stride,
                    allow_generated_text_retokenize=allow_generated_text_retokenize,
                )
                shadow_ran = bool(prefix_cells)
                cell_blockers.extend(ladder_blockers)

            round_source_counts[round_source] = round_source_counts.get(round_source, 0) + 1
            for pc in prefix_cells:
                all_prefix_cells.append(pc)
                if pc.get("generated_prefix_length", 0) > max_prefix_len:
                    max_prefix_len = int(pc["generated_prefix_length"])
                if pc.get("shadow_status") == GenerationShadowStatus.SHADOW_COMPLETE.value:
                    prefix_success += 1
                else:
                    prefix_blocked += 1

            safety = _generation_safety_gates(
                generation_completed=generation_completed,
                shadow_ran=shadow_ran,
            )
            generation_cells.append({
                "prompt_id": prompt_id,
                "prompt_preview": preview,
                "compressor": compressor,
                "max_new_tokens": max_new_tokens,
                "generation_completed": generation_completed,
                "exactkv_failures": gen_out.exactkv_failures,
                "token_exact_match": gen_out.token_exact_match,
                "generation_output_preview": _preview(gen_out.generation_output_text),
                "generation_output_token_count": len(gen_out.generation_output_token_ids or []),
                "generation_output_token_ids_available": bool(gen_out.generation_output_token_ids),
                "round_source": round_source,
                "prefix_shadow_cells": prefix_cells,
                "safety_gates": safety,
                "blockers": cell_blockers,
            })

    total_prefix = prefix_success + prefix_blocked
    safety_ok = all(
        gc.get("safety_gates", {}).get("shadow_used_for_token_commit") is False
        and gc.get("safety_gates", {}).get("generation_modified_by_shadow") is False
        and gc.get("safety_gates", {}).get("default_runtime_changed") is False
        for gc in generation_cells
    )

    if not safety_ok:
        status = "failed"
    elif gen_success == 0:
        status = "blocked"
    elif prefix_success == total_prefix and total_prefix > 0:
        status = "diagnostic_complete"
    elif prefix_success > 0:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_079_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "ladder_stride": ladder_stride,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable_compressors,
        "total_generation_cells": len(generation_cells),
        "generation_successful_cells": gen_success,
        "generation_blocked_cells": gen_blocked,
        "total_prefix_shadow_cells": total_prefix,
        "prefix_shadow_successful_cells": prefix_success,
        "prefix_shadow_blocked_cells": prefix_blocked,
        "max_generated_prefix_length": max_prefix_len,
        "round_source_counts": round_source_counts,
        "tolerance_policy_summary_by_prefix_length": _aggregate_tolerance_by_prefix(all_prefix_cells),
        "topk_agreement_summary_by_prefix_length": _aggregate_topk_by_prefix(all_prefix_cells),
        "first_status_change_summary": _aggregate_first_status_changes(generation_cells),
        "first_top1_mismatch_summary": _aggregate_first_top1_mismatches(generation_cells),
        "max_full_vs_streaming_drift_by_prefix_length": _max_drift_by_prefix(
            all_prefix_cells, "full_vs_streaming_metrics",
        ),
        "max_streaming_vs_materialized_drift_by_prefix_length": _max_drift_by_prefix(
            all_prefix_cells, "streaming_vs_materialized_metrics",
        ),
        "exactkv_failure_summary": _summarize_exactkv_failures(generation_cells),
        "generation_modified_by_shadow": False,
        "shadow_used_for_token_commit": False,
        "default_runtime_changed": False,
        "generation_cells": generation_cells,
        "blockers": blockers,
        "limitations": [
            "Post-hoc decode-prefix ladder observer; not live decode integration.",
            "Prefix ladder replay is fixed-sequence analysis, not token generation.",
            "Top-k agreement is supplementary; not exactness.",
            "No live per-round decode hooks.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP079_CLAIM_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "cli_flag": PROPOSED_SHADOW_CLI_FLAG,
    }


def validate_exp079_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "max_new_tokens",
        "compressors_requested",
        "compressors_run",
        "total_generation_cells",
        "generation_successful_cells",
        "total_prefix_shadow_cells",
        "prefix_shadow_successful_cells",
        "prefix_shadow_blocked_cells",
        "max_generated_prefix_length",
        "round_source_counts",
        "tolerance_policy_summary_by_prefix_length",
        "topk_agreement_summary_by_prefix_length",
        "first_status_change_summary",
        "first_top1_mismatch_summary",
        "exactkv_failure_summary",
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
        "generation_cells",
        "blockers",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_079_ID:
        errors.append("experiment_id mismatch")

    for flag in (
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    for idx, cell in enumerate(report.get("generation_cells", [])):
        if not isinstance(cell, dict):
            errors.append(f"generation_cells[{idx}] not dict")
            continue
        for ck in (
            "prompt_id",
            "compressor",
            "max_new_tokens",
            "generation_completed",
            "generation_output_token_ids_available",
            "round_source",
            "prefix_shadow_cells",
            "safety_gates",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"generation_cells[{idx}] missing {ck}")
        gates = cell.get("safety_gates") or {}
        if gates.get("shadow_used_for_token_commit") is not False:
            errors.append(f"generation_cells[{idx}].safety_gates.shadow_used_for_token_commit must be false")
        if gates.get("generation_modified_by_shadow") is not False:
            errors.append(f"generation_cells[{idx}].safety_gates.generation_modified_by_shadow must be false")
        if gates.get("default_runtime_changed") is not False:
            errors.append(f"generation_cells[{idx}].safety_gates.default_runtime_changed must be false")

    return errors


# --- Phase 16O: ExactKV round-log shadow observer ---

EXPERIMENT_080_ID = "exp080_round_log_shadow_observer"
DEFAULT_EXP080_REPORT = Path("reports/experiment_080_round_log_shadow_observer.json")
DEFAULT_EXP080_COMPRESSORS: tuple[str, ...] = DEFAULT_EXP079_COMPRESSORS
DEFAULT_EXP080_MAX_NEW_TOKENS = DEFAULT_EXP079_MAX_NEW_TOKENS
BLOCKED_MISSING_ROUND_LOG = "blocked_missing_round_log"

EXP080_CLAIM_NOTE = (
    "Post-hoc ExactKV round-log shadow observer (Phase 16O). Uses existing "
    "ExactKVResult round traces when available, then offline shadow diagnostics "
    "at round boundaries. Not live decode integration, vLLM, CUDA/Triton kernels, "
    "or default runtime change. Shadow logits/top-k are diagnostic only."
)


def default_exp080_prompts() -> list[tuple[str, str]]:
    """Four deterministic prompts for round-log shadow panel."""
    return default_exp079_prompts()


def _trace_value(trace: Any, key: str, default: Any = None) -> Any:
    if isinstance(trace, dict):
        return trace.get(key, default)
    return getattr(trace, key, default)


def _acceptance_value(acceptance: Any, key: str, default: Any = None) -> Any:
    if acceptance is None:
        return default
    if isinstance(acceptance, dict):
        return acceptance.get(key, default)
    return getattr(acceptance, key, default)


def extract_round_log_entries(
    gen_out: GenerationOutput,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract round-boundary metadata from existing ExactKVResult traces."""
    blockers: list[str] = []
    traces = gen_out.exactkv_traces
    if not traces:
        return [], ["missing ExactKV round log"]

    prompt_ids = gen_out.prompt_ids
    if prompt_ids is None:
        return [], ["prompt token IDs unavailable"]

    prompt_len = int(prompt_ids.shape[-1])
    final_gen_ids = list(gen_out.generation_output_token_ids or [])
    entries: list[dict[str, Any]] = []

    for trace in traces:
        round_idx = _trace_value(trace, "round_idx")
        draft_tokens = _trace_value(trace, "draft_tokens")
        acceptance = _trace_value(trace, "acceptance")
        full_before = _trace_value(trace, "full_seq_len_before")
        full_after = _trace_value(trace, "full_seq_len_after")

        draft_length: int | None
        if draft_tokens is None:
            draft_length = None
        else:
            draft_length = len(list(draft_tokens))

        accepted_count = _acceptance_value(acceptance, "num_accepted")
        rejected_count = _acceptance_value(acceptance, "num_rejected")
        correction = _acceptance_value(acceptance, "correction_token")
        rejected_or_corrected: int | None
        if rejected_count is None and correction is None:
            rejected_or_corrected = None
        else:
            rejected_or_corrected = int(rejected_count or 0) + (
                1 if correction is not None else 0
            )

        gen_len_after: int | None = None
        gen_ids_up_to: list[int] | None = None
        if full_after is not None:
            gen_len_after = int(full_after) - prompt_len
            if gen_len_after >= 0:
                gen_ids_up_to = final_gen_ids[:gen_len_after]

        entries.append({
            "round_index": round_idx,
            "prefix_length_before_round": full_before,
            "prefix_length_after_round": full_after,
            "draft_length": draft_length,
            "accepted_token_count": accepted_count,
            "rejected_or_corrected_token_count": rejected_or_corrected,
            "generated_token_ids_up_to_round": gen_ids_up_to,
            "final_generated_token_ids": final_gen_ids if final_gen_ids else None,
        })

    if not entries:
        blockers.append("round log contained no entries")
    return entries, blockers


def build_round_boundary_input_ids(
    gen_out: GenerationOutput,
    entry: dict[str, Any],
) -> tuple[torch.Tensor | None, list[str]]:
    """Build fixed sequence at an ExactKV round boundary (post-commit)."""
    plen_after = entry.get("prefix_length_after_round")
    if plen_after is None:
        return None, ["prefix_length_after_round unknown"]

    if gen_out.full_sequence_ids is not None:
        return gen_out.full_sequence_ids[:, : int(plen_after)], []

    prompt_ids = gen_out.prompt_ids
    gen_ids = entry.get("generated_token_ids_up_to_round")
    if prompt_ids is None:
        return None, ["prompt token IDs unavailable"]
    if gen_ids is None:
        return None, ["generated token IDs up to round unavailable"]

    gen_tensor = torch.tensor([gen_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
    return torch.cat([prompt_ids, gen_tensor], dim=1), []


def diagnostic_shadow_top1_fields(
    shadow_cell: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Explicit diagnostic-only top-1 fields from shadow replay output.

    These fields are for L3 proposal diagnostics only and must not drive token commits.
    """
    empty: dict[str, Any] = {
        "shadow_top1_token_id": None,
        "shadow_top1_token_text": None,
        "shadow_topk_token_ids": None,
    }
    if not shadow_cell or shadow_cell.get("blockers"):
        return empty

    sm_logit = shadow_cell.get("streaming_vs_materialized_logit_metrics") or {}
    top1 = shadow_cell.get("streaming_top1_token_id")
    if top1 is None:
        top1 = sm_logit.get("other_top1_token_id")

    top5 = shadow_cell.get("streaming_top5_token_ids")
    topk_ids: list[int] | None = None
    if top5 is not None:
        topk_ids = [int(x) for x in list(top5)]

    return {
        "shadow_top1_token_id": int(top1) if top1 is not None else None,
        "shadow_top1_token_text": shadow_cell.get("shadow_top1_token_text"),
        "shadow_topk_token_ids": topk_ids,
    }


def _shadow_round_cell(
    *,
    entry: dict[str, Any],
    input_ids: torch.Tensor,
    prompt_id: str,
    hf_model: Any | None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None,
    chunk_size: int,
    accumulator_mode: str,
    allow_parity_fail: bool,
    allow_shadow_fail: bool,
) -> dict[str, Any]:
    """Run offline shadow at one ExactKV round boundary."""
    base = {
        "round_index": entry.get("round_index"),
        "prefix_length_before_round": entry.get("prefix_length_before_round"),
        "prefix_length_after_round": entry.get("prefix_length_after_round"),
        "draft_length": entry.get("draft_length"),
        "accepted_token_count": entry.get("accepted_token_count"),
        "rejected_or_corrected_token_count": entry.get("rejected_or_corrected_token_count"),
    }

    if hf_model is None and shadow_replay_fn is None:
        return {
            **base,
            "shadow_sequence_length": int(input_ids.shape[-1]),
            "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
            "tolerance_policy_status": "blocked",
            "streaming_vs_materialized_metrics": None,
            "full_vs_streaming_metrics": None,
            "topk_agreement_metrics": None,
            "shadow_top1_token_id": None,
            "shadow_top1_token_text": None,
            "shadow_topk_token_ids": None,
            "streaming_top1_token_id": None,
            "streaming_top5_token_ids": None,
            "interpretation_note": "HF model unavailable for shadow replay.",
            "blockers": ["hf model missing for shadow replay"],
        }

    replay = shadow_replay_fn or default_offline_shadow_replay
    try:
        shadow_cell = replay(
            model=hf_model,
            input_ids=input_ids,
            prompt_id=prompt_id,
            chunk_size=chunk_size,
            accumulator_mode=accumulator_mode,
            allow_parity_fail=allow_parity_fail,
        )
    except Exception as exc:  # noqa: BLE001
        shadow_cell = {"blockers": [f"shadow replay failed: {type(exc).__name__}: {exc}"]}

    if shadow_cell.get("blockers") and not allow_shadow_fail:
        shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
    elif shadow_cell.get("blockers"):
        shadow_status = GenerationShadowStatus.SHADOW_BLOCKED.value
    else:
        shadow_status = GenerationShadowStatus.SHADOW_COMPLETE.value

    num_layers = int(shadow_cell.get("num_layers_replayed") or 24)
    tol_status, interp = apply_tolerance_policy_to_shadow_cell(
        shadow_cell, num_layers=num_layers,
    )
    sm = shadow_cell.get("streaming_vs_materialized_logit_metrics") or {}
    fs = shadow_cell.get("full_vs_streaming_logit_metrics") or {}
    diag = diagnostic_shadow_top1_fields(shadow_cell)
    topk = {
        "top1_agreement": sm.get("top1_agreement"),
        "top5_overlap": sm.get("top5_overlap"),
        "top10_overlap": sm.get("top10_overlap"),
        "shadow_top1_token_id": diag.get("shadow_top1_token_id"),
    }
    return {
        **base,
        "shadow_sequence_length": int(input_ids.shape[-1]),
        "shadow_status": shadow_status,
        "tolerance_policy_status": tol_status,
        "streaming_vs_materialized_metrics": sm,
        "full_vs_streaming_metrics": fs,
        "topk_agreement_metrics": topk,
        "shadow_top1_token_id": diag.get("shadow_top1_token_id"),
        "shadow_top1_token_text": diag.get("shadow_top1_token_text"),
        "shadow_topk_token_ids": diag.get("shadow_topk_token_ids"),
        "streaming_top1_token_id": shadow_cell.get("streaming_top1_token_id"),
        "streaming_top5_token_ids": shadow_cell.get("streaming_top5_token_ids"),
        "interpretation_note": interp,
        "blockers": list(shadow_cell.get("blockers") or []),
    }


def run_posthoc_shadow_from_live_snapshots(
    *,
    snapshots: Sequence[Any],
    prompt_id: str,
    hf_model: Any | None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    accumulator_mode: str = "float32",
    allow_parity_fail: bool = True,
    allow_shadow_fail: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run post-hoc shadow at live observer snapshot round boundaries."""
    if not snapshots:
        return [], ["missing live round observer snapshots"]

    blockers: list[str] = []
    cells: list[dict[str, Any]] = []
    for snap in snapshots:
        round_index = getattr(snap, "round_index", None)
        prefix_after = getattr(snap, "prefix_token_ids_after", None)
        if prefix_after is None:
            cells.append({
                "round_index": round_index,
                "shadow_sequence_length": 0,
                "tolerance_policy_status": "blocked",
                "streaming_vs_materialized_metrics": None,
                "full_vs_streaming_metrics": None,
                "topk_agreement_metrics": None,
                "interpretation_note": "Missing prefix_token_ids_after on live snapshot.",
                "blockers": ["live snapshot missing prefix_token_ids_after"],
            })
            continue

        input_ids = torch.tensor([list(prefix_after)], dtype=torch.long)
        meta = dict(getattr(snap, "metadata", ()) or ())
        entry = {
            "round_index": round_index,
            "prefix_length_before_round": meta.get("full_seq_len_before")
            or len(getattr(snap, "prefix_token_ids_before", ()) or ()),
            "prefix_length_after_round": meta.get("full_seq_len_after")
            or len(prefix_after),
            "draft_length": (
                len(snap.draft_token_ids) if getattr(snap, "draft_token_ids", None) else None
            ),
            "accepted_token_count": getattr(snap, "accepted_token_count", None),
            "rejected_or_corrected_token_count": getattr(
                snap, "rejected_or_corrected_token_count", None,
            ),
        }
        raw = _shadow_round_cell(
            entry=entry,
            input_ids=input_ids,
            prompt_id=prompt_id,
            hf_model=hf_model,
            shadow_replay_fn=shadow_replay_fn,
            chunk_size=chunk_size,
            accumulator_mode=accumulator_mode,
            allow_parity_fail=allow_parity_fail,
            allow_shadow_fail=allow_shadow_fail,
        )
        cells.append({
            "round_index": round_index,
            "shadow_sequence_length": raw.get("shadow_sequence_length", 0),
            "shadow_status": raw.get("shadow_status"),
            "tolerance_policy_status": raw.get("tolerance_policy_status"),
            "streaming_vs_materialized_metrics": raw.get("streaming_vs_materialized_metrics"),
            "full_vs_streaming_metrics": raw.get("full_vs_streaming_metrics"),
            "topk_agreement_metrics": raw.get("topk_agreement_metrics"),
            "shadow_top1_token_id": raw.get("shadow_top1_token_id"),
            "shadow_top1_token_text": raw.get("shadow_top1_token_text"),
            "shadow_topk_token_ids": raw.get("shadow_topk_token_ids"),
            "streaming_top1_token_id": raw.get("streaming_top1_token_id"),
            "streaming_top5_token_ids": raw.get("streaming_top5_token_ids"),
            "interpretation_note": raw.get("interpretation_note", ""),
            "blockers": list(raw.get("blockers") or []),
        })
    return cells, blockers


def aggregate_first_status_changes_from_shadow_cells(
    generation_cells: Sequence[dict[str, Any]],
    *,
    shadow_cells_key: str = "posthoc_shadow_cells",
) -> dict[str, Any]:
    """Aggregate first tolerance status change across shadow cell lists."""
    changes: list[dict[str, Any]] = []
    stable = 0
    for gc in generation_cells:
        scs = gc.get(shadow_cells_key) or []
        change = _first_status_change_for_round_cells(scs)
        if change is None:
            stable += 1
        else:
            changes.append({
                "prompt_id": gc.get("prompt_id"),
                "compressor": gc.get("compressor"),
                **change,
            })
    return {
        "cells_with_status_change": len(changes),
        "cells_all_stable": stable,
        "changes": changes,
    }


def aggregate_first_top1_mismatches_from_shadow_cells(
    generation_cells: Sequence[dict[str, Any]],
    *,
    shadow_cells_key: str = "posthoc_shadow_cells",
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    all_agree = 0
    for gc in generation_cells:
        scs = gc.get(shadow_cells_key) or []
        mismatch = _first_top1_mismatch_for_round_cells(scs)
        if mismatch is None:
            all_agree += 1
        else:
            mismatches.append({
                "prompt_id": gc.get("prompt_id"),
                "compressor": gc.get("compressor"),
                **mismatch,
            })
    return {
        "cells_with_top1_mismatch": len(mismatches),
        "cells_all_top1_agree": all_agree,
        "mismatches": mismatches,
    }


def run_round_log_shadow_for_generation(
    *,
    gen_out: GenerationOutput,
    prompt_id: str,
    hf_model: Any | None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None,
    chunk_size: int,
    accumulator_mode: str,
    allow_parity_fail: bool,
    allow_shadow_fail: bool,
    fallback_prefix_ladder: bool = False,
    ladder_stride: int = 1,
) -> tuple[list[dict[str, Any]], bool, bool, list[str]]:
    """Run post-hoc round-log shadow; optional prefix-ladder fallback."""
    entries, blockers = extract_round_log_entries(gen_out)
    had_round_log = bool(entries)
    used_fallback = False

    if not entries:
        if not fallback_prefix_ladder:
            return [], False, False, blockers
        ladder, ladder_blockers = build_decode_prefix_ladder(
            gen_out, ladder_stride=ladder_stride,
        )
        if not ladder:
            return [], False, False, blockers + ladder_blockers
        used_fallback = True
        prompt_len = int(gen_out.prompt_ids.shape[-1]) if gen_out.prompt_ids is not None else 0
        entries = []
        for k, input_ids in ladder:
            entries.append({
                "round_index": k,
                "prefix_length_before_round": prompt_len + max(0, k - 1) if k > 0 else prompt_len,
                "prefix_length_after_round": int(input_ids.shape[-1]),
                "draft_length": None,
                "accepted_token_count": None,
                "rejected_or_corrected_token_count": None,
                "generated_token_ids_up_to_round": (
                    gen_out.generation_output_token_ids[:k] if k > 0 else []
                ),
                "_fallback_input_ids": input_ids,
            })
        blockers = ladder_blockers

    round_cells: list[dict[str, Any]] = []
    for entry in entries:
        fallback_ids = entry.pop("_fallback_input_ids", None)
        if fallback_ids is not None:
            input_ids = fallback_ids
            seq_blockers: list[str] = []
        else:
            input_ids, seq_blockers = build_round_boundary_input_ids(gen_out, entry)
        if input_ids is None:
            round_cells.append({
                "round_index": entry.get("round_index"),
                "prefix_length_before_round": entry.get("prefix_length_before_round"),
                "prefix_length_after_round": entry.get("prefix_length_after_round"),
                "draft_length": entry.get("draft_length"),
                "accepted_token_count": entry.get("accepted_token_count"),
                "rejected_or_corrected_token_count": entry.get(
                    "rejected_or_corrected_token_count",
                ),
                "shadow_sequence_length": 0,
                "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
                "tolerance_policy_status": "blocked",
                "streaming_vs_materialized_metrics": None,
                "full_vs_streaming_metrics": None,
                "topk_agreement_metrics": None,
                "interpretation_note": "Could not build round-boundary sequence.",
                "blockers": seq_blockers,
            })
            continue
        round_cells.append(
            _shadow_round_cell(
                entry=entry,
                input_ids=input_ids,
                prompt_id=prompt_id,
                hf_model=hf_model,
                shadow_replay_fn=shadow_replay_fn,
                chunk_size=chunk_size,
                accumulator_mode=accumulator_mode,
                allow_parity_fail=allow_parity_fail,
                allow_shadow_fail=allow_shadow_fail,
            )
        )
    return round_cells, had_round_log, used_fallback, blockers


def _aggregate_tolerance_by_round(
    round_cells: Sequence[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    by_round: dict[str, dict[str, int]] = {}
    for cell in round_cells:
        rnd = str(cell.get("round_index", ""))
        status = cell.get("tolerance_policy_status")
        if status is None:
            continue
        by_round.setdefault(rnd, {})
        by_round[rnd][status] = by_round[rnd].get(status, 0) + 1
    return by_round


def _aggregate_topk_by_round(
    round_cells: Sequence[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    by_round: dict[str, dict[str, int]] = {}
    for cell in round_cells:
        rnd = str(cell.get("round_index", ""))
        topk = cell.get("topk_agreement_metrics") or {}
        by_round.setdefault(
            rnd, {"top1_agreement_true": 0, "top1_agreement_false": 0, "top1_agreement_unknown": 0},
        )
        agree = topk.get("top1_agreement")
        if agree is True:
            by_round[rnd]["top1_agreement_true"] += 1
        elif agree is False:
            by_round[rnd]["top1_agreement_false"] += 1
        else:
            by_round[rnd]["top1_agreement_unknown"] += 1
    return by_round


def _first_status_change_for_round_cells(
    round_cells: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not round_cells:
        return None
    base = round_cells[0].get("tolerance_policy_status")
    for cell in round_cells[1:]:
        status = cell.get("tolerance_policy_status")
        if status != base:
            return {
                "first_change_at_round_index": cell.get("round_index"),
                "from_status": base,
                "to_status": status,
            }
    return None


def _first_top1_mismatch_for_round_cells(
    round_cells: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    for cell in round_cells:
        topk = cell.get("topk_agreement_metrics") or {}
        if topk.get("top1_agreement") is False:
            return {
                "first_mismatch_at_round_index": cell.get("round_index"),
                "accepted_token_count": cell.get("accepted_token_count"),
                "prefix_length_after_round": cell.get("prefix_length_after_round"),
            }
    return None


def _aggregate_first_status_changes_by_round(
    generation_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    stable = 0
    for gc in generation_cells:
        rcs = gc.get("round_shadow_cells") or []
        change = _first_status_change_for_round_cells(rcs)
        if change is None:
            stable += 1
        else:
            changes.append({
                "prompt_id": gc.get("prompt_id"),
                "compressor": gc.get("compressor"),
                **change,
            })
    return {
        "cells_with_status_change": len(changes),
        "cells_all_stable": stable,
        "changes": changes,
    }


def _aggregate_first_top1_mismatches_by_round(
    generation_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    all_agree = 0
    for gc in generation_cells:
        rcs = gc.get("round_shadow_cells") or []
        mismatch = _first_top1_mismatch_for_round_cells(rcs)
        if mismatch is None:
            all_agree += 1
        else:
            mismatches.append({
                "prompt_id": gc.get("prompt_id"),
                "compressor": gc.get("compressor"),
                **mismatch,
            })
    return {
        "cells_with_top1_mismatch": len(mismatches),
        "cells_all_top1_agree": all_agree,
        "mismatches": mismatches,
    }


def _aggregate_accepted_prefix_correlation_summary(
    generation_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Descriptive correlation fields only; no causality claims."""
    mismatch_rounds: list[dict[str, Any]] = []
    partial_acceptance_rounds: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []

    for gc in generation_cells:
        for rc in gc.get("round_shadow_cells") or []:
            accepted = rc.get("accepted_token_count")
            draft_len = rc.get("draft_length")
            topk = rc.get("topk_agreement_metrics") or {}
            mismatch = topk.get("top1_agreement") is False
            round_idx = rc.get("round_index")

            if mismatch:
                mismatch_rounds.append({
                    "prompt_id": gc.get("prompt_id"),
                    "compressor": gc.get("compressor"),
                    "round_index": round_idx,
                    "accepted_token_count": accepted,
                    "prefix_length_after_round": rc.get("prefix_length_after_round"),
                })

            partial = (
                accepted is not None
                and draft_len is not None
                and int(accepted) < int(draft_len)
            )
            if partial:
                partial_acceptance_rounds.append({
                    "prompt_id": gc.get("prompt_id"),
                    "compressor": gc.get("compressor"),
                    "round_index": round_idx,
                    "accepted_token_count": accepted,
                    "draft_length": draft_len,
                })
                if mismatch:
                    overlap.append({
                        "prompt_id": gc.get("prompt_id"),
                        "compressor": gc.get("compressor"),
                        "round_index": round_idx,
                        "accepted_token_count": accepted,
                        "draft_length": draft_len,
                    })

    return {
        "description": (
            "Descriptive overlap between top-1 mismatch rounds and partial-acceptance "
            "rounds; not a causality claim."
        ),
        "mismatch_round_count": len(mismatch_rounds),
        "partial_acceptance_round_count": len(partial_acceptance_rounds),
        "overlap_mismatch_and_partial_acceptance_count": len(overlap),
        "mismatch_rounds": mismatch_rounds,
        "partial_acceptance_rounds": partial_acceptance_rounds,
        "overlap_rounds": overlap,
    }


def run_exp080_round_log_shadow_panel(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_new_tokens: int = DEFAULT_EXP080_MAX_NEW_TOKENS,
    compressors_requested: Sequence[str] = DEFAULT_EXP080_COMPRESSORS,
    draft_len: int = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    accumulator_mode: str = "float32",
    fallback_prefix_ladder: bool = False,
    ladder_stride: int = 1,
    allow_shadow_fail: bool = True,
    allow_parity_fail: bool = True,
    local_files_only: bool = False,
    generation_fn: Callable[..., GenerationOutput] | None = None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 080 post-hoc ExactKV round-log shadow observer."""
    prompt_panel = list(prompts) if prompts is not None else default_exp080_prompts()
    runnable_compressors, _blocked_compressors = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if not runnable_compressors:
        blockers.append("no compressors runnable via get_compressor API")

    runtime: Any | None = None
    if generation_fn is None:
        if runtime_loader is not None:
            try:
                runtime, _tokenizer = runtime_loader(
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")
        else:
            try:
                from exactkv.runtime.model_runtime import ModelRuntime

                runtime = ModelRuntime(model_id, device=device, dtype=dtype)
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    hf_model = getattr(runtime, "model", None) if runtime is not None else None
    baseline_cache: dict[str, list[int]] = {}

    generation_cells: list[dict[str, Any]] = []
    all_round_cells: list[dict[str, Any]] = []
    gen_success = 0
    gen_blocked = 0
    round_success = 0
    round_blocked = 0
    round_log_available = 0
    round_log_missing = 0
    fallback_used = 0
    max_rounds = 0

    for prompt_id, prompt_text in prompt_panel:
        preview = _preview(prompt_text)
        for compressor in runnable_compressors:
            cell_blockers: list[str] = []
            if runtime is None and generation_fn is None:
                gen_blocked += 1
                generation_cells.append({
                    "prompt_id": prompt_id,
                    "prompt_preview": preview,
                    "compressor": compressor,
                    "max_new_tokens": max_new_tokens,
                    "generation_completed": False,
                    "exactkv_failures": None,
                    "token_exact_match": None,
                    "generation_output_preview": "",
                    "generation_output_token_count": 0,
                    "round_log_available": False,
                    "round_count": 0,
                    "fallback_prefix_ladder_used": False,
                    "round_shadow_cells": [],
                    "safety_gates": _generation_safety_gates(
                        generation_completed=False, shadow_ran=False,
                    ),
                    "blockers": list(blockers),
                })
                continue

            if generation_fn is not None:
                gen_out = generation_fn(
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                )
            else:
                full_ids = baseline_cache.get(prompt_id)
                gen_out = run_exactkv_generation_with_baseline(
                    runtime=runtime,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    compressor_name=compressor,
                    draft_len=draft_len,
                    full_baseline_ids=full_ids,
                )
                if full_ids is None and gen_out.generation_completed:
                    try:
                        from exactkv.runtime.generation import generate_full_greedy

                        full_res = generate_full_greedy(runtime, prompt_text, max_new_tokens)
                        full_ids = full_res.generated_ids.squeeze().tolist()
                        if isinstance(full_ids, int):
                            full_ids = [full_ids]
                        baseline_cache[prompt_id] = full_ids
                    except Exception:  # noqa: BLE001
                        pass

            generation_completed = bool(gen_out.generation_completed)
            if generation_completed:
                gen_success += 1
            else:
                gen_blocked += 1
                cell_blockers.extend(gen_out.blockers)

            round_cells: list[dict[str, Any]] = []
            log_available = False
            used_fallback = False
            shadow_ran = False

            if generation_completed:
                round_cells, log_available, used_fallback, extract_blockers = (
                    run_round_log_shadow_for_generation(
                        gen_out=gen_out,
                        prompt_id=prompt_id,
                        hf_model=hf_model,
                        shadow_replay_fn=shadow_replay_fn,
                        chunk_size=chunk_size,
                        accumulator_mode=accumulator_mode,
                        allow_parity_fail=allow_parity_fail,
                        allow_shadow_fail=allow_shadow_fail,
                        fallback_prefix_ladder=fallback_prefix_ladder,
                        ladder_stride=ladder_stride,
                    )
                )
                cell_blockers.extend(extract_blockers)
                shadow_ran = bool(round_cells)
                if log_available:
                    round_log_available += 1
                elif not used_fallback:
                    round_log_missing += 1
                    cell_blockers.append(BLOCKED_MISSING_ROUND_LOG)
                if used_fallback:
                    fallback_used += 1

            round_count = len(round_cells)
            if round_count > max_rounds:
                max_rounds = round_count

            for rc in round_cells:
                all_round_cells.append(rc)
                if rc.get("shadow_status") == GenerationShadowStatus.SHADOW_COMPLETE.value:
                    round_success += 1
                else:
                    round_blocked += 1

            safety = _generation_safety_gates(
                generation_completed=generation_completed,
                shadow_ran=shadow_ran,
            )

            generation_cells.append({
                "prompt_id": prompt_id,
                "prompt_preview": preview,
                "compressor": compressor,
                "max_new_tokens": max_new_tokens,
                "generation_completed": generation_completed,
                "exactkv_failures": gen_out.exactkv_failures,
                "token_exact_match": gen_out.token_exact_match,
                "generation_output_preview": _preview(gen_out.generation_output_text),
                "generation_output_token_count": len(gen_out.generation_output_token_ids or []),
                "round_log_available": log_available,
                "round_count": round_count,
                "fallback_prefix_ladder_used": used_fallback,
                "round_shadow_cells": round_cells,
                "safety_gates": safety,
                "blockers": cell_blockers,
            })

    total_round = round_success + round_blocked
    safety_ok = all(
        gc.get("safety_gates", {}).get("shadow_used_for_token_commit") is False
        and gc.get("safety_gates", {}).get("generation_modified_by_shadow") is False
        and gc.get("safety_gates", {}).get("default_runtime_changed") is False
        for gc in generation_cells
    )

    if not safety_ok:
        status = "failed"
    elif gen_success == 0:
        status = "blocked"
    elif round_success == total_round and total_round > 0:
        status = "diagnostic_complete"
    elif round_success > 0:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_080_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "fallback_prefix_ladder": fallback_prefix_ladder,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable_compressors,
        "total_generation_cells": len(generation_cells),
        "generation_successful_cells": gen_success,
        "generation_blocked_cells": gen_blocked,
        "total_round_shadow_cells": total_round,
        "round_shadow_successful_cells": round_success,
        "round_shadow_blocked_cells": round_blocked,
        "round_log_available_cells": round_log_available,
        "round_log_missing_cells": round_log_missing,
        "fallback_prefix_ladder_used_cells": fallback_used,
        "max_rounds_observed": max_rounds,
        "exactkv_failure_summary": _summarize_exactkv_failures(generation_cells),
        "tolerance_policy_summary_by_round": _aggregate_tolerance_by_round(all_round_cells),
        "topk_agreement_summary_by_round": _aggregate_topk_by_round(all_round_cells),
        "first_status_change_summary": _aggregate_first_status_changes_by_round(generation_cells),
        "first_top1_mismatch_summary": _aggregate_first_top1_mismatches_by_round(generation_cells),
        "accepted_prefix_correlation_summary": _aggregate_accepted_prefix_correlation_summary(
            generation_cells,
        ),
        "generation_modified_by_shadow": False,
        "shadow_used_for_token_commit": False,
        "default_runtime_changed": False,
        "generation_cells": generation_cells,
        "blockers": blockers,
        "limitations": [
            "Post-hoc ExactKV round-log observer; not live decode integration.",
            "Uses existing ExactKVResult traces when available.",
            "Round-boundary replay is fixed-sequence analysis, not token generation.",
            "Top-k agreement is supplementary; not exactness.",
            "No live per-round decode hooks.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP080_CLAIM_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "cli_flag": PROPOSED_SHADOW_CLI_FLAG,
    }


def validate_exp080_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "max_new_tokens",
        "compressors_requested",
        "compressors_run",
        "total_generation_cells",
        "generation_successful_cells",
        "generation_blocked_cells",
        "total_round_shadow_cells",
        "round_shadow_successful_cells",
        "round_shadow_blocked_cells",
        "round_log_available_cells",
        "round_log_missing_cells",
        "fallback_prefix_ladder_used_cells",
        "max_rounds_observed",
        "exactkv_failure_summary",
        "tolerance_policy_summary_by_round",
        "topk_agreement_summary_by_round",
        "first_status_change_summary",
        "first_top1_mismatch_summary",
        "accepted_prefix_correlation_summary",
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
        "generation_cells",
        "blockers",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_080_ID:
        errors.append("experiment_id mismatch")

    for flag in (
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    for idx, cell in enumerate(report.get("generation_cells", [])):
        if not isinstance(cell, dict):
            errors.append(f"generation_cells[{idx}] not dict")
            continue
        for ck in (
            "prompt_id",
            "compressor",
            "max_new_tokens",
            "generation_completed",
            "round_log_available",
            "round_count",
            "round_shadow_cells",
            "safety_gates",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"generation_cells[{idx}] missing {ck}")
        gates = cell.get("safety_gates") or {}
        if gates.get("shadow_used_for_token_commit") is not False:
            errors.append(
                f"generation_cells[{idx}].safety_gates.shadow_used_for_token_commit must be false",
            )
        if gates.get("generation_modified_by_shadow") is not False:
            errors.append(
                f"generation_cells[{idx}].safety_gates.generation_modified_by_shadow must be false",
            )
        if gates.get("default_runtime_changed") is not False:
            errors.append(
                f"generation_cells[{idx}].safety_gates.default_runtime_changed must be false",
            )

    return errors
