"""L4 no-op opt-in scaffold (Phase 21A / Exp 102).

Stage 1 scaffolding only — records opt-in trace metadata externally.
Must not affect token commits or generator decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.attention.decode_time_shadow_observer import (
    _token_lists_match,
    default_exp083_prompts,
)
from exactkv.attention.live_round_observer import _run_baseline_generation
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    L4_OPT_IN_FLAG,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_102_ID = "exp102_l4_noop_opt_in_scaffold"
DEFAULT_EXP102_REPORT = Path(
    "reports/experiment_102_l4_noop_opt_in_scaffold.json",
)
PHASE_21A = "21A"
STAGE = "stage_1_noop_opt_in_scaffold"
MODE = "noop_trace_only"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21A = "phase21b_l4_noop_scaffold_panel_validation"
FORBIDDEN_NEXT_PHASE_21A = "l4_runtime_commit_implementation"

EXPERIMENT_103_ID = "exp103_l4_noop_scaffold_panel_validation"
DEFAULT_EXP103_REPORT = Path(
    "reports/experiment_103_l4_noop_scaffold_panel_validation.json",
)
PHASE_21B = "21B"
RECOMMENDED_NEXT_PHASE_21B = "phase21c_l4_trace_only_dry_run_design"
FORBIDDEN_NEXT_PHASE_21B = "l4_runtime_commit_implementation"

DEFAULT_PANEL_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
DEFAULT_PANEL_COMPRESSORS: tuple[str, ...] = (
    "noop",
    "int8",
    "int4_sim",
    "k8_v4_sim",
)
DEFAULT_MAX_NEW_TOKENS_VALUES: tuple[int, ...] = (4, 8)
DEFAULT_PANEL_PROMPTS = 4

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_COMPRESSORS: tuple[str, ...] = ("noop", "int8")
DEFAULT_MAX_NEW_TOKENS = 4
DEFAULT_MAX_PROMPTS = 2

EXPERIMENTAL_WARNING = (
    "Experimental L4 no-op scaffold: trace metadata only; "
    "no verifier-mediated acceptance; no performance or serving claim."
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

SCAFFOLD_RESOLVED_BLOCKER_IDS: frozenset[str] = frozenset(
    {
        "explicit_l4_design_spec",
        "verifier_mediated_acceptance_contract",
        "rollback_behavior_defined",
        "l4_test_matrix_defined",
        "l4_opt_in_flag_designed",
        "l4_synthetic_contract_tests_no_runtime",
        "exactkv_generator_integration_plan",
        "stage_1_noop_scaffold_design",
        "stage_1_noop_scaffold_panel_validation",
    },
)


@dataclass(frozen=True)
class L4NoopOptInConfig:
    """L4 no-op opt-in configuration (scaffold only)."""

    enabled: bool = False
    flag_name: str = L4_OPT_IN_FLAG
    experimental_warning: str = EXPERIMENTAL_WARNING
    stage: str = STAGE
    mode: str = MODE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4NoopSafetyGates:
    """Safety invariants for L4 no-op scaffold cells and reports."""

    l4_noop_scaffold_enabled: bool
    l4_runtime_commit_implemented: bool = False
    verifier_mediated_acceptance_performed: bool = False
    proposal_used_for_token_commit: bool = False
    proposal_exposed_to_generator: bool = False
    rollback_runtime_implemented: bool = False
    fallback_runtime_implemented: bool = False
    default_runtime_changed: bool = False
    generation_logic_changed: bool = False
    exactkv_generator_modified: bool = False
    production_cli_modified: bool = False
    research_script_flag_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4NoopScaffoldTrace:
    """Trace metadata recorded by no-op scaffold (no runtime commit effect)."""

    l4_noop_scaffold_enabled: bool
    stage: str
    mode: str
    flag_name: str
    experimental_warning: str
    decision_steps: tuple[str, ...]
    trace_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4NoopCellResult:
    """Result for one baseline vs no-op scaffold cell."""

    prompt_id: str
    prompt_preview: str
    compressor: str
    baseline_completed: bool
    noop_scaffold_completed: bool
    token_match: bool
    text_match: bool
    baseline_token_ids: tuple[int, ...]
    noop_scaffold_token_ids: tuple[int, ...]
    safety_gates: L4NoopSafetyGates
    scaffold_trace: L4NoopScaffoldTrace
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_preview": self.prompt_preview,
            "compressor": self.compressor,
            "baseline_completed": self.baseline_completed,
            "noop_scaffold_completed": self.noop_scaffold_completed,
            "token_match": self.token_match,
            "text_match": self.text_match,
            "baseline_token_ids": list(self.baseline_token_ids),
            "noop_scaffold_token_ids": list(self.noop_scaffold_token_ids),
            "safety_gates": self.safety_gates.to_dict(),
            "scaffold_trace": self.scaffold_trace.to_dict(),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class L4NoopScaffoldValidationResult:
    """Validation outcome for an L4 no-op scaffold report."""

    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4NoopScaffoldReport:
    """Top-level L4 no-op scaffold report aggregate."""

    experiment_id: str
    status: str
    stage: str
    mode: str
    config: L4NoopOptInConfig
    cells: tuple[L4NoopCellResult, ...]
    validation_result: L4NoopScaffoldValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "stage": self.stage,
            "mode": self.mode,
            "config": self.config.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "validation_result": self.validation_result.to_dict(),
        }


def default_l4_noop_opt_in_config(*, enabled: bool = False) -> L4NoopOptInConfig:
    """Default L4 no-op opt-in config (disabled unless explicitly enabled)."""
    return L4NoopOptInConfig(enabled=enabled)


def default_l4_noop_safety_gates(*, scaffold_enabled: bool) -> L4NoopSafetyGates:
    """Build default safety gates for no-op scaffold."""
    return L4NoopSafetyGates(l4_noop_scaffold_enabled=scaffold_enabled)


def build_l4_noop_scaffold_trace(config: L4NoopOptInConfig) -> L4NoopScaffoldTrace:
    """Build no-op scaffold trace metadata (no generation side effects)."""
    steps: tuple[str, ...]
    if config.enabled:
        steps = (
            "noop_scaffold_opt_in_recorded",
            "baseline_generation_invoked_unchanged",
            "no_verifier_mediated_acceptance",
            "no_proposal_commit",
            "trace_metadata_attached",
        )
    else:
        steps = (
            "noop_scaffold_disabled",
            "baseline_generation_only",
        )
    return L4NoopScaffoldTrace(
        l4_noop_scaffold_enabled=config.enabled,
        stage=config.stage,
        mode=config.mode,
        flag_name=config.flag_name,
        experimental_warning=config.experimental_warning,
        decision_steps=steps,
        trace_complete=True,
    )


def run_baseline_generation_external(
    *,
    runtime: Any,
    prompt: str,
    max_new_tokens: int,
    compressor_name: str,
    draft_len: int = 4,
) -> dict[str, Any]:
    """Run baseline generation without L4 scaffold (unchanged path)."""
    return _run_baseline_generation(
        runtime, prompt, max_new_tokens, compressor_name, draft_len,
    )


def run_noop_scaffold_generation_external(
    *,
    runtime: Any,
    prompt: str,
    max_new_tokens: int,
    compressor_name: str,
    config: L4NoopOptInConfig,
    draft_len: int = 4,
) -> tuple[dict[str, Any], L4NoopScaffoldTrace]:
    """Run generation with external no-op L4 scaffold wrapper (no token effect).

    The scaffold records opt-in trace metadata only. Generation uses the
    unchanged baseline path; no L4 interfaces are invoked at runtime.
    """
    trace = build_l4_noop_scaffold_trace(config)
    result = _run_baseline_generation(
        runtime, prompt, max_new_tokens, compressor_name, draft_len,
    )
    return result, trace


def _cell_token_ids(gen_out: dict[str, Any]) -> tuple[int, ...]:
    ids = gen_out.get("generated_token_ids") or []
    return tuple(ids) if isinstance(ids, list) else tuple()


def _build_noop_cell(
    *,
    prompt_id: str,
    prompt_text: str,
    compressor: str,
    max_new_tokens: int,
    config: L4NoopOptInConfig,
    runtime: Any | None,
    draft_len: int,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None,
    noop_scaffold_generation_fn: Callable[..., tuple[dict[str, Any], L4NoopScaffoldTrace]]
    | None,
) -> L4NoopCellResult:
    """Build one baseline vs no-op scaffold comparison cell."""
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
        baseline = run_baseline_generation_external(
            runtime=runtime,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            draft_len=draft_len,
        )
    else:
        baseline = {"generation_completed": False, "blockers": ["no runtime"]}

    if noop_scaffold_generation_fn is not None:
        noop_out, trace = noop_scaffold_generation_fn(
            prompt=prompt_text,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
        )
    elif runtime is not None:
        noop_out, trace = run_noop_scaffold_generation_external(
            runtime=runtime,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            config=config,
            draft_len=draft_len,
        )
    else:
        noop_out = {"generation_completed": False, "blockers": ["no runtime"]}
        trace = build_l4_noop_scaffold_trace(config)

    baseline_ok = bool(baseline.get("generation_completed"))
    noop_ok = bool(noop_out.get("generation_completed"))
    baseline_ids = _cell_token_ids(baseline)
    noop_ids = _cell_token_ids(noop_out)
    tok_match = _token_lists_match(baseline_ids, noop_ids)
    txt_match = baseline.get("generated_text") == noop_out.get("generated_text")

    if baseline_ok and noop_ok and not tok_match:
        blockers.append("baseline_vs_noop_scaffold_token_mismatch")
    if baseline_ok and noop_ok and not txt_match:
        blockers.append("baseline_vs_noop_scaffold_text_mismatch")

    gates = default_l4_noop_safety_gates(scaffold_enabled=config.enabled)

    return L4NoopCellResult(
        prompt_id=prompt_id,
        prompt_preview=preview,
        compressor=compressor,
        baseline_completed=baseline_ok,
        noop_scaffold_completed=noop_ok,
        token_match=tok_match,
        text_match=txt_match,
        baseline_token_ids=baseline_ids,
        noop_scaffold_token_ids=noop_ids,
        safety_gates=gates,
        scaffold_trace=trace,
        blockers=tuple(blockers),
    )


def _aggregate_safety_gate_summary(cells: Sequence[L4NoopCellResult]) -> dict[str, Any]:
    ok = 0
    for cell in cells:
        sg = cell.safety_gates
        if (
            not sg.l4_runtime_commit_implemented
            and not sg.verifier_mediated_acceptance_performed
            and not sg.proposal_used_for_token_commit
            and not sg.proposal_exposed_to_generator
            and not sg.default_runtime_changed
            and not sg.generation_logic_changed
            and not sg.production_cli_modified
        ):
            ok += 1
    total = len(cells)
    return {
        "cells_all_gates_ok": ok,
        "cells_with_gate_failure": total - ok,
        "total_cells": total,
    }


def validate_l4_noop_scaffold_report(report: dict[str, Any]) -> L4NoopScaffoldValidationResult:
    """Validate L4 no-op scaffold report safety invariants."""
    errors: list[str] = []

    required_top = (
        "experiment_id",
        "status",
        "stage",
        "mode",
        "config",
        "research_script_flag_used",
        "production_cli_modified",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "l4_runtime_commit_implemented",
        "verifier_mediated_acceptance_performed",
        "rollback_runtime_implemented",
        "fallback_runtime_implemented",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "cells",
        "validation_result",
        "allowed_next_phase",
        "forbidden_next_phase",
        "limitations",
    )
    for key in required_top:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_102_ID:
        errors.append("experiment_id mismatch")

    bool_must_be_false = (
        "l4_runtime_commit_implemented",
        "verifier_mediated_acceptance_performed",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "rollback_runtime_implemented",
        "fallback_runtime_implemented",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
    )
    for key in bool_must_be_false:
        if report.get(key) is not False:
            errors.append(f"{key} must be false")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21A:
        errors.append("allowed_next_phase must be phase21b_l4_noop_scaffold_panel_validation")

    if report.get("forbidden_next_phase") != FORBIDDEN_NEXT_PHASE_21A:
        errors.append("forbidden_next_phase must be l4_runtime_commit_implementation")

    cells = report.get("cells") or []
    for idx, cell in enumerate(cells):
        if not cell.get("baseline_completed") or not cell.get("noop_scaffold_completed"):
            continue
        if cell.get("token_match") is not True:
            errors.append(f"cells[{idx}] token_match failed for completed cell")
        if cell.get("text_match") is not True:
            errors.append(f"cells[{idx}] text_match failed for completed cell")
        sg = cell.get("safety_gates") or {}
        for key in bool_must_be_false:
            if sg.get(key) is True:
                errors.append(f"cells[{idx}].safety_gates.{key} must be false")

    return L4NoopScaffoldValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
    )


def validate_exp102_report(report: dict[str, Any]) -> list[str]:
    """Return validation error strings for Experiment 102 report."""
    return list(validate_l4_noop_scaffold_report(report).errors)


def _remaining_implementation_blockers() -> list[dict[str, str]]:
    mapping = {
        "explicit L4 design spec missing": "explicit_l4_design_spec",
        "ExactKVGenerator integration plan missing": "exactkv_generator_integration_plan",
        "fallback path not yet implemented for L4": "l4_fallback_path",
        "L4 opt-in flag not yet designed": "l4_opt_in_flag_designed",
        "verifier-mediated acceptance contract not yet defined": (
            "verifier_mediated_acceptance_contract"
        ),
        "rollback behavior not yet defined": "rollback_behavior_defined",
        "L4 test matrix not yet defined": "l4_test_matrix_defined",
        "no L4 baseline-vs-integrated parity panel": "l4_parity_panel",
        "no L4 exactkv_failures gate run": "l4_exactkv_failures_gate_run",
        "no active GPU memory measurement": "gpu_memory_measurement",
        "no performance benchmark": "performance_benchmark",
        "no serving integration": "serving_integration",
    }
    remaining: list[dict[str, str]] = []
    for text in L4_IMPLEMENTATION_BLOCKERS:
        bid = mapping.get(text, text.replace(" ", "_").lower()[:40])
        if bid not in SCAFFOLD_RESOLVED_BLOCKER_IDS:
            remaining.append({"blocker_id": bid, "description": text})
    for extra in (
        ("l4_runtime_fallback_implementation", "runtime L4 fallback path not implemented"),
        ("l4_runtime_rollback_implementation", "runtime L4 rollback path not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
        ("stage_2_trace_dry_run", "stage 2 trace-only L4 dry-run not implemented"),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("production_cli_l4_flag", "production CLI L4 flag not implemented"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def build_synthetic_exp102_report() -> dict[str, Any]:
    """Build a safe synthetic report for unit tests (no model)."""
    config = default_l4_noop_opt_in_config(enabled=True)
    trace = build_l4_noop_scaffold_trace(config)
    gates = default_l4_noop_safety_gates(scaffold_enabled=True)
    cell = L4NoopCellResult(
        prompt_id="p0",
        prompt_preview="hello",
        compressor="noop",
        baseline_completed=True,
        noop_scaffold_completed=True,
        token_match=True,
        text_match=True,
        baseline_token_ids=(100, 101, 102, 103),
        noop_scaffold_token_ids=(100, 101, 102, 103),
        safety_gates=gates,
        scaffold_trace=trace,
        blockers=(),
    )
    report = _report_from_cells(
        cells=(cell,),
        config=config,
        research_script_flag_used=True,
        model_id="synthetic",
        device="cpu",
        dtype="float32",
        compressors_requested=["noop"],
        compressors_run=["noop"],
        status="scaffold_complete",
        blocked_cells=0,
    )
    validation = validate_l4_noop_scaffold_report(report)
    report["validation_result"] = validation.to_dict()
    return report


def _report_from_cells(
    *,
    cells: Sequence[L4NoopCellResult],
    config: L4NoopOptInConfig,
    research_script_flag_used: bool,
    model_id: str,
    device: str,
    dtype: str,
    compressors_requested: Sequence[str],
    compressors_run: Sequence[str],
    status: str,
    blocked_cells: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    baseline_ok = sum(1 for c in cells if c.baseline_completed)
    noop_ok = sum(1 for c in cells if c.noop_scaffold_completed)
    tok_match = sum(1 for c in cells if c.token_match)
    txt_match = sum(1 for c in cells if c.text_match)
    total = len(cells)

    return {
        "experiment_id": EXPERIMENT_102_ID,
        "status": status,
        "phase": PHASE_21A,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "config": config.to_dict(),
        "research_script_flag_used": research_script_flag_used,
        "production_cli_modified": False,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "l4_runtime_commit_implemented": False,
        "verifier_mediated_acceptance_performed": False,
        "rollback_runtime_implemented": False,
        "fallback_runtime_implemented": False,
        "proposal_used_for_token_commit": False,
        "proposal_exposed_to_generator": False,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "compressors_requested": list(compressors_requested),
        "compressors_run": list(compressors_run),
        "total_cells": total,
        "successful_baseline_cells": baseline_ok,
        "successful_noop_scaffold_cells": noop_ok,
        "token_match_cells": tok_match,
        "text_match_cells": txt_match,
        "blocked_cells": blocked_cells,
        "safety_gate_summary": _aggregate_safety_gate_summary(cells),
        "cells": [c.to_dict() for c in cells],
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21A,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_21A,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 no-op opt-in scaffold only; not L4 runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Research-script flag only; production CLI unchanged.",
            "No verifier-mediated acceptance; no proposal commit.",
            "No model-output preservation claim generally.",
        ],
    }


def run_exp102_l4_noop_opt_in_scaffold(
    *,
    l4_noop_enabled: bool = True,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = DEFAULT_MAX_PROMPTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    compressors_requested: Sequence[str] = DEFAULT_COMPRESSORS,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_model_blocked: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    noop_scaffold_generation_fn: Callable[
        ..., tuple[dict[str, Any], L4NoopScaffoldTrace]
    ] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run Experiment 102 L4 no-op opt-in scaffold panel."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    config = default_l4_noop_opt_in_config(enabled=l4_noop_enabled)
    prompt_panel = (
        list(prompts) if prompts is not None else default_exp083_prompts()[:max_prompts]
    )
    runnable, _blocked_comp = resolve_panel_compressors(compressors_requested)

    runtime: Any | None = None
    load_blockers: list[str] = []
    if baseline_generation_fn is None or noop_scaffold_generation_fn is None:
        try:
            if runtime_loader is not None:
                runtime = runtime_loader(
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            else:
                from exactkv.runtime.model_runtime import ModelRuntime

                runtime = ModelRuntime(model_id, device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            load_blockers.append(f"model load failed: {type(exc).__name__}: {exc}")

    cells: list[L4NoopCellResult] = []
    blocked_cells = 0

    if runtime is None and (
        baseline_generation_fn is None or noop_scaffold_generation_fn is None
    ):
        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                trace = build_l4_noop_scaffold_trace(config)
                gates = default_l4_noop_safety_gates(scaffold_enabled=config.enabled)
                cells.append(
                    L4NoopCellResult(
                        prompt_id=prompt_id,
                        prompt_preview=prompt_text[:80],
                        compressor=compressor,
                        baseline_completed=False,
                        noop_scaffold_completed=False,
                        token_match=False,
                        text_match=False,
                        baseline_token_ids=(),
                        noop_scaffold_token_ids=(),
                        safety_gates=gates,
                        scaffold_trace=trace,
                        blockers=tuple(load_blockers),
                    ),
                )
                blocked_cells += 1
        status = "blocked" if allow_model_blocked else "failed"
    else:
        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                cell = _build_noop_cell(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    compressor=compressor,
                    max_new_tokens=max_new_tokens,
                    config=config,
                    runtime=runtime,
                    draft_len=draft_len,
                    baseline_generation_fn=baseline_generation_fn,
                    noop_scaffold_generation_fn=noop_scaffold_generation_fn,
                )
                cells.append(cell)
                if cell.blockers:
                    blocked_cells += 1

        total = len(cells)
        baseline_ok = sum(1 for c in cells if c.baseline_completed)
        tok_match = sum(1 for c in cells if c.token_match)
        if total == 0:
            status = "blocked"
        elif baseline_ok == total and tok_match == total:
            status = "scaffold_complete"
        elif baseline_ok > 0:
            status = "scaffold_partial"
        else:
            status = "failed"

    report = _report_from_cells(
        cells=tuple(cells),
        config=config,
        research_script_flag_used=l4_noop_enabled,
        model_id=model_id,
        device=device,
        dtype=dtype,
        compressors_requested=compressors_requested,
        compressors_run=runnable,
        status=status,
        blocked_cells=blocked_cells,
        max_new_tokens=max_new_tokens,
    )
    validation = validate_l4_noop_scaffold_report(report)
    report["validation_result"] = validation.to_dict()
    return report


def default_noop_panel_prompts(max_prompts: int = DEFAULT_PANEL_PROMPTS) -> list[tuple[str, str]]:
    """Deterministic prompts for L4 no-op scaffold panel validation."""
    return default_exp083_prompts()[:max_prompts]


def _cell_safety_gates_ok(gates: L4NoopSafetyGates | Mapping[str, Any]) -> bool:
    """Return True when no-op safety gates pass for a cell."""
    if isinstance(gates, L4NoopSafetyGates):
        d = gates.to_dict()
    else:
        d = gates
    return (
        d.get("l4_noop_scaffold_enabled") is True
        and d.get("l4_runtime_commit_implemented") is False
        and d.get("verifier_mediated_acceptance_performed") is False
        and d.get("proposal_used_for_token_commit") is False
        and d.get("proposal_exposed_to_generator") is False
        and d.get("rollback_runtime_implemented") is False
        and d.get("fallback_runtime_implemented") is False
        and d.get("default_runtime_changed") is False
        and d.get("generation_logic_changed") is False
        and d.get("production_cli_modified") is False
        and d.get("research_script_flag_only") is True
    )


def _panel_cell_to_dict(
    *,
    model_id: str,
    prompt_id: str,
    prompt_preview: str,
    compressor: str,
    max_new_tokens: int,
    cell: L4NoopCellResult,
    exactkv_failures_baseline: int | None,
    exactkv_failures_noop_scaffold: int | None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "prompt_preview": prompt_preview,
        "compressor": compressor,
        "max_new_tokens": max_new_tokens,
        "baseline_completed": cell.baseline_completed,
        "noop_scaffold_completed": cell.noop_scaffold_completed,
        "token_match": cell.token_match,
        "text_match": cell.text_match,
        "exactkv_failures_baseline": exactkv_failures_baseline,
        "exactkv_failures_noop_scaffold": exactkv_failures_noop_scaffold,
        "baseline_token_ids": list(cell.baseline_token_ids),
        "noop_scaffold_token_ids": list(cell.noop_scaffold_token_ids),
        "safety_gates": cell.safety_gates.to_dict(),
        "scaffold_trace": cell.scaffold_trace.to_dict(),
        "blockers": list(cell.blockers),
    }


def _build_noop_panel_cell(
    *,
    model_id: str,
    prompt_id: str,
    prompt_text: str,
    compressor: str,
    max_new_tokens: int,
    config: L4NoopOptInConfig,
    runtime: Any | None,
    draft_len: int,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None,
    noop_scaffold_generation_fn: Callable[..., tuple[dict[str, Any], L4NoopScaffoldTrace]]
    | None,
) -> tuple[dict[str, Any], bool]:
    """Build one panel cell dict and whether it failed validation."""
    preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
    blockers: list[str] = []

    if baseline_generation_fn is not None:
        baseline = baseline_generation_fn(
            prompt=prompt_text,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            model_id=model_id,
        )
    elif runtime is not None:
        baseline = run_baseline_generation_external(
            runtime=runtime,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            draft_len=draft_len,
        )
    else:
        baseline = {"generation_completed": False, "blockers": ["no runtime"]}

    if noop_scaffold_generation_fn is not None:
        noop_out, trace = noop_scaffold_generation_fn(
            prompt=prompt_text,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            model_id=model_id,
        )
    elif runtime is not None:
        noop_out, trace = run_noop_scaffold_generation_external(
            runtime=runtime,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            config=config,
            draft_len=draft_len,
        )
    else:
        noop_out = {"generation_completed": False, "blockers": ["no runtime"]}
        trace = build_l4_noop_scaffold_trace(config)

    baseline_ok = bool(baseline.get("generation_completed"))
    noop_ok = bool(noop_out.get("generation_completed"))
    baseline_ids = _cell_token_ids(baseline)
    noop_ids = _cell_token_ids(noop_out)
    tok_match = _token_lists_match(baseline_ids, noop_ids)
    txt_match = baseline.get("generated_text") == noop_out.get("generated_text")

    if baseline_ok and noop_ok and not tok_match:
        blockers.append("baseline_vs_noop_scaffold_token_mismatch")
    if baseline_ok and noop_ok and not txt_match:
        blockers.append("baseline_vs_noop_scaffold_text_mismatch")

    gates = default_l4_noop_safety_gates(scaffold_enabled=config.enabled)
    cell_result = L4NoopCellResult(
        prompt_id=prompt_id,
        prompt_preview=preview,
        compressor=compressor,
        baseline_completed=baseline_ok,
        noop_scaffold_completed=noop_ok,
        token_match=tok_match,
        text_match=txt_match,
        baseline_token_ids=baseline_ids,
        noop_scaffold_token_ids=noop_ids,
        safety_gates=gates,
        scaffold_trace=trace,
        blockers=tuple(blockers),
    )

    cell_dict = _panel_cell_to_dict(
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_preview=preview,
        compressor=compressor,
        max_new_tokens=max_new_tokens,
        cell=cell_result,
        exactkv_failures_baseline=baseline.get("exactkv_failures"),
        exactkv_failures_noop_scaffold=noop_out.get("exactkv_failures"),
    )

    failed = (
        not _cell_safety_gates_ok(gates)
        or (baseline_ok and noop_ok and (not tok_match or not txt_match))
    )
    if failed and not _cell_safety_gates_ok(gates):
        if "safety_gate_failed" not in blockers:
            blockers.append("safety_gate_failed")
        cell_dict["blockers"] = list(blockers)

    return cell_dict, failed


def _breakdown_metrics_from_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(cells)
    return {
        "total_cells": total,
        "successful_baseline_cells": sum(
            1 for c in cells if c.get("baseline_completed")
        ),
        "successful_noop_scaffold_cells": sum(
            1 for c in cells if c.get("noop_scaffold_completed")
        ),
        "token_match_cells": sum(1 for c in cells if c.get("token_match")),
        "text_match_cells": sum(1 for c in cells if c.get("text_match")),
        "exactkv_failures_baseline": sum(
            1 for c in cells if (c.get("exactkv_failures_baseline") or 0) > 0
        ),
        "exactkv_failures_noop_scaffold": sum(
            1 for c in cells if (c.get("exactkv_failures_noop_scaffold") or 0) > 0
        ),
    }


def aggregate_noop_panel_breakdowns(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Break down no-op panel metrics by model, compressor, prompt, max_new_tokens."""
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    by_compressor: dict[str, list[Mapping[str, Any]]] = {}
    by_prompt: dict[str, list[Mapping[str, Any]]] = {}
    by_max_new: dict[str, list[Mapping[str, Any]]] = {}

    for cell in cells:
        by_model.setdefault(str(cell.get("model_id", "unknown")), []).append(cell)
        by_compressor.setdefault(str(cell.get("compressor", "unknown")), []).append(cell)
        by_prompt.setdefault(str(cell.get("prompt_id", "unknown")), []).append(cell)
        by_max_new.setdefault(str(cell.get("max_new_tokens", "unknown")), []).append(cell)

    def _from_group(grouped: dict[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
        return {
            key: _breakdown_metrics_from_cells(group)
            for key, group in sorted(grouped.items())
        }

    return {
        "breakdowns_by_model": _from_group(by_model),
        "breakdowns_by_compressor": _from_group(by_compressor),
        "breakdowns_by_prompt": _from_group(by_prompt),
        "breakdowns_by_max_new_tokens": _from_group(by_max_new),
    }


def validate_exp103_panel_report(report: dict[str, Any]) -> L4NoopScaffoldValidationResult:
    """Validate Experiment 103 L4 no-op scaffold panel report."""
    errors: list[str] = []
    required_top = (
        "experiment_id",
        "status",
        "stage",
        "mode",
        "research_script_flag_used",
        "production_cli_modified",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "l4_runtime_commit_implemented",
        "verifier_mediated_acceptance_performed",
        "rollback_runtime_implemented",
        "fallback_runtime_implemented",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "models_requested",
        "models_loaded",
        "models_blocked",
        "device",
        "dtype",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens_values",
        "total_cells",
        "successful_baseline_cells",
        "successful_noop_scaffold_cells",
        "token_match_cells",
        "text_match_cells",
        "blocked_cells",
        "exactkv_failure_summary",
        "safety_gate_summary",
        "breakdowns_by_model",
        "breakdowns_by_compressor",
        "breakdowns_by_prompt",
        "breakdowns_by_max_new_tokens",
        "cells",
        "allowed_next_phase",
        "forbidden_next_phase",
        "limitations",
    )
    for key in required_top:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_103_ID:
        errors.append("experiment_id mismatch")

    bool_must_be_false = (
        "l4_runtime_commit_implemented",
        "verifier_mediated_acceptance_performed",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "rollback_runtime_implemented",
        "fallback_runtime_implemented",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
    )
    for key in bool_must_be_false:
        if report.get(key) is not False:
            errors.append(f"{key} must be false")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21B:
        errors.append("allowed_next_phase must be phase21c_l4_trace_only_dry_run_design")

    if report.get("forbidden_next_phase") != FORBIDDEN_NEXT_PHASE_21B:
        errors.append("forbidden_next_phase must be l4_runtime_commit_implementation")

    cell_required = (
        "model_id",
        "prompt_id",
        "compressor",
        "max_new_tokens",
        "baseline_completed",
        "noop_scaffold_completed",
        "token_match",
        "text_match",
        "safety_gates",
    )
    for idx, cell in enumerate(report.get("cells") or []):
        for ck in cell_required:
            if ck not in cell:
                errors.append(f"cells[{idx}] missing {ck}")
        if not cell.get("baseline_completed") or not cell.get("noop_scaffold_completed"):
            continue
        if cell.get("token_match") is not True:
            errors.append(f"cells[{idx}] token_match failed for completed cell")
        if cell.get("text_match") is not True:
            errors.append(f"cells[{idx}] text_match failed for completed cell")
        sg = cell.get("safety_gates") or {}
        for key in bool_must_be_false:
            if sg.get(key) is True:
                errors.append(f"cells[{idx}].safety_gates.{key} must be false")
        if sg.get("l4_noop_scaffold_enabled") is not True:
            errors.append(f"cells[{idx}].safety_gates.l4_noop_scaffold_enabled must be true")

    if report.get("status") == "panel_complete":
        total = report.get("total_cells") or 0
        if report.get("token_match_cells") != total:
            errors.append("panel_complete but token_match_cells != total_cells")
        if report.get("text_match_cells") != total:
            errors.append("panel_complete but text_match_cells != total_cells")

    return L4NoopScaffoldValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
    )


def build_synthetic_exp103_panel_report(
    *,
    num_cells: int = 4,
    unsafe: bool = False,
) -> dict[str, Any]:
    """Build synthetic panel report for unit tests."""
    config = default_l4_noop_opt_in_config(enabled=True)
    gates = default_l4_noop_safety_gates(scaffold_enabled=True)
    trace = build_l4_noop_scaffold_trace(config)
    cells: list[dict[str, Any]] = []
    for i in range(num_cells):
        cell_result = L4NoopCellResult(
            prompt_id=f"p{i}",
            prompt_preview=f"prompt {i}",
            compressor="noop" if i % 2 == 0 else "int8",
            baseline_completed=True,
            noop_scaffold_completed=True,
            token_match=not unsafe,
            text_match=not unsafe,
            baseline_token_ids=(100, 101),
            noop_scaffold_token_ids=(100, 101) if not unsafe else (999,),
            safety_gates=gates,
            scaffold_trace=trace,
            blockers=("baseline_vs_noop_scaffold_token_mismatch",) if unsafe else (),
        )
        cells.append(
            _panel_cell_to_dict(
                model_id="synthetic-model",
                prompt_id=f"p{i}",
                prompt_preview=f"prompt {i}",
                compressor=cell_result.compressor,
                max_new_tokens=4 if i % 2 == 0 else 8,
                cell=cell_result,
                exactkv_failures_baseline=0,
                exactkv_failures_noop_scaffold=0,
            ),
        )

    breakdowns = aggregate_noop_panel_breakdowns(cells)
    status = "failed" if unsafe else "panel_complete"
    report = {
        "experiment_id": EXPERIMENT_103_ID,
        "status": status,
        "phase": PHASE_21B,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "config": config.to_dict(),
        "research_script_flag_used": True,
        "production_cli_modified": False,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "l4_runtime_commit_implemented": False,
        "verifier_mediated_acceptance_performed": False,
        "rollback_runtime_implemented": False,
        "fallback_runtime_implemented": False,
        "proposal_used_for_token_commit": False,
        "proposal_exposed_to_generator": False,
        "models_requested": ["synthetic-model"],
        "models_loaded": ["synthetic-model"],
        "models_blocked": [],
        "model_block_summary": {"models_blocked_count": 0},
        "device": "cpu",
        "dtype": "float32",
        "compressors_requested": ["noop", "int8"],
        "compressors_run": ["noop", "int8"],
        "max_new_tokens_values": [4, 8],
        "total_cells": num_cells,
        "successful_baseline_cells": num_cells,
        "successful_noop_scaffold_cells": num_cells,
        "token_match_cells": 0 if unsafe else num_cells,
        "text_match_cells": 0 if unsafe else num_cells,
        "blocked_cells": 0,
        "exactkv_failure_summary": {
            "baseline_failures": 0,
            "noop_scaffold_failures": 0,
        },
        "safety_gate_summary": {
            "cells_all_gates_ok": num_cells,
            "cells_with_gate_failure": 0,
            "total_cells": num_cells,
        },
        **breakdowns,
        "cells": cells,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21B,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_21B,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 no-op scaffold panel validation only; not runtime commit.",
            "Panel-scoped parity only; not general model-output preservation.",
        ],
    }
    validation = validate_exp103_panel_report(report)
    report["validation_result"] = validation.to_dict()
    return report


def run_exp103_l4_noop_scaffold_panel_validation(
    *,
    l4_noop_enabled: bool = True,
    model_ids: Sequence[str] | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = DEFAULT_PANEL_PROMPTS,
    max_new_tokens_values: Sequence[int] = DEFAULT_MAX_NEW_TOKENS_VALUES,
    compressors_requested: Sequence[str] = DEFAULT_PANEL_COMPRESSORS,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_model_blocked: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    noop_scaffold_generation_fn: Callable[
        ..., tuple[dict[str, Any], L4NoopScaffoldTrace]
    ] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run Experiment 103 L4 no-op scaffold real-model panel validation."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    config = default_l4_noop_opt_in_config(enabled=l4_noop_enabled)
    models_requested = list(model_ids or DEFAULT_PANEL_MODEL_IDS)
    prompt_panel = (
        list(prompts) if prompts is not None else default_noop_panel_prompts(max_prompts)
    )
    runnable, _blocked_comp = resolve_panel_compressors(compressors_requested)
    mnt_values = list(max_new_tokens_values)

    cells: list[dict[str, Any]] = []
    models_loaded: list[str] = []
    models_blocked: list[dict[str, Any]] = []
    blocked_cells = 0
    failed_cells = 0
    baseline_ok = 0
    noop_ok = 0
    tok_match = 0
    txt_match = 0
    sg_ok = 0

    for model_id in models_requested:
        runtime: Any | None = None
        load_blockers: list[str] = []
        if baseline_generation_fn is None or noop_scaffold_generation_fn is None:
            try:
                if runtime_loader is not None:
                    runtime = runtime_loader(
                        model_id=model_id,
                        device=device,
                        dtype=dtype,
                        local_files_only=local_files_only,
                    )
                else:
                    from exactkv.runtime.model_runtime import ModelRuntime

                    runtime = ModelRuntime(model_id, device=device, dtype=dtype)
            except Exception as exc:  # noqa: BLE001
                load_blockers.append(f"{type(exc).__name__}: {exc}")

        if runtime is None and (
            baseline_generation_fn is None or noop_scaffold_generation_fn is None
        ):
            reason = "; ".join(load_blockers) or "model load failed"
            models_blocked.append({"model_id": model_id, "blocked_reason": reason})
            gates = default_l4_noop_safety_gates(scaffold_enabled=config.enabled)
            trace = build_l4_noop_scaffold_trace(config)
            for prompt_id, prompt_text in prompt_panel:
                preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
                for compressor in runnable:
                    for max_new in mnt_values:
                        blocked_cells += 1
                        cell_result = L4NoopCellResult(
                            prompt_id=prompt_id,
                            prompt_preview=preview,
                            compressor=compressor,
                            baseline_completed=False,
                            noop_scaffold_completed=False,
                            token_match=False,
                            text_match=False,
                            baseline_token_ids=(),
                            noop_scaffold_token_ids=(),
                            safety_gates=gates,
                            scaffold_trace=trace,
                            blockers=(f"model_blocked: {reason}",),
                        )
                        cells.append(
                            _panel_cell_to_dict(
                                model_id=model_id,
                                prompt_id=prompt_id,
                                prompt_preview=preview,
                                compressor=compressor,
                                max_new_tokens=max_new,
                                cell=cell_result,
                                exactkv_failures_baseline=None,
                                exactkv_failures_noop_scaffold=None,
                            ),
                        )
            if not allow_model_blocked:
                failed_cells += len(prompt_panel) * len(runnable) * len(mnt_values)
            continue

        models_loaded.append(model_id)
        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                for max_new in mnt_values:
                    cell, failed = _build_noop_panel_cell(
                        model_id=model_id,
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        compressor=compressor,
                        max_new_tokens=max_new,
                        config=config,
                        runtime=runtime,
                        draft_len=draft_len,
                        baseline_generation_fn=baseline_generation_fn,
                        noop_scaffold_generation_fn=noop_scaffold_generation_fn,
                    )
                    cells.append(cell)
                    if cell["baseline_completed"]:
                        baseline_ok += 1
                    if cell["noop_scaffold_completed"]:
                        noop_ok += 1
                    if cell["token_match"]:
                        tok_match += 1
                    if cell["text_match"]:
                        txt_match += 1
                    if _cell_safety_gates_ok(cell.get("safety_gates") or {}):
                        sg_ok += 1
                    if failed:
                        failed_cells += 1
                    if cell.get("blockers"):
                        blocked_cells += 1

    breakdowns = aggregate_noop_panel_breakdowns(cells)
    total = len(cells)

    if failed_cells > 0:
        status = "failed"
    elif baseline_ok == total and noop_ok == total and tok_match == total and total > 0:
        status = "panel_complete"
    elif baseline_ok > 0:
        status = "panel_partial"
    else:
        status = "blocked"

    exactkv_failure_summary = {
        "baseline_failures": sum(
            1 for c in cells if (c.get("exactkv_failures_baseline") or 0) > 0
        ),
        "noop_scaffold_failures": sum(
            1 for c in cells if (c.get("exactkv_failures_noop_scaffold") or 0) > 0
        ),
    }

    report = {
        "experiment_id": EXPERIMENT_103_ID,
        "status": status,
        "phase": PHASE_21B,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "config": config.to_dict(),
        "research_script_flag_used": l4_noop_enabled,
        "production_cli_modified": False,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "l4_runtime_commit_implemented": False,
        "verifier_mediated_acceptance_performed": False,
        "rollback_runtime_implemented": False,
        "fallback_runtime_implemented": False,
        "proposal_used_for_token_commit": False,
        "proposal_exposed_to_generator": False,
        "models_requested": models_requested,
        "models_loaded": models_loaded,
        "models_blocked": models_blocked,
        "model_block_summary": {
            "models_requested_count": len(models_requested),
            "models_loaded_count": len(models_loaded),
            "models_blocked_count": len(models_blocked),
        },
        "device": device,
        "dtype": dtype,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens_values": mnt_values,
        "total_cells": total,
        "successful_baseline_cells": baseline_ok,
        "successful_noop_scaffold_cells": noop_ok,
        "token_match_cells": tok_match,
        "text_match_cells": txt_match,
        "blocked_cells": blocked_cells,
        "exactkv_failure_summary": exactkv_failure_summary,
        "safety_gate_summary": {
            "cells_all_gates_ok": sg_ok,
            "cells_with_gate_failure": total - sg_ok,
            "total_cells": total,
        },
        **breakdowns,
        "cells": cells,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21B,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_21B,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 no-op scaffold panel validation only; not L4 runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Research-script flag only; production CLI unchanged.",
            "Panel-scoped token/text parity only; not general preservation claim.",
            "No verifier-mediated acceptance; no proposal commit.",
        ],
    }
    validation = validate_exp103_panel_report(report)
    report["validation_result"] = validation.to_dict()
    return report
