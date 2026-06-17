"""Generation-shadow wiring review utilities (Phase 16J).

Inspects the ExactKV codebase and produces a claim-safe plan for a future
opt-in generation-shadow observer. **Does not** wire streaming attention into
ExactKV generation or alter token generation behavior.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from exactkv.attention.streaming_quant_attention import FORBIDDEN_ATTENTION_CLAIMS

EXPERIMENT_075_ID = "exp075_generation_shadow_wiring_review"
DEFAULT_EXP075_REPORT = Path("reports/experiment_075_generation_shadow_wiring_review.json")

EXP075_CLAIM_NOTE = (
    "Generation-shadow wiring review (Phase 16J). Reviews hook points and safety "
    "gates for a future opt-in L1 generation observer. Not generation integration, "
    "vLLM, CUDA/Triton kernels, or ExactKV default runtime. Shadow logits/top-k "
    "are diagnostic only. No speed, throughput, latency, active GPU memory, or "
    "serving claim."
)

PROPOSED_SHADOW_CLI_FLAG = "--generation-shadow-observer"
PROPOSED_SHADOW_CLI_DEST = "generation_shadow_observer"

DEFAULT_RUNTIME_UNCHANGED_CLAIM = (
    "ExactKV default generation (ExactKVGenerator.generate) remains unchanged; "
    "shadow mode is not wired into the generator in this phase."
)

SHADOW_FORBIDDEN_CLAIMS: tuple[str, ...] = FORBIDDEN_ATTENTION_CLAIMS + (
    "exact generation preservation",
    "model-output preservation",
    "shadow exactness guarantee",
    "production correctness guarantee",
)

INSPECT_PATHS: tuple[str, ...] = (
    "exactkv/runtime/exactkv_generator.py",
    "exactkv/runtime/generation.py",
    "exactkv/runtime/prefill.py",
    "exactkv/runtime/model_runtime.py",
    "exactkv/runtime/experimental_cli.py",
    "exactkv/runtime/experimental.py",
    "exactkv/verification/engine.py",
    "exactkv/verification/acceptance.py",
    "exactkv/cache/full_state.py",
    "exactkv/cache/compressed_state.py",
    "exactkv/cache/restored_verifier_runner.py",
    "exactkv/compressors/base.py",
    "exactkv/benchmarks/runner.py",
    "exactkv/attention/streaming_quant_attention.py",
    "exactkv/attention/hf_full_replay_probe.py",
    "exactkv/attention/hf_multilayer_probe.py",
    "exactkv/attention/tolerance_policy.py",
    "examples/qwen_smoke.py",
)


class ShadowLevelId(str, Enum):
    L0_OFFLINE_REPLAY = "L0_offline_replay"
    L1_GENERATION_OBSERVER = "L1_generation_observer"
    L2_DRAFT_SHADOW = "L2_draft_shadow"
    L3_RESTORED_VERIFIER_SHADOW = "L3_restored_verifier_shadow"
    L4_RUNTIME_INTEGRATION = "L4_runtime_integration"


@dataclass
class ShadowLevelSpec:
    level_id: str
    name: str
    description: str
    implementation_status: str
    required_hooks: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    allowed_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    recommended_or_not: str = "not_recommended"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SAFETY_GATES: tuple[dict[str, str], ...] = (
    {"gate_id": "opt_in_only", "description": "Shadow mode must be explicitly enabled; never default-on."},
    {"gate_id": "default_runtime_unchanged", "description": "ExactKVGenerator and default CLI behavior unchanged when flag absent."},
    {"gate_id": "generated_tokens_unaffected", "description": "Shadow diagnostics must not alter committed or output token IDs."},
    {"gate_id": "no_streaming_token_commit", "description": "Streaming/shadow logits must never be used for argmax or token commit."},
    {"gate_id": "no_speed_memory_claim", "description": "Shadow reports must not claim throughput, latency, or GPU memory savings."},
    {"gate_id": "diagnostic_only_output", "description": "Shadow logits/top-k labeled diagnostic only; not exactness guarantees."},
    {"gate_id": "exactness_via_full_path", "description": "Exactness still judged against full generation / restored verifier paths."},
)


def _repo_root(start: Path | None = None) -> Path:
    if start is not None:
        return start
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "exactkv").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
        if (parent / "exactkv").is_dir() and (parent / "setup.py").is_file():
            return parent
    return here.parents[2]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _extract_symbols(source: str) -> dict[str, list[str]]:
    classes: list[str] = []
    functions: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"classes": classes, "functions": functions}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    return {"classes": sorted(set(classes)), "functions": sorted(set(functions))}


def _grep_patterns(source: str, patterns: Sequence[str]) -> dict[str, bool]:
    return {pat: bool(re.search(pat, source, re.MULTILINE)) for pat in patterns}


def inspect_codebase(
    root: Path | None = None,
    *,
    inspect_paths: Sequence[str] = INSPECT_PATHS,
) -> dict[str, Any]:
    """Inspect local ExactKV codebase for generation-shadow hook points."""
    repo = _repo_root(root)
    files_inspected: list[dict[str, Any]] = []
    missing_files: list[str] = []
    aggregate_patterns: dict[str, bool] = {}

    pattern_checks = {
        "ExactKVGenerator": r"class\s+ExactKVGenerator",
        "generate_method": r"def\s+generate\(",
        "prefill_to_full_state": r"def\s+prefill_to_full_state",
        "ModelRuntime_forward_logits": r"def\s+forward\(",
        "experimental_cli_flag": r"--experimental-restored-verifier",
        "streaming_compressed": r"streaming_compressed|attention_streaming_compressed",
        "tolerance_policy": r"AttentionTolerancePolicy|evaluate_offline_attention_cell",
        "restored_verifier": r"restored_verifier|RestoredVerifier",
        "verification_engine": r"class\s+VerificationEngine",
        "benchmark_runner": r"def\s+run_one\(",
    }

    for rel in inspect_paths:
        path = repo / rel
        if not path.is_file():
            missing_files.append(rel)
            continue
        text = _read_text(path)
        if text is None:
            missing_files.append(rel)
            continue
        symbols = _extract_symbols(text)
        hits = _grep_patterns(text, list(pattern_checks.values()))
        for key, found in zip(pattern_checks.keys(), hits.values(), strict=True):
            aggregate_patterns[key] = aggregate_patterns.get(key, False) or found
        files_inspected.append({
            "path": rel,
            "exists": True,
            "line_count": text.count("\n") + 1,
            "classes": symbols["classes"][:20],
            "functions": symbols["functions"][:30],
            "pattern_hits": dict(zip(pattern_checks.keys(), hits.values(), strict=True)),
        })

    detected = _detect_runtime_components(aggregate_patterns, files_inspected)
    return {
        "repo_root": str(repo),
        "files_inspected": files_inspected,
        "missing_files": missing_files,
        "aggregate_pattern_hits": aggregate_patterns,
        "detected_runtime_components": detected,
    }


def _detect_runtime_components(
    patterns: dict[str, bool],
    files: list[dict[str, Any]],
) -> list[dict[str, str]]:
    components: list[dict[str, str]] = []
    mapping = [
        ("ExactKVGenerator", "exactkv/runtime/exactkv_generator.py", "Main draft-verify-commit generation loop"),
        ("ModelRuntime", "exactkv/runtime/model_runtime.py", "HF model wrapper; forward exposes logits"),
        ("prefill_to_full_state", "exactkv/runtime/prefill.py", "Prompt → FullKVState prefill"),
        ("VerificationEngine", "exactkv/verification/engine.py", "Sequential/span verifier; does not commit tokens"),
        ("RestoredVerifierRunner", "exactkv/cache/restored_verifier_runner.py", "Experimental dual-cache verifier API"),
        ("ExperimentalCli", "exactkv/runtime/experimental_cli.py", "Explicit opt-in CLI flag pattern"),
        ("BenchmarkRunner", "exactkv/benchmarks/runner.py", "Full/lossy/ExactKV benchmark cells"),
        ("StreamingQuantAttention", "exactkv/attention/streaming_quant_attention.py", "Phase 16A tensor reference"),
        ("HfFullReplayProbe", "exactkv/attention/hf_full_replay_probe.py", "Phase 16F–16H offline replay/trace"),
        ("TolerancePolicy", "exactkv/attention/tolerance_policy.py", "Phase 16I offline interpretation policy"),
    ]
    present_paths = {f["path"] for f in files}
    for name, path, role in mapping:
        components.append({
            "component": name,
            "path": path,
            "role": role,
            "present": str(path in present_paths),
        })
    if patterns.get("experimental_cli_flag"):
        components.append({
            "component": "ExperimentalCliFlagPattern",
            "path": "exactkv/runtime/experimental_cli.py",
            "role": "Precedent for explicit opt-in flags (--experimental-restored-verifier)",
            "present": "True",
        })
    return components


def build_hook_point_review(inspection: dict[str, Any]) -> dict[str, Any]:
    """Concrete hook-point review for a future L1 generation observer."""
    patterns = inspection.get("aggregate_pattern_hits", {})
    l1_viable = all(
        patterns.get(k, False)
        for k in (
            "ExactKVGenerator",
            "generate_method",
            "prefill_to_full_state",
            "ModelRuntime_forward_logits",
            "streaming_compressed",
        )
    )
    return {
        "prompt_entry_points": [
            "ExactKVGenerator.generate(prompt, max_new_tokens)",
            "prefill_to_full_state(runtime, prompt) → FullKVState.prompt_ids",
            "ModelRuntime.encode(prompt) → input_ids",
            "benchmarks.runner.run_one → generate_full_greedy / ExactKVGenerator",
        ],
        "generation_output_points": [
            "ExactKVGenerator.generate → ExactKVResult.output_ids / output_text",
            "generate_full_greedy → FullGreedyResult.generated_ids",
            "ExactKVResult.prompt_ids + output_ids → full prefix for offline replay",
        ],
        "logit_observation_points": [
            "ModelRuntime.forward → ModelOutput.logits (available; not persisted by generator)",
            "prefill metadata next_token_id (first-token greedy prediction only)",
            "VerificationEngine verify paths (verifier tokens, not streaming shadow)",
            "hf_full_replay_probe offline replay (prefix logit at last position)",
        ],
        "shadow_comparison_sufficient": l1_viable,
        "minimal_future_cli_flag": PROPOSED_SHADOW_CLI_FLAG,
        "minimal_future_cli_dest": PROPOSED_SHADOW_CLI_DEST,
        "proposed_wrapper_flow": [
            "1. Run ExactKVGenerator.generate unchanged (authoritative tokens).",
            "2. Reconstruct prefix input_ids = cat(prompt_ids, output_ids).",
            "3. Run offline streaming-vs-materialized replay/trace (16F–16G) on prefix.",
            "4. Apply tolerance_policy (16I) to shadow metrics.",
            "5. Emit separate shadow report; never feed shadow logits to argmax.",
        ],
        "must_remain_forbidden": [
            "Modifying ExactKVGenerator._draft / _commit / _verify_draft_tokens",
            "Using streaming attention output for token selection",
            "Default-on shadow mode or environment-variable activation",
            "Claiming shadow top-k agreement as exact generation preservation",
        ],
        "precedent_opt_in_pattern": "exactkv/runtime/experimental_cli.py (--experimental-restored-verifier)",
    }


def _base_shadow_levels() -> list[ShadowLevelSpec]:
    diag_only = "Diagnostic only; not exactness or production correctness."
    return [
        ShadowLevelSpec(
            level_id=ShadowLevelId.L0_OFFLINE_REPLAY.value,
            name="Offline replay",
            description="Existing Phase 16F/16G full-prefix offline replay and divergence trace. No generation involvement.",
            implementation_status="implemented",
            required_hooks=["hf_full_replay_probe", "tolerance_policy"],
            risks=["Prefix-only; does not observe per-decode-step drift during generation."],
            blockers=[],
            allowed_claims=[
                "Offline streaming-vs-materialized drift measured on fixed prefixes",
                "Teacher-forced vs free-running divergence classification",
            ],
            forbidden_claims=list(SHADOW_FORBIDDEN_CLAIMS),
            recommended_or_not="implemented_baseline",
        ),
        ShadowLevelSpec(
            level_id=ShadowLevelId.L1_GENERATION_OBSERVER.value,
            name="Generation observer",
            description=(
                "Run normal generation unchanged; separately run offline/logit shadow "
                "on the same prompt/prefix; compare logits/top-k after the fact."
            ),
            implementation_status="not_implemented",
            required_hooks=[
                "Wrapper around ExactKVGenerator.generate (external to generator)",
                "prefix input_ids reconstruction from ExactKVResult",
                "run_exp072_trace_cell / tolerance_policy evaluate",
                f"Explicit CLI flag {PROPOSED_SHADOW_CLI_FLAG}",
            ],
            risks=[
                "Shadow replay cost is additional to generation (offline CPU replay).",
                "Qwen2.5-specific HF extraction may not generalize without panel extension.",
                "Prefix shadow does not capture intra-generation KV evolution per round.",
            ],
            blockers=[
                "No in-generator hook added yet (by design for Phase 16J).",
                "Per-round shadow during multi-round ExactKV loop not specified.",
            ],
            allowed_claims=[
                "Shadow metrics recorded beside unchanged generation output",
                "Supplementary top-k agreement on prefix shadow logits",
                "Tolerance-policy interpretation of shadow drift",
            ],
            forbidden_claims=list(SHADOW_FORBIDDEN_CLAIMS),
            recommended_or_not="recommended_next",
        ),
        ShadowLevelSpec(
            level_id=ShadowLevelId.L2_DRAFT_SHADOW.value,
            name="Draft shadow",
            description=(
                "Streaming compressed attention produces shadow draft/logit estimates; "
                "verifier/generator still commits only full-path tokens."
            ),
            implementation_status="not_implemented",
            required_hooks=[
                "Hook inside ExactKVGenerator._draft or parallel shadow forward",
                "Streaming attention module wired beside materialized draft path",
                "Shadow/draft logit comparison without token commit",
            ],
            risks=[
                "Risk of accidental draft path mutation if hooked inside _draft.",
                "DynamicCache sharing between draft and full state.",
            ],
            blockers=[
                "Draft path uses compressor.materialize_for_draft, not streaming_quant_attention.",
                "No safe in-generator hook designed yet.",
            ],
            allowed_claims=["Future research prototype only when explicitly opt-in"],
            forbidden_claims=list(SHADOW_FORBIDDEN_CLAIMS),
            recommended_or_not="not_recommended_yet",
        ),
        ShadowLevelSpec(
            level_id=ShadowLevelId.L3_RESTORED_VERIFIER_SHADOW.value,
            name="Restored verifier shadow",
            description="Future interaction with restored full-KV verifier experimental runtime.",
            implementation_status="not_implemented",
            required_hooks=[
                "restored_verifier_runner integration",
                "Shadow metrics alongside verify_sequential/verify_span",
            ],
            risks=["Verifier cache reload semantics; GPU/CPU residency complexity."],
            blockers=[
                "Restored verifier is separate experimental API (Phase 13B+).",
                "No streaming attention integration in verifier path.",
            ],
            allowed_claims=["Experimental dual-cache verify observations only"],
            forbidden_claims=list(SHADOW_FORBIDDEN_CLAIMS),
            recommended_or_not="not_recommended_yet",
        ),
        ShadowLevelSpec(
            level_id=ShadowLevelId.L4_RUNTIME_INTEGRATION.value,
            name="Runtime integration",
            description="Future production-style streaming attention inside default runtime.",
            implementation_status="forbidden_for_now",
            required_hooks=["ExactKVGenerator internal integration", "Default runtime change"],
            risks=["Token commit correctness", "Performance claim pressure", "Default behavior change"],
            blockers=[
                "Explicitly forbidden until offline evidence and shadow observer mature.",
                "Phase 16A–16I do not authorize default runtime wiring.",
            ],
            allowed_claims=[],
            forbidden_claims=list(SHADOW_FORBIDDEN_CLAIMS) + ["default runtime integration"],
            recommended_or_not="forbidden",
        ),
    ]


def recommend_next_level(
    hook_review: dict[str, Any],
    *,
    shadow_levels: Sequence[ShadowLevelSpec],
) -> tuple[str, str]:
    """Return (recommended_next_level, recommended_next_phase)."""
    if hook_review.get("shadow_comparison_sufficient"):
        return (
            ShadowLevelId.L1_GENERATION_OBSERVER.value,
            "Phase 16K: implement external L1 generation-shadow observer wrapper "
            f"with explicit {PROPOSED_SHADOW_CLI_FLAG} flag; still no ExactKVGenerator changes.",
        )
    return (
        ShadowLevelId.L0_OFFLINE_REPLAY.value,
        "Remain at L0 offline replay until hook inspection passes.",
    )


def run_exp075_generation_shadow_review(
    *,
    root: Path | None = None,
    inspect_paths: Sequence[str] = INSPECT_PATHS,
) -> dict[str, Any]:
    """Run Experiment 075 generation-shadow wiring review."""
    inspection = inspect_codebase(root, inspect_paths=inspect_paths)
    hook_review = build_hook_point_review(inspection)
    levels = _base_shadow_levels()
    recommended_level, recommended_phase = recommend_next_level(hook_review, shadow_levels=levels)

    required_hooks = [
        *hook_review.get("proposed_wrapper_flow", []),
        f"Future CLI: {PROPOSED_SHADOW_CLI_FLAG} (explicit opt-in; not added in 16J)",
    ]

    blockers = [
        "Phase 16J is review-only; no generation integration implemented.",
        "Streaming attention not wired into ExactKVGenerator.",
        *(
            lvl.blockers
            for lvl in levels
            if lvl.level_id == ShadowLevelId.L1_GENERATION_OBSERVER.value
        ),
    ]

    status = "review_complete"
    if inspection.get("missing_files"):
        status = "review_complete_with_missing_paths"

    return {
        "experiment_id": EXPERIMENT_075_ID,
        "status": status,
        "files_inspected": [f["path"] for f in inspection["files_inspected"]],
        "files_missing": inspection.get("missing_files", []),
        "detected_runtime_components": inspection["detected_runtime_components"],
        "hook_point_review": hook_review,
        "shadow_levels": [lvl.to_dict() for lvl in levels],
        "recommended_next_level": recommended_level,
        "recommended_next_phase": recommended_phase,
        "required_hooks": required_hooks,
        "blockers": blockers,
        "safety_gates": [dict(g) for g in SAFETY_GATES],
        "allowed_claims": [
            "Generation-shadow wiring review completed",
            "L1 generation observer identified as safest next engineering step",
            "Default ExactKV generation unchanged",
            "Shadow output would be diagnostic only",
            "Offline tolerance policy applies to shadow metrics",
        ],
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "default_runtime_unchanged_claim": DEFAULT_RUNTIME_UNCHANGED_CLAIM,
        "limitations": [
            "Review/planning phase only; no generation integration.",
            "Hook inspection is static (AST/pattern); no runtime execution.",
            "L1 observer prefix-shadow does not cover per-round decode drift.",
            "Qwen2.5-centric offline probes; broader panels deferred.",
            "No CUDA/Triton/vLLM/LMCache/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP075_CLAIM_NOTE,
    }


def validate_exp075_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "files_inspected",
        "detected_runtime_components",
        "shadow_levels",
        "recommended_next_level",
        "recommended_next_phase",
        "required_hooks",
        "blockers",
        "safety_gates",
        "allowed_claims",
        "forbidden_claims",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "default_runtime_unchanged_claim",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_075_ID:
        errors.append("experiment_id mismatch")

    levels = report.get("shadow_levels")
    if not isinstance(levels, list):
        errors.append("shadow_levels must be a list")
        return errors

    level_ids = {lvl.get("level_id") for lvl in levels if isinstance(lvl, dict)}
    for expected in ShadowLevelId:
        if expected.value not in level_ids:
            errors.append(f"missing shadow level {expected.value}")

    for idx, lvl in enumerate(levels):
        if not isinstance(lvl, dict):
            errors.append(f"shadow level {idx} not dict")
            continue
        for ck in (
            "level_id",
            "name",
            "description",
            "implementation_status",
            "required_hooks",
            "risks",
            "blockers",
            "allowed_claims",
            "forbidden_claims",
            "recommended_or_not",
        ):
            if ck not in lvl:
                errors.append(f"shadow level {idx} missing {ck}")

    gates = report.get("safety_gates")
    if not isinstance(gates, list) or len(gates) < len(SAFETY_GATES):
        errors.append("safety_gates incomplete")

    return errors
