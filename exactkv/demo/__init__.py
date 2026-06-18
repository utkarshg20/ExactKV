"""Claim-safe demo and validation utilities (Phase 17)."""

from exactkv.demo.broader_model_validation import (
    CLAIM_SCOPE_NOTE,
    DEFAULT_MODEL_IDS,
    EXPERIMENT_087_ID,
    OPTIONAL_MODEL_IDS,
    run_exp087_broader_model_validation_panel,
    validate_exp087_report,
)
from exactkv.demo.long_context_validation import (
    CLAIM_SCOPE_NOTE as LONG_CONTEXT_CLAIM_SCOPE_NOTE,
    DEFAULT_MODEL_ID,
    EXPERIMENT_088_ID,
    run_exp088_long_context_validation_panel,
    validate_exp088_report,
)
from exactkv.demo.phase17_claim_safe_demo import (
    BENCHMARK_GAP_LINE,
    DEMO_HOOK,
    DEMO_PROBLEM_STATEMENT,
    EXPERIMENT_086_ID,
    build_demo_cards,
    build_demo_sections,
    build_q_and_a,
    run_exp086_claim_safe_demo_packaging,
    validate_exp086_report,
)

__all__ = [
    "BENCHMARK_GAP_LINE",
    "DEMO_HOOK",
    "DEMO_PROBLEM_STATEMENT",
    "EXPERIMENT_086_ID",
    "EXPERIMENT_087_ID",
    "EXPERIMENT_088_ID",
    "CLAIM_SCOPE_NOTE",
    "LONG_CONTEXT_CLAIM_SCOPE_NOTE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_IDS",
    "OPTIONAL_MODEL_IDS",
    "build_demo_cards",
    "build_demo_sections",
    "build_q_and_a",
    "run_exp086_claim_safe_demo_packaging",
    "validate_exp086_report",
    "run_exp087_broader_model_validation_panel",
    "validate_exp087_report",
    "run_exp088_long_context_validation_panel",
    "validate_exp088_report",
]
