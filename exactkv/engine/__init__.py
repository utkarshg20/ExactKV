"""ExactKV engine — unified truth and divergence authority (Phase G)."""
from exactkv.engine.unified_truth_engine import (
    DEFAULT_KERNEL_CONSISTENCY_REPORT,
    DEFAULT_PHASE_A_INPUT,
    DEFAULT_PHASE_D_INPUT,
    DEFAULT_PHASE_F_INPUT,
    DEFAULT_PHASE_G_TRUTH_REPORT,
    DEFAULT_UNIFIED_DIVERGENCE_MAP,
    DEFAULT_LEADERBOARD_INPUT,
    FAILURE_REGIME_STABLE,
    FirstDivergenceAuthority,
    PHASE_G_ID,
    run_phase_g_unified_truth_engine,
    validate_kernel_consistency,
    validate_phase_g_report,
    write_phase_g_outputs,
)

__all__ = [
    "DEFAULT_KERNEL_CONSISTENCY_REPORT",
    "DEFAULT_LEADERBOARD_INPUT",
    "DEFAULT_PHASE_A_INPUT",
    "DEFAULT_PHASE_D_INPUT",
    "DEFAULT_PHASE_F_INPUT",
    "DEFAULT_PHASE_G_TRUTH_REPORT",
    "DEFAULT_UNIFIED_DIVERGENCE_MAP",
    "FAILURE_REGIME_STABLE",
    "FirstDivergenceAuthority",
    "PHASE_G_ID",
    "run_phase_g_unified_truth_engine",
    "validate_kernel_consistency",
    "validate_phase_g_report",
    "write_phase_g_outputs",
]
