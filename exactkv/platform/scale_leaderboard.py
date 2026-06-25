"""Repair and rebuild scale leaderboard aggregates (Release Gate R1).

Re-exports from leaderboard_aggregates for backward compatibility.
"""
from exactkv.platform.leaderboard_aggregates import (
    count_cells_by_model,
    rebuild_scale_leaderboard_from_raw,
    repair_phase_a_report_aggregates,
    validate_leaderboard_against_raw,
)

__all__ = [
    "count_cells_by_model",
    "repair_phase_a_report_aggregates",
    "rebuild_scale_leaderboard_from_raw",
    "validate_leaderboard_against_raw",
]
