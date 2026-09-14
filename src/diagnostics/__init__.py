"""File description: Public API for diagnostic experiment helpers."""

from diagnostics.config import (
    PoseDiagnosticConfig,
    VariantConfig,
    load_pose_diagnostic_config,
)
from diagnostics.prediction import compute_horizon_profile
from diagnostics.report import (
    horizon_profile_rows,
    print_pose_diagnostics,
    save_horizon_profile_figure,
)
from diagnostics.runner import run_pose_diagnostics
from diagnostics.windows import (
    MatchedMouseWindow,
    MatchedWindowData,
    build_matched_window_data,
)


__all__ = [
    "MatchedMouseWindow",
    "MatchedWindowData",
    "PoseDiagnosticConfig",
    "VariantConfig",
    "build_matched_window_data",
    "compute_horizon_profile",
    "horizon_profile_rows",
    "load_pose_diagnostic_config",
    "print_pose_diagnostics",
    "run_pose_diagnostics",
    "save_horizon_profile_figure",
]
