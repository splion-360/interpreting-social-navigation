"""File description: Public API for diagnostic experiment helpers."""

from diagnostics.calibration import calibration_profile
from diagnostics.config import (
    PoseDiagnosticConfig,
    VariantConfig,
    load_pose_diagnostic_config,
)
from diagnostics.distribution import (
    compare_feature_distributions,
    trajectory_window_features,
    window_feature_distributions,
)
from diagnostics.prediction import compute_horizon_profile
from diagnostics.report import (
    horizon_profile_rows,
    prediction_mode_rows,
    print_pose_diagnostics,
    save_calibration_profile_figure,
    save_horizon_profile_figure,
    save_prediction_mode_figure,
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
    "calibration_profile",
    "compare_feature_distributions",
    "compute_horizon_profile",
    "horizon_profile_rows",
    "load_pose_diagnostic_config",
    "prediction_mode_rows",
    "print_pose_diagnostics",
    "run_pose_diagnostics",
    "save_calibration_profile_figure",
    "save_horizon_profile_figure",
    "save_prediction_mode_figure",
    "trajectory_window_features",
    "window_feature_distributions",
]
