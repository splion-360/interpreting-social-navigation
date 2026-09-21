"""File description: Summary record builders for diagnostic workflows."""

from datetime import UTC, datetime
from typing import Any

import numpy as np

from evaluate import (
    MOTION_STRATA,
    _aggregate_metric_quantiles,
    _aggregate_metric_values,
)
from metrics import PRIMARY_METRIC_NAMES
from train import FlatFitConfig

from .config import PoseDiagnosticConfig
from .windows import MatchedMouseWindow, MatchedWindowData, matched_window_digest


def case_row(
    *,
    case: MatchedMouseWindow,
    model_name: str,
    mode: str,
    seed: int,
    motion_label: str,
    motion_score_px_s: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build one compact per-case row for downstream inspection."""

    return {
        "sequence_id": case.sequence_id,
        "start_frame": case.start_frame,
        "mouse_index": case.mouse_index,
        "model": model_name,
        "mode": mode,
        "seed": seed,
        "motion_group": motion_label,
        "motion_score_px_s": motion_score_px_s,
        "metrics": {
            name: value
            for name, value in metrics.items()
            if name in PRIMARY_METRIC_NAMES and isinstance(value, int | float)
        },
    }


def build_diagnostic_record(
    *,
    config: PoseDiagnosticConfig,
    dense_config: FlatFitConfig,
    data: MatchedWindowData,
    metric_values: dict[str, dict[str, dict[str, list[Any]]]],
    metric_values_by_motion: dict[str, dict[str, dict[str, dict[str, list[Any]]]]],
    horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]],
    calibration_values: dict[str, dict[str, list[float]]],
    prediction_mode_values: dict[str, dict[str, dict[str, list[Any]]]],
    prediction_mode_horizon_values: dict[
        str, dict[str, dict[int, dict[str, list[Any]]]]
    ],
    calibration_horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]],
    distribution_audit: dict[str, Any],
    paired_bootstrap: dict[str, Any],
    attention_values: dict[str, list[float]],
) -> dict[str, Any]:
    """Assemble a JSON-ready diagnostic record."""

    summaries: dict[str, Any] = {}
    for model_name, by_mode in metric_values.items():
        summaries[model_name] = {}
        for mode, values in by_mode.items():
            summaries[model_name][mode] = {
                "metrics": _aggregate_metric_values(values),
                "metric_quantiles": _aggregate_metric_quantiles(values),
                "metrics_by_motion": {
                    stratum: _aggregate_metric_values(group_values)
                    for stratum, group_values in metric_values_by_motion[model_name][
                        mode
                    ].items()
                    if group_values
                },
                "metric_quantiles_by_motion": {
                    stratum: _aggregate_metric_quantiles(group_values)
                    for stratum, group_values in metric_values_by_motion[model_name][
                        mode
                    ].items()
                    if group_values
                },
            }

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "lineage": {
            "dense_train_config_path": str(config.dense_triplet.train_config_path),
            "single_train_config_path": str(config.single_mouse.train_config_path),
            "disconnected_train_config_path": str(
                config.disconnected_triplet.train_config_path
            ),
            "dense_checkpoint_path": str(config.dense_triplet.checkpoint_path),
            "single_checkpoint_path": str(config.single_mouse.checkpoint_path),
            "disconnected_checkpoint_path": str(
                config.disconnected_triplet.checkpoint_path
            ),
            "test_config_path": str(config.test_config_path),
            "case_results_path": str(config.case_results_path)
            if config.case_results_path
            else None,
        },
        "window_contract": {
            "observation_length": dense_config.observation_length,
            "prediction_length": dense_config.prediction_length,
            "window_length": dense_config.window_length,
            "frame_step": dense_config.frame_step,
            "source_fps": dense_config.source_fps,
            "effective_fps": dense_config.source_fps / dense_config.frame_step,
            "observation_seconds": dense_config.observation_length
            * dense_config.frame_step
            / dense_config.source_fps,
            "prediction_seconds": dense_config.prediction_length
            * dense_config.frame_step
            / dense_config.source_fps,
            "triplet_windows": len(data.dense_windows),
            "mouse_cases": len(data.cases),
            "seeds": list(config.seeds),
            "window_digest": matched_window_digest(data.cases),
        },
        "motion_profile": {
            "score": "mean_keypoint_speed_px_s",
            "thresholds_px_s": {
                "low_max": data.motion_thresholds[0],
                "medium_max": data.motion_thresholds[1],
            },
            "counts": {
                stratum: data.motion_labels.count(stratum) for stratum in MOTION_STRATA
            },
        },
        "methods": summaries,
        "paired_delta_single_minus_dense": _paired_delta_summary(metric_values),
        "paired_bootstrap": paired_bootstrap,
        "horizon_profile": _horizon_profile(horizon_values),
        "prediction_mode_comparison": _prediction_mode_summary(
            prediction_mode_values,
            prediction_mode_horizon_values,
        ),
        "calibration": {
            model_name: _aggregate_float_values(values)
            for model_name, values in calibration_values.items()
        },
        "calibration_by_horizon": _horizon_profile(calibration_horizon_values),
        "distribution_audit": distribution_audit,
        "attention": _aggregate_float_values(attention_values),
    }


def _paired_delta_summary(
    metric_values: dict[str, dict[str, dict[str, list[Any]]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute single-minus-dense paired deltas for scalar metrics."""

    output: dict[str, dict[str, dict[str, float]]] = {}
    for mode in ("autoregressive", "teacher_forced"):
        output[mode] = {}
        for metric_name in PRIMARY_METRIC_NAMES:
            dense = metric_values["dense_triplet"][mode].get(metric_name)
            single = metric_values["single_mouse"][mode].get(metric_name)
            if dense is None or single is None:
                continue
            if isinstance(dense[0], np.ndarray) or isinstance(single[0], np.ndarray):
                continue
            deltas = np.asarray(single, dtype=np.float32) - np.asarray(
                dense, dtype=np.float32
            )
            finite = deltas[np.isfinite(deltas)]
            if finite.size == 0:
                continue
            output[mode][metric_name] = {
                "mean": float(finite.mean()),
                "p50": float(np.percentile(finite, 50)),
                "p99": float(np.percentile(finite, 99)),
                "single_better_rate": float((finite < 0).mean()),
            }
    return output


def _horizon_profile(
    horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]],
) -> dict[str, Any]:
    """Aggregate exact-step horizon metrics for compact JSON output."""

    summary: dict[str, Any] = {}
    for model_name, by_mode in horizon_values.items():
        summary[model_name] = {}
        for mode, by_horizon in by_mode.items():
            summary[model_name][mode] = {
                str(horizon): {
                    "metrics": _aggregate_metric_values(values),
                    "metric_quantiles": _aggregate_metric_quantiles(values),
                }
                for horizon, values in by_horizon.items()
                if values
            }
    return summary


def _prediction_mode_summary(
    metric_values: dict[str, dict[str, dict[str, list[Any]]]],
    horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]],
) -> dict[str, Any]:
    """Aggregate metrics for each conditioning and Gaussian-output mode."""

    summary: dict[str, Any] = {}
    for model_name, by_mode in metric_values.items():
        summary[model_name] = {}
        for mode, values in by_mode.items():
            summary[model_name][mode] = {
                "metrics": _aggregate_metric_values(values),
                "metric_quantiles": _aggregate_metric_quantiles(values),
                "horizon_profile": _horizon_profile(
                    {model_name: {mode: horizon_values[model_name][mode]}}
                )[model_name][mode],
            }
    return summary


def _aggregate_float_values(values: dict[str, list[float]]) -> dict[str, float]:
    """Average scalar diagnostic accumulators."""

    return {
        name: float(np.asarray(items, dtype=np.float32).mean())
        for name, items in values.items()
        if items
    }
