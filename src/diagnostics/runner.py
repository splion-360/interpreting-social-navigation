"""File description: Orchestration for paired diagnostic workflows."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tqdm.auto import tqdm

from evaluate import _load_checkpoint, _seconds_per_step, load_test_config
from logging_utils import configure_cli_logging, get_logger
from metrics import PRIMARY_METRIC_NAMES
from models import FlatSocialAttentionModel
from train import (
    FlatFitConfig,
    build_training_window_data,
    _graph_builder,
    _select_device,
    load_flat_fit_config,
)

from .config import (
    PoseDiagnosticConfig,
    PredictionStatistic,
    RolloutMode,
    load_pose_diagnostic_config,
)
from .distribution import compare_feature_distributions, window_feature_distributions
from .io import save_json, save_jsonl, save_jsonl_rows
from .prediction import (
    append_calibration_profile_values,
    append_float_values,
    append_horizon_values,
    append_values,
    attention_metrics,
    calibration_metrics,
    case_calibration_profile,
    case_metrics,
    predict_case,
)
from .records import build_diagnostic_record, case_row
from .report import (
    print_pose_diagnostics,
    save_calibration_profile_figure,
    save_horizon_profile_figure,
    save_prediction_mode_figure,
)
from .statistics import paired_cluster_bootstrap
from .windows import build_matched_window_data


LOGGER = get_logger(__name__)
MODEL_NAMES = ("dense_triplet", "single_mouse", "disconnected_triplet")
ROLLOUT_MODES = ("autoregressive", "teacher_forced")
PREDICTION_STATISTICS = ("sample", "mean")
PREDICTION_MODE_NAMES = tuple(
    f"{mode}_{statistic}"
    for mode in ROLLOUT_MODES
    for statistic in PREDICTION_STATISTICS
)
CALIBRATION_MODE_NAMES = (
    "autoregressive_sample",
    "autoregressive_mean",
    "teacher_forced",
)


def run_pose_diagnostics(config: PoseDiagnosticConfig) -> dict[str, Any]:
    """Run matched diagnostics for dense, single, and disconnected checkpoints.

    Args:
        config: Diagnostic input paths and sampling controls.

    Returns:
        JSON-ready summary record.
    """

    dense_config = load_flat_fit_config(config.dense_triplet.train_config_path)
    single_config = load_flat_fit_config(config.single_mouse.train_config_path)
    disconnected_config = load_flat_fit_config(
        config.disconnected_triplet.train_config_path
    )
    _ensure_disconnected_contract(dense_config, disconnected_config)
    test_config = load_test_config(config.test_config_path)
    data = build_matched_window_data(
        dense_config=dense_config,
        single_config=single_config,
        test_config=test_config,
        max_triplet_windows=config.max_triplet_windows,
        seed=config.seeds[0],
    )
    distribution_audit = {
        "dense_triplet": _build_distribution_audit(dense_config),
        "single_mouse": _build_distribution_audit(single_config),
    }
    device = _select_device(config.device or dense_config.device)
    LOGGER.info("diagnostics: using %s", device)
    models = {
        "dense_triplet": _load_model(
            checkpoint_path=config.dense_triplet.checkpoint_path,
            device=device,
        ),
        "single_mouse": _load_model(
            checkpoint_path=config.single_mouse.checkpoint_path,
            device=device,
        ),
        "disconnected_triplet": _load_model(
            checkpoint_path=config.disconnected_triplet.checkpoint_path,
            device=device,
        ),
    }
    build_graphs = {
        "dense_triplet": _graph_builder(dense_config.graph_variant),
        "single_mouse": _graph_builder(single_config.graph_variant),
        "disconnected_triplet": _graph_builder(disconnected_config.graph_variant),
    }
    metric_values: dict[str, dict[str, dict[str, list[Any]]]] = {
        model_name: {mode: {} for mode in ROLLOUT_MODES} for model_name in MODEL_NAMES
    }
    metric_values_by_motion: dict[str, dict[str, dict[str, dict[str, list[Any]]]]] = {
        model_name: {
            mode: {stratum: {} for stratum in ("low", "medium", "high")}
            for mode in ROLLOUT_MODES
        }
        for model_name in MODEL_NAMES
    }
    horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]] = {
        model_name: {
            mode: {
                horizon: {} for horizon in range(1, dense_config.prediction_length + 1)
            }
            for mode in ROLLOUT_MODES
        }
        for model_name in MODEL_NAMES
    }
    prediction_mode_values: dict[str, dict[str, dict[str, list[Any]]]] = {
        model_name: {mode: {} for mode in PREDICTION_MODE_NAMES}
        for model_name in MODEL_NAMES
    }
    prediction_mode_horizon_values: dict[
        str, dict[str, dict[int, dict[str, list[Any]]]]
    ] = {
        model_name: {
            mode: {
                horizon: {} for horizon in range(1, dense_config.prediction_length + 1)
            }
            for mode in PREDICTION_MODE_NAMES
        }
        for model_name in MODEL_NAMES
    }
    calibration_horizon_values: dict[
        str, dict[str, dict[int, dict[str, list[Any]]]]
    ] = {
        model_name: {
            mode: {
                horizon: {} for horizon in range(1, dense_config.prediction_length + 1)
            }
            for mode in CALIBRATION_MODE_NAMES
        }
        for model_name in MODEL_NAMES
    }
    calibration_values: dict[str, dict[str, list[float]]] = {
        model_name: {} for model_name in MODEL_NAMES
    }
    attention_values: dict[str, list[float]] = {}
    case_rows: list[dict[str, Any]] = []
    seconds_per_step = _seconds_per_step(dense_config)

    iterator = tqdm(data.cases, desc="Running paired pose diagnostics", unit="case")
    with torch.no_grad():
        for case_index, case in enumerate(iterator):
            dense_window = data.dense_windows[case.dense_index]
            single_window = data.single_windows[case.single_index]
            motion_label = data.motion_labels[case_index]
            for seed in config.seeds:
                model_inputs = {
                    "dense_triplet": (dense_window, dense_config, case.mouse_index, 0),
                    "single_mouse": (single_window, single_config, 0, 1),
                    "disconnected_triplet": (
                        dense_window,
                        disconnected_config,
                        case.mouse_index,
                        0,
                    ),
                }
                predictions_by_name = {}
                for model_name, (
                    window,
                    model_config,
                    _,
                    stream,
                ) in model_inputs.items():
                    for mode in ROLLOUT_MODES:
                        for statistic in PREDICTION_STATISTICS:
                            mode_name = f"{mode}_{statistic}"
                            predictions_by_name[(model_name, mode_name)] = predict_case(
                                model=models[model_name],
                                window=window,
                                mode=cast(RolloutMode, mode),
                                prediction_length=model_config.prediction_length,
                                observation_length=model_config.observation_length,
                                build_graph=build_graphs[model_name],
                                device=device,
                                seed=_case_seed(seed, case_index, stream),
                                prediction_statistic=cast(
                                    PredictionStatistic, statistic
                                ),
                            )
                for (model_name, mode_name), prediction in predictions_by_name.items():
                    window, _, mouse_index, _ = model_inputs[model_name]
                    mode, statistic = mode_name.rsplit("_", 1)
                    metrics = case_metrics(
                        prediction=prediction,
                        window=window,
                        normalizer=data.normalizer,
                        mouse_index=mouse_index,
                        seconds_per_step=seconds_per_step,
                    )
                    append_values(
                        prediction_mode_values[model_name][mode_name], metrics
                    )
                    append_horizon_values(
                        prediction_mode_horizon_values[model_name][mode_name],
                        prediction=prediction,
                        window=window,
                        normalizer=data.normalizer,
                        mouse_index=mouse_index,
                        seconds_per_step=seconds_per_step,
                        mode=cast(RolloutMode, mode),
                    )
                    calibration_mode = (
                        mode_name if mode == "autoregressive" else "teacher_forced"
                    )
                    if mode == "autoregressive" or statistic == "sample":
                        append_calibration_profile_values(
                            calibration_horizon_values[model_name][calibration_mode],
                            case_calibration_profile(
                                prediction=prediction,
                                window=window,
                                normalizer=data.normalizer,
                                mouse_index=mouse_index,
                            ),
                        )
                    if statistic != "sample":
                        continue
                    append_values(metric_values[model_name][mode], metrics)
                    append_values(
                        metric_values_by_motion[model_name][mode][motion_label],
                        metrics,
                    )
                    append_horizon_values(
                        horizon_values[model_name][mode],
                        prediction=prediction,
                        window=window,
                        normalizer=data.normalizer,
                        mouse_index=mouse_index,
                        seconds_per_step=seconds_per_step,
                        mode=cast(RolloutMode, mode),
                    )
                    if mode == "autoregressive":
                        append_float_values(
                            calibration_values[model_name],
                            calibration_metrics(
                                prediction=prediction,
                                window=window,
                                normalizer=data.normalizer,
                                mouse_index=mouse_index,
                            ),
                        )
                        case_rows.append(
                            case_row(
                                case=case,
                                model_name=model_name,
                                mode=mode,
                                seed=seed,
                                motion_label=motion_label,
                                motion_score_px_s=data.motion_scores_px_s[case_index],
                                metrics=metrics,
                            )
                        )
                append_float_values(
                    attention_values,
                    attention_metrics(
                        predictions_by_name[
                            ("dense_triplet", "autoregressive_sample")
                        ].rollout,
                        mouse_index=case.mouse_index,
                    ),
                )

    record = build_diagnostic_record(
        config=config,
        dense_config=dense_config,
        data=data,
        metric_values=metric_values,
        metric_values_by_motion=metric_values_by_motion,
        horizon_values=horizon_values,
        calibration_values=calibration_values,
        prediction_mode_values=prediction_mode_values,
        prediction_mode_horizon_values=prediction_mode_horizon_values,
        calibration_horizon_values=calibration_horizon_values,
        distribution_audit=distribution_audit,
        paired_bootstrap=_paired_bootstrap_summary(
            metric_values,
            clusters=np.asarray(
                [
                    f"{case.sequence_id}:{case.start_frame}"
                    for case in data.cases
                    for _ in config.seeds
                ]
            ),
            seed=config.seeds[0],
        ),
        attention_values=attention_values,
    )
    save_jsonl(record, config.results_path)
    if config.case_results_path is not None:
        save_jsonl_rows(case_rows, config.case_results_path)
    output_dir = config.results_path.parent
    save_jsonl(
        {
            "timestamp_utc": record["timestamp_utc"],
            "window_contract": record["window_contract"],
            "prediction_mode_comparison": record["prediction_mode_comparison"],
        },
        output_dir / "prediction_mode_comparison.jsonl",
    )
    save_json(record["distribution_audit"], output_dir / "distribution_audit.json")
    for statistic in ("p50", "p99"):
        save_horizon_profile_figure(
            record,
            output_dir / f"horizon_profile_{statistic}.png",
            statistic=statistic,
        )
        save_prediction_mode_figure(
            record,
            output_dir / f"prediction_mode_{statistic}.png",
            statistic=statistic,
        )
    save_calibration_profile_figure(
        record,
        output_dir / "calibration_by_horizon.png",
    )
    return record


def main() -> None:
    """Run paired diagnostics from the command line."""

    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Run matched diagnostics across flat graph variants."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/test__flat_pose_diagnostics_5fps.yml"),
    )
    parser.add_argument("--device")
    parser.add_argument("--max-triplet-windows", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--case-results-path", type=Path)
    args = parser.parse_args()
    config = _apply_cli_overrides(load_pose_diagnostic_config(args.config), args)
    record = run_pose_diagnostics(config)
    print_pose_diagnostics(record)
    LOGGER.info("result=%s", config.results_path)
    if config.case_results_path is not None:
        LOGGER.info("case_results=%s", config.case_results_path)


def _load_model(
    *, checkpoint_path: Path, device: torch.device
) -> FlatSocialAttentionModel:
    """Load a flat Social Attention checkpoint for inference."""

    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = FlatSocialAttentionModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _apply_cli_overrides(
    config: PoseDiagnosticConfig,
    args: argparse.Namespace,
) -> PoseDiagnosticConfig:
    """Apply optional command-line overrides to a diagnostic config."""

    updates: dict[str, Any] = {}
    if args.device is not None:
        updates["device"] = args.device
    if args.max_triplet_windows is not None:
        updates["max_triplet_windows"] = args.max_triplet_windows
    if args.seeds is not None:
        updates["seeds"] = tuple(
            int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()
        )
    if args.results_path is not None:
        updates["results_path"] = args.results_path
    if args.case_results_path is not None:
        updates["case_results_path"] = args.case_results_path
    return replace(config, **updates)


def _case_seed(seed: int, case_index: int, stream: int) -> int:
    """Build a deterministic seed for one stochastic diagnostic stream."""

    return int(seed + 1009 * case_index + 9176 * stream)


def _build_distribution_audit(config: FlatFitConfig) -> dict[str, Any]:
    """Compare selected training and validation trajectory distributions."""

    LOGGER.info("diagnostics: rebuilding train and validation window selections")
    data = build_training_window_data(config, show_progress=True)
    seconds_per_step = _seconds_per_step(config)
    train_features = window_feature_distributions(
        data.train_windows,
        seconds_per_step=seconds_per_step,
    )
    validation_features = window_feature_distributions(
        data.validation_windows,
        seconds_per_step=seconds_per_step,
    )
    return {
        "train_windows": len(data.train_windows),
        "validation_windows": len(data.validation_windows),
        "train_sequences": len(data.train_windows.sequences),
        "validation_sequences": len(data.validation_windows.sequences),
        "features": compare_feature_distributions(
            train_features=train_features,
            validation_features=validation_features,
        ),
    }


def _ensure_disconnected_contract(
    dense_config: FlatFitConfig,
    disconnected_config: FlatFitConfig,
) -> None:
    """Validate that the cross-edge ablation changes only compatible settings.

    Args:
        dense_config: Dense-triplet training configuration used as the reference.
        disconnected_config: Within-mouse triplet training configuration.

    Raises:
        ValueError: If temporal settings differ or the disconnected graph variant
            is not the expected within-mouse graph.
    """

    fields = (
        "window_length",
        "observation_length",
        "prediction_length",
        "stride",
        "frame_step",
        "source_fps",
    )
    mismatches = [
        field
        for field in fields
        if getattr(dense_config, field) != getattr(disconnected_config, field)
    ]
    if mismatches:
        raise ValueError(
            "disconnected diagnostic config differs on: " + ", ".join(mismatches)
        )
    if disconnected_config.graph_variant != "within_mouse_dense_keypoint":
        raise ValueError(
            "disconnected diagnostic variant must be within_mouse_dense_keypoint"
        )


def _paired_bootstrap_summary(
    metric_values: dict[str, dict[str, dict[str, list[Any]]]],
    *,
    clusters: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Compare disconnected and dense errors with clustered paired bootstrap.

    Args:
        metric_values: Per-model metric observations grouped by rollout mode.
        clusters: Source-window identifier for every paired mouse observation.
        seed: Random seed used by the bootstrap.

    Returns:
        Bootstrap summaries for each scalar metric and rollout mode.
    """

    summary: dict[str, Any] = {}
    for mode in ROLLOUT_MODES:
        summary[mode] = {}
        dense_values = metric_values["dense_triplet"][mode]
        disconnected_values = metric_values["disconnected_triplet"][mode]
        for metric_name in PRIMARY_METRIC_NAMES:
            reference = dense_values.get(metric_name)
            comparison = disconnected_values.get(metric_name)
            if reference is None or comparison is None:
                continue
            if isinstance(reference[0], np.ndarray) or isinstance(
                comparison[0], np.ndarray
            ):
                continue
            summary[mode][metric_name] = paired_cluster_bootstrap(
                np.asarray(reference),
                np.asarray(comparison),
                clusters=clusters,
                seed=seed,
            )
    return {
        "comparison": "disconnected_triplet_minus_dense_triplet",
        "unit_of_resampling": "source_triplet_window",
        "resamples": 10_000,
        "modes": summary,
    }
