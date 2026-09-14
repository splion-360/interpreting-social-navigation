"""File description: Orchestration for paired diagnostic workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from tqdm.auto import tqdm

from evaluate import _load_checkpoint, _seconds_per_step, load_test_config
from logging_utils import configure_cli_logging, get_logger
from models import FlatSocialAttentionModel
from train import _graph_builder, _select_device, load_flat_fit_config

from .config import PoseDiagnosticConfig, RolloutMode, load_pose_diagnostic_config
from .io import save_jsonl, save_jsonl_rows
from .prediction import (
    append_float_values,
    append_horizon_values,
    append_values,
    attention_metrics,
    calibration_metrics,
    case_metrics,
    predict_case,
)
from .records import build_diagnostic_record, case_row
from .report import print_pose_diagnostics
from .windows import build_matched_window_data


LOGGER = get_logger(__name__)
MODEL_NAMES = ("dense_triplet", "single_mouse")
ROLLOUT_MODES = ("autoregressive", "teacher_forced")


def run_pose_diagnostics(config: PoseDiagnosticConfig) -> dict[str, Any]:
    """Run paired diagnostics for dense-triplet and single-mouse checkpoints.

    Args:
        config: Diagnostic input paths and sampling controls.

    Returns:
        JSON-ready summary record.
    """

    dense_config = load_flat_fit_config(config.dense_triplet.train_config_path)
    single_config = load_flat_fit_config(config.single_mouse.train_config_path)
    test_config = load_test_config(config.test_config_path)
    data = build_matched_window_data(
        dense_config=dense_config,
        single_config=single_config,
        test_config=test_config,
        max_triplet_windows=config.max_triplet_windows,
        seed=config.seeds[0],
    )
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
    }
    build_graphs = {
        "dense_triplet": _graph_builder(dense_config.graph_variant),
        "single_mouse": _graph_builder(single_config.graph_variant),
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
                predictions_by_name = {
                    ("dense_triplet", "autoregressive"): predict_case(
                        model=models["dense_triplet"],
                        window=dense_window,
                        mode="autoregressive",
                        prediction_length=dense_config.prediction_length,
                        observation_length=dense_config.observation_length,
                        build_graph=build_graphs["dense_triplet"],
                        device=device,
                        seed=_case_seed(seed, case_index, 0),
                    ),
                    ("single_mouse", "autoregressive"): predict_case(
                        model=models["single_mouse"],
                        window=single_window,
                        mode="autoregressive",
                        prediction_length=single_config.prediction_length,
                        observation_length=single_config.observation_length,
                        build_graph=build_graphs["single_mouse"],
                        device=device,
                        seed=_case_seed(seed, case_index, 1),
                    ),
                    ("dense_triplet", "teacher_forced"): predict_case(
                        model=models["dense_triplet"],
                        window=dense_window,
                        mode="teacher_forced",
                        prediction_length=dense_config.prediction_length,
                        observation_length=dense_config.observation_length,
                        build_graph=build_graphs["dense_triplet"],
                        device=device,
                        seed=_case_seed(seed, case_index, 0),
                    ),
                    ("single_mouse", "teacher_forced"): predict_case(
                        model=models["single_mouse"],
                        window=single_window,
                        mode="teacher_forced",
                        prediction_length=single_config.prediction_length,
                        observation_length=single_config.observation_length,
                        build_graph=build_graphs["single_mouse"],
                        device=device,
                        seed=_case_seed(seed, case_index, 1),
                    ),
                }
                for (model_name, mode), prediction in predictions_by_name.items():
                    window = (
                        dense_window if model_name == "dense_triplet" else single_window
                    )
                    mouse_index = (
                        case.mouse_index if model_name == "dense_triplet" else 0
                    )
                    metrics = case_metrics(
                        prediction=prediction,
                        window=window,
                        normalizer=data.normalizer,
                        mouse_index=mouse_index,
                        seconds_per_step=seconds_per_step,
                    )
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
                            ("dense_triplet", "autoregressive")
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
        attention_values=attention_values,
    )
    save_jsonl(record, config.results_path)
    if config.case_results_path is not None:
        save_jsonl_rows(case_rows, config.case_results_path)
    return record


def main() -> None:
    """Run paired diagnostics from the command line."""

    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Run paired diagnostics between dense triplet and single-mouse models."
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
