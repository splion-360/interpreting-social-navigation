"""File description: Paired diagnostics for flat trajectory model variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from rich import box
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from data import (
    MabeDataset,
    MabeSequence,
    MabeWindowDataset,
    PoseNormalizer,
    Window,
    WindowSpec,
    mean_keypoint_speed_px_s,
    single_mouse_sequence_id,
    to_single_mouse_sequences,
)
from evaluate import (
    DEFAULT_TEST_CONFIG_PATH,
    KEYPOINT_GRAPH_VARIANTS,
    MOTION_STRATA,
    TestConfig,
    _aggregate_metric_quantiles,
    _aggregate_metric_values,
    _fit_training_normalizer,
    _json_safe,
    _load_checkpoint,
    _seconds_per_step,
    load_test_config,
)
from inference import (
    RolloutResult,
    rollout_flat_keypoint_model,
    sample_bivariate_gaussian,
)
from logging_utils import configure_cli_logging, get_logger
from loss import gaussian_2d_parameters
from metrics import PRIMARY_METRIC_NAMES, compute_pixel_metrics
from models import FlatSocialAttentionModel
from train import FlatFitConfig, _graph_builder, _select_device, load_flat_fit_config


ModelName = Literal["dense_triplet", "single_mouse"]
RolloutMode = Literal["autoregressive", "teacher_forced"]
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class VariantConfig:
    """Model variant used in paired diagnostics.

    Attributes:
        train_config_path: YAML file used to train the variant.
        checkpoint_path: Local checkpoint for the variant.
        display_name: Short name shown in diagnostic tables.
    """

    train_config_path: Path
    checkpoint_path: Path
    display_name: str


@dataclass(frozen=True)
class PoseDiagnosticConfig:
    """Configuration for dense-triplet versus single-mouse diagnostics.

    Attributes:
        dense_triplet: Dense triplet model inputs.
        single_mouse: Single-mouse model inputs.
        test_config_path: Held-out test configuration.
        results_path: JSONL destination for the diagnostic summary.
        case_results_path: Optional JSONL destination for per-case rows.
        max_triplet_windows: Number of triplet windows sampled before expanding
            into mouse-specific paired cases.
        seeds: Sampling seeds used to probe stochastic rollout sensitivity.
        device: Optional device override. When omitted, each train config decides.
    """

    dense_triplet: VariantConfig
    single_mouse: VariantConfig
    test_config_path: Path
    results_path: Path
    case_results_path: Path | None
    max_triplet_windows: int
    seeds: tuple[int, ...]
    device: str | None = None


@dataclass(frozen=True)
class MatchedMouseWindow:
    """One paired dense/single evaluation case.

    Attributes:
        sequence_id: Source triplet sequence ID.
        start_frame: Source start frame.
        mouse_index: Mouse index compared in both variants.
        dense_index: Index into the dense triplet window dataset.
        single_index: Index into the single-mouse window dataset.
    """

    sequence_id: str
    start_frame: int
    mouse_index: int
    dense_index: int
    single_index: int


@dataclass(frozen=True)
class MatchedWindowData:
    """Window datasets and pairing metadata for a fair variant comparison.

    Attributes:
        dense_windows: Triplet windows keyed by source sequence and start.
        single_windows: Single-mouse windows keyed by source, mouse, and start.
        cases: Mouse-specific pairings between the two datasets.
        normalizer: Pixel normalizer fitted from the training split.
        motion_labels: Shared low/medium/high labels for each case.
        motion_scores_px_s: Ground-truth motion scores for each case.
        motion_thresholds: Low and medium boundaries fitted on selected cases.
    """

    dense_windows: MabeWindowDataset
    single_windows: MabeWindowDataset
    cases: tuple[MatchedMouseWindow, ...]
    normalizer: PoseNormalizer
    motion_labels: tuple[str, ...]
    motion_scores_px_s: tuple[float, ...]
    motion_thresholds: tuple[float, float]


def load_pose_diagnostic_config(path: Path) -> PoseDiagnosticConfig:
    """Load paired diagnostic configuration from YAML.

    Args:
        path: YAML config path.

    Returns:
        Typed diagnostic configuration.
    """

    raw = yaml.safe_load(path.read_text()) or {}
    return PoseDiagnosticConfig(
        dense_triplet=_load_variant_config(raw["dense_triplet"]),
        single_mouse=_load_variant_config(raw["single_mouse"]),
        test_config_path=Path(raw.get("test_config_path", DEFAULT_TEST_CONFIG_PATH)),
        results_path=Path(raw["results_path"]),
        case_results_path=(
            Path(raw["case_results_path"])
            if raw.get("case_results_path") is not None
            else None
        ),
        max_triplet_windows=int(raw.get("max_triplet_windows", 30)),
        seeds=tuple(int(seed) for seed in raw.get("seeds", [42])),
        device=raw.get("device"),
    )


def build_matched_window_data(
    *,
    dense_config: FlatFitConfig,
    single_config: FlatFitConfig,
    test_config: TestConfig,
    max_triplet_windows: int,
    seed: int,
) -> MatchedWindowData:
    """Build shared mouse-window cases for dense and single-mouse variants.

    Args:
        dense_config: Dense triplet train config.
        single_config: Single-mouse train config.
        test_config: Held-out test config.
        max_triplet_windows: Number of triplet windows to sample.
        seed: Deterministic selection seed.

    Returns:
        Datasets plus an exact pairing between triplet and single-mouse windows.
    """

    _ensure_comparable_window_contracts(dense_config, single_config)
    normalizer = _fit_training_normalizer(dense_config)
    test_dataset = MabeDataset.from_file(test_config.data_path)
    source_sequences = test_dataset.select(test_dataset.sequence_ids)
    spec = _window_spec(dense_config)
    dense_keys = _sample_triplet_window_keys(
        sequences=source_sequences,
        spec=spec,
        max_triplet_windows=max_triplet_windows,
        seed=seed,
    )
    single_keys = tuple(
        (single_mouse_sequence_id(sequence_id, mouse_index), start_frame)
        for sequence_id, start_frame in dense_keys
        for mouse_index in range(3)
    )
    dense_windows = MabeWindowDataset(
        source_sequences,
        spec,
        normalizer=normalizer,
        window_keys=dense_keys,
    )
    single_windows = MabeWindowDataset(
        to_single_mouse_sequences(source_sequences),
        spec,
        normalizer=normalizer,
        window_keys=single_keys,
    )
    dense_index_by_key = {
        key: index for index, key in enumerate(dense_windows.window_keys)
    }
    single_index_by_key = {
        key: index for index, key in enumerate(single_windows.window_keys)
    }
    cases = tuple(
        MatchedMouseWindow(
            sequence_id=sequence_id,
            start_frame=start_frame,
            mouse_index=mouse_index,
            dense_index=dense_index_by_key[(sequence_id, start_frame)],
            single_index=single_index_by_key[
                (single_mouse_sequence_id(sequence_id, mouse_index), start_frame)
            ],
        )
        for sequence_id, start_frame in dense_keys
        for mouse_index in range(3)
    )
    scores = _case_motion_scores(
        dense_windows=dense_windows,
        cases=cases,
        normalizer=normalizer,
        seconds_per_step=_seconds_per_step(dense_config),
    )
    low_max, medium_max = np.quantile(scores, (1.0 / 3.0, 2.0 / 3.0))
    labels = tuple(
        _motion_label(score, low_max=low_max, medium_max=medium_max) for score in scores
    )
    return MatchedWindowData(
        dense_windows=dense_windows,
        single_windows=single_windows,
        cases=cases,
        normalizer=normalizer,
        motion_labels=labels,
        motion_scores_px_s=tuple(float(score) for score in scores),
        motion_thresholds=(float(low_max), float(medium_max)),
    )


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
        model_name: {mode: {} for mode in ("autoregressive", "teacher_forced")}
        for model_name in ("dense_triplet", "single_mouse")
    }
    metric_values_by_motion: dict[str, dict[str, dict[str, dict[str, list[Any]]]]] = {
        model_name: {
            mode: {stratum: {} for stratum in MOTION_STRATA}
            for mode in ("autoregressive", "teacher_forced")
        }
        for model_name in ("dense_triplet", "single_mouse")
    }
    horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]] = {
        model_name: {
            mode: {
                horizon: {} for horizon in range(1, dense_config.prediction_length + 1)
            }
            for mode in ("autoregressive", "teacher_forced")
        }
        for model_name in ("dense_triplet", "single_mouse")
    }
    calibration_values: dict[str, dict[str, list[float]]] = {
        model_name: {} for model_name in ("dense_triplet", "single_mouse")
    }
    attention_values: dict[str, list[float]] = {}
    case_rows: list[dict[str, Any]] = []

    iterator = tqdm(data.cases, desc="Running paired pose diagnostics", unit="case")
    with torch.no_grad():
        for case_index, case in enumerate(iterator):
            dense_window = data.dense_windows[case.dense_index]
            single_window = data.single_windows[case.single_index]
            label = data.motion_labels[case_index]
            for seed in config.seeds:
                dense_predictions = _predict_case(
                    model=models["dense_triplet"],
                    window=dense_window,
                    mode="autoregressive",
                    prediction_length=dense_config.prediction_length,
                    observation_length=dense_config.observation_length,
                    build_graph=build_graphs["dense_triplet"],
                    device=device,
                    seed=_case_seed(seed, case_index, 0),
                )
                single_predictions = _predict_case(
                    model=models["single_mouse"],
                    window=single_window,
                    mode="autoregressive",
                    prediction_length=single_config.prediction_length,
                    observation_length=single_config.observation_length,
                    build_graph=build_graphs["single_mouse"],
                    device=device,
                    seed=_case_seed(seed, case_index, 1),
                )
                dense_tf = _predict_case(
                    model=models["dense_triplet"],
                    window=dense_window,
                    mode="teacher_forced",
                    prediction_length=dense_config.prediction_length,
                    observation_length=dense_config.observation_length,
                    build_graph=build_graphs["dense_triplet"],
                    device=device,
                    seed=_case_seed(seed, case_index, 2),
                )
                single_tf = _predict_case(
                    model=models["single_mouse"],
                    window=single_window,
                    mode="teacher_forced",
                    prediction_length=single_config.prediction_length,
                    observation_length=single_config.observation_length,
                    build_graph=build_graphs["single_mouse"],
                    device=device,
                    seed=_case_seed(seed, case_index, 3),
                )
                predictions_by_name = {
                    ("dense_triplet", "autoregressive"): dense_predictions,
                    ("single_mouse", "autoregressive"): single_predictions,
                    ("dense_triplet", "teacher_forced"): dense_tf,
                    ("single_mouse", "teacher_forced"): single_tf,
                }
                for (model_name, mode), prediction in predictions_by_name.items():
                    metrics = _case_metrics(
                        prediction=prediction,
                        window=dense_window
                        if model_name == "dense_triplet"
                        else single_window,
                        normalizer=data.normalizer,
                        mouse_index=case.mouse_index
                        if model_name == "dense_triplet"
                        else 0,
                        seconds_per_step=_seconds_per_step(dense_config),
                    )
                    _append_values(metric_values[model_name][mode], metrics)
                    _append_values(
                        metric_values_by_motion[model_name][mode][label], metrics
                    )
                    _append_horizon_values(
                        horizon_values[model_name][mode],
                        prediction=prediction,
                        window=dense_window
                        if model_name == "dense_triplet"
                        else single_window,
                        normalizer=data.normalizer,
                        mouse_index=case.mouse_index
                        if model_name == "dense_triplet"
                        else 0,
                        seconds_per_step=_seconds_per_step(dense_config),
                    )
                    if mode == "autoregressive":
                        _append_float_values(
                            calibration_values[model_name],
                            _calibration_metrics(
                                prediction=prediction,
                                window=dense_window
                                if model_name == "dense_triplet"
                                else single_window,
                                normalizer=data.normalizer,
                                mouse_index=case.mouse_index
                                if model_name == "dense_triplet"
                                else 0,
                            ),
                        )
                        case_rows.append(
                            _case_row(
                                case=case,
                                model_name=model_name,
                                mode=mode,
                                seed=seed,
                                motion_label=label,
                                motion_score_px_s=data.motion_scores_px_s[case_index],
                                metrics=metrics,
                            )
                        )
                _append_float_values(
                    attention_values,
                    _attention_metrics(
                        dense_predictions.rollout, mouse_index=case.mouse_index
                    ),
                )

    record = _diagnostic_record(
        config=config,
        dense_config=dense_config,
        single_config=single_config,
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


def print_pose_diagnostics(
    record: dict[str, Any], *, console: Console | None = None
) -> None:
    """Print the paired diagnostic summary as Rich tables.

    Args:
        record: Diagnostic record returned by :func:`run_pose_diagnostics`.
        console: Optional Rich console.
    """

    output = console or Console()
    output.print(_diagnostic_overview_table(record))
    output.print(_diagnostic_metric_table(record, mode="autoregressive"))
    output.print(_diagnostic_metric_table(record, mode="teacher_forced"))
    output.print(_diagnostic_delta_table(record))
    output.print(_diagnostic_motion_table(record))
    output.print(_diagnostic_calibration_table(record))
    output.print(_diagnostic_attention_table(record))


def save_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append one JSON-safe record to a JSONL file.

    Args:
        record: JSON-serializable payload.
        path: Destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n"
        )


def save_jsonl_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Append multiple JSON-safe diagnostic rows to a JSONL file.

    Args:
        rows: Per-case diagnostic rows.
        path: Destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n"
            )


def _load_variant_config(raw: dict[str, Any]) -> VariantConfig:
    """Load one variant block from YAML."""

    return VariantConfig(
        train_config_path=Path(raw["train_config_path"]),
        checkpoint_path=Path(raw["checkpoint_path"]),
        display_name=str(raw["display_name"]),
    )


def _ensure_comparable_window_contracts(
    dense_config: FlatFitConfig,
    single_config: FlatFitConfig,
) -> None:
    """Ensure variants use the same temporal evaluation contract."""

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
        if getattr(dense_config, field) != getattr(single_config, field)
    ]
    if mismatches:
        raise ValueError(f"diagnostic configs differ on: {', '.join(mismatches)}")
    if dense_config.graph_variant not in KEYPOINT_GRAPH_VARIANTS:
        raise ValueError("dense diagnostic variant must use keypoint graphs")
    if single_config.graph_variant != "single_mouse_dense_keypoint":
        raise ValueError(
            "single diagnostic variant must be single_mouse_dense_keypoint"
        )


def _window_spec(config: FlatFitConfig) -> WindowSpec:
    """Build a window spec from a train config."""

    return WindowSpec(
        length=config.window_length,
        observation_length=config.observation_length,
        prediction_length=config.prediction_length,
        stride=config.stride,
        frame_step=config.frame_step,
    )


def _sample_triplet_window_keys(
    *,
    sequences: list[MabeSequence],
    spec: WindowSpec,
    max_triplet_windows: int,
    seed: int,
) -> tuple[tuple[str, int], ...]:
    """Select triplet windows uniformly across all eligible test windows."""

    all_windows = MabeWindowDataset(sequences, spec)
    rng = np.random.default_rng(seed)
    count = min(max_triplet_windows, len(all_windows))
    indices = rng.choice(len(all_windows), size=count, replace=False)
    return tuple(all_windows.window_keys[int(index)] for index in indices)


def _case_motion_scores(
    *,
    dense_windows: MabeWindowDataset,
    cases: tuple[MatchedMouseWindow, ...],
    normalizer: PoseNormalizer,
    seconds_per_step: float,
) -> np.ndarray:
    """Score matched mouse cases by ground-truth future keypoint speed."""

    scores = []
    for case in cases:
        window = dense_windows[case.dense_index]
        target = normalizer.inverse_transform(
            window.future_keypoints[:, case.mouse_index : case.mouse_index + 1]
        )
        scores.append(
            mean_keypoint_speed_px_s(target, seconds_per_step=seconds_per_step)
        )
    return np.asarray(scores, dtype=np.float32)


def _motion_label(score: float, *, low_max: float, medium_max: float) -> str:
    """Return a tertile motion label for one score."""

    if score <= low_max:
        return "low"
    if score <= medium_max:
        return "medium"
    return "high"


@dataclass(frozen=True)
class CasePrediction:
    """Prediction tensors for one case and rollout mode.

    Attributes:
        future_nodes: Normalized predicted future nodes.
        gaussian_outputs: Raw Gaussian parameters for future nodes.
        rollout: Full rollout object for autoregressive mode, if available.
    """

    future_nodes: np.ndarray
    gaussian_outputs: np.ndarray
    rollout: RolloutResult | None = None


def _predict_case(
    *,
    model: FlatSocialAttentionModel,
    window: Window,
    mode: RolloutMode,
    prediction_length: int,
    observation_length: int,
    build_graph: Any,
    device: torch.device,
    seed: int,
) -> CasePrediction:
    """Predict one window under autoregressive or teacher-forced inputs."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if mode == "autoregressive":
        rollout = rollout_flat_keypoint_model(
            model=model,
            observed_keypoints=window.observed_keypoints,
            prediction_length=prediction_length,
            build_graph=build_graph,
            device=device,
            generator=generator,
        )
        return CasePrediction(
            future_nodes=rollout.nodes[-prediction_length:],
            gaussian_outputs=rollout.gaussian_outputs,
            rollout=rollout,
        )

    graph = build_graph(window.keypoints[:-1])
    result = model.forward_with_state(
        nodes=torch.from_numpy(graph.nodes).to(device),
        edge_features=torch.from_numpy(graph.edge_features).to(device),
        edge_specs=graph.edge_specs,
        nodes_present=graph.nodes_present,
        edges_present=graph.edges_present,
        state=None,
    )
    future_outputs = result.outputs[
        observation_length - 1 : observation_length - 1 + prediction_length
    ]
    samples = sample_bivariate_gaussian(future_outputs, generator=generator)
    return CasePrediction(
        future_nodes=samples.detach().cpu().numpy(),
        gaussian_outputs=future_outputs.detach().cpu().numpy(),
        rollout=None,
    )


def _case_metrics(
    *,
    prediction: CasePrediction,
    window: Window,
    normalizer: PoseNormalizer,
    mouse_index: int,
    seconds_per_step: float,
) -> dict[str, Any]:
    """Compute pixel metrics for one matched mouse case."""

    pose_shape = window.future_keypoints.shape
    predicted = normalizer.inverse_transform(
        prediction.future_nodes.reshape(pose_shape)[:, mouse_index : mouse_index + 1]
    )
    target = normalizer.inverse_transform(
        window.future_keypoints[:, mouse_index : mouse_index + 1]
    )
    initial_pose = normalizer.inverse_transform(
        window.observed_keypoints[-1, mouse_index : mouse_index + 1]
    )
    return compute_pixel_metrics(
        predicted,
        target,
        initial_pose=initial_pose,
        seconds_per_step=seconds_per_step,
    ).to_numpy_dict()


def _append_horizon_values(
    destination: dict[int, dict[str, list[Any]]],
    *,
    prediction: CasePrediction,
    window: Window,
    normalizer: PoseNormalizer,
    mouse_index: int,
    seconds_per_step: float,
) -> None:
    """Append cumulative horizon metrics for one case prediction."""

    pose_shape = window.future_keypoints.shape
    predicted = normalizer.inverse_transform(
        prediction.future_nodes.reshape(pose_shape)[:, mouse_index : mouse_index + 1]
    )
    target = normalizer.inverse_transform(
        window.future_keypoints[:, mouse_index : mouse_index + 1]
    )
    initial_pose = normalizer.inverse_transform(
        window.observed_keypoints[-1, mouse_index : mouse_index + 1]
    )
    for horizon in range(1, predicted.shape[0] + 1):
        metrics = compute_pixel_metrics(
            predicted[:horizon],
            target[:horizon],
            initial_pose=initial_pose,
            seconds_per_step=seconds_per_step,
        ).to_numpy_dict()
        _append_values(destination[horizon], metrics)


def _calibration_metrics(
    *,
    prediction: CasePrediction,
    window: Window,
    normalizer: PoseNormalizer,
    mouse_index: int,
) -> dict[str, float]:
    """Summarize Gaussian calibration for one matched mouse case.

    The coverage metrics use standard chi-square thresholds for a 2D Gaussian:
    roughly half and 95 percent of calibrated samples should fall inside those
    ellipses.
    """

    pose_shape = window.future_keypoints.shape
    outputs = prediction.gaussian_outputs.reshape((*pose_shape[:-1], 5))[
        :, mouse_index : mouse_index + 1
    ]
    target = window.future_keypoints[:, mouse_index : mouse_index + 1]
    params = gaussian_2d_parameters(torch.from_numpy(outputs.astype(np.float32)))
    dx = torch.from_numpy(target[..., 0].astype(np.float32)) - params.mu_x
    dy = torch.from_numpy(target[..., 1].astype(np.float32)) - params.mu_y
    one_minus_rho_sq = torch.clamp(1.0 - params.rho.square(), min=1e-6)
    mahalanobis_sq = (
        (dx / params.sigma_x).square()
        + (dy / params.sigma_y).square()
        - (2.0 * params.rho * dx * dy / (params.sigma_x * params.sigma_y))
    ) / one_minus_rho_sq
    sigma_scale = torch.as_tensor(normalizer.scale, dtype=torch.float32)
    return {
        "sigma_x_px": float((params.sigma_x * sigma_scale[0]).mean().item()),
        "sigma_y_px": float((params.sigma_y * sigma_scale[1]).mean().item()),
        "mahalanobis_sq": float(mahalanobis_sq.mean().item()),
        "coverage_50": float((mahalanobis_sq <= 1.38629436).float().mean().item()),
        "coverage_95": float((mahalanobis_sq <= 5.99146455).float().mean().item()),
    }


def _attention_metrics(
    rollout: RolloutResult | None,
    *,
    mouse_index: int,
) -> dict[str, float]:
    """Measure dense-triplet attention mass within and across mice."""

    if rollout is None:
        return {}

    within = []
    cross = []
    entropy = []
    for frame_attention in rollout.attention_weights:
        for source_node, payload in frame_attention.items():
            if source_node // 12 != mouse_index:
                continue
            weights, target_nodes = payload
            values = weights.detach().cpu().numpy().astype(np.float32)
            same_mouse = np.asarray(
                [target // 12 == mouse_index for target in target_nodes],
                dtype=bool,
            )
            within.append(float(values[same_mouse].sum()))
            cross.append(float(values[~same_mouse].sum()))
            entropy.append(float(-(values * np.log(values + 1e-12)).sum()))
    return {
        "dense_within_mouse_attention_mass": float(np.mean(within)),
        "dense_cross_mouse_attention_mass": float(np.mean(cross)),
        "dense_attention_entropy": float(np.mean(entropy)),
    }


def _load_model(
    *, checkpoint_path: Path, device: torch.device
) -> FlatSocialAttentionModel:
    """Load a flat Social Attention checkpoint for inference."""

    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = FlatSocialAttentionModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _append_values(destination: dict[str, list[Any]], metrics: dict[str, Any]) -> None:
    """Append scalar and array metric values into an accumulator."""

    for name, value in metrics.items():
        destination.setdefault(name, []).append(value)


def _append_float_values(
    destination: dict[str, list[float]], metrics: dict[str, float]
) -> None:
    """Append scalar diagnostics into an accumulator."""

    for name, value in metrics.items():
        destination.setdefault(name, []).append(value)


def _case_seed(seed: int, case_index: int, stream: int) -> int:
    """Build a deterministic seed for one stochastic diagnostic stream."""

    return int(seed + 1009 * case_index + 9176 * stream)


def _case_row(
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


def _diagnostic_record(
    *,
    config: PoseDiagnosticConfig,
    dense_config: FlatFitConfig,
    single_config: FlatFitConfig,
    data: MatchedWindowData,
    metric_values: dict[str, dict[str, dict[str, list[Any]]]],
    metric_values_by_motion: dict[str, dict[str, dict[str, dict[str, list[Any]]]]],
    horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]],
    calibration_values: dict[str, dict[str, list[float]]],
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
            "dense_checkpoint_path": str(config.dense_triplet.checkpoint_path),
            "single_checkpoint_path": str(config.single_mouse.checkpoint_path),
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
            "window_digest": _matched_window_digest(data.cases),
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
        "horizon_summary": _horizon_summary(horizon_values),
        "calibration": {
            model_name: _aggregate_float_values(values)
            for model_name, values in calibration_values.items()
        },
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


def _horizon_summary(
    horizon_values: dict[str, dict[str, dict[int, dict[str, list[Any]]]]],
) -> dict[str, Any]:
    """Aggregate horizon-wise metrics for compact JSON output."""

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


def _aggregate_float_values(values: dict[str, list[float]]) -> dict[str, float]:
    """Average scalar diagnostic accumulators."""

    return {
        name: float(np.asarray(items, dtype=np.float32).mean())
        for name, items in values.items()
        if items
    }


def _matched_window_digest(cases: tuple[MatchedMouseWindow, ...]) -> str:
    """Fingerprint the paired case selection."""

    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.sequence_id.encode())
        digest.update(b"\0")
        digest.update(case.start_frame.to_bytes(8, byteorder="big", signed=False))
        digest.update(case.mouse_index.to_bytes(2, byteorder="big", signed=False))
    return digest.hexdigest()


def _diagnostic_overview_table(record: dict[str, Any]) -> Table:
    """Build a compact diagnostic lineage table."""

    contract = record["window_contract"]
    table = Table(title="Pose diagnostic contract", box=box.SIMPLE_HEAVY)
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", no_wrap=True)
    for name in (
        "effective_fps",
        "observation_seconds",
        "prediction_seconds",
        "triplet_windows",
        "mouse_cases",
        "seeds",
    ):
        table.add_row(name, str(contract[name]))
    return table


def _diagnostic_metric_table(record: dict[str, Any], *, mode: str) -> Table:
    """Build dense versus single metric table for one rollout mode."""

    table = Table(title=f"{mode} matched metrics", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    for model_name in ("dense_triplet", "single_mouse"):
        table.add_column(model_name, justify="right")
    for metric_name in PRIMARY_METRIC_NAMES:
        values = []
        has_value = False
        for model_name in ("dense_triplet", "single_mouse"):
            method = record["methods"][model_name][mode]
            mean = method["metrics"].get(metric_name)
            quantiles = method["metric_quantiles"].get(metric_name, {})
            cell = _mean_quantile_cell(mean, quantiles)
            values.append(cell or "-")
            has_value = has_value or cell is not None
        if has_value:
            table.add_row(metric_name, *values)
    return table


def _diagnostic_delta_table(record: dict[str, Any]) -> Table:
    """Build single-minus-dense paired delta table for autoregressive metrics."""

    table = Table(
        title="autoregressive paired delta: single_mouse - dense_triplet",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("mean", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p99", justify="right")
    table.add_column("single better", justify="right")
    for metric_name, values in record["paired_delta_single_minus_dense"][
        "autoregressive"
    ].items():
        table.add_row(
            metric_name,
            f"{values['mean']:.3f}",
            f"{values['p50']:.3f}",
            f"{values['p99']:.3f}",
            f"{values['single_better_rate']:.1%}",
        )
    return table


def _diagnostic_motion_table(record: dict[str, Any]) -> Table:
    """Build an autoregressive pose metrics table grouped by motion stratum."""

    table = Table(title="autoregressive pose metrics by motion", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("motion", no_wrap=True)
    table.add_column("dense_triplet", justify="right")
    table.add_column("single_mouse", justify="right")
    for metric_name in (
        "body_heading_error_deg",
        "bone_length_error_px",
        "skeleton_orientation_error_deg",
        "relative_ordering_error",
    ):
        for stratum in MOTION_STRATA:
            dense = (
                record["methods"]["dense_triplet"]["autoregressive"][
                    "metrics_by_motion"
                ]
                .get(stratum, {})
                .get(metric_name)
            )
            single = (
                record["methods"]["single_mouse"]["autoregressive"]["metrics_by_motion"]
                .get(stratum, {})
                .get(metric_name)
            )
            table.add_row(
                metric_name, stratum, _format_optional(dense), _format_optional(single)
            )
    return table


def _diagnostic_calibration_table(record: dict[str, Any]) -> Table:
    """Build Gaussian calibration summary table."""

    table = Table(title="autoregressive Gaussian calibration", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("dense_triplet", justify="right")
    table.add_column("single_mouse", justify="right")
    metrics = sorted(
        set(record["calibration"]["dense_triplet"])
        | set(record["calibration"]["single_mouse"])
    )
    for metric_name in metrics:
        table.add_row(
            metric_name,
            _format_optional(record["calibration"]["dense_triplet"].get(metric_name)),
            _format_optional(record["calibration"]["single_mouse"].get(metric_name)),
        )
    return table


def _diagnostic_attention_table(record: dict[str, Any]) -> Table:
    """Build dense-triplet attention summary table."""

    table = Table(title="dense-triplet attention summary", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    for name, value in record["attention"].items():
        table.add_row(name, _format_optional(value))
    return table


def _mean_quantile_cell(value: Any, quantiles: dict[str, float]) -> str | None:
    """Format a mean, p50, and p99 cell."""

    if not isinstance(value, int | float) or not np.isfinite(value):
        return None
    parts = [f"mean={value:.3f}"]
    for name in ("p50", "p99"):
        quantile = quantiles.get(name)
        if isinstance(quantile, int | float) and np.isfinite(quantile):
            parts.append(f"{name}={quantile:.3f}")
    return " ".join(parts)


def _format_optional(value: Any) -> str:
    """Format an optional scalar table value."""

    if not isinstance(value, int | float) or not np.isfinite(value):
        return "-"
    return f"{value:.3f}"


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
    args = parser.parse_args()
    config = load_pose_diagnostic_config(args.config)
    record = run_pose_diagnostics(config)
    print_pose_diagnostics(record)
    LOGGER.info("result=%s", config.results_path)
    if config.case_results_path is not None:
        LOGGER.info("case_results=%s", config.case_results_path)


if __name__ == "__main__":
    main()
