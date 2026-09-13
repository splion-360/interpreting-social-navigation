"""File description: Autoregressive evaluation for trajectory checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from torch import Tensor
from tqdm.auto import tqdm

from baseline import BaselineName, predict_motion_baseline, valid_baseline_names
from constants import (
    COORDINATES,
    DEFAULT_SOURCE_FPS,
    KEYPOINT_NAMES,
    MOUSE_SKELETON_EDGES,
    NUM_KEYPOINTS,
    NUM_MICE,
)
from data import (
    MabeDataset,
    MabeWindowDataset,
    PoseNormalizer,
    Window,
    WindowSpec,
    split_sequence_ids,
)
from loss import gaussian_2d_parameters
from metric import PRIMARY_METRIC_NAMES, compute_pixel_metrics, ground_truth_motion_px
from models import FlatSocialAttentionModel
from st_graph import GraphSequence
from train import (
    FlatFitConfig,
    _graph_builder,
    _select_device,
    load_flat_fit_config,
)


KEYPOINT_GRAPH_VARIANTS = {"dense_keypoint", "flat_sparse_keypoint"}
DEFAULT_TEST_CONFIG_PATH = Path("src/config/test.yml")
DEFAULT_BASELINE_CONFIG_PATH = Path("src/config/motion_baselines_pred12__evaluate.yml")
DEFAULT_RESULTS_ROOT = Path("outputs/evaluations")
DEFAULT_RESULTS_PATH = DEFAULT_RESULTS_ROOT / "dense_keypoint" / "results.jsonl"
TABLE_METRIC_NAMES = (
    "centroid_offset_px_by_mouse",
    "body_heading_error_deg_by_mouse",
    "body_frame_keypoint_error_px_by_mouse",
    "relative_ordering_error_by_mouse_axis",
    "edge_angle_error_deg_by_mouse",
    "edge_bone_length_error_px_by_mouse",
)
MOTION_STRATA = ("low", "medium", "high")
MotionStratum = Literal["low", "medium", "high"]
SKELETON_EDGE_CELLS = frozenset(
    (start, end)
    for edge in MOUSE_SKELETON_EDGES
    for start, end in (edge, (edge[1], edge[0]))
)
animate_prediction_comparison: Any | None = None
animate_single_mouse_prediction_comparison: Any | None = None
save_animation: Any | None = None


@dataclass(frozen=True)
class TestConfig:
    """Configuration for held-out test evaluation.

    Attributes:
        data_path: Path to the held-out MABe test file.
        results_path: Optional local JSONL ledger for evaluation records.
        max_windows: Optional cap on test windows.
        seed: Sampling seed.
        wandb: Whether to log evaluation metrics to W&B.
        wandb_project: W&B project for evaluation logging.
        wandb_run_name: Optional W&B run name for evaluation logging.
    """

    data_path: Path = Path("data/MaBe/mouse_triplet_test.npy")
    results_path: Path | None = None
    max_windows: int | None = 100
    seed: int = 42
    wandb: bool = False
    wandb_project: str = "interpreting-social-navigation"
    wandb_run_name: str | None = None


@dataclass(frozen=True)
class MotionBaselineConfig:
    """Configuration for deterministic motion-baseline evaluation.

    Attributes:
        baselines: Baseline strategies to evaluate.
        observation_length: Number of conditioning frames.
        prediction_length: Number of future frames.
        comparison_prediction_length: Longest future horizon used to align starts.
        stride: Raw-frame gap between consecutive evaluation window starts.
        frame_step: Raw-frame gap between sampled frames inside one window.
        source_fps: Source dataset frame rate before temporal downsampling.
        max_windows: Maximum number of held-out windows to evaluate.
        results_path: Local JSONL ledger for baseline evaluation records.
        seed: Sampling seed recorded for lineage.
        wandb: Whether to log baseline metrics to W&B.
        wandb_project: W&B project for baseline evaluation logging.
        wandb_run_name: Optional W&B run name prefix for baseline logging.
    """

    baselines: tuple[BaselineName, ...] = valid_baseline_names()
    observation_length: int = 8
    prediction_length: int = 12
    comparison_prediction_length: int = 60
    stride: int = 20
    frame_step: int = 1
    source_fps: float = DEFAULT_SOURCE_FPS
    max_windows: int | None = 1000
    results_path: Path = (
        DEFAULT_RESULTS_ROOT / "motion_baselines" / "pred12" / "results.jsonl"
    )
    seed: int = 42
    wandb: bool = False
    wandb_project: str = "interpreting-social-navigation"
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        """Validate timing values used for baseline windows and metric units."""

        if self.frame_step < 1:
            raise ValueError("frame_step must be at least 1")
        if self.source_fps <= 0:
            raise ValueError("source_fps must be positive")


@dataclass(frozen=True)
class MotionStratification:
    """Ground-truth motion strata for a fixed collection of windows.

    Attributes:
        labels: Per-window low-, medium-, or high-motion labels.
        low_max_px: Upper tertile boundary for low-motion windows.
        medium_max_px: Upper tertile boundary for medium-motion windows.
    """

    labels: tuple[MotionStratum, ...]
    low_max_px: float
    medium_max_px: float

    @property
    def counts(self) -> dict[str, int]:
        """Return the number of windows assigned to each stratum."""

        return {name: self.labels.count(name) for name in MOTION_STRATA}

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready motion-profile metadata."""

        return {
            "score": "mean_mouse_centroid_displacement_px",
            "thresholds_px": {
                "low_max": self.low_max_px,
                "medium_max": self.medium_max_px,
            },
            "counts": self.counts,
        }


def stratify_motion_scores(scores_px: np.ndarray) -> MotionStratification:
    """Partition window motion scores at their lower and upper tertiles.

    Args:
        scores_px: One ground-truth centroid-displacement score per window.

    Returns:
        Deterministic labels and the pixel thresholds used to create them.
    """

    low_max, medium_max = np.quantile(scores_px, (1.0 / 3.0, 2.0 / 3.0))
    labels: list[MotionStratum] = []
    for score in scores_px:
        if score <= low_max:
            labels.append("low")
        elif score <= medium_max:
            labels.append("medium")
        else:
            labels.append("high")
    return MotionStratification(
        labels=tuple(labels),
        low_max_px=float(low_max),
        medium_max_px=float(medium_max),
    )


def resolve_baseline_window_config(
    train_config: FlatFitConfig,
    baseline_config: MotionBaselineConfig,
) -> FlatFitConfig:
    """Apply a baseline experiment's window contract to the data configuration.

    Args:
        train_config: Base configuration providing data and normalization settings.
        baseline_config: Baseline-specific observation, prediction, and stride values.

    Returns:
        Copy of the base configuration with the requested window contract.
    """

    return replace(
        train_config,
        window_length=(
            baseline_config.observation_length + baseline_config.prediction_length
        ),
        observation_length=baseline_config.observation_length,
        prediction_length=baseline_config.prediction_length,
        stride=baseline_config.stride,
        frame_step=baseline_config.frame_step,
        source_fps=baseline_config.source_fps,
    )


@dataclass(frozen=True)
class RolloutResult:
    """Autoregressive trajectory rollout for one window.

    Attributes:
        nodes: Predicted node coordinates shaped `[time, nodes, 2]`.
        gaussian_outputs: Raw Gaussian parameters for predicted future frames.
        attention_weights: Per-frame attention weights from prediction steps.
    """

    nodes: np.ndarray
    gaussian_outputs: np.ndarray
    attention_weights: tuple[dict[int, tuple[Tensor, tuple[int, ...]]], ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate trajectory metrics for a checkpoint.

    Attributes:
        split: Evaluation split name.
        windows: Number of windows evaluated.
        metrics: Pixel-space evaluation metrics.
        device: Device used for model inference.
        checkpoint_epoch: Epoch stored in the evaluated checkpoint, if available.
        checkpoint_validation_loss: Validation loss stored in the checkpoint.
        motion_profile: Ground-truth motion thresholds and window counts.
        metrics_by_motion: Aggregate metrics for each motion stratum.
        window_digest: Fingerprint of selected sequence/start-frame keys.
    """

    split: str
    windows: int
    metrics: dict[str, Any]
    device: str
    checkpoint_epoch: int | None
    checkpoint_validation_loss: float | None
    motion_profile: dict[str, Any] | None = None
    metrics_by_motion: dict[str, dict[str, Any]] | None = None
    window_digest: str | None = None


@dataclass(frozen=True)
class BaselineEvaluationResult:
    """Aggregate metrics for one deterministic motion baseline.

    Attributes:
        baseline: Baseline strategy name.
        split: Evaluation split name.
        windows: Number of windows evaluated.
        metrics: Pixel-space evaluation metrics.
        motion_profile: Ground-truth motion thresholds and window counts.
        metrics_by_motion: Aggregate metrics for each motion stratum.
        evaluation_seconds: Wall-clock time spent predicting and scoring windows.
        window_digest: Fingerprint of selected sequence/start-frame keys.
    """

    baseline: BaselineName
    split: str
    windows: int
    metrics: dict[str, Any]
    motion_profile: dict[str, Any]
    metrics_by_motion: dict[str, dict[str, Any]]
    evaluation_seconds: float = 0.0
    window_digest: str | None = None


@dataclass(frozen=True)
class PredictionVideoResult:
    """Saved prediction-comparison video metadata.

    Attributes:
        path: MP4 output path.
        sequence_id: Source MABe sequence ID.
        start_frame: Start frame for the visualized window.
        graph_variant: Graph variant used for inference.
    """

    path: Path
    sequence_id: str
    start_frame: int
    graph_variant: str


def sample_bivariate_gaussian(
    outputs: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample 2D positions from raw bivariate Gaussian model outputs.

    Args:
        outputs: Raw Gaussian parameters shaped `[..., 5]`.
        generator: Optional random generator for reproducible sampling.

    Returns:
        Sampled coordinates shaped `[..., 2]`.
    """

    params = gaussian_2d_parameters(outputs)
    eps_x = torch.randn(
        params.mu_x.shape,
        generator=generator,
        device=outputs.device,
        dtype=outputs.dtype,
    )
    eps_y = torch.randn(
        params.mu_y.shape,
        generator=generator,
        device=outputs.device,
        dtype=outputs.dtype,
    )
    one_minus_rho_sq = torch.clamp(1 - params.rho.square(), min=1e-6)
    x = params.mu_x + params.sigma_x * eps_x
    y = params.mu_y + params.sigma_y * (
        params.rho * eps_x + torch.sqrt(one_minus_rho_sq) * eps_y
    )
    return torch.stack((x, y), dim=-1)


def rollout_flat_keypoint_model(
    *,
    model: FlatSocialAttentionModel,
    observed_keypoints: np.ndarray,
    prediction_length: int,
    build_graph: Callable[[np.ndarray], GraphSequence],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> RolloutResult:
    """Roll a flat keypoint model forward from observed frames.

    Args:
        model: Trained flat Social Attention model.
        observed_keypoints: Observed keypoints shaped `[time, 3, 12, 2]`.
        prediction_length: Number of future frames to generate.
        build_graph: Graph builder matching the checkpoint/config variant.
        device: Inference device.
        generator: Optional random generator for reproducible sampling.

    Returns:
        Predicted full sequence containing observed and generated nodes.
    """

    observed = observed_keypoints.astype(np.float32)
    total_length = observed.shape[0] + prediction_length
    rollout_keypoints = np.zeros(
        (total_length, NUM_MICE, NUM_KEYPOINTS, COORDINATES), dtype=np.float32
    )
    rollout_keypoints[: observed.shape[0]] = observed
    state = None
    attention: list[dict[int, tuple[Tensor, tuple[int, ...]]]] = []
    gaussian_outputs: list[np.ndarray] = []

    if observed.shape[0] > 1:
        warm_graph = build_graph(rollout_keypoints[: observed.shape[0] - 1])
        with torch.no_grad():
            warm_result = model.forward_with_state(
                nodes=torch.from_numpy(warm_graph.nodes).to(device),
                edge_features=torch.from_numpy(warm_graph.edge_features).to(device),
                edge_specs=warm_graph.edge_specs,
                nodes_present=warm_graph.nodes_present,
                edges_present=warm_graph.edges_present,
                state=None,
            )
        state = warm_result.state

    for step_idx in range(prediction_length):
        current_frame = observed.shape[0] - 1 + step_idx
        graph = build_graph(rollout_keypoints[: current_frame + 1])
        with torch.no_grad():
            result = model.forward_with_state(
                nodes=torch.from_numpy(
                    graph.nodes[current_frame : current_frame + 1]
                ).to(device),
                edge_features=torch.from_numpy(
                    graph.edge_features[current_frame : current_frame + 1]
                ).to(device),
                edge_specs=graph.edge_specs,
                nodes_present=(graph.nodes_present[current_frame],),
                edges_present=(graph.edges_present[current_frame],),
                state=state,
            )
        state = result.state
        output = result.outputs[0]
        next_nodes = sample_bivariate_gaussian(output, generator=generator)
        gaussian_outputs.append(output.detach().cpu().numpy())
        attention.extend(result.attention_weights)
        rollout_keypoints[current_frame + 1] = _flat_nodes_to_keypoints(
            next_nodes.detach().cpu().numpy()
        )

    rollout_graph = build_graph(rollout_keypoints)
    return RolloutResult(
        nodes=rollout_graph.nodes,
        gaussian_outputs=np.stack(gaussian_outputs, axis=0),
        attention_weights=tuple(attention),
    )


def evaluate_flat_checkpoint(
    *,
    config: FlatFitConfig,
    checkpoint_path: Path,
    split: Literal["validation", "test"] = "validation",
    test_data_path: Path | None = None,
    max_windows: int | None = None,
    seed: int | None = None,
    show_progress: bool = True,
) -> EvaluationResult:
    """Evaluate a flat keypoint checkpoint with autoregressive rollouts.

    Args:
        config: Training/evaluation configuration.
        checkpoint_path: Local checkpoint containing model weights.
        split: Evaluation split, either validation from train data or test data.
        test_data_path: Held-out test file used only when `split` is `test`.
        max_windows: Optional cap on evaluated windows.
        seed: Optional random seed for reproducible sampling.
        show_progress: Whether to print checkpoint and window progress.

    Returns:
        Mean ADE/FDE over selected validation windows.
    """

    if config.graph_variant not in KEYPOINT_GRAPH_VARIANTS:
        raise ValueError("autoregressive keypoint evaluation requires a keypoint graph")

    device = _select_device(config.device)
    windows, normalizer = _build_evaluation_data(
        config=config,
        split=split,
        test_data_path=test_data_path,
        max_windows=max_windows,
    )
    build_graph = _graph_builder(config.graph_variant)
    model = FlatSocialAttentionModel().to(device)
    if show_progress:
        print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if show_progress:
        print(f"Loaded checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    metric_values: dict[str, list[Any]] = {}
    motion_profile = _build_motion_profile(windows, normalizer)
    metrics_by_motion: dict[str, dict[str, list[Any]]] = {
        name: {} for name in MOTION_STRATA
    }
    with torch.no_grad():
        for window_index in tqdm(range(len(windows)), desc="Evaluating on test data"):
            window = windows[window_index]
            rollout = rollout_flat_keypoint_model(
                model=model,
                observed_keypoints=window.observed_keypoints,
                prediction_length=windows.spec.prediction_length,
                build_graph=build_graph,
                device=device,
                generator=generator,
            )
            metrics = _window_pixel_metrics(
                rollout=rollout,
                window=window,
                normalizer=normalizer,
                prediction_length=windows.spec.prediction_length,
                seconds_per_step=_seconds_per_step(config),
            )
            _append_metric_values(metric_values, metrics)
            _append_metric_values(
                metrics_by_motion[motion_profile.labels[window_index]],
                metrics,
            )

    return EvaluationResult(
        split=split,
        windows=len(windows),
        metrics=_aggregate_metric_values(metric_values),
        device=str(device),
        checkpoint_epoch=checkpoint.get("epoch"),
        checkpoint_validation_loss=checkpoint.get("validation_loss"),
        motion_profile=motion_profile.to_dict(),
        metrics_by_motion=_aggregate_metrics_by_motion(metrics_by_motion),
        window_digest=_window_selection_digest(windows),
    )


def evaluate_motion_baseline(
    *,
    train_config: FlatFitConfig,
    baseline: BaselineName,
    split: Literal["validation", "test"] = "test",
    test_data_path: Path | None = None,
    max_windows: int | None = None,
    show_progress: bool = True,
    index_window_length: int | None = None,
) -> BaselineEvaluationResult:
    """Evaluate one deterministic baseline on validation or held-out test windows.

    Args:
        train_config: Training configuration defining window shape and normalizer.
        baseline: Baseline strategy name.
        split: Evaluation split, either validation from train data or test data.
        test_data_path: Held-out test file used only when `split` is `test`.
        max_windows: Optional cap on evaluated windows.
        show_progress: Whether to print window progress.
        index_window_length: Optional longest comparison window used for indexing.

    Returns:
        Aggregated pixel-space metrics for the baseline.
    """

    windows, normalizer = _build_evaluation_data(
        config=train_config,
        split=split,
        test_data_path=test_data_path,
        max_windows=max_windows,
        index_window_length=index_window_length,
    )
    metric_values: dict[str, list[Any]] = {}
    motion_profile = _build_motion_profile(windows, normalizer)
    metrics_by_motion: dict[str, dict[str, list[Any]]] = {
        name: {} for name in MOTION_STRATA
    }
    iterator = tqdm(
        range(len(windows)),
        desc=f"Evaluating {baseline}",
        disable=not show_progress,
    )
    evaluation_started = perf_counter()
    for window_index in iterator:
        window = windows[window_index]
        observed_keypoints = normalizer.inverse_transform(window.observed_keypoints)
        target_future = normalizer.inverse_transform(window.future_keypoints)
        prediction = predict_motion_baseline(
            baseline,
            observed_keypoints=observed_keypoints,
            prediction_length=windows.spec.prediction_length,
        )
        metrics = compute_pixel_metrics(
            prediction.future_keypoints,
            target_future,
            initial_pose=observed_keypoints[-1],
            seconds_per_step=_seconds_per_step(train_config),
        ).to_numpy_dict()
        _append_metric_values(metric_values, metrics)
        _append_metric_values(
            metrics_by_motion[motion_profile.labels[window_index]],
            metrics,
        )

    return BaselineEvaluationResult(
        baseline=baseline,
        split=split,
        windows=len(windows),
        metrics=_aggregate_metric_values(metric_values),
        motion_profile=motion_profile.to_dict(),
        metrics_by_motion=_aggregate_metrics_by_motion(metrics_by_motion),
        evaluation_seconds=perf_counter() - evaluation_started,
        window_digest=_window_selection_digest(windows),
    )


def evaluate_motion_baselines(
    *,
    train_config: FlatFitConfig,
    baseline_config: MotionBaselineConfig,
    split: Literal["validation", "test"] = "test",
    test_data_path: Path | None = None,
    max_windows: int | None = None,
    show_progress: bool = True,
) -> list[BaselineEvaluationResult]:
    """Evaluate every configured deterministic motion baseline.

    Args:
        train_config: Training configuration defining window shape and normalizer.
        baseline_config: Baseline strategies and logging defaults.
        split: Evaluation split, either validation or held-out test.
        test_data_path: Held-out test file used only when `split` is `test`.
        max_windows: Optional cap on evaluated windows.
        show_progress: Whether to print window progress.

    Returns:
        Aggregate result for each configured baseline.
    """

    return [
        evaluate_motion_baseline(
            train_config=train_config,
            baseline=baseline,
            split=split,
            test_data_path=test_data_path,
            max_windows=max_windows,
            show_progress=show_progress,
            index_window_length=(
                baseline_config.observation_length
                + baseline_config.comparison_prediction_length
            ),
        )
        for baseline in baseline_config.baselines
    ]


def save_test_prediction_video(
    *,
    config: FlatFitConfig,
    test_config: TestConfig,
    checkpoint_path: Path,
    sequence_id: str | None = None,
    window_index: int = 0,
    output_root: Path = Path("outputs/visualizations"),
    fps: int = 8,
    interval_ms: int = 120,
    seed: int | None = None,
) -> PredictionVideoResult:
    """Run test-set inference for one window and save a comparison MP4.

    Args:
        config: Training configuration that defines the model and graph variant.
        test_config: Held-out test configuration with the test `.npy` path.
        checkpoint_path: Local checkpoint containing model weights.
        sequence_id: Optional MABe sequence ID. Defaults to the first test sequence.
        window_index: Deterministic window index within the selected sequence set.
        output_root: Base output folder. The graph variant and sequence ID are appended.
        fps: Frames per second for the saved MP4.
        interval_ms: Matplotlib animation interval.
        seed: Optional random seed for reproducible Gaussian sampling.

    Returns:
        Metadata for the saved video.
    """

    window, actual_keypoints, predicted_future = _sample_test_prediction(
        config=config,
        test_config=test_config,
        sequence_id=sequence_id,
        window_index=window_index,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )
    animate, save = _prediction_visualization_functions()
    animation_obj = animate(
        actual_keypoints=actual_keypoints,
        predicted_future_keypoints=predicted_future,
        observation_length=config.observation_length,
        sequence_id=window.sequence_id,
        step=1,
        interval_ms=interval_ms,
    )
    output_path = output_root / config.graph_variant / f"{window.sequence_id}.mp4"
    saved_path = save(animation_obj, output_path, fps=fps)
    return PredictionVideoResult(
        path=saved_path,
        sequence_id=window.sequence_id,
        start_frame=window.start_frame,
        graph_variant=config.graph_variant,
    )


def save_single_mouse_prediction_video(
    *,
    config: FlatFitConfig,
    test_config: TestConfig,
    checkpoint_path: Path,
    mouse_index: int,
    sequence_id: str | None = None,
    window_index: int = 0,
    output_root: Path = Path("outputs/visualizations"),
    fps: int = 8,
    interval_ms: int = 120,
    seed: int | None = None,
) -> PredictionVideoResult:
    """Run test-set inference and save a zoomed one-mouse comparison MP4.

    Args:
        config: Training configuration that defines the model and graph variant.
        test_config: Held-out test configuration with the test `.npy` path.
        checkpoint_path: Local checkpoint containing model weights.
        mouse_index: Mouse index to visualize, from `0` to `2`.
        sequence_id: Optional MABe sequence ID. Defaults to the first test sequence.
        window_index: Deterministic window index within the selected sequence set.
        output_root: Base output folder. The graph variant is appended.
        fps: Frames per second for the saved MP4.
        interval_ms: Matplotlib animation interval.
        seed: Optional random seed for reproducible Gaussian sampling.

    Returns:
        Metadata for the saved video.
    """

    if not 0 <= mouse_index < NUM_MICE:
        raise ValueError(f"mouse_index must be between 0 and {NUM_MICE - 1}")

    window, actual_keypoints, predicted_future = _sample_test_prediction(
        config=config,
        test_config=test_config,
        sequence_id=sequence_id,
        window_index=window_index,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )
    animate, save = _single_mouse_visualization_functions()
    animation_obj = animate(
        actual_keypoints=actual_keypoints,
        predicted_future_keypoints=predicted_future,
        observation_length=config.observation_length,
        mouse_index=mouse_index,
        sequence_id=window.sequence_id,
        step=1,
        interval_ms=interval_ms,
    )
    output_path = (
        output_root
        / config.graph_variant
        / f"{window.sequence_id}__mouse_{mouse_index}.mp4"
    )
    saved_path = save(animation_obj, output_path, fps=fps)
    return PredictionVideoResult(
        path=saved_path,
        sequence_id=window.sequence_id,
        start_frame=window.start_frame,
        graph_variant=config.graph_variant,
    )


def _sample_test_prediction(
    *,
    config: FlatFitConfig,
    test_config: TestConfig,
    sequence_id: str | None,
    window_index: int,
    checkpoint_path: Path,
    seed: int | None,
) -> tuple[Window, np.ndarray, np.ndarray]:
    """Sample one normalized test window and return pixel-space prediction arrays.

    Args:
        config: Training configuration that defines the model and graph variant.
        test_config: Held-out test configuration with the test `.npy` path.
        sequence_id: Optional MABe sequence ID.
        window_index: Deterministic window index within the selected sequence set.
        checkpoint_path: Local checkpoint containing model weights.
        seed: Optional random seed for reproducible Gaussian sampling.

    Returns:
        Source window, actual keypoints, and predicted future in pixel coordinates.
    """

    if config.graph_variant not in KEYPOINT_GRAPH_VARIANTS:
        raise ValueError("prediction videos require a keypoint graph")

    normalizer = _fit_training_normalizer(config)
    window = _test_window(
        config=config,
        test_config=test_config,
        normalizer=normalizer,
        sequence_id=sequence_id,
        window_index=window_index,
    )
    device = _select_device(config.device)
    build_graph = _graph_builder(config.graph_variant)
    model = FlatSocialAttentionModel().to(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generator = torch.Generator(device=device)
    resolved_seed = seed if seed is not None else test_config.seed
    if resolved_seed is not None:
        generator.manual_seed(resolved_seed)

    with torch.no_grad():
        rollout = rollout_flat_keypoint_model(
            model=model,
            observed_keypoints=window.observed_keypoints,
            prediction_length=config.prediction_length,
            build_graph=build_graph,
            device=device,
            generator=generator,
        )

    actual_keypoints = normalizer.inverse_transform(window.keypoints)
    predicted_future = normalizer.inverse_transform(
        rollout.nodes[config.observation_length :].reshape(
            config.prediction_length,
            NUM_MICE,
            NUM_KEYPOINTS,
            COORDINATES,
        )
    )
    return window, actual_keypoints, predicted_future


def _prediction_visualization_functions() -> tuple[Any, Any]:
    """Load triplet visualization functions when video generation is requested."""

    global animate_prediction_comparison, save_animation
    if animate_prediction_comparison is None or save_animation is None:
        from data.visualization import (
            animate_prediction_comparison as loaded_animate,
        )
        from data.visualization import save_animation as loaded_save

        animate_prediction_comparison = loaded_animate
        save_animation = loaded_save
    return animate_prediction_comparison, save_animation


def _single_mouse_visualization_functions() -> tuple[Any, Any]:
    """Load one-mouse visualization functions when video generation is requested."""

    global animate_single_mouse_prediction_comparison, save_animation
    if animate_single_mouse_prediction_comparison is None or save_animation is None:
        from data.visualization import (
            animate_single_mouse_prediction_comparison as loaded_animate,
        )
        from data.visualization import save_animation as loaded_save

        animate_single_mouse_prediction_comparison = loaded_animate
        save_animation = loaded_save
    return animate_single_mouse_prediction_comparison, save_animation


def _flat_nodes_to_keypoints(nodes: np.ndarray) -> np.ndarray:
    """Convert flat keypoint nodes back to `[3, 12, 2]` keypoints."""

    return nodes.reshape(NUM_MICE, NUM_KEYPOINTS, COORDINATES).astype(np.float32)


def _window_pixel_metrics(
    *,
    rollout: RolloutResult,
    window: Window,
    normalizer: PoseNormalizer,
    prediction_length: int,
    seconds_per_step: float,
) -> dict[str, Any]:
    """Compute pixel-space metrics for one evaluated window.

    Args:
        rollout: Model rollout over observed and predicted frames.
        window: Normalized source window with ground-truth future frames.
        normalizer: Pixel-coordinate normalizer used by the dataloader.
        prediction_length: Number of predicted future frames.
        seconds_per_step: Seconds represented by each sampled prediction step.

    Returns:
        Pixel-space trajectory, pose, and structure metrics.
    """

    predicted_future = normalizer.inverse_transform(
        rollout.nodes[-prediction_length:].reshape(
            prediction_length,
            NUM_MICE,
            NUM_KEYPOINTS,
            COORDINATES,
        )
    )
    initial_pose = normalizer.inverse_transform(window.observed_keypoints[-1])
    target_future = normalizer.inverse_transform(window.future_keypoints)
    return compute_pixel_metrics(
        predicted_future,
        target_future,
        initial_pose=initial_pose,
        seconds_per_step=seconds_per_step,
    ).to_numpy_dict()


def _build_motion_profile(
    windows: MabeWindowDataset,
    normalizer: PoseNormalizer,
) -> MotionStratification:
    """Build ground-truth motion strata for fixed evaluation windows.

    Args:
        windows: Selected normalized evaluation windows.
        normalizer: Transform used to restore pixel coordinates.

    Returns:
        Tertile-based motion labels derived without model predictions.
    """

    scores = []
    for window_index in range(len(windows)):
        window = windows[window_index]
        scores.append(
            ground_truth_motion_px(
                normalizer.inverse_transform(window.observed_keypoints[-1]),
                normalizer.inverse_transform(window.future_keypoints),
            )
        )
    return stratify_motion_scores(np.asarray(scores, dtype=np.float32))


def _seconds_per_step(config: FlatFitConfig) -> float:
    """Return seconds represented by one sampled trajectory step."""

    return config.frame_step / config.source_fps


def _window_selection_digest(windows: MabeWindowDataset) -> str:
    """Fingerprint selected sequence and start-frame keys in evaluation order."""

    digest = hashlib.sha256()
    for sequence_id, start_frame in windows.window_keys:
        digest.update(sequence_id.encode())
        digest.update(b"\0")
        digest.update(start_frame.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _append_metric_values(
    destination: dict[str, list[Any]],
    metrics: dict[str, float | np.ndarray],
) -> None:
    """Append one window's metrics to an aggregate accumulator."""

    for name, value in metrics.items():
        destination.setdefault(name, []).append(value)


def _aggregate_metrics_by_motion(
    values_by_motion: dict[str, dict[str, list[Any]]],
) -> dict[str, dict[str, Any]]:
    """Aggregate metric accumulators separately for each motion stratum."""

    return {
        stratum: _aggregate_metric_values(values) if values else {}
        for stratum, values in values_by_motion.items()
    }


def _aggregate_metric_values(metric_values: dict[str, list[Any]]) -> dict[str, Any]:
    """Average scalar and array metrics across evaluated windows.

    Args:
        metric_values: Per-window metric values keyed by metric name.

    Returns:
        JSON-ready aggregate metrics.
    """

    aggregates: dict[str, Any] = {}
    for name, values in metric_values.items():
        first_value = values[0]
        if isinstance(first_value, np.ndarray):
            aggregates[name] = _array_to_json_list(
                _nanmean_stacked([np.asarray(value) for value in values])
            )
        else:
            scalar_values = np.asarray(values, dtype=np.float32)
            finite_values = scalar_values[~np.isnan(scalar_values)]
            aggregates[name] = (
                float(finite_values.mean()) if finite_values.size else float("nan")
            )
    return aggregates


def _nanmean_stacked(values: list[np.ndarray]) -> np.ndarray:
    """Average same-shaped arrays while preserving all-`nan` positions."""

    stacked = np.stack(values, axis=0)
    valid = ~np.isnan(stacked)
    counts = valid.sum(axis=0)
    sums = np.where(valid, stacked, 0.0).sum(axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float32),
        where=counts > 0,
    )


def _array_to_json_list(value: np.ndarray) -> list[Any]:
    """Convert arrays with `nan` sentinels into JSON lists with `null`."""

    json_value = value.astype(object)
    json_value[np.isnan(value)] = None
    return json_value.tolist()


def print_evaluation_metrics(
    metrics: dict[str, Any],
    *,
    console: Console | None = None,
) -> None:
    """Print scalar metrics and diagnostic tables for the evaluation CLI.

    Args:
        metrics: JSON-ready metric dictionary returned by evaluation.
        console: Optional Rich console for tests or custom render settings.
    """

    output = console or Console()
    output.print(_primary_metrics_table(metrics))

    table_metrics = {
        name: metrics[name] for name in TABLE_METRIC_NAMES if name in metrics
    }
    if not table_metrics:
        return

    output.print()
    output.print(_keypoint_index_table())
    output.print()
    output.print(
        "[dim]Image offsets: positive x = right, positive y = down. "
        "Pairwise legend: blue cells = known MABe skeleton edges, "
        "green = lowest error pair, red = highest error pair, "
        "underlined = also a known skeleton edge.[/dim]"
    )

    centroid_offsets = table_metrics.get("centroid_offset_px_by_mouse")
    if centroid_offsets is not None:
        output.print()
        output.print("[bold]centroid_offset_px_by_mouse[/bold]")
        output.print(_centroid_offset_table(centroid_offsets))

    heading = table_metrics.get("body_heading_error_deg_by_mouse")
    if heading is not None:
        output.print()
        output.print("[bold]body_heading_error_deg_by_mouse[/bold]")
        output.print(_body_heading_table(heading))

    local_pose = table_metrics.get("body_frame_keypoint_error_px_by_mouse")
    if local_pose is not None:
        output.print()
        output.print("[bold]body_frame_keypoint_error_px_by_mouse[/bold]")
        output.print(_mouse_keypoint_table(local_pose, title="local pose error px"))

    ordering = table_metrics.get("relative_ordering_error_by_mouse_axis")
    if ordering is not None:
        output.print()
        output.print("[bold]relative_ordering_error_by_mouse_axis[/bold]")
        output.print(_relative_ordering_table(ordering))

    _print_matrix_tables(
        console=output,
        title="edge_angle_error_deg_by_mouse",
        values=table_metrics.get("edge_angle_error_deg_by_mouse"),
    )
    _print_matrix_tables(
        console=output,
        title="edge_bone_length_error_px_by_mouse",
        values=table_metrics.get("edge_bone_length_error_px_by_mouse"),
    )


def print_motion_stratified_metrics(
    *,
    metrics_by_motion: dict[str, dict[str, Any]],
    motion_profile: dict[str, Any],
    console: Console | None = None,
) -> None:
    """Print primary metrics grouped by ground-truth window motion.

    Args:
        metrics_by_motion: Aggregate metrics keyed by motion stratum.
        motion_profile: Score definition, thresholds, and stratum counts.
        console: Optional Rich console for tests or custom render settings.
    """

    output = console or Console()
    counts = motion_profile["counts"]
    thresholds = motion_profile["thresholds_px"]
    table = Table(
        title="Metrics by ground-truth motion stratum",
        caption=(
            f"low <= {thresholds['low_max']:.3f} px; "
            f"medium <= {thresholds['medium_max']:.3f} px; high above"
        ),
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("metric", style="cyan", no_wrap=True)
    for stratum in MOTION_STRATA:
        table.add_column(
            f"{stratum} (n={counts[stratum]})",
            justify="right",
            no_wrap=True,
        )
    for name in PRIMARY_METRIC_NAMES:
        values = [metrics_by_motion[stratum].get(name) for stratum in MOTION_STRATA]
        if not any(isinstance(value, int | float) for value in values):
            continue
        table.add_row(
            name,
            *(
                f"{value:.6f}" if isinstance(value, int | float) else "-"
                for value in values
            ),
        )
    output.print(table)


def _primary_metrics_table(metrics: dict[str, Any]) -> Table:
    """Build the primary scalar metrics table for CLI output."""

    table = Table(title="Primary evaluation metrics", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right", no_wrap=True)
    for name in PRIMARY_METRIC_NAMES:
        value = metrics.get(name)
        if isinstance(value, int | float):
            table.add_row(name, f"{value:.6f}")
    return table


def _keypoint_index_table() -> Table:
    """Build the keypoint-index legend table for CLI output."""

    table = Table(title="Keypoint indices", box=box.SIMPLE_HEAVY)
    table.add_column("index", justify="right", style="cyan", no_wrap=True)
    table.add_column("keypoint", style="white", no_wrap=True)
    for index, name in enumerate(KEYPOINT_NAMES):
        table.add_row(f"{index:02d}", name)
    return table


def _centroid_offset_table(values: Any) -> Table:
    """Build the per-mouse signed centroid offset table."""

    matrix = np.asarray(values, dtype=np.float32)
    table = Table(title="centroid_offset_px_by_mouse", box=box.SIMPLE_HEAVY)
    table.add_column("mouse", justify="right", style="cyan", no_wrap=True)
    table.add_column("x_px", justify="right", no_wrap=True)
    table.add_column("y_px", justify="right", no_wrap=True)
    for mouse_index, row in enumerate(matrix):
        table.add_row(str(mouse_index), *(_format_table_value(value) for value in row))
    return table


def _mouse_keypoint_table(values: Any, *, title: str) -> Table:
    """Build a compact per-mouse, per-keypoint metric table."""

    matrix = np.asarray(values, dtype=np.float32)
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("mouse", justify="right", style="cyan", no_wrap=True)
    for index in range(NUM_KEYPOINTS):
        table.add_column(f"{index:02d}", justify="right", no_wrap=True)
    for mouse_index, row in enumerate(matrix):
        table.add_row(
            str(mouse_index),
            *(_format_table_value(value) for value in row),
        )
    return table


def _relative_ordering_table(values: Any) -> Table:
    """Build the per-mouse body-frame ordering error table."""

    matrix = np.asarray(values, dtype=np.float32)
    table = Table(title="relative_ordering_error_by_mouse_axis", box=box.SIMPLE_HEAVY)
    table.add_column("mouse", justify="right", style="cyan", no_wrap=True)
    table.add_column("forward", justify="right", no_wrap=True)
    table.add_column("lateral", justify="right", no_wrap=True)
    for mouse_index, row in enumerate(matrix):
        table.add_row(str(mouse_index), *(_format_table_value(value) for value in row))
    return table


def _body_heading_table(values: Any) -> Table:
    """Build the per-mouse body heading error table."""

    table = Table(title="body_heading_error_deg_by_mouse", box=box.SIMPLE_HEAVY)
    table.add_column("mouse", justify="right", style="cyan", no_wrap=True)
    table.add_column("error_deg", justify="right", no_wrap=True)
    for mouse_index, value in enumerate(np.asarray(values, dtype=np.float32)):
        table.add_row(str(mouse_index), _format_table_value(value))
    return table


def _print_matrix_tables(*, console: Console, title: str, values: Any) -> None:
    """Print one dense keypoint-pair matrix per mouse when values are present."""

    if values is None:
        return

    matrices = np.asarray(values, dtype=np.float32)
    console.print()
    console.print(f"[bold]{title}[/bold]")
    for mouse_index, matrix in enumerate(matrices):
        console.print()
        console.print(_keypoint_matrix_table(matrix, title=f"mouse_{mouse_index}"))


def _keypoint_matrix_table(matrix: np.ndarray, *, title: str) -> Table:
    """Build one dense `[12, 12]` keypoint-pair matrix for CLI output."""

    min_cells, max_cells = _extreme_pair_cells(matrix)
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("kp", justify="right", style="cyan", no_wrap=True)
    for index in range(NUM_KEYPOINTS):
        table.add_column(f"{index:02d}", justify="right", no_wrap=True)
    for row_index, row in enumerate(matrix):
        table.add_row(
            f"{row_index:02d}",
            *(
                _format_matrix_cell(
                    value,
                    row_index=row_index,
                    column_index=column_index,
                    min_cells=min_cells,
                    max_cells=max_cells,
                )
                for column_index, value in enumerate(row)
            ),
        )
    return table


def _format_matrix_cell(
    value: Any,
    *,
    row_index: int,
    column_index: int,
    min_cells: frozenset[tuple[int, int]],
    max_cells: frozenset[tuple[int, int]],
) -> Text:
    """Format and style one keypoint-pair metric cell."""

    style = _matrix_cell_style(
        row_index=row_index,
        column_index=column_index,
        min_cells=min_cells,
        max_cells=max_cells,
    )
    return Text(_format_table_value(value), style=style)


def _matrix_cell_style(
    *,
    row_index: int,
    column_index: int,
    min_cells: frozenset[tuple[int, int]],
    max_cells: frozenset[tuple[int, int]],
) -> str:
    """Return the Rich style for one diagnostic matrix cell."""

    if row_index == column_index:
        return "dim"

    cell = (row_index, column_index)
    is_skeleton_edge = cell in SKELETON_EDGE_CELLS
    if cell in max_cells:
        return (
            "bold white on red underline" if is_skeleton_edge else "bold white on red"
        )
    if cell in min_cells:
        return (
            "bold black on green underline"
            if is_skeleton_edge
            else "bold black on green"
        )
    if is_skeleton_edge:
        return "black on sky_blue1"
    return ""


def _extreme_pair_cells(
    matrix: np.ndarray,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """Find symmetric lowest and highest off-diagonal cells in one matrix."""

    upper_mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1) & ~np.isnan(matrix)
    if not np.any(upper_mask):
        return frozenset(), frozenset()

    pair_indices = np.argwhere(upper_mask)
    pair_values = matrix[upper_mask]
    min_index = int(np.argmin(pair_values))
    max_index = int(np.argmax(pair_values))
    min_pair = (int(pair_indices[min_index, 0]), int(pair_indices[min_index, 1]))
    max_pair = (int(pair_indices[max_index, 0]), int(pair_indices[max_index, 1]))
    return _symmetric_cells(min_pair), _symmetric_cells(max_pair)


def _symmetric_cells(pair: tuple[int, int]) -> frozenset[tuple[int, int]]:
    """Return both directed cells for an unordered keypoint pair."""

    start, end = pair
    return frozenset(((start, end), (end, start)))


def _format_table_value(value: Any) -> str:
    """Format one metric table cell, using `.` for same-keypoint entries."""

    if value is None:
        return "."
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return "."
    return f"{numeric_value:.2f}"


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a trajectory checkpoint across script/module entry points.

    Args:
        path: Local `.pt` checkpoint file.
        device: Device used to map checkpoint tensors.

    Returns:
        Checkpoint dictionary.
    """

    main_module = cast(Any, sys.modules["__main__"])
    if not hasattr(main_module, "FlatFitConfig"):
        main_module.FlatFitConfig = FlatFitConfig
    return torch.load(path, map_location=device, weights_only=False)


def _build_evaluation_windows(
    *,
    config: FlatFitConfig,
    split: Literal["validation", "test"],
    test_data_path: Path | None,
    max_windows: int | None,
    index_window_length: int | None = None,
) -> MabeWindowDataset:
    """Build validation or test windows for checkpoint evaluation.

    Args:
        config: Training/evaluation configuration.
        split: Evaluation split name.
        test_data_path: Held-out test file used only when `split` is `test`.
        max_windows: Optional window cap for fast evaluation.
        index_window_length: Optional longest comparison window used for indexing.

    Returns:
        Window dataset for the requested split.
    """

    windows, _ = _build_evaluation_data(
        config=config,
        split=split,
        test_data_path=test_data_path,
        max_windows=max_windows,
        index_window_length=index_window_length,
    )
    return windows


def _build_evaluation_data(
    *,
    config: FlatFitConfig,
    split: Literal["validation", "test"],
    test_data_path: Path | None,
    max_windows: int | None,
    index_window_length: int | None = None,
) -> tuple[MabeWindowDataset, PoseNormalizer]:
    """Build normalized evaluation windows and their fitted normalizer.

    Args:
        config: Training/evaluation configuration.
        split: Evaluation split name.
        test_data_path: Held-out test file used only when `split` is `test`.
        max_windows: Optional window cap for fast evaluation.
        index_window_length: Optional longest comparison window used for indexing.

    Returns:
        Window dataset and train-split-fitted normalizer.
    """

    train_dataset = MabeDataset.from_file(config.data_path)
    train_ids, validation_ids = _training_validation_ids(config, train_dataset)
    normalizer = PoseNormalizer.fit(train_dataset.select(train_ids))
    spec = WindowSpec(
        length=config.window_length,
        observation_length=config.observation_length,
        prediction_length=config.prediction_length,
        stride=config.stride,
        frame_step=config.frame_step,
    )

    if split == "validation":
        return (
            MabeWindowDataset(
                train_dataset.select(validation_ids),
                spec,
                normalizer=normalizer,
                max_windows=max_windows or config.max_validation_windows,
                index_window_length=index_window_length,
            ),
            normalizer,
        )

    if test_data_path is None:
        raise ValueError("test split requires --test-data or --test-config")
    test_dataset = MabeDataset.from_file(test_data_path)
    return (
        MabeWindowDataset(
            test_dataset.select(test_dataset.sequence_ids),
            spec,
            normalizer=normalizer,
            max_windows=max_windows,
            index_window_length=index_window_length,
        ),
        normalizer,
    )


def _fit_training_normalizer(config: FlatFitConfig) -> PoseNormalizer:
    """Fit the pose normalizer using only training split sequences."""

    train_dataset = MabeDataset.from_file(config.data_path)
    train_ids, _ = _training_validation_ids(config, train_dataset)
    return PoseNormalizer.fit(train_dataset.select(train_ids))


def _training_validation_ids(
    config: FlatFitConfig,
    dataset: MabeDataset,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split training-file sequence IDs into train and validation IDs."""

    return split_sequence_ids(
        dataset.sequence_ids,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )


def _test_window(
    *,
    config: FlatFitConfig,
    test_config: TestConfig,
    normalizer: PoseNormalizer,
    sequence_id: str | None,
    window_index: int,
) -> Window:
    """Load one deterministic normalized window from the held-out test file."""

    test_dataset = MabeDataset.from_file(test_config.data_path)
    selected_ids = (
        [sequence_id] if sequence_id is not None else test_dataset.sequence_ids
    )
    windows = MabeWindowDataset(
        test_dataset.select(selected_ids),
        WindowSpec(
            length=config.window_length,
            observation_length=config.observation_length,
            prediction_length=config.prediction_length,
            stride=config.stride,
            frame_step=config.frame_step,
        ),
        normalizer=normalizer,
    )
    return windows[window_index]


def load_test_config(path: Path = DEFAULT_TEST_CONFIG_PATH) -> TestConfig:
    """Load held-out test evaluation configuration.

    Args:
        path: YAML configuration file.

    Returns:
        Fully typed held-out test configuration.
    """

    raw = yaml.safe_load(path.read_text()) or {}
    values = {
        "data_path": raw.get("data_path", TestConfig.data_path),
        "results_path": raw.get("results_path", TestConfig.results_path),
        "max_windows": raw.get("max_windows", TestConfig.max_windows),
        "seed": raw.get("seed", TestConfig.seed),
        "wandb": raw.get("wandb", TestConfig.wandb),
        "wandb_project": raw.get("wandb_project", TestConfig.wandb_project),
        "wandb_run_name": raw.get("wandb_run_name", TestConfig.wandb_run_name),
    }
    values["data_path"] = Path(values["data_path"])
    if values["results_path"] is not None:
        values["results_path"] = Path(values["results_path"])
    return TestConfig(**values)


def load_motion_baseline_config(
    path: Path = DEFAULT_BASELINE_CONFIG_PATH,
) -> MotionBaselineConfig:
    """Load deterministic baseline evaluation configuration.

    Args:
        path: YAML configuration file.

    Returns:
        Fully typed baseline evaluation configuration.
    """

    raw = yaml.safe_load(path.read_text()) or {}
    baselines = tuple(raw.get("baselines", MotionBaselineConfig.baselines))
    valid_names = set(valid_baseline_names())
    unknown = sorted(set(baselines) - valid_names)
    if unknown:
        raise ValueError(f"unknown baseline names: {unknown}")

    values = {
        "baselines": baselines,
        "observation_length": raw.get(
            "observation_length", MotionBaselineConfig.observation_length
        ),
        "prediction_length": raw.get(
            "prediction_length", MotionBaselineConfig.prediction_length
        ),
        "comparison_prediction_length": raw.get(
            "comparison_prediction_length",
            MotionBaselineConfig.comparison_prediction_length,
        ),
        "stride": raw.get("stride", MotionBaselineConfig.stride),
        "frame_step": raw.get("frame_step", MotionBaselineConfig.frame_step),
        "source_fps": raw.get(
            "source_fps",
            raw.get("frame_rate_hz", MotionBaselineConfig.source_fps),
        ),
        "max_windows": raw.get("max_windows", MotionBaselineConfig.max_windows),
        "results_path": raw.get("results_path", MotionBaselineConfig.results_path),
        "seed": raw.get("seed", MotionBaselineConfig.seed),
        "wandb": raw.get("wandb", MotionBaselineConfig.wandb),
        "wandb_project": raw.get("wandb_project", MotionBaselineConfig.wandb_project),
        "wandb_run_name": raw.get(
            "wandb_run_name", MotionBaselineConfig.wandb_run_name
        ),
    }
    values["results_path"] = Path(values["results_path"])
    return MotionBaselineConfig(**values)


def save_evaluation_record(record: dict[str, Any], path: Path) -> None:
    """Append one evaluation record to a local JSONL ledger.

    Args:
        record: JSON-serializable evaluation metadata and metrics.
        path: Destination `.jsonl` file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n"
        )


def _json_safe(value: Any) -> Any:
    """Replace non-finite numeric values recursively with JSON nulls."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return None
    return value


def default_model_results_path(train_config: FlatFitConfig) -> Path:
    """Return the default JSONL ledger path for a model evaluation.

    Args:
        train_config: Training configuration that identifies the graph variant.

    Returns:
        Source-specific path under the evaluation output directory.
    """

    return DEFAULT_RESULTS_ROOT / train_config.graph_variant / "results.jsonl"


def log_evaluation_to_wandb(
    *,
    record: dict[str, Any],
    project: str,
    run_name: str | None,
) -> None:
    """Log evaluation metrics and lineage metadata to W&B.

    Args:
        record: Evaluation record used as run config and metric source.
        project: W&B project name.
        run_name: Optional W&B run name.
    """

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requires the optional dependency: "
            'python -m pip install -e ".[wandb]"'
        ) from exc

    run = wandb.init(
        project=project,
        name=run_name,
        job_type="evaluate",
        config=record,
    )
    run.log(
        {
            **{
                f"eval/{name}": value
                for name, value in record["metrics"].items()
                if isinstance(value, int | float)
            },
            "eval/windows": record["windows"],
        }
    )
    run.finish()


def build_evaluation_record(
    *,
    result: EvaluationResult,
    train_config: FlatFitConfig,
    train_config_path: Path,
    checkpoint_path: Path,
    split: str,
    test_config_path: Path | None,
    test_data_path: Path | None,
    max_windows: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """Build a durable metadata record for one evaluation run.

    Args:
        result: Aggregate evaluation metrics.
        train_config: Training configuration used to build the model/data setup.
        train_config_path: Path to the training YAML.
        checkpoint_path: Evaluated checkpoint.
        split: Evaluation split name.
        test_config_path: Test YAML path when evaluating the held-out test split.
        test_data_path: Test file path when evaluating the held-out test split.
        max_windows: Window cap used for evaluation.
        seed: Sampling seed used for evaluation.

    Returns:
        JSON-serializable evaluation record.
    """

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "split": split,
        "train_config_path": str(train_config_path),
        "test_config_path": str(test_config_path) if test_config_path else None,
        "checkpoint_path": str(checkpoint_path),
        "test_data_path": str(test_data_path) if test_data_path else None,
        "windows": result.windows,
        "max_windows": max_windows,
        "sampling": "bivariate_gaussian",
        "seed": seed,
        "device": result.device,
        "model": {
            "family": train_config.model,
            "graph_variant": train_config.graph_variant,
            "window_length": train_config.window_length,
            "observation_length": train_config.observation_length,
            "prediction_length": train_config.prediction_length,
            "stride": train_config.stride,
            "frame_step": train_config.frame_step,
            "source_fps": train_config.source_fps,
            "effective_fps": train_config.source_fps / train_config.frame_step,
        },
        "checkpoint": {
            "epoch": result.checkpoint_epoch,
            "validation_loss": result.checkpoint_validation_loss,
        },
        "metrics": {
            **result.metrics,
        },
        "motion_profile": result.motion_profile,
        "metrics_by_motion": result.metrics_by_motion,
    }


def build_baseline_evaluation_record(
    *,
    result: BaselineEvaluationResult,
    train_config: FlatFitConfig,
    train_config_path: Path,
    baseline_config: MotionBaselineConfig,
    baseline_config_path: Path,
    split: str,
    test_config_path: Path | None,
    test_data_path: Path | None,
    max_windows: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """Build a durable metadata record for one baseline evaluation.

    Args:
        result: Aggregate baseline metrics.
        train_config: Training configuration used for windowing and normalization.
        train_config_path: Path to the training YAML.
        baseline_config: Baseline experiment and frame-rate configuration.
        baseline_config_path: Path to the baseline YAML.
        split: Evaluation split name.
        test_config_path: Test YAML path when evaluating the held-out split.
        test_data_path: Test file path when evaluating the held-out split.
        max_windows: Window cap used for evaluation.
        seed: Seed recorded for lineage.

    Returns:
        JSON-serializable evaluation record.
    """

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "split": split,
        "train_config_path": str(train_config_path),
        "baseline_config_path": str(baseline_config_path),
        "test_config_path": str(test_config_path) if test_config_path else None,
        "test_data_path": str(test_data_path) if test_data_path else None,
        "windows": result.windows,
        "max_windows": max_windows,
        "runtime_seconds": result.evaluation_seconds,
        "source_fps": baseline_config.source_fps,
        "frame_step": baseline_config.frame_step,
        "effective_fps": baseline_config.source_fps / baseline_config.frame_step,
        "prediction_horizon_seconds": (
            baseline_config.prediction_length
            * baseline_config.frame_step
            / baseline_config.source_fps
        ),
        "comparison_prediction_length": (baseline_config.comparison_prediction_length),
        "sampling": "deterministic",
        "seed": seed,
        "model": {
            "family": "motion_baseline",
            "baseline": result.baseline,
            "graph_variant": train_config.graph_variant,
            "window_length": train_config.window_length,
            "observation_length": train_config.observation_length,
            "prediction_length": train_config.prediction_length,
            "stride": train_config.stride,
            "frame_step": train_config.frame_step,
            "source_fps": train_config.source_fps,
            "effective_fps": train_config.source_fps / train_config.frame_step,
        },
        "metrics": {
            **result.metrics,
        },
        "motion_profile": result.motion_profile,
        "metrics_by_motion": result.metrics_by_motion,
    }


def _resolve_results_path(
    *,
    explicit_path: Path | None,
    baseline_config: MotionBaselineConfig | None,
    test_config: TestConfig | None,
    train_config: FlatFitConfig,
) -> Path:
    """Resolve the source-specific result ledger path for an evaluation run.

    Args:
        explicit_path: CLI override path.
        baseline_config: Baseline config when evaluating deterministic baselines.
        test_config: Held-out test config when evaluating the test split.
        train_config: Training config used by model/checkpoint evaluation.

    Returns:
        JSONL path where this evaluation record should be appended.
    """

    if explicit_path is not None:
        return explicit_path
    if baseline_config is not None:
        return baseline_config.results_path
    if test_config is not None and test_config.results_path is not None:
        return test_config.results_path
    return default_model_results_path(train_config)


def main() -> None:
    """Run checkpoint evaluation from the command line."""

    parser = argparse.ArgumentParser(description="Evaluate trajectory checkpoints.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/dense_keypoint__train.yml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--test-config", type=Path, default=DEFAULT_TEST_CONFIG_PATH)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--no-save-result", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.checkpoint is None and args.baseline_config is None:
        parser.error("either --checkpoint or --baseline-config is required")
    if args.checkpoint is not None and args.baseline_config is not None:
        parser.error("use either --checkpoint or --baseline-config, not both")

    test_config = load_test_config(args.test_config) if args.split == "test" else None
    baseline_config = (
        load_motion_baseline_config(args.baseline_config)
        if args.baseline_config is not None
        else None
    )
    test_data_path = args.test_data or (
        test_config.data_path if test_config is not None else None
    )
    train_config = load_flat_fit_config(args.config)
    if baseline_config is not None:
        train_config = resolve_baseline_window_config(train_config, baseline_config)
    max_windows = args.max_windows or args.max_validation_windows
    if max_windows is None and baseline_config is not None:
        max_windows = baseline_config.max_windows
    if max_windows is None and test_config is not None:
        max_windows = test_config.max_windows
    seed = (
        args.seed
        if args.seed is not None
        else (
            baseline_config.seed
            if baseline_config is not None
            else (test_config.seed if test_config is not None else train_config.seed)
        )
    )
    wandb_enabled = (
        args.wandb
        if args.wandb is not None
        else (
            baseline_config.wandb
            if baseline_config is not None
            else (test_config.wandb if test_config is not None else False)
        )
    )
    wandb_project = args.wandb_project or (
        baseline_config.wandb_project
        if baseline_config is not None
        else (
            test_config.wandb_project
            if test_config is not None
            else train_config.wandb_project
        )
    )
    wandb_run_name = args.wandb_run_name or (
        baseline_config.wandb_run_name
        if baseline_config is not None
        else (test_config.wandb_run_name if test_config is not None else None)
    )
    results_path = _resolve_results_path(
        explicit_path=args.results_path,
        baseline_config=baseline_config,
        test_config=test_config,
        train_config=train_config,
    )

    if baseline_config is not None:
        results = evaluate_motion_baselines(
            train_config=train_config,
            baseline_config=baseline_config,
            split=args.split,
            test_data_path=test_data_path,
            max_windows=max_windows,
            show_progress=True,
        )
        for baseline_result in results:
            record = build_baseline_evaluation_record(
                result=baseline_result,
                train_config=train_config,
                train_config_path=args.config,
                baseline_config=baseline_config,
                baseline_config_path=args.baseline_config,
                split=args.split,
                test_config_path=args.test_config if args.split == "test" else None,
                test_data_path=test_data_path,
                max_windows=max_windows,
                seed=seed,
            )
            if not args.no_save_result:
                save_evaluation_record(record, results_path)
            if wandb_enabled:
                log_evaluation_to_wandb(
                    record=record,
                    project=wandb_project,
                    run_name=(
                        f"{wandb_run_name}-{baseline_result.baseline}"
                        if wandb_run_name
                        else None
                    ),
                )
            print(f"\nbaseline={baseline_result.baseline}")
            print_evaluation_metrics(baseline_result.metrics)
            print_motion_stratified_metrics(
                metrics_by_motion=baseline_result.metrics_by_motion,
                motion_profile=baseline_result.motion_profile,
            )
        if not args.no_save_result:
            print(f"result={results_path}")
        return

    checkpoint_result = evaluate_flat_checkpoint(
        config=train_config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        test_data_path=test_data_path,
        max_windows=max_windows,
        seed=seed,
        show_progress=True,
    )
    record = build_evaluation_record(
        result=checkpoint_result,
        train_config=train_config,
        train_config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        test_config_path=args.test_config if args.split == "test" else None,
        test_data_path=test_data_path,
        max_windows=max_windows,
        seed=seed,
    )
    if not args.no_save_result:
        save_evaluation_record(record, results_path)
    if wandb_enabled:
        log_evaluation_to_wandb(
            record=record,
            project=wandb_project,
            run_name=wandb_run_name,
        )
    print_evaluation_metrics(checkpoint_result.metrics)
    if (
        checkpoint_result.metrics_by_motion is not None
        and checkpoint_result.motion_profile is not None
    ):
        print_motion_stratified_metrics(
            metrics_by_motion=checkpoint_result.metrics_by_motion,
            motion_profile=checkpoint_result.motion_profile,
        )
    if not args.no_save_result:
        print(f"result={results_path}")


if __name__ == "__main__":
    main()
