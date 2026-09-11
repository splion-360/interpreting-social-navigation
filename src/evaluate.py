"""File description: Autoregressive evaluation for trajectory checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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

from constants import (
    COORDINATES,
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
from data.visualization import (
    animate_prediction_comparison,
    animate_single_mouse_prediction_comparison,
    save_animation,
)
from loss import gaussian_2d_parameters
from metric import compute_pixel_metrics
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
DEFAULT_RESULTS_PATH = Path("outputs/evaluations/results.jsonl")
TABLE_METRIC_NAMES = (
    "body_heading_error_deg_by_mouse",
    "edge_angle_error_deg_by_mouse",
    "edge_bone_length_error_px_by_mouse",
)
SKELETON_EDGE_CELLS = frozenset(
    (start, end)
    for edge in MOUSE_SKELETON_EDGES
    for start, end in (edge, (edge[1], edge[0]))
)


@dataclass(frozen=True)
class TestConfig:
    """Configuration for held-out test evaluation.

    Attributes:
        data_path: Path to the held-out MABe test file.
        results_path: Local JSONL ledger for evaluation records.
        max_windows: Optional cap on test windows.
        seed: Sampling seed.
        wandb: Whether to log evaluation metrics to W&B.
        wandb_project: W&B project for evaluation logging.
        wandb_run_name: Optional W&B run name for evaluation logging.
    """

    data_path: Path = Path("data/MaBe/mouse_triplet_test.npy")
    results_path: Path = DEFAULT_RESULTS_PATH
    max_windows: int | None = 100
    seed: int = 42
    wandb: bool = False
    wandb_project: str = "interpreting-social-navigation"
    wandb_run_name: str | None = None


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
    """

    split: str
    windows: int
    metrics: dict[str, Any]
    device: str
    checkpoint_epoch: int | None
    checkpoint_validation_loss: float | None


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
            )
            for name, value in metrics.items():
                metric_values.setdefault(name, []).append(value)

    return EvaluationResult(
        split=split,
        windows=len(windows),
        metrics=_aggregate_metric_values(metric_values),
        device=str(device),
        checkpoint_epoch=checkpoint.get("epoch"),
        checkpoint_validation_loss=checkpoint.get("validation_loss"),
    )


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
    animation_obj = animate_prediction_comparison(
        actual_keypoints=actual_keypoints,
        predicted_future_keypoints=predicted_future,
        observation_length=config.observation_length,
        sequence_id=window.sequence_id,
        step=1,
        interval_ms=interval_ms,
    )
    output_path = output_root / config.graph_variant / f"{window.sequence_id}.mp4"
    saved_path = save_animation(animation_obj, output_path, fps=fps)
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
    animation_obj = animate_single_mouse_prediction_comparison(
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
    saved_path = save_animation(animation_obj, output_path, fps=fps)
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


def _flat_nodes_to_keypoints(nodes: np.ndarray) -> np.ndarray:
    """Convert flat keypoint nodes back to `[3, 12, 2]` keypoints."""

    return nodes.reshape(NUM_MICE, NUM_KEYPOINTS, COORDINATES).astype(np.float32)


def _window_pixel_metrics(
    *,
    rollout: RolloutResult,
    window: Window,
    normalizer: PoseNormalizer,
    prediction_length: int,
) -> dict[str, Any]:
    """Compute pixel-space metrics for one evaluated window.

    Args:
        rollout: Model rollout over observed and predicted frames.
        window: Normalized source window with ground-truth future frames.
        normalizer: Pixel-coordinate normalizer used by the dataloader.
        prediction_length: Number of predicted future frames.

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
    target_future = normalizer.inverse_transform(window.future_keypoints)
    return compute_pixel_metrics(predicted_future, target_future).to_numpy_dict()


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
            aggregates[name] = float(np.mean(values))
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
    for name, value in metrics.items():
        if isinstance(value, int | float):
            output.print(f"{name}={value:.6f}")

    table_metrics = {name: metrics[name] for name in TABLE_METRIC_NAMES if name in metrics}
    if not table_metrics:
        return

    output.print()
    output.print(_keypoint_index_table())
    output.print()
    output.print(
        "[dim]Legend: blue cells = known MABe skeleton edges, "
        "green = lowest error pair, red = highest error pair, "
        "underlined = also a known skeleton edge.[/dim]"
    )

    heading = table_metrics.get("body_heading_error_deg_by_mouse")
    if heading is not None:
        output.print()
        output.print("[bold]body_heading_error_deg_by_mouse[/bold]")
        output.print(_body_heading_table(heading))

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


def _keypoint_index_table() -> Table:
    """Build the keypoint-index legend table for CLI output."""

    table = Table(title="Keypoint indices", box=box.SIMPLE_HEAVY)
    table.add_column("index", justify="right", style="cyan", no_wrap=True)
    table.add_column("keypoint", style="white", no_wrap=True)
    for index, name in enumerate(KEYPOINT_NAMES):
        table.add_row(f"{index:02d}", name)
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
        return "bold white on red underline" if is_skeleton_edge else "bold white on red"
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
) -> MabeWindowDataset:
    """Build validation or test windows for checkpoint evaluation.

    Args:
        config: Training/evaluation configuration.
        split: Evaluation split name.
        test_data_path: Held-out test file used only when `split` is `test`.
        max_windows: Optional window cap for fast evaluation.

    Returns:
        Window dataset for the requested split.
    """

    windows, _ = _build_evaluation_data(
        config=config,
        split=split,
        test_data_path=test_data_path,
        max_windows=max_windows,
    )
    return windows


def _build_evaluation_data(
    *,
    config: FlatFitConfig,
    split: Literal["validation", "test"],
    test_data_path: Path | None,
    max_windows: int | None,
) -> tuple[MabeWindowDataset, PoseNormalizer]:
    """Build normalized evaluation windows and their fitted normalizer.

    Args:
        config: Training/evaluation configuration.
        split: Evaluation split name.
        test_data_path: Held-out test file used only when `split` is `test`.
        max_windows: Optional window cap for fast evaluation.

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
    )

    if split == "validation":
        return (
            MabeWindowDataset(
                train_dataset.select(validation_ids),
                spec,
                normalizer=normalizer,
                max_windows=max_windows or config.max_validation_windows,
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
    values["results_path"] = Path(values["results_path"])
    return TestConfig(**values)


def save_evaluation_record(record: dict[str, Any], path: Path) -> None:
    """Append one evaluation record to a local JSONL ledger.

    Args:
        record: JSON-serializable evaluation metadata and metrics.
        path: Destination `.jsonl` file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


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
        },
        "checkpoint": {
            "epoch": result.checkpoint_epoch,
            "validation_loss": result.checkpoint_validation_loss,
        },
        "metrics": {
            **result.metrics,
        },
    }


def main() -> None:
    """Run checkpoint evaluation from the command line."""

    parser = argparse.ArgumentParser(description="Evaluate trajectory checkpoints.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/dense_keypoint__train.yml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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

    test_config = load_test_config(args.test_config) if args.split == "test" else None
    test_data_path = args.test_data or (
        test_config.data_path if test_config is not None else None
    )
    max_windows = (
        args.max_windows
        or args.max_validation_windows
        or (test_config.max_windows if test_config is not None else None)
    )
    train_config = load_flat_fit_config(args.config)
    seed = (
        args.seed
        if args.seed is not None
        else (test_config.seed if test_config is not None else train_config.seed)
    )
    wandb_enabled = (
        args.wandb
        if args.wandb is not None
        else (test_config.wandb if test_config is not None else False)
    )
    wandb_project = args.wandb_project or (
        test_config.wandb_project
        if test_config is not None
        else train_config.wandb_project
    )
    wandb_run_name = args.wandb_run_name or (
        test_config.wandb_run_name if test_config is not None else None
    )
    results_path = args.results_path or (
        test_config.results_path if test_config is not None else DEFAULT_RESULTS_PATH
    )

    result = evaluate_flat_checkpoint(
        config=train_config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        test_data_path=test_data_path,
        max_windows=max_windows,
        seed=seed,
        show_progress=True,
    )
    record = build_evaluation_record(
        result=result,
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
    print_evaluation_metrics(result.metrics)
    if not args.no_save_result:
        print(f"result={results_path}")


if __name__ == "__main__":
    main()
