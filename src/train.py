"""File description: Training entry point for trajectory model experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from data import (
    DEFAULT_MOTION_MIX,
    MOTION_STRATA,
    MabeDataset,
    MabeSequence,
    MabeWindowDataset,
    MotionProfile,
    MotionStratum,
    MotionThresholds,
    PoseNormalizer,
    WindowSpec,
    build_motion_profile,
    expand_single_mouse_window_keys,
    fill_missing_keypoints,
    fit_motion_thresholds,
    sample_motion_balanced_window_keys,
    source_sequence_id,
    split_sequence_ids,
    to_single_mouse_sequences,
    window_motion_scores_px_s,
)
from data.schema import DEFAULT_SOURCE_FPS
from inference import rollout_flat_keypoint_model
from logging_utils import configure_cli_logging, get_logger
from loss import bivariate_gaussian_horizon_nll, bivariate_gaussian_nll
from metrics import compute_pixel_metrics
from models import FlatSocialAttentionModel
from st_graph import (
    GraphSequence,
    build_dense_keypoint_graph,
    build_flat_sparse_keypoint_graph,
    build_mouse_level_graph,
    build_single_mouse_dense_keypoint_graph,
    build_within_mouse_dense_keypoint_graph,
)


DEFAULT_TRAIN_CONFIG_PATH = Path("src/config/train__flat_dense_triplet_30fps.yml")
PATH_CONFIG_FIELDS = {
    "data_path",
    "checkpoint_dir",
    "resume_checkpoint",
    "resume_download_dir",
}
GRAPH_BUILDERS = {
    "dense_keypoint": build_dense_keypoint_graph,
    "flat_sparse_keypoint": build_flat_sparse_keypoint_graph,
    "mouse_level": build_mouse_level_graph,
    "single_mouse_dense_keypoint": build_single_mouse_dense_keypoint_graph,
    "within_mouse_dense_keypoint": build_within_mouse_dense_keypoint_graph,
}
VALIDATION_METRIC_GRAPH_VARIANTS = {
    "dense_keypoint",
    "flat_sparse_keypoint",
    "single_mouse_dense_keypoint",
    "within_mouse_dense_keypoint",
}
VALIDATION_METRIC_ALIASES = {
    "centroid_ade_px": "cADE",
    "centroid_fde_px": "cFDE",
    "keypoint_ade_px": "kADE",
    "keypoint_fde_px": "kFDE",
    "centroid_velocity_error_px_s": "CVE",
    "keypoint_velocity_error_px_s": "KVE",
    "body_heading_error_deg": "BHE",
    "body_frame_keypoint_ade_px": "BFK-ADE",
    "body_frame_keypoint_fde_px": "BFK-FDE",
    "bone_length_error_px": "BLE",
    "skeleton_orientation_error_deg": "SOE",
    "relative_ordering_error": "ROE",
}
VALIDATION_CLI_METRICS = (
    "cFDE",
    "kFDE",
    "CVE",
    "KVE",
    "BHE",
    "ROE",
)
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FlatWarmupConfig:
    """Configuration for a short flat-model training smoke run.

    Attributes:
        data_path: Path to `mouse_triplet_train.npy`.
        sequence_index: Sorted sequence index used for the smoke run.
        window_length: Number of frames loaded from the sequence.
        graph_variant: Graph builder used by the smoke run.
        steps: Optimizer steps to run.
        learning_rate: Adam learning rate.
        seed: Torch random seed.
        device: Requested device, such as `cpu`, `cuda`, or `auto`.
    """

    data_path: Path = Path("data/mabe/raw/mouse_triplet_train.npy")
    sequence_index: int = 0
    window_length: int = 9
    graph_variant: str = "dense_keypoint"
    steps: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class FlatWarmupResult:
    """Summary metrics from a warm-up run.

    Attributes:
        initial_loss: Loss before the first optimizer step.
        final_loss: Loss after the final optimizer step.
        steps: Number of optimizer steps completed.
        device: Device used for the run.
    """

    initial_loss: float
    final_loss: float
    steps: int
    device: str


@dataclass(frozen=True)
class FlatFitConfig:
    """Configuration for flat-model training.

    Attributes:
        model: Model family to train.
        data_path: Path to `mouse_triplet_train.npy`.
        epochs: Number of training epochs.
        batch_size: Number of windows per optimizer step.
        window_length: Number of frames per training window.
        observation_length: Number of conditioning frames.
        prediction_length: Number of forecast target frames.
        graph_variant: Graph builder used for flat model inputs.
        single_mouse_window_source: Whether single-mouse windows are sampled
            independently or expanded from selected source triplet windows.
        stride: Raw-frame gap between consecutive window starts.
        frame_step: Raw-frame gap between sampled frames inside one window.
        source_fps: Source dataset frame rate before temporal downsampling.
        max_train_windows: Optional training-window cap for debug runs.
        max_validation_windows: Optional validation-window cap for debug runs.
        validation_fraction: Fraction of sequences used for validation.
        motion_sampling: Whether to sample training windows by motion stratum.
        motion_group_mix: Desired low/medium/high training-window proportions.
        motion_score: Name of the motion score used for grouping.
        workers: Number of CPU worker processes used for data-prep scoring.
        learning_rate: Adam learning rate.
        grad_clip: Gradient clipping threshold.
        seed: Random seed for deterministic splits and model initialization.
        device: Requested device, such as `cpu`, `cuda`, or `auto`.
        wandb: Whether to log metrics to Weights & Biases.
        wandb_project: Weights & Biases project name.
        wandb_run_name: Optional Weights & Biases run name.
        checkpoint_dir: Directory for best-checkpoint files.
        save_checkpoints: Whether to save the best validation checkpoint.
        checkpoint_frequency: Optional interval for saving epoch checkpoints.
        wandb_artifact_name: W&B artifact name for the best checkpoint.
        resume_checkpoint: Optional local checkpoint restored before training.
        resume_wandb_artifact: Optional W&B artifact restored before training.
        resume_download_dir: Directory for downloaded W&B artifacts.
    """

    model: str = "flat"
    data_path: Path = Path("data/mabe/raw/mouse_triplet_train.npy")
    epochs: int = 100
    batch_size: int = 8
    window_length: int = 20
    observation_length: int = 8
    prediction_length: int = 12
    graph_variant: str = "dense_keypoint"
    single_mouse_window_source: str = "independent"
    stride: int = 20
    frame_step: int = 1
    source_fps: float = DEFAULT_SOURCE_FPS
    max_train_windows: int | None = None
    max_validation_windows: int | None = None
    validation_fraction: float = 0.2
    motion_sampling: bool = False
    motion_group_mix: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MOTION_MIX)
    )
    motion_score: str = "mean_keypoint_speed_px_s"
    workers: int = 0
    learning_rate: float = 1e-3
    grad_clip: float = 10.0
    seed: int = 42
    device: str = "auto"
    wandb: bool = False
    wandb_project: str = "interpreting-social-navigation"
    wandb_run_name: str | None = None
    checkpoint_dir: Path = Path("checkpoints/flat")
    save_checkpoints: bool = True
    checkpoint_frequency: int | None = None
    wandb_artifact_name: str = "flat-best-checkpoint"
    resume_checkpoint: Path | None = None
    resume_wandb_artifact: str | None = None
    resume_download_dir: Path = Path("checkpoints/wandb")

    def __post_init__(self) -> None:
        """Validate timing values that affect sampling and metric units."""

        if self.frame_step < 1:
            raise ValueError("frame_step must be at least 1")
        if self.source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if self.motion_score != "mean_keypoint_speed_px_s":
            raise ValueError("motion_score must be mean_keypoint_speed_px_s")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.single_mouse_window_source not in {"independent", "triplet"}:
            raise ValueError(
                "single_mouse_window_source must be independent or triplet"
            )
        if (
            self.single_mouse_window_source == "triplet"
            and self.graph_variant != "single_mouse_dense_keypoint"
        ):
            raise ValueError(
                "triplet window sourcing requires single_mouse_dense_keypoint"
            )


@dataclass(frozen=True)
class FlatFitResult:
    """Summary metrics from a flat-model training run.

    Attributes:
        best_epoch: Epoch with the lowest validation loss.
        best_validation_loss: Lowest observed validation loss.
        final_train_loss: Training loss from the final epoch.
        final_validation_loss: Validation loss from the final epoch.
        checkpoint_path: Best-checkpoint path when checkpointing is enabled.
        wandb_artifact_name: W&B artifact name when a checkpoint is logged.
        resumed_from: Checkpoint source restored before training.
        device: Device used for the run.
    """

    best_epoch: int
    best_validation_loss: float
    final_train_loss: float
    final_validation_loss: float
    checkpoint_path: Path | None
    wandb_artifact_name: str | None
    resumed_from: str | None
    device: str


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute-device metadata for a training run.

    Attributes:
        requested: Device value requested by config or CLI.
        resolved: Device actually passed to PyTorch tensors/modules.
        torch_version: Installed PyTorch version.
        cuda_available: Whether PyTorch can use CUDA.
        cuda_device_count: Number of CUDA devices visible to PyTorch.
        cuda_device_index: Selected CUDA device index, when CUDA is used.
        cuda_device_name: Selected CUDA device name, when CUDA is used.
        cuda_version: CUDA version used by the PyTorch build.
        cudnn_version: cuDNN version visible to PyTorch.
    """

    requested: str
    resolved: str
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_index: int | None
    cuda_device_name: str | None
    cuda_version: str | None
    cudnn_version: int | None


@dataclass(frozen=True)
class MotionSamplingReport:
    """Motion-aware training-window selection metadata.

    Attributes:
        train_candidates: Profile over every training candidate window.
        train_selected: Profile over the sampled training windows.
        validation: Profile over validation windows labeled with train thresholds.
    """

    train_candidates: MotionProfile
    train_selected: MotionProfile
    validation: MotionProfile

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready motion sampling metadata."""

        return {
            "enabled": True,
            "score": "mean_keypoint_speed_px_s",
            "thresholds_px_s": self.train_candidates.thresholds.to_dict(),
            "train_candidate_counts": self.train_candidates.counts,
            "train_selected_counts": self.train_selected.counts,
            "validation_counts": self.validation.counts,
        }


@dataclass(frozen=True)
class TrainingWindowData:
    """Resolved train and validation windows for one training run.

    Attributes:
        dataset: Loaded MABe dataset backing the split.
        train_windows: Training windows, optionally motion-balanced.
        validation_windows: Validation windows.
        motion_report: Motion sampling metadata when grouping is enabled.
    """

    dataset: MabeDataset
    train_windows: MabeWindowDataset
    validation_windows: MabeWindowDataset
    motion_report: MotionSamplingReport | None


@dataclass(frozen=True)
class EpochLossSummary:
    """Loss summary for one train or validation epoch.

    Attributes:
        loss: Mean loss across the epoch.
        loss_by_motion: Mean validation loss per motion stratum.
        counts_by_motion: Number of validation windows per motion stratum.
    """

    loss: float
    loss_by_motion: dict[MotionStratum, float]
    counts_by_motion: dict[MotionStratum, int]


@dataclass(frozen=True)
class ValidationMetricSummary:
    """Autoregressive validation metrics for one epoch.

    Attributes:
        metrics: Scalar metrics keyed by abbreviation.
        metrics_by_motion: Scalar metrics keyed by motion stratum and abbreviation.
        counts_by_motion: Number of windows per motion stratum.
    """

    metrics: dict[str, float]
    metrics_by_motion: dict[MotionStratum, dict[str, float]]
    counts_by_motion: dict[MotionStratum, int]


@dataclass
class ValidationMotionMetricTable:
    """Cumulative validation motion table for W&B.

    Attributes:
        rows: Rows logged so far across completed epochs.
    """

    rows: list[tuple[float | int | str, ...]]

    @classmethod
    def empty(cls) -> ValidationMotionMetricTable:
        """Create an empty cumulative table."""

        return cls(rows=[])


def run_flat_warmup(config: FlatWarmupConfig) -> FlatWarmupResult:
    """Run a short flat-model training smoke test.

    Args:
        config: Warm-up configuration.

    Returns:
        Initial and final loss values from the smoke run.
    """

    torch.manual_seed(config.seed)
    device = _select_device(config.device)
    _print_device_info(_device_info(config.device, device))
    nodes, edges, targets, edge_specs = _load_flat_next_frame_batch(config)

    nodes = nodes.to(device)
    edges = edges.to(device)
    targets = targets.to(device)
    model = FlatSocialAttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    initial_loss = 0.0
    final_loss = 0.0
    for step_idx in range(config.steps):
        optimizer.zero_grad()
        predictions = model(nodes, edges, edge_specs)
        loss = bivariate_gaussian_nll(predictions, targets)
        loss.backward()
        optimizer.step()

        value = float(loss.detach().cpu())
        if step_idx == 0:
            initial_loss = value
        final_loss = value

    return FlatWarmupResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=config.steps,
        device=str(device),
    )


def run_flat_fit(config: FlatFitConfig, *, show_progress: bool = True) -> FlatFitResult:
    """Train the flat trajectory model with CLI and optional W&B logging.

    Args:
        config: Training configuration.
        show_progress: Whether to render tqdm progress bars.

    Returns:
        Training summary with best validation loss and checkpoint path.
    """

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    window_data = build_training_window_data(config, show_progress=show_progress)
    train_windows = window_data.train_windows
    validation_windows = window_data.validation_windows
    device = _select_device(config.device)
    device_info = _device_info(config.device, device)
    _print_device_info(device_info)
    _print_motion_sampling_report(window_data.motion_report)
    model = FlatSocialAttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    run = _start_wandb(config, device_info, window_data.motion_report)

    start_epoch, best_validation_loss, resumed_from = _restore_training_state(
        config=config,
        run=run,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    best_epoch = start_epoch - 1
    best_checkpoint_path = config.checkpoint_dir / "flat_best.pt"
    final_train_loss = 0.0
    final_validation_loss = 0.0
    validation_motion_table = ValidationMotionMetricTable.empty()

    for epoch in range(start_epoch, config.epochs + 1):
        train_summary = _run_epoch(
            model=model,
            windows=train_windows,
            optimizer=optimizer,
            config=config,
            device=device,
            epoch=epoch,
            split="train",
            show_progress=show_progress,
        )
        validation_summary = _run_epoch(
            model=model,
            windows=validation_windows,
            optimizer=None,
            config=config,
            device=device,
            epoch=epoch,
            split="validation",
            show_progress=show_progress,
            motion_labels=(
                window_data.motion_report.validation.labels
                if window_data.motion_report is not None
                else None
            ),
        )
        final_train_loss = train_summary.loss
        final_validation_loss = validation_summary.loss
        validation_metrics = _run_validation_metrics(
            model=model,
            windows=validation_windows,
            config=config,
            device=device,
            motion_labels=(
                window_data.motion_report.validation.labels
                if window_data.motion_report is not None
                else None
            ),
            show_progress=show_progress,
        )

        LOGGER.info(
            "epoch=%s/%s train_loss=%.6f validation_loss=%.6f",
            epoch,
            config.epochs,
            final_train_loss,
            final_validation_loss,
        )
        _print_validation_motion_losses(validation_summary)
        _print_validation_metrics(validation_metrics)
        _wandb_log(
            run,
            {
                "epoch": epoch,
                "train/loss": final_train_loss,
                "validation/loss": final_validation_loss,
                **{
                    f"validation/{name}": value
                    for name, value in validation_metrics.metrics.items()
                },
                **{
                    f"validation/loss/{stratum}": loss
                    for stratum, loss in validation_summary.loss_by_motion.items()
                },
                **{
                    f"validation/{stratum}/{name}": value
                    for stratum, values in validation_metrics.metrics_by_motion.items()
                    for name, value in values.items()
                },
            },
        )
        _wandb_log_validation_motion_table(
            run,
            epoch,
            validation_summary,
            validation_metrics,
            validation_motion_table,
        )

        is_best_checkpoint = final_validation_loss < best_validation_loss
        if is_best_checkpoint:
            best_epoch = epoch
            best_validation_loss = final_validation_loss
            if config.save_checkpoints:
                config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    validation_loss=final_validation_loss,
                    motion_report=window_data.motion_report,
                )
                _wandb_log_checkpoint(
                    run=run,
                    checkpoint_path=best_checkpoint_path,
                    artifact_name=config.wandb_artifact_name,
                    epoch=epoch,
                    validation_loss=final_validation_loss,
                    motion_report=window_data.motion_report,
                    aliases=["best", f"epoch-{epoch}"],
                )
        if _should_save_epoch_checkpoint(config, epoch):
            epoch_checkpoint_path = config.checkpoint_dir / f"flat_epoch_{epoch:04d}.pt"
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(
                path=epoch_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                validation_loss=final_validation_loss,
                motion_report=window_data.motion_report,
            )
            if not is_best_checkpoint:
                _wandb_log_checkpoint(
                    run=run,
                    checkpoint_path=epoch_checkpoint_path,
                    artifact_name=config.wandb_artifact_name,
                    epoch=epoch,
                    validation_loss=final_validation_loss,
                    motion_report=window_data.motion_report,
                    aliases=[f"epoch-{epoch}"],
                )

    _finish_wandb(run)
    return FlatFitResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        checkpoint_path=best_checkpoint_path if config.save_checkpoints else None,
        wandb_artifact_name=(
            config.wandb_artifact_name
            if config.wandb and config.save_checkpoints
            else None
        ),
        resumed_from=resumed_from,
        device=str(device),
    )


def _load_flat_next_frame_batch(
    config: FlatWarmupConfig,
) -> tuple[Tensor, Tensor, Tensor, tuple]:
    """Load one normalized window as a next-frame prediction batch."""

    dataset = MabeDataset.from_file(config.data_path)
    sequence_id = dataset.sequence_ids[config.sequence_index]
    sequence = dataset.sequences[sequence_id]
    normalizer = PoseNormalizer.fit([sequence])
    window = MabeWindowDataset(
        [sequence],
        WindowSpec(
            length=config.window_length,
            observation_length=config.window_length - 1,
            prediction_length=1,
            stride=config.window_length,
        ),
        normalizer=normalizer,
        max_windows=1,
    ).first()

    build_graph = _graph_builder(config.graph_variant)
    input_graph = build_graph(window.keypoints[:-1])
    target_graph = build_graph(window.keypoints[1:])
    return (
        torch.from_numpy(input_graph.nodes),
        torch.from_numpy(input_graph.edge_features),
        torch.from_numpy(target_graph.nodes),
        input_graph.edge_specs,
    )


def _build_window_datasets(
    config: FlatFitConfig,
) -> tuple[MabeWindowDataset, MabeWindowDataset]:
    """Load train and validation window datasets."""

    window_data = build_training_window_data(config)
    return window_data.train_windows, window_data.validation_windows


def build_training_window_data(
    config: FlatFitConfig,
    *,
    show_progress: bool = False,
) -> TrainingWindowData:
    """Load train and validation windows with optional motion-aware sampling.

    Args:
        config: Training configuration.
        show_progress: Whether to print data-preparation progress.

    Returns:
        Training windows, validation windows, and motion-sampling metadata.
    """

    if show_progress:
        LOGGER.info("data: loading %s", config.data_path)
    dataset = MabeDataset.from_file(config.data_path)
    if show_progress:
        LOGGER.info("data: splitting sequences and fitting pixel normalizer")
    train_ids, validation_ids = split_sequence_ids(
        dataset.sequence_ids,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    normalizer = PoseNormalizer.fit(dataset.select(train_ids))
    spec = WindowSpec(
        length=config.window_length,
        observation_length=config.observation_length,
        prediction_length=config.prediction_length,
        stride=config.stride,
        frame_step=config.frame_step,
    )
    train_sequences = dataset.select(train_ids)
    validation_sequences = dataset.select(validation_ids)
    matched_single_mouse = config.single_mouse_window_source == "triplet"
    if not matched_single_mouse:
        train_sequences = _sequences_for_graph_variant(
            train_sequences, config.graph_variant
        )
        validation_sequences = _sequences_for_graph_variant(
            validation_sequences,
            config.graph_variant,
        )
    if config.motion_sampling:
        train_windows, validation_windows, motion_report = (
            _build_motion_sampled_windows(
                train_sequences=train_sequences,
                validation_sequences=validation_sequences,
                spec=spec,
                normalizer=normalizer,
                config=config,
                show_progress=show_progress,
                matched_single_mouse=matched_single_mouse,
            )
        )
    else:
        train_windows = _build_selected_windows(
            sequences=train_sequences,
            spec=spec,
            normalizer=normalizer,
            max_windows=config.max_train_windows,
            matched_single_mouse=matched_single_mouse,
        )
        validation_windows = _build_selected_windows(
            sequences=validation_sequences,
            spec=spec,
            normalizer=normalizer,
            max_windows=config.max_validation_windows,
            matched_single_mouse=matched_single_mouse,
        )
        motion_report = None
    return TrainingWindowData(
        dataset=dataset,
        train_windows=train_windows,
        validation_windows=validation_windows,
        motion_report=motion_report,
    )


def _sequences_for_graph_variant(
    sequences: list[MabeSequence],
    graph_variant: str,
) -> list[MabeSequence]:
    """Return the sequence view expected by a graph variant.

    Args:
        sequences: Source sequences after train/validation splitting.
        graph_variant: Graph variant selected by the experiment config.

    Returns:
        Original triplet sequences or extracted single-mouse sequences.
    """

    if graph_variant == "single_mouse_dense_keypoint":
        return to_single_mouse_sequences(sequences)
    return sequences


def _build_motion_sampled_windows(
    *,
    train_sequences: list[MabeSequence],
    validation_sequences: list[MabeSequence],
    spec: WindowSpec,
    normalizer: PoseNormalizer,
    config: FlatFitConfig,
    show_progress: bool,
    matched_single_mouse: bool,
) -> tuple[MabeWindowDataset, MabeWindowDataset, MotionSamplingReport]:
    """Build train windows after fitting motion groups on train candidates.

    Args:
        train_sequences: Training sequences from the sequence-aware split.
        validation_sequences: Validation sequences from the sequence-aware split.
        spec: Window specification.
        normalizer: Train-fitted pose normalizer.
        config: Training configuration.
        show_progress: Whether to print data-preparation progress.
        matched_single_mouse: Whether selected triplet keys should be expanded
            into one training window per mouse.

    Returns:
        Motion-balanced train and validation windows plus sampling metadata.
    """

    if show_progress:
        LOGGER.info("data: indexing training motion candidates")
        LOGGER.info("data: motion scoring workers=%s", config.workers)
    train_candidates = MabeWindowDataset(train_sequences, spec)
    candidate_scores = window_motion_scores_px_s(
        train_candidates,
        seconds_per_step=_seconds_per_step(config),
        show_progress=show_progress,
        desc="data: scoring train candidates",
        workers=config.workers,
    )
    thresholds = fit_motion_thresholds(candidate_scores)
    train_candidate_profile = _motion_profile_from_scores(
        candidate_scores,
        thresholds=thresholds,
    )
    if show_progress:
        LOGGER.info("data: sampling motion-balanced training windows")
    selected_keys = sample_motion_balanced_window_keys(
        window_keys=train_candidates.window_keys,
        labels=train_candidate_profile.labels,
        mix=config.motion_group_mix,
        max_windows=config.max_train_windows,
        seed=config.seed,
    )
    train_selected_pixels = MabeWindowDataset(
        train_sequences,
        spec,
        window_keys=selected_keys,
    )
    train_selected_profile = build_motion_profile(
        train_selected_pixels,
        thresholds=thresholds,
        seconds_per_step=_seconds_per_step(config),
        show_progress=show_progress,
        desc="data: labeling selected train windows",
        workers=config.workers,
    )
    if show_progress:
        LOGGER.info("data: indexing validation motion candidates")
    validation_pixels = MabeWindowDataset(
        validation_sequences,
        spec,
    )
    validation_candidate_profile = build_motion_profile(
        validation_pixels,
        thresholds=thresholds,
        seconds_per_step=_seconds_per_step(config),
        show_progress=show_progress,
        desc="data: labeling validation candidates",
        workers=config.workers,
    )
    if show_progress:
        LOGGER.info("data: sampling motion-balanced validation windows")
    validation_keys = sample_motion_balanced_window_keys(
        window_keys=validation_pixels.window_keys,
        labels=validation_candidate_profile.labels,
        mix=config.motion_group_mix,
        max_windows=config.max_validation_windows,
        seed=config.seed + 1,
    )
    validation_selected_pixels = MabeWindowDataset(
        validation_sequences,
        spec,
        window_keys=validation_keys,
    )
    validation_profile = build_motion_profile(
        validation_selected_pixels,
        thresholds=thresholds,
        seconds_per_step=_seconds_per_step(config),
        show_progress=show_progress,
        desc="data: labeling selected validation windows",
        workers=config.workers,
    )
    if matched_single_mouse:
        train_source_sequences = train_sequences
        validation_source_sequences = validation_sequences
        mice_per_window = int(train_source_sequences[0].keypoints.shape[1])
        train_sequences = to_single_mouse_sequences(train_source_sequences)
        validation_sequences = to_single_mouse_sequences(validation_source_sequences)
        selected_keys = expand_single_mouse_window_keys(
            selected_keys,
            train_source_sequences,
        )
        validation_keys = expand_single_mouse_window_keys(
            validation_keys,
            validation_source_sequences,
        )
        train_selected_profile = _repeat_motion_profile(
            train_selected_profile,
            repeats=mice_per_window,
        )
        validation_profile = _repeat_motion_profile(
            validation_profile,
            repeats=mice_per_window,
        )
    train_windows = MabeWindowDataset(
        train_sequences,
        spec,
        normalizer=normalizer,
        window_keys=selected_keys,
    )
    validation_windows = MabeWindowDataset(
        validation_sequences,
        spec,
        normalizer=normalizer,
        window_keys=validation_keys,
    )
    return (
        train_windows,
        validation_windows,
        MotionSamplingReport(
            train_candidates=train_candidate_profile,
            train_selected=train_selected_profile,
            validation=validation_profile,
        ),
    )


def _build_selected_windows(
    *,
    sequences: list[MabeSequence],
    spec: WindowSpec,
    normalizer: PoseNormalizer,
    max_windows: int | None,
    matched_single_mouse: bool,
) -> MabeWindowDataset:
    """Build capped windows, optionally expanding each triplet key by mouse.

    Args:
        sequences: Source sequences after the train/validation split.
        spec: Window specification.
        normalizer: Train-fitted pixel normalizer.
        max_windows: Maximum source windows to select.
        matched_single_mouse: Whether to expand source keys into mouse samples.

    Returns:
        Normalized windows for the configured graph input.
    """

    if not matched_single_mouse:
        return MabeWindowDataset(
            sequences,
            spec,
            normalizer=normalizer,
            max_windows=max_windows,
        )
    source_windows = MabeWindowDataset(sequences, spec, max_windows=max_windows)
    return MabeWindowDataset(
        to_single_mouse_sequences(sequences),
        spec,
        normalizer=normalizer,
        window_keys=expand_single_mouse_window_keys(
            source_windows.window_keys,
            sequences,
        ),
    )


def _repeat_motion_profile(profile: MotionProfile, *, repeats: int) -> MotionProfile:
    """Repeat source-window motion labels for each extracted mouse.

    Args:
        profile: Motion profile defined on source triplet windows.
        repeats: Number of mice extracted from every source window.

    Returns:
        Motion profile aligned with expanded single-mouse windows.
    """

    return MotionProfile(
        labels=tuple(label for label in profile.labels for _ in range(repeats)),
        thresholds=profile.thresholds,
        scores_px_s=tuple(
            score for score in profile.scores_px_s for _ in range(repeats)
        ),
    )


def _motion_profile_from_scores(
    scores_px_s: np.ndarray,
    *,
    thresholds: MotionThresholds,
) -> MotionProfile:
    """Build motion labels from already-computed window scores.

    Args:
        scores_px_s: Per-window motion scores in pixels per second.
        thresholds: Train-fitted thresholds used to assign strata.

    Returns:
        Motion profile without rescoring the same windows.
    """

    return MotionProfile(
        labels=tuple(thresholds.label(float(score)) for score in scores_px_s),
        thresholds=thresholds,
        scores_px_s=tuple(float(score) for score in scores_px_s),
    )


def _run_epoch(
    *,
    model: FlatSocialAttentionModel,
    windows: MabeWindowDataset,
    optimizer: torch.optim.Optimizer | None,
    config: FlatFitConfig,
    device: torch.device,
    epoch: int,
    split: str,
    show_progress: bool,
    motion_labels: tuple[MotionStratum, ...] | None = None,
) -> EpochLossSummary:
    """Run one train or validation epoch."""

    model.train(optimizer is not None)
    batch_losses = []
    motion_losses: dict[MotionStratum, list[float]] = {
        name: [] for name in MOTION_STRATA
    }
    iterator = range(0, len(windows), config.batch_size)
    progress = tqdm(
        iterator,
        desc=f"{split} epoch {epoch}",
        unit="batch",
        disable=not show_progress,
    )
    for start in progress:
        batch_indices = range(start, min(start + config.batch_size, len(windows)))
        with torch.set_grad_enabled(optimizer is not None):
            loss, window_losses = _batch_loss(
                model,
                windows,
                batch_indices,
                config,
                device,
            )
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        loss_value = float(loss.detach().cpu())
        batch_losses.append(loss_value)
        if motion_labels is not None:
            for offset, index in enumerate(batch_indices):
                motion_losses[motion_labels[index]].append(window_losses[offset])
        progress.set_postfix(loss=f"{loss_value:.6f}")

    loss_by_motion = {
        stratum: float(np.mean(values))
        for stratum, values in motion_losses.items()
        if values
    }
    return EpochLossSummary(
        loss=float(np.mean(batch_losses)),
        loss_by_motion=loss_by_motion,
        counts_by_motion={
            stratum: len(values) for stratum, values in motion_losses.items()
        },
    )


def _batch_loss(
    model: FlatSocialAttentionModel,
    windows: MabeWindowDataset,
    batch_indices: range,
    config: FlatFitConfig,
    device: torch.device,
) -> tuple[Tensor, list[float]]:
    """Compute mean prediction-horizon loss for a window batch."""

    build_graph = _graph_builder(config.graph_variant)
    losses = []
    for index in batch_indices:
        window = windows[index]
        input_graph = build_graph(window.keypoints[:-1])
        target_graph = build_graph(window.keypoints[1:])
        nodes = torch.from_numpy(input_graph.nodes).to(device)
        edges = torch.from_numpy(input_graph.edge_features).to(device)
        targets = torch.from_numpy(target_graph.nodes).to(device)
        target_mask = _nodes_present_mask(target_graph).to(device)
        state = model.initial_state(
            node_count=input_graph.node_count,
            edge_count=input_graph.edge_count,
            device=device,
            dtype=nodes.dtype,
        )
        result = model.forward_with_state(
            nodes=nodes,
            edge_features=edges,
            edge_specs=input_graph.edge_specs,
            nodes_present=input_graph.nodes_present,
            edges_present=input_graph.edges_present,
            state=state,
        )
        losses.append(
            bivariate_gaussian_horizon_nll(
                result.outputs,
                targets,
                observation_length=window.observation_length,
                mask=target_mask,
            )
        )
    loss_values = [float(loss.detach().cpu()) for loss in losses]
    return torch.stack(losses).mean(), loss_values


def _run_validation_metrics(
    *,
    model: FlatSocialAttentionModel,
    windows: MabeWindowDataset,
    config: FlatFitConfig,
    device: torch.device,
    motion_labels: tuple[MotionStratum, ...] | None,
    show_progress: bool,
) -> ValidationMetricSummary:
    """Run autoregressive validation metrics for the current model.

    Args:
        model: Model after the current epoch update.
        windows: Normalized validation windows.
        config: Training configuration.
        device: Device used for inference.
        motion_labels: Optional per-window motion labels from train thresholds.
        show_progress: Whether to render a validation-metrics progress bar.

    Returns:
        Overall and motion-stratified scalar validation metrics.
    """

    if config.graph_variant not in VALIDATION_METRIC_GRAPH_VARIANTS:
        return ValidationMetricSummary(
            metrics={},
            metrics_by_motion={},
            counts_by_motion=dict.fromkeys(MOTION_STRATA, 0),
        )

    normalizer = windows.normalizer
    if normalizer is None:
        raise ValueError("validation metrics require normalized windows")

    build_graph = _graph_builder(config.graph_variant)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    metric_values: dict[str, list[float]] = {}
    by_motion: dict[MotionStratum, dict[str, list[float]]] = {
        name: {} for name in MOTION_STRATA
    }
    counts_by_motion = dict.fromkeys(MOTION_STRATA, 0)
    was_training = model.training
    model.eval()

    iterator = tqdm(
        range(len(windows)),
        desc="validation metrics",
        unit="window",
        disable=not show_progress,
    )
    with torch.no_grad():
        for window_index in iterator:
            window = windows[window_index]
            rollout = rollout_flat_keypoint_model(
                model=model,
                observed_keypoints=window.observed_keypoints,
                prediction_length=config.prediction_length,
                build_graph=build_graph,
                device=device,
                generator=generator,
            )
            predicted_future = normalizer.inverse_transform(
                rollout.nodes[config.observation_length :].reshape(
                    window.future_keypoints.shape
                )
            )
            target_future = normalizer.inverse_transform(window.future_keypoints)
            initial_pose = normalizer.inverse_transform(window.observed_keypoints[-1])
            metrics = _abbreviated_validation_metrics(
                compute_pixel_metrics(
                    predicted_future,
                    target_future,
                    initial_pose=initial_pose,
                    seconds_per_step=_seconds_per_step(config),
                ).to_numpy_dict()
            )
            _append_scalar_metrics(metric_values, metrics)
            if motion_labels is not None:
                stratum = motion_labels[window_index]
                counts_by_motion[stratum] += 1
                _append_scalar_metrics(by_motion[stratum], metrics)

    model.train(was_training)
    return ValidationMetricSummary(
        metrics=_aggregate_scalar_metrics(metric_values),
        metrics_by_motion={
            stratum: _aggregate_scalar_metrics(values)
            for stratum, values in by_motion.items()
            if values
        },
        counts_by_motion=counts_by_motion,
    )


def _abbreviated_validation_metrics(
    metrics: dict[str, float | np.ndarray],
) -> dict[str, float]:
    """Keep validation metrics required for training logs.

    Args:
        metrics: Full pixel-space metric dictionary.

    Returns:
        Scalar metrics keyed by the agreed training-log abbreviations.
    """

    return {
        alias: float(value)
        for metric_name, alias in VALIDATION_METRIC_ALIASES.items()
        if isinstance((value := metrics[metric_name]), int | float)
    }


def _append_scalar_metrics(
    destination: dict[str, list[float]],
    metrics: dict[str, float],
) -> None:
    """Append scalar metrics to an aggregation dictionary."""

    for name, value in metrics.items():
        destination.setdefault(name, []).append(value)


def _aggregate_scalar_metrics(
    metric_values: dict[str, list[float]],
) -> dict[str, float]:
    """Average scalar metric values while ignoring undefined windows."""

    aggregated = {}
    for name, values in metric_values.items():
        array = np.asarray(values, dtype=np.float32)
        finite = array[np.isfinite(array)]
        aggregated[name] = float(finite.mean()) if finite.size else float("nan")
    return aggregated


def _save_checkpoint(
    *,
    path: Path,
    model: FlatSocialAttentionModel,
    optimizer: torch.optim.Optimizer,
    config: FlatFitConfig,
    epoch: int,
    validation_loss: float,
    motion_report: MotionSamplingReport | None,
) -> None:
    """Save model, optimizer, and training metadata to a checkpoint file.

    Args:
        path: Destination checkpoint path.
        model: Model whose parameters are saved.
        optimizer: Optimizer whose state is saved.
        config: Training config used for this run.
        epoch: Epoch represented by this checkpoint.
        validation_loss: Validation loss recorded at this epoch.
        motion_report: Motion sampling metadata for the training run.
    """

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "config": _plain_config(config),
            "motion_profile": (
                motion_report.to_dict() if motion_report is not None else None
            ),
        },
        path,
    )


def _should_save_epoch_checkpoint(config: FlatFitConfig, epoch: int) -> bool:
    """Return whether a periodic checkpoint should be saved for this epoch."""

    return (
        config.save_checkpoints
        and config.checkpoint_frequency is not None
        and config.checkpoint_frequency > 0
        and epoch % config.checkpoint_frequency == 0
    )


def _start_wandb(
    config: FlatFitConfig,
    device_info: DeviceInfo,
    motion_report: MotionSamplingReport | None = None,
) -> Any | None:
    """Start a W&B run when requested."""

    if not config.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requires the optional dependency: "
            'python -m pip install -e ".[wandb]"'
        ) from exc

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "model": "flat",
            "configured_model": config.model,
            "data_path": str(config.data_path),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "window_length": config.window_length,
            "observation_length": config.observation_length,
            "prediction_length": config.prediction_length,
            "graph_variant": config.graph_variant,
            "stride": config.stride,
            "frame_step": config.frame_step,
            "source_fps": config.source_fps,
            "effective_fps": config.source_fps / config.frame_step,
            "motion_sampling": config.motion_sampling,
            "motion_group_mix": config.motion_group_mix,
            "motion_score": config.motion_score,
            "workers": config.workers,
            "motion_profile": (
                motion_report.to_dict() if motion_report is not None else None
            ),
            "validation_metric_aliases": VALIDATION_METRIC_ALIASES,
            "learning_rate": config.learning_rate,
            "grad_clip": config.grad_clip,
            "seed": config.seed,
            "device": config.device,
            "checkpoint_frequency": config.checkpoint_frequency,
            "resolved_device": device_info.resolved,
            "torch_version": device_info.torch_version,
            "cuda_available": device_info.cuda_available,
            "cuda_device_count": device_info.cuda_device_count,
            "cuda_device_index": device_info.cuda_device_index,
            "cuda_device_name": device_info.cuda_device_name,
            "cuda_version": device_info.cuda_version,
            "cudnn_version": device_info.cudnn_version,
            "resume_checkpoint": (
                str(config.resume_checkpoint)
                if config.resume_checkpoint is not None
                else None
            ),
            "resume_wandb_artifact": config.resume_wandb_artifact,
        },
    )


def _wandb_log(run: Any | None, metrics: dict[str, float | int]) -> None:
    """Log metrics to W&B when a run exists."""

    if run is not None:
        run.log(metrics)


def _wandb_log_checkpoint(
    *,
    run: Any | None,
    checkpoint_path: Path,
    artifact_name: str,
    epoch: int,
    validation_loss: float,
    aliases: list[str],
    motion_report: MotionSamplingReport | None = None,
) -> None:
    """Version a checkpoint as a W&B model artifact when logging is enabled.

    Args:
        run: Active W&B run, or `None` when W&B is disabled.
        checkpoint_path: Local checkpoint file to upload.
        artifact_name: Stable W&B artifact name.
        epoch: Epoch represented by the checkpoint.
        validation_loss: Validation loss for the checkpoint.
        motion_report: Motion sampling metadata for the training run.
        aliases: Artifact aliases to assign to this checkpoint version.
    """

    if run is None:
        return

    import wandb

    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata={
            "epoch": epoch,
            "validation_loss": validation_loss,
            "motion_profile": (
                motion_report.to_dict() if motion_report is not None else None
            ),
        },
    )
    artifact.add_file(str(checkpoint_path))
    logged = run.log_artifact(artifact, aliases=aliases)
    logged.wait()


def _restore_training_state(
    *,
    config: FlatFitConfig,
    run: Any | None,
    model: FlatSocialAttentionModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, str | None]:
    """Restore checkpoint state before training when configured.

    Args:
        config: Training configuration.
        run: Active W&B run, or `None` when W&B is disabled.
        model: Model receiving checkpoint weights.
        optimizer: Optimizer receiving checkpoint state.
        device: Device used to map checkpoint tensors.

    Returns:
        Next epoch number, previous best validation loss, and restore source.
    """

    checkpoint_path = _resolve_resume_checkpoint(config, run)
    if checkpoint_path is None:
        return 1, float("inf"), None

    checkpoint = _load_training_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint["epoch"])
    validation_loss = float(checkpoint["validation_loss"])
    return epoch + 1, validation_loss, str(checkpoint_path)


def _load_training_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint saved from either script or module execution.

    Args:
        path: Local checkpoint file.
        device: Device used to map tensors.

    Returns:
        Checkpoint dictionary.
    """

    main_module = cast(Any, sys.modules["__main__"])
    if not hasattr(main_module, "FlatFitConfig"):
        main_module.FlatFitConfig = FlatFitConfig
    return torch.load(path, map_location=device, weights_only=False)


def _resolve_resume_checkpoint(
    config: FlatFitConfig,
    run: Any | None,
) -> Path | None:
    """Resolve a local or W&B checkpoint source.

    Args:
        config: Training configuration.
        run: Active W&B run for artifact download.

    Returns:
        Local checkpoint path, or `None` when training starts fresh.
    """

    if config.resume_checkpoint is not None:
        return config.resume_checkpoint
    if config.resume_wandb_artifact is None:
        return None
    if run is None:
        raise RuntimeError("resuming from a W&B artifact requires --wandb")

    artifact = run.use_artifact(config.resume_wandb_artifact, type="model")
    download_dir = Path(artifact.download(root=str(config.resume_download_dir)))
    checkpoint_path = download_dir / "flat_best.pt"
    if checkpoint_path.exists():
        return checkpoint_path

    matches = sorted(download_dir.glob("*.pt"))
    if not matches:
        raise FileNotFoundError(
            f"no .pt checkpoint found in downloaded artifact: {download_dir}"
        )
    return matches[0]


def _finish_wandb(run: Any | None) -> None:
    """Finish a W&B run when a run exists."""

    if run is not None:
        run.finish()


def _print_motion_sampling_report(
    motion_report: MotionSamplingReport | None,
) -> None:
    """Print fitted motion thresholds and group counts before training.

    Args:
        motion_report: Motion sampling metadata, or `None` when disabled.
    """

    if motion_report is None:
        return

    thresholds = motion_report.train_candidates.thresholds
    LOGGER.info(
        "motion_sampling: score=mean_keypoint_speed_px_s low_max=%.6f "
        "medium_max=%.6f train_selected=%s validation=%s",
        thresholds.low_max_px_s,
        thresholds.medium_max_px_s,
        motion_report.train_selected.counts,
        motion_report.validation.counts,
    )


def _print_validation_motion_losses(summary: EpochLossSummary) -> None:
    """Print validation losses grouped by motion stratum.

    Args:
        summary: Epoch loss summary from the validation split.
    """

    if not summary.loss_by_motion:
        return

    values = " ".join(
        f"{stratum}_loss={summary.loss_by_motion[stratum]:.6f}"
        for stratum in MOTION_STRATA
        if stratum in summary.loss_by_motion
    )
    LOGGER.info("validation_by_motion: %s", values)


def _print_validation_metrics(summary: ValidationMetricSummary) -> None:
    """Print the most useful validation metrics after each epoch.

    Args:
        summary: Autoregressive validation metric summary.
    """

    if not summary.metrics:
        return

    values = " ".join(
        f"{name}={summary.metrics[name]:.3f}"
        for name in VALIDATION_CLI_METRICS
        if name in summary.metrics
    )
    LOGGER.info("validation_metrics: %s", values)


def _wandb_log_validation_motion_table(
    run: Any | None,
    epoch: int,
    loss_summary: EpochLossSummary,
    metric_summary: ValidationMetricSummary,
    table_history: ValidationMotionMetricTable,
) -> None:
    """Log cumulative validation motion-stratum losses and metrics to W&B.

    Args:
        run: Active W&B run, or `None` when W&B is disabled.
        epoch: Epoch number for the table rows.
        loss_summary: Validation loss summary.
        metric_summary: Autoregressive validation metric summary.
        table_history: Accumulated validation table rows from previous epochs.
    """

    if run is None or not metric_summary.metrics_by_motion:
        return

    import wandb

    for stratum in MOTION_STRATA:
        if stratum not in metric_summary.metrics_by_motion:
            continue
        metrics = metric_summary.metrics_by_motion[stratum]
        table_history.rows.append(
            (
                epoch,
                stratum,
                metric_summary.counts_by_motion[stratum],
                loss_summary.loss_by_motion.get(stratum, float("nan")),
                *(
                    metrics.get(alias, float("nan"))
                    for alias in VALIDATION_METRIC_ALIASES.values()
                ),
            )
        )

    table = wandb.Table(
        columns=[
            "epoch",
            "motion_group",
            "windows",
            "loss",
            *VALIDATION_METRIC_ALIASES.values(),
        ]
    )
    for row in table_history.rows:
        table.add_data(*row)
    run.log({"validation/motion_metrics_table": table, "epoch": epoch})


def _seconds_per_step(config: FlatFitConfig) -> float:
    """Return seconds represented by one sampled trajectory step."""

    return config.frame_step / config.source_fps


def _device_info(requested: str, resolved: torch.device) -> DeviceInfo:
    """Collect PyTorch device information for logs and experiment metadata.

    Args:
        requested: Device value requested by config or CLI.
        resolved: Device selected for tensors and modules.

    Returns:
        Device metadata safe to print or send to W&B.
    """

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    cuda_device_index = None
    cuda_device_name = None

    if resolved.type == "cuda" and cuda_available:
        cuda_device_index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
        cuda_device_name = torch.cuda.get_device_name(cuda_device_index)

    return DeviceInfo(
        requested=requested,
        resolved=str(resolved),
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_index=cuda_device_index,
        cuda_device_name=cuda_device_name,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
    )


def _print_device_info(device_info: DeviceInfo) -> None:
    """Print the selected compute device before training work begins.

    Args:
        device_info: Device metadata to print.
    """

    LOGGER.info("device: %s", _device_display_name(device_info))


def _device_display_name(device_info: DeviceInfo) -> str:
    """Return the concise device label shown in training logs."""

    return device_info.cuda_device_name or device_info.resolved


def load_flat_fit_config(
    path: Path = DEFAULT_TRAIN_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> FlatFitConfig:
    """Load flat-model training configuration from YAML and CLI overrides.

    Args:
        path: YAML configuration file.
        overrides: Explicit command-line values that should replace YAML values.

    Returns:
        Fully typed flat-model training configuration.
    """

    raw = yaml.safe_load(path.read_text()) or {}
    values = asdict(FlatFitConfig())
    values.update(
        {
            field.name: raw[field.name]
            for field in fields(FlatFitConfig)
            if field.name in raw
        }
    )
    if overrides is not None:
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

    for key in PATH_CONFIG_FIELDS:
        if values[key] is not None:
            values[key] = Path(values[key])

    return FlatFitConfig(**values)


def show_flat_fit_setup(config: FlatFitConfig) -> None:
    """Print the resolved training setup without running optimization.

    Args:
        config: Training configuration to inspect.
    """

    window_data = build_training_window_data(config, show_progress=True)
    dataset = window_data.dataset
    train_windows = window_data.train_windows
    validation_windows = window_data.validation_windows
    train_sequences = [
        train_windows.sequences[item] for item in train_windows.sequences
    ]
    normalizer = train_windows.normalizer
    resolved_device = _select_device(config.device)
    first_graph = _graph_builder(config.graph_variant)(
        train_windows.first().keypoints[:-1]
    )
    model = FlatSocialAttentionModel()
    report = {
        "config": _plain_config(config),
        "data": {
            "path": str(config.data_path),
            "total_sequences": len(dataset.sequence_ids),
            "train_sequences": len(
                {
                    source_sequence_id(sequence_id)
                    for sequence_id in train_windows.sequences
                }
            ),
            "validation_sequences": len(
                {
                    source_sequence_id(sequence_id)
                    for sequence_id in validation_windows.sequences
                }
            ),
            "train_samples": len(train_windows.sequences),
            "validation_samples": len(validation_windows.sequences),
            "train_windows": len(train_windows),
            "validation_windows": len(validation_windows),
            "sequence_frames": _frame_summary(dataset),
            "missing_keypoints": _missing_keypoint_summary(dataset),
            "window": {
                "length": train_windows.spec.length,
                "observation_length": train_windows.spec.observation_length,
                "prediction_length": train_windows.spec.prediction_length,
                "stride": train_windows.spec.stride,
                "frame_step": train_windows.spec.frame_step,
                "raw_span": train_windows.spec.raw_span,
                "source_fps": config.source_fps,
                "effective_fps": config.source_fps / config.frame_step,
                "observation_seconds": (
                    config.observation_length * config.frame_step / config.source_fps
                ),
                "prediction_seconds": (
                    config.prediction_length * config.frame_step / config.source_fps
                ),
            },
            "normalization": {
                "method": "forward/backward fill intermittent missing keypoints, preserve fully missing keypoints as zero, scale observed pixel coordinates by image width/height",
                "offset_xy": (
                    _rounded_list(normalizer.offset) if normalizer is not None else None
                ),
                "scale_xy": (
                    _rounded_list(normalizer.scale) if normalizer is not None else None
                ),
                "raw_train_coordinate_range_including_zero_sentinels": _coordinate_range(
                    train_sequences
                ),
            },
            "motion_sampling": (
                window_data.motion_report.to_dict()
                if window_data.motion_report is not None
                else {
                    "enabled": False,
                    "score": config.motion_score,
                    "motion_group_mix": config.motion_group_mix,
                    "workers": config.workers,
                }
            ),
        },
        "device": _device_display_name(_device_info(config.device, resolved_device)),
        "graph": {
            "variant": first_graph.variant,
            "node_count": first_graph.node_count,
            "edge_count": first_graph.edge_count,
        },
        "model": {
            "class": model.__class__.__name__,
            "architecture": str(model),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "optimization": {
            "resolved_device": str(resolved_device),
            "loss": "bivariate_gaussian_horizon_nll",
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "grad_clip": config.grad_clip,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        },
        "checkpointing": {
            "enabled": config.save_checkpoints,
            "best_checkpoint_path": str(config.checkpoint_dir / "flat_best.pt"),
            "epoch_checkpoint_pattern": str(
                config.checkpoint_dir / "flat_epoch_{epoch:04d}.pt"
            ),
            "checkpoint_frequency": config.checkpoint_frequency,
            "resume_checkpoint": (
                str(config.resume_checkpoint)
                if config.resume_checkpoint is not None
                else None
            ),
            "resume_wandb_artifact": config.resume_wandb_artifact,
            "resume_download_dir": str(config.resume_download_dir),
        },
        "wandb": {
            "enabled": config.wandb,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
            "artifact": (
                f"{config.wandb_artifact_name}:best"
                if config.wandb and config.save_checkpoints
                else None
            ),
        },
    }
    LOGGER.info("%s", yaml.safe_dump(report, sort_keys=False).rstrip())


def _select_device(requested: str) -> torch.device:
    """Resolve a requested training device."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _graph_builder(variant: str) -> Callable[[np.ndarray], GraphSequence]:
    """Return the graph builder for a configured graph variant.

    Args:
        variant: Name of a graph variant in `GRAPH_BUILDERS`.

    Returns:
        Callable that converts keypoints into a graph sequence.
    """

    if variant not in GRAPH_BUILDERS:
        raise ValueError(f"unknown graph variant: {variant}")
    return GRAPH_BUILDERS[variant]


def _nodes_present_mask(graph: GraphSequence) -> Tensor:
    """Build a boolean node-presence mask from graph metadata.

    Args:
        graph: Graph sequence with `nodes_present` frame metadata.

    Returns:
        Boolean tensor shaped `[time, nodes]`.
    """

    mask = torch.zeros((graph.nodes.shape[0], graph.node_count), dtype=torch.bool)
    for frame_idx, node_ids in enumerate(graph.nodes_present):
        mask[frame_idx, list(node_ids)] = True
    return mask


def _plain_config(config: FlatFitConfig) -> dict[str, Any]:
    """Convert a training config to YAML-safe values."""

    values = asdict(config)
    for key in PATH_CONFIG_FIELDS:
        values[key] = str(values[key]) if values[key] is not None else None
    return values


def _frame_summary(dataset: MabeDataset) -> dict[str, float | int]:
    """Summarize sequence lengths in frames."""

    lengths = np.asarray(
        [
            dataset.sequences[sequence_id].num_frames
            for sequence_id in dataset.sequence_ids
        ]
    )
    return {
        "min": int(lengths.min()),
        "mean": float(np.round(lengths.mean(), 2)),
        "max": int(lengths.max()),
    }


def _missing_keypoint_summary(dataset: MabeDataset) -> dict[str, float | int]:
    """Summarize zero-sentinel keypoints in the loaded dataset."""

    missing = 0
    total = 0
    for sequence_id in dataset.sequence_ids:
        mask = np.all(dataset.sequences[sequence_id].keypoints == 0, axis=-1)
        missing += int(mask.sum())
        total += int(mask.size)

    return {
        "entries": missing,
        "total": total,
        "fraction": float(np.round(missing / total, 6)),
    }


def _coordinate_range(sequences: list[MabeSequence]) -> dict[str, list[float]]:
    """Compute raw coordinate ranges after missing-keypoint filling."""

    filled = [
        fill_missing_keypoints(sequence.keypoints).reshape(-1, 2).astype(np.float32)
        for sequence in sequences
    ]
    stacked = np.concatenate(filled, axis=0)
    return {
        "min_xy": _rounded_list(stacked.min(axis=0)),
        "max_xy": _rounded_list(stacked.max(axis=0)),
    }


def _rounded_list(values: np.ndarray) -> list[float]:
    """Convert an array to a short list of rounded floats."""

    return [float(item) for item in np.round(values.astype(float), 4)]


def _fit_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect explicit fit-command overrides from argparse values."""

    overrides = {
        "model": args.model,
        "data_path": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_length": args.window_length,
        "observation_length": args.observation_length,
        "prediction_length": args.prediction_length,
        "graph_variant": args.graph_variant,
        "single_mouse_window_source": args.single_mouse_window_source,
        "stride": args.stride,
        "frame_step": args.frame_step,
        "source_fps": args.source_fps,
        "max_train_windows": args.max_train_windows,
        "max_validation_windows": args.max_validation_windows,
        "validation_fraction": args.validation_fraction,
        "motion_sampling": args.motion_sampling,
        "motion_score": args.motion_score,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "device": args.device,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_artifact_name": args.wandb_artifact_name,
        "checkpoint_dir": args.checkpoint_dir,
        "save_checkpoints": args.save_checkpoints,
        "checkpoint_frequency": args.checkpoint_frequency,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_wandb_artifact": args.resume_wandb_artifact,
        "resume_download_dir": args.resume_download_dir,
    }
    if args.no_checkpoint:
        overrides["save_checkpoints"] = False
    return overrides


def main() -> None:
    """Run a training command."""

    configure_cli_logging()
    parser = argparse.ArgumentParser(description="Train trajectory models.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    warmup = subcommands.add_parser("warmup", help="Run a short flat-model smoke test.")
    warmup.add_argument("--data", type=Path, default=FlatWarmupConfig.data_path)
    warmup.add_argument(
        "--sequence-index", type=int, default=FlatWarmupConfig.sequence_index
    )
    warmup.add_argument(
        "--window-length", type=int, default=FlatWarmupConfig.window_length
    )
    warmup.add_argument(
        "--graph-variant",
        choices=sorted(GRAPH_BUILDERS),
        default=FlatWarmupConfig.graph_variant,
    )
    warmup.add_argument("--steps", type=int, default=FlatWarmupConfig.steps)
    warmup.add_argument(
        "--learning-rate", type=float, default=FlatWarmupConfig.learning_rate
    )
    warmup.add_argument("--seed", type=int, default=FlatWarmupConfig.seed)
    warmup.add_argument("--device", default=FlatWarmupConfig.device)

    fit = subcommands.add_parser("fit", help="Train a trajectory model.")
    fit.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    fit.add_argument("--show-config", action="store_true")
    fit.add_argument("--model", choices=["flat"])
    fit.add_argument("--data", type=Path)
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--batch-size", type=int)
    fit.add_argument("--window-length", type=int)
    fit.add_argument("--observation-length", type=int)
    fit.add_argument("--prediction-length", type=int)
    fit.add_argument("--graph-variant", choices=sorted(GRAPH_BUILDERS))
    fit.add_argument(
        "--single-mouse-window-source",
        choices=("independent", "triplet"),
    )
    fit.add_argument("--stride", type=int)
    fit.add_argument("--frame-step", type=int)
    fit.add_argument("--source-fps", type=float)
    fit.add_argument("--max-train-windows", type=int)
    fit.add_argument("--max-validation-windows", type=int)
    fit.add_argument("--validation-fraction", type=float)
    fit.add_argument("--motion-sampling", action=argparse.BooleanOptionalAction)
    fit.add_argument("--motion-score")
    fit.add_argument("--workers", type=int)
    fit.add_argument("--learning-rate", type=float)
    fit.add_argument("--grad-clip", type=float)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--device")
    fit.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    fit.add_argument("--wandb-project")
    fit.add_argument("--wandb-run-name")
    fit.add_argument("--wandb-artifact-name")
    fit.add_argument("--checkpoint-dir", type=Path)
    fit.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction)
    fit.add_argument("--checkpoint-frequency", type=int)
    fit.add_argument("--no-checkpoint", action="store_true")
    fit.add_argument("--resume-checkpoint", type=Path)
    fit.add_argument("--resume-wandb-artifact")
    fit.add_argument("--resume-download-dir", type=Path)
    args = parser.parse_args()

    if args.command == "warmup":
        warmup_result = run_flat_warmup(
            FlatWarmupConfig(
                data_path=args.data,
                sequence_index=args.sequence_index,
                window_length=args.window_length,
                graph_variant=args.graph_variant,
                steps=args.steps,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device=args.device,
            )
        )
        LOGGER.info("device=%s", warmup_result.device)
        LOGGER.info("steps=%s", warmup_result.steps)
        LOGGER.info("initial_loss=%.6f", warmup_result.initial_loss)
        LOGGER.info("final_loss=%.6f", warmup_result.final_loss)
        return

    fit_config = load_flat_fit_config(args.config, _fit_overrides(args))
    if fit_config.model != "flat":
        raise ValueError(f"unknown model: {fit_config.model}")
    if args.show_config:
        show_flat_fit_setup(fit_config)
        return

    fit_result = run_flat_fit(fit_config)
    LOGGER.info("device=%s", fit_result.device)
    LOGGER.info("best_epoch=%s", fit_result.best_epoch)
    LOGGER.info("best_validation_loss=%.6f", fit_result.best_validation_loss)
    LOGGER.info("final_train_loss=%.6f", fit_result.final_train_loss)
    LOGGER.info("final_validation_loss=%.6f", fit_result.final_validation_loss)
    if fit_result.checkpoint_path is not None:
        LOGGER.info("checkpoint=%s", fit_result.checkpoint_path)
    if fit_result.wandb_artifact_name is not None:
        LOGGER.info("wandb_artifact=%s:best", fit_result.wandb_artifact_name)
    if fit_result.resumed_from is not None:
        LOGGER.info("resumed_from=%s", fit_result.resumed_from)


if __name__ == "__main__":
    main()
