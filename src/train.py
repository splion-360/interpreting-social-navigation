"""File description: Training entry point for trajectory model experiments."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from data import (
    MabeDataset,
    MabeSequence,
    MabeWindowDataset,
    PoseNormalizer,
    WindowSpec,
    fill_missing_keypoints,
    split_sequence_ids,
)
from loss import bivariate_gaussian_horizon_nll, bivariate_gaussian_nll
from models import FlatSocialAttentionModel
from st_graph import (
    GraphSequence,
    build_dense_keypoint_graph,
    build_flat_sparse_keypoint_graph,
)


DEFAULT_TRAIN_CONFIG_PATH = Path("config/train.yml")
PATH_CONFIG_FIELDS = {
    "data_path",
    "checkpoint_dir",
    "resume_checkpoint",
    "resume_download_dir",
}
GRAPH_BUILDERS = {
    "dense_keypoint": build_dense_keypoint_graph,
    "flat_sparse_keypoint": build_flat_sparse_keypoint_graph,
}


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

    data_path: Path = Path("data/MaBe/mouse_triplet_train.npy")
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
        stride: Frame stride between windows.
        max_train_windows: Optional training-window cap for debug runs.
        max_validation_windows: Optional validation-window cap for debug runs.
        validation_fraction: Fraction of sequences used for validation.
        learning_rate: Adam learning rate.
        grad_clip: Gradient clipping threshold.
        seed: Random seed for deterministic splits and model initialization.
        device: Requested device, such as `cpu`, `cuda`, or `auto`.
        wandb: Whether to log metrics to Weights & Biases.
        wandb_project: Weights & Biases project name.
        wandb_run_name: Optional Weights & Biases run name.
        checkpoint_dir: Directory for best-checkpoint files.
        save_checkpoints: Whether to save the best validation checkpoint.
        wandb_artifact_name: W&B artifact name for the best checkpoint.
        resume_checkpoint: Optional local checkpoint restored before training.
        resume_wandb_artifact: Optional W&B artifact restored before training.
        resume_download_dir: Directory for downloaded W&B artifacts.
    """

    model: str = "flat"
    data_path: Path = Path("data/MaBe/mouse_triplet_train.npy")
    epochs: int = 100
    batch_size: int = 8
    window_length: int = 20
    observation_length: int = 8
    prediction_length: int = 12
    graph_variant: str = "dense_keypoint"
    stride: int = 20
    max_train_windows: int | None = None
    max_validation_windows: int | None = None
    validation_fraction: float = 0.2
    learning_rate: float = 1e-3
    grad_clip: float = 10.0
    seed: int = 42
    device: str = "auto"
    wandb: bool = False
    wandb_project: str = "interpreting-social-navigation"
    wandb_run_name: str | None = None
    checkpoint_dir: Path = Path("checkpoints/flat")
    save_checkpoints: bool = True
    wandb_artifact_name: str = "flat-best-checkpoint"
    resume_checkpoint: Path | None = None
    resume_wandb_artifact: str | None = None
    resume_download_dir: Path = Path("checkpoints/wandb")


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


def run_flat_warmup(config: FlatWarmupConfig) -> FlatWarmupResult:
    """Run a short flat-model training smoke test.

    Args:
        config: Warm-up configuration.

    Returns:
        Initial and final loss values from the smoke run.
    """

    torch.manual_seed(config.seed)
    device = _select_device(config.device)
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
    device = _select_device(config.device)
    train_windows, validation_windows = _build_window_datasets(config)
    model = FlatSocialAttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    run = _start_wandb(config)

    start_epoch, best_validation_loss, resumed_from = _restore_training_state(
        config=config,
        run=run,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    best_epoch = start_epoch - 1
    checkpoint_path = config.checkpoint_dir / "flat_best.pt"
    final_train_loss = 0.0
    final_validation_loss = 0.0

    for epoch in range(start_epoch, config.epochs + 1):
        final_train_loss = _run_epoch(
            model=model,
            windows=train_windows,
            optimizer=optimizer,
            config=config,
            device=device,
            epoch=epoch,
            split="train",
            show_progress=show_progress,
        )
        final_validation_loss = _run_epoch(
            model=model,
            windows=validation_windows,
            optimizer=None,
            config=config,
            device=device,
            epoch=epoch,
            split="validation",
            show_progress=show_progress,
        )

        print(
            f"epoch={epoch}/{config.epochs} "
            f"train_loss={final_train_loss:.6f} "
            f"validation_loss={final_validation_loss:.6f}"
        )
        _wandb_log(
            run,
            {
                "epoch": epoch,
                "train/loss": final_train_loss,
                "validation/loss": final_validation_loss,
            },
        )

        if final_validation_loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = final_validation_loss
            if config.save_checkpoints:
                config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_loss": final_validation_loss,
                        "config": config,
                    },
                    checkpoint_path,
                )
                _wandb_log_checkpoint(
                    run=run,
                    checkpoint_path=checkpoint_path,
                    artifact_name=config.wandb_artifact_name,
                    epoch=epoch,
                    validation_loss=final_validation_loss,
                )

    _finish_wandb(run)
    return FlatFitResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        checkpoint_path=checkpoint_path if config.save_checkpoints else None,
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

    dataset = MabeDataset.from_file(config.data_path)
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
    )
    train_windows = MabeWindowDataset(
        dataset.select(train_ids),
        spec,
        normalizer=normalizer,
        max_windows=config.max_train_windows,
    )
    validation_windows = MabeWindowDataset(
        dataset.select(validation_ids),
        spec,
        normalizer=normalizer,
        max_windows=config.max_validation_windows,
    )
    return train_windows, validation_windows


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
) -> float:
    """Run one train or validation epoch."""

    model.train(optimizer is not None)
    batch_losses = []
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
            loss = _batch_loss(model, windows, batch_indices, config, device)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        loss_value = float(loss.detach().cpu())
        batch_losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.6f}")

    return float(np.mean(batch_losses))


def _batch_loss(
    model: FlatSocialAttentionModel,
    windows: MabeWindowDataset,
    batch_indices: range,
    config: FlatFitConfig,
    device: torch.device,
) -> Tensor:
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
    return torch.stack(losses).mean()


def _start_wandb(config: FlatFitConfig) -> Any | None:
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
            "learning_rate": config.learning_rate,
            "grad_clip": config.grad_clip,
            "seed": config.seed,
            "device": config.device,
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
) -> None:
    """Version a checkpoint as a W&B model artifact when logging is enabled.

    Args:
        run: Active W&B run, or `None` when W&B is disabled.
        checkpoint_path: Local checkpoint file to upload.
        artifact_name: Stable W&B artifact name.
        epoch: Epoch represented by the checkpoint.
        validation_loss: Validation loss for the checkpoint.
    """

    if run is None:
        return

    import wandb

    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata={"epoch": epoch, "validation_loss": validation_loss},
    )
    artifact.add_file(str(checkpoint_path))
    logged = run.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint["epoch"])
    validation_loss = float(checkpoint["validation_loss"])
    return epoch + 1, validation_loss, str(checkpoint_path)


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

    dataset = MabeDataset.from_file(config.data_path)
    train_windows, validation_windows = _build_window_datasets(config)
    train_sequences = [
        train_windows.sequences[item] for item in train_windows.sequences
    ]
    normalizer = train_windows.normalizer
    first_graph = _graph_builder(config.graph_variant)(
        train_windows.first().keypoints[:-1]
    )
    model = FlatSocialAttentionModel()
    report = {
        "config": _plain_config(config),
        "data": {
            "path": str(config.data_path),
            "total_sequences": len(dataset.sequence_ids),
            "train_sequences": len(train_windows.sequences),
            "validation_sequences": len(validation_windows.sequences),
            "train_windows": len(train_windows),
            "validation_windows": len(validation_windows),
            "sequence_frames": _frame_summary(dataset),
            "missing_keypoints": _missing_keypoint_summary(dataset),
            "window": {
                "length": train_windows.spec.length,
                "observation_length": train_windows.spec.observation_length,
                "prediction_length": train_windows.spec.prediction_length,
                "stride": train_windows.spec.stride,
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
        },
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
            "resolved_device": str(_select_device(config.device)),
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
    print(yaml.safe_dump(report, sort_keys=False))


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
        "stride": args.stride,
        "max_train_windows": args.max_train_windows,
        "max_validation_windows": args.max_validation_windows,
        "validation_fraction": args.validation_fraction,
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
        "resume_checkpoint": args.resume_checkpoint,
        "resume_wandb_artifact": args.resume_wandb_artifact,
        "resume_download_dir": args.resume_download_dir,
    }
    if args.no_checkpoint:
        overrides["save_checkpoints"] = False
    return overrides


def main() -> None:
    """Run a training command."""

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
    fit.add_argument("--stride", type=int)
    fit.add_argument("--max-train-windows", type=int)
    fit.add_argument("--max-validation-windows", type=int)
    fit.add_argument("--validation-fraction", type=float)
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
        print(f"device={warmup_result.device}")
        print(f"steps={warmup_result.steps}")
        print(f"initial_loss={warmup_result.initial_loss:.6f}")
        print(f"final_loss={warmup_result.final_loss:.6f}")
        return

    fit_config = load_flat_fit_config(args.config, _fit_overrides(args))
    if fit_config.model != "flat":
        raise ValueError(f"unknown model: {fit_config.model}")
    if args.show_config:
        show_flat_fit_setup(fit_config)
        return

    fit_result = run_flat_fit(fit_config)
    print(f"device={fit_result.device}")
    print(f"best_epoch={fit_result.best_epoch}")
    print(f"best_validation_loss={fit_result.best_validation_loss:.6f}")
    print(f"final_train_loss={fit_result.final_train_loss:.6f}")
    print(f"final_validation_loss={fit_result.final_validation_loss:.6f}")
    if fit_result.checkpoint_path is not None:
        print(f"checkpoint={fit_result.checkpoint_path}")
    if fit_result.wandb_artifact_name is not None:
        print(f"wandb_artifact={fit_result.wandb_artifact_name}:best")
    if fit_result.resumed_from is not None:
        print(f"resumed_from={fit_result.resumed_from}")


if __name__ == "__main__":
    main()
