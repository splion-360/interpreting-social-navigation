"""File description: Training entry point for trajectory model experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from data import (
    MabeDataset,
    MabeWindowDataset,
    PoseNormalizer,
    WindowSpec,
    split_sequence_ids,
)
from flat_model import FlatSocialAttentionModel
from loss import bivariate_gaussian_nll
from social_attention import build_flat_sparse_keypoint_graph


@dataclass(frozen=True)
class FlatWarmupConfig:
    """Configuration for a short flat-model training smoke run.

    Attributes:
        data_path: Path to `mouse_triplet_train.npy`.
        sequence_index: Sorted sequence index used for the smoke run.
        window_length: Number of frames loaded from the sequence.
        steps: Optimizer steps to run.
        learning_rate: Adam learning rate.
        seed: Torch random seed.
        device: Requested device, such as `cpu`, `cuda`, or `auto`.
    """

    data_path: Path = Path("data/MaBe/mouse_triplet_train.npy")
    sequence_index: int = 0
    window_length: int = 9
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
        data_path: Path to `mouse_triplet_train.npy`.
        epochs: Number of training epochs.
        batch_size: Number of windows per optimizer step.
        window_length: Number of frames per next-frame training window.
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
    """

    data_path: Path = Path("data/MaBe/mouse_triplet_train.npy")
    epochs: int = 20
    batch_size: int = 16
    window_length: int = 9
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
        device: Device used for the run.
    """

    best_epoch: int
    best_validation_loss: float
    final_train_loss: float
    final_validation_loss: float
    checkpoint_path: Path | None
    wandb_artifact_name: str | None
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

    best_epoch = 0
    best_validation_loss = float("inf")
    checkpoint_path = config.checkpoint_dir / "flat_best.pt"
    final_train_loss = 0.0
    final_validation_loss = 0.0

    for epoch in range(1, config.epochs + 1):
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

    input_graph = build_flat_sparse_keypoint_graph(window.keypoints[:-1])
    target_graph = build_flat_sparse_keypoint_graph(window.keypoints[1:])
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
        observation_length=config.window_length - 1,
        prediction_length=1,
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
            loss = _batch_loss(model, windows, batch_indices, device)
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
    device: torch.device,
) -> Tensor:
    """Compute the mean next-frame prediction loss for a window batch."""

    losses = []
    for index in batch_indices:
        window = windows[index]
        input_graph = build_flat_sparse_keypoint_graph(window.keypoints[:-1])
        target_graph = build_flat_sparse_keypoint_graph(window.keypoints[1:])
        nodes = torch.from_numpy(input_graph.nodes).to(device)
        edges = torch.from_numpy(input_graph.edge_features).to(device)
        targets = torch.from_numpy(target_graph.nodes).to(device)
        predictions = model(nodes, edges, input_graph.edge_specs)
        losses.append(bivariate_gaussian_nll(predictions, targets))
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
            "data_path": str(config.data_path),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "window_length": config.window_length,
            "stride": config.stride,
            "learning_rate": config.learning_rate,
            "grad_clip": config.grad_clip,
            "seed": config.seed,
            "device": config.device,
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
    run.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])


def _finish_wandb(run: Any | None) -> None:
    """Finish a W&B run when a run exists."""

    if run is not None:
        run.finish()


def _select_device(requested: str) -> torch.device:
    """Resolve a requested training device."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


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
    warmup.add_argument("--steps", type=int, default=FlatWarmupConfig.steps)
    warmup.add_argument(
        "--learning-rate", type=float, default=FlatWarmupConfig.learning_rate
    )
    warmup.add_argument("--seed", type=int, default=FlatWarmupConfig.seed)
    warmup.add_argument("--device", default=FlatWarmupConfig.device)

    fit = subcommands.add_parser("fit", help="Train a trajectory model.")
    fit.add_argument("--model", choices=["flat"], default="flat")
    fit.add_argument("--data", type=Path, default=FlatFitConfig.data_path)
    fit.add_argument("--epochs", type=int, default=FlatFitConfig.epochs)
    fit.add_argument("--batch-size", type=int, default=FlatFitConfig.batch_size)
    fit.add_argument("--window-length", type=int, default=FlatFitConfig.window_length)
    fit.add_argument("--stride", type=int, default=FlatFitConfig.stride)
    fit.add_argument("--max-train-windows", type=int)
    fit.add_argument("--max-validation-windows", type=int)
    fit.add_argument(
        "--validation-fraction", type=float, default=FlatFitConfig.validation_fraction
    )
    fit.add_argument("--learning-rate", type=float, default=FlatFitConfig.learning_rate)
    fit.add_argument("--grad-clip", type=float, default=FlatFitConfig.grad_clip)
    fit.add_argument("--seed", type=int, default=FlatFitConfig.seed)
    fit.add_argument("--device", default=FlatFitConfig.device)
    fit.add_argument("--wandb", action="store_true")
    fit.add_argument("--wandb-project", default=FlatFitConfig.wandb_project)
    fit.add_argument("--wandb-run-name")
    fit.add_argument("--wandb-artifact-name", default=FlatFitConfig.wandb_artifact_name)
    fit.add_argument(
        "--checkpoint-dir", type=Path, default=FlatFitConfig.checkpoint_dir
    )
    fit.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    if args.command == "warmup":
        warmup_result = run_flat_warmup(
            FlatWarmupConfig(
                data_path=args.data,
                sequence_index=args.sequence_index,
                window_length=args.window_length,
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

    fit_result = run_flat_fit(
        FlatFitConfig(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            window_length=args.window_length,
            stride=args.stride,
            max_train_windows=args.max_train_windows,
            max_validation_windows=args.max_validation_windows,
            validation_fraction=args.validation_fraction,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            seed=args.seed,
            device=args.device,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            checkpoint_dir=args.checkpoint_dir,
            save_checkpoints=not args.no_checkpoint,
            wandb_artifact_name=args.wandb_artifact_name,
        )
    )
    print(f"device={fit_result.device}")
    print(f"best_epoch={fit_result.best_epoch}")
    print(f"best_validation_loss={fit_result.best_validation_loss:.6f}")
    print(f"final_train_loss={fit_result.final_train_loss:.6f}")
    print(f"final_validation_loss={fit_result.final_validation_loss:.6f}")
    if fit_result.checkpoint_path is not None:
        print(f"checkpoint={fit_result.checkpoint_path}")
    if fit_result.wandb_artifact_name is not None:
        print(f"wandb_artifact={fit_result.wandb_artifact_name}:best")


if __name__ == "__main__":
    main()
