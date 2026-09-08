"""File description: Training entry point for trajectory model experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from data import MabeDataset, MabeWindowDataset, PoseNormalizer, WindowSpec
from graphs import build_flat_sparse_keypoint_graph
from losses import bivariate_gaussian_nll
from models import FlatSocialAttentionModel


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
    args = parser.parse_args()

    result = run_flat_warmup(
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
    print(f"device={result.device}")
    print(f"steps={result.steps}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")


if __name__ == "__main__":
    main()
