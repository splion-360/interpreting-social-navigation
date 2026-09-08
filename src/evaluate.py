"""File description: Autoregressive evaluation for trajectory checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from config.mabe import COORDINATES, NUM_KEYPOINTS, NUM_MICE
from loss import gaussian_2d_parameters
from models import FlatSocialAttentionModel
from st_graph import GraphSequence
from train import (
    FlatFitConfig,
    _build_window_datasets,
    _graph_builder,
    _nodes_present_mask,
    _select_device,
    load_flat_fit_config,
)


KEYPOINT_GRAPH_VARIANTS = {"dense_keypoint", "flat_sparse_keypoint"}


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
        windows: Number of validation windows evaluated.
        ade: Average displacement error across predicted frames.
        fde: Final displacement error on the last predicted frame.
        device: Device used for model inference.
    """

    windows: int
    ade: float
    fde: float
    device: str


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
    sample: bool = True,
    generator: torch.Generator | None = None,
) -> RolloutResult:
    """Roll a flat keypoint model forward from observed frames.

    Args:
        model: Trained flat Social Attention model.
        observed_keypoints: Observed keypoints shaped `[time, 3, 12, 2]`.
        prediction_length: Number of future frames to generate.
        build_graph: Graph builder matching the checkpoint/config variant.
        device: Inference device.
        sample: Whether to sample from Gaussians; if false, use Gaussian means.
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
                nodes=torch.from_numpy(graph.nodes[current_frame : current_frame + 1]).to(
                    device
                ),
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
        next_nodes = (
            sample_bivariate_gaussian(output, generator=generator)
            if sample
            else _gaussian_means(output)
        )
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
    max_windows: int | None = None,
    sample: bool = True,
    seed: int | None = None,
) -> EvaluationResult:
    """Evaluate a flat keypoint checkpoint with autoregressive rollouts.

    Args:
        config: Training/evaluation configuration.
        checkpoint_path: Local checkpoint containing model weights.
        max_windows: Optional cap on validation windows.
        sample: Whether to sample from Gaussian predictions.
        seed: Optional random seed for reproducible sampling.

    Returns:
        Mean ADE/FDE over selected validation windows.
    """

    if config.graph_variant not in KEYPOINT_GRAPH_VARIANTS:
        raise ValueError("autoregressive keypoint evaluation requires a keypoint graph")

    device = _select_device(config.device)
    _, validation_windows = _build_window_datasets(
        FlatFitConfig(
            **{
                **_config_values(config),
                "max_validation_windows": max_windows or config.max_validation_windows,
            }
        )
    )
    build_graph = _graph_builder(config.graph_variant)
    model = FlatSocialAttentionModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    ade_values = []
    fde_values = []
    with torch.no_grad():
        for window in validation_windows:
            rollout = rollout_flat_keypoint_model(
                model=model,
                observed_keypoints=window.observed_keypoints,
                prediction_length=validation_windows.spec.prediction_length,
                build_graph=build_graph,
                device=device,
                sample=sample,
                generator=generator if sample else None,
            )
            target_graph = build_graph(window.keypoints)
            mask = _nodes_present_mask(target_graph)[
                validation_windows.spec.observation_length :
            ].cpu()
            prediction = torch.from_numpy(
                rollout.nodes[validation_windows.spec.observation_length :]
            )
            target = torch.from_numpy(
                target_graph.nodes[validation_windows.spec.observation_length :]
            )
            distances = torch.linalg.norm(prediction - target, dim=-1)
            ade_values.append(float(distances[mask].mean()))
            fde_values.append(float(distances[-1][mask[-1]].mean()))

    return EvaluationResult(
        windows=len(validation_windows),
        ade=float(np.mean(ade_values)),
        fde=float(np.mean(fde_values)),
        device=str(device),
    )


def _flat_nodes_to_keypoints(nodes: np.ndarray) -> np.ndarray:
    """Convert flat keypoint nodes back to `[3, 12, 2]` keypoints."""

    return nodes.reshape(NUM_MICE, NUM_KEYPOINTS, COORDINATES).astype(np.float32)


def _config_values(config: FlatFitConfig) -> dict[str, Any]:
    """Return dataclass values that can reconstruct `FlatFitConfig`."""

    return {field.name: getattr(config, field.name) for field in fields(config)}


def _gaussian_means(outputs: Tensor) -> Tensor:
    """Extract Gaussian mean coordinates from raw model outputs."""

    params = gaussian_2d_parameters(outputs)
    return torch.stack((params.mu_x, params.mu_y), dim=-1)


def main() -> None:
    """Run checkpoint evaluation from the command line."""

    parser = argparse.ArgumentParser(description="Evaluate trajectory checkpoints.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/dense_keypoint__train.yml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mean", action="store_true")
    args = parser.parse_args()

    result = evaluate_flat_checkpoint(
        config=load_flat_fit_config(args.config),
        checkpoint_path=args.checkpoint,
        max_windows=args.max_validation_windows,
        sample=not args.mean,
        seed=args.seed,
    )
    print(f"device={result.device}")
    print(f"windows={result.windows}")
    print(f"ade={result.ade:.6f}")
    print(f"fde={result.fde:.6f}")


if __name__ == "__main__":
    main()
