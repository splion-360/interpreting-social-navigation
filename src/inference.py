"""File description: Shared autoregressive inference helpers for trajectory models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from loss import gaussian_2d_parameters
from models import FlatSocialAttentionModel
from st_graph import GraphSequence


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


def sample_bivariate_gaussian(
    outputs: Tensor,
    *,
    generator: torch.Generator | None = None,
    standard_normal: Tensor | None = None,
) -> Tensor:
    """Sample 2D positions from raw bivariate Gaussian model outputs.

    Args:
        outputs: Raw Gaussian parameters shaped `[..., 5]`.
        generator: Optional random generator for reproducible sampling.
        standard_normal: Optional paired standard-normal draws shaped `[..., 2]`.

    Returns:
        Sampled coordinates shaped `[..., 2]`.
    """

    params = gaussian_2d_parameters(outputs)
    if standard_normal is None:
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
    else:
        eps_x = standard_normal[..., 0]
        eps_y = standard_normal[..., 1]
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
    standard_normal: Tensor | None = None,
) -> RolloutResult:
    """Roll a flat keypoint model forward from observed frames.

    Args:
        model: Trained flat Social Attention model.
        observed_keypoints: Observed keypoints shaped `[time, mice, keypoints, 2]`.
        prediction_length: Number of future frames to generate.
        build_graph: Graph builder matching the checkpoint/config variant.
        device: Inference device.
        generator: Optional random generator for reproducible sampling.
        standard_normal: Optional draws shaped `[prediction, nodes, 2]`.

    Returns:
        Predicted full sequence containing observed and generated nodes.
    """

    observed = observed_keypoints.astype(np.float32)
    total_length = observed.shape[0] + prediction_length
    pose_shape = observed.shape[1:]
    rollout_keypoints = np.zeros((total_length, *pose_shape), dtype=np.float32)
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
        if standard_normal is None:
            next_nodes = sample_bivariate_gaussian(output, generator=generator)
        else:
            next_nodes = sample_bivariate_gaussian(
                output,
                standard_normal=standard_normal[step_idx],
            )
        gaussian_outputs.append(output.detach().cpu().numpy())
        attention.extend(result.attention_weights)
        rollout_keypoints[current_frame + 1] = flat_nodes_to_keypoints(
            next_nodes.detach().cpu().numpy(),
            pose_shape=pose_shape,
        )

    rollout_graph = build_graph(rollout_keypoints)
    return RolloutResult(
        nodes=rollout_graph.nodes,
        gaussian_outputs=np.stack(gaussian_outputs, axis=0),
        attention_weights=tuple(attention),
    )


def flat_nodes_to_keypoints(
    nodes: np.ndarray,
    *,
    pose_shape: tuple[int, ...],
) -> np.ndarray:
    """Convert flat keypoint nodes back to pose-shaped keypoints.

    Args:
        nodes: Flat keypoint node coordinates shaped `[mice * keypoints, 2]`.
        pose_shape: Output pose shape `[mice, keypoints, 2]`.

    Returns:
        Keypoints shaped like `pose_shape`.
    """

    return nodes.reshape(pose_shape).astype(np.float32)
