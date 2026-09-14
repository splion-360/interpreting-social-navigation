"""File description: Prediction and scoring helpers for diagnostic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from data import PoseNormalizer, Window
from inference import (
    RolloutResult,
    rollout_flat_keypoint_model,
    sample_bivariate_gaussian,
)
from loss import gaussian_2d_parameters
from metrics import compute_pixel_metrics
from models import FlatSocialAttentionModel

from .config import RolloutMode


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


def predict_case(
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
    """Predict one window under autoregressive or teacher-forced inputs.

    Args:
        model: Loaded flat trajectory model.
        window: Normalized MABe window.
        mode: Rollout policy to use.
        prediction_length: Number of future frames to predict.
        observation_length: Number of observed frames in the window.
        build_graph: Graph builder for the variant.
        device: Torch device used for inference.
        seed: Sampling seed.

    Returns:
        Predicted future nodes and Gaussian outputs.
    """

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


def case_metrics(
    *,
    prediction: CasePrediction,
    window: Window,
    normalizer: PoseNormalizer,
    mouse_index: int,
    seconds_per_step: float,
) -> dict[str, Any]:
    """Compute pixel metrics for one matched mouse case."""

    predicted, target, initial_pose = _case_pixel_arrays(
        prediction=prediction,
        window=window,
        normalizer=normalizer,
        mouse_index=mouse_index,
    )
    return compute_pixel_metrics(
        predicted,
        target,
        initial_pose=initial_pose,
        seconds_per_step=seconds_per_step,
    ).to_numpy_dict()


def append_horizon_values(
    destination: dict[int, dict[str, list[Any]]],
    *,
    prediction: CasePrediction,
    window: Window,
    normalizer: PoseNormalizer,
    mouse_index: int,
    seconds_per_step: float,
) -> None:
    """Append cumulative horizon metrics for one case prediction."""

    predicted, target, initial_pose = _case_pixel_arrays(
        prediction=prediction,
        window=window,
        normalizer=normalizer,
        mouse_index=mouse_index,
    )
    for horizon in range(1, predicted.shape[0] + 1):
        metrics = compute_pixel_metrics(
            predicted[:horizon],
            target[:horizon],
            initial_pose=initial_pose,
            seconds_per_step=seconds_per_step,
        ).to_numpy_dict()
        append_values(destination[horizon], metrics)


def calibration_metrics(
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


def attention_metrics(
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


def append_values(destination: dict[str, list[Any]], metrics: dict[str, Any]) -> None:
    """Append scalar and array metric values into an accumulator."""

    for name, value in metrics.items():
        destination.setdefault(name, []).append(value)


def append_float_values(
    destination: dict[str, list[float]], metrics: dict[str, float]
) -> None:
    """Append scalar diagnostics into an accumulator."""

    for name, value in metrics.items():
        destination.setdefault(name, []).append(value)


def _case_pixel_arrays(
    *,
    prediction: CasePrediction,
    window: Window,
    normalizer: PoseNormalizer,
    mouse_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one normalized case prediction and target into pixels."""

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
    return predicted, target, initial_pose
