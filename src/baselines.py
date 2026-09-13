"""File description: Deterministic motion baselines for trajectory evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from data.schema import COORDINATES, NUM_KEYPOINTS, NUM_MICE


BaselineName = Literal[
    "persistence",
    "rigid_constant_velocity",
    "keypoint_constant_velocity",
]


@dataclass(frozen=True)
class MotionBaselinePrediction:
    """Predicted future trajectory from a deterministic baseline.

    Attributes:
        name: Baseline identifier.
        future_keypoints: Predicted future shaped `[time, 3, 12, 2]`.
    """

    name: BaselineName
    future_keypoints: np.ndarray


def predict_motion_baseline(
    name: BaselineName,
    observed_keypoints: np.ndarray,
    prediction_length: int,
) -> MotionBaselinePrediction:
    """Predict future keypoints with a named deterministic baseline.

    Args:
        name: Baseline strategy name.
        observed_keypoints: Observed poses shaped `[time, 3, 12, 2]`.
        prediction_length: Number of future frames to predict.

    Returns:
        Named baseline prediction in the same coordinate system as input.
    """

    predictors: dict[BaselineName, Callable[[np.ndarray, int], np.ndarray]] = {
        "persistence": predict_persistence,
        "rigid_constant_velocity": predict_rigid_constant_velocity,
        "keypoint_constant_velocity": predict_keypoint_constant_velocity,
    }
    return MotionBaselinePrediction(
        name=name,
        future_keypoints=predictors[name](observed_keypoints, prediction_length),
    )


def predict_persistence(
    observed_keypoints: np.ndarray,
    prediction_length: int,
) -> np.ndarray:
    """Repeat the final observed pose through the prediction horizon.

    Args:
        observed_keypoints: Observed poses shaped `[time, 3, 12, 2]`.
        prediction_length: Number of future frames to predict.

    Returns:
        Predicted future shaped `[prediction_length, 3, 12, 2]`.
    """

    return np.repeat(observed_keypoints[-1][None], prediction_length, axis=0).astype(
        np.float32
    )


def predict_rigid_constant_velocity(
    observed_keypoints: np.ndarray,
    prediction_length: int,
) -> np.ndarray:
    """Translate each mouse by its centroid velocity while preserving pose.

    Args:
        observed_keypoints: Observed poses shaped `[time, 3, 12, 2]`.
        prediction_length: Number of future frames to predict.

    Returns:
        Rigidly translated future poses shaped `[prediction_length, 3, 12, 2]`.
    """

    centroids = observed_keypoints.mean(axis=2)
    velocity = _linear_velocity(centroids)
    future_steps = np.arange(1, prediction_length + 1, dtype=np.float32)
    translations = future_steps[:, None, None] * velocity[None]
    return (observed_keypoints[-1][None] + translations[:, :, None, :]).astype(
        np.float32
    )


def predict_keypoint_constant_velocity(
    observed_keypoints: np.ndarray,
    prediction_length: int,
) -> np.ndarray:
    """Extrapolate every keypoint independently with linear velocity.

    Args:
        observed_keypoints: Observed poses shaped `[time, 3, 12, 2]`.
        prediction_length: Number of future frames to predict.

    Returns:
        Per-keypoint extrapolated future shaped `[prediction_length, 3, 12, 2]`.
    """

    velocity = _linear_velocity(observed_keypoints)
    future_steps = np.arange(1, prediction_length + 1, dtype=np.float32)
    return (
        observed_keypoints[-1][None] + future_steps[:, None, None, None] * velocity
    ).astype(np.float32)


def _linear_velocity(values: np.ndarray) -> np.ndarray:
    """Estimate per-frame linear velocity along the first axis.

    Args:
        values: Time-indexed values with any trailing shape.

    Returns:
        Least-squares slope over time with the trailing shape preserved.
    """

    time = np.arange(values.shape[0], dtype=np.float32)
    centered_time = time - time.mean()
    centered_values = values.astype(np.float32) - values.astype(np.float32).mean(axis=0)
    denominator = float(np.sum(centered_time**2))
    if denominator == 0.0:
        return np.zeros(values.shape[1:], dtype=np.float32)
    time_shape = (-1,) + (1,) * (values.ndim - 1)
    numerator = np.sum(centered_time.reshape(time_shape) * centered_values, axis=0)
    return numerator / denominator


def valid_baseline_names() -> tuple[BaselineName, ...]:
    """Return the supported deterministic baseline names."""

    return (
        "persistence",
        "rigid_constant_velocity",
        "keypoint_constant_velocity",
    )


def expected_pose_shape() -> tuple[int, int, int]:
    """Return the expected non-time pose shape for MABe triplets."""

    return (NUM_MICE, NUM_KEYPOINTS, COORDINATES)
