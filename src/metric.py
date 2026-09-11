"""File description: Pixel-space metrics for mouse trajectory predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from constants import MOUSE_SKELETON_EDGES, NUM_KEYPOINTS


BODY_HEADING_EDGE = (9, 3)


@dataclass(frozen=True)
class PixelMetricBundle:
    """Evaluation metrics computed in MABe pixel coordinates.

    Attributes:
        keypoint_ade_px: Mean keypoint displacement error over the horizon.
        keypoint_fde_px: Mean keypoint displacement error at the final frame.
        centroid_ade_px: Mean per-mouse centroid displacement over the horizon.
        centroid_fde_px: Mean per-mouse centroid displacement at the final frame.
        skeleton_orientation_error_deg: Mean pairwise keypoint angle error in degrees.
        bone_length_error_px: Mean pairwise keypoint distance error in pixels.
        body_heading_error_deg: Mean back-axis heading error in degrees.
        body_heading_error_deg_by_mouse: Per-mouse back-axis heading error.
        edge_angle_error_deg_by_mouse: Dense per-mouse pairwise angle matrices.
        edge_bone_length_error_px_by_mouse: Dense per-mouse pairwise distance matrices.
    """

    keypoint_ade_px: float
    keypoint_fde_px: float
    centroid_ade_px: float
    centroid_fde_px: float
    skeleton_orientation_error_deg: float
    bone_length_error_px: float
    body_heading_error_deg: float
    body_heading_error_deg_by_mouse: np.ndarray
    edge_angle_error_deg_by_mouse: np.ndarray
    edge_bone_length_error_px_by_mouse: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return metric names and JSON-ready values as a plain dictionary."""

        return {
            name: _metric_value_to_json(value)
            for name, value in self.to_numpy_dict().items()
        }

    def to_numpy_dict(self) -> dict[str, float | np.ndarray]:
        """Return metric names with arrays preserved for aggregation."""

        return {
            "keypoint_ade_px": self.keypoint_ade_px,
            "keypoint_fde_px": self.keypoint_fde_px,
            "centroid_ade_px": self.centroid_ade_px,
            "centroid_fde_px": self.centroid_fde_px,
            "skeleton_orientation_error_deg": self.skeleton_orientation_error_deg,
            "bone_length_error_px": self.bone_length_error_px,
            "body_heading_error_deg": self.body_heading_error_deg,
            "body_heading_error_deg_by_mouse": self.body_heading_error_deg_by_mouse,
            "edge_angle_error_deg_by_mouse": self.edge_angle_error_deg_by_mouse,
            "edge_bone_length_error_px_by_mouse": (
                self.edge_bone_length_error_px_by_mouse
            ),
        }


def compute_pixel_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    min_edge_length_px: float = 1.0,
) -> PixelMetricBundle:
    """Compute trajectory, pose, and structure metrics in pixel coordinates.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum true/predicted edge length for orientation.

    Returns:
        Pixel-space metric bundle for the predicted horizon.
    """

    edge_angle_matrix = edge_angle_error_matrix_deg(
        predicted,
        target,
        min_edge_length_px=min_edge_length_px,
    )
    edge_bone_matrix = edge_bone_length_error_matrix_px(predicted, target)
    heading_by_mouse = body_heading_error_deg_by_mouse(
        predicted,
        target,
        min_edge_length_px=min_edge_length_px,
    )
    return PixelMetricBundle(
        keypoint_ade_px=keypoint_ade_px(predicted, target),
        keypoint_fde_px=keypoint_fde_px(predicted, target),
        centroid_ade_px=centroid_ade_px(predicted, target),
        centroid_fde_px=centroid_fde_px(predicted, target),
        skeleton_orientation_error_deg=_nanmean(edge_angle_matrix),
        bone_length_error_px=_nanmean(edge_bone_matrix),
        body_heading_error_deg=_nanmean(heading_by_mouse),
        body_heading_error_deg_by_mouse=heading_by_mouse,
        edge_angle_error_deg_by_mouse=edge_angle_matrix,
        edge_bone_length_error_px_by_mouse=edge_bone_matrix,
    )


def keypoint_ade_px(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean keypoint displacement over all predicted frames."""

    return float(_keypoint_distances(predicted, target).mean())


def keypoint_fde_px(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean keypoint displacement on the final predicted frame."""

    return float(_keypoint_distances(predicted, target)[-1].mean())


def centroid_ade_px(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean per-mouse centroid displacement over all predicted frames."""

    return float(_centroid_distances(predicted, target).mean())


def centroid_fde_px(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean per-mouse centroid displacement on the final predicted frame."""

    return float(_centroid_distances(predicted, target)[-1].mean())


def skeleton_orientation_error_deg(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    min_edge_length_px: float = 1.0,
) -> float:
    """Return mean angle error across all keypoint-pair vectors.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum true/predicted edge length included.

    Returns:
        Mean angular error in degrees, or `nan` when no valid edge exists.
    """

    return _nanmean(
        edge_angle_error_matrix_deg(
            predicted,
            target,
            min_edge_length_px=min_edge_length_px,
        )
    )


def bone_length_error_px(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean pairwise keypoint distance error in pixels."""

    return _nanmean(edge_bone_length_error_matrix_px(predicted, target))


def edge_angle_error_matrix_deg(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    min_edge_length_px: float = 1.0,
) -> np.ndarray:
    """Return per-mouse pairwise keypoint angle error matrices.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum true/predicted pair length included.

    Returns:
        Dense angle matrices shaped `[3, 12, 12]`. Every off-diagonal keypoint
        pair is populated; diagonals are `nan`.
    """

    predicted_vectors = _pairwise_keypoint_vectors(predicted)
    target_vectors = _pairwise_keypoint_vectors(target)
    angles = _angle_errors_deg(
        predicted_vectors,
        target_vectors,
        min_length_px=min_edge_length_px,
    )
    return _clear_pairwise_diagonal(_nanmean_axis(angles, axis=0))


def edge_bone_length_error_matrix_px(
    predicted: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return per-mouse pairwise keypoint distance error matrices.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.

    Returns:
        Dense distance-error matrices shaped `[3, 12, 12]`. Every off-diagonal
        keypoint pair is populated; diagonals are `nan`.
    """

    predicted_lengths = np.linalg.norm(_pairwise_keypoint_vectors(predicted), axis=-1)
    target_lengths = np.linalg.norm(_pairwise_keypoint_vectors(target), axis=-1)
    return _clear_pairwise_diagonal(np.abs(predicted_lengths - target_lengths).mean(axis=0))


def body_heading_error_deg(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    min_edge_length_px: float = 1.0,
) -> float:
    """Return mean body heading error from tail base to neck.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum true/predicted heading vector length.

    Returns:
        Mean heading angle error in degrees.
    """

    return _nanmean(
        body_heading_error_deg_by_mouse(
            predicted,
            target,
            min_edge_length_px=min_edge_length_px,
        )
    )


def body_heading_error_deg_by_mouse(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    min_edge_length_px: float = 1.0,
) -> np.ndarray:
    """Return body heading error separately for each mouse.

    The heading vector follows the mouse back from tail base to neck.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum true/predicted heading vector length.

    Returns:
        Per-mouse heading angle errors shaped `[3]`.
    """

    rear_keypoint, front_keypoint = BODY_HEADING_EDGE
    predicted_vectors = predicted[..., front_keypoint, :] - predicted[
        ..., rear_keypoint, :
    ]
    target_vectors = target[..., front_keypoint, :] - target[..., rear_keypoint, :]
    angles = _angle_errors_deg(
        predicted_vectors,
        target_vectors,
        min_length_px=min_edge_length_px,
    )
    return _nanmean_axis(angles, axis=0)


def skeleton_edge_vectors(keypoints: np.ndarray) -> np.ndarray:
    """Return anatomical edge vectors for poses.

    Args:
        keypoints: Pose array shaped `[time, mice, keypoints, 2]`.

    Returns:
        Edge vectors shaped `[time, mice, anatomical_edges, 2]`.
    """

    starts, ends = np.asarray(MOUSE_SKELETON_EDGES, dtype=np.int64).T
    return keypoints[..., ends, :] - keypoints[..., starts, :]


def _pairwise_keypoint_vectors(keypoints: np.ndarray) -> np.ndarray:
    """Return every within-mouse keypoint-pair vector.

    Args:
        keypoints: Pose array shaped `[time, mice, keypoints, 2]`.

    Returns:
        Pairwise vectors shaped `[time, mice, keypoints, keypoints, 2]`, where
        each cell `[i, j]` stores the vector from keypoint `i` to keypoint `j`.
    """

    return keypoints[:, :, None, :, :] - keypoints[:, :, :, None, :]


def _keypoint_distances(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean distances for corresponding keypoints."""

    return np.linalg.norm(predicted.astype(np.float32) - target.astype(np.float32), axis=-1)


def _centroid_distances(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between per-mouse keypoint centroids."""

    predicted_centroids = predicted.astype(np.float32).mean(axis=2)
    target_centroids = target.astype(np.float32).mean(axis=2)
    return np.linalg.norm(predicted_centroids - target_centroids, axis=-1)


def _angle_errors_deg(
    predicted_vectors: np.ndarray,
    target_vectors: np.ndarray,
    *,
    min_length_px: float,
) -> np.ndarray:
    """Return angle errors with invalid tiny vectors marked as `nan`."""

    predicted_lengths = np.linalg.norm(predicted_vectors, axis=-1)
    target_lengths = np.linalg.norm(target_vectors, axis=-1)
    valid = (predicted_lengths >= min_length_px) & (target_lengths >= min_length_px)
    dot = np.sum(predicted_vectors * target_vectors, axis=-1)
    cosine = dot / np.maximum(predicted_lengths * target_lengths, 1e-8)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.where(valid, angles, np.nan)


def _clear_pairwise_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Mark same-keypoint pairwise entries as `nan`."""

    cleared = matrix.astype(np.float32, copy=True)
    keypoint_indices = np.arange(NUM_KEYPOINTS)
    cleared[:, keypoint_indices, keypoint_indices] = np.nan
    return cleared


def _nanmean(values: np.ndarray) -> float:
    """Return a finite-aware mean without warning on all-`nan` input."""

    valid = ~np.isnan(values)
    if not np.any(valid):
        return float("nan")
    return float(values[valid].mean())


def _nanmean_axis(values: np.ndarray, *, axis: int) -> np.ndarray:
    """Return a finite-aware mean along one axis."""

    valid = ~np.isnan(values)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float32),
        where=counts > 0,
    )


def _metric_value_to_json(value: float | np.ndarray) -> Any:
    """Convert metric values to JSON-safe scalars or lists."""

    if isinstance(value, np.ndarray):
        json_value = value.astype(object)
        json_value[np.isnan(value)] = None
        return json_value.tolist()
    return value
