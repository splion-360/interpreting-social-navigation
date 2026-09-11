"""File description: Pixel-space metrics for mouse trajectory predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from constants import MOUSE_SKELETON_EDGES


@dataclass(frozen=True)
class PixelMetricBundle:
    """Evaluation metrics computed in MABe pixel coordinates.

    Attributes:
        keypoint_ade_px: Mean keypoint displacement error over the horizon.
        keypoint_fde_px: Mean keypoint displacement error at the final frame.
        centroid_ade_px: Mean per-mouse centroid displacement over the horizon.
        centroid_fde_px: Mean per-mouse centroid displacement at the final frame.
        skeleton_orientation_error_deg: Mean anatomical edge angle error in degrees.
        bone_length_error_px: Mean anatomical edge length error in pixels.
    """

    keypoint_ade_px: float
    keypoint_fde_px: float
    centroid_ade_px: float
    centroid_fde_px: float
    skeleton_orientation_error_deg: float
    bone_length_error_px: float

    def to_dict(self) -> dict[str, float]:
        """Return metric names and values as a plain dictionary."""

        return asdict(self)


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

    return PixelMetricBundle(
        keypoint_ade_px=keypoint_ade_px(predicted, target),
        keypoint_fde_px=keypoint_fde_px(predicted, target),
        centroid_ade_px=centroid_ade_px(predicted, target),
        centroid_fde_px=centroid_fde_px(predicted, target),
        skeleton_orientation_error_deg=skeleton_orientation_error_deg(
            predicted,
            target,
            min_edge_length_px=min_edge_length_px,
        ),
        bone_length_error_px=bone_length_error_px(predicted, target),
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
    """Return mean angle error across anatomical skeleton edge vectors.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum true/predicted edge length included.

    Returns:
        Mean angular error in degrees, or `nan` when no valid edge exists.
    """

    predicted_vectors = skeleton_edge_vectors(predicted)
    target_vectors = skeleton_edge_vectors(target)
    predicted_lengths = np.linalg.norm(predicted_vectors, axis=-1)
    target_lengths = np.linalg.norm(target_vectors, axis=-1)
    valid = (predicted_lengths >= min_edge_length_px) & (
        target_lengths >= min_edge_length_px
    )
    if not np.any(valid):
        return float("nan")

    dot = np.sum(predicted_vectors * target_vectors, axis=-1)
    cosine = dot / np.maximum(predicted_lengths * target_lengths, 1e-8)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angles[valid].mean())


def bone_length_error_px(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean anatomical edge length error in pixels."""

    predicted_lengths = np.linalg.norm(skeleton_edge_vectors(predicted), axis=-1)
    target_lengths = np.linalg.norm(skeleton_edge_vectors(target), axis=-1)
    return float(np.abs(predicted_lengths - target_lengths).mean())


def skeleton_edge_vectors(keypoints: np.ndarray) -> np.ndarray:
    """Return anatomical edge vectors for poses.

    Args:
        keypoints: Pose array shaped `[time, mice, keypoints, 2]`.

    Returns:
        Edge vectors shaped `[time, mice, anatomical_edges, 2]`.
    """

    starts, ends = np.asarray(MOUSE_SKELETON_EDGES, dtype=np.int64).T
    return keypoints[..., ends, :] - keypoints[..., starts, :]


def _keypoint_distances(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean distances for corresponding keypoints."""

    return np.linalg.norm(predicted.astype(np.float32) - target.astype(np.float32), axis=-1)


def _centroid_distances(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between per-mouse keypoint centroids."""

    predicted_centroids = predicted.astype(np.float32).mean(axis=2)
    target_centroids = target.astype(np.float32).mean(axis=2)
    return np.linalg.norm(predicted_centroids - target_centroids, axis=-1)
