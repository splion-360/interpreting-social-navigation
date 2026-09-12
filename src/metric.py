"""File description: Pixel-space metrics for mouse trajectory predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from constants import MOUSE_SKELETON_EDGES, NUM_KEYPOINTS


BODY_HEADING_EDGE = (9, 3)
BODY_FRAME_ORIGIN_KEYPOINT = 6
BODY_FRAME_AXES = ("forward", "lateral")


@dataclass(frozen=True)
class PixelMetricBundle:
    """Evaluation metrics computed in MABe pixel coordinates.

    Attributes:
        keypoint_ade_px: Mean keypoint displacement error over the horizon.
        keypoint_fde_px: Mean keypoint displacement error at the final frame.
        centroid_ade_px: Mean per-mouse centroid displacement over the horizon.
        centroid_fde_px: Mean per-mouse centroid displacement at the final frame.
        centroid_x_offset_px: Signed mean centroid x-offset in image coordinates.
        centroid_y_offset_px: Signed mean centroid y-offset in image coordinates.
        centroid_offset_px_by_mouse: Signed per-mouse centroid offset.
        centroid_velocity_error_px_per_frame: Mean centroid velocity-vector error.
        keypoint_velocity_error_px_per_frame: Mean keypoint velocity-vector error.
        displacement_magnitude_error_px: Mean net centroid travel-distance error.
        displacement_direction_error_deg: Mean net centroid direction error.
        displacement_gain: Predicted-to-target net centroid displacement ratio.
        relative_ordering_error: Mean local-body-frame keypoint ordering error.
        relative_ordering_error_forward: Ordering error along the body axis.
        relative_ordering_error_lateral: Ordering error across the body axis.
        body_frame_keypoint_error_px_by_mouse: Per-mouse/keypoint local pose error.
        relative_ordering_error_by_mouse_axis: Per-mouse ordering error by axis.
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
    centroid_x_offset_px: float
    centroid_y_offset_px: float
    centroid_offset_px_by_mouse: np.ndarray
    centroid_velocity_error_px_per_frame: float
    keypoint_velocity_error_px_per_frame: float
    displacement_magnitude_error_px: float
    displacement_direction_error_deg: float
    displacement_gain: float
    relative_ordering_error: float
    relative_ordering_error_forward: float
    relative_ordering_error_lateral: float
    body_frame_keypoint_error_px_by_mouse: np.ndarray
    relative_ordering_error_by_mouse_axis: np.ndarray
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
            "centroid_x_offset_px": self.centroid_x_offset_px,
            "centroid_y_offset_px": self.centroid_y_offset_px,
            "centroid_offset_px_by_mouse": self.centroid_offset_px_by_mouse,
            "centroid_velocity_error_px_per_frame": (
                self.centroid_velocity_error_px_per_frame
            ),
            "keypoint_velocity_error_px_per_frame": (
                self.keypoint_velocity_error_px_per_frame
            ),
            "displacement_magnitude_error_px": self.displacement_magnitude_error_px,
            "displacement_direction_error_deg": (self.displacement_direction_error_deg),
            "displacement_gain": self.displacement_gain,
            "relative_ordering_error": self.relative_ordering_error,
            "relative_ordering_error_forward": self.relative_ordering_error_forward,
            "relative_ordering_error_lateral": self.relative_ordering_error_lateral,
            "body_frame_keypoint_error_px_by_mouse": (
                self.body_frame_keypoint_error_px_by_mouse
            ),
            "relative_ordering_error_by_mouse_axis": (
                self.relative_ordering_error_by_mouse_axis
            ),
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
    initial_pose: np.ndarray | None = None,
    min_edge_length_px: float = 1.0,
) -> PixelMetricBundle:
    """Compute trajectory, pose, and structure metrics in pixel coordinates.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        initial_pose: Final observed pose immediately before the prediction horizon.
            When omitted, the first target frame is used as the motion reference.
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
    body_frame_errors = body_frame_keypoint_errors_px(
        predicted,
        target,
        min_edge_length_px=min_edge_length_px,
    )
    ordering_by_mouse_axis = relative_ordering_error_by_mouse_axis(
        predicted,
        target,
        min_edge_length_px=min_edge_length_px,
    )
    centroid_offsets = centroid_offset_px_by_mouse(predicted, target)
    motion_origin = target[0] if initial_pose is None else initial_pose
    return PixelMetricBundle(
        keypoint_ade_px=keypoint_ade_px(predicted, target),
        keypoint_fde_px=keypoint_fde_px(predicted, target),
        centroid_ade_px=centroid_ade_px(predicted, target),
        centroid_fde_px=centroid_fde_px(predicted, target),
        centroid_x_offset_px=float(centroid_offsets[:, 0].mean()),
        centroid_y_offset_px=float(centroid_offsets[:, 1].mean()),
        centroid_offset_px_by_mouse=centroid_offsets,
        centroid_velocity_error_px_per_frame=(
            centroid_velocity_error_px_per_frame(
                predicted,
                target,
                initial_pose=motion_origin,
            )
        ),
        keypoint_velocity_error_px_per_frame=(
            keypoint_velocity_error_px_per_frame(
                predicted,
                target,
                initial_pose=motion_origin,
            )
        ),
        displacement_magnitude_error_px=displacement_magnitude_error_px(
            predicted,
            target,
            initial_pose=motion_origin,
        ),
        displacement_direction_error_deg=displacement_direction_error_deg(
            predicted,
            target,
            initial_pose=motion_origin,
        ),
        displacement_gain=displacement_gain(
            predicted,
            target,
            initial_pose=motion_origin,
        ),
        relative_ordering_error=_nanmean(ordering_by_mouse_axis),
        relative_ordering_error_forward=_nanmean(ordering_by_mouse_axis[:, 0]),
        relative_ordering_error_lateral=_nanmean(ordering_by_mouse_axis[:, 1]),
        body_frame_keypoint_error_px_by_mouse=_nanmean_axis(
            body_frame_errors,
            axis=0,
        ),
        relative_ordering_error_by_mouse_axis=ordering_by_mouse_axis,
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


def centroid_offset_px_by_mouse(
    predicted: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Return signed centroid offsets for each mouse.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.

    Returns:
        Signed offsets shaped `[mice, 2]`. Positive x means right; positive y
        means down in MABe image coordinates.
    """

    predicted_centroids = predicted.astype(np.float32).mean(axis=2)
    target_centroids = target.astype(np.float32).mean(axis=2)
    return (predicted_centroids - target_centroids).mean(axis=0)


def centroid_velocity_error_px_per_frame(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    initial_pose: np.ndarray,
) -> float:
    """Return mean per-frame centroid velocity-vector error.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        initial_pose: Final observed pose shaped `[mice, keypoints, 2]`.

    Returns:
        Mean Euclidean velocity error in pixels per frame.
    """

    predicted_centroids = predicted.astype(np.float32).mean(axis=2)
    target_centroids = target.astype(np.float32).mean(axis=2)
    initial_centroids = initial_pose.astype(np.float32).mean(axis=1)
    predicted_velocity = np.diff(
        np.concatenate((initial_centroids[None], predicted_centroids), axis=0),
        axis=0,
    )
    target_velocity = np.diff(
        np.concatenate((initial_centroids[None], target_centroids), axis=0),
        axis=0,
    )
    return float(np.linalg.norm(predicted_velocity - target_velocity, axis=-1).mean())


def keypoint_velocity_error_px_per_frame(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    initial_pose: np.ndarray,
) -> float:
    """Return mean per-frame keypoint velocity-vector error.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        initial_pose: Final observed pose shaped `[mice, keypoints, 2]`.

    Returns:
        Mean Euclidean velocity error in pixels per frame.
    """

    predicted_velocity = np.diff(
        np.concatenate((initial_pose[None], predicted), axis=0),
        axis=0,
    )
    target_velocity = np.diff(
        np.concatenate((initial_pose[None], target), axis=0),
        axis=0,
    )
    return float(np.linalg.norm(predicted_velocity - target_velocity, axis=-1).mean())


def displacement_magnitude_error_px(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    initial_pose: np.ndarray,
) -> float:
    """Return mean error in net centroid displacement magnitude.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        initial_pose: Final observed pose shaped `[mice, keypoints, 2]`.

    Returns:
        Mean absolute travel-distance error in pixels.
    """

    predicted_displacement, target_displacement = _centroid_displacements(
        predicted,
        target,
        initial_pose=initial_pose,
    )
    predicted_magnitude = np.linalg.norm(predicted_displacement, axis=-1)
    target_magnitude = np.linalg.norm(target_displacement, axis=-1)
    return float(np.abs(predicted_magnitude - target_magnitude).mean())


def displacement_direction_error_deg(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    initial_pose: np.ndarray,
    min_displacement_px: float = 1.0,
) -> float:
    """Return mean angle error between predicted and target net displacement.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        initial_pose: Final observed pose shaped `[mice, keypoints, 2]`.
        min_displacement_px: Minimum displacement needed to define a direction.

    Returns:
        Mean direction error in degrees, or `nan` when motion is too small.
    """

    predicted_displacement, target_displacement = _centroid_displacements(
        predicted,
        target,
        initial_pose=initial_pose,
    )
    return _nanmean(
        _angle_errors_deg(
            predicted_displacement,
            target_displacement,
            min_length_px=min_displacement_px,
        )
    )


def displacement_gain(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    initial_pose: np.ndarray,
    min_displacement_px: float = 1.0,
) -> float:
    """Return predicted-to-target net centroid displacement ratio.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        initial_pose: Final observed pose shaped `[mice, keypoints, 2]`.
        min_displacement_px: Minimum target displacement included in the ratio.

    Returns:
        Mean displacement ratio. One is ideal, zero is persistence, and values
        above one indicate over-predicted travel.
    """

    predicted_displacement, target_displacement = _centroid_displacements(
        predicted,
        target,
        initial_pose=initial_pose,
    )
    predicted_magnitude = np.linalg.norm(predicted_displacement, axis=-1)
    target_magnitude = np.linalg.norm(target_displacement, axis=-1)
    valid = target_magnitude >= min_displacement_px
    ratios = np.divide(
        predicted_magnitude,
        target_magnitude,
        out=np.full(target_magnitude.shape, np.nan, dtype=np.float32),
        where=valid,
    )
    return _nanmean(ratios)


def ground_truth_motion_px(
    initial_pose: np.ndarray,
    target_future: np.ndarray,
) -> float:
    """Return mean per-mouse centroid travel over a prediction horizon.

    Args:
        initial_pose: Final observed pose shaped `[mice, keypoints, 2]`.
        target_future: Ground-truth future shaped `[time, mice, keypoints, 2]`.

    Returns:
        Mean final centroid displacement across mice in pixels.
    """

    initial_centroids = initial_pose.astype(np.float32).mean(axis=1)
    final_centroids = target_future[-1].astype(np.float32).mean(axis=1)
    return float(np.linalg.norm(final_centroids - initial_centroids, axis=-1).mean())


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
    return _clear_pairwise_diagonal(
        np.abs(predicted_lengths - target_lengths).mean(axis=0)
    )


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
    predicted_vectors = (
        predicted[..., front_keypoint, :] - predicted[..., rear_keypoint, :]
    )
    target_vectors = target[..., front_keypoint, :] - target[..., rear_keypoint, :]
    angles = _angle_errors_deg(
        predicted_vectors,
        target_vectors,
        min_length_px=min_edge_length_px,
    )
    return _nanmean_axis(angles, axis=0)


def body_frame_keypoint_errors_px(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    min_edge_length_px: float = 1.0,
) -> np.ndarray:
    """Return keypoint errors in the ground-truth mouse body frame.

    Each predicted mouse and target mouse is expressed in the target mouse's
    body frame for the same timestep. This measures where predicted semantic
    keypoints land relative to the actual mouse's position and heading.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum heading-vector length for a valid body frame.

    Returns:
        Local pose errors shaped `[time, mice, keypoints]`.
    """

    predicted_local = body_frame_coordinates(
        predicted,
        reference_keypoints=target,
        min_edge_length_px=min_edge_length_px,
    )
    target_local = body_frame_coordinates(target, min_edge_length_px=min_edge_length_px)
    return np.linalg.norm(predicted_local - target_local, axis=-1)


def body_frame_coordinates(
    keypoints: np.ndarray,
    *,
    reference_keypoints: np.ndarray | None = None,
    min_edge_length_px: float = 1.0,
) -> np.ndarray:
    """Express keypoints in each mouse's center-back body frame.

    The origin is the center-back keypoint. The forward axis is the tail-base to
    neck vector, and the lateral axis is its perpendicular. By default, the
    frame is built from `keypoints`; pass `reference_keypoints` to project one
    pose into another pose's body frame.

    Args:
        keypoints: Pose array shaped `[time, mice, keypoints, 2]`.
        reference_keypoints: Optional pose array that defines the body frame.
        min_edge_length_px: Minimum heading-vector length for a valid body frame.

    Returns:
        Local coordinates shaped `[time, mice, keypoints, 2]`, with invalid body
        frames marked as `nan`.
    """

    reference = keypoints if reference_keypoints is None else reference_keypoints
    origin = reference[..., BODY_FRAME_ORIGIN_KEYPOINT, :]
    rear_keypoint, front_keypoint = BODY_HEADING_EDGE
    forward = reference[..., front_keypoint, :] - reference[..., rear_keypoint, :]
    lengths = np.linalg.norm(forward, axis=-1, keepdims=True)
    valid = lengths >= min_edge_length_px
    forward_axis = np.divide(
        forward,
        np.maximum(lengths, 1e-8),
        out=np.zeros_like(forward, dtype=np.float32),
        where=valid,
    )
    lateral_axis = np.stack((-forward_axis[..., 1], forward_axis[..., 0]), axis=-1)
    centered = keypoints - origin[..., None, :]
    local = np.stack(
        (
            np.sum(centered * forward_axis[..., None, :], axis=-1),
            np.sum(centered * lateral_axis[..., None, :], axis=-1),
        ),
        axis=-1,
    )
    return np.where(valid[..., None, :], local, np.nan)


def relative_ordering_error_by_mouse_axis(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    min_edge_length_px: float = 1.0,
) -> np.ndarray:
    """Return local keypoint ordering violations by mouse and body-frame axis.

    The comparison is frame-specific: the predicted pose and ground-truth pose
    are both expressed in the ground-truth mouse-local coordinate frame for the
    same timestep.

    Args:
        predicted: Predicted poses shaped `[time, mice, keypoints, 2]`.
        target: Ground-truth poses shaped `[time, mice, keypoints, 2]`.
        min_edge_length_px: Minimum heading-vector length for a valid body frame.

    Returns:
        Ordering violation rates shaped `[mice, 2]`, where axis 0 is forward and
        axis 1 is lateral.
    """

    predicted_local = body_frame_coordinates(
        predicted,
        reference_keypoints=target,
        min_edge_length_px=min_edge_length_px,
    )
    target_local = body_frame_coordinates(target, min_edge_length_px=min_edge_length_px)
    predicted_order = np.sign(_pairwise_axis_offsets(predicted_local))
    target_order = np.sign(_pairwise_axis_offsets(target_local))
    violations = predicted_order != target_order
    valid = (
        _off_diagonal_pair_mask()[None, None, :, :, None]
        & ~np.isnan(predicted_order)
        & ~np.isnan(target_order)
    )
    return _nanmean_axes(
        np.where(valid, violations.astype(np.float32), np.nan),
        axes=(0, 2, 3),
    )


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


def _pairwise_axis_offsets(local_keypoints: np.ndarray) -> np.ndarray:
    """Return keypoint-pair offsets for each local body-frame axis."""

    return local_keypoints[:, :, None, :, :] - local_keypoints[:, :, :, None, :]


def _off_diagonal_pair_mask() -> np.ndarray:
    """Return the valid non-self keypoint-pair mask."""

    return ~np.eye(NUM_KEYPOINTS, dtype=bool)


def _keypoint_distances(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean distances for corresponding keypoints."""

    return np.linalg.norm(
        predicted.astype(np.float32) - target.astype(np.float32), axis=-1
    )


def _centroid_distances(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between per-mouse keypoint centroids."""

    predicted_centroids = predicted.astype(np.float32).mean(axis=2)
    target_centroids = target.astype(np.float32).mean(axis=2)
    return np.linalg.norm(predicted_centroids - target_centroids, axis=-1)


def _centroid_displacements(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    initial_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final predicted and target centroid displacement vectors."""

    initial_centroids = initial_pose.astype(np.float32).mean(axis=1)
    predicted_centroids = predicted.astype(np.float32).mean(axis=2)
    target_centroids = target.astype(np.float32).mean(axis=2)
    return (
        predicted_centroids[-1] - initial_centroids,
        target_centroids[-1] - initial_centroids,
    )


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


def _nanmean_axes(values: np.ndarray, *, axes: tuple[int, ...]) -> np.ndarray:
    """Return a finite-aware mean along multiple axes."""

    valid = ~np.isnan(values)
    counts = valid.sum(axis=axes)
    sums = np.where(valid, values, 0.0).sum(axis=axes)
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
