"""File description: Tests for pixel-space trajectory and pose metrics."""

import numpy as np

from constants import MOUSE_SKELETON_EDGES
from metric import (
    body_frame_keypoint_errors_px,
    body_heading_error_deg,
    bone_length_error_px,
    centroid_ade_px,
    centroid_fde_px,
    compute_pixel_metrics,
    edge_angle_error_matrix_deg,
    edge_bone_length_error_matrix_px,
    keypoint_ade_px,
    keypoint_fde_px,
    relative_ordering_error_by_mouse_axis,
    skeleton_edge_vectors,
    skeleton_orientation_error_deg,
)


def make_pose() -> np.ndarray:
    """Create one synthetic pose shaped `[1, 1, 12, 2]`."""

    pose = np.zeros((1, 1, 12, 2), dtype=np.float32)
    for keypoint_idx in range(12):
        pose[0, 0, keypoint_idx] = np.array([keypoint_idx, 0], dtype=np.float32)
    return pose


def test_pixel_metrics_are_zero_for_perfect_prediction() -> None:
    pose = make_pose()

    metrics = compute_pixel_metrics(pose, pose)

    assert metrics.keypoint_ade_px == 0.0
    assert metrics.keypoint_fde_px == 0.0
    assert metrics.centroid_ade_px == 0.0
    assert metrics.centroid_fde_px == 0.0
    assert metrics.body_frame_keypoint_ade_px == 0.0
    assert metrics.body_frame_keypoint_fde_px == 0.0
    assert metrics.relative_ordering_error == 0.0
    assert metrics.relative_ordering_error_forward == 0.0
    assert metrics.relative_ordering_error_lateral == 0.0
    assert metrics.body_frame_keypoint_error_px_by_mouse.shape == (1, 12)
    assert metrics.relative_ordering_error_by_mouse_axis.shape == (1, 2)
    assert metrics.skeleton_orientation_error_deg == 0.0
    assert metrics.bone_length_error_px == 0.0
    assert metrics.body_heading_error_deg == 0.0
    assert metrics.body_heading_error_deg_by_mouse.shape == (1,)
    assert metrics.edge_angle_error_deg_by_mouse.shape == (1, 12, 12)
    assert metrics.edge_bone_length_error_px_by_mouse.shape == (1, 12, 12)
    assert metrics.to_dict()["keypoint_ade_px"] == 0.0
    assert metrics.to_dict()["edge_angle_error_deg_by_mouse"][0][0][0] is None
    assert metrics.to_dict()["edge_angle_error_deg_by_mouse"][0][0][5] == 0.0


def test_translation_affects_trajectory_and_actual_frame_pose_not_structure() -> None:
    target = make_pose()
    predicted = target + np.array([10, 0], dtype=np.float32)

    assert keypoint_ade_px(predicted, target) == 10.0
    assert keypoint_fde_px(predicted, target) == 10.0
    assert centroid_ade_px(predicted, target) == 10.0
    assert centroid_fde_px(predicted, target) == 10.0
    assert skeleton_orientation_error_deg(predicted, target) == 0.0
    assert bone_length_error_px(predicted, target) == 0.0
    assert body_frame_keypoint_errors_px(predicted, target).mean() == 10.0


def test_skeleton_orientation_detects_right_angle_rotation() -> None:
    target = make_pose()
    predicted = np.zeros_like(target)
    predicted[..., 0] = -target[..., 1]
    predicted[..., 1] = target[..., 0]

    assert skeleton_orientation_error_deg(predicted, target) == 90.0
    assert bone_length_error_px(predicted, target) == 0.0


def test_bone_length_error_detects_uniform_scaling() -> None:
    target = make_pose()
    predicted = target * 2.0
    expected = np.mean(
        [
            abs(pair_distance * 2.0 - pair_distance)
            for pair_distance in range(1, 12)
            for _ in range(2 * (12 - pair_distance))
        ]
    )

    np.testing.assert_allclose(bone_length_error_px(predicted, target), expected)


def test_edge_bone_length_matrix_reports_all_keypoint_pairs() -> None:
    target = make_pose()
    predicted = target.copy()
    predicted[..., 11, 0] += 3.0

    matrix = edge_bone_length_error_matrix_px(predicted, target)

    assert matrix.shape == (1, 12, 12)
    assert matrix[0, 10, 11] == 3.0
    assert matrix[0, 11, 10] == 3.0
    assert matrix[0, 0, 5] == 0.0
    assert np.isnan(matrix[0, 0, 0])


def test_edge_angle_matrix_reports_all_keypoint_pairs() -> None:
    target = make_pose()
    predicted = target.copy()
    predicted[..., 11, :] = np.array([10.0, 1.0], dtype=np.float32)

    matrix = edge_angle_error_matrix_deg(predicted, target)

    assert matrix.shape == (1, 12, 12)
    np.testing.assert_allclose(matrix[0, 10, 11], 90.0)
    np.testing.assert_allclose(matrix[0, 11, 10], 90.0)
    assert matrix[0, 0, 5] == 0.0
    assert np.isnan(matrix[0, 0, 0])


def test_body_heading_error_uses_tail_base_to_neck_axis() -> None:
    target = make_pose()
    predicted = target.copy()
    predicted[..., 3, :] = np.array([9.0, -6.0], dtype=np.float32)

    assert body_heading_error_deg(predicted, target) == 90.0


def test_body_frame_keypoint_error_uses_actual_mouse_frame() -> None:
    target = make_pose()
    predicted = np.zeros_like(target)
    predicted[..., 0] = -target[..., 1] + 20.0
    predicted[..., 1] = target[..., 0] + 30.0

    np.testing.assert_allclose(body_heading_error_deg(predicted, target), 90.0)
    assert body_frame_keypoint_errors_px(predicted, target).mean() > 0.0


def test_relative_ordering_error_detects_keypoint_swaps() -> None:
    target = make_pose()
    predicted = target.copy()
    predicted[..., [0, 11], :] = predicted[..., [11, 0], :]

    ordering = relative_ordering_error_by_mouse_axis(predicted, target)

    assert ordering.shape == (1, 2)
    assert 0.0 < ordering[0, 0] < 1.0
    assert ordering[0, 1] == 0.0


def test_relative_ordering_uses_actual_mouse_frame() -> None:
    target = make_pose()
    predicted = np.zeros_like(target)
    predicted[..., 0] = -target[..., 1] + 20.0
    predicted[..., 1] = target[..., 0] + 30.0

    ordering = relative_ordering_error_by_mouse_axis(predicted, target)

    assert body_frame_keypoint_errors_px(predicted, target).mean() > 0.0
    assert ordering[0, 0] > 0.0
    assert ordering[0, 1] > 0.0


def test_skeleton_edge_vectors_returns_anatomical_edges() -> None:
    vectors = skeleton_edge_vectors(make_pose())

    assert vectors.shape == (1, 1, len(MOUSE_SKELETON_EDGES), 2)
