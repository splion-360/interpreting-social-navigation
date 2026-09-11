"""File description: Tests for pixel-space trajectory and pose metrics."""

import numpy as np

from constants import MOUSE_SKELETON_EDGES
from metric import (
    bone_length_error_px,
    centroid_ade_px,
    centroid_fde_px,
    compute_pixel_metrics,
    keypoint_ade_px,
    keypoint_fde_px,
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
    assert metrics.skeleton_orientation_error_deg == 0.0
    assert metrics.bone_length_error_px == 0.0
    assert metrics.to_dict()["keypoint_ade_px"] == 0.0


def test_translation_affects_trajectory_not_pose_structure() -> None:
    target = make_pose()
    predicted = target + np.array([10, 0], dtype=np.float32)

    assert keypoint_ade_px(predicted, target) == 10.0
    assert keypoint_fde_px(predicted, target) == 10.0
    assert centroid_ade_px(predicted, target) == 10.0
    assert centroid_fde_px(predicted, target) == 10.0
    assert skeleton_orientation_error_deg(predicted, target) == 0.0
    assert bone_length_error_px(predicted, target) == 0.0


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
            abs((end - start) * 2.0 - (end - start))
            for start, end in MOUSE_SKELETON_EDGES
        ]
    )

    np.testing.assert_allclose(bone_length_error_px(predicted, target), expected)


def test_skeleton_edge_vectors_returns_anatomical_edges() -> None:
    vectors = skeleton_edge_vectors(make_pose())

    assert vectors.shape == (1, 1, len(MOUSE_SKELETON_EDGES), 2)
