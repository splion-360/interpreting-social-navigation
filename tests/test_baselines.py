"""File description: Tests for deterministic motion baselines."""

import numpy as np

from baselines import (
    predict_keypoint_constant_velocity,
    predict_motion_baseline,
    predict_persistence,
    predict_rigid_constant_velocity,
    valid_baseline_names,
)


def make_observed_keypoints() -> np.ndarray:
    """Create a small observed trajectory shaped `[time, 3, 12, 2]`."""

    base = np.zeros((4, 3, 12, 2), dtype=np.float32)
    for frame in range(base.shape[0]):
        base[frame, ..., 0] = np.arange(12, dtype=np.float32) + frame
        base[frame, ..., 1] = np.arange(3, dtype=np.float32)[:, None] + frame * 2
    return base


def test_persistence_repeats_last_observed_pose() -> None:
    observed = make_observed_keypoints()

    predicted = predict_persistence(observed, prediction_length=3)

    assert predicted.shape == (3, 3, 12, 2)
    np.testing.assert_array_equal(predicted[0], observed[-1])
    np.testing.assert_array_equal(predicted[-1], observed[-1])


def test_rigid_constant_velocity_preserves_last_pose_offsets() -> None:
    observed = make_observed_keypoints()

    predicted = predict_rigid_constant_velocity(observed, prediction_length=2)

    np.testing.assert_allclose(predicted[0, ..., 0] - observed[-1, ..., 0], 1.0)
    np.testing.assert_allclose(predicted[0, ..., 1] - observed[-1, ..., 1], 2.0)
    np.testing.assert_allclose(predicted[1, ..., 0] - observed[-1, ..., 0], 2.0)
    np.testing.assert_allclose(predicted[1, ..., 1] - observed[-1, ..., 1], 4.0)
    expected_offsets = np.repeat(
        observed[-1:, :, 1] - observed[-1:, :, 0],
        repeats=2,
        axis=0,
    )
    np.testing.assert_allclose(
        predicted[:, :, 1] - predicted[:, :, 0],
        expected_offsets,
    )


def test_keypoint_constant_velocity_extrapolates_each_keypoint() -> None:
    observed = make_observed_keypoints()
    observed[:, :, 4, 0] *= 2.0

    predicted = predict_keypoint_constant_velocity(observed, prediction_length=1)

    np.testing.assert_allclose(predicted[0, :, 0, 0], observed[-1, :, 0, 0] + 1.0)
    np.testing.assert_allclose(predicted[0, :, 4, 0], observed[-1, :, 4, 0] + 2.0)


def test_predict_motion_baseline_returns_named_prediction() -> None:
    observed = make_observed_keypoints()

    prediction = predict_motion_baseline(
        "persistence",
        observed_keypoints=observed,
        prediction_length=2,
    )

    assert prediction.name == "persistence"
    assert prediction.future_keypoints.shape == (2, 3, 12, 2)
    assert "rigid_constant_velocity" in valid_baseline_names()
