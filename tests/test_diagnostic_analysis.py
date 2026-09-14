"""File description: Tests for diagnostic calibration and distribution analysis."""

import numpy as np
import pytest
import torch

from data import MabeSequence, MabeWindowDataset, WindowSpec
from diagnostics import (
    calibration_profile,
    compare_feature_distributions,
    trajectory_window_features,
    window_feature_distributions,
)
from inference import bivariate_gaussian_mean


def test_bivariate_gaussian_mean_returns_parameter_means() -> None:
    outputs = torch.zeros((2, 3, 5))
    outputs[..., 0] = 2.0
    outputs[..., 1] = -3.0

    means = bivariate_gaussian_mean(outputs)

    assert means.shape == (2, 3, 2)
    torch.testing.assert_close(means[..., 0], torch.full((2, 3), 2.0))
    torch.testing.assert_close(means[..., 1], torch.full((2, 3), -3.0))


def test_calibration_profile_reports_exact_step_ellipse_coverage() -> None:
    outputs = np.zeros((2, 1, 5), dtype=np.float32)
    target = np.asarray([[[0.0, 0.0]], [[2.0, 0.0]]], dtype=np.float32)

    profile = calibration_profile(
        outputs,
        target,
        coordinate_scale=np.asarray([10.0, 20.0], dtype=np.float32),
    )

    assert profile == [
        {
            "horizon_step": 1,
            "coverage_50": 1.0,
            "coverage_95": 1.0,
            "mahalanobis_sq": 0.0,
            "sigma_x_px": 10.0,
            "sigma_y_px": 20.0,
        },
        {
            "horizon_step": 2,
            "coverage_50": 0.0,
            "coverage_95": 1.0,
            "mahalanobis_sq": 4.0,
            "sigma_x_px": 10.0,
            "sigma_y_px": 20.0,
        },
    ]


def test_compare_feature_distributions_reports_standardized_shift() -> None:
    comparison = compare_feature_distributions(
        train_features={"speed": np.asarray([0.0, 1.0, 2.0])},
        validation_features={"speed": np.asarray([2.0, 3.0, 4.0])},
    )

    speed = comparison["speed"]
    assert speed["train_mean"] == 1.0
    assert speed["validation_mean"] == 3.0
    assert speed["standardized_mean_difference"] == pytest.approx(2.0)
    assert speed["ks_statistic"] == pytest.approx(2.0 / 3.0)


def test_trajectory_window_features_capture_motion_and_heading() -> None:
    base_pose = np.zeros((3, 12, 2), dtype=np.float32)
    base_pose[:, :, 0] = np.arange(12, dtype=np.float32)[None] + 100.0
    base_pose[1, :, 0] += 10.0
    base_pose[2, :, 0] += 20.0
    keypoints = np.stack([base_pose + [step, 0.0] for step in range(3)])

    features = trajectory_window_features(keypoints, seconds_per_step=0.2)

    assert features["mean_keypoint_speed_px_s"] == pytest.approx(5.0)
    assert features["mean_keypoint_acceleration_px_s2"] == pytest.approx(0.0)
    assert features["body_heading_cos"] == pytest.approx(-1.0)
    assert features["body_heading_sin"] == pytest.approx(0.0, abs=1e-6)
    assert features["missing_keypoint_fraction"] == 0.0
    assert "local_nose_forward_px" in features
    assert "local_tail_tip_lateral_px" in features


def test_window_features_preserve_source_missingness_for_single_mouse() -> None:
    keypoints = np.ones((3, 1, 12, 2), dtype=np.float32) * 100.0
    keypoints[1, 0, 0] = 0.0
    windows = MabeWindowDataset(
        [MabeSequence(sequence_id="single", keypoints=keypoints)],
        WindowSpec(length=3, observation_length=2, prediction_length=1),
    )

    features = window_feature_distributions(windows, seconds_per_step=0.2)

    assert features["missing_keypoint_fraction"][0] == pytest.approx(1.0 / 36.0)
    assert "mean_inter_mouse_distance_px" not in features
