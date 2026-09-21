"""File description: Train and validation trajectory-distribution comparisons."""

import numpy as np
from scipy.stats import ks_2samp

from data import KEYPOINT_NAMES, MabeWindowDataset
from data.schema import (
    BODY_FRAME_ORIGIN_KEYPOINT,
    BODY_HEADING_EDGE,
    MOUSE_SKELETON_EDGES,
)


def trajectory_window_features(
    keypoints_px: np.ndarray,
    *,
    seconds_per_step: float,
) -> dict[str, float]:
    """Summarize motion, arena position, and local pose for one window.

    Args:
        keypoints_px: Pixel positions shaped `[time, mice, keypoints, 2]`.
        seconds_per_step: Seconds between adjacent sampled frames.

    Returns:
        Scalar features suitable for train/validation distribution comparison.
    """

    keypoints = keypoints_px.astype(np.float32)
    velocity = np.diff(keypoints, axis=0) / seconds_per_step
    acceleration = np.diff(velocity, axis=0) / seconds_per_step
    centroids = keypoints.mean(axis=2)
    heading = (
        keypoints[..., BODY_HEADING_EDGE[1], :]
        - keypoints[..., BODY_HEADING_EDGE[0], :]
    )
    heading /= np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-6)
    origins = keypoints[..., BODY_FRAME_ORIGIN_KEYPOINT, :]
    relative = keypoints - origins[..., None, :]
    lateral = np.stack((-heading[..., 1], heading[..., 0]), axis=-1)
    local_forward = np.sum(relative * heading[..., None, :], axis=-1)
    local_lateral = np.sum(relative * lateral[..., None, :], axis=-1)
    skeleton_lengths = np.stack(
        [
            np.linalg.norm(
                keypoints[..., target, :] - keypoints[..., source, :], axis=-1
            )
            for source, target in MOUSE_SKELETON_EDGES
        ],
        axis=-1,
    )
    features = {
        "mean_keypoint_speed_px_s": float(np.linalg.norm(velocity, axis=-1).mean()),
        "mean_keypoint_acceleration_px_s2": float(
            np.linalg.norm(acceleration, axis=-1).mean()
        ),
        "centroid_x_px": float(centroids[..., 0].mean()),
        "centroid_y_px": float(centroids[..., 1].mean()),
        "body_heading_cos": float(heading[..., 0].mean()),
        "body_heading_sin": float(heading[..., 1].mean()),
        "mean_skeleton_edge_length_px": float(skeleton_lengths.mean()),
        "missing_keypoint_fraction": float(np.all(keypoints == 0.0, axis=-1).mean()),
    }
    if keypoints.shape[1] > 1:
        inter_mouse_distances = np.stack(
            [
                np.linalg.norm(centroids[:, target] - centroids[:, source], axis=-1)
                for source in range(keypoints.shape[1])
                for target in range(source + 1, keypoints.shape[1])
            ],
            axis=-1,
        )
        features["mean_inter_mouse_distance_px"] = float(inter_mouse_distances.mean())
    for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
        features[f"local_{keypoint_name}_forward_px"] = float(
            local_forward[..., keypoint_index].mean()
        )
        features[f"local_{keypoint_name}_lateral_px"] = float(
            local_lateral[..., keypoint_index].mean()
        )
    return features


def window_feature_distributions(
    windows: MabeWindowDataset,
    *,
    seconds_per_step: float,
) -> dict[str, np.ndarray]:
    """Collect scalar feature distributions from a selected window dataset.

    Args:
        windows: Normalized or pixel-space selected windows.
        seconds_per_step: Seconds between adjacent sampled frames.

    Returns:
        Arrays of feature values keyed by feature name.
    """

    values: dict[str, list[float]] = {}
    for index in range(len(windows)):
        window = windows[index]
        sequence = windows.sequences[window.sequence_id]
        raw_keypoints = sequence.keypoints[window.frame_indices]
        keypoints = (
            windows.normalizer.inverse_transform(window.keypoints)
            if windows.normalizer is not None
            else window.keypoints
        )
        features = trajectory_window_features(
            keypoints,
            seconds_per_step=seconds_per_step,
        )
        features["missing_keypoint_fraction"] = float(
            np.all(raw_keypoints == 0.0, axis=-1).mean()
        )
        for name, value in features.items():
            values.setdefault(name, []).append(value)
    return {
        name: np.asarray(feature_values, dtype=np.float32)
        for name, feature_values in values.items()
    }


def compare_feature_distributions(
    *,
    train_features: dict[str, np.ndarray],
    validation_features: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compare matching scalar features across train and validation windows.

    Args:
        train_features: Feature values keyed by descriptive name.
        validation_features: Validation values with matching keys.

    Returns:
        Distribution summaries, standardized mean differences, and KS statistics.
    """

    comparison = {}
    for name, train_values in train_features.items():
        validation_values = validation_features[name]
        pooled_std = np.sqrt(
            (np.var(train_values, ddof=1) + np.var(validation_values, ddof=1)) / 2.0
        )
        mean_difference = float(validation_values.mean() - train_values.mean())
        comparison[name] = {
            "train_mean": float(train_values.mean()),
            "train_p50": float(np.percentile(train_values, 50)),
            "train_p99": float(np.percentile(train_values, 99)),
            "validation_mean": float(validation_values.mean()),
            "validation_p50": float(np.percentile(validation_values, 50)),
            "validation_p99": float(np.percentile(validation_values, 99)),
            "standardized_mean_difference": (
                mean_difference / float(pooled_std) if pooled_std > 0 else 0.0
            ),
            "ks_statistic": float(ks_2samp(train_values, validation_values).statistic),
        }
    return comparison
