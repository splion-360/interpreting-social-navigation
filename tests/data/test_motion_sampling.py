"""File description: Tests for motion scoring and stratified window sampling."""

import numpy as np

from data.motion_sampling import (
    DEFAULT_MOTION_MIX,
    MotionThresholds,
    build_motion_profile,
    mean_keypoint_speed_px_s,
    sample_motion_balanced_window_keys,
)


def test_mean_keypoint_speed_uses_future_pixel_steps() -> None:
    future = np.zeros((3, 3, 12, 2), dtype=np.float32)
    future[1, ..., 0] = 2.0
    future[2, ..., 0] = 4.0

    score = mean_keypoint_speed_px_s(future, seconds_per_step=0.5)

    assert score == 4.0


def test_motion_profile_applies_train_fitted_thresholds() -> None:
    class FakeWindow:
        def __init__(self, speed: float) -> None:
            self.future_keypoints = np.zeros((2, 3, 12, 2), dtype=np.float32)
            self.future_keypoints[1, ..., 0] = speed

    class FakeWindows:
        def __init__(self, speeds: list[float]) -> None:
            self.windows = [FakeWindow(speed) for speed in speeds]

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int) -> FakeWindow:
            return self.windows[index]

    profile = build_motion_profile(
        FakeWindows([1.0, 5.0, 10.0]),  # type: ignore[arg-type]
        thresholds=MotionThresholds(low_max_px_s=2.0, medium_max_px_s=6.0),
        seconds_per_step=1.0,
    )

    assert profile.labels == ("low", "medium", "high")
    assert profile.counts == {"low": 1, "medium": 1, "high": 1}


def test_motion_balanced_sampling_uses_configured_mix_and_seed() -> None:
    window_keys = tuple((f"seq_{index}", index) for index in range(50))
    labels = (
        ("low",) * 10
        + ("medium",) * 20
        + ("high",) * 20
    )

    selected = sample_motion_balanced_window_keys(
        window_keys=window_keys,
        labels=labels,
        mix=DEFAULT_MOTION_MIX,
        max_windows=10,
        seed=42,
    )
    selected_again = sample_motion_balanced_window_keys(
        window_keys=window_keys,
        labels=labels,
        mix=DEFAULT_MOTION_MIX,
        max_windows=10,
        seed=42,
    )

    selected_labels = [labels[window_keys.index(key)] for key in selected]
    assert selected == selected_again
    assert selected_labels.count("low") == 2
    assert selected_labels.count("medium") == 4
    assert selected_labels.count("high") == 4
