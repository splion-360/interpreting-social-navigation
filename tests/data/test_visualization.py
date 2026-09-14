"""File description: Tests for mouse-triplets visualization helpers."""

from typing import Any, cast

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data.schema import KEYPOINT_NAMES
from data.visualization import (
    KEYPOINT_COLORS,
    animate_prediction_comparison,
    animate_single_mouse_prediction_comparison,
    plot_pose_frame,
)


def test_plot_pose_frame_can_color_and_label_each_keypoint() -> None:
    pose = np.arange(3 * 12 * 2, dtype=np.float32).reshape(3, 12, 2)
    fig, ax = plt.subplots()

    plot_pose_frame(
        ax,
        pose,
        track=pose[None, ...],
        predicted_pose=pose + 1,
        color_by_keypoint=True,
        show_keypoint_legend=True,
        show_all_keypoint_tracks=True,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == list(KEYPOINT_NAMES)
    assert len(KEYPOINT_COLORS) == len(KEYPOINT_NAMES)
    assert {text.get_text() for text in ax.texts} == {"0", "1", "2", "p0", "p1", "p2"}

    plt.close(fig)


def test_single_mouse_prediction_animation_has_two_zoomed_panels() -> None:
    actual = np.arange(20 * 3 * 12 * 2, dtype=np.float32).reshape(20, 3, 12, 2)
    predicted = actual[8:, ...] + np.float32(2.0)

    animation = animate_single_mouse_prediction_comparison(
        actual_keypoints=actual,
        predicted_future_keypoints=predicted,
        observation_length=8,
        mouse_index=1,
        sequence_id="seq",
    )

    animation_state = cast(Any, animation)
    animation_state._func(8)
    axes = animation_state._fig.axes
    assert len(axes) == 2
    assert axes[0].get_title() == "Actual (future)"
    assert axes[1].get_title() == "Prediction"
    assert [text.get_text() for text in axes[0].texts] == ["1"]
    assert [text.get_text() for text in axes[1].texts] == ["1"]
    assert axes[1].get_legend() is not None

    animation_state._draw_was_started = True
    plt.close(animation_state._fig)


def test_prediction_animation_supports_single_mouse_pose() -> None:
    actual = np.arange(20 * 1 * 12 * 2, dtype=np.float32).reshape(20, 1, 12, 2)
    predicted = actual[8:, ...] + np.float32(2.0)

    animation = animate_prediction_comparison(
        actual_keypoints=actual,
        predicted_future_keypoints=predicted,
        observation_length=8,
        sequence_id="seq__mouse_0",
    )

    animation_state = cast(Any, animation)
    animation_state._func(8)
    axes = animation_state._fig.axes
    assert [text.get_text() for text in axes[0].texts] == ["0", "p0"]

    animation_state._draw_was_started = True
    plt.close(animation_state._fig)
