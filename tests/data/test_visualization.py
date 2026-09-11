"""File description: Tests for mouse-triplets visualization helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from constants import KEYPOINT_NAMES
from data.visualization import KEYPOINT_COLORS, plot_pose_frame


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

    plt.close(fig)
