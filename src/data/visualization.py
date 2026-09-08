"""File description: Visualization helpers for mouse-triplets pose sequences."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes

from config.mabe import FRAME_HEIGHT, FRAME_WIDTH, KEYPOINT_NAMES

MOUSE_COLORS = ("lawngreen", "skyblue", "tomato")
MOUSE_TRACK_COLORS = ("green", "blue", "red")

MOUSE_SKELETON_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (3, 6),
    (6, 9),
    (9, 10),
    (10, 11),
    (4, 5),
    (5, 8),
    (8, 9),
    (9, 7),
    (7, 4),
)


def plot_pose_frame(
    ax: Axes,
    pose: np.ndarray,
    *,
    colors: tuple[str, str, str] = MOUSE_COLORS,
    track: np.ndarray | None = None,
    track_keypoint: int = 6,
    predicted_pose: np.ndarray | None = None,
) -> None:
    """Plot one frame shaped [3, 12, 2]."""

    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for mouse_idx, color in enumerate(colors):
        mouse_pose = pose[mouse_idx]
        ax.scatter(mouse_pose[:, 0], mouse_pose[:, 1], s=10, color=color)
        for start, end in MOUSE_SKELETON_EDGES:
            segment = mouse_pose[[start, end]]
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1)

        if track is not None:
            path = track[:, mouse_idx, track_keypoint]
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=MOUSE_TRACK_COLORS[mouse_idx],
                linewidth=1,
                alpha=0.7,
            )

        if predicted_pose is not None:
            predicted_mouse_pose = predicted_pose[mouse_idx]
            ax.scatter(
                predicted_mouse_pose[:, 0],
                predicted_mouse_pose[:, 1],
                s=14,
                marker="x",
                color=color,
            )
            for start, end in MOUSE_SKELETON_EDGES:
                segment = predicted_mouse_pose[[start, end]]
                ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1, linestyle="--")


def animate_pose_sequence(
    keypoints: np.ndarray,
    *,
    sequence_id: str = "sequence",
    start_frame: int = 0,
    stop_frame: int | None = None,
    step: int = 10,
    interval_ms: int = 100,
    show_track: bool = True,
) -> animation.FuncAnimation:
    """Create a matplotlib animation from keypoints shaped [frames, 3, 12, 2]."""

    if stop_frame is None:
        stop_frame = keypoints.shape[0]
    frame_indices = list(range(start_frame, min(stop_frame, keypoints.shape[0]), step))
    if not frame_indices:
        raise ValueError("animation frame range is empty")

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx: int) -> tuple[Axes]:
        ax.clear()
        track = keypoints[: frame_idx + 1] if show_track else None
        plot_pose_frame(ax, keypoints[frame_idx], track=track)
        ax.set_title(f"{sequence_id} frame {frame_idx}")
        return (ax,)

    return animation.FuncAnimation(
        fig, update, frames=frame_indices, interval=interval_ms, blit=False
    )


def animate_prediction_comparison(
    actual_keypoints: np.ndarray,
    predicted_future_keypoints: np.ndarray,
    *,
    observation_length: int,
    sequence_id: str = "sequence",
    step: int = 1,
    interval_ms: int = 100,
) -> animation.FuncAnimation:
    """Animate observed history, actual future, and predicted future poses.

    `actual_keypoints` must be shaped [observed + future, 3, 12, 2].
    `predicted_future_keypoints` must be shaped [future, 3, 12, 2].
    """

    expected_future = actual_keypoints.shape[0] - observation_length
    if predicted_future_keypoints.shape[0] != expected_future:
        raise ValueError("predicted_future_keypoints length must match the future horizon")

    frame_indices = list(range(0, actual_keypoints.shape[0], step))
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx: int) -> tuple[Axes]:
        ax.clear()
        predicted_pose = None
        if frame_idx >= observation_length:
            predicted_pose = predicted_future_keypoints[frame_idx - observation_length]
        plot_pose_frame(
            ax,
            actual_keypoints[frame_idx],
            track=actual_keypoints[: frame_idx + 1],
            predicted_pose=predicted_pose,
        )
        phase = "observed" if frame_idx < observation_length else "future"
        ax.set_title(f"{sequence_id} frame {frame_idx} ({phase})")
        return (ax,)

    return animation.FuncAnimation(
        fig, update, frames=frame_indices, interval=interval_ms, blit=False
    )


def save_animation(animation_obj: animation.FuncAnimation, path: str | Path, fps: int = 10) -> Path:
    """Save an animation to disk using matplotlib's configured writers."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation_obj.save(output_path, fps=fps)
    return output_path


__all__ = [
    "KEYPOINT_NAMES",
    "MOUSE_COLORS",
    "MOUSE_SKELETON_EDGES",
    "animate_pose_sequence",
    "animate_prediction_comparison",
    "plot_pose_frame",
    "save_animation",
]
