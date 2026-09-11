"""File description: Visualization helpers for mouse-triplets pose sequences."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from constants import FRAME_HEIGHT, FRAME_WIDTH, KEYPOINT_NAMES, MOUSE_SKELETON_EDGES

MOUSE_COLORS = ("lawngreen", "skyblue", "tomato")
MOUSE_TRACK_COLORS = ("green", "blue", "red")
KEYPOINT_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "gold",
    "black",
)


def plot_pose_frame(
    ax: Axes,
    pose: np.ndarray,
    *,
    colors: tuple[str, str, str] = MOUSE_COLORS,
    track: np.ndarray | None = None,
    track_keypoint: int = 6,
    predicted_pose: np.ndarray | None = None,
    color_by_keypoint: bool = False,
    show_keypoint_legend: bool = False,
    show_all_keypoint_tracks: bool = False,
) -> None:
    """Plot one frame of mouse-triplets keypoints.

    Args:
        ax: Matplotlib axes that receives the frame.
        pose: Ground-truth pose shaped `[3, 12, 2]`.
        colors: Per-mouse colors used when `color_by_keypoint` is false.
        track: Optional history shaped `[time, 3, 12, 2]`.
        track_keypoint: Keypoint index to trail when drawing one track per mouse.
        predicted_pose: Optional predicted future pose shaped `[3, 12, 2]`.
        color_by_keypoint: Use stable body-part colors instead of mouse colors.
        show_keypoint_legend: Add a compact legend for the body-part colors.
        show_all_keypoint_tracks: Trail every keypoint, useful for eval videos.
    """

    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for mouse_idx, color in enumerate(colors):
        mouse_pose = pose[mouse_idx]
        point_colors = KEYPOINT_COLORS if color_by_keypoint else color
        ax.scatter(mouse_pose[:, 0], mouse_pose[:, 1], s=10, color=point_colors)
        for start, end in MOUSE_SKELETON_EDGES:
            segment = mouse_pose[[start, end]]
            line_color = KEYPOINT_COLORS[start] if color_by_keypoint else color
            ax.plot(segment[:, 0], segment[:, 1], color=line_color, linewidth=1)

        if track is not None:
            if show_all_keypoint_tracks:
                for keypoint_idx, track_color in enumerate(KEYPOINT_COLORS):
                    path = track[:, mouse_idx, keypoint_idx]
                    ax.plot(
                        path[:, 0],
                        path[:, 1],
                        color=track_color,
                        linewidth=0.8,
                        alpha=0.35,
                    )
            else:
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
                color=point_colors,
            )
            for start, end in MOUSE_SKELETON_EDGES:
                segment = predicted_mouse_pose[[start, end]]
                line_color = KEYPOINT_COLORS[start] if color_by_keypoint else color
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=line_color,
                    linewidth=1,
                    linestyle="--",
                )

    if show_keypoint_legend:
        _add_keypoint_legend(ax)


def _add_keypoint_legend(ax: Axes) -> None:
    """Add a compact keypoint-color legend to the frame.

    Args:
        ax: Matplotlib axes that receives the legend.
    """

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=name,
            markersize=4,
        )
        for name, color in zip(KEYPOINT_NAMES, KEYPOINT_COLORS, strict=True)
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=6,
        framealpha=0.85,
        borderpad=0.3,
        labelspacing=0.25,
    )


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
            color_by_keypoint=True,
            show_keypoint_legend=True,
            show_all_keypoint_tracks=True,
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
    "KEYPOINT_COLORS",
    "KEYPOINT_NAMES",
    "MOUSE_COLORS",
    "MOUSE_SKELETON_EDGES",
    "animate_pose_sequence",
    "animate_prediction_comparison",
    "plot_pose_frame",
    "save_animation",
]
