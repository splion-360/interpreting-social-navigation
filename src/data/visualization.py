"""File description: Visualization helpers for mouse-triplets pose sequences."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from data.schema import FRAME_HEIGHT, FRAME_WIDTH, KEYPOINT_NAMES, MOUSE_SKELETON_EDGES


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
    show_mouse_labels: bool = True,
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
        show_mouse_labels: Add mouse IDs near each mouse centroid.
    """

    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for mouse_idx in range(pose.shape[0]):
        color = colors[mouse_idx % len(colors)]
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
                    color=MOUSE_TRACK_COLORS[mouse_idx % len(MOUSE_TRACK_COLORS)],
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

        if show_mouse_labels:
            _add_mouse_label(ax, mouse_pose, str(mouse_idx), color=color)
            if predicted_pose is not None:
                _add_mouse_label(
                    ax,
                    predicted_pose[mouse_idx],
                    f"p{mouse_idx}",
                    color=color,
                    predicted=True,
                )

    if show_keypoint_legend:
        _add_keypoint_legend(ax)


def _add_mouse_label(
    ax: Axes,
    mouse_pose: np.ndarray,
    label: str,
    *,
    color: str,
    predicted: bool = False,
) -> None:
    """Add one mouse ID label near the mouse centroid.

    Args:
        ax: Matplotlib axes that receives the label.
        mouse_pose: Mouse keypoints shaped `[12, 2]`.
        label: Text label to draw.
        color: Label edge color.
        predicted: Whether the label marks a predicted mouse.
    """

    centroid = mouse_pose.mean(axis=0)
    ax.text(
        float(centroid[0]),
        float(centroid[1]),
        label,
        color="black",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "circle,pad=0.25",
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.85,
            "linestyle": "--" if predicted else "-",
        },
    )


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


def animate_single_mouse_prediction_comparison(
    actual_keypoints: np.ndarray,
    predicted_future_keypoints: np.ndarray,
    *,
    observation_length: int,
    mouse_index: int,
    sequence_id: str = "sequence",
    step: int = 1,
    interval_ms: int = 100,
    padding_px: float = 40.0,
) -> animation.FuncAnimation:
    """Animate one mouse as adjacent actual and rollout panels.

    Args:
        actual_keypoints: Ground-truth sequence shaped `[time, 3, 12, 2]`.
        predicted_future_keypoints: Predicted future shaped `[future, 3, 12, 2]`.
        observation_length: Number of observed frames before prediction starts.
        mouse_index: Mouse to visualize.
        sequence_id: Source sequence label shown in the title.
        step: Animation frame step.
        interval_ms: Matplotlib animation interval.
        padding_px: Extra pixels around the selected mouse trajectory.

    Returns:
        Matplotlib animation with two adjacent zoomed panels.
    """

    expected_future = actual_keypoints.shape[0] - observation_length
    if predicted_future_keypoints.shape[0] != expected_future:
        raise ValueError(
            "predicted_future_keypoints length must match the future horizon"
        )

    actual_mouse = actual_keypoints[:, mouse_index]
    predicted_future_mouse = predicted_future_keypoints[:, mouse_index]
    rollout_mouse = np.concatenate(
        (actual_mouse[:observation_length], predicted_future_mouse), axis=0
    )
    x_limits, y_limits = _mouse_zoom_limits(
        actual_mouse,
        predicted_future_mouse,
        padding_px=padding_px,
    )

    frame_indices = list(range(0, actual_keypoints.shape[0], step))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    def update(frame_idx: int) -> tuple[Axes, Axes]:
        phase = "observed" if frame_idx < observation_length else "future"
        actual_ax, rollout_ax = axes
        actual_ax.clear()
        rollout_ax.clear()
        _plot_single_mouse_frame(
            actual_ax,
            actual_mouse[frame_idx],
            track=actual_mouse[: frame_idx + 1],
            title=f"Actual ({phase})",
            x_limits=x_limits,
            y_limits=y_limits,
            mouse_label=str(mouse_index),
        )
        _plot_single_mouse_frame(
            rollout_ax,
            rollout_mouse[frame_idx],
            track=rollout_mouse[: frame_idx + 1],
            title="Observed context"
            if frame_idx < observation_length
            else "Prediction",
            x_limits=x_limits,
            y_limits=y_limits,
            predicted=frame_idx >= observation_length,
            show_keypoint_legend=True,
            mouse_label=str(mouse_index),
        )
        fig.suptitle(f"{sequence_id} mouse {mouse_index} frame {frame_idx}")
        return actual_ax, rollout_ax

    return animation.FuncAnimation(
        fig, update, frames=frame_indices, interval=interval_ms, blit=False
    )


def _plot_single_mouse_frame(
    ax: Axes,
    pose: np.ndarray,
    *,
    track: np.ndarray,
    title: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    predicted: bool = False,
    show_keypoint_legend: bool = False,
    show_mouse_label: bool = True,
    mouse_label: str = "0",
) -> None:
    """Plot one mouse frame inside fixed zoom limits.

    Args:
        ax: Matplotlib axes that receives the mouse frame.
        pose: Mouse keypoints shaped `[12, 2]`.
        track: Mouse trajectory history shaped `[time, 12, 2]`.
        title: Panel title.
        x_limits: Zoomed x-axis limits.
        y_limits: Zoomed y-axis limits.
        predicted: Draw pose with prediction markers when true.
        show_keypoint_legend: Add keypoint legend to this panel.
        show_mouse_label: Add the selected mouse ID to this panel.
        mouse_label: Mouse ID shown near the centroid.
    """

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

    for keypoint_idx, color in enumerate(KEYPOINT_COLORS):
        path = track[:, keypoint_idx]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.9, alpha=0.45)

    marker = "x" if predicted else "o"
    line_style = "--" if predicted else "-"
    ax.scatter(pose[:, 0], pose[:, 1], s=18, marker=marker, color=KEYPOINT_COLORS)
    for start, end in MOUSE_SKELETON_EDGES:
        segment = pose[[start, end]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=KEYPOINT_COLORS[start],
            linewidth=1.2,
            linestyle=line_style,
        )

    if show_mouse_label:
        _add_mouse_label(
            ax,
            pose,
            mouse_label,
            color="black",
            predicted=predicted,
        )

    if show_keypoint_legend:
        _add_keypoint_legend(ax)


def _mouse_zoom_limits(
    actual_mouse: np.ndarray,
    predicted_future_mouse: np.ndarray,
    *,
    padding_px: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute fixed zoom limits around one mouse rollout.

    Args:
        actual_mouse: Ground-truth mouse keypoints shaped `[time, 12, 2]`.
        predicted_future_mouse: Predicted mouse keypoints shaped `[future, 12, 2]`.
        padding_px: Extra pixels around the trajectory bounds.

    Returns:
        x-axis limits and reversed y-axis limits for image coordinates.
    """

    points = np.concatenate(
        (actual_mouse.reshape(-1, 2), predicted_future_mouse.reshape(-1, 2)),
        axis=0,
    )
    x_min, y_min = np.min(points, axis=0) - padding_px
    x_max, y_max = np.max(points, axis=0) + padding_px
    return (
        (max(0.0, float(x_min)), min(float(FRAME_WIDTH), float(x_max))),
        (min(float(FRAME_HEIGHT), float(y_max)), max(0.0, float(y_min))),
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
        raise ValueError(
            "predicted_future_keypoints length must match the future horizon"
        )

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


def save_animation(
    animation_obj: animation.FuncAnimation, path: str | Path, fps: int = 10
) -> Path:
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
    "animate_single_mouse_prediction_comparison",
    "plot_pose_frame",
    "save_animation",
]
