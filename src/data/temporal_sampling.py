"""File description: Temporal sampling diagnostics for MABe windows."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from data.mabe import (
    MabeSequence,
    MabeWindowDataset,
    WindowSpec,
    fill_missing_keypoints,
)


@dataclass(frozen=True)
class TemporalSamplingDiagnostics:
    """Summary of motion retained after temporal downsampling.

    Attributes:
        source_fps: Source dataset frame rate.
        frame_step: Raw-frame gap between sampled time steps.
        effective_fps: Frame rate after downsampling.
        observation_seconds: Duration represented by observed samples.
        prediction_seconds: Duration represented by future samples.
        raw_span: Raw frames needed for one sampled window.
        windows: Number of windows inspected.
        motion_score_correlation: Correlation between raw and sampled mean
            keypoint speed.
        retained_motion_ratio_mean: Mean sampled/raw path-length ratio.
        retained_motion_ratio_median: Median sampled/raw path-length ratio.
        retained_motion_ratio_p10: Tenth-percentile sampled/raw path-length ratio.
        retained_motion_ratio_p90: Ninetieth-percentile sampled/raw path-length ratio.
        worst_windows: Windows with the lowest retained-motion ratio.
    """

    source_fps: float
    frame_step: int
    effective_fps: float
    observation_seconds: float
    prediction_seconds: float
    raw_span: int
    windows: int
    motion_score_correlation: float
    retained_motion_ratio_mean: float
    retained_motion_ratio_median: float
    retained_motion_ratio_p10: float
    retained_motion_ratio_p90: float
    worst_windows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the diagnostics as JSON-ready values."""

        return asdict(self)


def temporal_sampling_diagnostics(
    sequences: list[MabeSequence],
    *,
    spec: WindowSpec,
    source_fps: float,
    max_windows: int | None = None,
    worst_count: int = 5,
) -> TemporalSamplingDiagnostics:
    """Compare raw and downsampled motion over identical source spans.

    Args:
        sequences: Source MABe sequences to inspect.
        spec: Window specification including the downsampling frame step.
        source_fps: Source dataset frame rate before downsampling.
        max_windows: Optional cap for fast notebook diagnostics.
        worst_count: Number of lowest-retention windows to include.

    Returns:
        Standard downsampling diagnostics based on path-length retention.
    """

    windows = MabeWindowDataset(
        sequences,
        spec,
        fill_missing=False,
        max_windows=max_windows,
    )
    raw_scores: list[float] = []
    sampled_scores: list[float] = []
    retained_ratios: list[float] = []
    inspected: list[dict[str, Any]] = []
    duration_seconds = (spec.raw_span - 1) / source_fps

    for sequence_id, start_frame in windows.window_keys:
        sequence = windows.sequences[sequence_id]
        raw_indices = np.arange(start_frame, start_frame + spec.raw_span)
        sampled_indices = start_frame + np.arange(spec.length) * spec.frame_step
        raw_keypoints = fill_missing_keypoints(sequence.keypoints[raw_indices])
        sampled_keypoints = fill_missing_keypoints(sequence.keypoints[sampled_indices])
        raw_path = keypoint_path_length_px(raw_keypoints)
        sampled_path = keypoint_path_length_px(sampled_keypoints)
        raw_score = raw_path / duration_seconds
        sampled_score = sampled_path / duration_seconds
        retained_ratio = sampled_path / raw_path if raw_path > 0 else np.nan

        raw_scores.append(raw_score)
        sampled_scores.append(sampled_score)
        retained_ratios.append(float(retained_ratio))
        inspected.append(
            {
                "sequence_id": sequence_id,
                "start_frame": start_frame,
                "first_sampled_frame": int(sampled_indices[0]),
                "last_sampled_frame": int(sampled_indices[-1]),
                "raw_path_length_px": raw_path,
                "sampled_path_length_px": sampled_path,
                "retained_motion_ratio": float(retained_ratio),
            }
        )

    ratios = np.asarray(retained_ratios, dtype=np.float32)
    finite_ratios = ratios[np.isfinite(ratios)]
    worst = sorted(
        (item for item in inspected if np.isfinite(item["retained_motion_ratio"])),
        key=lambda item: item["retained_motion_ratio"],
    )[:worst_count]

    ratio_summary = _ratio_summary(finite_ratios)
    return TemporalSamplingDiagnostics(
        source_fps=source_fps,
        frame_step=spec.frame_step,
        effective_fps=source_fps / spec.frame_step,
        observation_seconds=spec.observation_length * spec.frame_step / source_fps,
        prediction_seconds=spec.prediction_length * spec.frame_step / source_fps,
        raw_span=spec.raw_span,
        windows=len(windows),
        motion_score_correlation=_correlation(raw_scores, sampled_scores),
        retained_motion_ratio_mean=ratio_summary["mean"],
        retained_motion_ratio_median=ratio_summary["median"],
        retained_motion_ratio_p10=ratio_summary["p10"],
        retained_motion_ratio_p90=ratio_summary["p90"],
        worst_windows=tuple(worst),
    )


def keypoint_path_length_px(keypoints: np.ndarray) -> float:
    """Return mean keypoint path length over one pose sequence.

    Args:
        keypoints: Pose array shaped `[time, mice, keypoints, 2]`.

    Returns:
        Mean path length across mice and keypoints in pixels.
    """

    step_distances = np.linalg.norm(np.diff(keypoints.astype(np.float32), axis=0), axis=-1)
    return float(step_distances.sum(axis=0).mean())


def _correlation(left: list[float], right: list[float]) -> float:
    """Return Pearson correlation, or `nan` when it is undefined."""

    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if len(left_array) < 2 or left_array.std() == 0.0 or right_array.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _ratio_summary(ratios: np.ndarray) -> dict[str, float]:
    """Return retained-motion ratio summaries with undefined cases as `nan`."""

    if ratios.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(np.mean(ratios)),
        "median": float(np.median(ratios)),
        "p10": float(np.percentile(ratios, 10)),
        "p90": float(np.percentile(ratios, 90)),
    }
