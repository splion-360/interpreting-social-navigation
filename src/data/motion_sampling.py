"""File description: Motion scoring and stratified window sampling utilities."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from data.mabe import MabeWindowDataset


MotionStratum = Literal["low", "medium", "high"]
MOTION_STRATA: tuple[MotionStratum, ...] = ("low", "medium", "high")
DEFAULT_MOTION_MIX = {"low": 0.2, "medium": 0.4, "high": 0.4}
_WORKER_WINDOWS: MabeWindowDataset | None = None
_WORKER_SECONDS_PER_STEP = 1.0


@dataclass(frozen=True)
class MotionThresholds:
    """Train-fitted motion thresholds.

    Attributes:
        low_max_px_s: Maximum mean keypoint speed for low-motion windows.
        medium_max_px_s: Maximum mean keypoint speed for medium-motion windows.
    """

    low_max_px_s: float
    medium_max_px_s: float

    def label(self, score_px_s: float) -> MotionStratum:
        """Return the motion stratum for a score.

        Args:
            score_px_s: Mean keypoint speed in pixels per second.

        Returns:
            Low, medium, or high motion label.
        """

        if score_px_s <= self.low_max_px_s:
            return "low"
        if score_px_s <= self.medium_max_px_s:
            return "medium"
        return "high"

    def to_dict(self) -> dict[str, float]:
        """Return JSON-ready threshold values."""

        return {
            "low_max": self.low_max_px_s,
            "medium_max": self.medium_max_px_s,
        }


@dataclass(frozen=True)
class MotionProfile:
    """Motion labels and thresholds for a fixed window collection.

    Attributes:
        labels: Per-window motion labels.
        thresholds: Thresholds used to assign labels.
        scores_px_s: Per-window mean keypoint speed scores.
    """

    labels: tuple[MotionStratum, ...]
    thresholds: MotionThresholds
    scores_px_s: tuple[float, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Return the number of windows assigned to each stratum."""

        return {name: self.labels.count(name) for name in MOTION_STRATA}

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready motion-profile metadata."""

        return {
            "score": "mean_keypoint_speed_px_s",
            "thresholds_px_s": self.thresholds.to_dict(),
            "counts": self.counts,
        }


def mean_keypoint_speed_px_s(
    target_future: np.ndarray,
    *,
    seconds_per_step: float,
) -> float:
    """Compute mean keypoint speed within the ground-truth future horizon.

    Args:
        target_future: Future keypoints shaped `[time, mice, keypoints, 2]`.
        seconds_per_step: Seconds between adjacent sampled frames.

    Returns:
        Mean keypoint speed in pixels per second.
    """

    if target_future.shape[0] < 2:
        return 0.0
    deltas = np.diff(target_future.astype(np.float32), axis=0)
    speeds = np.linalg.norm(deltas, axis=-1) / seconds_per_step
    return float(speeds.mean())


def window_motion_scores_px_s(
    windows: MabeWindowDataset,
    *,
    seconds_per_step: float,
    show_progress: bool = False,
    desc: str = "Scoring motion windows",
    workers: int = 0,
) -> np.ndarray:
    """Score each window by future mean keypoint speed.

    Args:
        windows: Pixel-space windows to score.
        seconds_per_step: Seconds between adjacent sampled frames.
        show_progress: Whether to show scoring progress.
        desc: Progress-bar description.
        workers: Number of forked worker processes. Use `0` or `1` for serial
            scoring.

    Returns:
        One score per window in pixels per second.
    """

    if workers > 1:
        return _parallel_window_motion_scores_px_s(
            windows,
            seconds_per_step=seconds_per_step,
            show_progress=show_progress,
            desc=desc,
            workers=workers,
        )
    iterator = tqdm(
        range(len(windows)),
        desc=desc,
        unit="window",
        disable=not show_progress,
    )
    return np.asarray(
        [
            mean_keypoint_speed_px_s(
                windows[index].future_keypoints,
                seconds_per_step=seconds_per_step,
            )
            for index in iterator
        ],
        dtype=np.float32,
    )


def fit_motion_thresholds(scores_px_s: np.ndarray) -> MotionThresholds:
    """Fit low/medium/high boundaries from training scores.

    Args:
        scores_px_s: Training-candidate window scores in pixels per second.

    Returns:
        Tertile thresholds fit only from the provided training scores.
    """

    low_max, medium_max = np.quantile(scores_px_s, (1.0 / 3.0, 2.0 / 3.0))
    return MotionThresholds(low_max_px_s=float(low_max), medium_max_px_s=float(medium_max))


def build_motion_profile(
    windows: MabeWindowDataset,
    *,
    thresholds: MotionThresholds,
    seconds_per_step: float,
    show_progress: bool = False,
    desc: str = "Building motion profile",
    workers: int = 0,
) -> MotionProfile:
    """Assign motion labels to a fixed window collection.

    Args:
        windows: Pixel-space windows to label.
        thresholds: Train-fitted thresholds.
        seconds_per_step: Seconds between adjacent sampled frames.
        show_progress: Whether to show scoring progress.
        desc: Progress-bar description.
        workers: Number of forked worker processes. Use `0` or `1` for serial
            scoring.

    Returns:
        Motion labels, thresholds, and scores for the window collection.
    """

    scores = window_motion_scores_px_s(
        windows,
        seconds_per_step=seconds_per_step,
        show_progress=show_progress,
        desc=desc,
        workers=workers,
    )
    return MotionProfile(
        labels=tuple(thresholds.label(float(score)) for score in scores),
        thresholds=thresholds,
        scores_px_s=tuple(float(score) for score in scores),
    )


def sample_motion_balanced_window_keys(
    *,
    window_keys: Sequence[tuple[str, int]],
    labels: Sequence[MotionStratum],
    mix: Mapping[str, float],
    max_windows: int | None,
    seed: int,
) -> tuple[tuple[str, int], ...]:
    """Select deterministic training windows with a configured motion mix.

    Args:
        window_keys: Candidate `(sequence_id, start_frame)` pointers.
        labels: Motion label for each candidate key.
        mix: Desired low/medium/high proportions.
        max_windows: Maximum selected windows after grouping.
        seed: Random seed for deterministic sampling and shuffling.

    Returns:
        Sampled window keys in deterministic shuffled order.
    """

    rng = np.random.default_rng(seed)
    if max_windows is None:
        indices = rng.permutation(len(window_keys))
        return tuple(window_keys[int(index)] for index in indices)

    grouped = {
        stratum: [index for index, label in enumerate(labels) if label == stratum]
        for stratum in MOTION_STRATA
    }
    selected: list[int] = []
    for stratum, count in _allocate_counts(max_windows, mix).items():
        candidates = np.asarray(grouped[stratum], dtype=np.int64)
        if candidates.size == 0:
            continue
        take = min(count, int(candidates.size))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())

    if len(selected) < max_windows:
        remaining = np.asarray(
            [index for index in range(len(window_keys)) if index not in set(selected)],
            dtype=np.int64,
        )
        take = min(max_windows - len(selected), int(remaining.size))
        if take:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())

    shuffled = np.asarray(selected, dtype=np.int64)
    rng.shuffle(shuffled)
    return tuple(window_keys[int(index)] for index in shuffled)


def _parallel_window_motion_scores_px_s(
    windows: MabeWindowDataset,
    *,
    seconds_per_step: float,
    show_progress: bool,
    desc: str,
    workers: int,
) -> np.ndarray:
    """Score windows with forked worker processes.

    Args:
        windows: Pixel-space windows to score.
        seconds_per_step: Seconds between sampled frames.
        show_progress: Whether to show scoring progress.
        desc: Progress-bar description.
        workers: Number of worker processes.

    Returns:
        One score per window in pixels per second.
    """

    context = mp.get_context("fork")
    chunksize = max(1, len(windows) // (workers * 16))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_motion_score_worker,
        initargs=(windows, seconds_per_step),
    ) as executor:
        scores = executor.map(
            _score_motion_window_index,
            range(len(windows)),
            chunksize=chunksize,
        )
        iterator = tqdm(
            scores,
            total=len(windows),
            desc=f"{desc} ({workers} workers)",
            unit="window",
            disable=not show_progress,
        )
        return np.fromiter(iterator, dtype=np.float32, count=len(windows))


def _init_motion_score_worker(
    windows: MabeWindowDataset,
    seconds_per_step: float,
) -> None:
    """Store scoring inputs once per worker process."""

    global _WORKER_WINDOWS, _WORKER_SECONDS_PER_STEP
    _WORKER_WINDOWS = windows
    _WORKER_SECONDS_PER_STEP = seconds_per_step


def _score_motion_window_index(index: int) -> float:
    """Score one worker-owned window by mean keypoint speed."""

    if _WORKER_WINDOWS is None:
        raise RuntimeError("motion scoring worker was not initialized")
    return mean_keypoint_speed_px_s(
        _WORKER_WINDOWS[index].future_keypoints,
        seconds_per_step=_WORKER_SECONDS_PER_STEP,
    )


def _allocate_counts(
    total: int,
    mix: Mapping[str, float],
) -> dict[MotionStratum, int]:
    """Allocate an integer sample count to each motion stratum."""

    raw = {stratum: total * float(mix[stratum]) for stratum in MOTION_STRATA}
    counts = {stratum: int(np.floor(value)) for stratum, value in raw.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(
        MOTION_STRATA,
        key=lambda stratum: raw[stratum] - counts[stratum],
        reverse=True,
    )
    for stratum in ranked[:remainder]:
        counts[stratum] += 1
    return counts
