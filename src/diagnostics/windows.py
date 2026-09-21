"""File description: Matched window construction for diagnostic comparisons."""

import hashlib
from dataclasses import dataclass

import numpy as np

from data import (
    MabeDataset,
    MabeSequence,
    MabeWindowDataset,
    PoseNormalizer,
    WindowSpec,
    mean_keypoint_speed_px_s,
    single_mouse_sequence_id,
    to_single_mouse_sequences,
)
from evaluate import (
    KEYPOINT_GRAPH_VARIANTS,
    TestConfig,
    _fit_training_normalizer,
    _seconds_per_step,
)
from train import FlatFitConfig


@dataclass(frozen=True)
class MatchedMouseWindow:
    """One paired dense/single evaluation case.

    Attributes:
        sequence_id: Source triplet sequence ID.
        start_frame: Source start frame.
        mouse_index: Mouse index compared in both variants.
        dense_index: Index into the dense triplet window dataset.
        single_index: Index into the single-mouse window dataset.
    """

    sequence_id: str
    start_frame: int
    mouse_index: int
    dense_index: int
    single_index: int


@dataclass(frozen=True)
class MatchedWindowData:
    """Window datasets and pairing metadata for a fair variant comparison.

    Attributes:
        dense_windows: Triplet windows keyed by source sequence and start.
        single_windows: Single-mouse windows keyed by source, mouse, and start.
        cases: Mouse-specific pairings between the two datasets.
        normalizer: Pixel normalizer fitted from the training split.
        motion_labels: Shared low/medium/high labels for each case.
        motion_scores_px_s: Ground-truth motion scores for each case.
        motion_thresholds: Low and medium boundaries fitted on selected cases.
    """

    dense_windows: MabeWindowDataset
    single_windows: MabeWindowDataset
    cases: tuple[MatchedMouseWindow, ...]
    normalizer: PoseNormalizer
    motion_labels: tuple[str, ...]
    motion_scores_px_s: tuple[float, ...]
    motion_thresholds: tuple[float, float]


def build_matched_window_data(
    *,
    dense_config: FlatFitConfig,
    single_config: FlatFitConfig,
    test_config: TestConfig,
    max_triplet_windows: int,
    seed: int,
) -> MatchedWindowData:
    """Build shared mouse-window cases for dense and single-mouse variants.

    Args:
        dense_config: Dense triplet train config.
        single_config: Single-mouse train config.
        test_config: Held-out test config.
        max_triplet_windows: Number of triplet windows to sample.
        seed: Deterministic selection seed.

    Returns:
        Datasets plus an exact pairing between triplet and single-mouse windows.
    """

    _ensure_comparable_window_contracts(dense_config, single_config)
    normalizer = _fit_training_normalizer(dense_config)
    test_dataset = MabeDataset.from_file(test_config.data_path)
    source_sequences = test_dataset.select(test_dataset.sequence_ids)
    spec = _window_spec(dense_config)
    dense_keys = _sample_triplet_window_keys(
        sequences=source_sequences,
        spec=spec,
        max_triplet_windows=max_triplet_windows,
        seed=seed,
    )
    single_keys = tuple(
        (single_mouse_sequence_id(sequence_id, mouse_index), start_frame)
        for sequence_id, start_frame in dense_keys
        for mouse_index in range(3)
    )
    dense_windows = MabeWindowDataset(
        source_sequences,
        spec,
        normalizer=normalizer,
        window_keys=dense_keys,
    )
    single_windows = MabeWindowDataset(
        to_single_mouse_sequences(source_sequences),
        spec,
        normalizer=normalizer,
        window_keys=single_keys,
    )
    dense_index_by_key = {
        key: index for index, key in enumerate(dense_windows.window_keys)
    }
    single_index_by_key = {
        key: index for index, key in enumerate(single_windows.window_keys)
    }
    cases = tuple(
        MatchedMouseWindow(
            sequence_id=sequence_id,
            start_frame=start_frame,
            mouse_index=mouse_index,
            dense_index=dense_index_by_key[(sequence_id, start_frame)],
            single_index=single_index_by_key[
                (single_mouse_sequence_id(sequence_id, mouse_index), start_frame)
            ],
        )
        for sequence_id, start_frame in dense_keys
        for mouse_index in range(3)
    )
    scores = _case_motion_scores(
        dense_windows=dense_windows,
        cases=cases,
        normalizer=normalizer,
        seconds_per_step=_seconds_per_step(dense_config),
    )
    low_max, medium_max = np.quantile(scores, (1.0 / 3.0, 2.0 / 3.0))
    labels = tuple(
        _motion_label(score, low_max=low_max, medium_max=medium_max) for score in scores
    )
    return MatchedWindowData(
        dense_windows=dense_windows,
        single_windows=single_windows,
        cases=cases,
        normalizer=normalizer,
        motion_labels=labels,
        motion_scores_px_s=tuple(float(score) for score in scores),
        motion_thresholds=(float(low_max), float(medium_max)),
    )


def matched_window_digest(cases: tuple[MatchedMouseWindow, ...]) -> str:
    """Fingerprint the paired case selection."""

    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.sequence_id.encode())
        digest.update(b"\0")
        digest.update(case.start_frame.to_bytes(8, byteorder="big", signed=False))
        digest.update(case.mouse_index.to_bytes(2, byteorder="big", signed=False))
    return digest.hexdigest()


def _ensure_comparable_window_contracts(
    dense_config: FlatFitConfig,
    single_config: FlatFitConfig,
) -> None:
    """Ensure variants use the same temporal evaluation contract."""

    fields = (
        "window_length",
        "observation_length",
        "prediction_length",
        "stride",
        "frame_step",
        "source_fps",
    )
    mismatches = [
        field
        for field in fields
        if getattr(dense_config, field) != getattr(single_config, field)
    ]
    if mismatches:
        raise ValueError(f"diagnostic configs differ on: {', '.join(mismatches)}")
    if dense_config.graph_variant not in KEYPOINT_GRAPH_VARIANTS:
        raise ValueError("dense diagnostic variant must use keypoint graphs")
    if single_config.graph_variant != "single_mouse_dense_keypoint":
        raise ValueError(
            "single diagnostic variant must be single_mouse_dense_keypoint"
        )


def _window_spec(config: FlatFitConfig) -> WindowSpec:
    """Build a window spec from a train config."""

    return WindowSpec(
        length=config.window_length,
        observation_length=config.observation_length,
        prediction_length=config.prediction_length,
        stride=config.stride,
        frame_step=config.frame_step,
    )


def _sample_triplet_window_keys(
    *,
    sequences: list[MabeSequence],
    spec: WindowSpec,
    max_triplet_windows: int,
    seed: int,
) -> tuple[tuple[str, int], ...]:
    """Select triplet windows uniformly across all eligible test windows."""

    all_windows = MabeWindowDataset(sequences, spec)
    rng = np.random.default_rng(seed)
    count = min(max_triplet_windows, len(all_windows))
    indices = rng.choice(len(all_windows), size=count, replace=False)
    return tuple(all_windows.window_keys[int(index)] for index in indices)


def _case_motion_scores(
    *,
    dense_windows: MabeWindowDataset,
    cases: tuple[MatchedMouseWindow, ...],
    normalizer: PoseNormalizer,
    seconds_per_step: float,
) -> np.ndarray:
    """Score matched mouse cases by ground-truth future keypoint speed."""

    scores = []
    for case in cases:
        window = dense_windows[case.dense_index]
        target = normalizer.inverse_transform(
            window.future_keypoints[:, case.mouse_index : case.mouse_index + 1]
        )
        scores.append(
            mean_keypoint_speed_px_s(target, seconds_per_step=seconds_per_step)
        )
    return np.asarray(scores, dtype=np.float32)


def _motion_label(score: float, *, low_max: float, medium_max: float) -> str:
    """Return a tertile motion label for one score."""

    if score <= low_max:
        return "low"
    if score <= medium_max:
        return "medium"
    return "high"
