"""File description: MABe mouse-triplets data loading and window sampling."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.schema import (
    COORDINATES,
    DEFAULT_FRAME_STEP,
    DEFAULT_OBSERVATION_LENGTH,
    DEFAULT_PREDICTION_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SPLIT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_WINDOW_STRIDE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)


@dataclass(frozen=True)
class MabeSequence:
    """One MABe mouse-triplets sequence.

    Attributes:
        sequence_id: Stable sequence identifier from the source dataset.
        keypoints: Pose array shaped `[frames, 3, 12, 2]`.
        annotations: Optional label array shaped `[labels, frames]`.
    """

    sequence_id: str
    keypoints: np.ndarray
    annotations: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])


@dataclass(frozen=True)
class MabeDataset:
    """Loaded MABe mouse-triplets dictionary.

    Attributes:
        sequences: Mapping from sequence ID to parsed sequence.
        vocabulary: Label names from the source file, such as `chases` and `lights`.
    """

    sequences: dict[str, MabeSequence]
    vocabulary: tuple[str, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> MabeDataset:
        """Load a CaltechDATA/AIcrowd MABe `.npy` dictionary.

        Args:
            path: Path to a MABe file such as `mouse_triplet_train.npy`.

        Returns:
            Parsed dataset with source sequence IDs preserved.
        """

        raw = np.load(Path(path), allow_pickle=True).item()
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> MabeDataset:
        """Create a dataset from an already-loaded MABe dictionary.

        Args:
            raw: Dictionary with `sequences` and optional `vocabulary` keys.

        Returns:
            Parsed dataset.
        """

        sequences = {}
        for sequence_id, sequence in raw["sequences"].items():
            sequences[str(sequence_id)] = MabeSequence(
                sequence_id=str(sequence_id),
                keypoints=np.asarray(sequence["keypoints"]),
                annotations=(
                    np.asarray(sequence["annotations"])
                    if "annotations" in sequence and sequence["annotations"] is not None
                    else None
                ),
            )

        vocabulary = tuple(str(item) for item in raw.get("vocabulary", ()))
        return cls(sequences=sequences, vocabulary=vocabulary)

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.sequences))

    def select(self, sequence_ids: Sequence[str]) -> list[MabeSequence]:
        """Return sequences in the requested order.

        Args:
            sequence_ids: Source sequence IDs to select.

        Returns:
            List of matching sequences.
        """

        return [self.sequences[sequence_id] for sequence_id in sequence_ids]


@dataclass(frozen=True)
class WindowSpec:
    """Windowing policy for trajectory forecasting examples.

    Attributes:
        length: Total number of frames per window.
        stride: Raw-frame gap between consecutive window starts.
        frame_step: Raw-frame gap between sampled frames inside one window.
        observation_length: Number of conditioning frames.
        prediction_length: Number of future frames.
    """

    length: int = DEFAULT_SEQUENCE_LENGTH
    stride: int = DEFAULT_WINDOW_STRIDE
    frame_step: int = DEFAULT_FRAME_STEP
    observation_length: int = DEFAULT_OBSERVATION_LENGTH
    prediction_length: int = DEFAULT_PREDICTION_LENGTH

    def __post_init__(self) -> None:
        if self.observation_length + self.prediction_length != self.length:
            raise ValueError("observation_length + prediction_length must equal length")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.frame_step < 1:
            raise ValueError("frame_step must be at least 1")

    @property
    def raw_span(self) -> int:
        """Return raw frames needed to sample one complete window."""

        return (self.length - 1) * self.frame_step + 1


@dataclass(frozen=True)
class Window:
    """A contiguous trajectory window.

    Attributes:
        sequence_id: Source sequence ID.
        start_frame: First frame index in the source sequence.
        frame_indices: Raw source-frame indices sampled into this window.
        keypoints: Window keypoints shaped `[time, 3, 12, 2]`.
        observation_length: Number of observed frames before the prediction horizon.
        annotations: Optional labels sliced to the same time span.
    """

    sequence_id: str
    start_frame: int
    frame_indices: np.ndarray
    keypoints: np.ndarray
    observation_length: int
    annotations: np.ndarray | None = None

    @property
    def observed_keypoints(self) -> np.ndarray:
        return self.keypoints[: self.observation_length]

    @property
    def future_keypoints(self) -> np.ndarray:
        return self.keypoints[self.observation_length :]


def split_sequence_ids(
    sequence_ids: Sequence[str],
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split sequence IDs without frame-level leakage.

    Args:
        sequence_ids: Sequence IDs from `MabeDataset.sequence_ids`.
        validation_fraction: Fraction of sequences assigned to validation.
        seed: Random seed for deterministic shuffling.

    Returns:
        `(train_ids, validation_ids)` sorted within each split.
    """

    ids = np.array(sorted(sequence_ids), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = ids[rng.permutation(len(ids))]
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation = tuple(sorted(str(item) for item in shuffled[:validation_count]))
    train = tuple(sorted(str(item) for item in shuffled[validation_count:]))
    return train, validation


def missing_keypoint_mask(keypoints: np.ndarray) -> np.ndarray:
    """Return a mask for missing keypoints.

    Args:
        keypoints: Pose array ending in the coordinate dimension.

    Returns:
        Boolean array where a keypoint is missing if both coordinates are zero.
    """

    return np.all(keypoints == 0, axis=-1)


def fill_missing_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Fill zero-valued keypoint holes across time.

    Args:
        keypoints: Pose array shaped `[frames, mice, keypoints, coordinates]`.

    Returns:
        Copy of `keypoints` where missing entries are forward-filled. Initial holes use the
        first later observed value. Keypoints missing for the full sequence stay zero.
    """

    filled = np.asarray(keypoints).copy()
    mask = missing_keypoint_mask(filled)

    for mouse_idx in range(filled.shape[1]):
        for keypoint_idx in range(filled.shape[2]):
            missing_frames = mask[:, mouse_idx, keypoint_idx]
            if not np.any(missing_frames):
                continue

            observed = np.flatnonzero(~missing_frames)
            if observed.size == 0:
                continue

            first_observed = int(observed[0])
            filled[:first_observed, mouse_idx, keypoint_idx] = filled[
                first_observed, mouse_idx, keypoint_idx
            ]

            for frame_idx in range(first_observed + 1, filled.shape[0]):
                if missing_frames[frame_idx]:
                    filled[frame_idx, mouse_idx, keypoint_idx] = filled[
                        frame_idx - 1, mouse_idx, keypoint_idx
                    ]

    return filled


@dataclass(frozen=True)
class PoseNormalizer:
    """Image-size pose normalizer for MABe pixel coordinates.

    Attributes:
        offset: Coordinate offset shaped `[2]`.
        scale: Coordinate scale shaped `[2]`.
    """

    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, sequences: Iterable[MabeSequence]) -> PoseNormalizer:
        """Create the fixed MABe pixel-coordinate normalizer.

        Args:
            sequences: Unused training sequences kept for caller symmetry.

        Returns:
            Normalizer that maps pixel coordinates to image-size units.
        """

        _ = sequences
        return cls(
            offset=np.zeros(COORDINATES, dtype=np.float32),
            scale=np.array([FRAME_WIDTH, FRAME_HEIGHT], dtype=np.float32),
        )

    def transform(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize keypoints by MABe image dimensions.

        Args:
            keypoints: Pose tensor with final coordinate dimension `[x, y]`.

        Returns:
            Float32 tensor where observed pixels are scaled by `[width, height]`.
        """

        missing = missing_keypoint_mask(keypoints)
        normalized = (keypoints.astype(np.float32) - self.offset) / self.scale
        normalized[missing] = 0.0
        return normalized.astype(np.float32)

    def inverse_transform(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert normalized keypoints back to pixel coordinates.

        Args:
            keypoints: Normalized pose tensor.

        Returns:
            Float32 pose tensor in pixel coordinates.
        """

        missing = missing_keypoint_mask(keypoints)
        restored = keypoints.astype(np.float32) * self.scale + self.offset
        restored[missing] = 0.0
        return restored.astype(np.float32)


class MabeWindowDataset:
    """Deterministic window dataset over loaded MABe sequences.

    Args:
        sequences: Source sequences to window.
        spec: Window length, stride, and observed/future split.
        fill_missing: Whether to fill zero-valued keypoint holes before returning a window.
        normalizer: Optional training-fitted normalizer.
        max_windows: Optional cap for debug subsets.
        window_keys: Optional explicit `(sequence_id, start_frame)` selection.
        index_window_length: Optional longer window length used only to determine
            valid start frames for comparisons across forecasting horizons.
    """

    def __init__(
        self,
        sequences: Sequence[MabeSequence],
        spec: WindowSpec | None = None,
        *,
        fill_missing: bool = True,
        normalizer: PoseNormalizer | None = None,
        max_windows: int | None = None,
        window_keys: Sequence[tuple[str, int]] | None = None,
        index_window_length: int | None = None,
    ) -> None:
        self.sequences = {sequence.sequence_id: sequence for sequence in sequences}
        self.spec = spec or WindowSpec()
        self.fill_missing = fill_missing
        self.normalizer = normalizer
        self._index = (
            tuple(window_keys)
            if window_keys is not None
            else self._build_index(max_windows, index_window_length)
        )

    def _build_index(
        self,
        max_windows: int | None,
        index_window_length: int | None,
    ) -> tuple[tuple[str, int], ...]:
        """Build deterministic `(sequence_id, start_frame)` window pointers.

        Args:
            max_windows: Optional maximum number of pointers to return.
            index_window_length: Optional length used to restrict valid starts while
                leaving the window length returned by ``__getitem__`` unchanged.

        Returns:
            Sequence and start-frame pointers in deterministic order.
        """

        index: list[tuple[str, int]] = []
        required_steps = index_window_length or self.spec.length
        required_length = (required_steps - 1) * self.spec.frame_step + 1
        for sequence_id in sorted(self.sequences):
            sequence = self.sequences[sequence_id]
            stop = sequence.num_frames - required_length + 1
            for start in range(0, max(0, stop), self.spec.stride):
                index.append((sequence_id, start))
                if max_windows is not None and len(index) >= max_windows:
                    return tuple(index)
        return tuple(index)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def window_keys(self) -> tuple[tuple[str, int], ...]:
        """Return the deterministic sequence and start-frame selection."""

        return self._index

    def __getitem__(self, index: int) -> Window:
        sequence_id, start = self._index[index]
        sequence = self.sequences[sequence_id]
        frame_indices = start + np.arange(self.spec.length) * self.spec.frame_step

        keypoints = sequence.keypoints[frame_indices]
        if self.fill_missing:
            keypoints = fill_missing_keypoints(keypoints)
        keypoints = keypoints.astype(np.float32)
        if self.normalizer is not None:
            keypoints = self.normalizer.transform(keypoints)

        annotations = (
            sequence.annotations[:, frame_indices]
            if sequence.annotations is not None
            else None
        )
        return Window(
            sequence_id=sequence_id,
            start_frame=start,
            frame_indices=frame_indices.astype(np.int64),
            keypoints=keypoints,
            observation_length=self.spec.observation_length,
            annotations=annotations,
        )

    def first(self) -> Window:
        """Return the first deterministic window."""

        return self[0]
