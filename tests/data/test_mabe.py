"""File description: Tests for MABe loading, preprocessing, and windowing."""

from pathlib import Path
from typing import Any

import numpy as np

from data import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MabeDataset,
    MabeWindowDataset,
    PoseNormalizer,
    WindowSpec,
    fill_missing_keypoints,
    source_sequence_id,
    split_sequence_ids,
    temporal_sampling_diagnostics,
    to_single_mouse_sequences,
)


def make_keypoints(frames: int = 30) -> np.ndarray:
    values = np.arange(frames * 3 * 12 * 2, dtype=np.float32)
    return values.reshape(frames, 3, 12, 2)


def write_mabe_file(path: Path) -> None:
    payload: dict[str, Any] = {
        "vocabulary": ["chases", "lights"],
        "sequences": {
            "seq_b": {
                "keypoints": make_keypoints(),
                "annotations": np.zeros((2, 30), dtype=np.float32),
            },
            "seq_a": {
                "keypoints": make_keypoints() + 1000,
                "annotations": np.ones((2, 30), dtype=np.float32),
            },
        },
    }
    np.save(path, np.asarray(payload, dtype=object))


def test_loads_mabe_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(path)

    dataset = MabeDataset.from_file(path)

    assert dataset.vocabulary == ("chases", "lights")
    assert dataset.sequence_ids == ("seq_a", "seq_b")
    assert dataset.sequences["seq_a"].keypoints.shape == (30, 3, 12, 2)
    annotations = dataset.sequences["seq_a"].annotations
    assert annotations is not None
    assert annotations.shape == (2, 30)


def test_split_sequence_ids_is_deterministic() -> None:
    sequence_ids = [f"seq_{idx}" for idx in range(10)]

    first = split_sequence_ids(sequence_ids, validation_fraction=0.2, seed=123)
    second = split_sequence_ids(sequence_ids, validation_fraction=0.2, seed=123)

    assert first == second
    train, validation = first
    assert len(train) == 8
    assert len(validation) == 2
    assert not set(train).intersection(validation)


def test_fill_missing_keypoints_forward_fills_from_first_observed_frame() -> None:
    keypoints = make_keypoints(frames=4)
    keypoints[0, 0, 0] = 0
    keypoints[2, 0, 0] = 0

    filled = fill_missing_keypoints(keypoints)

    np.testing.assert_array_equal(filled[0, 0, 0], keypoints[1, 0, 0])
    np.testing.assert_array_equal(filled[2, 0, 0], filled[1, 0, 0])


def test_fill_missing_keypoints_keeps_whole_sequence_missing_as_zero() -> None:
    keypoints = make_keypoints(frames=4)
    keypoints[:, 0, 0] = 0

    filled = fill_missing_keypoints(keypoints)

    np.testing.assert_array_equal(filled[:, 0, 0], np.zeros((4, 2), dtype=np.float32))


def test_window_dataset_returns_contiguous_windows_with_annotations() -> None:
    dataset = MabeDataset.from_dict(
        {
            "vocabulary": ["chases"],
            "sequences": {
                "seq": {
                    "keypoints": make_keypoints(frames=8),
                    "annotations": np.arange(8, dtype=np.float32).reshape(1, 8),
                },
            },
        }
    )
    windows = MabeWindowDataset(
        dataset.select(["seq"]),
        WindowSpec(length=5, stride=2, observation_length=2, prediction_length=3),
    )

    assert len(windows) == 2
    window = windows[1]
    assert window.sequence_id == "seq"
    assert window.start_frame == 2
    assert window.keypoints.shape == (5, 3, 12, 2)
    assert window.observed_keypoints.shape == (2, 3, 12, 2)
    assert window.future_keypoints.shape == (3, 3, 12, 2)
    np.testing.assert_array_equal(window.annotations, np.array([[2, 3, 4, 5, 6]], dtype=np.float32))


def test_window_dataset_samples_with_frame_step_inside_window() -> None:
    dataset = MabeDataset.from_dict(
        {
            "sequences": {
                "seq": {
                    "keypoints": make_keypoints(frames=260),
                    "annotations": np.arange(260, dtype=np.float32).reshape(1, 260),
                },
            },
        }
    )
    spec = WindowSpec(
        length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        frame_step=12,
    )
    windows = MabeWindowDataset(dataset.select(["seq"]), spec)

    window = windows[0]

    np.testing.assert_array_equal(window.frame_indices, np.arange(0, 240, 12)[:20])
    assert window.start_frame == 0
    assert windows.window_keys[1] == ("seq", 20)
    np.testing.assert_array_equal(
        window.keypoints,
        make_keypoints(frames=260)[window.frame_indices],
    )
    np.testing.assert_array_equal(
        window.annotations,
        np.arange(0, 240, 12, dtype=np.float32)[:20].reshape(1, 20),
    )
    assert spec.raw_span == 229


def test_single_mouse_sequences_extract_after_source_split() -> None:
    dataset = MabeDataset.from_dict(
        {
            "sequences": {
                "seq_a": {"keypoints": make_keypoints(frames=8)},
                "seq_b": {"keypoints": make_keypoints(frames=8) + 1000},
            }
        }
    )
    train_ids, validation_ids = split_sequence_ids(
        dataset.sequence_ids,
        validation_fraction=0.5,
        seed=42,
    )

    train_sequences = to_single_mouse_sequences(dataset.select(train_ids))
    validation_sequences = to_single_mouse_sequences(dataset.select(validation_ids))
    train_sources = {
        source_sequence_id(sequence.sequence_id) for sequence in train_sequences
    }
    validation_sources = {
        source_sequence_id(sequence.sequence_id) for sequence in validation_sequences
    }

    assert len(train_sequences) == len(train_ids) * 3
    assert len(validation_sequences) == len(validation_ids) * 3
    assert train_sources.isdisjoint(validation_sources)
    assert train_sequences[0].keypoints.shape == (8, 1, 12, 2)
    assert train_sequences[0].sequence_id.endswith("__mouse_0")


def test_temporal_sampling_diagnostics_reports_motion_retention() -> None:
    keypoints = np.zeros((240, 3, 12, 2), dtype=np.float32)
    keypoints[..., 0] = np.arange(1, 241, dtype=np.float32)[:, None, None]
    dataset = MabeDataset.from_dict({"sequences": {"seq": {"keypoints": keypoints}}})
    spec = WindowSpec(
        length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        frame_step=12,
    )

    diagnostics = temporal_sampling_diagnostics(
        dataset.select(["seq"]),
        spec=spec,
        source_fps=30.0,
        max_windows=1,
    )

    assert diagnostics.effective_fps == 2.5
    assert diagnostics.observation_seconds == 3.2
    assert diagnostics.prediction_seconds == 4.8
    assert diagnostics.raw_span == 229
    assert diagnostics.retained_motion_ratio_mean == 1.0
    assert diagnostics.worst_windows[0]["last_sampled_frame"] == 228


def test_pose_normalizer_round_trips_filled_keypoints() -> None:
    dataset = MabeDataset.from_dict(
        {
            "sequences": {
                "seq": {
                    "keypoints": make_keypoints(frames=6),
                },
            },
        }
    )
    sequence = dataset.sequences["seq"]

    normalizer = PoseNormalizer.fit([sequence])
    normalized = normalizer.transform(sequence.keypoints)
    restored = normalizer.inverse_transform(normalized)

    np.testing.assert_allclose(restored, sequence.keypoints, rtol=1e-6, atol=1e-5)


def test_pose_normalizer_scales_pixels_and_preserves_zero_sentinel() -> None:
    keypoints = make_keypoints(frames=6)
    keypoints[:, 0, 0] = 0
    dataset = MabeDataset.from_dict(
        {
            "sequences": {
                "seq": {
                    "keypoints": keypoints,
                },
            },
        }
    )

    normalizer = PoseNormalizer.fit([dataset.sequences["seq"]])
    normalized = normalizer.transform(keypoints)

    np.testing.assert_array_equal(
        normalized[:, 0, 0], np.zeros((6, 2), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        normalizer.scale,
        np.array([FRAME_WIDTH, FRAME_HEIGHT], dtype=np.float32),
    )
    np.testing.assert_allclose(
        normalized[0, 0, 1],
        keypoints[0, 0, 1] / np.array([FRAME_WIDTH, FRAME_HEIGHT], dtype=np.float32),
    )
