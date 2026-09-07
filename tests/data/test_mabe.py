"""File description: Tests for MABe loading, preprocessing, and windowing."""

from pathlib import Path

import numpy as np
import pytest

from social_nav.data import (
    MabeDataset,
    MabeWindowDataset,
    PoseNormalizer,
    WindowSpec,
    fill_missing_keypoints,
    split_sequence_ids,
)


def make_keypoints(frames: int = 30) -> np.ndarray:
    values = np.arange(frames * 3 * 12 * 2, dtype=np.float32)
    return values.reshape(frames, 3, 12, 2)


def write_mabe_file(path: Path) -> None:
    np.save(
        path,
        {
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
        },
    )


def test_loads_mabe_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(path)

    dataset = MabeDataset.from_file(path)

    assert dataset.vocabulary == ("chases", "lights")
    assert dataset.sequence_ids == ("seq_a", "seq_b")
    assert dataset.sequences["seq_a"].keypoints.shape == (30, 3, 12, 2)
    assert dataset.sequences["seq_a"].annotations.shape == (2, 30)


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


def test_fill_missing_keypoints_rejects_keypoint_missing_for_whole_sequence() -> None:
    keypoints = make_keypoints(frames=4)
    keypoints[:, 0, 0] = 0

    with pytest.raises(ValueError, match="missing for the whole sequence"):
        fill_missing_keypoints(keypoints)


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
