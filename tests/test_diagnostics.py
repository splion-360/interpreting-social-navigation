"""File description: Tests for paired model-behavior diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from data import MabeDataset
from diagnostics import (
    build_matched_window_data,
    compute_horizon_profile,
    horizon_profile_rows,
    load_pose_diagnostic_config,
    paired_cluster_bootstrap,
)
from train import FlatFitConfig


def make_keypoints(frames: int = 80) -> np.ndarray:
    """Create deterministic triplet keypoints shaped `[frames, 3, 12, 2]`."""

    values = np.arange(frames * 3 * 12 * 2, dtype=np.float32)
    return values.reshape(frames, 3, 12, 2)


def write_mabe_file(path: Path, *, prefix: str, sequences: int = 4) -> None:
    """Write a tiny MABe-style `.npy` fixture."""

    payload = {
        "vocabulary": [],
        "sequences": {
            f"{prefix}_{index}": {"keypoints": make_keypoints() + np.float32(index)}
            for index in range(sequences)
        },
    }
    np.save(path, np.asarray(payload, dtype=object))


def test_load_pose_diagnostic_config_reads_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "test__diagnostics.yml"
    config_path.write_text(
        "\n".join(
            [
                "dense_triplet:",
                "  train_config_path: src/config/train__flat_dense_triplet_5fps.yml",
                "  checkpoint_path: checkpoints/dense.pt",
                "  display_name: dense",
                "single_mouse:",
                "  train_config_path: src/config/train__flat_dense_single_mouse_5fps.yml",
                "  checkpoint_path: checkpoints/single.pt",
                "  display_name: single",
                "disconnected_triplet:",
                "  train_config_path: src/config/train__flat_within_mouse_triplet_5fps.yml",
                "  checkpoint_path: checkpoints/disconnected.pt",
                "  display_name: disconnected",
                "test_config_path: src/config/test__mabe.yml",
                f"results_path: {tmp_path / 'results.jsonl'}",
                f"case_results_path: {tmp_path / 'cases.jsonl'}",
                "max_triplet_windows: 5",
                "seeds: [42, 43]",
                "device: cpu",
            ]
        )
    )

    config = load_pose_diagnostic_config(config_path)

    assert config.dense_triplet.display_name == "dense"
    assert config.single_mouse.checkpoint_path == Path("checkpoints/single.pt")
    assert config.disconnected_triplet.checkpoint_path == Path(
        "checkpoints/disconnected.pt"
    )
    assert config.results_path == tmp_path / "results.jsonl"
    assert config.case_results_path == tmp_path / "cases.jsonl"
    assert config.max_triplet_windows == 5
    assert config.seeds == (42, 43)
    assert config.device == "cpu"


def test_paired_cluster_bootstrap_resamples_whole_triplet_windows() -> None:
    reference = np.asarray([1.0, 2.0, 3.0, 4.0])
    comparison = np.asarray([2.0, 3.0, 5.0, 6.0])
    clusters = np.asarray(["window-a", "window-a", "window-b", "window-b"])

    summary = paired_cluster_bootstrap(
        reference,
        comparison,
        clusters=clusters,
        seed=42,
        resamples=1000,
    )

    assert summary["pairs"] == 4
    assert summary["clusters"] == 2
    assert summary["mean_delta"] == pytest.approx(1.5)
    assert summary["p50_delta"] == pytest.approx(1.5)
    assert summary["comparison_better_rate"] == 0.0
    assert summary["mean_delta_ci95_low"] == pytest.approx(1.0)
    assert summary["mean_delta_ci95_high"] == pytest.approx(2.0)


def test_matched_window_data_expands_each_triplet_into_mouse_cases(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    write_mabe_file(train_path, prefix="train")
    write_mabe_file(test_path, prefix="test", sequences=3)
    dense_config = FlatFitConfig(
        data_path=train_path,
        graph_variant="dense_keypoint",
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        frame_step=1,
    )
    single_config = FlatFitConfig(
        data_path=train_path,
        graph_variant="single_mouse_dense_keypoint",
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        frame_step=1,
    )

    data = build_matched_window_data(
        dense_config=dense_config,
        single_config=single_config,
        test_config=type("TestConfig", (), {"data_path": test_path})(),
        max_triplet_windows=4,
        seed=42,
    )

    assert len(data.dense_windows) == 4
    assert len(data.single_windows) == 12
    assert len(data.cases) == 12
    assert {case.mouse_index for case in data.cases} == {0, 1, 2}
    for case in data.cases:
        dense_window = data.dense_windows[case.dense_index]
        single_window = data.single_windows[case.single_index]
        assert dense_window.sequence_id == case.sequence_id
        assert single_window.sequence_id == (
            f"{case.sequence_id}__mouse_{case.mouse_index}"
        )
        assert dense_window.start_frame == single_window.start_frame == case.start_frame


def test_matched_window_data_requires_same_window_contract(tmp_path: Path) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    write_mabe_file(train_path, prefix="train")
    write_mabe_file(test_path, prefix="test")
    dense_config = FlatFitConfig(
        data_path=train_path,
        graph_variant="dense_keypoint",
        window_length=20,
        observation_length=8,
        prediction_length=12,
    )
    single_config = FlatFitConfig(
        data_path=train_path,
        graph_variant="single_mouse_dense_keypoint",
        window_length=21,
        observation_length=8,
        prediction_length=13,
    )

    with pytest.raises(ValueError, match="diagnostic configs differ"):
        build_matched_window_data(
            dense_config=dense_config,
            single_config=single_config,
            test_config=type("TestConfig", (), {"data_path": test_path})(),
            max_triplet_windows=4,
            seed=42,
        )


def test_matched_window_data_uses_held_out_sequences(tmp_path: Path) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    write_mabe_file(train_path, prefix="train")
    write_mabe_file(test_path, prefix="heldout", sequences=2)
    dense_config = FlatFitConfig(
        data_path=train_path,
        graph_variant="dense_keypoint",
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
    )
    single_config = FlatFitConfig(
        data_path=train_path,
        graph_variant="single_mouse_dense_keypoint",
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
    )

    data = build_matched_window_data(
        dense_config=dense_config,
        single_config=single_config,
        test_config=type("TestConfig", (), {"data_path": test_path})(),
        max_triplet_windows=2,
        seed=42,
    )

    assert all(case.sequence_id.startswith("heldout_") for case in data.cases)
    assert MabeDataset.from_file(test_path).sequence_ids == ("heldout_0", "heldout_1")


def test_horizon_profile_reports_exact_step_errors() -> None:
    initial_pose = np.zeros((1, 12, 2), dtype=np.float32)
    target = np.zeros((3, 1, 12, 2), dtype=np.float32)
    predicted = np.zeros_like(target)
    predicted[:, :, :, 0] = np.asarray([1.0, 3.0, 6.0])[:, None, None]

    profile = compute_horizon_profile(
        predicted,
        target,
        initial_pose=initial_pose,
        seconds_per_step=0.2,
    )

    assert [row["horizon_step"] for row in profile] == [1, 2, 3]
    assert [row["centroid_displacement_error_px"] for row in profile] == [
        1.0,
        3.0,
        6.0,
    ]
    assert [row["centroid_velocity_error_px_s"] for row in profile] == [
        5.0,
        10.0,
        15.0,
    ]
    assert [row["keypoint_velocity_error_px_s"] for row in profile] == [
        5.0,
        10.0,
        15.0,
    ]


def test_teacher_forced_horizon_velocity_uses_previous_target_pose() -> None:
    initial_pose = np.zeros((1, 12, 2), dtype=np.float32)
    target = np.zeros((3, 1, 12, 2), dtype=np.float32)
    predicted = np.zeros_like(target)
    predicted[:, :, :, 0] = np.asarray([1.0, 3.0, 6.0])[:, None, None]

    profile = compute_horizon_profile(
        predicted,
        target,
        initial_pose=initial_pose,
        seconds_per_step=0.2,
        mode="teacher_forced",
    )

    assert [row["centroid_velocity_error_px_s"] for row in profile] == [
        5.0,
        15.0,
        30.0,
    ]


def test_horizon_profile_rows_builds_tidy_plot_data() -> None:
    record = {
        "window_contract": {"effective_fps": 5.0},
        "horizon_profile": {
            "dense_triplet": {
                "autoregressive": {
                    "1": {
                        "metrics": {"bone_length_error_px": 4.0},
                        "metric_quantiles": {
                            "bone_length_error_px": {"p50": 3.0, "p99": 9.0}
                        },
                    }
                }
            }
        },
    }

    rows = horizon_profile_rows(record, statistic="p50")

    assert rows == [
        {
            "model": "dense_triplet",
            "mode": "autoregressive",
            "horizon_step": 1,
            "horizon_seconds": 0.2,
            "metric": "bone_length_error_px",
            "value": 3.0,
        }
    ]
