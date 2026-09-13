"""File description: Tests for autoregressive checkpoint evaluation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import evaluate as evaluate_module
import inference as inference_module
from evaluate import (
    EvaluationResult,
    _aggregate_metric_values,
    _build_evaluation_windows,
    _extreme_pair_cells,
    _matrix_cell_style,
    _resolve_results_path,
    build_baseline_evaluation_record,
    build_evaluation_record,
    default_model_results_path,
    evaluate_flat_checkpoint,
    evaluate_motion_baseline,
    load_motion_baseline_config,
    load_test_config,
    print_evaluation_metrics,
    print_motion_stratified_metrics,
    resolve_baseline_window_config,
    rollout_flat_keypoint_model,
    sample_bivariate_gaussian,
    save_evaluation_record,
    save_single_mouse_prediction_video,
    save_test_prediction_video,
    stratify_motion_scores,
)
from models import FlatSocialAttentionModel
from st_graph import build_dense_keypoint_graph, build_single_mouse_dense_keypoint_graph
from train import FlatFitConfig


def make_keypoints(frames: int = 4) -> np.ndarray:
    """Create deterministic keypoints shaped `[frames, 3, 12, 2]`."""

    values = np.arange(frames * 3 * 12 * 2, dtype=np.float32)
    return values.reshape(frames, 3, 12, 2)


def write_mabe_file(path, *, sequence_prefix: str, sequences: int = 3) -> None:
    """Write a small MABe-style file for evaluation tests."""

    payload = {
        "vocabulary": [],
        "sequences": {
            f"{sequence_prefix}_{idx}": {
                "keypoints": make_keypoints(frames=24) + np.float32(idx)
            }
            for idx in range(sequences)
        },
    }
    np.save(path, np.asarray(payload, dtype=object))


def test_sample_bivariate_gaussian_returns_coordinate_samples() -> None:
    outputs = torch.zeros((2, 36, 5))
    generator = torch.Generator().manual_seed(7)

    samples = sample_bivariate_gaussian(outputs, generator=generator)

    assert samples.shape == (2, 36, 2)
    assert torch.isfinite(samples).all()


def test_motion_stratification_is_deterministic_and_balanced() -> None:
    profile = stratify_motion_scores(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    )

    assert profile.labels == ("low", "low", "medium", "medium", "high", "high")
    assert profile.counts == {"low": 2, "medium": 2, "high": 2}
    assert profile.low_max_px_s == pytest.approx(5.0 / 3.0)
    assert profile.medium_max_px_s == pytest.approx(10.0 / 3.0)


def test_scalar_metric_aggregation_ignores_undefined_windows() -> None:
    metrics = _aggregate_metric_values({"direction_error_deg": [np.nan, 90.0]})

    assert metrics["direction_error_deg"] == 90.0


def test_rollout_reuses_sampled_positions_to_recompute_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = make_keypoints(frames=2)
    recorded_edges = []

    class IncrementModel:
        def forward_with_state(self, **kwargs):
            nodes = kwargs["nodes"]
            edge_features = kwargs["edge_features"]
            recorded_edges.append(edge_features[0, 0].detach().cpu().numpy())
            outputs = torch.zeros((1, nodes.shape[1], 5), dtype=nodes.dtype)
            outputs[0, :, :2] = nodes[0] + 1.0
            return SimpleNamespace(
                outputs=outputs,
                state=kwargs["state"],
                attention_weights=({},),
            )

    def fake_sample(outputs, *, generator=None):
        del generator
        return outputs[:, :2]

    monkeypatch.setattr(inference_module, "sample_bivariate_gaussian", fake_sample)
    rollout = rollout_flat_keypoint_model(
        model=cast(Any, IncrementModel()),
        observed_keypoints=observed,
        prediction_length=2,
        build_graph=build_dense_keypoint_graph,
        device=torch.device("cpu"),
    )

    observed_graph = build_dense_keypoint_graph(observed)
    np.testing.assert_array_equal(recorded_edges[1], observed_graph.edge_features[1, 0])
    np.testing.assert_array_equal(
        rollout.nodes[2], observed_graph.nodes[1] + np.float32(1.0)
    )
    np.testing.assert_array_equal(
        rollout.nodes[3], observed_graph.nodes[1] + np.float32(2.0)
    )
    np.testing.assert_array_equal(recorded_edges[2], np.array([-1.0, -1.0]))


def test_rollout_preserves_single_mouse_pose_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = make_keypoints(frames=2)[:, 0:1]

    class IncrementModel:
        def forward_with_state(self, **kwargs):
            nodes = kwargs["nodes"]
            outputs = torch.zeros((1, nodes.shape[1], 5), dtype=nodes.dtype)
            outputs[0, :, :2] = nodes[0] + 1.0
            return SimpleNamespace(
                outputs=outputs,
                state=kwargs["state"],
                attention_weights=({},),
            )

    def fake_sample(outputs, *, generator=None):
        del generator
        return outputs[:, :2]

    monkeypatch.setattr(inference_module, "sample_bivariate_gaussian", fake_sample)
    rollout = rollout_flat_keypoint_model(
        model=cast(Any, IncrementModel()),
        observed_keypoints=observed,
        prediction_length=2,
        build_graph=build_single_mouse_dense_keypoint_graph,
        device=torch.device("cpu"),
    )

    assert rollout.nodes.shape == (4, 12, 2)


def test_build_evaluation_windows_can_read_separate_test_file(tmp_path) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    write_mabe_file(train_path, sequence_prefix="train")
    write_mabe_file(test_path, sequence_prefix="test", sequences=2)
    config = FlatFitConfig(
        data_path=train_path,
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        max_validation_windows=1,
    )

    validation = _build_evaluation_windows(
        config=config,
        split="validation",
        test_data_path=None,
        max_windows=None,
    )
    test = _build_evaluation_windows(
        config=config,
        split="test",
        test_data_path=test_path,
        max_windows=1,
    )

    assert next(iter(validation.sequences)).startswith("train_")
    assert next(iter(test.sequences)).startswith("test_")
    assert len(validation) == 1
    assert len(test) == 1


def test_build_evaluation_windows_extracts_single_mouse_test_samples(tmp_path) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    write_mabe_file(train_path, sequence_prefix="train", sequences=3)
    write_mabe_file(test_path, sequence_prefix="test", sequences=1)
    config = FlatFitConfig(
        data_path=train_path,
        graph_variant="single_mouse_dense_keypoint",
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        max_validation_windows=1,
    )

    test = _build_evaluation_windows(
        config=config,
        split="test",
        test_data_path=test_path,
        max_windows=2,
    )

    assert len(test) == 2
    assert test[0].sequence_id == "test_0__mouse_0"
    assert test[0].keypoints.shape == (20, 1, 12, 2)


def test_horizon_comparison_uses_identical_sequence_start_windows(tmp_path) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    payload = {
        "vocabulary": [],
        "sequences": {
            f"test_{index}": {"keypoints": make_keypoints(frames=100)}
            for index in range(2)
        },
    }
    write_mabe_file(train_path, sequence_prefix="train")
    np.save(test_path, np.asarray(payload, dtype=object))
    short_config = FlatFitConfig(
        data_path=train_path,
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
    )
    long_config = FlatFitConfig(
        data_path=train_path,
        window_length=68,
        observation_length=8,
        prediction_length=60,
        stride=20,
    )

    short_windows = _build_evaluation_windows(
        config=short_config,
        split="test",
        test_data_path=test_path,
        max_windows=None,
        index_window_length=68,
    )
    long_windows = _build_evaluation_windows(
        config=long_config,
        split="test",
        test_data_path=test_path,
        max_windows=None,
        index_window_length=68,
    )

    short_keys = [
        (short_windows[index].sequence_id, short_windows[index].start_frame)
        for index in range(len(short_windows))
    ]
    long_keys = [
        (long_windows[index].sequence_id, long_windows[index].start_frame)
        for index in range(len(long_windows))
    ]
    assert short_keys == long_keys


def test_load_test_config_reads_held_out_test_file_path(tmp_path) -> None:
    config_path = tmp_path / "test__mabe.yml"
    config_path.write_text(
        "\n".join(
            [
                f"data_path: {tmp_path / 'mouse_triplet_test.npy'}",
                f"results_path: {tmp_path / 'results.jsonl'}",
                "max_windows: 3",
                "seed: 7",
                "wandb: true",
                "wandb_project: eval-project",
                "wandb_run_name: eval-run",
            ]
        )
    )

    config = load_test_config(config_path)

    assert config.data_path == tmp_path / "mouse_triplet_test.npy"
    assert config.results_path == tmp_path / "results.jsonl"
    assert config.max_windows == 3
    assert config.seed == 7
    assert config.wandb is True
    assert config.wandb_project == "eval-project"
    assert config.wandb_run_name == "eval-run"


def test_load_test_config_allows_source_specific_default_results_path(tmp_path) -> None:
    config_path = tmp_path / "test__mabe.yml"
    config_path.write_text(
        "\n".join(
            [
                f"data_path: {tmp_path / 'mouse_triplet_test.npy'}",
                "results_path: null",
            ]
        )
    )

    config = load_test_config(config_path)

    assert config.results_path is None


def test_load_motion_baseline_config_reads_baseline_names(tmp_path) -> None:
    config_path = tmp_path / "benchmark__motion_baselines_pred30.yml"
    config_path.write_text(
        "\n".join(
            [
                "baselines:",
                "  - persistence",
                "  - rigid_constant_velocity",
                "observation_length: 8",
                "prediction_length: 30",
                "comparison_prediction_length: 60",
                "stride: 20",
                "max_windows: 1000",
                "source_fps: 30.0",
                "frame_step: 12",
                f"results_path: {tmp_path / 'baseline_results.jsonl'}",
                "seed: 42",
                "wandb: false",
                "wandb_project: baseline-project",
                "wandb_run_name: baseline-run",
            ]
        )
    )

    config = load_motion_baseline_config(config_path)

    assert config.baselines == ("persistence", "rigid_constant_velocity")
    assert config.observation_length == 8
    assert config.prediction_length == 30
    assert config.comparison_prediction_length == 60
    assert config.stride == 20
    assert config.max_windows == 1000
    assert config.source_fps == 30.0
    assert config.frame_step == 12
    assert config.results_path == tmp_path / "baseline_results.jsonl"
    assert config.seed == 42
    assert config.wandb is False
    assert config.wandb_project == "baseline-project"
    assert config.wandb_run_name == "baseline-run"


def test_baseline_window_config_changes_only_the_horizon_contract() -> None:
    train_config = FlatFitConfig(
        observation_length=8,
        prediction_length=12,
        window_length=20,
        stride=20,
        graph_variant="dense_keypoint",
    )
    baseline_config = evaluate_module.MotionBaselineConfig(
        observation_length=8,
        prediction_length=60,
        stride=20,
        frame_step=12,
        source_fps=30.0,
    )

    resolved = resolve_baseline_window_config(train_config, baseline_config)

    assert resolved.observation_length == 8
    assert resolved.prediction_length == 60
    assert resolved.window_length == 68
    assert resolved.stride == 20
    assert resolved.frame_step == 12
    assert resolved.source_fps == 30.0
    assert resolved.graph_variant == "dense_keypoint"


def test_model_evaluation_results_path_defaults_to_graph_variant() -> None:
    config = FlatFitConfig(graph_variant="dense_keypoint")

    assert default_model_results_path(config) == Path(
        "outputs/evaluations/flat_dense_triplet_30fps/results.jsonl"
    )


def test_model_evaluation_results_path_uses_experiment_slug() -> None:
    config = FlatFitConfig(
        graph_variant="within_mouse_dense_keypoint",
        frame_step=6,
    )

    assert default_model_results_path(config) == Path(
        "outputs/evaluations/flat_within_mouse_triplet_5fps/results.jsonl"
    )


def test_results_path_resolution_keeps_baselines_separate() -> None:
    train_config = FlatFitConfig(graph_variant="dense_keypoint")
    baseline_config = evaluate_module.MotionBaselineConfig()

    assert _resolve_results_path(
        explicit_path=None,
        baseline_config=baseline_config,
        test_config=evaluate_module.TestConfig(results_path=None),
        train_config=train_config,
    ) == Path("outputs/evaluations/motion_baselines_30fps/pred12/results.jsonl")


def test_evaluate_motion_baseline_uses_test_windows(tmp_path) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    write_mabe_file(train_path, sequence_prefix="train")
    write_mabe_file(test_path, sequence_prefix="test", sequences=2)
    config = FlatFitConfig(
        data_path=train_path,
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        max_validation_windows=1,
    )

    result = evaluate_motion_baseline(
        train_config=config,
        baseline="persistence",
        split="test",
        test_data_path=test_path,
        max_windows=1,
        show_progress=False,
    )

    assert result.baseline == "persistence"
    assert result.split == "test"
    assert result.windows == 1
    assert "centroid_ade_px" in result.metrics
    assert "centroid_velocity_error_px_s" in result.metrics
    assert "displacement_gain" in result.metrics
    assert "relative_ordering_error" in result.metrics
    assert sum(result.motion_profile["counts"].values()) == 1
    assert set(result.metrics_by_motion) == {"low", "medium", "high"}


def test_save_evaluation_record_appends_jsonl(tmp_path) -> None:
    results_path = tmp_path / "outputs" / "evaluations" / "results.jsonl"
    record = {
        "split": "test",
        "windows": 2,
        "metrics": {"ade": 1.25, "fde": 2.5},
    }

    save_evaluation_record(record, results_path)

    assert results_path.read_text().strip() == (
        '{"metrics": {"ade": 1.25, "fde": 2.5}, "split": "test", "windows": 2}'
    )


def test_save_evaluation_record_writes_undefined_metrics_as_json_null(tmp_path) -> None:
    results_path = tmp_path / "results.jsonl"

    save_evaluation_record(
        {"metrics": {"displacement_direction_error_deg": float("nan")}},
        results_path,
    )

    assert '"displacement_direction_error_deg": null' in results_path.read_text()


def test_build_evaluation_record_tracks_lineage() -> None:
    record = build_evaluation_record(
        result=EvaluationResult(
            split="test",
            windows=5,
            metrics={
                "keypoint_ade_px": 1.0,
                "keypoint_fde_px": 2.0,
                "centroid_ade_px": 0.5,
                "centroid_fde_px": 1.5,
                "skeleton_orientation_error_deg": 3.0,
                "bone_length_error_px": 4.0,
            },
            device="cpu",
            checkpoint_epoch=50,
            checkpoint_validation_loss=0.5,
        ),
        train_config=FlatFitConfig(graph_variant="dense_keypoint"),
        train_config_path=Path("src/config/train__flat_dense_triplet_30fps.yml"),
        checkpoint_path=Path("checkpoints/flat_dense_triplet_30fps/flat_best.pt"),
        split="test",
        test_config_path=Path("src/config/test__mabe.yml"),
        test_data_path=Path("data/mabe/raw/mouse_triplet_test.npy"),
        max_windows=10,
        seed=42,
    )

    assert record["split"] == "test"
    assert record["checkpoint"]["epoch"] == 50
    assert record["model"]["graph_variant"] == "dense_keypoint"
    assert record["metrics"] == {
        "keypoint_ade_px": 1.0,
        "keypoint_fde_px": 2.0,
        "centroid_ade_px": 0.5,
        "centroid_fde_px": 1.5,
        "skeleton_orientation_error_deg": 3.0,
        "bone_length_error_px": 4.0,
    }
    assert record["sampling"] == "bivariate_gaussian"


def test_baseline_record_persists_motion_strata() -> None:
    result = evaluate_module.BaselineEvaluationResult(
        baseline="persistence",
        split="test",
        windows=3,
        metrics={"centroid_ade_px": 2.0},
        motion_profile={
            "score": "mean_keypoint_speed_px_s",
            "thresholds_px_s": {"low_max": 1.0, "medium_max": 2.0},
            "counts": {"low": 1, "medium": 1, "high": 1},
        },
        metrics_by_motion={
            "low": {
                "centroid_ade_px": 1.0,
                "skeleton_orientation_error_deg": 4.0,
                "bone_length_error_px": 7.0,
            },
            "medium": {
                "centroid_ade_px": 2.0,
                "skeleton_orientation_error_deg": 5.0,
                "bone_length_error_px": 8.0,
            },
            "high": {
                "centroid_ade_px": 3.0,
                "skeleton_orientation_error_deg": 6.0,
                "bone_length_error_px": 9.0,
            },
        },
        evaluation_seconds=1.25,
    )
    baseline_config = evaluate_module.MotionBaselineConfig(
        observation_length=8,
        prediction_length=30,
        source_fps=30.0,
        frame_step=12,
    )

    record = build_baseline_evaluation_record(
        result=result,
        train_config=FlatFitConfig(),
        train_config_path=Path("src/config/train__flat_dense_triplet_30fps.yml"),
        baseline_config=baseline_config,
        baseline_config_path=Path("src/config/benchmark__motion_baselines_pred30.yml"),
        split="test",
        test_config_path=Path("src/config/test__mabe.yml"),
        test_data_path=Path("data/mabe/raw/mouse_triplet_test.npy"),
        max_windows=3,
        seed=42,
    )

    assert record["motion_profile"] == result.motion_profile
    assert record["metrics_by_motion"] == result.metrics_by_motion
    assert record["runtime_seconds"] == 1.25
    assert record["source_fps"] == 30.0
    assert record["frame_step"] == 12
    assert record["effective_fps"] == 2.5
    assert record["prediction_horizon_seconds"] == 12.0


def test_print_evaluation_metrics_includes_diagnostic_tables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    edge_angle = np.full((3, 12, 12), np.nan, dtype=np.float32)
    edge_bone = np.full((3, 12, 12), np.nan, dtype=np.float32)
    edge_angle[:, 0, 1] = 12.5
    edge_angle[:, 1, 0] = 12.5
    edge_bone[:, 0, 1] = 4.25
    edge_bone[:, 1, 0] = 4.25

    print_evaluation_metrics(
        {
            "centroid_ade_px": 1.25,
            "centroid_fde_px": 2.5,
            "keypoint_ade_px": 1.5,
            "keypoint_fde_px": 3.0,
            "centroid_x_offset_px": 4.0,
            "centroid_y_offset_px": -2.0,
            "body_heading_error_deg": 15.0,
            "relative_ordering_error": 0.125,
            "relative_ordering_error_forward": 0.1,
            "relative_ordering_error_lateral": 0.15,
            "centroid_offset_px_by_mouse": [
                [1.0, 2.0],
                [3.0, -4.0],
                [5.0, -6.0],
            ],
            "body_heading_error_deg_by_mouse": [10.0, 20.0, 30.0],
            "body_frame_keypoint_error_px_by_mouse": np.full(
                (3, 12), 0.75, dtype=np.float32
            ).tolist(),
            "relative_ordering_error_by_mouse_axis": [
                [0.1, 0.2],
                [0.0, 0.3],
                [0.4, 0.5],
            ],
            "edge_angle_error_deg_by_mouse": edge_angle.tolist(),
            "edge_bone_length_error_px_by_mouse": edge_bone.tolist(),
        }
    )

    output = capsys.readouterr().out
    assert "Primary evaluation" in output
    assert "centroid_ade_px" in output
    assert "centroid_x_offset_px" in output
    assert "keypoint_ade_px" in output
    assert "relative_ordering_error" in output
    assert "Keypoint indices" in output
    assert "nose" in output
    assert "centroid_offset_px_by_mouse" in output
    assert "body_heading_error_deg_by_mouse" in output
    assert "body_frame_keypoint_error_px_by_mouse" in output
    assert "relative_ordering_error_by_mouse_axis" in output
    assert "edge_angle_error_deg_by_mouse" in output
    assert "edge_bone_length_error_px_by_mouse" in output
    assert "mouse_0" in output
    assert "  12.50" in output
    assert "   4.25" in output


def test_print_motion_stratified_metrics_includes_counts_and_thresholds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_motion_stratified_metrics(
        metrics_by_motion={
            "low": {
                "centroid_ade_px": 1.0,
                "skeleton_orientation_error_deg": 4.0,
                "bone_length_error_px": 7.0,
            },
            "medium": {
                "centroid_ade_px": 2.0,
                "skeleton_orientation_error_deg": 5.0,
                "bone_length_error_px": 8.0,
            },
            "high": {
                "centroid_ade_px": 3.0,
                "skeleton_orientation_error_deg": 6.0,
                "bone_length_error_px": 9.0,
            },
        },
        motion_profile={
            "score": "mean_keypoint_speed_px_s",
            "thresholds_px_s": {"low_max": 4.0, "medium_max": 9.0},
            "counts": {"low": 2, "medium": 3, "high": 4},
        },
    )

    output = capsys.readouterr().out
    assert "Metrics by ground-truth motion stratum" in output
    assert "low (n=2)" in output
    assert "medium (n=3)" in output
    assert "high (n=4)" in output
    assert "low <= 4.000 px/s" in output
    assert "medium <= 9.000 px/s" in output
    assert "centroid_ade_px" in output
    assert "skeleton_orientation_error_deg" in output
    assert "bone_length_error_px" in output


def test_diagnostic_table_styles_mark_edges_and_extremes() -> None:
    matrix = np.full((12, 12), np.nan, dtype=np.float32)
    matrix[0, 1] = 2.0
    matrix[1, 0] = 2.0
    matrix[0, 5] = 7.0
    matrix[5, 0] = 7.0
    matrix[3, 6] = 1.0
    matrix[6, 3] = 1.0

    min_cells, max_cells = _extreme_pair_cells(matrix)

    assert min_cells == frozenset(((3, 6), (6, 3)))
    assert max_cells == frozenset(((0, 5), (5, 0)))
    assert (
        _matrix_cell_style(
            row_index=0,
            column_index=1,
            min_cells=min_cells,
            max_cells=max_cells,
        )
        == "black on sky_blue1"
    )
    assert (
        _matrix_cell_style(
            row_index=3,
            column_index=6,
            min_cells=min_cells,
            max_cells=max_cells,
        )
        == "bold black on green underline"
    )
    assert (
        _matrix_cell_style(
            row_index=0,
            column_index=5,
            min_cells=min_cells,
            max_cells=max_cells,
        )
        == "bold white on red"
    )


def test_save_test_prediction_video_uses_requested_output_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    checkpoint_path = tmp_path / "flat_best.pt"
    output_root = tmp_path / "outputs" / "visualizations"
    write_mabe_file(train_path, sequence_prefix="train")
    write_mabe_file(test_path, sequence_prefix="test", sequences=1)
    model = FlatSocialAttentionModel()
    torch.save({"model_state_dict": model.state_dict(), "epoch": 50}, checkpoint_path)

    saved = {}

    def fake_animate_prediction_comparison(
        actual_keypoints,
        predicted_future_keypoints,
        *,
        observation_length,
        sequence_id,
        step,
        interval_ms,
    ):
        saved["actual_shape"] = actual_keypoints.shape
        saved["predicted_shape"] = predicted_future_keypoints.shape
        saved["observation_length"] = observation_length
        saved["sequence_id"] = sequence_id
        saved["interval_ms"] = interval_ms
        return object()

    def fake_save_animation(animation_obj, path, fps):
        del animation_obj
        saved["path"] = Path(path)
        saved["fps"] = fps
        return Path(path)

    monkeypatch.setattr(
        evaluate_module,
        "animate_prediction_comparison",
        fake_animate_prediction_comparison,
    )
    monkeypatch.setattr(evaluate_module, "save_animation", fake_save_animation)

    result = save_test_prediction_video(
        config=FlatFitConfig(
            data_path=train_path,
            graph_variant="dense_keypoint",
            device="cpu",
            window_length=20,
            observation_length=8,
            prediction_length=12,
            stride=20,
        ),
        test_config=evaluate_module.TestConfig(data_path=test_path),
        checkpoint_path=checkpoint_path,
        sequence_id="test_0",
        output_root=output_root,
        fps=6,
        seed=1,
        filename_suffix="__20260913_120000",
    )

    assert result.path == output_root / "dense_keypoint" / "test_0__20260913_120000.mp4"
    assert result.sequence_id == "test_0"
    assert result.start_frame == 0
    assert saved["path"] == result.path
    assert saved["fps"] == 6
    assert saved["actual_shape"] == (20, 3, 12, 2)
    assert saved["predicted_shape"] == (12, 3, 12, 2)
    assert saved["observation_length"] == 8


def test_save_single_mouse_prediction_video_uses_mouse_output_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "mouse_triplet_train.npy"
    test_path = tmp_path / "mouse_triplet_test.npy"
    checkpoint_path = tmp_path / "flat_best.pt"
    output_root = tmp_path / "outputs" / "visualizations"
    write_mabe_file(train_path, sequence_prefix="train")
    write_mabe_file(test_path, sequence_prefix="test", sequences=1)
    model = FlatSocialAttentionModel()
    torch.save({"model_state_dict": model.state_dict(), "epoch": 50}, checkpoint_path)

    saved = {}

    def fake_animate_single_mouse_prediction_comparison(
        actual_keypoints,
        predicted_future_keypoints,
        *,
        observation_length,
        mouse_index,
        sequence_id,
        step,
        interval_ms,
    ):
        saved["actual_shape"] = actual_keypoints.shape
        saved["predicted_shape"] = predicted_future_keypoints.shape
        saved["observation_length"] = observation_length
        saved["mouse_index"] = mouse_index
        saved["sequence_id"] = sequence_id
        saved["interval_ms"] = interval_ms
        return object()

    def fake_save_animation(animation_obj, path, fps):
        del animation_obj
        saved["path"] = Path(path)
        saved["fps"] = fps
        return Path(path)

    monkeypatch.setattr(
        evaluate_module,
        "animate_single_mouse_prediction_comparison",
        fake_animate_single_mouse_prediction_comparison,
    )
    monkeypatch.setattr(evaluate_module, "save_animation", fake_save_animation)

    result = save_single_mouse_prediction_video(
        config=FlatFitConfig(
            data_path=train_path,
            graph_variant="dense_keypoint",
            device="cpu",
            window_length=20,
            observation_length=8,
            prediction_length=12,
            stride=20,
        ),
        test_config=evaluate_module.TestConfig(data_path=test_path),
        checkpoint_path=checkpoint_path,
        mouse_index=1,
        sequence_id="test_0",
        output_root=output_root,
        fps=6,
        seed=1,
        filename_suffix="__20260913_120000",
    )

    assert (
        result.path
        == output_root / "dense_keypoint" / "test_0__mouse_1__20260913_120000.mp4"
    )
    assert result.sequence_id == "test_0"
    assert result.start_frame == 0
    assert saved["path"] == result.path
    assert saved["fps"] == 6
    assert saved["actual_shape"] == (20, 3, 12, 2)
    assert saved["predicted_shape"] == (12, 3, 12, 2)
    assert saved["observation_length"] == 8
    assert saved["mouse_index"] == 1


def test_mouse_level_checkpoint_evaluation_requires_decoder(tmp_path) -> None:
    with pytest.raises(ValueError, match="keypoint graph"):
        evaluate_flat_checkpoint(
            config=FlatFitConfig(
                graph_variant="mouse_level",
                data_path=tmp_path / "missing.npy",
                device="cpu",
            ),
            checkpoint_path=tmp_path / "missing.pt",
        )
