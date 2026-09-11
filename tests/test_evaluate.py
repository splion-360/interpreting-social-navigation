"""File description: Tests for autoregressive checkpoint evaluation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate as evaluate_module
from evaluate import (
    EvaluationResult,
    _build_evaluation_windows,
    build_evaluation_record,
    evaluate_flat_checkpoint,
    load_test_config,
    rollout_flat_keypoint_model,
    sample_bivariate_gaussian,
    save_evaluation_record,
    save_single_mouse_prediction_video,
    save_test_prediction_video,
)
from models import FlatSocialAttentionModel
from st_graph import build_dense_keypoint_graph
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

    monkeypatch.setattr(evaluate_module, "sample_bivariate_gaussian", fake_sample)
    rollout = rollout_flat_keypoint_model(
        model=IncrementModel(),
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


def test_load_test_config_reads_held_out_test_file_path(tmp_path) -> None:
    config_path = tmp_path / "test.yml"
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
        train_config_path=Path("src/config/dense_keypoint__train.yml"),
        checkpoint_path=Path("checkpoints/dense_keypoint/flat_best.pt"),
        split="test",
        test_config_path=Path("src/config/test.yml"),
        test_data_path=Path("data/MaBe/mouse_triplet_test.npy"),
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
    )

    assert result.path == output_root / "dense_keypoint" / "test_0.mp4"
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
    )

    assert result.path == output_root / "dense_keypoint" / "test_0__mouse_1.mp4"
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
