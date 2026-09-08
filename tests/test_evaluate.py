"""File description: Tests for autoregressive checkpoint evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluate import (
    _build_evaluation_windows,
    evaluate_flat_checkpoint,
    rollout_flat_keypoint_model,
    sample_bivariate_gaussian,
)
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


def test_rollout_reuses_sampled_positions_to_recompute_edges() -> None:
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

    rollout = rollout_flat_keypoint_model(
        model=IncrementModel(),
        observed_keypoints=observed,
        prediction_length=2,
        build_graph=build_dense_keypoint_graph,
        device=torch.device("cpu"),
        sample=False,
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
        test_data_path=test_path,
        window_length=20,
        observation_length=8,
        prediction_length=12,
        stride=20,
        max_validation_windows=1,
    )

    validation = _build_evaluation_windows(
        config=config,
        split="validation",
        max_windows=None,
    )
    test = _build_evaluation_windows(config=config, split="test", max_windows=1)

    assert next(iter(validation.sequences)).startswith("train_")
    assert next(iter(test.sequences)).startswith("test_")
    assert len(validation) == 1
    assert len(test) == 1


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
