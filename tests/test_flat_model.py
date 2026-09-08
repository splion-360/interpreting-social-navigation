"""File description: Tests for flat graph trajectory models."""

import numpy as np
import torch

from flat_model import FlatSocialAttentionModel
from loss import gaussian_2d_parameters
from social_attention import build_flat_sparse_keypoint_graph


def make_keypoints(frames: int = 4) -> np.ndarray:
    """Create deterministic keypoints shaped `[frames, 3, 12, 2]`."""

    values = np.arange(frames * 3 * 12 * 2, dtype=np.float32)
    return values.reshape(frames, 3, 12, 2)


def test_flat_social_attention_model_outputs_gaussian_params_per_keypoint() -> None:
    graph = build_flat_sparse_keypoint_graph(make_keypoints())
    model = FlatSocialAttentionModel()

    outputs = model(
        torch.from_numpy(graph.nodes),
        torch.from_numpy(graph.edge_features),
        graph.edge_specs,
    )
    params = gaussian_2d_parameters(outputs)

    assert outputs.shape == (4, 36, 5)
    assert torch.isfinite(outputs).all()
    assert torch.all(params.sigma_x > 0)
    assert torch.all(params.sigma_y > 0)
    assert torch.all(params.rho < 1)
    assert torch.all(params.rho > -1)


def test_flat_social_attention_model_keeps_zero_edge_features_zero() -> None:
    model = FlatSocialAttentionModel()
    zeros = torch.zeros((2, 4, 2))

    edge_embeddings = model.edge_encoder(zeros)

    assert torch.count_nonzero(edge_embeddings) == 0
