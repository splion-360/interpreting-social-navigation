"""File description: Tests for flat graph trajectory models."""

import numpy as np
import torch

from loss import gaussian_2d_parameters
from models import EdgeAttention, EdgeRNN, FlatSocialAttentionModel, NodeRNN
from st_graph import build_flat_sparse_keypoint_graph


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


def test_flat_social_attention_model_uses_srnn_components() -> None:
    model = FlatSocialAttentionModel()

    assert isinstance(model.node_rnn, NodeRNN)
    assert isinstance(model.temporal_edge_rnn, EdgeRNN)
    assert isinstance(model.spatial_edge_rnn, EdgeRNN)
    assert isinstance(model.edge_attention, EdgeAttention)


def test_flat_social_attention_model_backpropagates_through_recurrent_path() -> None:
    graph = build_flat_sparse_keypoint_graph(make_keypoints(frames=3))
    model = FlatSocialAttentionModel()

    outputs = model(
        torch.from_numpy(graph.nodes),
        torch.from_numpy(graph.edge_features),
        graph.edge_specs,
    )
    loss = outputs.square().mean()
    loss.backward()

    assert model.node_rnn.cell.weight_hh.grad is not None
    assert model.temporal_edge_rnn.cell.weight_hh.grad is not None
    assert model.spatial_edge_rnn.cell.weight_hh.grad is not None
    assert model.edge_attention.temporal_projection.weight.grad is not None
