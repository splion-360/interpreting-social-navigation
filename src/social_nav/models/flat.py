"""File description: Flat graph models for trajectory Gaussian prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from social_nav.graphs import EdgeSpec


@dataclass(frozen=True)
class FlatSocialAttentionConfig:
    """Configuration for the flat social-attention model.

    Attributes:
        node_input_size: Coordinate feature size for each node.
        edge_input_size: Coordinate-delta feature size for each edge.
        embedding_size: Shared hidden size for node and edge embeddings.
        output_size: Number of raw Gaussian parameters predicted per node.
    """

    node_input_size: int = 2
    edge_input_size: int = 2
    embedding_size: int = 64
    output_size: int = 5


class FlatSocialAttentionModel(nn.Module):
    """Predict per-node Gaussian parameters from a flat spatio-temporal graph.

    This module keeps graph construction outside the model. It consumes node
    coordinates, edge coordinate deltas, and the stable edge contract produced
    by `social_nav.graphs`.
    """

    def __init__(self, config: FlatSocialAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FlatSocialAttentionConfig()
        size = self.config.embedding_size
        self.node_encoder = nn.Linear(self.config.node_input_size, size)
        self.edge_encoder = nn.Linear(self.config.edge_input_size, size, bias=False)
        self.output_head = nn.Sequential(
            nn.Linear(size * 3, size),
            nn.ReLU(),
            nn.Linear(size, self.config.output_size),
        )

    def forward(
        self,
        nodes: Tensor,
        edge_features: Tensor,
        edge_specs: Sequence[EdgeSpec],
    ) -> Tensor:
        """Predict raw Gaussian parameters for each node and frame.

        Args:
            nodes: Node coordinates shaped `[time, nodes, 2]`.
            edge_features: Edge deltas shaped `[time, edges, 2]`.
            edge_specs: Stable edge identities matching `edge_features`.

        Returns:
            Raw Gaussian predictions shaped `[time, nodes, 5]`.
        """

        node_embeddings = torch.relu(self.node_encoder(nodes.float()))
        edge_embeddings = torch.relu(self.edge_encoder(edge_features.float()))
        temporal_context = self._temporal_context(edge_embeddings, edge_specs, nodes.shape[1])
        social_context = self._social_attention(edge_embeddings, temporal_context, edge_specs)
        return self.output_head(
            torch.cat((node_embeddings, temporal_context, social_context), dim=-1)
        )

    def _temporal_context(
        self,
        edge_embeddings: Tensor,
        edge_specs: Sequence[EdgeSpec],
        node_count: int,
    ) -> Tensor:
        """Gather temporal self-edge embeddings per node."""

        contexts = []
        for node_id in range(node_count):
            edge_id = edge_specs.index(EdgeSpec(source=node_id, target=node_id, kind="temporal"))
            contexts.append(edge_embeddings[:, edge_id])
        return torch.stack(contexts, dim=1)

    def _social_attention(
        self,
        edge_embeddings: Tensor,
        temporal_context: Tensor,
        edge_specs: Sequence[EdgeSpec],
    ) -> Tensor:
        """Attend over incoming spatial edges for each target node."""

        contexts = []
        for node_id in range(temporal_context.shape[1]):
            incoming = [
                edge_id
                for edge_id, edge in enumerate(edge_specs)
                if edge.kind == "spatial" and edge.target == node_id
            ]
            if not incoming:
                contexts.append(torch.zeros_like(temporal_context[:, node_id]))
                continue

            values = edge_embeddings[:, incoming]
            query = temporal_context[:, node_id].unsqueeze(1)
            scores = (query * values).sum(dim=-1) / sqrt(self.config.embedding_size)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            contexts.append((weights * values).sum(dim=1))

        return torch.stack(contexts, dim=1)
