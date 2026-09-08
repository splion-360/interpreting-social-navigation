"""File description: SRNN-style flat model for trajectory Gaussian prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from st_graph import EdgeSpec


@dataclass(frozen=True)
class FlatSocialAttentionConfig:
    """Configuration for the flat SRNN social-attention model.

    Attributes:
        node_input_size: Coordinate feature size for each node.
        edge_input_size: Coordinate-delta feature size for each edge.
        node_embedding_size: Embedded node-input size.
        edge_embedding_size: Embedded edge-input size.
        node_rnn_size: Hidden size for node recurrence.
        edge_rnn_size: Hidden size for temporal and spatial edge recurrence.
        attention_size: Hidden size used to score temporal-to-spatial attention.
        output_size: Number of raw Gaussian parameters predicted per node.
        dropout: Dropout probability applied after input embeddings.
    """

    node_input_size: int = 2
    edge_input_size: int = 2
    node_embedding_size: int = 64
    edge_embedding_size: int = 64
    node_rnn_size: int = 128
    edge_rnn_size: int = 128
    attention_size: int = 64
    output_size: int = 5
    dropout: float = 0.0


class NodeRNN(nn.Module):
    """Node recurrence that combines position, temporal edge, and social context."""

    def __init__(self, config: FlatSocialAttentionConfig) -> None:
        super().__init__()
        self.position_encoder = nn.Linear(
            config.node_input_size, config.node_embedding_size
        )
        self.edge_context_encoder = nn.Linear(
            config.edge_rnn_size * 2, config.node_embedding_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.cell = nn.LSTMCell(config.node_embedding_size * 2, config.node_rnn_size)
        self.output = nn.Linear(config.node_rnn_size, config.output_size)

    def forward(
        self,
        positions: Tensor,
        temporal_context: Tensor,
        social_context: Tensor,
        hidden: Tensor,
        cell: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance node recurrence for all nodes in one frame.

        Args:
            positions: Node coordinates shaped `[nodes, 2]`.
            temporal_context: Temporal edge hidden states shaped `[nodes, edge_hidden]`.
            social_context: Social context shaped `[nodes, edge_hidden]`.
            hidden: Previous node hidden states shaped `[nodes, node_hidden]`.
            cell: Previous node cell states shaped `[nodes, node_hidden]`.

        Returns:
            Raw Gaussian outputs, next hidden states, and next cell states.
        """

        encoded_position = self.dropout(torch.relu(self.position_encoder(positions)))
        edge_context = torch.cat((temporal_context, social_context), dim=-1)
        encoded_edges = self.dropout(
            torch.relu(self.edge_context_encoder(edge_context))
        )
        next_hidden, next_cell = self.cell(
            torch.cat((encoded_position, encoded_edges), dim=-1),
            (hidden, cell),
        )
        return self.output(next_hidden), next_hidden, next_cell


class EdgeRNN(nn.Module):
    """Shared recurrence for temporal or spatial edge features."""

    def __init__(self, config: FlatSocialAttentionConfig) -> None:
        super().__init__()
        self.encoder = nn.Linear(config.edge_input_size, config.edge_embedding_size)
        self.dropout = nn.Dropout(config.dropout)
        self.cell = nn.LSTMCell(config.edge_embedding_size, config.edge_rnn_size)

    def forward(
        self,
        features: Tensor,
        hidden: Tensor,
        cell: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance edge recurrence for selected edges in one frame.

        Args:
            features: Edge features shaped `[edges, 2]`.
            hidden: Previous edge hidden states shaped `[edges, edge_hidden]`.
            cell: Previous edge cell states shaped `[edges, edge_hidden]`.

        Returns:
            Next hidden and cell states for the selected edges.
        """

        encoded = self.dropout(torch.relu(self.encoder(features)))
        return self.cell(encoded, (hidden, cell))


class EdgeAttention(nn.Module):
    """Dot-product attention from a node temporal edge to its spatial edges."""

    def __init__(self, config: FlatSocialAttentionConfig) -> None:
        super().__init__()
        self.temporal_projection = nn.Linear(
            config.edge_rnn_size, config.attention_size
        )
        self.spatial_projection = nn.Linear(config.edge_rnn_size, config.attention_size)
        self.attention_size = config.attention_size

    def forward(self, temporal: Tensor, spatial: Tensor) -> tuple[Tensor, Tensor]:
        """Attend from one temporal edge state over incoming spatial edge states.

        Args:
            temporal: Temporal hidden state shaped `[edge_hidden]`.
            spatial: Spatial hidden states shaped `[edges, edge_hidden]`.

        Returns:
            Weighted spatial context and attention weights.
        """

        temporal_query = self.temporal_projection(temporal)
        spatial_keys = self.spatial_projection(spatial)
        scores = spatial_keys @ temporal_query
        scores = scores * (spatial.shape[0] / self.attention_size**0.5)
        weights = torch.softmax(scores, dim=0)
        return weights @ spatial, weights


class FlatSocialAttentionModel(nn.Module):
    """Predict per-node Gaussian parameters with SRNN-style social attention."""

    def __init__(self, config: FlatSocialAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FlatSocialAttentionConfig()
        self.node_rnn = NodeRNN(self.config)
        self.temporal_edge_rnn = EdgeRNN(self.config)
        self.spatial_edge_rnn = EdgeRNN(self.config)
        self.edge_attention = EdgeAttention(self.config)

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

        time_steps, node_count, _ = nodes.shape
        device = nodes.device
        dtype = nodes.dtype
        edge_count = len(edge_specs)
        node_hidden = torch.zeros(
            node_count, self.config.node_rnn_size, device=device, dtype=dtype
        )
        node_cell = torch.zeros_like(node_hidden)
        edge_hidden = torch.zeros(
            edge_count, self.config.edge_rnn_size, device=device, dtype=dtype
        )
        edge_cell = torch.zeros_like(edge_hidden)
        outputs = []

        temporal_edge_ids, spatial_edge_ids = _edge_kind_ids(edge_specs)
        incoming_spatial = _incoming_spatial_edge_ids(edge_specs, node_count)

        for frame_idx in range(time_steps):
            temporal_context = torch.zeros(
                node_count, self.config.edge_rnn_size, device=device, dtype=dtype
            )
            social_context = torch.zeros_like(temporal_context)

            if frame_idx > 0:
                temporal_hidden, temporal_cell = self.temporal_edge_rnn(
                    edge_features[frame_idx, temporal_edge_ids].float(),
                    edge_hidden[temporal_edge_ids],
                    edge_cell[temporal_edge_ids],
                )
                edge_hidden[temporal_edge_ids] = temporal_hidden
                edge_cell[temporal_edge_ids] = temporal_cell
                temporal_targets = torch.tensor(
                    [edge_specs[edge_id].target for edge_id in temporal_edge_ids],
                    device=device,
                )
                temporal_context[temporal_targets] = temporal_hidden

            spatial_hidden, spatial_cell = self.spatial_edge_rnn(
                edge_features[frame_idx, spatial_edge_ids].float(),
                edge_hidden[spatial_edge_ids],
                edge_cell[spatial_edge_ids],
            )
            edge_hidden[spatial_edge_ids] = spatial_hidden
            edge_cell[spatial_edge_ids] = spatial_cell

            for node_id, edge_ids in enumerate(incoming_spatial):
                if not edge_ids:
                    continue
                social_context[node_id], _ = self.edge_attention(
                    temporal_context[node_id],
                    edge_hidden[torch.tensor(edge_ids, device=device)],
                )

            frame_output, node_hidden, node_cell = self.node_rnn(
                nodes[frame_idx].float(),
                temporal_context,
                social_context,
                node_hidden,
                node_cell,
            )
            outputs.append(frame_output)

        return torch.stack(outputs, dim=0)


def _edge_kind_ids(edge_specs: Sequence[EdgeSpec]) -> tuple[list[int], list[int]]:
    """Separate temporal and spatial edge IDs."""

    temporal = []
    spatial = []
    for edge_id, edge in enumerate(edge_specs):
        if edge.kind == "temporal":
            temporal.append(edge_id)
        else:
            spatial.append(edge_id)
    return temporal, spatial


def _incoming_spatial_edge_ids(
    edge_specs: Sequence[EdgeSpec],
    node_count: int,
) -> list[list[int]]:
    """Collect incoming spatial edge IDs for each target node."""

    incoming: list[list[int]] = [[] for _ in range(node_count)]
    for edge_id, edge in enumerate(edge_specs):
        if edge.kind == "spatial":
            incoming[edge.target].append(edge_id)
    return incoming
