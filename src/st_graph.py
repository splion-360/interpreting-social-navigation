"""File description: Spatio-temporal graph builders for mouse trajectory modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from data.schema import COORDINATES, NUM_KEYPOINTS, NUM_MICE


EdgeKind = Literal["temporal", "spatial"]


@dataclass(frozen=True)
class EdgeSpec:
    """Stable edge identity for a spatio-temporal graph.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        kind: Whether the edge is temporal or spatial.
    """

    source: int
    target: int
    kind: EdgeKind


@dataclass(frozen=True)
class GraphSequence:
    """Graph features for one pose sequence.

    Attributes:
        variant: Graph variant name.
        nodes: Node coordinates shaped `[time, nodes, 2]`.
        edge_features: Edge position vectors shaped `[time, edges, 2]`.
        edge_specs: Stable edge identities matching the edge-feature axis.
        nodes_present: Node IDs available at each frame.
        edges_present: Edge IDs available at each frame.
    """

    variant: str
    nodes: np.ndarray
    edge_features: np.ndarray
    edge_specs: tuple[EdgeSpec, ...]
    nodes_present: tuple[tuple[int, ...], ...]
    edges_present: tuple[tuple[int, ...], ...]

    @property
    def node_count(self) -> int:
        """Return the number of graph nodes."""

        return int(self.nodes.shape[1])

    @property
    def edge_count(self) -> int:
        """Return the number of graph edges."""

        return len(self.edge_specs)

    def edge_id(self, source: int, target: int, kind: EdgeKind) -> int:
        """Return the stable edge ID for a source, target, and kind.

        Args:
            source: Source node ID.
            target: Target node ID.
            kind: Edge kind.

        Returns:
            Index into the edge-feature axis.
        """

        return self.edge_specs.index(EdgeSpec(source=source, target=target, kind=kind))


def flat_keypoint_node_id(mouse_id: int, keypoint_id: int) -> int:
    """Map a mouse/keypoint pair to the flat graph node ID.

    Args:
        mouse_id: Mouse index in `[0, 3)`.
        keypoint_id: Keypoint index in `[0, 12)`.

    Returns:
        Stable keypoint-node ID.
    """

    return mouse_id * NUM_KEYPOINTS + keypoint_id


def build_flat_sparse_keypoint_graph(keypoints: np.ndarray) -> GraphSequence:
    """Build a flat keypoint graph with sparse cross-mouse spatial edges.

    The graph uses temporal self-edges for same-node motion and directed
    spatial edges for interactions within each frame. For the sparse MABe
    triplet baseline, spatial edges connect keypoints across mice only,
    excluding same-mouse keypoint-to-keypoint edges.

    Args:
        keypoints: Pose sequence shaped `[time, 3, 12, 2]`.

    Returns:
        Graph sequence with 36 nodes and 900 total edge specs.
    """

    _ensure_pose_shape(keypoints)
    nodes = keypoints.astype(np.float32).reshape(
        keypoints.shape[0], NUM_MICE * NUM_KEYPOINTS, COORDINATES
    )
    temporal_edges = tuple(
        EdgeSpec(source=node_id, target=node_id, kind="temporal")
        for node_id in range(nodes.shape[1])
    )
    spatial_edges = tuple(_flat_sparse_spatial_edges())
    return _build_graph_sequence(
        variant="flat_sparse_keypoint",
        nodes=nodes,
        edge_specs=temporal_edges + spatial_edges,
    )


def build_dense_keypoint_graph(keypoints: np.ndarray) -> GraphSequence:
    """Build a dense keypoint graph with one edge slot per node pair.

    Diagonal slots are temporal self-edges and off-diagonal slots are directed
    spatial edges. With three mice and twelve keypoints per mouse, this yields
    36 nodes and 1,296 edge slots.

    Args:
        keypoints: Pose sequence shaped `[time, 3, 12, 2]`.

    Returns:
        Graph sequence with dense `[time, nodes * nodes, 2]` edge features.
    """

    _ensure_pose_shape(keypoints)
    nodes = keypoints.astype(np.float32).reshape(
        keypoints.shape[0], NUM_MICE * NUM_KEYPOINTS, COORDINATES
    )
    edge_specs = tuple(
        EdgeSpec(
            source=source,
            target=target,
            kind="temporal" if source == target else "spatial",
        )
        for source in range(nodes.shape[1])
        for target in range(nodes.shape[1])
    )
    return _build_graph_sequence(
        variant="dense_keypoint",
        nodes=nodes,
        edge_specs=edge_specs,
    )


def build_single_mouse_dense_keypoint_graph(keypoints: np.ndarray) -> GraphSequence:
    """Build a dense keypoint graph for one extracted mouse.

    Diagonal slots are temporal self-edges and off-diagonal slots are directed
    spatial edges. With one mouse and twelve keypoints, this yields 12 nodes
    and 144 edge slots.

    Args:
        keypoints: Pose sequence shaped `[time, 1, 12, 2]`.

    Returns:
        Graph sequence with dense `[time, 12 * 12, 2]` edge features.
    """

    _ensure_single_mouse_pose_shape(keypoints)
    nodes = keypoints.astype(np.float32).reshape(
        keypoints.shape[0], NUM_KEYPOINTS, COORDINATES
    )
    edge_specs = tuple(
        EdgeSpec(
            source=source,
            target=target,
            kind="temporal" if source == target else "spatial",
        )
        for source in range(nodes.shape[1])
        for target in range(nodes.shape[1])
    )
    return _build_graph_sequence(
        variant="single_mouse_dense_keypoint",
        nodes=nodes,
        edge_specs=edge_specs,
    )


def build_within_mouse_dense_keypoint_graph(keypoints: np.ndarray) -> GraphSequence:
    """Build a triplet graph with independent dense keypoint graphs per mouse.

    Each mouse keeps dense keypoint-to-keypoint slots within its own body, but
    no spatial edge connects keypoints from different mice. With three mice and
    twelve keypoints per mouse, this yields 36 nodes and 432 edge slots.

    Args:
        keypoints: Pose sequence shaped `[time, 3, 12, 2]`.

    Returns:
        Graph sequence with three disconnected within-mouse keypoint graphs.
    """

    _ensure_pose_shape(keypoints)
    nodes = keypoints.astype(np.float32).reshape(
        keypoints.shape[0], NUM_MICE * NUM_KEYPOINTS, COORDINATES
    )
    edge_specs = tuple(
        EdgeSpec(
            source=flat_keypoint_node_id(mouse_id, source_keypoint),
            target=flat_keypoint_node_id(mouse_id, target_keypoint),
            kind="temporal" if source_keypoint == target_keypoint else "spatial",
        )
        for mouse_id in range(NUM_MICE)
        for source_keypoint in range(NUM_KEYPOINTS)
        for target_keypoint in range(NUM_KEYPOINTS)
    )
    return _build_graph_sequence(
        variant="within_mouse_dense_keypoint",
        nodes=nodes,
        edge_specs=edge_specs,
    )


def build_mouse_level_graph(keypoints: np.ndarray) -> GraphSequence:
    """Build a mouse-level graph from per-mouse pose centroids.

    Args:
        keypoints: Pose sequence shaped `[time, 3, 12, 2]`.

    Returns:
        Graph sequence with 3 mouse nodes and directed inter-mouse spatial edges.
    """

    _ensure_pose_shape(keypoints)
    nodes = keypoints.astype(np.float32).mean(axis=2)
    temporal_edges = tuple(
        EdgeSpec(source=mouse_id, target=mouse_id, kind="temporal")
        for mouse_id in range(NUM_MICE)
    )
    spatial_edges = tuple(
        EdgeSpec(source=source, target=target, kind="spatial")
        for source in range(NUM_MICE)
        for target in range(NUM_MICE)
        if source != target
    )
    return _build_graph_sequence(
        variant="mouse_level",
        nodes=nodes,
        edge_specs=temporal_edges + spatial_edges,
    )


def _flat_sparse_spatial_edges() -> tuple[EdgeSpec, ...]:
    """Return deterministic directed cross-mouse keypoint edge specs."""

    return tuple(
        EdgeSpec(
            source=flat_keypoint_node_id(source_mouse, source_keypoint),
            target=flat_keypoint_node_id(target_mouse, target_keypoint),
            kind="spatial",
        )
        for source_mouse in range(NUM_MICE)
        for source_keypoint in range(NUM_KEYPOINTS)
        for target_mouse in range(NUM_MICE)
        for target_keypoint in range(NUM_KEYPOINTS)
        if source_mouse != target_mouse
    )


def _ensure_pose_shape(keypoints: np.ndarray) -> None:
    """Validate the fixed MABe triplet pose shape expected by graph builders."""

    expected_tail = (NUM_MICE, NUM_KEYPOINTS, COORDINATES)
    if keypoints.ndim != 4 or keypoints.shape[1:] != expected_tail:
        raise ValueError(
            f"keypoints must be shaped [time, {NUM_MICE}, {NUM_KEYPOINTS}, 2]"
        )


def _ensure_single_mouse_pose_shape(keypoints: np.ndarray) -> None:
    """Validate the extracted single-mouse pose shape expected by graph builders."""

    expected_tail = (1, NUM_KEYPOINTS, COORDINATES)
    if keypoints.ndim != 4 or keypoints.shape[1:] != expected_tail:
        raise ValueError(f"keypoints must be shaped [time, 1, {NUM_KEYPOINTS}, 2]")


def _build_graph_sequence(
    *,
    variant: str,
    nodes: np.ndarray,
    edge_specs: tuple[EdgeSpec, ...],
) -> GraphSequence:
    """Materialize edge features and frame-level presence metadata."""

    edge_features = np.zeros((nodes.shape[0], len(edge_specs), 2), dtype=np.float32)
    temporal_edge_ids: list[int] = []
    spatial_edge_ids: list[int] = []

    for edge_idx, edge in enumerate(edge_specs):
        if edge.kind == "temporal":
            temporal_edge_ids.append(edge_idx)
            edge_features[1:, edge_idx] = (
                nodes[:-1, edge.source] - nodes[1:, edge.target]
            )
        else:
            spatial_edge_ids.append(edge_idx)
            edge_features[:, edge_idx] = nodes[:, edge.source] - nodes[:, edge.target]

    nodes_present = tuple(tuple(range(nodes.shape[1])) for _ in range(nodes.shape[0]))
    edges_present = (
        (tuple(spatial_edge_ids),)
        if nodes.shape[0] == 1
        else (tuple(spatial_edge_ids),)
        + tuple(tuple(range(len(edge_specs))) for _ in range(nodes.shape[0] - 1))
    )

    return GraphSequence(
        variant=variant,
        nodes=nodes,
        edge_features=edge_features,
        edge_specs=edge_specs,
        nodes_present=nodes_present,
        edges_present=edges_present,
    )
