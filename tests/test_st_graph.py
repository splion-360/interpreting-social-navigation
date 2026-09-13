"""File description: Tests for social-attention graph builders."""

import numpy as np
import pytest

from st_graph import (
    build_dense_keypoint_graph,
    build_flat_sparse_keypoint_graph,
    build_mouse_level_graph,
    build_single_mouse_dense_keypoint_graph,
    build_within_mouse_dense_keypoint_graph,
    flat_keypoint_node_id,
)


def make_keypoints(frames: int = 3) -> np.ndarray:
    """Create deterministic keypoints shaped `[frames, 3, 12, 2]`."""

    keypoints = np.zeros((frames, 3, 12, 2), dtype=np.float32)
    for frame_idx in range(frames):
        for mouse_idx in range(3):
            for keypoint_idx in range(12):
                keypoints[frame_idx, mouse_idx, keypoint_idx] = (
                    frame_idx + mouse_idx * 100 + keypoint_idx * 10,
                    frame_idx * 2 + mouse_idx * 1000 + keypoint_idx,
                )
    return keypoints


def test_flat_keypoint_graph_has_expected_nodes_and_edges() -> None:
    keypoints = make_keypoints()

    graph = build_flat_sparse_keypoint_graph(keypoints)

    assert graph.nodes.shape == (3, 36, 2)
    assert graph.edge_features.shape == (3, 900, 2)
    assert len(graph.edge_specs) == 900
    assert graph.edge_count == 900
    assert len(set(graph.edge_specs)) == graph.edge_count
    assert graph.nodes_present[0] == tuple(range(36))


def test_dense_keypoint_graph_has_one_slot_per_node_pair() -> None:
    keypoints = make_keypoints()

    graph = build_dense_keypoint_graph(keypoints)

    assert graph.variant == "dense_keypoint"
    assert graph.nodes.shape == (3, 36, 2)
    assert graph.edge_features.shape == (3, 36 * 36, 2)
    assert graph.edge_count == 36 * 36
    assert len(set(graph.edge_specs)) == graph.edge_count
    assert graph.nodes_present[0] == tuple(range(36))


def test_single_mouse_dense_keypoint_graph_has_one_slot_per_node_pair() -> None:
    keypoints = make_keypoints()[:, 1:2]

    graph = build_single_mouse_dense_keypoint_graph(keypoints)

    assert graph.variant == "single_mouse_dense_keypoint"
    assert graph.nodes.shape == (3, 12, 2)
    assert graph.edge_features.shape == (3, 12 * 12, 2)
    assert graph.edge_count == 12 * 12
    assert len(set(graph.edge_specs)) == graph.edge_count
    assert graph.nodes_present[0] == tuple(range(12))
    assert len(graph.edges_present[0]) == 12 * 11
    assert len(graph.edges_present[1]) == 12 * 12


def test_within_mouse_dense_keypoint_graph_has_independent_mouse_graphs() -> None:
    keypoints = make_keypoints()

    graph = build_within_mouse_dense_keypoint_graph(keypoints)

    assert graph.variant == "within_mouse_dense_keypoint"
    assert graph.nodes.shape == (3, 36, 2)
    assert graph.edge_features.shape == (3, 3 * 12 * 12, 2)
    assert graph.edge_count == 3 * 12 * 12
    assert len(set(graph.edge_specs)) == graph.edge_count
    assert graph.nodes_present[0] == tuple(range(36))
    assert len(graph.edges_present[0]) == 3 * 12 * 11
    assert len(graph.edges_present[1]) == 3 * 12 * 12


def test_flat_keypoint_graph_uses_stable_node_ids() -> None:
    keypoints = make_keypoints()

    graph = build_flat_sparse_keypoint_graph(keypoints)

    assert flat_keypoint_node_id(mouse_id=2, keypoint_id=11) == 35
    np.testing.assert_array_equal(graph.nodes[0, 35], keypoints[0, 2, 11])


def test_dense_keypoint_graph_uses_dense_source_target_indexing() -> None:
    keypoints = make_keypoints()

    graph = build_dense_keypoint_graph(keypoints)
    source = flat_keypoint_node_id(mouse_id=0, keypoint_id=1)
    target = flat_keypoint_node_id(mouse_id=2, keypoint_id=3)

    assert graph.edge_id(source=source, target=target, kind="spatial") == (
        source * 36 + target
    )
    assert graph.edge_id(source=target, target=target, kind="temporal") == (
        target * 36 + target
    )


def test_single_mouse_dense_graph_uses_dense_source_target_indexing() -> None:
    graph = build_single_mouse_dense_keypoint_graph(make_keypoints()[:, 0:1])

    assert graph.edge_id(source=1, target=3, kind="spatial") == 15
    assert graph.edge_id(source=3, target=3, kind="temporal") == 39


def test_within_mouse_dense_graph_uses_mouse_local_dense_indexing() -> None:
    graph = build_within_mouse_dense_keypoint_graph(make_keypoints())
    source = flat_keypoint_node_id(mouse_id=2, keypoint_id=1)
    target = flat_keypoint_node_id(mouse_id=2, keypoint_id=3)
    temporal = flat_keypoint_node_id(mouse_id=1, keypoint_id=3)

    assert graph.edge_id(source=source, target=target, kind="spatial") == 303
    assert graph.edge_id(source=temporal, target=temporal, kind="temporal") == 183


def test_within_mouse_dense_graph_excludes_cross_mouse_edges() -> None:
    graph = build_within_mouse_dense_keypoint_graph(make_keypoints())

    with pytest.raises(ValueError):
        graph.edge_id(
            source=flat_keypoint_node_id(mouse_id=0, keypoint_id=1),
            target=flat_keypoint_node_id(mouse_id=2, keypoint_id=1),
            kind="spatial",
        )


def test_flat_keypoint_graph_features_match_source_minus_target_vectors() -> None:
    keypoints = make_keypoints()

    graph = build_flat_sparse_keypoint_graph(keypoints)
    temporal_edge = graph.edge_id(source=0, target=0, kind="temporal")
    spatial_edge = graph.edge_id(
        source=flat_keypoint_node_id(mouse_id=0, keypoint_id=1),
        target=flat_keypoint_node_id(mouse_id=2, keypoint_id=3),
        kind="spatial",
    )

    assert temporal_edge not in graph.edges_present[0]
    assert temporal_edge in graph.edges_present[1]
    np.testing.assert_array_equal(
        graph.edge_features[1, temporal_edge], np.array([-1, -2])
    )

    source = keypoints[0, 0, 1]
    target = keypoints[0, 2, 3]
    np.testing.assert_array_equal(graph.edge_features[0, spatial_edge], source - target)


def test_dense_keypoint_graph_features_match_source_minus_target_vectors() -> None:
    keypoints = make_keypoints()

    graph = build_dense_keypoint_graph(keypoints)
    temporal_edge = graph.edge_id(source=0, target=0, kind="temporal")
    spatial_edge = graph.edge_id(
        source=flat_keypoint_node_id(mouse_id=0, keypoint_id=1),
        target=flat_keypoint_node_id(mouse_id=0, keypoint_id=3),
        kind="spatial",
    )

    assert temporal_edge not in graph.edges_present[0]
    assert temporal_edge in graph.edges_present[1]
    np.testing.assert_array_equal(
        graph.edge_features[1, temporal_edge], np.array([-1, -2])
    )

    source = keypoints[0, 0, 1]
    target = keypoints[0, 0, 3]
    np.testing.assert_array_equal(graph.edge_features[0, spatial_edge], source - target)


def test_single_mouse_dense_graph_features_match_source_minus_target_vectors() -> None:
    keypoints = make_keypoints()[:, 0:1]

    graph = build_single_mouse_dense_keypoint_graph(keypoints)
    temporal_edge = graph.edge_id(source=0, target=0, kind="temporal")
    spatial_edge = graph.edge_id(source=1, target=3, kind="spatial")

    assert temporal_edge not in graph.edges_present[0]
    assert temporal_edge in graph.edges_present[1]
    np.testing.assert_array_equal(
        graph.edge_features[1, temporal_edge], np.array([-1, -2])
    )
    np.testing.assert_array_equal(
        graph.edge_features[0, spatial_edge],
        keypoints[0, 0, 1] - keypoints[0, 0, 3],
    )


def test_within_mouse_dense_graph_features_match_source_minus_target_vectors() -> None:
    keypoints = make_keypoints()

    graph = build_within_mouse_dense_keypoint_graph(keypoints)
    temporal = flat_keypoint_node_id(mouse_id=2, keypoint_id=0)
    temporal_edge = graph.edge_id(source=temporal, target=temporal, kind="temporal")
    source = flat_keypoint_node_id(mouse_id=2, keypoint_id=1)
    target = flat_keypoint_node_id(mouse_id=2, keypoint_id=3)
    spatial_edge = graph.edge_id(source=source, target=target, kind="spatial")

    assert temporal_edge not in graph.edges_present[0]
    assert temporal_edge in graph.edges_present[1]
    np.testing.assert_array_equal(
        graph.edge_features[1, temporal_edge], np.array([-1, -2])
    )
    np.testing.assert_array_equal(
        graph.edge_features[0, spatial_edge],
        keypoints[0, 2, 1] - keypoints[0, 2, 3],
    )


def test_mouse_level_graph_uses_three_mouse_centroids() -> None:
    keypoints = make_keypoints()

    graph = build_mouse_level_graph(keypoints)

    assert graph.nodes.shape == (3, 3, 2)
    assert graph.edge_features.shape == (3, 9, 2)
    assert graph.edge_count == 9
    assert len(set(graph.edge_specs)) == graph.edge_count
    np.testing.assert_array_equal(graph.nodes[0, 2], keypoints[0, 2].mean(axis=0))


def test_mouse_level_graph_uses_directed_inter_mouse_edges() -> None:
    keypoints = make_keypoints()

    graph = build_mouse_level_graph(keypoints)
    forward = graph.edge_id(source=0, target=2, kind="spatial")
    backward = graph.edge_id(source=2, target=0, kind="spatial")

    assert forward != backward
    np.testing.assert_array_equal(
        graph.edge_features[0, forward], graph.nodes[0, 0] - graph.nodes[0, 2]
    )
    np.testing.assert_array_equal(
        graph.edge_features[0, backward],
        graph.nodes[0, 2] - graph.nodes[0, 0],
    )


def test_graph_builders_reject_wrong_pose_shape() -> None:
    keypoints = np.zeros((3, 36, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="keypoints must be shaped"):
        build_flat_sparse_keypoint_graph(keypoints)

    with pytest.raises(ValueError, match="keypoints must be shaped"):
        build_dense_keypoint_graph(keypoints)

    with pytest.raises(ValueError, match="keypoints must be shaped"):
        build_mouse_level_graph(keypoints)

    with pytest.raises(ValueError, match="keypoints must be shaped"):
        build_single_mouse_dense_keypoint_graph(make_keypoints())

    with pytest.raises(ValueError, match="keypoints must be shaped"):
        build_within_mouse_dense_keypoint_graph(keypoints)
