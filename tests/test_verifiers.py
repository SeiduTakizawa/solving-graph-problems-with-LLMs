"""Verifier tests: valid answers must pass, every kind of broken answer must be caught."""
from pathlib import Path

import networkx as nx

from harness.verifiers import verify_mst, verify_shortest_path

# 0-1-2-3 is the shortest way from 0 to 3 (3 steps); 0-4-5-6-3 is a longer detour (4 steps).
GRAPH = nx.Graph([(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 3)])
FROM_0_TO_3 = {"source": 0, "target": 3}


def test_shortest_path_passes():
    assert verify_shortest_path(GRAPH, FROM_0_TO_3, [0, 1, 2, 3]) is None


def test_empty_path_is_caught():
    assert verify_shortest_path(GRAPH, FROM_0_TO_3, []) is not None


def test_wrong_start_is_caught():
    assert verify_shortest_path(GRAPH, FROM_0_TO_3, [1, 2, 3]) is not None


def test_wrong_end_is_caught():
    assert verify_shortest_path(GRAPH, FROM_0_TO_3, [0, 1, 2]) is not None


def test_fake_edge_is_caught():
    # 0-3 is not an edge in G: the model "teleported".
    error = verify_shortest_path(GRAPH, FROM_0_TO_3, [0, 3])
    assert error is not None
    assert "0" in error and "3" in error  # the message should say which step is wrong


def test_longer_path_is_caught():
    # A real path, but not the shortest one.
    error = verify_shortest_path(GRAPH, FROM_0_TO_3, [0, 4, 5, 6, 3])
    assert error is not None
    assert "shortest" in error


def test_path_to_itself():
    assert verify_shortest_path(GRAPH, {"source": 2, "target": 2}, [2]) is None


def test_networkx_paths_pass_on_the_dataset():
    # Every shortest path networkx finds on the small dev graphs must be accepted.
    graphs_dir = Path(__file__).parents[1] / "data" / "graphs" / "er" / "small"
    for graph_id in range(50):
        graph = nx.read_adjlist(graphs_dir / f"{graph_id}.txt", nodetype=int)
        for source, paths in nx.all_pairs_shortest_path(graph):
            for target, path in paths.items():
                assert verify_shortest_path(graph, {"source": source, "target": target}, path) is None


# MST. The graphs are unweighted, so any spanning tree (forest, if G is not connected) is a valid answer.

# A square 0-1-2-3-0 with a tail 3-4, and a separate component 5-6.
MST_GRAPH = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (5, 6)])
NO_PARAMS = {}


def test_mst_passes():
    assert verify_mst(MST_GRAPH, NO_PARAMS, [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6]]) is None


def test_mst_any_spanning_forest_passes():
    # Dropping a different edge of the square is just as good.
    assert verify_mst(MST_GRAPH, NO_PARAMS, [[1, 2], [2, 3], [3, 0], [3, 4], [5, 6]]) is None


def test_mst_edge_direction_does_not_matter():
    assert verify_mst(MST_GRAPH, NO_PARAMS, [[1, 0], [2, 1], [3, 2], [4, 3], [6, 5]]) is None


def test_mst_of_graph_without_edges_is_empty():
    assert verify_mst(nx.empty_graph(3), NO_PARAMS, []) is None


def test_mst_fake_edge_is_caught():
    # 0-2 is a diagonal of the square, not an edge of G.
    error = verify_mst(MST_GRAPH, NO_PARAMS, [[0, 1], [0, 2], [2, 3], [3, 4], [5, 6]])
    assert error is not None
    assert "0" in error and "2" in error  # the message should name the bad edge


def test_mst_missing_node_is_caught():
    assert verify_mst(MST_GRAPH, NO_PARAMS, [[0, 1], [1, 2], [2, 3], [3, 99], [5, 6]]) is not None


def test_mst_self_loop_is_caught():
    assert verify_mst(MST_GRAPH, NO_PARAMS, [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [2, 2]]) is not None


def test_mst_repeated_edge_is_caught():
    error = verify_mst(MST_GRAPH, NO_PARAMS, [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [1, 0]])  # 0-1 twice
    assert error is not None


def test_mst_cycle_is_caught():
    # The whole square is a cycle.
    error = verify_mst(MST_GRAPH, NO_PARAMS, [[0, 1], [1, 2], [2, 3], [3, 0], [3, 4], [5, 6]])
    assert error is not None
    assert "cycle" in error


def test_mst_missing_edges_are_caught():
    # A valid forest, but node 4 is left out (and so is the 5-6 component).
    error = verify_mst(MST_GRAPH, NO_PARAMS, [[0, 1], [1, 2], [2, 3]])
    assert error is not None
    assert "2" in error  # two edges are missing


def test_networkx_msts_pass_on_the_dataset():
    # The MST networkx finds on every dataset graph (all sizes) must be accepted.
    for path in sorted((Path(__file__).parents[1] / "data" / "graphs" / "er").glob("*/*.txt")):
        graph = nx.read_adjlist(path, nodetype=int)
        edges = [[u, v] for u, v in nx.minimum_spanning_edges(graph, data=False)]
        assert verify_mst(graph, NO_PARAMS, edges) is None, path
