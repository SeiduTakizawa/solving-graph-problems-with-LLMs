"""Verifier tests: valid answers must pass, every kind of broken answer must be caught."""
from pathlib import Path

import networkx as nx

from harness.verifiers import verify_shortest_path

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
