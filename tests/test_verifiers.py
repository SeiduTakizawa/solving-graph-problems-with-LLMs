"""Verifier tests: valid answers must pass, every kind of broken answer must be caught."""
from pathlib import Path

import networkx as nx

from harness.verifiers import verify_connectivity, verify_cycle, verify_mst, verify_neighbors, verify_shortest_path

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


# Connectivity: a "yes" must come with a real path; a "no" is not checked.
# Same graph as for MST: a square 0-1-2-3-0 with a tail 3-4, and a separate component 5-6.

def test_connectivity_yes_with_path_passes():
    assert verify_connectivity(MST_GRAPH, {"source": 0, "target": 4}, True, path=[0, 3, 4]) is None


def test_connectivity_any_path_counts_not_only_the_shortest():
    assert verify_connectivity(MST_GRAPH, {"source": 0, "target": 4}, True, path=[0, 1, 2, 3, 4]) is None


def test_connectivity_yes_without_path_is_sent_back():
    error = verify_connectivity(MST_GRAPH, {"source": 0, "target": 4}, True)
    assert error is not None and "path" in error


def test_connectivity_fake_path_is_caught():
    # 2-5 is not an edge: there is no path from 0 to 5 at all.
    error = verify_connectivity(MST_GRAPH, {"source": 0, "target": 5}, True, path=[0, 1, 2, 5])
    assert error is not None and "2-5" in error


def test_connectivity_path_to_the_wrong_node_is_caught():
    assert verify_connectivity(MST_GRAPH, {"source": 0, "target": 4}, True, path=[0, 1, 2]) is not None


def test_connectivity_no_is_not_checked():
    # Even a wrong "no" passes: proving there is no path would mean solving the question again.
    assert verify_connectivity(MST_GRAPH, {"source": 0, "target": 4}, False) is None


# Cycle: a "yes" must come with a real cycle; a "no" is not checked.

def test_cycle_yes_with_cycle_passes():
    assert verify_cycle(MST_GRAPH, {}, True, cycle=[0, 1, 2, 3]) is None


def test_cycle_with_first_node_repeated_passes():
    assert verify_cycle(MST_GRAPH, {}, True, cycle=[0, 1, 2, 3, 0]) is None


def test_cycle_yes_without_cycle_is_sent_back():
    error = verify_cycle(MST_GRAPH, {}, True)
    assert error is not None and "cycle" in error


def test_cycle_that_does_not_close_is_caught():
    # 0-1-2-3 is a path; closing it needs 3-0, which is an edge. 1-2-3-4 needs 4-1, which is not.
    error = verify_cycle(MST_GRAPH, {}, True, cycle=[1, 2, 3, 4])
    assert error is not None and "4-1" in error


def test_cycle_back_and_forth_is_caught():
    assert verify_cycle(MST_GRAPH, {}, True, cycle=[5, 6]) is not None  # just the edge 5-6, walked twice


def test_cycle_with_repeated_node_is_caught():
    assert verify_cycle(MST_GRAPH, {}, True, cycle=[0, 1, 2, 3, 2, 1]) is not None


def test_cycle_no_is_not_checked():
    assert verify_cycle(MST_GRAPH, {}, False) is None


def test_networkx_cycles_pass_on_the_dataset():
    for path in sorted((Path(__file__).parents[1] / "data" / "graphs" / "er").glob("*/*.txt")):
        graph = nx.read_adjlist(path, nodetype=int)
        for cycle in nx.cycle_basis(graph):
            assert verify_cycle(graph, {}, True, cycle=cycle) is None, path


# Neighbors: every listed node must be a real neighbor; a missing one is not checked.

def test_neighbors_pass_in_any_order():
    assert verify_neighbors(MST_GRAPH, {"node": 3}, [4, 0, 2]) is None


def test_neighbor_that_is_not_one_is_caught():
    error = verify_neighbors(MST_GRAPH, {"node": 3}, [0, 2, 4, 1])
    assert error is not None and "1" in error


def test_node_is_not_its_own_neighbor():
    assert verify_neighbors(MST_GRAPH, {"node": 3}, [3]) is not None


def test_repeated_neighbor_is_caught():
    assert verify_neighbors(MST_GRAPH, {"node": 3}, [0, 2, 2, 4]) is not None


def test_missing_neighbor_is_not_checked():
    # 4 is missing, but that can't be seen without listing the neighbors again.
    assert verify_neighbors(MST_GRAPH, {"node": 3}, [0, 2]) is None
