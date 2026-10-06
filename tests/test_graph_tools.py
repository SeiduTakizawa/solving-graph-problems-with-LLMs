"""Graph tools checked against networkx on every graph in the dataset."""
import inspect
from pathlib import Path

import networkx as nx
import pytest

from harness.tools.graph_tools import GRAPH_TOOLS, TOOL_FUNCTIONS, run_tool

GRAPH_FILES = sorted((Path(__file__).parents[1] / "data" / "graphs" / "er").glob("*/*.txt"))


@pytest.fixture(scope="module")
def graphs():
    return [nx.read_adjlist(path, nodetype=int) for path in GRAPH_FILES]


def test_dataset_found(graphs):
    assert len(graphs) == 300  # 100 each of small, medium, large


def test_get_neighbors_matches_networkx(graphs):
    for graph in graphs:
        for node in graph.nodes:
            result = run_tool(graph, "get_neighbors", {"node": node})
            assert result == {"neighbors": sorted(graph.neighbors(node))}


def test_has_edge_matches_networkx(graphs):
    for graph in graphs:
        nodes = sorted(graph.nodes)
        for u in nodes:
            for v in nodes:
                assert run_tool(graph, "has_edge", {"u": u, "v": v}) == {"has_edge": graph.has_edge(u, v)}


def test_has_edge_missing_node_is_an_error():
    result = run_tool(nx.path_graph(3), "has_edge", {"u": 0, "v": 99})
    assert "does not exist" in result["error"]


def test_graph_info_matches_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "graph_info", {})
        assert result == {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "directed": False}


def test_shortest_path_matches_networkx(graphs):
    for graph in graphs:
        nodes = sorted(graph.nodes)
        for source in nodes:
            for target in nodes:
                result = run_tool(graph, "shortest_path", {"source": source, "target": target})
                if nx.has_path(graph, source, target):
                    assert result["reachable"] is True
                    assert result["length"] == nx.shortest_path_length(graph, source, target)
                    path = result["path"]
                    assert path[0] == source and path[-1] == target
                    assert len(path) == result["length"] + 1
                    assert all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))  # a real path in G
                else:
                    assert result == {"reachable": False}


def test_shortest_path_missing_node_is_an_error():
    result = run_tool(nx.path_graph(3), "shortest_path", {"source": 0, "target": 99})
    assert "does not exist" in result["error"]


def test_connected_components_matches_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "connected_components", {})
        expected = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
        assert result == {"count": len(expected), "components": expected}


def is_real_cycle(graph, cycle):
    closing = list(zip(cycle, cycle[1:] + cycle[:1]))  # every step, including last node back to the first
    return len(cycle) >= 3 and len(set(cycle)) == len(cycle) and all(graph.has_edge(u, v) for u, v in closing)


def test_has_cycle_matches_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "has_cycle", {})
        assert result["has_cycle"] == (len(nx.cycle_basis(graph)) > 0)
        if result["has_cycle"]:
            assert is_real_cycle(graph, result["cycle"])
        else:
            assert result == {"has_cycle": False}


def test_has_cycle_small_cases():
    assert run_tool(nx.path_graph(4), "has_cycle", {}) == {"has_cycle": False}   # 0-1-2-3
    triangle = run_tool(nx.cycle_graph(3), "has_cycle", {})
    assert triangle["has_cycle"] is True and sorted(triangle["cycle"]) == [0, 1, 2]
    assert run_tool(nx.empty_graph(3), "has_cycle", {}) == {"has_cycle": False}  # no edges


def test_minimum_spanning_tree_matches_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "minimum_spanning_tree", {})
        edges = result["edges"]
        assert result["num_edges"] == len(edges)
        assert all(u < v for u, v in edges) and edges == sorted(edges)  # same graph, same result
        assert all(graph.has_edge(u, v) for u, v in edges)
        forest = nx.Graph(edges)
        forest.add_nodes_from(graph)
        assert nx.is_forest(forest)
        assert nx.number_connected_components(forest) == nx.number_connected_components(graph)  # spans G


def test_minimum_spanning_tree_small_cases():
    assert run_tool(nx.path_graph(4), "minimum_spanning_tree", {}) == {"num_edges": 3, "edges": [[0, 1], [1, 2], [2, 3]]}
    assert run_tool(nx.cycle_graph(3), "minimum_spanning_tree", {})["num_edges"] == 2   # triangle: drop one edge
    assert run_tool(nx.empty_graph(3), "minimum_spanning_tree", {}) == {"num_edges": 0, "edges": []}
    two_parts = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4)])  # a triangle and a separate edge: a forest of 2 trees
    assert run_tool(two_parts, "minimum_spanning_tree", {})["num_edges"] == 3


def test_minimum_spanning_tree_directed_is_an_error():
    result = run_tool(nx.DiGraph([(0, 1), (1, 2)]), "minimum_spanning_tree", {})
    assert "directed" in result["error"]


# Tools never raise: every bad input comes back as an error the model can read.

def test_missing_node_is_an_error():
    result = run_tool(nx.path_graph(3), "get_neighbors", {"node": 99})
    assert "does not exist" in result["error"]


def test_unknown_tool_is_an_error():
    result = run_tool(nx.path_graph(3), "delete_graph", {})
    assert "Unknown tool" in result["error"]


@pytest.mark.parametrize("args", [{}, {"vertex": 1}, {"node": 1, "extra": 2}])
def test_bad_arguments_are_an_error(args):
    result = run_tool(nx.path_graph(3), "get_neighbors", args)
    assert "Bad arguments" in result["error"]


# The descriptions the model sees must match the real functions.

def test_every_tool_has_a_description_and_vice_versa():
    described = {t["function"]["name"] for t in GRAPH_TOOLS}
    assert described == set(TOOL_FUNCTIONS)


@pytest.mark.parametrize("schema", GRAPH_TOOLS, ids=lambda t: t["function"]["name"])
def test_described_arguments_match_function(schema):
    fn = TOOL_FUNCTIONS[schema["function"]["name"]]
    params = schema["function"]["parameters"]
    real_args = [p for p in inspect.signature(fn).parameters if p != "graph"]

    assert sorted(params["properties"]) == sorted(real_args)
    assert set(params["required"]) <= set(params["properties"])  # can't require an argument that doesn't exist
