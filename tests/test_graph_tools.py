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


def test_has_cycle_matches_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "has_cycle", {})
        assert result == {"has_cycle": len(nx.cycle_basis(graph)) > 0}


def test_has_cycle_small_cases():
    assert run_tool(nx.path_graph(4), "has_cycle", {}) == {"has_cycle": False}   # 0-1-2-3
    assert run_tool(nx.cycle_graph(3), "has_cycle", {}) == {"has_cycle": True}   # triangle
    assert run_tool(nx.empty_graph(3), "has_cycle", {}) == {"has_cycle": False}  # no edges


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
