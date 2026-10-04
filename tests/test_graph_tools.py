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


def test_count_edges_matches_networkx(graphs):
    for graph in graphs:
        assert run_tool(graph, "count_edges", {}) == {"edges": graph.number_of_edges()}


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
