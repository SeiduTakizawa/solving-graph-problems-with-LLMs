"""code_value: inside run_python, tools return plain values (like networkx) and errors raise. No Docker needed."""
import networkx as nx
import pytest

from harness.sandbox.runner import code_value
from harness.tools.graph_tools import TOOLS, run_tool

# A square 0-1-2-3-0 with a tail 3-4, and a separate component 5-6.
G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (5, 6)])
TREE = nx.path_graph(4)  # 0-1-2-3: no cycle
DAG = nx.DiGraph([(0, 1), (1, 2)])
NOT_A_DAG = nx.DiGraph([(0, 1), (1, 2), (2, 0)])


def value(name, graph=G, **args):
    return code_value(name, run_tool(graph, name, args))


# TODO 1: errors raise, with the tool's message

def test_error_raises_value_error():
    with pytest.raises(ValueError, match="Node 99 does not exist"):
        value("get_neighbors", node=99)


def test_bad_arguments_raise_too():
    with pytest.raises(ValueError, match="Bad arguments"):
        value("neighborhood", node=0, k=0)


# TODO 2: one main value

@pytest.mark.parametrize("name, args, expected", [
    ("get_neighbors", {"node": 3}, [0, 2, 4]),
    ("degree", {"node": 3}, 3),
    ("has_edge", {"u": 0, "v": 1}, True),
    ("has_edge", {"u": 0, "v": 2}, False),
    ("connected_components", {}, [[0, 1, 2, 3, 4], [5, 6]]),
    ("neighborhood", {"node": 4, "k": 2}, [0, 2, 3]),
    ("distances_from", {"node": 4}, [[4], [3], [0, 2], [1]]),
])
def test_single_value(name, args, expected):
    assert value(name, **args) == expected


def test_minimum_spanning_tree_is_the_edge_list():
    edges = value("minimum_spanning_tree")
    assert isinstance(edges, list) and len(edges) == 5  # 7 nodes - 2 components


def test_the_loop_from_the_failed_runs_now_works():
    # Exactly what the model wrote in C1: loop over get_neighbors(...) as over a list of nodes.
    seen = [neighbor for neighbor in value("get_neighbors", node=3)]
    assert seen == [0, 2, 4]


# TODO 3: "maybe" tools: evidence or None

def test_shortest_path_is_the_path_or_none():
    assert value("shortest_path", source=4, target=1) in ([4, 3, 0, 1], [4, 3, 2, 1])
    assert value("shortest_path", source=0, target=5) is None


def test_has_path_is_the_path_or_none():
    assert value("has_path", source=0, target=4)[-1] == 4
    assert value("has_path", source=0, target=5) is None


def test_has_cycle_is_the_cycle_or_none():
    cycle = value("has_cycle")
    assert sorted(cycle) == [0, 1, 2, 3]
    assert value("has_cycle", graph=TREE) is None


def test_topological_sort_is_the_order_or_none():
    assert value("topological_sort", graph=DAG) == [0, 1, 2]
    assert value("topological_sort", graph=NOT_A_DAG) is None


# TODO 4: several equal parts: the dict stays

def test_graph_info_stays_a_dict():
    assert value("graph_info") == {"nodes": 7, "edges": 6, "directed": False}


def test_is_bipartite_stays_a_dict():
    assert value("is_bipartite", graph=TREE)["bipartite"] is True


# Every tool must be covered: a new tool that still returns its raw dict in code would bring the bug back.

KEEPS_DICT = {"graph_info", "is_bipartite"}


@pytest.mark.parametrize("name", sorted(set(TOOLS) - KEEPS_DICT))
def test_no_tool_returns_its_raw_dict_in_code(name):
    args = {"node": 0, "u": 0, "v": 1, "source": 0, "target": 4, "k": 1}
    graph = DAG if name == "topological_sort" else G
    fields = TOOLS[name].args.model_fields
    result = run_tool(graph, name, {key: v for key, v in args.items() if key in fields})
    assert code_value(name, result) != result, f"{name} still returns its tool dict in code"
