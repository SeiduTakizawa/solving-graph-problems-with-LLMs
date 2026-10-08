"""Graph tools checked against networkx on every graph in the dataset."""
import inspect
import json
from pathlib import Path

import networkx as nx
import pytest

from harness.tools.graph_tools import GRAPH_TOOLS, TOOLS, run_tool

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


def test_degree_matches_networkx(graphs):
    for graph in graphs:
        for node in graph.nodes:
            assert run_tool(graph, "degree", {"node": node}) == {"degree": graph.degree(node)}


def test_degree_of_directed_graph_is_in_plus_out():
    g = nx.DiGraph([(0, 1), (2, 1), (1, 3)])  # node 1: two edges in, one out
    assert run_tool(g, "degree", {"node": 1}) == {"degree": 3, "in_degree": 2, "out_degree": 1}


def test_degree_of_isolated_node_is_zero():
    g = nx.Graph([(0, 1)])
    g.add_node(2)
    assert run_tool(g, "degree", {"node": 2}) == {"degree": 0}


def test_degree_missing_node_is_an_error():
    result = run_tool(nx.path_graph(3), "degree", {"node": 99})
    assert "does not exist" in result["error"]


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


def test_has_path_matches_networkx(graphs):
    for graph in graphs:
        for source in graph.nodes:
            for target in graph.nodes:
                result = run_tool(graph, "has_path", {"source": source, "target": target})
                assert result["reachable"] == nx.has_path(graph, source, target)
                if result["reachable"]:
                    path = result["path"]
                    assert path[0] == source and path[-1] == target
                    assert all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))
                else:
                    assert result == {"reachable": False}


def test_has_path_follows_edge_directions():
    g = nx.DiGraph([(0, 1), (1, 2)])
    assert run_tool(g, "has_path", {"source": 0, "target": 2})["path"] == [0, 1, 2]
    assert run_tool(g, "has_path", {"source": 2, "target": 0}) == {"reachable": False}


def test_has_path_missing_node_is_an_error():
    assert "does not exist" in run_tool(nx.path_graph(3), "has_path", {"source": 0, "target": 9})["error"]


def check_bipartite_evidence(graph, result):
    """The evidence must prove the answer: a proper 2-coloring, or a real odd cycle."""
    g = graph.to_undirected() if graph.is_directed() else graph
    if result["bipartite"]:
        a, b = set(result["side_a"]), set(result["side_b"])
        assert a | b == set(g.nodes) and not a & b  # every node on exactly one side
        assert all((u in a) != (v in a) for u, v in g.edges)  # every edge goes between the sides
    else:
        cycle = result["odd_cycle"]
        assert len(cycle) % 2 == 1 and len(set(cycle)) == len(cycle)
        assert all(g.has_edge(u, v) for u, v in zip(cycle, cycle[1:] + cycle[:1]))


def test_is_bipartite_matches_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "is_bipartite", {})
        assert result["bipartite"] == nx.is_bipartite(graph)
        check_bipartite_evidence(graph, result)


@pytest.mark.parametrize("graph, expected", [
    (nx.path_graph(5), True),
    (nx.cycle_graph(6), True),     # even cycle
    (nx.cycle_graph(5), False),    # odd cycle
    (nx.complete_bipartite_graph(3, 4), True),
    (nx.complete_graph(4), False),  # has triangles
    (nx.empty_graph(3), True),     # no edges: trivially bipartite
    (nx.Graph([(0, 1), (2, 3), (3, 4), (4, 2)]), False),  # odd cycle in the second component only
    (nx.DiGraph([(0, 1), (1, 2), (2, 0)]), False),         # directions ignored: a triangle
], ids=["path", "even_cycle", "odd_cycle", "K3_4", "K4", "no_edges", "second_component", "directed_triangle"])
def test_is_bipartite_small_cases(graph, expected):
    result = run_tool(graph, "is_bipartite", {})
    assert result["bipartite"] is expected
    check_bipartite_evidence(graph, result)


def test_is_bipartite_on_random_graphs():
    for seed in range(200):
        graph = nx.gnp_random_graph(12, 0.2, seed=seed)
        result = run_tool(graph, "is_bipartite", {})
        assert result["bipartite"] == nx.is_bipartite(graph)
        check_bipartite_evidence(graph, result)


def check_order(graph, order):
    position = {node: i for i, node in enumerate(order)}
    assert sorted(order) == sorted(graph.nodes)
    assert all(position[u] < position[v] for u, v in graph.edges)


def test_topological_sort_on_random_dags():
    for seed in range(100):
        g = nx.gnp_random_graph(15, 0.2, seed=seed, directed=True)
        dag = nx.DiGraph((u, v) for u, v in g.edges if u < v)  # edges only go "up": no cycles
        dag.add_nodes_from(g)
        result = run_tool(dag, "topological_sort", {})
        assert result["is_dag"] is True
        check_order(dag, result["order"])


def test_topological_sort_finds_a_cycle():
    for seed in range(100):
        g = nx.gnp_random_graph(10, 0.3, seed=seed, directed=True)
        result = run_tool(g, "topological_sort", {})
        assert result["is_dag"] == nx.is_directed_acyclic_graph(g)
        if result["is_dag"]:
            check_order(g, result["order"])
        else:
            cycle = result["cycle"]
            assert all(g.has_edge(u, v) for u, v in zip(cycle, cycle[1:] + cycle[:1]))


def test_topological_sort_is_always_the_same():
    g = nx.DiGraph([(3, 1), (2, 1), (0, 2)])
    assert run_tool(g, "topological_sort", {})["order"] == [0, 2, 3, 1]


def test_topological_sort_of_undirected_graph_is_an_error():
    assert "directed" in run_tool(nx.path_graph(3), "topological_sort", {})["error"]


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


def test_missing_node_error_gives_count_and_range():
    assert run_tool(nx.path_graph(5), "get_neighbors", {"node": 9})["error"] == (
        "Node 9 does not exist. G has 5 nodes, with ids 0 to 4.")
    gaps = nx.Graph([(0, 5), (5, 10)])  # ids 0, 5, 10
    assert "between 0 and 10 (not every id" in run_tool(gaps, "degree", {"node": 3})["error"]


def test_missing_node_error_stays_short_on_a_big_graph():
    # It used to list every node: ~59,000 characters on 10,000 nodes, for one typo.
    big = nx.gnm_random_graph(10_000, 25_000, seed=1)
    for name, args in [("get_neighbors", {"node": 123_456}), ("has_edge", {"u": 0, "v": -1}),
                       ("shortest_path", {"source": 0, "target": 10_000}), ("has_path", {"source": -5, "target": 1})]:
        error = run_tool(big, name, args)["error"]
        assert len(error) < 120 and "10000 nodes" in error, (name, error)


# Argument validation (Pydantic): clear errors instead of crashes or wrong answers.

def test_node_id_as_string_is_accepted():
    # "3" is clearly node 3; before Pydantic it was looked up as the string "3" and reported missing.
    assert run_tool(nx.path_graph(5), "get_neighbors", {"node": "3"}) == {"neighbors": [2, 4]}


def test_true_is_not_a_node_id():
    result = run_tool(nx.path_graph(3), "get_neighbors", {"node": True})  # True == 1 in Python
    assert "Bad arguments" in result["error"] and "true/false" in result["error"]


def test_missing_argument_names_the_field():
    result = run_tool(nx.path_graph(3), "shortest_path", {"source": 0})
    assert "Bad arguments for shortest_path" in result["error"] and "target" in result["error"]


def test_extra_argument_names_the_field():
    result = run_tool(nx.path_graph(3), "graph_info", {"verbose": True})
    assert "verbose" in result["error"]


def test_fractional_node_id_is_an_error():
    assert "Bad arguments" in run_tool(nx.path_graph(3), "get_neighbors", {"node": 1.5})["error"]


def test_arguments_that_are_not_an_object_are_an_error():
    assert "Bad arguments" in run_tool(nx.path_graph(3), "get_neighbors", [1])["error"]


# The descriptions the model sees are generated from the argument models, so they must match the functions.

def test_every_tool_has_a_description():
    assert [t["function"]["name"] for t in GRAPH_TOOLS] == list(TOOLS)


@pytest.mark.parametrize("schema", GRAPH_TOOLS, ids=lambda t: t["function"]["name"])
def test_described_arguments_match_function(schema):
    fn = TOOLS[schema["function"]["name"]].function
    params = schema["function"]["parameters"]
    real_args = [p for p in inspect.signature(fn).parameters if p != "graph"]

    assert sorted(params["properties"]) == sorted(real_args)
    assert set(params["required"]) <= set(params["properties"])  # can't require an argument that doesn't exist


@pytest.mark.parametrize("schema", GRAPH_TOOLS, ids=lambda t: t["function"]["name"])
def test_schema_is_short(schema):
    # Pydantic adds titles and additionalProperties; they only cost tokens, so they are trimmed.
    text = json.dumps(schema)
    assert '"title"' not in text and "additionalProperties" not in text
