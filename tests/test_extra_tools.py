"""The opt-in tools from the GDS Agent review (articulation points, bridges, k-core, triangles, coloring),
checked against networkx on every graph in the dataset, plus the opt-in plumbing (off by default)."""
from pathlib import Path

import networkx as nx
import pytest

from harness.loop import AgentConfig, run_agent
from harness.sandbox import run_python_tool
from harness.tools.graph_tools import DEFAULT_TOOLS, GRAPH_TOOLS, TOOLS, run_tool

GRAPH_FILES = sorted((Path(__file__).parents[1] / "data" / "graphs" / "er").glob("*/*.txt"))
OPT_IN = ["articulation_points", "bridges", "k_core", "triangles", "greedy_coloring"]


@pytest.fixture(scope="module")
def graphs():
    return [nx.read_adjlist(path, nodetype=int) for path in GRAPH_FILES]


def components(graph):
    return nx.number_connected_components(graph)


# --- Against networkx, and the evidence checks out ---

def test_articulation_points_match_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "articulation_points", {})
        assert result["nodes"] == sorted(nx.articulation_points(graph)) and result["count"] == len(result["nodes"])


def test_articulation_points_really_split_the_graph(graphs):
    for graph in graphs[:20]:
        for node in run_tool(graph, "articulation_points", {})["nodes"]:
            rest = graph.copy()
            rest.remove_node(node)
            assert components(rest) > components(graph)


def test_bridges_match_networkx(graphs):
    for graph in graphs:
        result = run_tool(graph, "bridges", {})
        assert result["edges"] == sorted([min(e), max(e)] for e in nx.bridges(graph))
        assert all(u < v for u, v in result["edges"]) and result["count"] == len(result["edges"])


def test_bridges_really_split_the_graph(graphs):
    for graph in graphs[:20]:
        for u, v in run_tool(graph, "bridges", {})["edges"]:
            rest = graph.copy()
            rest.remove_edge(u, v)
            assert components(rest) == components(graph) + 1


def test_k_core_matches_networkx(graphs):
    for graph in graphs:
        max_k = max(nx.core_number(graph).values())
        for k in range(1, max_k + 2):
            result = run_tool(graph, "k_core", {"k": k})
            assert result["nodes"] == sorted(nx.k_core(graph, k)) and result["max_k"] == max_k


def test_every_k_core_node_has_k_neighbors_inside(graphs):
    for graph in graphs[:20]:
        result = run_tool(graph, "k_core", {"k": 2})
        inside = set(result["nodes"])
        assert all(len(inside & set(graph.neighbors(n))) >= 2 for n in inside)


def test_k_core_ignores_self_loops():
    g = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 3), (3, 0)])
    assert run_tool(g, "k_core", {"k": 2})["nodes"] == [0, 1, 2]


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_k_core_bad_k_is_an_error(k):
    assert run_tool(nx.path_graph(3), "k_core", {"k": k})["error"].startswith("Bad arguments for k_core: k:")


def test_triangles_match_networkx(graphs):
    for graph in graphs:
        for node in graph.nodes:
            result = run_tool(graph, "triangles", {"node": node})
            assert result["count"] == nx.triangles(graph, node) == len(result["triangles"])
            for a, b, c in result["triangles"]:
                assert a == node and b < c and graph.has_edge(a, b) and graph.has_edge(a, c) and graph.has_edge(b, c)


def test_triangles_missing_node_is_an_error():
    assert "does not exist" in run_tool(nx.path_graph(3), "triangles", {"node": 9})["error"]


def test_greedy_coloring_is_proper(graphs):
    for graph in graphs:
        result = run_tool(graph, "greedy_coloring", {})
        color = {n: i for i, group in enumerate(result["classes"]) for n in group}
        assert sorted(color) == sorted(graph.nodes)  # every node exactly once
        assert all(color[u] != color[v] for u, v in graph.edges)
        assert result["colors_used"] == len(result["classes"]) and all(result["classes"])


def test_greedy_coloring_of_a_bipartite_graph_uses_two_colors():
    assert run_tool(nx.cycle_graph(6), "greedy_coloring", {})["colors_used"] == 2


def test_greedy_coloring_with_a_self_loop_is_an_error():
    assert "edge to itself" in run_tool(nx.Graph([(0, 1), (1, 1)]), "greedy_coloring", {})["error"]


def test_empty_graph():
    empty = nx.Graph()
    assert run_tool(empty, "k_core", {"k": 1}) == {"k": 1, "count": 0, "nodes": [], "max_k": 0}
    assert run_tool(empty, "greedy_coloring", {}) == {"colors_used": 0, "classes": []}


@pytest.mark.parametrize("name, args", [("articulation_points", {}), ("bridges", {}), ("k_core", {"k": 1}),
                                        ("triangles", {"node": 0})])
def test_directed_graph_is_an_error(name, args):
    assert "undirected graphs only" in run_tool(nx.DiGraph([(0, 1)]), name, args)["error"]


# --- Opt-in: nothing changes unless asked for ---

def test_new_tools_are_opt_in():
    assert [name for name in TOOLS if name not in DEFAULT_TOOLS] == OPT_IN
    assert list(DEFAULT_TOOLS) == list(TOOLS)[:len(DEFAULT_TOOLS)]  # the default text keeps its order


def offered(config):
    seen = {}

    def model(messages, tools):
        seen["names"] = [t["function"]["name"] for t in tools]
        seen["tools"] = tools
        return {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
            "name": "cannot_answer", "arguments": "{}"}}]}

    run_agent("?", nx.path_graph(3), call_model=model, config=config)
    return seen


def test_loop_offers_only_default_tools_by_default():
    assert offered(AgentConfig())["names"] == list(DEFAULT_TOOLS) + ["submit_answer", "cannot_answer"]


def test_loop_offers_an_opt_in_tool_when_asked():
    names = offered(AgentConfig(graph_tools=DEFAULT_TOOLS + ("bridges",)))["names"]
    assert "bridges" in names and "k_core" not in names


def test_run_python_text_is_unchanged_without_opt_in_tools():
    assert "Also available" not in run_python_tool("tools")["function"]["description"]
    assert "Also available: k_core(k)." in run_python_tool("tools", ("k_core",))["function"]["description"]


def test_run_python_names_offered_opt_in_tools():
    seen = offered(AgentConfig(graph_tools=DEFAULT_TOOLS + ("triangles",), python="tools"))
    code_tool = next(t for t in seen["tools"] if t["function"]["name"] == "run_python")
    assert "Also available: triangles(node)." in code_tool["function"]["description"]


def test_schemas_exist_for_new_tools():
    names = [t["function"]["name"] for t in GRAPH_TOOLS]
    assert all(name in names for name in OPT_IN)
