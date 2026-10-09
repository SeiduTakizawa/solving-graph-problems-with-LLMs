"""Dataset questions and grading for each task. Used only by the eval runner, never by the agent.

The dataset's questions contain the whole edge list; only the parameters (node ids) are kept, and the
question the agent sees is rebuilt from the harness's template, so the graph never enters the prompt.
"""
import json
import random
import re
from pathlib import Path

import networkx as nx

from harness.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "data" / "graphs" / "er"
QUESTIONS_DIR = ROOT / "data" / "graphs_questions" / "er"

# Graphs 0-49 are for developing the harness, 50-99 are kept untouched for final results.
SPLITS = {"dev": range(0, 50), "test": range(50, 100)}

# How to read each task's parameters out of the dataset's question text.
PARAM_PATTERNS = {
    "node_degree": (r"degree of node (\d+)\?$", ["node"]),
    "connected_nodes": (r"neighbors of node (\d+)\?$", ["node"]),
    "edge_existence": (r"edge between nodes (\d+) and (\d+)\?$", ["u", "v"]),
    "connectivity": (r"path between nodes (\d+) and (\d+)\?$", ["source", "target"]),
    "shortest_path": (r"between nodes\s+(\d+) and (\d+)\?$", ["source", "target"]),
}


def load_graph(size: str, graph_id: int) -> nx.Graph:
    return nx.read_adjlist(GRAPHS_DIR / size / f"{graph_id}.txt", nodetype=int)


def parse_params(task: str, question: str) -> dict:
    if task not in PARAM_PATTERNS:
        return {}  # whole-graph questions have no parameters
    pattern, names = PARAM_PATTERNS[task]
    return dict(zip(names, map(int, re.search(pattern, question).groups())))


def waypoint_params(graph: nx.Graph, rng: random.Random, count: int = 2) -> list[dict]:
    """Questions for shortest_path_via: source, via and target in one component, with the waypoint off every
    shortest source-target path, so a plain shortest_path(source, target) gives a wrong answer."""
    found = []
    nodes = sorted(graph.nodes)
    for _ in range(500):
        source, via, target = rng.sample(nodes, 3)
        if not nx.has_path(graph, source, via) or not nx.has_path(graph, via, target):
            continue
        detour = nx.shortest_path_length(graph, source, via) + nx.shortest_path_length(graph, via, target)
        if detour > nx.shortest_path_length(graph, source, target):
            found.append({"source": source, "via": via, "target": target})
            if len(found) == count:
                break
    return found


def busy_node_params(graph: nx.Graph, rng: random.Random, count: int = 1, with_k: bool = False) -> list[dict]:
    """Questions about one node with at least 3 neighbors, so there is something to loop over.
    with_k: also a hop distance, 1 on the dense large graphs (2 hops is already almost every node), else 2."""
    busy = [n for n in sorted(graph.nodes) if graph.degree(n) >= 3]
    params = [{"node": n} for n in rng.sample(busy, min(count, len(busy)))]
    if with_k:
        k = 1 if graph.number_of_nodes() > 20 else 2
        params = [{**p, "k": k} for p in params]
    return params


# Tasks whose questions are generated from the graphs (fixed seed per graph), not read from the dataset.
GENERATED = {
    "shortest_path_via": waypoint_params,
    "hop_max_degree": lambda graph, rng: busy_node_params(graph, rng, with_k=True),
    "common_neighbors_max": busy_node_params,
    "triangle_count": busy_node_params,
}


def load_tasks(task: str, size: str, split: str, n: int) -> list[dict]:
    """Up to n questions of one task, spread over the graphs of the split (1st question of every graph,
    then the 2nd, ...)."""
    per_graph = []
    for graph_id in SPLITS[split]:
        if task in GENERATED:  # the same questions every time: the seed is the task, size and graph
            params = GENERATED[task](load_graph(size, graph_id), random.Random(f"{task}/{size}/{graph_id}"))
            per_graph.append([(graph_id, p) for p in params])
            continue
        questions = json.loads((QUESTIONS_DIR / size / f"{graph_id}.txt").read_text())["edgelist"].get(task, [])
        if isinstance(questions, str):
            questions = [questions]
        per_graph.append([(graph_id, parse_params(task, q)) for q in questions])

    items = []
    for round_ in range(max((len(p) for p in per_graph), default=0)):
        for pairs in per_graph:
            if round_ < len(pairs):
                graph_id, params = pairs[round_]
                items.append({"task": task, "graph_id": graph_id, "params": params,
                              "question": TASKS[task].make_question(params)})
    return items[:n]


def reference_answer(task: str, graph: nx.Graph, params: dict):
    """A correct answer, in the task's answer format."""
    if task == "node_count":
        return graph.number_of_nodes()
    if task == "edge_count":
        return graph.number_of_edges()
    if task == "node_degree":
        return graph.degree(params["node"])
    if task == "connected_nodes":
        return sorted(graph.neighbors(params["node"]))
    if task == "edge_existence":
        return graph.has_edge(params["u"], params["v"])
    if task == "connectivity":
        return nx.has_path(graph, params["source"], params["target"])
    if task == "connected_components_count":
        return nx.number_connected_components(graph)
    if task == "cycle_check":
        return len(nx.cycle_basis(graph)) > 0
    if task == "shortest_path":
        return nx.shortest_path(graph, params["source"], params["target"])  # one of possibly several
    if task == "mst":
        return [[u, v] for u, v in nx.minimum_spanning_edges(graph, data=False)]  # one of possibly many
    if task == "shortest_path_via":
        first = nx.shortest_path(graph, params["source"], params["via"])
        return first + nx.shortest_path(graph, params["via"], params["target"])[1:]
    if task == "hop_max_degree":  # highest degree, ties to the smallest id
        within = nx.single_source_shortest_path_length(graph, params["node"], cutoff=params["k"])
        return min((n for n in within if n != params["node"]), key=lambda n: (-graph.degree(n), n))
    if task == "common_neighbors_max":
        mine = set(graph.neighbors(params["node"]))
        return min((n for n in graph.nodes if n != params["node"]),
                   key=lambda n: (-len(mine & set(graph.neighbors(n))), n))
    if task == "triangle_count":
        return nx.triangles(graph, params["node"])
    raise ValueError(f"Unknown task {task}")


def is_correct(task: str, graph: nx.Graph, params: dict, answer) -> bool:
    if answer is None:  # cannot_answer, loop_detected, max_steps, ...
        return False
    if task == "connected_nodes":  # any order of the neighbors is fine
        return isinstance(answer, list) and sorted(answer) == reference_answer(task, graph, params)
    if task == "shortest_path":  # any shortest path is fine, not only the one networkx found
        return isinstance(answer, list) and verify_path(graph, params, answer)
    if task == "mst":  # any spanning forest is fine: the graphs are unweighted, so all have the same weight
        return isinstance(answer, list) and is_spanning_forest(graph, answer)
    if task == "shortest_path_via":  # any shortest route through the waypoint
        return isinstance(answer, list) and is_route_via(graph, params, answer)
    expected = reference_answer(task, graph, params)
    return answer == expected and type(answer) is type(expected)  # True must not count as 1


def verify_path(graph: nx.Graph, params: dict, path: list) -> bool:
    """Grading check for shortest_path, written independently of the harness's verifier."""
    source, target = params["source"], params["target"]
    return (len(path) > 0 and path[0] == source and path[-1] == target
            and all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))
            and len(path) - 1 == nx.shortest_path_length(graph, source, target))


def is_spanning_forest(graph: nx.Graph, edges: list) -> bool:
    """Grading check for mst, written independently of the harness's verifier: the edges are real edges
    of G, there are no repeats, and together they connect exactly what G connects without a cycle."""
    if not all(isinstance(e, list) and len(e) == 2 and graph.has_edge(*e) for e in edges):
        return False
    if len({frozenset(e) for e in edges}) != len(edges):
        return False
    forest = nx.Graph(edges)
    forest.add_nodes_from(graph)
    return nx.is_forest(forest) and nx.number_connected_components(forest) == nx.number_connected_components(graph)


def is_route_via(graph: nx.Graph, params: dict, route: list) -> bool:
    """Grading check for shortest_path_via, written independently of the harness's verifier."""
    s, v, t = params["source"], params["via"], params["target"]
    if not route or route[0] != s or route[-1] != t or v not in route:
        return False
    if not all(graph.has_edge(a, b) for a, b in zip(route, route[1:])):
        return False
    return len(route) - 1 == nx.shortest_path_length(graph, s, v) + nx.shortest_path_length(graph, v, t)


# --- Expected tool calls (process metrics, eval/analysis/process.py) ---
# Per task, the calls an ideal run makes. Each entry is one expected call ("slot"): the tools that can make it
# (equivalents), each with the arguments it should get, as {tool argument: question parameter}. Modelled on the
# expected_tools / expected_parameters of the GDS Agent benchmark (github.com/brs96/gds-agent-benchmarks), so the
# numbers are comparable. Written from the task, not from any run.
EXPECTED_CALLS = {
    "node_count": [{"graph_info": {}}],
    "edge_count": [{"graph_info": {}}],
    "node_degree": [{"degree": {"node": "node"}, "get_neighbors": {"node": "node"}}],
    "connected_nodes": [{"get_neighbors": {"node": "node"}}],
    "edge_existence": [{"has_edge": {"u": "u", "v": "v"}, "get_neighbors": {}}],
    "connectivity": [{"has_path": {"source": "source", "target": "target"},
                      "shortest_path": {"source": "source", "target": "target"}, "connected_components": {}}],
    "connected_components_count": [{"connected_components": {}}],
    "cycle_check": [{"has_cycle": {}}],
    "shortest_path": [{"shortest_path": {"source": "source", "target": "target"}}],
    "mst": [{"minimum_spanning_tree": {}}],
    "shortest_path_via": [{"shortest_path": {"source": "source", "target": "via"}},
                          {"shortest_path": {"source": "via", "target": "target"}}],
    "hop_max_degree": [{"neighborhood": {"node": "node", "k": "k"}, "distances_from": {"node": "node"},
                        "get_neighbors": {"node": "node"}},
                       {"degree": {}}],
    "common_neighbors_max": [{"get_neighbors": {"node": "node"}}],
    "triangle_count": [{"get_neighbors": {"node": "node"}}],
}
