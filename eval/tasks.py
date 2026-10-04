"""Dataset questions and grading for each task. Used only by the eval runner, never by the agent.

The dataset's questions contain the whole edge list; only the parameters (node ids) are kept, and the
question the agent sees is rebuilt from the harness's template, so the graph never enters the prompt.
"""
import json
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


def load_tasks(task: str, size: str, split: str, n: int) -> list[dict]:
    """Up to n questions of one task, spread over the graphs of the split (1st question of every graph,
    then the 2nd, ...)."""
    per_graph = []
    for graph_id in SPLITS[split]:
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
    raise ValueError(f"Unknown task {task}")


def is_correct(task: str, graph: nx.Graph, params: dict, answer) -> bool:
    if answer is None:  # cannot_answer, loop_detected, max_steps, ...
        return False
    if task == "connected_nodes":  # any order of the neighbors is fine
        return isinstance(answer, list) and sorted(answer) == reference_answer(task, graph, params)
    if task == "shortest_path":  # any shortest path is fine, not only the one networkx found
        return isinstance(answer, list) and verify_path(graph, params, answer)
    expected = reference_answer(task, graph, params)
    return answer == expected and type(answer) is type(expected)  # True must not count as 1


def verify_path(graph: nx.Graph, params: dict, path: list) -> bool:
    """Grading check for shortest_path, written independently of the harness's verifier."""
    source, target = params["source"], params["target"]
    return (len(path) > 0 and path[0] == source and path[-1] == target
            and all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))
            and len(path) - 1 == nx.shortest_path_length(graph, source, target))
