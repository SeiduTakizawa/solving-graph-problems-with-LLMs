"""Verifiers: plain-Python checks of the model's final answer, run before the harness accepts it.

Each verifier takes the graph, the question's parameters and the submitted answer, and returns
None if the answer is valid, or an error message the model can act on.
"""
import networkx as nx


def verify_shortest_path(graph: nx.Graph, params: dict, answer: list[int]) -> str | None:
    """Check that `answer` is a shortest path from params["source"] to params["target"].

    Return None if it is, otherwise a short message saying what is wrong.
    """
    source, target = params["source"], params["target"]
    path = answer

    # 1. An empty path is never valid.
    if len(path) == 0:
        return "The path is empty."

    # 2. The path must start at source and end at target (either end wrong is an error, hence "or").
    if path[0] != source or path[-1] != target:
        return f"The path must start at {source} and end at {target}."

    # 3. Every step (u, v) of the path must be a real edge of G.
    for u, v in zip(path, path[1:]):
        if not graph.has_edge(u, v):
            return f"{u}-{v} is not an edge in G."

    # 4. The path must be the shortest. Its length is the number of steps.
    length, best = len(path) - 1, nx.shortest_path_length(graph, source, target)
    if length > best:
        return f"This path is valid but not the shortest: it takes {length} steps, a path with {best} exists."

    return None
