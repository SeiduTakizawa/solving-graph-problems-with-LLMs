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


def verify_mst(graph: nx.Graph, params: dict, answer: list[list[int]]) -> str | None:
    """Check that `answer` (a list of [u, v] edges) is a minimum spanning tree of G,
    or a minimum spanning forest if G is not connected.

    The dataset's graphs are unweighted, so every spanning forest has the same weight and is minimal:
    it is enough to check that the answer is a spanning forest. (With weights you would also compare
    its total weight against the minimum.)

    Return None if it is, otherwise a short message saying what is wrong (name the bad edge if there is one).
    """
    edges = answer

    # 1. Every edge must be a real edge of G (this also catches nodes that don't exist, and self-loops).
    for u, v in edges:
        if not graph.has_edge(u, v):
            return f"{u}-{v} is not an edge in G."

    # 2. No edge may appear twice. [2, 5] and [5, 2] are the same edge, so compare them as sets.
    seen = set()
    for u, v in edges:
        if frozenset((u, v)) in seen:
            return f"The edge {u}-{v} appears more than once."
        seen.add(frozenset((u, v)))

    # 3. The edges must not form a cycle. Union-find: every node points towards the root of its tree.
    #    An edge whose ends already have the same root closes a cycle.
    parent = {}

    def root(node):
        while parent.get(node, node) != node:
            node = parent[node]
        return node

    for u, v in edges:
        ru, rv = root(u), root(v)
        if ru == rv:
            return f"The edge {u}-{v} closes a cycle: its ends are already joined by the other edges."
        parent[ru] = rv

    # 4. The forest must span G. A forest on n nodes with c trees has exactly n - c edges, and after 1-3 the
    #    answer is a forest inside G, so it can have at most that many: the only way left to fail is too few.
    missing = graph.number_of_nodes() - nx.number_connected_components(graph) - len(edges)
    if missing > 0:
        return (f"{missing} edges are missing: a spanning tree must connect every node that G connects "
                "(one tree per connected component).")

    return None
