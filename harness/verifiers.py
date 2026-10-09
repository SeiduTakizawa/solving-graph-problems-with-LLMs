"""Verifiers: plain-Python checks of the model's final answer, run before the harness accepts it.

Each verifier takes the graph, the question's parameters and the submitted answer, and returns
None if the answer is valid, or an error message the model can act on.
"""
import networkx as nx

from harness.brief import brief


def check_path(graph: nx.Graph, source: int, target: int, path) -> str | None:
    """None if `path` is a real path in G from source to target, otherwise what is wrong with it."""
    # 1. An empty path is never valid.
    if not isinstance(path, list) or len(path) == 0:
        return "The path is empty."

    # 2. The path must start at source and end at target (either end wrong is an error, hence "or").
    if path[0] != source or path[-1] != target:
        return f"The path must start at {source} and end at {target}."

    # 3. Every step (u, v) of the path must be a real edge of G.
    for u, v in zip(path, path[1:]):
        if not graph.has_edge(u, v):
            return f"{u}-{v} is not an edge in G."

    return None


def verify_shortest_path(graph: nx.Graph, params: dict, answer: list[int]) -> str | None:
    """Check that `answer` is a shortest path from params["source"] to params["target"].

    Return None if it is, otherwise a short message saying what is wrong.
    """
    source, target = params["source"], params["target"]
    path = answer

    # 1-3. It must be a real path from source to target.
    error = check_path(graph, source, target, path)
    if error:
        return error

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


# The next three are partial checks. A "no" (no path, no cycle) and a missing neighbor can't be checked without
# solving the question again, so those are accepted as they are: only the part with evidence is checked.

def verify_connectivity(graph: nx.Graph, params: dict, answer: bool, path: list[int] | None = None) -> str | None:
    """A "yes, there is a path" must come with that path, and the path must be real. A "no" is not checked."""
    if answer is False:
        return None
    source, target = params["source"], params["target"]
    if not path:
        return f"You answered true: also send the path you found from {source} to {target} in the path field."
    return check_path(graph, source, target, path)


def verify_cycle(graph: nx.Graph, params: dict, answer: bool, cycle: list[int] | None = None) -> str | None:
    """A "yes, there is a cycle" must come with a real cycle of G. A "no" is not checked."""
    if answer is False:
        return None
    if not cycle:
        return "You answered true: also send the nodes of one cycle, in order, in the cycle field."
    if not isinstance(cycle, list):
        return "The cycle must be a list of nodes."

    # Accept the cycle written with its first node repeated at the end, e.g. [0, 1, 2, 0].
    if len(cycle) > 1 and cycle[0] == cycle[-1]:
        cycle = cycle[:-1]

    # 1. In a simple undirected graph a cycle has at least 3 nodes (0-1-0 just walks one edge back and forth).
    if len(cycle) < 3:
        return f"{brief(cycle)} is not a cycle: a cycle needs at least 3 different nodes."

    # 2. No node twice: otherwise it is a walk, not a cycle.
    seen = set()
    for node in cycle:
        if node in seen:  # name the repeated node instead of repeating the whole cycle back
            return f"The cycle visits node {node} more than once."
        seen.add(node)

    # 3. Every step must be a real edge, including the one closing the cycle (last node back to the first).
    for u, v in zip(cycle, cycle[1:] + cycle[:1]):
        if not graph.has_edge(u, v):
            return f"{u}-{v} is not an edge in G."

    return None


def verify_neighbors(graph: nx.Graph, params: dict, answer: list[int]) -> str | None:
    """Every listed node must really be a neighbor of params["node"], listed once.
    A missing neighbor is not checked."""
    node = params["node"]
    if len(set(answer)) != len(answer):
        return "Some nodes are listed more than once."
    for other in answer:
        if not graph.has_edge(node, other):
            return f"{other} is not a neighbor of {node}."
    return None


def verify_path_via(graph: nx.Graph, params: dict, answer: list[int]) -> str | None:
    """Check that `answer` is a shortest route from source to target that passes through `via`.

    A route may visit a node twice (the best way to the waypoint and on can overlap). It is shortest when its
    length equals dist(source, via) + dist(via, target).
    """
    source, via, target = params["source"], params["via"], params["target"]
    route = answer

    # 1-3. A real route from source to target (every step an edge).
    error = check_path(graph, source, target, route)
    if error:
        return error

    # 4. It must pass through the waypoint.
    if via not in route:
        return f"The route must pass through node {via}."

    # 5. And be as short as possible.
    length = len(route) - 1
    best = nx.shortest_path_length(graph, source, via) + nx.shortest_path_length(graph, via, target)
    if length > best:
        return f"This route is valid but not the shortest: it takes {length} steps, a route with {best} exists."
    return None


# How to deliver a long evidence list (2026-10-09: right counts of 390 and 730 triangles were rejected because the
# model copied 2-5 of them by hand and ran out of steps). Added to every rejection of missing or partial evidence.
LONG_LIST_HOW = (" If the list is long, don't copy it: in run_python set `result` to the full list; the reply then "
                 "shows a result_handle (e.g. \"result_2\"), and you send that handle in the {field} field. A list a "
                 "tool returned as a handle can be sent by its handle too.")

# Counts with evidence (v8): the answer comes with what was counted. Each listed item is checked, and the count
# must match the list, so an overcount or an invented item is caught. A missed item is not (finding it would mean
# solving the question again), like verify_neighbors.

def verify_components_count(graph: nx.Graph, params: dict, answer: int, components: list | None = None) -> str | None:
    """The components must split G's nodes (each node exactly once), with no edge between two of them, and there
    must be `answer` of them. No edge between them means no real component was split in two: an overcount is
    caught. Two components listed as one (an undercount) is not checked."""
    if not components:
        return ("Also send the components you counted, one list of nodes per component, in the components field."
                + LONG_LIST_HOW.format(field="components"))
    if not isinstance(components, list) or not all(isinstance(c, list) and c for c in components):
        return "components must be a list of non-empty lists of nodes."
    if len(components) != answer:
        return (f"You answered {answer} but listed {len(components)} components: send all of them."
                + LONG_LIST_HOW.format(field="components"))
    where = {}
    for i, component in enumerate(components):
        for node in component:
            if node not in graph:
                return f"Node {node} is not in G."
            if node in where:
                return f"Node {node} is listed in more than one component (or twice)."
            where[node] = i
    if len(where) != graph.number_of_nodes():
        missing = next(n for n in graph.nodes if n not in where)
        return f"Every node must be in a component: node {missing} is missing."
    for u, v in graph.edges:
        if where[u] != where[v]:
            return f"{u}-{v} is an edge, so {u} and {v} are in the same component."
    return None


def verify_triangle_count(graph: nx.Graph, params: dict, answer: int, triangles: list | None = None) -> str | None:
    """Every listed triangle must include params["node"], be a real triangle of G, and be listed once; there must
    be `answer` of them. A missed triangle is not checked."""
    node = params["node"]
    if not triangles:  # none listed: fine for a count of 0
        return None if answer == 0 else ("Also send the triangles you counted, each as its three nodes, in the "
                                         "triangles field." + LONG_LIST_HOW.format(field="triangles"))
    if not isinstance(triangles, list) or not all(isinstance(t, list) for t in triangles):
        return "triangles must be a list of [a, b, c] node lists."
    if len(triangles) != answer:
        return (f"You answered {answer} but listed {len(triangles)} triangles: send all of them."
                + LONG_LIST_HOW.format(field="triangles"))
    seen = set()
    for triangle in triangles:
        nodes = frozenset(triangle)
        if len(triangle) != 3 or len(nodes) != 3:
            return f"{brief(triangle)} is not three different nodes."
        if node not in nodes:
            return f"{brief(triangle)} does not include node {node}."
        a, b, c = triangle
        for u, v in ((a, b), (b, c), (a, c)):
            if not graph.has_edge(u, v):
                return f"{brief(triangle)} is not a triangle: {u}-{v} is not an edge in G."
        if nodes in seen:
            return f"The triangle {brief(sorted(nodes))} is listed twice."
        seen.add(nodes)
    return None
