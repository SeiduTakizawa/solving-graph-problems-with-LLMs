"""Graph tools the agent can call. Each tool takes the graph plus its arguments and returns a dict.

Tools never raise: bad input comes back as {"error": "..."} so the model can correct itself.

Each tool's arguments are a Pydantic model. It is the single source of truth: run_tool validates the
model's arguments with it, and the JSON schema the model sees (GRAPH_TOOLS) is generated from it.
"""
from dataclasses import dataclass
from typing import Annotated, Callable

import networkx as nx
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError


def _not_a_bool(value):
    # In Python True == 1, so a plain int would accept true as node 1. "3" is fine: it is clearly node 3.
    if isinstance(value, bool):
        raise ValueError("a node id must be an integer, not true/false")
    return value


NodeId = Annotated[int, BeforeValidator(_not_a_bool)]


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown argument is an error, not silently ignored


class NodeArgs(NoArgs):
    node: NodeId


class EdgeArgs(NoArgs):
    u: NodeId
    v: NodeId


class PathArgs(NoArgs):
    source: NodeId
    target: NodeId


def missing_node(graph: nx.Graph, *nodes: int) -> dict | None:
    """The error for the first of `nodes` that isn't in G, or None if they all are."""
    for node in nodes:
        if node not in graph:
            return {"error": f"Node {node} does not exist. Nodes are {sorted(graph.nodes)}."}
    return None


def get_neighbors(graph: nx.Graph, node: int) -> dict:
    if error := missing_node(graph, node):
        return error
    return {"neighbors": sorted(graph.neighbors(node))}


def degree(graph: nx.Graph, node: int) -> dict:
    """The degree of a node: how many edges touch it.

    Return:
      - if the node doesn't exist: {"error": "Node ... does not exist. ..."}
      - undirected G: {"degree": <number of neighbors>}
      - directed G:   {"degree": <in + out>, "in_degree": <...>, "out_degree": <...>}
    The model doesn't need to know whether G is directed: the tool handles both.
    """
    if error := missing_node(graph, node):
        return error
    if graph.is_directed():
        return {"degree": graph.degree(node), "in_degree": graph.in_degree(node), "out_degree": graph.out_degree(node)}
    return {"degree": graph.degree(node)}


def has_edge(graph: nx.Graph, u: int, v: int) -> dict:
    if error := missing_node(graph, u, v):
        return error
    return {"has_edge": graph.has_edge(u, v)}


def graph_info(graph: nx.Graph) -> dict:
    """Basic facts about G.

    Return: {"nodes": <number of nodes>, "edges": <number of edges>, "directed": <True/False>}
    """
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
    }


def shortest_path(graph: nx.Graph, source: int, target: int) -> dict:
    """A shortest path from source to target.

    Return:
      - if a node doesn't exist: {"error": "Node ... does not exist. ..."}  (like get_neighbors)
      - if there is a path:      {"reachable": True, "length": <number of edges>, "path": [source, ..., target]}
      - if there is no path:     {"reachable": False}
    """
    if error := missing_node(graph, source, target):
        return error
    try:
        path = nx.shortest_path(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return {"reachable": False}
    return {"reachable": True, "length": len(path) - 1, "path": path}


def connected_components(graph: nx.Graph) -> dict:
    """The connected components of G.

    Return: {"count": <number of components>, "components": [[nodes], [nodes], ...]}
    Sort the nodes inside each component, and sort the components by their smallest node,
    so the result is always the same for the same graph.
    """
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
    return {"count": len(components), "components": components}


def has_cycle(graph: nx.Graph) -> dict:
    """Whether G contains a cycle, and one cycle if it does.

    Return: {"has_cycle": True, "cycle": [nodes in order]} or {"has_cycle": False}
    """
    try:
        edges = nx.find_cycle(graph)  # [(0, 1), (1, 2), (2, 0)] -> cycle [0, 1, 2]
        return {"has_cycle": True, "cycle": [u for u, v in edges]}
    except nx.NetworkXNoCycle:
        return {"has_cycle": False}


def minimum_spanning_tree(graph: nx.Graph) -> dict:
    """A minimum spanning tree of G, or a minimum spanning forest (one tree per component) if G is not connected."""
    # Spanning trees are only defined here for undirected graphs.
    if graph.is_directed():
        return {"error": "The graph is directed. Expected an undirected graph."}

    mst_edges = list(nx.minimum_spanning_edges(graph, data=False))

    edges = []
    for u, v in mst_edges:
        edges.append([min(u, v), max(u, v)])

    edges.sort()

    return {
        "num_edges": len(edges),
        "edges": edges
    }


@dataclass(frozen=True)
class Tool:
    function: Callable  # function(graph, **args) -> dict
    args: type[BaseModel]
    description: str  # what the model is told the tool does


# Every tool the agent can be offered, in the order the model sees them.
TOOLS = {
    "get_neighbors": Tool(get_neighbors, NodeArgs, "Return the list of neighbors of a node in G."),
    "degree": Tool(degree, NodeArgs, "Return the degree of a node (for a directed graph: in-degree + out-degree)."),
    "has_edge": Tool(has_edge, EdgeArgs, "Return whether there is an edge between nodes u and v in G."),
    "graph_info": Tool(graph_info, NoArgs, "Return the number of nodes and edges of G, and whether it is directed."),
    "shortest_path": Tool(shortest_path, PathArgs, "Return a shortest path between two nodes and its length "
                                                   "(number of edges), or reachable=false if there is no path."),
    "connected_components": Tool(connected_components, NoArgs,
                                 "Return the number of connected components of G and the nodes in each one."),
    "has_cycle": Tool(has_cycle, NoArgs, "Return whether G contains a cycle, and the nodes of one cycle if it does."),
    "minimum_spanning_tree": Tool(minimum_spanning_tree, NoArgs,
                                  "Return the edges of a minimum spanning tree of G (a minimum spanning forest, "
                                  "one tree per connected component, if G is not connected)."),
}


def json_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema, trimmed to what the LLM needs: no titles (they only cost tokens) and no
    additionalProperties (extra arguments are still rejected by run_tool)."""
    schema = model.model_json_schema()

    def trim(node):
        if isinstance(node, dict):
            return {k: trim(v) for k, v in node.items() if k not in ("title", "additionalProperties")}
        return node

    schema = trim(schema)
    return {"type": "object", "properties": schema.get("properties", {}), "required": schema.get("required", [])}


# What the model is told about each tool (name, what it does, which arguments it takes).
GRAPH_TOOLS = [
    {"type": "function",
     "function": {"name": name, "description": tool.description, "parameters": json_schema(tool.args)}}
    for name, tool in TOOLS.items()
]


def describe_errors(error: ValidationError) -> str:
    """Pydantic's errors as one short line for the model, e.g. "node: Field required"."""
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg'].removeprefix('Value error, ')}"
                     for e in error.errors())


def run_tool(graph: nx.Graph, name: str, args: dict) -> dict:
    """Validate the arguments, run one tool by name and return its result."""
    if name not in TOOLS:
        return {"error": f"Unknown tool {name}. Available tools: {sorted(TOOLS)}."}
    tool = TOOLS[name]
    try:
        valid = tool.args.model_validate(args)
    except ValidationError as e:  # missing, extra or wrongly typed arguments
        return {"error": f"Bad arguments for {name}: {describe_errors(e)}"}
    return tool.function(graph, **valid.model_dump())
