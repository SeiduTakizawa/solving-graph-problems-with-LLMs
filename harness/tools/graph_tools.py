"""Graph tools the agent can call. Each tool takes the graph plus its arguments and returns a dict.

Tools never raise: bad input comes back as {"error": "..."} so the model can correct itself.
"""
import networkx as nx


def get_neighbors(graph: nx.Graph, node: int) -> dict:
    if node not in graph:
        return {"error": f"Node {node} does not exist. Nodes are {sorted(graph.nodes)}."}
    return {"neighbors": sorted(graph.neighbors(node))}


def degree(graph: nx.Graph, node: int) -> dict:
    """The degree of a node: how many edges touch it.

    Return:
      - if the node doesn't exist: {"error": "Node ... does not exist. ..."}
      - undirected G: {"degree": <number of neighbors>}
      - directed G:   {"degree": <in + out>, "in_degree": <...>, "out_degree": <...>}
    The model doesn't need to know whether G is directed: the tool handles both.
    """
    if node not in graph:
        return {"error": f"Node {node} does not exist. Nodes are {sorted(graph.nodes)}."}
    if graph.is_directed():
        return {"degree": graph.degree(node), "in_degree": graph.in_degree(node), "out_degree": graph.out_degree(node)}
    return {"degree": graph.degree(node)}


def has_edge(graph: nx.Graph, u: int, v: int) -> dict:
    for node in (u, v):
        if node not in graph:
            return {"error": f"Node {node} does not exist. Nodes are {sorted(graph.nodes)}."}
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
    if source not in graph:
        return {"error":f"Node {source} does not exist. Nodes are {sorted(graph.nodes)}."}
    if target not in graph:
        return {
            "error": f"Node {target} does not exist. Nodes are {sorted(graph.nodes)}."}
    try:
        path = nx.shortest_path(graph, source=source, target=target)
        return{
            "reachable":True,
            "length":len(path)-1,
            "path": path
        }
    except nx.NetworkXNoPath:
        return{
            "reachable":False
        }


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


# Tool name -> function. run_tool looks tools up here.
TOOL_FUNCTIONS = {
    "get_neighbors": get_neighbors,
    "degree": degree,
    "has_edge": has_edge,
    "graph_info": graph_info,
    "shortest_path": shortest_path,
    "connected_components": connected_components,
    "has_cycle": has_cycle,
    "minimum_spanning_tree": minimum_spanning_tree,
}

# What the model is told about each tool (name, what it does, which arguments it takes).
GRAPH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "Return the list of neighbors of a node in G.",
            "parameters": {
                "type": "object",
                "properties": {"node": {"type": "integer"}},
                "required": ["node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "degree",
            "description": "Return the degree of a node (for a directed graph: in-degree + out-degree).",
            "parameters": {
                "type": "object",
                "properties": {"node": {"type": "integer"}},
                "required": ["node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "has_edge",
            "description": "Return whether there is an edge between nodes u and v in G.",
            "parameters": {
                "type": "object",
                "properties": {"u": {"type": "integer"}, "v": {"type": "integer"}},
                "required": ["u", "v"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_info",
            "description": "Return the number of nodes and edges of G, and whether it is directed.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shortest_path",
            "description": "Return a shortest path between two nodes and its length (number of edges), "
                           "or reachable=false if there is no path.",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "integer"}, "target": {"type": "integer"}},
                "required": ["source", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "connected_components",
            "description": "Return the number of connected components of G and the nodes in each one.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "has_cycle",
            "description": "Return whether G contains a cycle, and the nodes of one cycle if it does.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "minimum_spanning_tree",
            "description": "Return the edges of a minimum spanning tree of G (a minimum spanning forest, "
                           "one tree per connected component, if G is not connected).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def run_tool(graph: nx.Graph, name: str, args: dict) -> dict:
    """Run one tool by name and return its result."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool {name}. Available tools: {sorted(TOOL_FUNCTIONS)}."}
    try:
        return TOOL_FUNCTIONS[name](graph, **args)
    except TypeError as e:  # wrong or missing arguments, e.g. get_neighbors without "node"
        return {"error": f"Bad arguments for {name}: {e}"}
