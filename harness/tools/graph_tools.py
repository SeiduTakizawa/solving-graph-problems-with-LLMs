"""Graph tools the agent can call. Each tool takes the graph plus its arguments and returns a dict.

Tools never raise: bad input comes back as {"error": "..."} so the model can correct itself.
"""
import networkx as nx


def get_neighbors(graph: nx.Graph, node: int) -> dict:
    if node not in graph:
        return {"error": f"Node {node} does not exist. Nodes are {sorted(graph.nodes)}."}
    return {"neighbors": sorted(graph.neighbors(node))}


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
    """Whether G contains a cycle.

    Return: {"has_cycle": True} or {"has_cycle": False}
    """
    try:
        nx.find_cycle(graph)
        return {"has_cycle": True}
    except nx.NetworkXNoCycle:
        return {"has_cycle": False}


# Tool name -> function. run_tool looks tools up here.
TOOL_FUNCTIONS = {
    "get_neighbors": get_neighbors,
    "graph_info": graph_info,
    "shortest_path": shortest_path,
    "connected_components": connected_components,
    "has_cycle": has_cycle,
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
            "description": "Return whether G contains a cycle.",
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
