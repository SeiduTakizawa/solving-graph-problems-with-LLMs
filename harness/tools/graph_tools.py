"""Graph tools the agent can call. Each tool takes the graph plus its arguments and returns a dict.

Tools never raise: bad input comes back as {"error": "..."} so the model can correct itself.
"""
import networkx as nx


def get_neighbors(graph: nx.Graph, node: int) -> dict:
    if node not in graph:
        return {"error": f"Node {node} does not exist. Nodes are {sorted(graph.nodes)}."}
    return {"neighbors": sorted(graph.neighbors(node))}


def count_edges(graph: nx.Graph) -> dict:
    return {"edges": graph.number_of_edges()}


# Tool name -> function. run_tool looks tools up here.
TOOL_FUNCTIONS = {
    "get_neighbors": get_neighbors,
    "count_edges": count_edges,
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
            "name": "count_edges",
            "description": "Return the number of edges in G.",
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
