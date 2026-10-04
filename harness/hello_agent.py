"""Your first agent: a model + 2 tools + a loop.

Run it:   uv run python -m harness.hello_agent
Test it:  uv run pytest tests/test_hello_agent.py
"""
import json

import litellm
import networkx as nx

MODEL = "ollama_chat/qwen3:8b"

SYSTEM_PROMPT = (
    "You answer questions about a graph G. You cannot see G directly; use the tools to inspect it. "
    "When you know the answer, call submit_answer."
)

# What the model is told about each tool (name, what it does, which arguments it takes).
TOOLS = [
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
            "name": "submit_answer",
            "description": "Submit your final answer. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_edges",
            "description": "Return the number of edges this current graph has.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def run_tool(graph: nx.Graph, name: str, args: dict) -> dict:
    """Run one tool and return its result. Never raises: errors go back to the model."""
    if name == "get_neighbors":
        node = args.get("node")
        if node not in graph:
            return {"error": f"Node {node} does not exist. Nodes are {sorted(graph.nodes)}."}
        return {"neighbors": sorted(graph.neighbors(node))}
    elif name == "count_edges":
        return {"edges": graph.number_of_edges()}
    return {"error": f"Unknown tool {name}."}


def call_model(messages: list[dict]) -> dict:
    """Send the conversation to the model, return its reply as a plain dict."""
    response = litellm.completion(model=MODEL, messages=messages, tools=TOOLS)
    return response.choices[0].message.model_dump()


def run_agent(question: str, graph: nx.Graph, call_model=call_model, max_steps: int = 10):
    """Run the agent loop. Returns (answer, messages); answer is None if it never submitted."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        # 1. Ask the model, and remember what it said.
        reply = call_model(messages)
        messages.append(reply)

        # 2. Run each tool it asked for and show it the result.
        tool_calls = reply["tool_calls"] or []
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])  # arguments arrive as a JSON string

            if name == "submit_answer":
                return args["answer"], messages

            try:
                result = run_tool(graph, name, args)
            except Exception as e:  # a buggy tool must not kill the agent
                result = {"error": f"Tool {name} crashed: {e}"}
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],  # tells the model which request this answers
                "content": json.dumps(result),
            })

        # 3. Plain-text reply (no tool calls): nudge it back to the tools.
        if not tool_calls:
            messages.append({
                "role": "user",
                "content": "Please use the tools to inspect G, or call submit_answer with your final answer.",
            })

    return None, messages  # ran out of steps


if __name__ == "__main__":
    graph = nx.read_adjlist("data/graphs/er/small/0.txt", nodetype=int)
    answer, messages = run_agent("How many edges does G have?", graph)

    for m in messages:
        print(f"[{m['role']}]", m.get("content") or "", m.get("tool_calls") or "")
    print("\nAnswer:", answer, "| correct:", graph.number_of_edges())
