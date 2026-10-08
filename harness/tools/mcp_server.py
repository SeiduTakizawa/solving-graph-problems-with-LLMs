"""MCP server: the harness's graph tools for other harnesses (Claude Code, generic agents, ...).

Our own loop calls the tools directly (run_tool). This server is the door for everyone else, so comparisons
use exactly the same tools: same names, same descriptions, same schemas (GRAPH_TOOLS, generated from the
Pydantic models), same validation and errors (run_tool), same result handles.

One server process serves one graph, given on the command line. With the stdio transport the client launches
a fresh process per question, so handles never leak from one question into the next.

    uv run python -m harness.tools.mcp_server --graph data/graphs/er/small/0.txt

Claude Code:  claude mcp add graph -- uv run python -m harness.tools.mcp_server --graph <file>

Known difference from our loop: read_result is listed from the start (our loop adds it after the first handle,
to keep the tool list unchanged on small graphs). submit_answer is not served yet: how baselines submit, and
whether our verifiers check them, is decided in M6.
"""
import argparse
import json

import anyio
import mcp_types as types
import networkx as nx
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from harness.brief import brief
from harness.tools.graph_tools import GRAPH_TOOLS, TOOLS, run_tool
from harness.tools.handles import HANDLE_LIMIT, READ_RESULT, HandleStore

INSTRUCTIONS = ("Tools for answering questions about one graph G. You cannot see G directly; "
                "use the tools to inspect it.")


def make_server(graph: nx.Graph, handle_limit: int | None = HANDLE_LIMIT) -> Server:
    """An MCP server for one graph. Nothing here talks to stdout: with stdio, stdout *is* the protocol."""
    store = HandleStore(handle_limit) if handle_limit else None
    schemas = GRAPH_TOOLS + ([READ_RESULT] if store else [])
    tools = [types.Tool(name=s["function"]["name"], description=s["function"]["description"],
                        input_schema=s["function"]["parameters"]) for s in schemas]

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        name, args = params.name, params.arguments or {}
        if name == "read_result" and store:
            result = store.read(args)
        elif name in TOOLS:
            try:
                result = run_tool(graph, name, args)  # validates the arguments, never raises on bad input
            except Exception as e:  # a buggy tool must not kill the server
                result = {"error": f"Tool {name} crashed: {brief(str(e))}"}
            if store and "error" not in result:
                result = store.compact(result)
        else:
            result = {"error": f"Unknown tool {name}. Available tools: {sorted(t.name for t in tools)}."}
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(result))],
                                    structured_content=result, is_error="error" in result)

    return Server("graph-tools", instructions=INSTRUCTIONS, on_list_tools=list_tools, on_call_tool=call_tool)


async def serve_stdio(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the graph tools for one graph over MCP (stdio).")
    parser.add_argument("--graph", required=True, help="graph file (networkx adjacency list)")
    parser.add_argument("--no-handles", action="store_true", help="send big results in full (ablation)")
    args = parser.parse_args()

    graph = nx.read_adjlist(args.graph, nodetype=int)
    anyio.run(serve_stdio, make_server(graph, handle_limit=None if args.no_handles else HANDLE_LIMIT))
