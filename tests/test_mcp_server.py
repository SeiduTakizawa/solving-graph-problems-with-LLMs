"""MCP server: other harnesses must get exactly the tools, results and errors our own loop gets."""
import json
import sys
from pathlib import Path

import anyio
import networkx as nx
import pytest
from mcp import Client, StdioServerParameters

from harness.tools.graph_tools import GRAPH_TOOLS, TOOLS, run_tool
from harness.tools.handles import READ_RESULT
from harness.tools.mcp_server import make_server

ROOT = Path(__file__).parents[1]
SMALL_0 = ROOT / "data" / "graphs" / "er" / "small" / "0.txt"


def with_client(graph, fn, **server_options):
    """Run `fn(client)` against an in-process server for `graph`, from a plain (non-async) test."""
    async def main():
        async with Client(make_server(graph, **server_options)) as client:
            return await fn(client)
    return anyio.run(main)


def text_of(result):
    return json.loads(result.content[0].text)


def test_tools_are_exactly_the_harness_tools():
    async def listed(client):
        return (await client.list_tools()).tools

    tools = with_client(nx.path_graph(3), listed)
    expected = GRAPH_TOOLS + [READ_RESULT]
    assert [t.name for t in tools] == [s["function"]["name"] for s in expected]
    for tool, schema in zip(tools, expected):
        assert tool.description == schema["function"]["description"]
        assert tool.input_schema == schema["function"]["parameters"]  # same schema the model sees in our loop


def test_results_match_run_tool_on_the_dataset():
    # Every tool, on a few dataset graphs, through MCP: identical to calling run_tool directly.
    for path in sorted((ROOT / "data" / "graphs" / "er").glob("*/[0-2].txt")):
        graph = nx.read_adjlist(path, nodetype=int)
        node = min(graph.nodes)
        calls = []
        for name, tool in TOOLS.items():
            fields = list(tool.args.model_fields)
            calls.append((name, dict(zip(fields, [node, max(graph.nodes)]))))

        async def call_all(client):
            return [(name, args, await client.call_tool(name, args)) for name, args in calls]

        for name, args, result in with_client(graph, call_all):
            expected = run_tool(graph, name, args)
            assert text_of(result) == expected, (path, name)
            assert result.structured_content == expected
            assert result.is_error == ("error" in expected)


@pytest.mark.parametrize("name, args, message", [
    ("shortest_path", {"source": 0}, "target: Field required"),
    ("get_neighbors", {"node": True}, "not true/false"),
    ("get_neighbors", {"node": 99}, "does not exist"),
    ("delete_graph", {}, "Unknown tool"),
])
def test_errors_match_our_loop(name, args, message):
    async def call(client):
        return await client.call_tool(name, args)

    result = with_client(nx.path_graph(3), call)
    assert result.is_error and message in text_of(result)["error"]


def test_handles_work_through_mcp():
    big = nx.gnm_random_graph(5_000, 12_000, seed=2)

    async def mst_then_read(client):
        mst = text_of(await client.call_tool("minimum_spanning_tree", {}))
        page = text_of(await client.call_tool("read_result", {"handle": mst["edges"]["handle"], "limit": 3}))
        return mst, page

    mst, page = with_client(big, mst_then_read)
    assert mst["edges"]["handle"] == "result_1" and mst["edges"]["length"] == mst["num_edges"]
    assert len(page["items"]) == 3 and page["more"] is True


def test_handles_can_be_switched_off():
    big = nx.gnm_random_graph(5_000, 12_000, seed=2)

    async def mst(client):
        return text_of(await client.call_tool("minimum_spanning_tree", {})), (await client.list_tools()).tools

    result, tools = with_client(big, mst, handle_limit=None)
    assert isinstance(result["edges"], list) and "read_result" not in [t.name for t in tools]


def test_server_runs_over_stdio_as_a_subprocess():
    # The way Claude Code uses it: launch the server as a process and talk over stdin/stdout.
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "harness.tools.mcp_server", "--graph", str(SMALL_0)],
                                   cwd=str(ROOT))

    async def main():
        async with Client(params) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            return names, text_of(await client.call_tool("degree", {"node": 4}))

    names, degree = anyio.run(main)
    assert "degree" in names and degree == {"degree": 2}
