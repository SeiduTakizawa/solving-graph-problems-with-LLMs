"""Result handles: big outputs are stored, the model sees a summary, and can read or submit them by handle."""
import json
from pathlib import Path

import networkx as nx
import pytest

from harness.loop import AgentConfig, run_agent
from harness.tools.graph_tools import TOOLS, run_tool
from harness.tools.handles import HANDLE_LIMIT, HandleStore, size
from harness.verifiers import verify_mst

GRAPH_FILES = sorted((Path(__file__).parents[1] / "data" / "graphs" / "er").glob("*/*.txt"))


@pytest.fixture(scope="module")
def big_graph():
    # 10,000 nodes, ~25,000 edges, connected enough to have one giant component.
    return nx.gnm_random_graph(10_000, 25_000, seed=1)


def test_size_counts_plain_values():
    assert size(7) == 1
    assert size([1, 2, 3]) == 3
    assert size([[0, 1], [1, 2]]) == 4  # 2 edges = 4 numbers
    assert size({"a": [1, 2], "b": 3}) == 3


def test_small_results_are_unchanged():
    store = HandleStore()
    result = {"neighbors": [1, 2, 3], "degree": 3}
    assert store.compact(result) == result and not store.values


def test_big_list_becomes_a_handle():
    store = HandleStore()
    edges = [[i, i + 1] for i in range(9_999)]
    shown = store.compact({"num_edges": 9_999, "edges": edges})
    assert shown["num_edges"] == 9_999  # small values stay
    assert shown["edges"] == {"handle": "result_1", "length": 9_999, "first_5": edges[:5]}
    assert "read_result" in shown["note"]
    assert store.resolve("result_1") is edges
    assert len(json.dumps(shown)) < 500  # instead of ~110,000 characters


def test_big_items_get_their_own_handles():
    # Components: 2 huge ones and a small one. Each huge one is stored; the small one stays visible.
    store = HandleStore()
    components = [list(range(9_000)), list(range(9_000, 9_900)), [9_900, 9_901]]
    shown = store.compact({"count": 3, "components": components})["components"]
    assert shown[0]["handle"] == "result_1" and shown[0]["length"] == 9_000
    assert shown[1]["handle"] == "result_2" and shown[1]["length"] == 900
    assert shown[2] == [9_900, 9_901]


def test_many_big_items_are_stored_as_one_list():
    store = HandleStore(limit=10)
    lists = [list(range(20)) for _ in range(30)]  # 30 big items: even their summaries are too many
    shown = store.compact({"x": lists})["x"]
    assert store.resolve(shown["handle"]) is lists and shown["length"] == 30
    assert size(shown["first_5"]) < 100  # the preview shows summaries, not 5 lists of 20


def test_read_result_pages_through_a_value():
    store = HandleStore()
    store.put(list(range(1_000)))
    page = store.read({"handle": "result_1", "offset": 990, "limit": 50})
    assert page["items"] == list(range(990, 1_000)) and page["more"] is False
    first = store.read({"handle": "result_1"})  # defaults: offset 0, limit 50
    assert first["items"] == list(range(50)) and first["more"] is True and first["length"] == 1_000


@pytest.mark.parametrize("args, message", [
    ({"handle": "result_7"}, "Unknown handle"),
    ({"handle": "result_1", "offset": -1}, "offset"),
    ({"handle": "result_1", "limit": 500}, "limit"),
    ({}, "handle"),
])
def test_read_result_errors(args, message):
    store = HandleStore()
    store.put([1, 2, 3])
    assert message in store.read(args)["error"]


def test_resolve_leaves_other_values_alone():
    store = HandleStore()
    store.put([1])
    assert store.resolve([4, 5]) == [4, 5] and store.resolve("hello") == "hello" and store.resolve(3) == 3


def test_the_current_dataset_never_makes_a_handle():
    # Guarantees that earlier experiments behave exactly as before handles existed.
    for path in GRAPH_FILES:
        graph = nx.read_adjlist(path, nodetype=int)
        store = HandleStore()
        for name, tool in TOOLS.items():
            fields = list(tool.args.model_fields)
            if not fields:
                calls = [{}]
            elif fields == ["node"]:
                calls = [{"node": n} for n in graph.nodes]
            else:  # two nodes: try every pair from the first node
                first = min(graph.nodes)
                calls = [dict(zip(fields, (first, n))) for n in graph.nodes]
            for args in calls:
                result = run_tool(graph, name, args)
                assert store.compact(result) == result, (path, name, args)
        assert not store.values


def test_big_graph_tool_output_stays_small(big_graph):
    store = HandleStore()
    for name in ("minimum_spanning_tree", "connected_components", "is_bipartite"):
        full = run_tool(big_graph, name, {})
        shown = store.compact(full)
        assert len(json.dumps(shown)) < 2_000, name  # the model sees a summary, not tens of thousands of numbers


def test_submitted_handle_is_checked_as_the_full_answer(big_graph):
    # End to end: MST of a 10,000-node graph, read a page of it, then submit it by handle.
    seen_tools = []

    def model(messages, tools):
        seen_tools.append([t["function"]["name"] for t in tools])
        step = len(seen_tools)
        if step == 1:
            call = ("minimum_spanning_tree", {})
        elif step == 2:
            call = ("read_result", {"handle": "result_1", "offset": 100, "limit": 3})
        else:
            call = ("submit_answer", {"answer": "result_1"})
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{step}", "type": "function", "function": {"name": call[0], "arguments": json.dumps(call[1])}}]}

    result = run_agent("What is a minimum spanning tree of G?", big_graph, answer_type="edge_list",
                       verify=lambda answer: verify_mst(big_graph, {}, answer), call_model=model)

    assert result.status == "submitted" and result.rejected == 0
    assert len(result.answer) == big_graph.number_of_nodes() - nx.number_connected_components(big_graph)
    assert "read_result" not in seen_tools[0]  # not offered before any handle exists
    assert "read_result" in seen_tools[1]  # offered once there is one
    tool_messages = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
    assert tool_messages[0]["edges"]["handle"] == "result_1"
    assert len(tool_messages[1]["items"]) == 3


def test_handles_can_be_switched_off(big_graph):
    def model(messages, tools):
        if len(messages) == 2:
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "connected_components", "arguments": "{}"}}]}
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "submit_answer", "arguments": '{"answer": 1}'}}]}

    result = run_agent("How many components?", big_graph, config=AgentConfig(handle_limit=None), call_model=model)
    shown = json.loads(result.messages[3]["content"])
    assert isinstance(shown["components"][0], list)  # the full list, no handle
    assert HANDLE_LIMIT == 200  # the documented default


def test_wrapped_handle_is_resolved():
    # What qwen3 actually sent on a 10,000-node graph: the schema said "array", so it wrapped the handle.
    store = HandleStore()
    edges = [[0, 1]]
    store.put(edges)
    assert store.resolve(["result_1"]) is edges
    assert store.resolve(["something"]) == ["something"]  # not a handle: unchanged


def test_submit_answer_accepts_handles_once_one_exists(big_graph):
    schemas = []

    def model(messages, tools):
        submit = next(t for t in tools if t["function"]["name"] == "submit_answer")
        schemas.append(submit["function"]["parameters"]["properties"]["answer"])
        if len(schemas) == 1:
            name, args = "minimum_spanning_tree", {}
        else:
            name, args = "submit_answer", {"answer": ["result_1"]}  # the wrapped form
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{len(schemas)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    result = run_agent("MST?", big_graph, answer_type="edge_list",
                       verify=lambda answer: verify_mst(big_graph, {}, answer), call_model=model)
    assert "anyOf" not in schemas[0]  # before any handle: exactly the old schema
    assert {"type": "string"}.items() <= schemas[1]["anyOf"][1].items()  # after: a handle string is allowed
    assert result.status == "submitted" and len(result.answer) > 9_000


def test_type_error_mentions_handles_when_there_are_some(big_graph):
    def model(messages, tools):
        if len(messages) == 2:
            name, args = "minimum_spanning_tree", {}
        elif len(messages) == 4:
            name, args = "submit_answer", {"answer": "edges"}  # neither edges nor a handle
        else:
            name, args = "cannot_answer", {"reason": "giving up"}
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{len(messages)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    result = run_agent("MST?", big_graph, answer_type="edge_list", call_model=model)
    assert 'handle, e.g. "result_1"' in json.loads(result.messages[5]["content"])["error"]
