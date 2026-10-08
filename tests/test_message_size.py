"""Nothing the model reads may grow with the graph. Checked on a 10,000-node graph, with long wrong answers.

Tool results are covered by result handles (tests/test_handles.py); this covers everything else that can repeat
a value back to the model: errors, verifier messages, read_result pages, crash messages.
"""
import json

import networkx as nx
import pytest

from harness.answers import check_answer
from harness.brief import brief
from harness.tools.graph_tools import run_tool
from harness.tools.handles import HandleStore
from harness.verifiers import verify_connectivity, verify_cycle, verify_mst, verify_neighbors, verify_shortest_path

LIMIT = 300  # characters: generous for one message, far below anything proportional to 10,000 nodes


@pytest.fixture(scope="module")
def big():
    return nx.gnm_random_graph(10_000, 25_000, seed=1)


def test_brief_keeps_short_values_and_shortens_long_ones():
    assert brief([1, 2, 3]) == "[1, 2, 3]"
    long = brief(list(range(10_000)))
    assert len(long) <= 160 and long.startswith("[0, 1, 2") and "(10000 items)" in long
    assert "(5000 characters)" in brief("x" * 5_000)


def test_wrong_answer_is_not_repeated_in_full(big):
    edges_as_strings = [[str(u), str(v)] for u, v in list(big.edges)[:9_999]]
    error = check_answer(edges_as_strings, "edge_list")
    assert len(error) < LIMIT and "(9999 items)" in error


def test_verifier_messages_stay_short(big):
    long_cycle_with_repeat = list(range(5_000)) + [3]
    messages = [
        verify_cycle(big, {}, True, cycle=long_cycle_with_repeat),
        verify_cycle(big, {}, True, cycle=list(range(5_000))),  # not a real cycle in G
        verify_mst(big, {}, [[0, 1]] * 2 + [[u, v] for u, v in big.edges]),
        verify_neighbors(big, {"node": 0}, list(range(1, 9_000))),
        verify_shortest_path(big, {"source": 0, "target": 1}, list(range(9_000))),
        verify_connectivity(big, {"source": 0, "target": 1}, True, path=list(range(9_000))),
    ]
    for message in messages:
        assert message is None or len(message) < LIMIT, message[:200]
    assert messages[0] == "The cycle visits node 3 more than once."


def test_missing_node_errors_stay_short(big):
    for name, args in [("get_neighbors", {"node": -1}), ("degree", {"node": 10**9}),
                       ("has_edge", {"u": 0, "v": -1}), ("shortest_path", {"source": 0, "target": -1}),
                       ("has_path", {"source": -1, "target": 0})]:
        assert len(run_tool(big, name, args)["error"]) < LIMIT


def test_read_result_pages_stay_short():
    store = HandleStore()
    shown = store.compact({"x": [list(range(i * 300, i * 300 + 300)) for i in range(50)]})  # 50 big items
    page = store.read({"handle": shown["x"]["handle"], "limit": 100})
    assert len(json.dumps(page)) < 20 * LIMIT  # 50 small summaries, not 15,000 numbers


def test_unknown_handle_error_stays_short():
    store = HandleStore()
    store.compact({"components": [list(range(i * 300, i * 300 + 300)) for i in range(500)]})
    error = store.read({"handle": "result_99999"})["error"]
    assert len(error) < LIMIT and "more" in error


def test_tool_crash_message_stays_short(monkeypatch):
    from harness.loop import run_agent

    def crash(graph, name, args):
        raise ValueError("bad: " + str(list(range(10_000))))

    monkeypatch.setattr("harness.loop.run_tool", crash)
    replies = iter([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "graph_info", "arguments": "{}"}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "cannot_answer", "arguments": '{"reason": "x"}'}}]},
    ])
    result = run_agent("?", nx.path_graph(3), call_model=lambda m, t: next(replies))
    assert len(result.messages[3]["content"]) < LIMIT
