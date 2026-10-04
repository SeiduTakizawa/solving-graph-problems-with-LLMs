"""Tests for the hello agent loop, using a fake model that replies from a script."""
import json

import networkx as nx

from harness.hello_agent import run_agent

# Same graph as data/graphs/er/small/0.txt; node 4 has neighbors [2, 5].
GRAPH = nx.Graph([(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)])


def tool_call(name, args, call_id="call_1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }


def text_reply(text):
    return {"role": "assistant", "content": text, "tool_calls": None}


def fake_model(*replies):
    """A 'model' that returns the given replies in order."""
    script = iter(replies)
    return lambda messages: next(script)


def test_submits_answer():
    model = fake_model(tool_call("submit_answer", {"answer": 2}))
    answer, _ = run_agent("What is the degree of node 4?", GRAPH, call_model=model)
    assert answer == 2


def test_tool_result_goes_back_to_model():
    model = fake_model(
        tool_call("get_neighbors", {"node": 4}, call_id="call_A"),
        tool_call("submit_answer", {"answer": 2}, call_id="call_B"),
    )
    answer, messages = run_agent("What is the degree of node 4?", GRAPH, call_model=model)

    assert answer == 2
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_A"
    assert json.loads(tool_msg["content"]) == {"neighbors": [2, 5]}


def test_message_history_order():
    model = fake_model(
        tool_call("get_neighbors", {"node": 4}),
        tool_call("submit_answer", {"answer": 2}),
    )
    _, messages = run_agent("What is the degree of node 4?", GRAPH, call_model=model)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]


def test_stops_after_max_steps():
    # A model that keeps asking for neighbors forever.
    looping = lambda messages: tool_call("get_neighbors", {"node": 4})
    answer, _ = run_agent("What is the degree of node 4?", GRAPH, call_model=looping, max_steps=3)
    assert answer is None


def test_bonus_text_reply_gets_a_nudge():
    model = fake_model(
        text_reply("I think the degree is 2."),
        tool_call("submit_answer", {"answer": 2}),
    )
    answer, messages = run_agent("What is the degree of node 4?", GRAPH, call_model=model)
    assert answer == 2
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant"]


def test_crashing_tool_does_not_crash_agent(monkeypatch):
    # Replace run_tool with one that always crashes, like the EdgeView bug did.
    def broken_tool(graph, name, args):
        raise TypeError("Object of type EdgeView is not JSON serializable")

    monkeypatch.setattr("harness.hello_agent.run_tool", broken_tool)
    model = fake_model(
        tool_call("count_edges", {}),
        tool_call("submit_answer", {"answer": 5}),
    )
    answer, messages = run_agent("How many edges does G have?", GRAPH, call_model=model)

    assert answer == 5  # the agent kept going after the crash
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert "EdgeView" in json.loads(tool_msg["content"])["error"]  # and the model saw the error
