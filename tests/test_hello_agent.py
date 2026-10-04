"""Tests for the hello agent loop, using a fake model that replies from a script."""
import json

import networkx as nx

from harness.hello_agent import FORMAT_ERROR, NUDGE, check_answer, run_agent
from harness.trace import Trace, read_trace

# Same graph as data/graphs/er/small/0.txt; node 4 has neighbors [2, 5].
GRAPH = nx.Graph([(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)])
QUESTION = "What is the degree of node 4?"


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
    return lambda messages, tools: next(script)


def with_stats(reply, prompt_tokens=100, completion_tokens=20, latency_s=0.5):
    """Attach the bookkeeping that the real call_model adds."""
    return {**reply, "extra": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                               "latency_s": latency_s}}


def roles(result):
    return [m["role"] for m in result.messages]


# --- The basic loop ---

def test_submits_answer():
    result = run_agent(QUESTION, GRAPH, call_model=fake_model(tool_call("submit_answer", {"answer": 2})))
    assert result.answer == 2
    assert result.status == "submitted"


def test_tool_result_goes_back_to_model():
    model = fake_model(
        tool_call("get_neighbors", {"node": 4}, call_id="call_A"),
        tool_call("submit_answer", {"answer": 2}, call_id="call_B"),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)

    assert result.answer == 2
    tool_msg = next(m for m in result.messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_A"
    assert json.loads(tool_msg["content"]) == {"neighbors": [2, 5]}


def test_message_history_order():
    model = fake_model(
        tool_call("get_neighbors", {"node": 4}),
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert roles(result) == ["system", "user", "assistant", "tool", "assistant"]


def test_stops_after_max_steps():
    # A model that asks about a different node each time (so it is not a repeat loop).
    nodes = iter(range(100))
    wandering = lambda messages, tools: tool_call("get_neighbors", {"node": next(nodes) % 7})
    result = run_agent(QUESTION, GRAPH, call_model=wandering, max_steps=3)
    assert result.answer is None
    assert result.status == "max_steps"


def test_crashing_tool_does_not_crash_agent(monkeypatch):
    # Replace run_tool with one that always crashes, like the EdgeView bug did.
    def broken_tool(graph, name, args):
        raise TypeError("Object of type EdgeView is not JSON serializable")

    monkeypatch.setattr("harness.hello_agent.run_tool", broken_tool)
    model = fake_model(
        tool_call("graph_info", {}),
        tool_call("submit_answer", {"answer": 5}),
    )
    result = run_agent("How many edges does G have?", GRAPH, call_model=model)

    assert result.answer == 5  # the agent kept going after the crash
    tool_msg = next(m for m in result.messages if m["role"] == "tool")
    assert "EdgeView" in json.loads(tool_msg["content"])["error"]  # and the model saw the error


# --- Answer types (the cycle question was answered with 0 for "no") ---

def test_yes_no_answer():
    model = fake_model(tool_call("submit_answer", {"answer": False}))
    result = run_agent("Is there a cycle in G?", GRAPH, answer_type="yes_no", call_model=model)
    assert result.status == "submitted" and result.answer is False


def test_node_list_answer():
    model = fake_model(tool_call("submit_answer", {"answer": [2, 5]}))
    result = run_agent("Which nodes are the neighbors of node 4?", GRAPH, answer_type="node_list", call_model=model)
    assert result.answer == [2, 5]


def test_wrong_answer_type_is_sent_back_to_the_model():
    model = fake_model(
        tool_call("submit_answer", {"answer": 0}, call_id="call_A"),      # 0 instead of false
        tool_call("submit_answer", {"answer": False}, call_id="call_B"),
    )
    result = run_agent("Is there a cycle in G?", GRAPH, answer_type="yes_no", call_model=model)

    assert result.status == "submitted" and result.answer is False  # second try accepted
    error = json.loads(result.messages[3]["content"])["error"]
    assert result.messages[3]["tool_call_id"] == "call_A"
    assert "true or false" in error


def test_model_is_offered_the_right_answer_type():
    seen = {}

    def model(messages, tools):
        submit = next(t for t in tools if t["function"]["name"] == "submit_answer")
        seen["schema"] = submit["function"]["parameters"]["properties"]["answer"]
        return tool_call("submit_answer", {"answer": True})

    run_agent("Is there a cycle in G?", GRAPH, answer_type="yes_no", call_model=model)
    assert seen["schema"] == {"type": "boolean"}


def test_check_answer():
    assert check_answer(3, "number") is None
    assert check_answer(True, "number") is not None   # in Python True is an int; not a number answer
    assert check_answer("3", "number") is not None
    assert check_answer(False, "yes_no") is None
    assert check_answer(0, "yes_no") is not None
    assert check_answer([1, 2], "node_list") is None
    assert check_answer([], "node_list") is None
    assert check_answer(2, "node_list") is not None
    assert check_answer([1, True], "node_list") is not None


# --- Giving up honestly (the node-99 and clique runs) ---

def test_cannot_answer_ends_the_run():
    model = fake_model(tool_call("cannot_answer", {"reason": "No tool can find cliques."}))
    result = run_agent("Find the biggest clique", GRAPH, call_model=model)

    assert result.status == "cannot_answer"
    assert result.answer is None
    assert result.reason == "No tool can find cliques."


# --- Plain-text replies (the clique and node-10990 runs) ---

def test_text_reply_gets_a_nudge():
    model = fake_model(
        text_reply("I think the degree is 2."),
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert result.answer == 2
    assert roles(result) == ["system", "user", "assistant", "user", "assistant"]
    assert result.messages[3]["content"] == NUDGE
    assert "cannot_answer" in NUDGE  # no pressure to guess: giving up is offered


def test_tool_call_written_as_text_gets_a_format_error():
    # Exactly what qwen3 wrote in the node-10990 run.
    model = fake_model(
        text_reply('submit_answer\n{"answer": 0}\n</tool_call>'),
        tool_call("submit_answer", {"answer": 0}),
    )
    result = run_agent("Get neighbors of node 10990", GRAPH, call_model=model)
    assert result.messages[3]["content"] == FORMAT_ERROR


def test_prose_mentioning_tools_is_not_a_format_error():
    # From the clique run: it names the tools but is not trying to call them.
    model = fake_model(
        text_reply("The tools (`get_neighbors` and `count_edges`) cannot find cliques."),
        tool_call("cannot_answer", {"reason": "no clique tool"}),
    )
    result = run_agent("Find the biggest clique", GRAPH, call_model=model)
    assert result.messages[3]["content"] == NUDGE


# --- Loop detection (the node-10990 run repeated itself 9 times) ---

def test_repeated_text_reply_stops_the_run():
    stuck = lambda messages, tools: text_reply('submit_answer\n{"answer": 0}\n</tool_call>')
    result = run_agent("Get neighbors of node 10990", GRAPH, call_model=stuck, max_steps=10)
    assert result.status == "loop_detected"
    assert roles(result).count("assistant") == 3  # stopped early, not after 10


def test_repeated_tool_call_stops_the_run():
    # Same call each time, with a fresh random id like a real model would send.
    ids = iter(range(100))
    stuck = lambda messages, tools: tool_call("get_neighbors", {"node": 4}, call_id=f"call_{next(ids)}")
    result = run_agent(QUESTION, GRAPH, call_model=stuck, max_steps=10)
    assert result.status == "loop_detected"


def test_asking_twice_is_not_a_loop():
    model = fake_model(
        tool_call("get_neighbors", {"node": 4}),
        tool_call("get_neighbors", {"node": 4}),
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert result.status == "submitted"


# --- Logging ---

def test_trace_logs_the_run(tmp_path):
    trace = Trace(tmp_path / "run.jsonl")
    model = fake_model(
        with_stats(tool_call("get_neighbors", {"node": 4})),
        with_stats(tool_call("submit_answer", {"answer": 2})),
    )
    run_agent(QUESTION, GRAPH, call_model=model, trace=trace)

    events = read_trace(trace.path)
    assert [e["event"] for e in events] == ["run_start", "model_call", "tool_call", "model_call", "run_end"]
    assert {e["run_id"] for e in events} == {trace.run_id}
    assert events[2]["result"] == {"neighbors": [2, 5]}

    end = events[-1]
    assert end["answer"] == 2 and end["status"] == "submitted" and end["steps"] == 2
    assert end["prompt_tokens"] == 200 and end["completion_tokens"] == 40  # summed over both calls


def test_trace_logs_how_the_run_ended(tmp_path):
    trace = Trace(tmp_path / "run.jsonl")
    model = fake_model(text_reply("hmm"), tool_call("cannot_answer", {"reason": "no clique tool"}))
    run_agent("Find the biggest clique", GRAPH, call_model=model, trace=trace)

    events = read_trace(trace.path)
    assert [e["event"] for e in events] == ["run_start", "model_call", "nudge", "model_call", "run_end"]
    assert events[2]["kind"] == "no_tool_call"
    assert events[-1]["status"] == "cannot_answer" and events[-1]["reason"] == "no clique tool"


def test_stats_are_not_sent_to_the_model():
    model = fake_model(
        with_stats(tool_call("get_neighbors", {"node": 4})),
        with_stats(tool_call("submit_answer", {"answer": 2})),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert all("extra" not in m for m in result.messages)
