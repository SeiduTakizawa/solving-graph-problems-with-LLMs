"""Tests for the agent loop, using a fake model that replies from a script."""
import json

import networkx as nx

from harness.answers import check_answer
from harness.loop import AgentConfig, run_agent
from harness.prompts import FORMAT_ERROR, NUDGE
from harness.tools.graph_tools import TOOLS
from harness.trace import Trace, read_trace
from harness.verifiers import verify_connectivity, verify_shortest_path

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
    result = run_agent(QUESTION, GRAPH, call_model=wandering, config=AgentConfig(max_steps=3))
    assert result.answer is None
    assert result.status == "max_steps"


def test_crashing_tool_does_not_crash_agent(monkeypatch):
    # Replace run_tool with one that always crashes, like the EdgeView bug did.
    def broken_tool(graph, name, args):
        raise TypeError("Object of type EdgeView is not JSON serializable")

    monkeypatch.setattr("harness.loop.run_tool", broken_tool)
    model = fake_model(
        tool_call("graph_info", {}),
        tool_call("submit_answer", {"answer": 5}),
    )
    result = run_agent("How many edges does G have?", GRAPH, call_model=model)

    assert result.answer == 5  # the agent kept going after the crash
    tool_msg = next(m for m in result.messages if m["role"] == "tool")
    assert "EdgeView" in json.loads(tool_msg["content"])["error"]  # and the model saw the error


def raw_tool_call(name, raw_arguments, call_id="call_1"):
    """Like tool_call, but with the arguments string exactly as the model wrote it (possibly broken)."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw_arguments}}]}


def test_broken_json_arguments_go_back_to_the_model():
    model = fake_model(
        raw_tool_call("get_neighbors", '{"node": 4', call_id="call_A"),  # unfinished JSON
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)

    assert result.status == "submitted" and result.answer == 2  # the run survived
    error = json.loads(result.messages[3]["content"])["error"]
    assert result.messages[3]["tool_call_id"] == "call_A" and "not valid JSON" in error


def test_arguments_that_are_not_an_object_go_back_to_the_model():
    model = fake_model(raw_tool_call("get_neighbors", "[4]"), tool_call("submit_answer", {"answer": 2}))
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert "JSON object" in json.loads(result.messages[3]["content"])["error"]


def test_broken_json_in_submit_answer_is_not_a_crash():
    model = fake_model(raw_tool_call("submit_answer", '{"answer": '), tool_call("submit_answer", {"answer": 2}))
    assert run_agent(QUESTION, GRAPH, call_model=model).answer == 2


def test_empty_arguments_mean_no_arguments():
    model = fake_model(raw_tool_call("graph_info", ""), tool_call("submit_answer", {"answer": 5}))
    result = run_agent("How many edges does G have?", GRAPH, call_model=model)
    assert json.loads(result.messages[3]["content"]) == {"nodes": 7, "edges": 5, "directed": False}


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
    assert check_answer(True, "yes_no_with_path") is None
    assert check_answer(1, "yes_no_with_cycle") is not None
    assert check_answer([[0, 1], [1, 2]], "edge_list") is None
    assert check_answer([[0, 1, 2]], "edge_list") is not None  # not a pair
    assert check_answer(None, "number") is not None  # submit_answer without an answer


# --- Verifier: a wrong final answer goes back to the model ---

PATH_QUESTION = "What is the shortest path from node 0 to node 5?"
check_path_0_to_5 = lambda answer: verify_shortest_path(GRAPH, {"source": 0, "target": 5}, answer)


def test_verifier_rejects_then_accepts():
    model = fake_model(
        tool_call("submit_answer", {"answer": [0, 5]}, call_id="call_A"),        # 0-5 is not an edge
        tool_call("submit_answer", {"answer": [0, 2, 4, 5]}, call_id="call_B"),  # the real shortest path
    )
    result = run_agent(PATH_QUESTION, GRAPH, answer_type="node_list", call_model=model, verify=check_path_0_to_5)

    assert result.status == "submitted" and result.answer == [0, 2, 4, 5]
    assert result.rejected == 1
    feedback = result.messages[3]
    assert feedback["tool_call_id"] == "call_A"
    assert json.loads(feedback["content"]) == {"error": "0-5 is not an edge in G."}


def test_valid_answer_passes_the_verifier():
    model = fake_model(tool_call("submit_answer", {"answer": [0, 2, 4, 5]}))
    result = run_agent(PATH_QUESTION, GRAPH, answer_type="node_list", call_model=model, verify=check_path_0_to_5)
    assert result.answer == [0, 2, 4, 5] and result.rejected == 0


def test_verifier_only_sees_answers_of_the_right_type():
    seen = []

    def verify(answer):
        seen.append(answer)
        return None

    model = fake_model(
        tool_call("submit_answer", {"answer": 5}),          # wrong type: stopped by check_answer first
        tool_call("submit_answer", {"answer": [0, 2, 4, 5]}),
    )
    run_agent(PATH_QUESTION, GRAPH, answer_type="node_list", call_model=model, verify=verify)
    assert seen == [[0, 2, 4, 5]]


def test_verifier_rejection_is_logged(tmp_path):
    trace = Trace(tmp_path / "run.jsonl")
    model = fake_model(
        tool_call("submit_answer", {"answer": [0, 5]}),
        tool_call("submit_answer", {"answer": [0, 2, 4, 5]}),
    )
    run_agent(PATH_QUESTION, GRAPH, answer_type="node_list", call_model=model, verify=check_path_0_to_5,
              trace=trace)

    events = read_trace(trace.path)
    rejected = [e for e in events if e["event"] == "verifier_rejected"]
    assert len(rejected) == 1 and rejected[0]["answer"] == [0, 5]
    assert events[-1]["rejected"] == 1 and events[0]["verified"] is True


# --- Evidence: a "yes" can carry the path or cycle behind it, for the verifier ---

def test_submit_answer_offers_the_evidence_field():
    seen = {}

    def model(messages, tools):
        submit = next(t for t in tools if t["function"]["name"] == "submit_answer")
        seen["params"] = submit["function"]["parameters"]
        return tool_call("submit_answer", {"answer": False})

    run_agent("Is there a path between nodes 0 and 5?", GRAPH, answer_type="yes_no_with_path", call_model=model)
    assert seen["params"]["properties"]["answer"] == {"type": "boolean"}
    assert "path" in seen["params"]["properties"]
    assert seen["params"]["required"] == ["answer"]  # a "no" needs no evidence


def test_evidence_is_passed_to_the_verifier_and_kept():
    seen = []

    def verify(answer, path=None):
        seen.append((answer, path))
        return None

    model = fake_model(tool_call("submit_answer", {"answer": True, "path": [0, 2, 4, 5]}))
    result = run_agent("Is there a path between nodes 0 and 5?", GRAPH, answer_type="yes_no_with_path",
                       call_model=model, verify=verify)
    assert seen == [(True, [0, 2, 4, 5])]
    assert result.answer is True and result.evidence == {"path": [0, 2, 4, 5]}


def test_no_evidence_for_plain_answer_types():
    seen = []
    model = fake_model(tool_call("submit_answer", {"answer": [0, 2, 4, 5], "path": [9]}))  # extra field ignored
    result = run_agent(PATH_QUESTION, GRAPH, answer_type="node_list", call_model=model,
                       verify=lambda answer: seen.append(answer))
    assert seen == [[0, 2, 4, 5]] and result.evidence is None


def test_yes_without_evidence_is_sent_back():
    check = lambda answer, path=None: verify_connectivity(GRAPH, {"source": 0, "target": 5}, answer, path)
    model = fake_model(
        tool_call("submit_answer", {"answer": True}, call_id="call_A"),
        tool_call("submit_answer", {"answer": True, "path": [0, 2, 4, 5]}, call_id="call_B"),
    )
    result = run_agent("Is there a path between nodes 0 and 5?", GRAPH, answer_type="yes_no_with_path",
                       call_model=model, verify=check)
    assert result.status == "submitted" and result.rejected == 1
    assert "path" in json.loads(result.messages[3]["content"])["error"]


# --- Offering a subset of tools (ablations) ---

def test_only_offered_tools_are_shown():
    seen = {}

    def model(messages, tools):
        seen["names"] = {t["function"]["name"] for t in tools}
        return tool_call("submit_answer", {"answer": 2})

    run_agent(QUESTION, GRAPH, call_model=model, config=AgentConfig(graph_tools=("get_neighbors",)))
    assert seen["names"] == {"get_neighbors", "submit_answer", "cannot_answer"}


def test_tool_not_offered_is_refused():
    without_has_edge = tuple(name for name in TOOLS if name != "has_edge")
    model = fake_model(
        tool_call("has_edge", {"u": 4, "v": 5}),  # the model calls a tool it was not given
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model, config=AgentConfig(graph_tools=without_has_edge))
    assert "Unknown tool has_edge" in json.loads(result.messages[3]["content"])["error"]


# --- Thinking is logged, not re-sent ---

def test_thinking_is_not_sent_back_but_is_logged(tmp_path):
    trace = Trace(tmp_path / "run.jsonl")
    model = fake_model(
        {**tool_call("get_neighbors", {"node": 4}), "reasoning_content": "Let me look at node 4..."},
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model, trace=trace)

    assert all("reasoning_content" not in m for m in result.messages)
    first_call = next(e for e in read_trace(trace.path) if e["event"] == "model_call")
    assert first_call["reasoning"] == "Let me look at node 4..."


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


def test_strict_mode_gives_a_format_error():
    # Exactly what qwen3 wrote in the node-10990 run; with rescue off it is not executed.
    model = fake_model(
        text_reply('submit_answer\n{"answer": 0}\n</tool_call>'),
        tool_call("submit_answer", {"answer": 0}),
    )
    result = run_agent("Get neighbors of node 10990", GRAPH, call_model=model, config=AgentConfig(rescue=False))
    assert result.messages[3]["content"] == FORMAT_ERROR
    assert result.rescued == 0


def test_unparseable_text_tool_call_gets_a_format_error():
    # Looks like a tool call but the JSON is broken, so even lenient mode can't run it.
    model = fake_model(
        text_reply('submit_answer\n{"answer": }\n</tool_call>'),
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert result.messages[3]["content"] == FORMAT_ERROR
    assert result.rescued == 0


def test_prose_mentioning_tools_is_not_a_format_error():
    # From the clique run: it names the tools but is not trying to call them.
    model = fake_model(
        text_reply("The tools (`get_neighbors` and `count_edges`) cannot find cliques."),
        tool_call("cannot_answer", {"reason": "no clique tool"}),
    )
    result = run_agent("Find the biggest clique", GRAPH, call_model=model)
    assert result.messages[3]["content"] == NUDGE


# --- Lenient mode: tool calls written as text are rescued (graph 14, node 10990, neighbors runs) ---

def test_rescues_submit_written_as_text():
    # The neighbors-of-node-4 run: correct answer, written as text.
    model = fake_model(
        tool_call("get_neighbors", {"node": 4}),
        text_reply('The neighbors of node 4 are nodes 2 and 5.\n\nsubmit_answer\n{"answer": [2, 5]}\n</tool_call>'),
    )
    result = run_agent("Which nodes are the neighbors of node 4?", GRAPH, answer_type="node_list", call_model=model)
    assert result.status == "submitted" and result.answer == [2, 5]
    assert result.rescued == 1


def test_rescues_graph_tool_in_qwen_format():
    model = fake_model(
        text_reply('<tool_call>\n{"name": "get_neighbors", "arguments": {"node": 4}}\n</tool_call>'),
        tool_call("submit_answer", {"answer": 2}),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)

    assert result.answer == 2 and result.rescued == 1
    # The history stays valid: the rescued call gets an id, and the tool result answers that id.
    call = result.messages[2]["tool_calls"][0]
    tool_msg = result.messages[3]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == call["id"]
    assert json.loads(tool_msg["content"]) == {"neighbors": [2, 5]}


def test_rescued_answer_is_still_type_checked():
    model = fake_model(
        text_reply('submit_answer\n{"answer": 0}\n</tool_call>'),  # 0 for a yes/no question
        tool_call("submit_answer", {"answer": False}),
    )
    result = run_agent("Is there a cycle in G?", GRAPH, answer_type="yes_no", call_model=model)
    assert result.answer is False and result.rescued == 1
    assert "true or false" in json.loads(result.messages[3]["content"])["error"]


def test_rescue_is_logged(tmp_path):
    trace = Trace(tmp_path / "run.jsonl")
    model = fake_model(text_reply('submit_answer\n{"answer": 2}\n</tool_call>'))
    run_agent(QUESTION, GRAPH, call_model=model, trace=trace)

    events = read_trace(trace.path)
    assert [e["event"] for e in events] == ["run_start", "model_call", "rescued_tool_call", "run_end"]
    assert events[2]["name"] == "submit_answer" and events[2]["args"] == {"answer": 2}
    assert events[-1]["rescued"] == 1 and events[-1]["status"] == "submitted"


# --- Loop detection (the node-10990 run repeated itself 9 times) ---

def test_repeated_text_reply_stops_the_run():
    stuck = lambda messages, tools: text_reply('submit_answer\n{"answer": 0}\n</tool_call>')
    result = run_agent("Get neighbors of node 10990", GRAPH, call_model=stuck, config=AgentConfig(max_steps=10, rescue=False))
    assert result.status == "loop_detected"
    assert roles(result).count("assistant") == 3  # stopped early, not after 10


def test_repeated_tool_call_stops_the_run():
    # Same call each time, with a fresh random id like a real model would send.
    ids = iter(range(100))
    stuck = lambda messages, tools: tool_call("get_neighbors", {"node": 4}, call_id=f"call_{next(ids)}")
    result = run_agent(QUESTION, GRAPH, call_model=stuck, config=AgentConfig(max_steps=10))
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


def test_result_carries_steps_and_token_totals():
    # The runner reads these from the result instead of searching the trace file.
    model = fake_model(
        with_stats(tool_call("get_neighbors", {"node": 4}), prompt_tokens=100, completion_tokens=20, latency_s=0.5),
        with_stats(tool_call("submit_answer", {"answer": 2}), prompt_tokens=150, completion_tokens=10, latency_s=0.25),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert (result.steps, result.prompt_tokens, result.completion_tokens) == (2, 250, 30)
    assert result.model_latency_s == 0.75


def test_stats_are_not_sent_to_the_model():
    model = fake_model(
        with_stats(tool_call("get_neighbors", {"node": 4})),
        with_stats(tool_call("submit_answer", {"answer": 2})),
    )
    result = run_agent(QUESTION, GRAPH, call_model=model)
    assert all("extra" not in m for m in result.messages)


def test_configured_model_is_used_and_logged(monkeypatch, tmp_path):
    used = []

    def fake_call_model(messages, tools, model):
        used.append(model)
        return tool_call("submit_answer", {"answer": 2})

    monkeypatch.setattr("harness.models.call_model", fake_call_model)
    trace = Trace(tmp_path / "run.jsonl")
    run_agent(QUESTION, GRAPH, config=AgentConfig(model="openai/some-cheap-model"), trace=trace)

    assert used == ["openai/some-cheap-model"]
    assert read_trace(trace.path)[0]["model"] == "openai/some-cheap-model"


def test_prompt_version_is_logged(tmp_path):
    from harness.prompts import PROMPT_VERSION

    trace = Trace(tmp_path / "run.jsonl")
    run_agent(QUESTION, GRAPH, call_model=fake_model(tool_call("submit_answer", {"answer": 2})), trace=trace)
    assert read_trace(trace.path)[0]["prompt_version"] == PROMPT_VERSION
