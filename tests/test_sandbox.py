"""run_python sandbox (Docker). Skipped where Docker isn't available."""
import json

import networkx as nx
import pytest

from harness.loop import AgentConfig, run_agent
from harness.sandbox import STDOUT_SHOWN, shown_reply
from harness.sandbox.docker import DockerSandbox, docker_binary

pytestmark = pytest.mark.skipif(docker_binary() is None, reason="Docker not available")

SMALL = nx.Graph([(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)])  # data/graphs/er/small/0.txt


@pytest.fixture(scope="module")
def sandbox():
    with DockerSandbox(SMALL, mode="tools") as sb:
        yield sb


def test_tools_return_plain_values_in_code(sandbox):
    # Like networkx, not like the JSON tools: see code_value in harness/sandbox/runner.py.
    assert sandbox.run("result = get_neighbors(4)")["result"] == [2, 5]
    assert sandbox.run("result = degree(node=4)")["result"] == 2  # keyword arguments work too
    assert sandbox.run("result = shortest_path(0, 3)")["result"] == [0, 2, 4, 5, 3]
    assert sandbox.run("result = shortest_path(0, 1)")["result"] is None  # different components


def test_a_tool_error_stops_the_code(sandbox):
    error = sandbox.run("x = 1\nn = get_neighbors(99)\nresult = n")["error"]
    assert error.startswith("line 2: ValueError: Node 99 does not exist")


def test_variables_persist_and_stdout_is_captured(sandbox):
    sandbox.run("total = sum(degree(n) for n in range(7))")
    reply = sandbox.run("print('edges:', total // 2)\nresult = total")
    assert reply["stdout"] == "edges: 5\n" and reply["result"] == 10


def test_a_composed_answer_in_one_call(sandbox):
    # The kind of thing code is for: many tool calls and a comparison in one action.
    code = "result = max(get_neighbors(5), key=degree)"
    assert sandbox.run(code)["result"] in (3, 4)


def test_a_redefined_tool_is_restored(sandbox):
    # From combine_qwen35_C: this recursion broke get_neighbors for every later call of the question.
    reply = sandbox.run("def get_neighbors(n): return get_neighbors(n)\nresult = get_neighbors(4)")
    assert "RecursionError" in reply["error"]
    assert "redefined get_neighbors" in reply["note"]
    after = sandbox.run("result = get_neighbors(4)")
    assert after["result"] == [2, 5] and after["note"] is None


def test_own_helpers_are_not_flagged(sandbox):
    assert sandbox.run("def deg2(n): return 2 * degree(n)\nresult = deg2(4)")["note"] is None


def test_errors_point_at_the_models_line(sandbox):
    assert sandbox.run("x = [1]\ny = x[3]")["error"] == "line 2: IndexError: list index out of range"


@pytest.mark.parametrize("code, expected", [
    ("import networkx", "not available"),                                                 # tools mode: no bypass
    ("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)", "unreachable"),  # no network
    ("open('/opt/harness/x.py', 'w')", "Read-only"),                                       # no writing the code
])
def test_lockdown(sandbox, code, expected):
    assert expected in sandbox.run(code)["error"]


def test_only_the_harness_code_is_visible(sandbox):
    # Reference answers live in eval/ and results/: they must not be reachable from the sandbox.
    reply = sandbox.run("import os\nresult = [sorted(os.listdir('/opt')), os.path.exists('/opt/eval'), "
                        "os.path.exists('/opt/harness/../results')]")
    assert reply["result"] == [["harness"], False, False]


def test_networkx_mode_has_the_graph():
    with DockerSandbox(SMALL, mode="networkx") as sb:
        assert sb.run("result = nx.number_connected_components(G)")["result"] == 2


def test_a_timeout_restarts_the_sandbox():
    with DockerSandbox(SMALL, timeout_s=2) as sb:
        sb.run("kept = 1")
        assert "longer than 2 s" in sb.run("while True: pass")["error"]
        after = sb.run("result = degree(4)")
        assert after["result"] == 2 and "restarted" in after["note"]
        assert "NameError" in sb.run("result = kept")["error"]  # earlier variables are gone


def test_too_much_memory_is_stopped():
    with DockerSandbox(SMALL, timeout_s=10) as sb:
        error = sb.run("x = bytearray(2_000_000_000)")["error"]
        assert "MemoryError" in error or "too much memory" in error


def test_long_output_is_shortened_for_the_model():
    shown = shown_reply({"stdout": "x" * 50_000, "result": None, "error": None})
    assert len(shown["stdout"]) < STDOUT_SHOWN + 200 and "put large values in `result`" in shown["stdout"]


def test_run_python_in_the_loop():
    calls = iter([
        ("run_python", {"code": "result = max(get_neighbors(5), key=degree)"}),
        ("submit_answer", {"answer": 4}),
    ])

    def model(messages, tools):
        name, args = next(calls)
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    result = run_agent("Which neighbor of node 5 has the highest degree?", SMALL, call_model=model,
                       config=AgentConfig(python="tools"))
    assert result.status == "submitted"
    assert json.loads(result.messages[3]["content"])["result"] in (3, 4)


def test_big_result_becomes_a_handle_in_the_loop():
    big = nx.gnm_random_graph(5_000, 12_000, seed=2)
    calls = iter([("run_python", {"code": "result = minimum_spanning_tree()"}),
                  ("cannot_answer", {"reason": "done"})])

    def model(messages, tools):
        name, args = next(calls)
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    result = run_agent("?", big, call_model=model, config=AgentConfig(python="tools"))
    shown = json.loads(result.messages[3]["content"])
    assert shown["result"]["handle"] == "result_1" and len(result.messages[3]["content"]) < 2_000


def test_run_python_is_off_by_default():
    seen = []
    run_agent("?", SMALL, call_model=lambda m, t: seen.append([x["function"]["name"] for x in t]) or
              {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
                  "name": "cannot_answer", "arguments": "{}"}}]})
    assert "run_python" not in seen[0]


def test_opt_in_tools_are_only_in_code_when_offered(sandbox):
    assert "not defined" in sandbox.run("result = triangles(4)")["error"]  # the default sandbox
    with DockerSandbox(SMALL, tools=("degree", "triangles")) as sb:
        assert sb.run("result = triangles(4)")["result"] == []
        assert sb.run("result = len(k_core(1))")["error"]  # only what was asked for


def test_a_code_list_can_be_submitted_by_handle():
    # v9: a run_python list of 20+ numbers gets a result_handle, and submit_answer then accepts it.
    big = nx.complete_graph(8)
    calls = iter([("run_python", {"code": "n = get_neighbors(0)\nresult = [[0, a, b] for i, a in enumerate(n) "
                                          "for b in n[i + 1:] if has_edge(a, b)]"}),
                  ("submit_answer", {"answer": 21, "triangles": "result_1"})])

    def model(messages, tools):
        name, args = next(calls)
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    from functools import partial
    from harness.verifiers import verify_triangle_count
    result = run_agent("?", big, answer_type="number_with_triangles", call_model=model,
                       verify=partial(verify_triangle_count, big, {"node": 0}), config=AgentConfig(python="tools"))
    assert json.loads(result.messages[3]["content"])["result_handle"] == "result_1"
    assert result.status == "submitted" and len(result.evidence["triangles"]) == 21
