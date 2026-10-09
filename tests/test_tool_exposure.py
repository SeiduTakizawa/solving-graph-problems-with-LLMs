"""M4 wiring: tool families and the more_tools escape hatch, the routed runner, the `any` answer type, and the
two count verifiers with evidence (components, triangles)."""
import argparse
import json
from pathlib import Path

import networkx as nx
import pytest

from eval.runner import make_route, paraphrased, tool_exposure
from eval.tasks import EXPECTED_CALLS, load_tasks
from harness.answers import check_answer
from harness.loop import AgentConfig, run_agent
from harness.tasks import TASKS
from harness.tools.graph_tools import DEFAULT_TOOLS, TOOL_CATEGORIES, TOOLS, run_tool
from harness.verifiers import verify_components_count, verify_triangle_count

GRAPH_FILES = sorted((Path(__file__).parents[1] / "data" / "graphs" / "er").glob("*/*.txt"))


@pytest.fixture(scope="module")
def graphs():
    return [nx.read_adjlist(path, nodetype=int) for path in GRAPH_FILES]


# --- Families and categories ---

def test_every_tool_is_in_exactly_one_category():
    listed = [t for names in TOOL_CATEGORIES.values() for t in names]
    assert sorted(listed) == sorted(TOOLS)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_family_has_the_tools_its_expected_calls_name(task):
    family = TASKS[task].tools
    assert family and set(family) <= set(DEFAULT_TOOLS)  # opt-in tools stay out of families
    for slot in EXPECTED_CALLS[task]:
        defaults = [t for t in slot if t in DEFAULT_TOOLS]
        assert any(t in family for t in defaults), f"{task}: none of {defaults} in its family"


# --- more_tools in the loop ---

def call(name, args, call_id="c"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def scripted(*replies):
    seen = []
    script = iter(replies)

    def model(messages, tools):
        seen.append([t["function"]["name"] for t in tools])
        return next(script)
    return model, seen


def test_more_tools_adds_a_category():
    model, seen = scripted(call("more_tools", {"category": "paths"}), call("shortest_path", {"source": 0, "target": 3}),
                           call("submit_answer", {"answer": [0, 1, 2, 3]}))
    result = run_agent("?", nx.path_graph(4), answer_type="node_list", call_model=model,
                       config=AgentConfig(graph_tools=("degree",), more_tools=True))
    assert seen[0] == ["degree", "more_tools", "submit_answer", "cannot_answer"]
    assert seen[1][:5] == ["degree", "shortest_path", "has_path", "distances_from", "neighborhood"]
    assert json.loads(result.messages[3]["content"])["added"] == ["shortest_path", "has_path", "distances_from",
                                                                  "neighborhood"]
    assert result.status == "submitted"


def test_more_tools_lists_only_what_is_left_and_disappears_when_used_up():
    model, seen = scripted(call("more_tools", {"category": "basics"}), call("cannot_answer", {"reason": "x"}))
    run_agent("?", nx.path_graph(4), call_model=model,
              config=AgentConfig(graph_tools=("shortest_path", "has_path", "distances_from", "neighborhood",
                                              "connected_components", "has_cycle", "minimum_spanning_tree",
                                              "is_bipartite", "topological_sort"), more_tools=True))
    assert "more_tools" in seen[0] and "more_tools" not in seen[1]  # basics was the only category left


def test_more_tools_with_a_bad_category_is_an_error():
    model, _ = scripted(call("more_tools", {"category": "magic"}), call("cannot_answer", {"reason": "x"}))
    result = run_agent("?", nx.path_graph(4), call_model=model, config=AgentConfig(graph_tools=(), more_tools=True))
    assert "Unknown or used-up category" in json.loads(result.messages[3]["content"])["error"]


def test_more_tools_is_off_by_default():
    model, seen = scripted(call("cannot_answer", {"reason": "x"}))
    run_agent("?", nx.path_graph(4), call_model=model)
    assert "more_tools" not in seen[0]


# --- The runner's routing and tool exposure ---

def runner_args(**overrides):
    defaults = dict(router="oracle", router_model=None, model="m", tools="all", code_only=False, without_tool=[])
    return argparse.Namespace(**{**defaults, **overrides})


def test_tool_exposure_modes():
    task = TASKS["connectivity"]
    assert tool_exposure(runner_args(tools="all"), task) == {}
    assert tool_exposure(runner_args(tools="task"), task) == {"graph_tools": task.tools, "more_tools": False}
    assert tool_exposure(runner_args(tools="hybrid"), task) == {"graph_tools": task.tools, "more_tools": True}
    assert tool_exposure(runner_args(tools="agent"), task) == {"graph_tools": (), "more_tools": True}
    assert tool_exposure(runner_args(tools="hybrid"), None) == {}  # unknown route: all tools


def test_oracle_and_regex_routes():
    item = {"task": "node_degree", "params": {"node": 4}, "question": "What is the degree of node 4?"}
    oracle = make_route(runner_args(), item)
    assert (oracle.task, oracle.params, oracle.router) == ("node_degree", {"node": 4}, "oracle")
    assert make_route(runner_args(router="regex"), item).task == "node_degree"


def test_paraphrased_questions_keep_the_templates_instructions():
    items = paraphrased(load_tasks("hop_max_degree", "small", "dev", 2), "dev")
    assert items[0]["question"] != items[0]["template_question"]
    assert items[0]["question"].endswith("If several are tied, answer with the smallest node id.")
    assert "{" not in items[1]["question"]


# --- The `any` answer type ---

@pytest.mark.parametrize("answer, ok", [(3, True), (True, True), ([1, 2], True), ([[0, 1], [1, 2]], True),
                                        ("three", False), ([[0, 1, 2]], False), (None, False)])
def test_any_answer_type(answer, ok):
    assert (check_answer(answer, "any") is None) == ok


# --- Count verifiers with evidence ---

def test_components_from_the_tool_pass_on_the_dataset(graphs):
    for graph in graphs:
        result = run_tool(graph, "connected_components", {})
        assert verify_components_count(graph, {}, result["count"], components=result["components"]) is None


def test_a_split_component_is_caught():
    g = nx.Graph([(0, 1), (1, 2), (3, 4)])
    assert "same component" in verify_components_count(g, {}, 3, components=[[0, 1], [2], [3, 4]])


@pytest.mark.parametrize("answer, components, message", [
    (2, None, "Also send"),
    (3, [[0, 1, 2], [3, 4]], "listed 2"),
    (2, [[0, 1, 2], [3]], "node 4 is missing"),
    (2, [[0, 1, 2], [3, 4, 0]], "more than one component"),
    (2, [[0, 1, 2], [3, 4, 9]], "not in G"),
])
def test_bad_component_evidence(answer, components, message):
    g = nx.Graph([(0, 1), (1, 2), (3, 4)])
    assert message in verify_components_count(g, {}, answer, components=components)


def test_merged_components_are_not_caught_by_design():
    g = nx.Graph([(0, 1), (1, 2), (3, 4)])
    assert verify_components_count(g, {}, 1, components=[[0, 1, 2, 3, 4]]) is None  # an undercount: unchecked


def test_triangles_from_the_tool_pass_on_the_dataset(graphs):
    for graph in graphs[:100]:
        for node in list(graph.nodes)[:5]:
            result = run_tool(graph, "triangles", {"node": node})
            assert verify_triangle_count(graph, {"node": node}, result["count"], triangles=result["triangles"]) is None


@pytest.mark.parametrize("answer, triangles, message", [
    (1, None, "Also send"),
    (2, [[0, 1, 2]], "listed 1"),
    (1, [[0, 1, 3]], "not an edge"),
    (1, [[1, 2, 3]], "does not include node 0"),
    (2, [[0, 1, 2], [2, 1, 0]], "listed twice"),
    (1, [[0, 1]], "not three different nodes"),
])
def test_bad_triangle_evidence(answer, triangles, message):
    g = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)])
    assert message in verify_triangle_count(g, {"node": 0}, answer, triangles=triangles)


def test_zero_triangles_need_no_list():
    assert verify_triangle_count(nx.path_graph(3), {"node": 1}, 0) is None
