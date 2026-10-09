"""Router (M4). check_route and the eval plumbing are done; regex_route and llm_route are TODO(you): their tests
fail until they are written (harness/router.py says what each must do)."""
import json

import networkx as nx
import pytest

from eval.routing import outcome, routing_questions, same_params
from harness.prompts import ROUTER_PROMPT, ROUTER_TASKS
from harness.router import (ROUTE_TOOL, UNKNOWN, RouteArgs, Route, check_route, llm_route, regex_route,
                            task_params)
from harness.tasks import TASKS

G = nx.path_graph(10)


def route(task, confidence=1.0, **params):
    return Route(task, params, confidence, "test")


# --- check_route (done) ---

def test_task_params_come_from_the_templates():
    assert task_params("node_count") == []
    assert task_params("node_degree") == ["node"]
    assert task_params("hop_max_degree") == ["k", "node"]  # in order of appearance, {node} only once
    assert task_params("shortest_path_via") == ["source", "target", "via"]


def test_a_good_route_passes_and_extra_params_are_dropped():
    checked = check_route(route("connectivity", source=1, target=5, node=3), G)
    assert checked.known and checked.params == {"source": 1, "target": 5}


@pytest.mark.parametrize("bad, reason", [
    (route("diameter"), "no task named"),
    (route("node_degree", confidence=0.3, node=1), "below"),
    (route("node_degree"), "needs node"),
    (route("shortest_path", source=1), "needs target"),
    (route("node_degree", node=99), "not in G"),
    (route("node_degree", node=True), "whole numbers"),
    (route("hop_max_degree", node=1, k=0), "k must be"),
])
def test_a_bad_route_becomes_unknown_with_a_reason(bad, reason):
    checked = check_route(bad, G)
    assert not checked.known and checked.params == {} and reason in checked.reason


def test_a_missing_confidence_is_accepted():
    assert check_route(route("node_count", confidence=None), G).known


def test_a_pair_in_the_other_fields_counts():
    assert check_route(route("connectivity", u=1, v=5), G).params == {"source": 1, "target": 5}
    assert check_route(route("edge_existence", source=2, target=3), G).params == {"u": 2, "v": 3}


def test_unknown_stays_unknown():
    assert not check_route(route(UNKNOWN), G).known


def test_without_a_graph_node_ids_are_not_checked():
    assert check_route(route("node_degree", node=99)).known


def test_the_route_tool_lists_every_task():
    schema = ROUTE_TOOL["function"]["parameters"]
    assert schema["properties"]["task"]["enum"] == [*TASKS, UNKNOWN]
    assert schema["required"] == ["task"]
    assert schema["properties"]["confidence"]["type"] == "number"


def test_the_router_prompt_describes_every_task():
    assert set(ROUTER_TASKS) == set(TASKS) and all(f"- {t}:" in ROUTER_PROMPT for t in TASKS)


# --- eval plumbing (done) ---

def test_routing_questions_fill_in_real_params():
    items = routing_questions("dev")
    assert {i["task"] for i in items} == {*TASKS, UNKNOWN}
    for item in items:
        assert "{" not in item["question"]
        assert set(item["params"]) == set(task_params(item["task"]) if item["task"] != UNKNOWN else [])


def test_dev_and_test_phrasings_differ():
    dev = {i["question"] for i in routing_questions("dev")}
    assert not dev & {i["question"] for i in routing_questions("test")}


def test_symmetric_params():
    assert same_params("edge_existence", {"u": 1, "v": 2}, {"u": 2, "v": 1})
    assert not same_params("shortest_path", {"source": 1, "target": 2}, {"source": 2, "target": 1})


def test_outcomes():
    item = {"task": "node_degree", "params": {"node": 3}}
    assert outcome(item, route("node_degree", node=3)) == "right"
    assert outcome(item, route("node_degree", node=4)) == "wrong"
    assert outcome(item, route(UNKNOWN)) == "safe_miss"
    assert outcome({"task": UNKNOWN, "params": {}}, route("node_count")) == "false_known"


# --- regex_route (TODO(you)) ---

@pytest.mark.parametrize("task", sorted(TASKS))
def test_regex_routes_the_dataset_templates(task):
    # The floor: the harness's own wording must route right, with every parameter.
    params = {name: 10 + i for i, name in enumerate(task_params(task))}
    if "k" in params:
        params["k"] = 2
    got = regex_route(TASKS[task].make_question(params))
    assert (got.task, got.params, got.router, got.confidence) == (task, params, "regex", 1.0)


@pytest.mark.parametrize("question, task", [
    ("How many neighbors does node 4 have?", "node_degree"),      # "how many" + neighbors: a number
    ("Which nodes are the neighbours of node 4?", "connected_nodes"),
    ("What is the shortest path from 1 to 7?", "shortest_path"),  # not connectivity
    ("Is there a path between nodes 1 and 7", "connectivity"),     # no question mark
])
def test_regex_handles_small_variations(question, task):
    assert regex_route(question).task == task


def test_regex_says_unknown_when_nothing_matches():
    got = regex_route("What is the diameter of G?")
    assert got.task == UNKNOWN and got.confidence == 0.0 and got.router == "regex"


# --- llm_route (TODO(you)), with a fake model ---

def fake_model(reply_args=None, text=None, prompt_tokens=400, completion_tokens=50):
    seen = {}

    def call(messages, tools, model=None, think=None):
        seen.update(messages=messages, tools=tools, think=think)
        calls = None if reply_args is None else [{"id": "c", "type": "function", "function": {
            "name": "route", "arguments": reply_args if isinstance(reply_args, str) else json.dumps(reply_args)}}]
        return {"role": "assistant", "content": text or "", "tool_calls": calls,
                "extra": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "latency_s": 0.9}}
    return call, seen


def test_llm_route_reads_the_tool_call():
    call, seen = fake_model({"task": "connectivity", "source": 3, "target": 17, "confidence": 0.95})
    got = llm_route("can I get from 3 to 17?", "m", call_model=call)
    assert (got.task, got.params, got.confidence, got.router) == ("connectivity", {"source": 3, "target": 17}, 0.95, "llm")
    assert (got.prompt_tokens, got.completion_tokens, got.latency_s) == (400, 50, 0.9)


def test_llm_route_sends_the_menu_and_turns_thinking_off():
    call, seen = fake_model({"task": "node_count", "confidence": 1})
    llm_route("how many nodes?", "m", call_model=call)
    assert seen["messages"][0] == {"role": "system", "content": ROUTER_PROMPT}
    assert seen["messages"][1] == {"role": "user", "content": "how many nodes?"}
    assert seen["tools"] == [ROUTE_TOOL] and seen["think"] is False


@pytest.mark.parametrize("args, text", [
    (None, "It is connectivity."),                            # answered in text, no tool call
    ("{not json", None),                                      # broken JSON
    ({"task": "diameter", "confidence": 1}, None),            # not a task
    ({"task": "node_degree", "node": "four"}, None),          # not a node id
])
def test_llm_route_never_crashes_on_a_bad_reply(args, text):
    call, _ = fake_model(args, text)
    got = llm_route("?", "m", call_model=call)
    assert got.task == UNKNOWN and got.reason and got.prompt_tokens == 400  # the cost is kept


def test_optional_fields_are_plain_integers_and_none_words_count_as_left_out():
    assert ROUTE_TOOL["function"]["parameters"]["properties"]["node"]["type"] == "integer"
    args = RouteArgs.model_validate({"task": "node_count", "confidence": 1, "node": "None", "u": "null"})
    assert args.node is None and args.u is None


def test_route_args_reject_true_as_a_node():
    with pytest.raises(Exception):
        RouteArgs.model_validate({"task": "node_degree", "node": True, "confidence": 1})
