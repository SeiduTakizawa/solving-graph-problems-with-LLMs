"""Waypoint task: generated questions, grading, and the trajectory classes."""
import networkx as nx

from eval.analysis.trajectory import classify
from eval.tasks import SPLITS, is_correct, load_graph, load_tasks, reference_answer

P = {"source": 0, "via": 2, "target": 4}


def call(name, **args):
    return (name, args)


def test_ideal_path_in_either_order_and_direction():
    calls = [call("shortest_path", source=2, target=4), call("shortest_path", source=2, target=0)]
    assert classify("shortest_path_via", P, calls, True) == {
        "kind": "ideal", "shortcut": False, "has_path_leg": False, "manual": False, "calls": 2}


def test_extra_calls_and_flags():
    calls = [call("shortest_path", source=0, target=4), call("get_neighbors", node=2),
             call("shortest_path", source=0, target=2), call("shortest_path", source=2, target=4)]
    r = classify("shortest_path_via", P, calls, True)
    assert r["kind"] == "extra calls" and r["shortcut"] and r["manual"] and r["calls"] == 4


def test_right_calls_but_wrong_answer():
    calls = [call("shortest_path", source=0, target=2), call("shortest_path", source=2, target=4)]
    assert classify("shortest_path_via", P, calls, False)["kind"] == "right calls, wrong"


def test_a_different_route_to_a_correct_answer_is_not_a_failure():
    # Seen with qwen3:8b: has_path for the first leg, the shortcut call, then the way back read from the first
    # result. Correct answer, so it's an alternative path, flagged, not a failure.
    calls = [call("has_path", source=0, target=2), call("shortest_path", source=0, target=4)]
    r = classify("shortest_path_via", P, calls, True)
    assert r["kind"] == "alternative path" and r["has_path_leg"] and r["shortcut"]
    assert classify("shortest_path_via", P, calls, False)["kind"] == "wrong path"
    assert classify("shortest_path_via", P, [], False)["kind"] == "wrong path"


def test_generated_questions_force_a_detour_and_are_reproducible():
    for size in ("small", "large"):
        items = load_tasks("shortest_path_via", size, "dev", 100)
        assert items == load_tasks("shortest_path_via", size, "dev", 100)  # fixed seeds
        for item in items:
            assert item["graph_id"] in SPLITS["dev"]
            g, p = load_graph(size, item["graph_id"]), item["params"]
            assert len({p["source"], p["via"], p["target"]}) == 3
            plain = nx.shortest_path(g, p["source"], p["target"])
            assert not is_correct("shortest_path_via", g, p, plain)  # the shortcut is always wrong
            assert is_correct("shortest_path_via", g, p, reference_answer("shortest_path_via", g, p))
