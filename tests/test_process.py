"""Process metrics (tool precision / recall / F1, params, exact) on hand-made call lists."""
import pytest

from eval.analysis.process import process_metrics, tools_in_code
from eval.tasks import EXPECTED_CALLS, load_tasks
from harness.tasks import TASKS
from harness.tools.graph_tools import TOOLS


def call(name, **args):
    return {"name": name, "args": args}


def test_one_right_call_is_perfect():
    m = process_metrics("node_degree", {"node": 4}, [call("degree", node=4), call("submit_answer", answer=2)])
    assert (m["precision"], m["recall"], m["f1"], m["params"], m["exact"]) == (1, 1, 1, 1, True)


def test_an_equivalent_tool_counts():
    assert process_metrics("node_degree", {"node": 4}, [call("get_neighbors", node=4)])["recall"] == 1


def test_exploring_lowers_precision_not_recall():
    m = process_metrics("node_degree", {"node": 4}, [call("graph_info"), call("degree", node=4)])
    assert m["recall"] == 1 and m["precision"] == 0.5 and not m["exact"] and m["unexpected"] == ["graph_info"]


def test_repeating_a_call_lowers_precision():
    # As in theirs: one expected call made twice is still one expected call, out of two calls.
    m = process_metrics("node_degree", {"node": 4}, [call("degree", node=4), call("degree", node=4)])
    assert m["precision"] == 0.5 and m["exact"]  # repeats are not "unexpected" tools, as in theirs


def test_wrong_argument():
    assert process_metrics("node_degree", {"node": 4}, [call("degree", node=5)])["params"] == 0


def test_node_ids_as_strings_match():
    assert process_metrics("node_degree", {"node": 4}, [call("degree", node="4")])["params"] == 1


def test_swapped_pair_matches_on_undirected_graphs():
    assert process_metrics("edge_existence", {"u": 1, "v": 2}, [call("has_edge", u=2, v=1)])["params"] == 1
    assert process_metrics("edge_existence", {"u": 1, "v": 2}, [call("has_edge", u=1, v=3)])["params"] == 0.5


def test_missing_call():
    m = process_metrics("shortest_path", {"source": 0, "target": 5}, [call("get_neighbors", node=0)])
    assert m["recall"] == 0 and m["f1"] == 0 and m["params"] is None


def test_two_legs_of_a_waypoint_route():
    params = {"source": 0, "via": 3, "target": 5}
    both = process_metrics("shortest_path_via", params,
                           [call("shortest_path", source=0, target=3), call("shortest_path", source=3, target=5)])
    assert (both["precision"], both["recall"], both["params"]) == (1, 1, 1)
    one = process_metrics("shortest_path_via", params, [call("shortest_path", source=0, target=5)])
    assert one["params"] == 0.5  # each leg's best match gets one of its two arguments right


def test_code_that_calls_a_tool_fills_the_slot():
    code = "best = max(get_neighbors(10), key=lambda n: degree(n))\nresult = best"
    m = process_metrics("hop_max_degree", {"node": 10, "k": 1}, [call("run_python", code=code)])
    assert m["recall"] == 1 and m["precision"] == 1 and m["params"] is None  # code arguments can't be read


def test_networkx_methods_are_not_our_tools():
    assert tools_in_code("d = G.degree(5)\nn = nx.neighbors(G, 5)") == set()
    assert tools_in_code("d = degree(5)") == {"degree"}


def test_answer_tools_are_not_scored():
    m = process_metrics("node_degree", {"node": 4},
                        [call("degree", node=4), call("submit_answer", answer=3), call("submit_answer", answer=2)])
    assert m["precision"] == 1


def test_task_without_expected_calls():
    assert process_metrics("not_a_task", {}, [call("degree", node=1)]) is None


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_has_expected_calls_that_fit_its_params(task):
    assert task in EXPECTED_CALLS
    params = load_tasks(task, "small", "dev", n=1)[0]["params"]
    for slot in EXPECTED_CALLS[task]:
        for tool, spec in slot.items():
            assert tool in TOOLS, f"{task}: {tool} is not a tool"
            assert set(spec.values()) <= set(params), f"{task}: {tool} needs a parameter the question lacks"
