"""Task registry (harness side) and dataset loading + grading (eval side)."""
import networkx as nx
import pytest

from eval.tasks import SPLITS, is_correct, load_graph, load_tasks, reference_answer
from harness.hello_agent import ANSWER_TYPES, check_answer
from harness.tasks import TASKS

ALL_TASKS = sorted(TASKS)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_registry_entry_is_valid(task):
    assert TASKS[task].answer_type in ANSWER_TYPES
    assert TASKS[task].name == task


@pytest.mark.parametrize("task", ALL_TASKS)
def test_dataset_questions_load(task):
    items = load_tasks(task, "small", "dev", n=30)
    assert len(items) == 30 or task == "connectivity"  # connectivity has only 28 small dev questions
    for item in items:
        assert item["graph_id"] in SPLITS["dev"]
        assert "edgelist" not in item["question"] and "[(" not in item["question"]  # graph never in the prompt
        assert all(isinstance(v, int) for v in item["params"].values())


def test_test_split_is_separate():
    dev = {i["graph_id"] for i in load_tasks("node_degree", "small", "dev", n=1000)}
    test = {i["graph_id"] for i in load_tasks("node_degree", "small", "test", n=1000)}
    assert dev and test and not dev & test


def test_questions_spread_over_graphs():
    items = load_tasks("node_degree", "small", "dev", n=50)
    assert len({i["graph_id"] for i in items}) == 50  # one per graph before any graph gets a second


@pytest.mark.parametrize("task", ALL_TASKS)
def test_reference_answer_is_graded_correct_and_has_the_right_type(task):
    for item in load_tasks(task, "small", "dev", n=20):
        graph = load_graph("small", item["graph_id"])
        answer = reference_answer(task, graph, item["params"])
        assert check_answer(answer, TASKS[task].answer_type) is None
        assert is_correct(task, graph, item["params"], answer)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_no_answer_is_wrong(task):
    item = load_tasks(task, "small", "dev", n=1)[0]
    assert not is_correct(task, load_graph("small", item["graph_id"]), item["params"], None)


# Grading details

G = nx.Graph([(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)])  # from 0 to 3: shortest is 0-4-3, longer is 0-1-2-3


def test_any_shortest_path_counts():
    square = nx.cycle_graph(4)  # 0-1-2-3-0: two shortest paths from 0 to 2
    params = {"source": 0, "target": 2}
    assert is_correct("shortest_path", square, params, [0, 1, 2])
    assert is_correct("shortest_path", square, params, [0, 3, 2])


def test_longer_or_fake_path_is_wrong():
    params = {"source": 0, "target": 3}
    assert not is_correct("shortest_path", G, params, [0, 1, 2, 3])  # real but longer than 0-4-3
    assert not is_correct("shortest_path", G, params, [0, 3])  # 0-3 is not an edge


def test_neighbors_in_any_order():
    assert is_correct("connected_nodes", G, {"node": 0}, [4, 1])
    assert not is_correct("connected_nodes", G, {"node": 0}, [1])


def test_true_is_not_one():
    # In Python True == 1, but a yes/no answer is not a count.
    assert not is_correct("node_degree", nx.path_graph(2), {"node": 0}, True)
    assert not is_correct("cycle_check", nx.cycle_graph(3), {}, 1)
