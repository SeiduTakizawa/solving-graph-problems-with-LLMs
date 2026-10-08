"""Report: numbers come out right from hand-made result rows."""
import json

import pytest

from eval.analysis import report as rep


def row(task="cycle_check", graph_id=0, params=None, run=1, correct=True, **extra):
    return {"run": run, "model": "fake", "task": task, "graph_id": graph_id, "params": params or {},
            "answer_type": "yes_no_with_cycle", "verified": True, "status": "submitted", "answer": True,
            "correct": correct, "rejected": 0, "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 1.0,
            **extra}


def test_bootstrap_ci_is_reproducible_and_contains_the_accuracy():
    rows = [row(graph_id=i, correct=i % 4 != 0) for i in range(40)]  # 75% correct
    low, high = rep.bootstrap_ci(rows)
    assert low < 0.75 < high
    assert rep.bootstrap_ci(rows) == (low, high)  # fixed seed: same logs, same interval


def test_bootstrap_ci_all_correct_is_a_point():
    assert rep.bootstrap_ci([row(graph_id=i) for i in range(10)]) == (1.0, 1.0)


def test_runs_of_one_question_stay_together():
    # One question answered 3 times counts as one question, not three.
    s = rep.summarize([row(run=r) for r in (1, 2, 3)])
    assert s["questions"] == 1 and s["runs"] == 3 and s["answers"] == 3


@pytest.mark.parametrize("changes, checked", [
    ({}, True),                                          # a "yes" with evidence, checker on
    ({"answer": False}, False),                          # a "no" is accepted without a look
    ({"verified": False}, False),                        # checkers off
    ({"status": "max_steps", "answer": None}, False),    # no answer at all
    ({"answer_type": "node_list", "answer": [1]}, True),  # plain answer types are always looked at
])
def test_was_checked(changes, checked):
    assert rep.was_checked(row(**changes)) is checked


def test_crashed_runs_are_left_out_of_tokens_and_time():
    rows = [row(), row(graph_id=1, status="error: boom", prompt_tokens=None, completion_tokens=None, correct=False)]
    s = rep.summarize(rows)
    assert s["tokens"] == 120 and s["time"] == 1.0
    assert s["accuracy"] == 0.5 and s["no_answer"] == 1


def test_report_reads_old_and_new_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "RESULTS_DIR", tmp_path)
    (tmp_path / "exp").mkdir()
    old = {k: v for k, v in row(graph_id=1).items() if k not in ("task", "params")} | {"node": 3}  # before the registry
    lines = [row(), old]
    (tmp_path / "exp" / "results.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    text, summaries = rep.report(["exp"], by_task=True)
    assert "| exp | fake |" in text and "node_degree" in text and "cycle_check" in text
    assert summaries["exp"]["questions"] == 2


def test_strict_accuracy_does_not_count_rescued_answers():
    rows = [row(graph_id=0), row(graph_id=1, rescued=1), row(graph_id=2, correct=False)]
    s = rep.summarize(rows)
    assert s["accuracy"] == pytest.approx(2 / 3) and s["strict"] == pytest.approx(1 / 3)


# --- Code use (traces.jsonl) ---

def call(run_id, name, error=None):
    return {"run_id": run_id, "event": "tool_call", "name": name, "result": {"error": error} if error else {"ok": 1}}


def test_code_use_counts():
    rows = [row(run_id="a", python="tools"), row(run_id="b", graph_id=1, python="tools")]
    events = {
        "a": [call("a", "get_neighbors"), call("a", "run_python", error="line 1: NameError"), call("a", "run_python"),
              call("a", "submit_answer")],
        "b": [{"run_id": "b", "event": "nudge", "kind": "empty_reply"}, call("b", "degree"), call("b", "submit_answer")],
    }
    c = rep.code_use(rows, events)
    assert c["with_code"] == 0.5 and c["mixed"] == 0.5  # only run a wrote code, and it also used a tool
    assert c["code_calls"] == 1.0 and c["code_errors"] == 0.5  # 2 code calls over 2 questions, 1 of them failed
    assert c["tool_calls"] == 1.0  # get_neighbors + degree; submit_answer is not a graph tool
    assert c["empty_replies"] == 1


def test_code_not_offered_says_so():
    c = rep.code_use([row(run_id="a")], {"a": [call("a", "degree")]})
    assert rep.code_line(["exp"], c)[1] == "not offered"


def test_report_with_code_reads_the_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "RESULTS_DIR", tmp_path)
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "results.jsonl").write_text(json.dumps(row(run_id="a", python="tools")) + "\n")
    (tmp_path / "exp" / "traces.jsonl").write_text(json.dumps(call("a", "run_python")) + "\n")
    text, _ = rep.report(["exp"], by_task=True, code=True)
    assert "## Code use" in text and "| exp | 100% | 0% | 1.0 | 0% | 0.0 | 0 |" in text
