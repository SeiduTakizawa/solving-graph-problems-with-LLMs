"""Compaction: near the context window, older tool results and code are shortened; recent rounds stay as they are."""
import json

import networkx as nx

from harness import models
from harness.compaction import KEEP_LAST, SHORT_RESULT, compact
from harness.loop import AgentConfig, run_agent
from harness.prompts import SHORTENED_CODE, SHORTENED_RESULT
from harness.trace import Trace, read_trace

BIG = nx.complete_graph(100)  # get_neighbors gives 99 ids: long enough to shorten, short of a handle (200)
WINDOW = models.OLLAMA_NUM_CTX
FULL = int(0.8 * WINDOW)  # above the 75% threshold
OVERFLOWING = int(0.95 * WINDOW)  # above the 90% "stop" line


def call(name, args, call_id):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(call_id, value):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(value)}


def conversation(rounds=4):
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}]
    for i in range(rounds):
        messages += [call("get_neighbors", {"node": i}, f"c{i}"), result(f"c{i}", {"neighbors": list(range(100))})]
    return messages


# --- compact() on its own ---

def test_older_results_are_shortened_recent_ones_kept():
    messages = conversation(rounds=4)
    out, shortened = compact(messages)
    assert shortened == 4 - KEEP_LAST
    assert out[:2] == messages[:2]  # system prompt and question untouched
    assert out[3]["content"].startswith(SHORTENED_RESULT) and out[5]["content"].startswith(SHORTENED_RESULT)
    assert out[6:] == messages[6:]  # the last two rounds word for word
    assert len(out[3]["content"]) < len(messages[3]["content"]) // 2


def test_no_message_is_deleted_and_every_call_keeps_its_reply():
    messages = conversation(rounds=5)
    out, _ = compact(messages)
    assert [m["role"] for m in out] == [m["role"] for m in messages]
    assert [m.get("tool_call_id") for m in out] == [m.get("tool_call_id") for m in messages]


def test_small_results_stay():
    messages = conversation(rounds=1) + [call("degree", {"node": 0}, "d"), result("d", {"degree": 99})]
    messages += conversation(rounds=2)[2:]
    out, _ = compact(messages)
    assert out[5] == messages[5]  # {"degree": 99} is far under SHORT_RESULT: nothing to gain


def test_compacting_twice_shortens_nothing_new():
    out, _ = compact(conversation(rounds=4))
    again, shortened = compact(out)
    assert shortened == 0 and again == out


def test_original_messages_are_not_changed():
    messages = conversation(rounds=4)
    before = json.dumps(messages)
    compact(messages)
    assert json.dumps(messages) == before  # the trace already logged them as they were


def test_old_code_is_cut_recent_code_kept():
    code = "neighbors = get_neighbors(0)\n" + "x = 1\n" * 100
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                call("run_python", {"code": code}, "p1"), result("p1", {"result": 1}),
                call("get_neighbors", {"node": 1}, "c1"), result("c1", {"neighbors": [0]}),
                call("run_python", {"code": code}, "p2"), result("p2", {"result": 2})]
    out, shortened = compact(messages)
    old_code = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])["code"]
    assert shortened == 1 and old_code.startswith(SHORTENED_CODE) and len(old_code) < 300
    assert out[6] == messages[6]  # the latest code stays


def test_nothing_to_do_with_few_rounds():
    messages = conversation(rounds=KEEP_LAST)
    assert compact(messages) == (messages, 0)


# --- in the loop ---

def scripted(*steps):
    """A fake model: each step is (reply, prompt_tokens). Records the messages it was sent."""
    seen, script = [], iter(steps)

    def model(messages, tools):
        seen.append(json.loads(json.dumps(messages)))
        reply, prompt_tokens = next(script)
        return {**reply, "extra": {"prompt_tokens": prompt_tokens, "completion_tokens": 10, "latency_s": 0.1}}
    return model, seen


def test_loop_compacts_when_the_prompt_passes_75_percent(tmp_path):
    model, seen = scripted(
        (call("get_neighbors", {"node": 0}, "a"), 1000),
        (call("get_neighbors", {"node": 1}, "b"), 2000),
        (call("get_neighbors", {"node": 2}, "c"), FULL),  # after this step: compact
        (call("submit_answer", {"answer": 99}, "d"), 1000),
    )
    trace = Trace(tmp_path / "run.jsonl")
    out = run_agent("Degree of node 0?", BIG, call_model=model, trace=trace)
    assert out.status == "submitted" and out.compactions == 1
    last_prompt = seen[-1]
    assert last_prompt[3]["content"].startswith(SHORTENED_RESULT)  # round 1's result, as the model saw it
    assert not last_prompt[5]["content"].startswith(SHORTENED_RESULT)  # rounds 2 and 3 kept
    event = next(e for e in read_trace(trace.path) if e["event"] == "compacted")
    assert event["step"] == 3 and event["shortened"] == 1 and event["chars_after"] < event["chars_before"]


def test_below_the_threshold_nothing_changes():
    model, seen = scripted(
        (call("get_neighbors", {"node": 0}, "a"), 1000),
        (call("get_neighbors", {"node": 1}, "b"), 1000),
        (call("get_neighbors", {"node": 2}, "c"), int(0.7 * WINDOW)),
        (call("submit_answer", {"answer": 99}, "d"), 1000),
    )
    out = run_agent("Degree of node 0?", BIG, call_model=model)
    assert out.compactions == 0
    assert all(not str(m.get("content")).startswith(SHORTENED_RESULT) for m in seen[-1])


def test_full_and_nothing_left_to_shorten_stops_loudly():
    model, _ = scripted((call("get_neighbors", {"node": 0}, "a"), OVERFLOWING))
    out = run_agent("Degree of node 0?", BIG, call_model=model)
    assert out.status == "context_full" and out.answer is None
    assert str(WINDOW) in out.reason


def test_compaction_can_be_turned_off():
    model, _ = scripted(
        (call("get_neighbors", {"node": 0}, "a"), OVERFLOWING),
        (call("submit_answer", {"answer": 99}, "b"), OVERFLOWING),
    )
    out = run_agent("Degree of node 0?", BIG, call_model=model, config=AgentConfig(compact_at=None))
    assert out.status == "submitted" and out.compactions == 0  # the ablation: no compaction, no stop


def test_api_models_have_no_window_so_no_compaction():
    model, _ = scripted(
        (call("get_neighbors", {"node": 0}, "a"), OVERFLOWING),
        (call("submit_answer", {"answer": 99}, "b"), 1000),
    )
    out = run_agent("Degree of node 0?", BIG, call_model=model, config=AgentConfig(model="openai/gpt-5"))
    assert out.status == "submitted" and out.compactions == 0


def test_short_result_threshold_is_sane():
    assert len(json.dumps({"neighbors": list(range(99))})) > SHORT_RESULT  # the test graph really gets shortened
