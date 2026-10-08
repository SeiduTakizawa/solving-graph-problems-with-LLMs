"""call_model retries temporary errors (a crashed or busy server) and raises real ones at once."""
from types import SimpleNamespace

import litellm
import pytest

from harness import models


def fake_response():
    message = SimpleNamespace(model_dump=lambda: {"role": "assistant", "content": "", "tool_calls": None})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def cuda_crash():
    return litellm.InternalServerError(message="CUDA error: the launch timed out", llm_provider="ollama_chat",
                                       model="qwen3:8b")


@pytest.fixture
def no_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(models.time, "sleep", waits.append)
    return waits


def scripted_completion(monkeypatch, *outcomes):
    """litellm.completion that raises or returns the given outcomes in order; returns the list of calls."""
    calls, script = [], iter(outcomes)

    def completion(**kwargs):
        calls.append(kwargs)
        outcome = next(script)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(models.litellm, "completion", completion)
    return calls


def test_no_error_no_retry(monkeypatch, no_waiting):
    calls = scripted_completion(monkeypatch, fake_response())
    reply = models.call_model([], [])
    assert len(calls) == 1 and no_waiting == []
    assert "retries" not in reply["extra"]  # normal traces stay unchanged


def test_transient_error_is_retried(monkeypatch, no_waiting):
    calls = scripted_completion(monkeypatch, cuda_crash(), cuda_crash(), fake_response())
    reply = models.call_model([], [])
    assert len(calls) == 3
    assert no_waiting == list(models.RETRY_WAITS_S[:2])
    assert reply["extra"]["retries"] == 2
    assert "CUDA error" in reply["extra"]["retry_errors"][0]
    assert reply["extra"]["prompt_tokens"] == 10  # the successful call's numbers


def test_gives_up_after_the_last_retry(monkeypatch, no_waiting):
    scripted_completion(monkeypatch, *[cuda_crash() for _ in range(len(models.RETRY_WAITS_S) + 1)])
    with pytest.raises(litellm.InternalServerError):
        models.call_model([], [])
    assert no_waiting == list(models.RETRY_WAITS_S)


def test_real_errors_are_not_retried(monkeypatch, no_waiting):
    wrong_model = litellm.NotFoundError(message="model 'qwen9:99b' not found", model="qwen9:99b",
                                        llm_provider="ollama_chat")
    calls = scripted_completion(monkeypatch, wrong_model, fake_response())
    with pytest.raises(litellm.NotFoundError):
        models.call_model([], [])
    assert len(calls) == 1 and no_waiting == []


def test_retries_reach_the_trace(monkeypatch, no_waiting, tmp_path):
    import networkx as nx

    from harness.loop import run_agent
    from harness.trace import Trace, read_trace

    submit = fake_response()
    submit.choices[0].message.model_dump = lambda: {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "submit_answer", "arguments": '{"answer": 1}'}}]}
    scripted_completion(monkeypatch, cuda_crash(), submit)
    trace = Trace(tmp_path / "run.jsonl")
    result = run_agent("What is the degree of node 0?", nx.path_graph(2), trace=trace)

    assert result.status == "submitted"
    model_call = next(e for e in read_trace(trace.path) if e["event"] == "model_call")
    assert model_call["retries"] == 1 and "CUDA error" in model_call["retry_errors"][0]


def test_ollama_models_get_a_bigger_context(monkeypatch, no_waiting):
    # Ollama's default is 4,096 tokens; qwen3.5's replies were cut off silently at that edge.
    calls = scripted_completion(monkeypatch, fake_response(), fake_response())
    models.call_model([], [], model="ollama_chat/qwen3.5:9b")
    models.call_model([], [], model="openai/gpt-5")
    assert calls[0]["num_ctx"] == models.OLLAMA_NUM_CTX
    assert "num_ctx" not in calls[1]  # API models keep the provider's own context


def test_a_prompt_near_the_limit_is_flagged(monkeypatch, no_waiting):
    big = fake_response()
    big.usage.prompt_tokens = models.OLLAMA_NUM_CTX - 100
    scripted_completion(monkeypatch, fake_response(), big)
    assert "near_context_limit" not in models.call_model([], [], model="ollama_chat/qwen3:8b")["extra"]
    assert models.call_model([], [], model="ollama_chat/qwen3:8b")["extra"]["near_context_limit"] == models.OLLAMA_NUM_CTX
