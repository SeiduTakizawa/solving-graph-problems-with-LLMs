"""Compaction: when the conversation gets close to the model's context window, shrink its older parts.

Why: past the window the model server cuts the prompt or the reply, silently (Ollama's 4k default cut qwen3.5's
replies mid-thought). Shrinking old tool results first keeps the run going with what matters still in view.

Rules (plain Python, no LLM summary: free, and the same run always compacts the same way):
  - the system prompt and the question are never touched
  - the last `keep_last` rounds (an assistant message and the tool results / nudge after it) stay word for word
  - older tool results longer than SHORT_RESULT characters become a short version (`brief`) plus a note
  - older run_python code longer than SHORT_CODE characters is cut to its start (its result is what matters)
  - messages are never deleted: every tool call keeps its reply, as the chat format requires
Handles ("result_3") are not touched: their full values stay in the HandleStore.
"""
import json

from harness.brief import brief
from harness.prompts import SHORTENED_CODE, SHORTENED_RESULT

KEEP_LAST = 2  # rounds kept word for word
SHORT_RESULT = 300  # characters: tool results up to this size are cheap and stay as they are
SHORT_CODE = 200


def round_starts(messages: list[dict]) -> list[int]:
    """Indices of the assistant messages: each one starts a round."""
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def shorten_result(content: str) -> str:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        value = content
    if isinstance(value, dict):  # each field on its own: "neighbors: [0, 1, 2, ...] (99 items)"
        body = ", ".join(f"{key}: {brief(field, 60)}" for key, field in value.items())
    else:
        body = brief(value, 120)
    return f"{SHORTENED_RESULT} {brief(body, 200)}"


def shorten_calls(message: dict) -> tuple[dict, bool]:
    """The assistant message with long run_python code cut short; whether anything changed."""
    calls, changed = [], False
    for call in message.get("tool_calls") or []:
        function = call["function"]
        if function["name"] == "run_python":
            try:
                args = json.loads(function["arguments"]) if isinstance(function["arguments"], str) else function["arguments"]
            except json.JSONDecodeError:
                args = None
            code = args.get("code") if isinstance(args, dict) else None
            if isinstance(code, str) and len(code) > SHORT_CODE and not code.startswith(SHORTENED_CODE):
                short = f"{SHORTENED_CODE}\n{code[:SHORT_CODE]}..."
                call = {**call, "function": {**function, "arguments": json.dumps({**args, "code": short})}}
                changed = True
        calls.append(call)
    return ({**message, "tool_calls": calls} if changed else message), changed


def compact(messages: list[dict], keep_last: int = KEEP_LAST) -> tuple[list[dict], int]:
    """A shorter copy of the conversation, and how many messages were shortened (0: nothing left to shrink)."""
    starts = round_starts(messages)
    if len(starts) <= keep_last:
        return list(messages), 0
    protected_from = starts[-keep_last] if keep_last else len(messages)
    out, shortened = [], 0
    for i, message in enumerate(messages):
        if i < 2 or i >= protected_from:  # system prompt, question, and the recent rounds
            out.append(message)
            continue
        if message["role"] == "tool":
            content = message.get("content") or ""
            if len(content) > SHORT_RESULT and not content.startswith(SHORTENED_RESULT):
                message = {**message, "content": shorten_result(content)}
                shortened += 1
        elif message["role"] == "assistant":
            message, changed = shorten_calls(message)
            shortened += changed
        out.append(message)
    return out, shortened


def size(messages: list[dict]) -> int:
    """Characters in the conversation (a rough size for the trace; tokens come from the model server)."""
    return len(json.dumps(messages, default=str))
