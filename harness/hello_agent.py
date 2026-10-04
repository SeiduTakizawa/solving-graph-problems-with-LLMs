"""Your first agent: a model + graph tools + a loop. The tools live in harness/tools/graph_tools.py.

Run it:   uv run python -m harness.hello_agent "What is the degree of node 4?"
Test it:  uv run pytest tests/test_hello_agent.py
"""
import json
import re
import sys
import time
from dataclasses import dataclass

import litellm
import networkx as nx

from harness.tools.graph_tools import GRAPH_TOOLS, run_tool
from harness.trace import Trace

MODEL = "ollama_chat/qwen3:8b"

SYSTEM_PROMPT = (
    "You answer questions about a graph G. You cannot see G directly; use the tools to inspect it. "
    "When you know the answer, call submit_answer. "
    "If the question cannot be answered with the available tools, call cannot_answer instead of guessing."
)

# Ending tools. Not graph tools: the loop handles them itself instead of calling run_tool.
SUBMIT_ANSWER = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your final answer. This ends the task.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        },
    },
}

CANNOT_ANSWER = {
    "type": "function",
    "function": {
        "name": "cannot_answer",
        "description": "Use this when the question cannot be answered with the available tools, "
                       "or the question itself is invalid. This ends the task.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
}

TOOLS = GRAPH_TOOLS + [SUBMIT_ANSWER, CANNOT_ANSWER]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]

# Sent when the model replies with plain text instead of calling a tool.
NUDGE = (
    "You did not call a tool. Use the tools to inspect G, call submit_answer with your final answer, "
    "or call cannot_answer if the question cannot be answered."
)
# Sent when the plain text looks like a tool call the model wrote out by hand (it was not executed).
FORMAT_ERROR = (
    "Your last message contains a tool call written as plain text, so it was NOT executed. "
    "Do not write tool calls in your reply; call the tool through the function-calling interface."
)

MAX_REPEATS = 3  # stop if the model sends the same reply this many times in a row


@dataclass
class RunResult:
    answer: int | None
    status: str  # "submitted", "cannot_answer", "max_steps" or "loop_detected"
    messages: list[dict]
    reason: str | None = None  # why it could not answer, or why the run was stopped


def call_model(messages: list[dict]) -> dict:
    """Send the conversation to the model, return its reply as a plain dict.

    Token counts and latency go under "extra"; run_agent removes it before the reply
    joins the history, so the model never sees it.
    """
    start = time.time()
    response = litellm.completion(model=MODEL, messages=messages, tools=TOOLS)
    reply = response.choices[0].message.model_dump()
    reply["extra"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "latency_s": round(time.time() - start, 3),
    }
    return reply


def looks_like_text_tool_call(content: str | None) -> bool:
    """True if a plain-text reply contains a hand-written tool call, e.g. 'submit_answer\n{"answer": 0}'."""
    if not content:
        return False
    if "tool_call>" in content:  # leaked <tool_call> / </tool_call> tags
        return True
    names = "|".join(TOOL_NAMES)
    return re.search(rf"^\s*({names})\s*\(?\s*\{{", content, re.MULTILINE) is not None


def reply_signature(reply: dict) -> tuple:
    """What the model said, ignoring the random tool-call ids, so identical replies compare equal."""
    calls = tuple((c["function"]["name"], c["function"]["arguments"]) for c in reply.get("tool_calls") or [])
    return (reply.get("content") or "").strip(), calls


def run_agent(question: str, graph: nx.Graph, call_model=call_model, max_steps: int = 10,
              trace: Trace | None = None) -> RunResult:
    """Run the agent loop until the model submits, gives up, repeats itself, or runs out of steps.

    If a Trace is given, every model call, tool call and the outcome are logged to it.
    """
    log = trace.log if trace else (lambda event, **data: None)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}

    def finish(answer, status, steps, reason=None):
        log("run_end", answer=answer, status=status, reason=reason, steps=steps, **totals)
        return RunResult(answer, status, messages, reason)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    log("run_start", model=MODEL, question=question, max_steps=max_steps)
    previous, repeats = None, 0

    for step in range(1, max_steps + 1):
        # 1. Ask the model, and remember what it said.
        reply = call_model(messages)
        stats = reply.pop("extra", {})  # bookkeeping only, not part of the conversation
        for key in totals:
            totals[key] += stats.get(key, 0)
        messages.append(reply)
        log("model_call", step=step, content=reply.get("content"), tool_calls=reply.get("tool_calls"), **stats)

        # Stuck? The same reply MAX_REPEATS times in a row will not get any better.
        signature = reply_signature(reply)
        repeats = repeats + 1 if signature == previous else 1
        previous = signature
        if repeats >= MAX_REPEATS:
            return finish(None, "loop_detected", step, reason=f"same reply {repeats} times in a row")

        # 2. Run each tool it asked for and show it the result.
        tool_calls = reply["tool_calls"] or []
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])  # arguments arrive as a JSON string

            if name == "submit_answer":
                return finish(args.get("answer"), "submitted", step)
            if name == "cannot_answer":
                return finish(None, "cannot_answer", step, reason=args.get("reason"))

            try:
                result = run_tool(graph, name, args)
            except Exception as e:  # a buggy tool must not kill the agent
                result = {"error": f"Tool {name} crashed: {e}"}
            log("tool_call", step=step, name=name, args=args, result=result)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],  # tells the model which request this answers
                "content": json.dumps(result),
            })

        # 3. Plain-text reply: say what went wrong, so the model can fix it instead of repeating it.
        if not tool_calls:
            format_error = looks_like_text_tool_call(reply.get("content"))
            log("nudge", step=step, kind="format_error" if format_error else "no_tool_call")
            messages.append({"role": "user", "content": FORMAT_ERROR if format_error else NUDGE})

    return finish(None, "max_steps", max_steps)  # ran out of steps


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the biggest clique in the graph?"
    graph = nx.read_adjlist("data/graphs/er/small/0.txt", nodetype=int)
    trace = Trace("results/harness_runs/hello_agent.jsonl")
    result = run_agent(question, graph, trace=trace)

    for m in result.messages:
        print(f"[{m['role']}]", m.get("content") or "", m.get("tool_calls") or "")
    print(f"\nStatus: {result.status} | answer: {result.answer}" + (f" | reason: {result.reason}" if result.reason else ""))
    print(f"Trace: {trace.path} (run_id {trace.run_id})")
