"""Your first agent: a model + graph tools + a loop. The tools live in harness/tools/graph_tools.py.

Run it:   uv run python -m harness.hello_agent "What is the degree of node 4?"
          uv run python -m harness.hello_agent "Is there a cycle in G?" --answer-type yes_no
Test it:  uv run pytest tests/test_hello_agent.py
"""
import json
import re
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

# The shape of the final answer, per kind of question. submit_answer is built to match.
ANSWER_TYPES = {
    "number": {"schema": {"type": "integer"}, "description": "a whole number"},
    "yes_no": {"schema": {"type": "boolean"}, "description": "true or false"},
    "node_list": {"schema": {"type": "array", "items": {"type": "integer"}}, "description": "a list of node ids"},
}


def make_submit_answer(answer_type: str) -> dict:
    """The submit_answer tool for one answer type. Ends the task; the loop handles it itself."""
    expected = ANSWER_TYPES[answer_type]
    return {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": f"Submit your final answer, which must be {expected['description']}. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": expected["schema"]},
                "required": ["answer"],
            },
        },
    }


def check_answer(answer, answer_type: str) -> str | None:
    """None if the answer has the right shape, otherwise an error message for the model."""
    # Careful: in Python True/False are also ints, so a number must not be a bool and vice versa.
    is_int = lambda x: isinstance(x, int) and not isinstance(x, bool)
    ok = {
        "number": is_int(answer),
        "yes_no": isinstance(answer, bool),
        "node_list": isinstance(answer, list) and all(is_int(x) for x in answer),
    }[answer_type]
    if ok:
        return None
    return (f"Invalid answer {json.dumps(answer)}: the answer must be {ANSWER_TYPES[answer_type]['description']}. "
            "Call submit_answer again with the right type.")


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

TOOL_NAMES = [t["function"]["name"] for t in GRAPH_TOOLS] + ["submit_answer", "cannot_answer"]

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
    answer: int | bool | list[int] | None
    status: str  # "submitted", "cannot_answer", "max_steps" or "loop_detected"
    messages: list[dict]
    reason: str | None = None  # why it could not answer, or why the run was stopped
    rescued: int = 0  # tool calls written as text that the harness parsed and ran anyway
    rejected: int = 0  # submitted answers the verifier sent back


def call_model(messages: list[dict], tools: list[dict]) -> dict:
    """Send the conversation and the available tools to the model, return its reply as a plain dict.

    Token counts and latency go under "extra"; run_agent removes it before the reply
    joins the history, so the model never sees it.
    """
    start = time.time()
    response = litellm.completion(model=MODEL, messages=messages, tools=tools)
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


def parse_text_tool_call(content: str | None) -> tuple[str, dict] | None:
    """Recover a tool call the model wrote as text instead of calling it. Returns (name, args) or None.

    Handles the two forms qwen3 produces:
      submit_answer\n{"answer": [2, 5]}\n</tool_call>                        (name, then arguments)
      <tool_call>{"name": "get_neighbors", "arguments": {"node": 4}}</tool_call>  (its own format)
    """
    if not content:
        return None
    decoder = json.JSONDecoder()

    def json_at(index):
        try:
            value, _ = decoder.raw_decode(content, index)
            return value
        except json.JSONDecodeError:
            return None

    for match in re.finditer(r"\{", content):
        value = json_at(match.start())
        if (isinstance(value, dict) and value.get("name") in TOOL_NAMES
                and isinstance(value.get("arguments"), dict)):
            return value["name"], value["arguments"]

    names = "|".join(TOOL_NAMES)
    for match in re.finditer(rf"^\s*({names})\s*\(?\s*(?=\{{)", content, re.MULTILINE):
        value = json_at(match.end())
        if isinstance(value, dict):
            return match.group(1), value
    return None


def reply_signature(reply: dict) -> tuple:
    """What the model said, ignoring the random tool-call ids, so identical replies compare equal."""
    calls = tuple((c["function"]["name"], c["function"]["arguments"]) for c in reply.get("tool_calls") or [])
    return (reply.get("content") or "").strip(), calls


def run_agent(question: str, graph: nx.Graph, answer_type: str = "number", call_model=call_model,
              max_steps: int = 10, trace: Trace | None = None, rescue: bool = True,
              verify=None) -> RunResult:
    """Run the agent loop until the model submits, gives up, repeats itself, or runs out of steps.

    answer_type ("number", "yes_no" or "node_list") sets what submit_answer accepts.
    rescue: run tool calls the model wrote as text (each one is logged as rescued_tool_call).
    With rescue=False they only get a FORMAT_ERROR (strict mode).
    verify: optional check of the final answer, verify(answer) -> error message or None.
    A rejected answer goes back to the model as an error, like a wrong answer type.

    If a Trace is given, every model call, tool call and the outcome are logged to it.
    """
    log = trace.log if trace else (lambda event, **data: None)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
    rescued = rejected = 0

    def finish(answer, status, steps, reason=None):
        log("run_end", answer=answer, status=status, reason=reason, steps=steps, rescued=rescued,
            rejected=rejected, **totals)
        return RunResult(answer, status, messages, reason, rescued, rejected)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = GRAPH_TOOLS + [make_submit_answer(answer_type), CANNOT_ANSWER]
    log("run_start", model=MODEL, question=question, answer_type=answer_type, max_steps=max_steps,
        rescue=rescue, verified=verify is not None)
    previous, repeats = None, 0

    for step in range(1, max_steps + 1):
        # 1. Ask the model, and remember what it said.
        reply = call_model(messages, tools)
        stats = reply.pop("extra", {})  # bookkeeping only, not part of the conversation
        for key in totals:
            totals[key] += stats.get(key, 0)
        messages.append(reply)
        log("model_call", step=step, content=reply.get("content"), tool_calls=reply.get("tool_calls"), **stats)

        # Lenient mode: a tool call written as text is turned into a real one (and logged as such).
        if rescue and not reply.get("tool_calls"):
            parsed = parse_text_tool_call(reply.get("content"))
            if parsed:
                name, args = parsed
                reply["tool_calls"] = [{"id": f"rescued_{step}", "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}]
                rescued += 1
                log("rescued_tool_call", step=step, name=name, args=args)

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
                error = check_answer(args.get("answer"), answer_type)
                if error is None and verify is not None:
                    error = verify(args["answer"])  # right type; is it also a valid answer?
                    if error:
                        rejected += 1
                        log("verifier_rejected", step=step, answer=args["answer"], error=error)
                if error is None:
                    return finish(args["answer"], "submitted", step)
                result = {"error": error}  # wrong shape or invalid: tell the model and let it try again
            elif name == "cannot_answer":
                return finish(None, "cannot_answer", step, reason=args.get("reason"))
            else:
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
    import argparse

    parser = argparse.ArgumentParser(description="Ask the agent one question about data/graphs/er/small/0.txt.")
    parser.add_argument("question", nargs="?", default="is there any cycles?")
    parser.add_argument("--answer-type", default="number", choices=list(ANSWER_TYPES))
    args = parser.parse_args()

    graph = nx.read_adjlist("data/graphs/er/large/0.txt", nodetype=int)
    trace = Trace("results/harness_runs/hello_agent.jsonl")
    result = run_agent(args.question, graph, answer_type=args.answer_type, trace=trace)

    for m in result.messages:
        print(f"[{m['role']}]", m.get("content") or "", m.get("tool_calls") or "")
    print(f"\nStatus: {result.status} | answer: {result.answer}" + (f" | reason: {result.reason}" if result.reason else ""))
    print(f"Trace: {trace.path} (run_id {trace.run_id})")
