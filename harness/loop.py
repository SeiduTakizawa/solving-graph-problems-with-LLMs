"""The agent loop: ask the model, run the tools it asks for, feed the results back, until it finishes."""
import json
from dataclasses import asdict, dataclass
from functools import partial

import networkx as nx

from harness import models
from harness.answers import CANNOT_ANSWER, check_answer, evidence_fields, make_submit_answer
from harness.brief import brief
from harness.parsing import looks_like_text_tool_call, parse_text_tool_call
from harness.prompts import FORMAT_ERROR, NUDGE, SYSTEM_PROMPT
from harness.tools.graph_tools import GRAPH_TOOLS, run_tool
from harness.tools.handles import HANDLE_LIMIT, READ_RESULT, HandleStore
from harness.trace import Trace


@dataclass(frozen=True)
class AgentConfig:
    """Settings that stay the same across the questions of one experiment."""
    model: str = models.DEFAULT_MODEL
    max_steps: int = 10
    max_repeats: int = 3  # stop if the model sends the same reply this many times in a row
    rescue: bool = True  # run tool calls written as text (lenient); False = only a FORMAT_ERROR (strict)
    graph_tools: tuple[str, ...] | None = None  # names of the graph tools to offer; None = all
    handle_limit: int | None = HANDLE_LIMIT  # results with more numbers than this become handles; None = off


@dataclass
class RunResult:
    answer: int | bool | list | None  # a number, true/false, a node list or an edge list
    status: str  # "submitted", "cannot_answer", "max_steps" or "loop_detected"
    messages: list[dict]
    reason: str | None = None  # why it could not answer, or why the run was stopped
    rescued: int = 0  # tool calls written as text that the harness parsed and ran anyway
    rejected: int = 0  # submitted answers the verifier sent back
    evidence: dict | None = None  # extra fields sent with the answer, e.g. {"path": [...]} behind a "yes"
    steps: int = 0  # model calls made
    prompt_tokens: int = 0  # summed over all model calls
    completion_tokens: int = 0
    model_latency_s: float = 0.0  # time spent waiting for the model (tools not included)


def parse_arguments(raw) -> tuple[dict, str | None]:
    """A tool call's arguments as a dict, or an error message for the model if they aren't a JSON object."""
    if isinstance(raw, dict):  # some providers already send a dict
        return raw, None
    try:
        args = json.loads(raw or "{}")  # arguments normally arrive as a JSON string; empty means none
    except json.JSONDecodeError as e:
        return {}, f"Your arguments are not valid JSON ({e.msg}). Send them again as a JSON object."
    if not isinstance(args, dict):
        return {}, "Your arguments must be a JSON object, e.g. {\"node\": 4}."
    return args, None


def reply_signature(reply: dict) -> tuple:
    """What the model said, ignoring the random tool-call ids, so identical replies compare equal."""
    calls = tuple((c["function"]["name"], c["function"]["arguments"]) for c in reply.get("tool_calls") or [])
    return (reply.get("content") or "").strip(), calls


def run_agent(question: str, graph: nx.Graph, answer_type: str = "number", verify=None,
              config: AgentConfig = AgentConfig(), call_model=None, trace: Trace | None = None) -> RunResult:
    """Run the agent loop until the model submits, gives up, repeats itself, or runs out of steps.

    answer_type (see harness/answers.py) sets what submit_answer accepts.
    verify: optional check of the final answer, verify(answer, **evidence) -> error message or None.
    Evidence is only sent for answer types that have it (e.g. verify(answer, path=[...]) for yes_no_with_path).
    A rejected answer goes back to the model as an error, like a wrong answer type.
    call_model: replaces the real model, call_model(messages, tools) -> reply (used by the tests).
    If a Trace is given, every model call, tool call and the outcome are logged to it.
    """
    call_model = call_model or partial(models.call_model, model=config.model)
    log = trace.log if trace else (lambda event, **data: None)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
    rescued = rejected = 0

    def finish(answer, status, steps, reason=None, evidence=None):
        log("run_end", answer=answer, evidence=evidence, status=status, reason=reason, steps=steps,
            rescued=rescued, rejected=rejected, **totals)
        return RunResult(answer, status, messages, reason=reason, rescued=rescued, rejected=rejected,
                         evidence=evidence, steps=steps, prompt_tokens=totals["prompt_tokens"],
                         completion_tokens=totals["completion_tokens"], model_latency_s=totals["latency_s"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    graph_tools = [t for t in GRAPH_TOOLS
                   if config.graph_tools is None or t["function"]["name"] in config.graph_tools]
    offered = {t["function"]["name"] for t in graph_tools}
    submit_tool = make_submit_answer(answer_type)
    tools = graph_tools + [submit_tool, CANNOT_ANSWER]
    log("run_start", question=question, answer_type=answer_type, verified=verify is not None,
        **{**asdict(config), "graph_tools": sorted(offered)})
    previous, repeats = None, 0
    store = HandleStore(config.handle_limit) if config.handle_limit else None

    for step in range(1, config.max_steps + 1):
        # 1. Ask the model, and remember what it said.
        reply = call_model(messages, tools)
        stats = reply.pop("extra", {})  # bookkeeping only, not part of the conversation
        for key in totals:
            totals[key] += stats.get(key, 0)
        # The model's hidden thinking is logged but not sent back: it would be re-read (and paid for)
        # on every later step, and the model does not need its old thinking.
        reasoning = reply.pop("reasoning_content", None)
        reply.pop("thinking_blocks", None)
        messages.append(reply)
        log("model_call", step=step, content=reply.get("content"), tool_calls=reply.get("tool_calls"),
            reasoning=reasoning, **stats)

        # Lenient mode: a tool call written as text is turned into a real one (and logged as such).
        if config.rescue and not reply.get("tool_calls"):
            parsed = parse_text_tool_call(reply.get("content"))
            if parsed:
                name, args = parsed
                reply["tool_calls"] = [{"id": f"rescued_{step}", "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}]
                rescued += 1
                log("rescued_tool_call", step=step, name=name, args=args)

        # Stuck? The same reply max_repeats times in a row will not get any better.
        signature = reply_signature(reply)
        repeats = repeats + 1 if signature == previous else 1
        previous = signature
        if repeats >= config.max_repeats:
            return finish(None, "loop_detected", step, reason=f"same reply {repeats} times in a row")

        # 2. Run each tool it asked for and show it the result.
        tool_calls = reply["tool_calls"] or []
        for call in tool_calls:
            name = call["function"]["name"]
            args, args_error = parse_arguments(call["function"]["arguments"])

            if args_error:  # broken JSON: tell the model instead of crashing the run
                result = {"error": args_error}
            elif name == "submit_answer":
                if store:  # a handle ("result_1") stands for the stored value it names
                    args = {key: store.resolve(value) for key, value in args.items()}
                evidence = {field: args.get(field) for field in evidence_fields(answer_type)}
                error = check_answer(args.get("answer"), answer_type)
                if error and store and store.values:
                    error += ' A stored result can be submitted by its handle, e.g. "result_1".'
                if error is None and verify is not None:
                    error = verify(args["answer"], **evidence)  # right type; is it also a valid answer?
                    if error:
                        rejected += 1
                        log("verifier_rejected", step=step, answer=args["answer"], evidence=evidence, error=error)
                if error is None:
                    return finish(args["answer"], "submitted", step, evidence=evidence or None)
                result = {"error": error}  # wrong shape or invalid: tell the model and let it try again
            elif name == "cannot_answer":
                return finish(None, "cannot_answer", step, reason=args.get("reason"))
            elif name not in offered:
                result = {"error": f"Unknown tool {name}. Available tools: {sorted(offered)}."}
            elif name == "read_result":
                result = store.read(args)
            else:
                try:
                    result = run_tool(graph, name, args)
                except Exception as e:  # a buggy tool must not kill the agent
                    result = {"error": f"Tool {name} crashed: {brief(str(e))}"}
                if store and "error" not in result:
                    result = store.compact(result)
                    # read_result is offered from the first handle on, not before: on graphs where nothing is
                    # ever big (the whole current dataset) the tool list stays exactly as it was.
                    if store.values and "read_result" not in offered:
                        tools.insert(len(graph_tools), READ_RESULT)
                        offered.add("read_result")
                        # ...and submit_answer starts accepting handles (its schema said "array" until now).
                        tools[tools.index(submit_tool)] = submit_tool = make_submit_answer(answer_type, handles=True)
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

    return finish(None, "max_steps", config.max_steps)  # ran out of steps
