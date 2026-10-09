"""Router (M4): from a plain-English question to a task in the registry, plus its parameters.

    "can I get from node 3 to node 17?"  ->  Route(task="connectivity", params={"source": 3, "target": 17}, ...)

The task decides everything else: the answer type, the verifier, (later) the tool family and the skill. So a wrong
route is worse than no route: it can put the wrong verifier on a correct answer. Hence `unknown`: when the router
isn't sure, or its parameters don't fit the task or the graph, the run gets all tools and no verifier (reported as
unverified). Never a guessed verifier.

Routers (compared in M4 on accuracy, latency and cost, eval/routing.py):
  - regex_route: patterns over the question text (the baseline, and the floor)
  - llm_route:   a small LLM calls a single `route` tool (ROUTE_TOOL). A tool call, not Ollama's `format` option:
                 with qwen3.5 and thinking off, Ollama 0.30 ignores `format` (checked 2026-10-09)
  - later: embeddings (nomic-embed-text), Jev (classification only)
The eval runner also has an "oracle" route: the dataset's own task and parameters (today's behavior).
"""
import json
import re
import string
from dataclasses import dataclass, replace
from typing import Annotated, Callable, Literal

import networkx as nx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from harness.brief import brief
from harness.prompts import ROUTER_PROMPT
from harness.tasks import TASKS
from harness.tools.graph_tools import _not_a_bool, describe_errors

UNKNOWN = "unknown"
PAIRS = (("source", "target"), ("u", "v"))  # the LLM sometimes puts a path's two nodes in u, v (dev run, r2)
MIN_CONFIDENCE = 0.6  # below this, a route becomes unknown (a first guess; set it from the routing eval, dev only)


def task_params(task: str) -> list[str]:
    """The parameters a task needs, read from its question template: "degree of node {node}?" -> ["node"]."""
    names = [field for _, field, _, _ in string.Formatter().parse(TASKS[task].question) if field]
    return list(dict.fromkeys(names))  # unique, in order ({node} appears twice in hop_max_degree)


@dataclass(frozen=True)
class Route:
    task: str  # a TASKS name, or UNKNOWN
    params: dict  # e.g. {"source": 3, "target": 17}; {} for whole-graph questions and for unknown
    confidence: float | None  # 0 to 1; None: the model didn't say (accepted, see check_route)
    router: str  # which router made it: "regex", "llm", "oracle", ...
    reason: str | None = None  # why it is unknown, when it is
    prompt_tokens: int = 0  # the router's own cost (0 for regex and oracle)
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def known(self) -> bool:
        return self.task != UNKNOWN


def unknown(route: Route, reason: str) -> Route:
    """The same route marked unknown (keeping its cost and the router's name, for the report)."""
    return replace(route, task=UNKNOWN, params={}, reason=reason)


def check_route(route: Route, graph: nx.Graph | None = None, min_confidence: float = MIN_CONFIDENCE) -> Route:
    """Accept a route only if it can be trusted with a verifier; otherwise return it as unknown, with the reason.

    Checks: a real task, confident enough (when a confidence was given: a missing one is accepted, since
    self-reported confidence isn't calibrated anyway), exactly the parameters the task needs (whole numbers; a node
    pair given as u, v counts as source, target and the other way round), k >= 1, and (when the graph is given)
    every node parameter is a node of G.
    """
    if not route.known:
        return route
    if route.task not in TASKS:
        return unknown(route, f"no task named {route.task!r}")
    if route.confidence is not None and route.confidence < min_confidence:
        return unknown(route, f"confidence {route.confidence:.2f} is below {min_confidence}")
    needed = task_params(route.task)
    given = dict(route.params)
    for want, other in (PAIRS, PAIRS[::-1]):  # a node pair in the other pair's fields: same nodes, same order
        if all(name in needed and given.get(name) is None for name in want) and all(given.get(n) is not None for n in other):
            given.update(zip(want, (given[n] for n in other)))
    missing = [name for name in needed if given.get(name) is None]
    if missing:
        return unknown(route, f"{route.task} needs {', '.join(missing)}")
    params = {name: given[name] for name in needed}  # extra parameters are dropped
    if any(isinstance(v, bool) or not isinstance(v, int) for v in params.values()):
        return unknown(route, f"parameters must be whole numbers: {params}")
    if params.get("k", 1) < 1:
        return unknown(route, "k must be at least 1")
    if graph is not None:
        absent = [v for name, v in params.items() if name != "k" and v not in graph]
        if absent:
            return unknown(route, f"node {absent[0]} is not in G")
    return replace(route, params=params)


# --- The LLM router's tool. Every parameter any task uses is an optional field; check_route keeps the ones the
# --- chosen task needs. Named fields rather than "the numbers in order": a paraphrase can reorder them.

def _none_words(value):
    # The model sometimes writes "None" / "null" / "" for a field it means to leave out: read that as left out.
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return _not_a_bool(value)


NodeParam = Annotated[int | None, BeforeValidator(_none_words)]


class RouteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal[tuple([*TASKS, UNKNOWN])]  # type: ignore[valid-type]  # the 14 task names, or "unknown"
    # Right after the task: last in the list, the model often left it out (dev run, 2026-10-09). Optional: in
    # second place it was still left out 17 times in 94 (r2), and a reply without it isn't a wrong reply.
    confidence: float | None = Field(None, ge=0, le=1, description="how sure you are of the task, from 0 to 1")
    node: NodeParam = Field(None, description="the node the question is about")
    u: NodeParam = Field(None, description="first node of an edge")
    v: NodeParam = Field(None, description="second node of an edge")
    source: NodeParam = Field(None, description="start node of a path")
    target: NodeParam = Field(None, description="end node of a path")
    via: NodeParam = Field(None, description="node a route must pass through")
    k: NodeParam = Field(None, description="a number of hops (edges)")


def route_tool() -> dict:
    """The schema the model sees. Optional fields are shown as plain integers to leave out when not needed:
    shown as anyOf [integer, null], qwen3.5 filled every one with the string "None" (dev run, 2026-10-09)."""
    from harness.tools.graph_tools import json_schema
    schema = json_schema(RouteArgs)
    for name, field in schema["properties"].items():
        if "anyOf" in field:
            kind = next(option["type"] for option in field["anyOf"] if option["type"] != "null")
            schema["properties"][name] = {"type": kind, "description": field["description"] + ("" if name == "confidence" else " (leave out if none)"),
                                          **{k: v for k, v in field["anyOf"][0].items() if k in ("minimum", "maximum")}}
    return {"type": "function",
            "function": {"name": "route", "description": "Report which task the question asks for, and its node ids.",
                         "parameters": schema}}


ROUTE_TOOL = route_tool()


# --- The routers.

# A node id, optionally after "node" / "vertex" (and "nodes" / "vertices" before a pair).
N = r"(?:nodes? |vertex |vertices )?"

# Most specific first: the first task with a matching pattern wins. "shortest ... via" before "shortest path",
# "shortest path" before "path", "edge between" / "adjacent" before "connected", "how many neighbors" (a number)
# before "neighbors of" (a list), every question about a node before the whole-graph ones.
REGEX_PATTERNS = [
    ("shortest_path_via", [rf"(?:shortest|quickest|fastest).*?from {N}(?P<source>\d+) to {N}(?P<target>\d+).*?"
                           rf"(?:through|via|visit|passing|pass) {N}(?P<via>\d+)"]),
    ("hop_max_degree", [rf"(?:at most|within) (?P<k>\d+) .*?{N}(?P<node>\d+).*?(?:highest|largest|most|max)"]),
    ("common_neighbors_max", [rf"(?=.*(?:in common|common neighbou?r|shar\w* .*neighbou?r))(?=.*?{N}(?P<node>\d+))"]),
    ("triangle_count", [rf"triangles?.*?{N}(?P<node>\d+)"]),
    ("shortest_path", [rf"(?:shortest|quickest|fastest|fewest).*?(?:from|between) {N}(?P<source>\d+) "
                       rf"(?:to|and|->) {N}(?P<target>\d+)"]),
    ("edge_existence", [rf"edge (?:between|from) {N}(?P<u>\d+) (?:and|to) {N}(?P<v>\d+)",
                        rf"{N}(?P<u>\d+) and {N}(?P<v>\d+),? (?:are |be )?(?:adjacent|directly connected)"]),
    ("connectivity", [rf"(?:path|route) (?:between|from) {N}(?P<source>\d+) (?:and|to) {N}(?P<target>\d+)",
                      rf"(?P<target>\d+) (?:is )?reachable from {N}(?P<source>\d+)",
                      rf"from {N}(?P<source>\d+) to {N}(?P<target>\d+)"]),
    ("node_degree", [rf"degree of {N}(?P<node>\d+)",
                     rf"how many (?:neighbou?rs|edges).*?{N}(?P<node>\d+)"]),
    ("connected_nodes", [rf"neighbou?rs of {N}(?P<node>\d+)", rf"adjacent to {N}(?P<node>\d+)"]),
    ("connected_components_count", [r"how many (?:connected )?components", r"number of (?:connected )?components"]),
    ("node_count", [r"how many (?:nodes|vertices)", r"number of (?:nodes|vertices)"]),
    ("edge_count", [r"how many (?:edges|links)", r"number of (?:edges|links)"]),
    ("mst", [r"minimum spanning (?:tree|forest)", r"\bmst\b"]),
    ("cycle_check", [r"\bcycles?\b", r"\bacyclic\b"]),
]
COMPILED = [(task, [re.compile(p, re.IGNORECASE) for p in patterns]) for task, patterns in REGEX_PATTERNS]


def regex_route(question: str) -> Route:
    """The baseline: the first pattern that matches decides the task; its named groups are the parameters.
    Confidence is 1.0 on a match (a regex can't be unsure) and 0.0 with no match (unknown)."""
    for task, patterns in COMPILED:
        for pattern in patterns:
            if match := pattern.search(question):
                params = {name: int(value) for name, value in match.groupdict().items() if value is not None}
                return Route(task, params, 1.0, "regex")
    return Route(UNKNOWN, {}, 0.0, "regex", reason="no pattern matched")


def llm_route(question: str, model: str, call_model: Callable | None = None) -> Route:
    """The LLM router: one call, thinking off, the model must call the `route` tool (ROUTER_PROMPT is the menu).
    A reply without a valid `route` call becomes unknown, with the reason; the call's cost is kept either way."""
    if call_model is None:
        from harness.models import call_model
    messages = [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": question}]
    reply = call_model(messages, [ROUTE_TOOL], model=model, think=False)
    extra = reply.get("extra") or {}
    empty = Route(UNKNOWN, {}, 0.0, "llm", prompt_tokens=extra.get("prompt_tokens", 0),
                  completion_tokens=extra.get("completion_tokens", 0), latency_s=extra.get("latency_s", 0.0))

    calls = [c for c in reply.get("tool_calls") or [] if c["function"]["name"] == "route"]
    if not calls:
        return unknown(empty, "the model did not call route" + (f": {brief(reply.get('content'))}"
                                                                if reply.get("content") else ""))
    arguments = calls[0]["function"]["arguments"]
    try:
        args = RouteArgs.model_validate(json.loads(arguments) if isinstance(arguments, str) else arguments)
    except json.JSONDecodeError:
        return unknown(empty, f"route arguments are not JSON: {brief(arguments)}")
    except ValidationError as e:
        return unknown(empty, f"bad route arguments: {describe_errors(e)}")
    params = {name: value for name, value in args.model_dump(exclude={"task", "confidence"}).items()
              if value is not None}
    return replace(empty, task=args.task, params=params, confidence=args.confidence, reason=None)
