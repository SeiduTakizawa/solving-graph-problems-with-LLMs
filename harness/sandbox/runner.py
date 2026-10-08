"""Runs INSIDE the sandbox container: executes the model's code against one graph.

Started once per question by the host (harness/sandbox/docker.py), then reads one JSON request per line from stdin
and writes one JSON reply per line to stdout:

    {"code": "..."}  ->  {"stdout": "...", "result": <value of `result`>, "error": "..." or null}

Variables persist between calls (like a notebook). The graph tools are plain functions: get_neighbors(4) or
get_neighbors(node=4) both work and return plain values, not the tools' dicts (see code_value). Mode "tools": only those functions,
networkx can't be imported. Mode "networkx": G (the graph) and nx are available too.
"""
import builtins
import contextlib
import io
import json
import sys
import traceback

import networkx as nx

from harness.tools.graph_tools import TOOLS, run_tool

STDOUT_LIMIT = 20_000  # characters kept per call; the host shortens further before the model sees it


def code_value(name: str, result: dict):
    """A tool's result as code should see it: a plain value, like networkx would give, and errors as exceptions.

    Why: as JSON for the model, a tool result is a dict ({"neighbors": [2, 5]}). But in code the model writes
    `for n in get_neighbors(4):`, which loops over the dict's *keys* and silently does the wrong thing, and an
    {"error": ...} dict just flows on as if it were a result. That's why every forced-code run failed (C1, 0/5).

    Rules:
      - an error result raises ValueError(<the error message>), so the code stops loudly at the right line
      - tools with one main value return just that value      get_neighbors(4) -> [2, 5],  degree(4) -> 2
      - "maybe" tools return their evidence, or None if there is none
                                                              shortest_path(0, 5) -> [0, 2, 4, 5]  or None
      - tools whose result has several equal parts (graph_info, is_bipartite) keep their dict
    """
    # 1. Errors stop the code loudly, at the line that called the tool.
    if "error" in result:
        raise ValueError(result["error"])

    # 2. Tools with one main value return just that value.
    single_value_keys = {
        "get_neighbors": "neighbors",
        "degree": "degree",
        "has_edge": "has_edge",
        "connected_components": "components",
        "minimum_spanning_tree": "edges",
        "distances_from": "layers",
        "neighborhood": "nodes",
    }
    if name in single_value_keys:
        return result[single_value_keys[name]]

    # 3. "Maybe" tools: their evidence, or None when there is none (so `if shortest_path(a, b):` works).
    evidence_keys = {"shortest_path": "path", "has_path": "path", "has_cycle": "cycle", "topological_sort": "order"}
    if name in evidence_keys:
        return result.get(evidence_keys[name])

    # 4. Several equal parts (graph_info, is_bipartite): the dict stays.
    return result


def tool_function(graph, name):
    fields = list(TOOLS[name].args.model_fields)

    def call(*args, **kwargs):
        kwargs.update(zip(fields, args))  # positional arguments in the order of the tool's fields
        return code_value(name, run_tool(graph, name, kwargs))

    call.__name__ = name
    call.__doc__ = TOOLS[name].description
    return call


def safe_builtins(allow_networkx: bool) -> dict:
    """In "tools" mode the code must not reach networkx (it would bypass the tools and their evidence).
    The tools themselves already imported it, so `import networkx` is blocked at the import statement."""
    allowed = dict(vars(builtins))
    if allow_networkx:
        return allowed
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("networkx", "harness"):
            raise ImportError(f"'{name}' is not available here: use the graph tool functions.")
        return real_import(name, *args, **kwargs)

    allowed["__import__"] = guarded_import
    return allowed


def describe_error(error: BaseException) -> str:
    """The error as the model needs it: the line of *its* code that failed and the message, not our internals."""
    frames = [f for f in traceback.extract_tb(error.__traceback__) if f.filename == "<run_python>"]
    where = f"line {frames[-1].lineno}: " if frames else ""
    return f"{where}{type(error).__name__}: {error}"


def jsonable(value):
    """Tuples and sets become lists, anything else unknown becomes its repr."""
    try:
        return json.loads(json.dumps(value, default=lambda v: sorted(v) if isinstance(v, (set, frozenset)) else repr(v)))
    except (TypeError, ValueError):
        return repr(value)


def restore(namespace: dict, protected: dict) -> str | None:
    """Put back any tool function (or G / nx) the code replaced, and say so.

    Why: variables persist between calls, so a redefinition would stick for the rest of the question. A model wrote
    `def get_neighbors(n): return get_neighbors(n)` (infinite recursion), and every later call crashed too.
    """
    replaced = sorted(name for name, value in protected.items() if namespace.get(name) is not value)
    namespace.update(protected)
    if not replaced:
        return None
    return (f"You redefined {', '.join(replaced)}; the original function was restored for your next calls. "
            "Use the tool functions as they are and pick other names for your own helpers.")


def main(graph_file: str, directed: bool, mode: str) -> None:
    graph = nx.read_adjlist(graph_file, nodetype=int, create_using=nx.DiGraph if directed else nx.Graph)
    protected = {name: tool_function(graph, name) for name in TOOLS}
    if mode == "networkx":
        protected.update(G=graph, nx=nx)
    namespace = dict(protected)
    namespace["__builtins__"] = safe_builtins(allow_networkx=mode == "networkx")

    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        code = json.loads(line)["code"]
        out, error = io.StringIO(), None
        namespace.pop("result", None)
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(code, "<run_python>", "exec"), namespace)
        except BaseException as e:  # noqa: BLE001 - report every failure to the model, keep serving
            error = describe_error(e)
        reply = {"stdout": out.getvalue()[:STDOUT_LIMIT], "result": jsonable(namespace.get("result")), "error": error,
                 "note": restore(namespace, protected)}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] == "directed", sys.argv[3])
