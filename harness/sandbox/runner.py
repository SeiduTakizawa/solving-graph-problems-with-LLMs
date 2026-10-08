"""Runs INSIDE the sandbox container: executes the model's code against one graph.

Started once per question by the host (harness/sandbox/docker.py), then reads one JSON request per line from stdin
and writes one JSON reply per line to stdout:

    {"code": "..."}  ->  {"stdout": "...", "result": <value of `result`>, "error": "..." or null}

Variables persist between calls (like a notebook). The graph tools are plain functions: get_neighbors(4) or
get_neighbors(node=4) both work and return the same dicts as the tools. Mode "tools": only those functions,
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


def tool_function(graph, name):
    fields = list(TOOLS[name].args.model_fields)

    def call(*args, **kwargs):
        kwargs.update(zip(fields, args))  # positional arguments in the order of the tool's fields
        return run_tool(graph, name, kwargs)

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


def main(graph_file: str, directed: bool, mode: str) -> None:
    graph = nx.read_adjlist(graph_file, nodetype=int, create_using=nx.DiGraph if directed else nx.Graph)
    namespace = {name: tool_function(graph, name) for name in TOOLS}
    if mode == "networkx":
        namespace.update(G=graph, nx=nx)
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
        reply = {"stdout": out.getvalue()[:STDOUT_LIMIT], "result": jsonable(namespace.get("result")), "error": error}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] == "directed", sys.argv[3])
