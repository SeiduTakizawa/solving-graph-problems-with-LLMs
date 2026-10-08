"""run_python: the model writes code that runs in a sandbox against the graph (M3 step 5).

Backends live in this package (docker.py; a WASM backend can follow for a public, no-Docker setup). Modes:
    "tools"     the code can call our graph tools as functions, nothing else (networkx is blocked)
    "networkx"  also G (the graph) and nx: the escape hatch, for comparison
"""
STDOUT_SHOWN = 2_000  # characters of printed output the model sees; big values belong in `result` (handles)


def run_python_tool(mode: str) -> dict:
    extra = (" G (the graph, a networkx graph) and nx (networkx) are also available." if mode == "networkx"
             else " Only these functions are available: networkx cannot be imported.")
    return {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python code against G in a sandbox. The graph tools are available as functions with "
                           "the same arguments and results, e.g. get_neighbors(4) -> {\"neighbors\": [...]}." + extra +
                           " Variables persist between calls. Put the value you want back in a variable named "
                           "`result`; printed output is shown too (shortened if long). No network or files; "
                           "each call may run at most 10 seconds.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    }


def shown_reply(reply: dict) -> dict:
    """What the model sees: empty fields dropped, long printed output shortened."""
    out = {}
    stdout = reply.get("stdout") or ""
    if stdout:
        out["stdout"] = stdout if len(stdout) <= STDOUT_SHOWN else (
            stdout[:STDOUT_SHOWN] + f"... ({len(stdout)} characters; put large values in `result` instead of printing them)")
    if reply.get("result") is not None:
        out["result"] = reply["result"]
    for key in ("error", "note"):
        if reply.get(key):
            out[key] = reply[key]
    return out or {"result": None, "note": "The code ran without printing anything or setting `result`."}
