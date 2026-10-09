"""Every piece of text the harness itself says to the model."""

# Logged in every run's run_start, so results can be traced back to the exact wording the model saw.
# Bump it whenever model-facing text changes: this file, the submit_answer / cannot_answer / read_result texts, or
# any tool description in harness/tools/graph_tools.py. Add a line below saying what changed.
#   v1: everything up to and including the M2 pilots (commit c5a41e5, 2026-10-07). Runs without the field are v1,
#       except the M3 runs made on the MacBook (2026-10-07/08, e.g. claude_code_mcp_demo): those predate the field
#       and sit between v1 and v3; their exact prompts are in the git history of that day.
#   v2: v1 + distances_from and neighborhood (GPU-machine commit c36824b, kept on branch backup-gpu-2026-10-07).
#       Used by m2_large_verify, m2_large_noverify and claude_code_vs_harness_large.
#   v3: the merge: M3 (Pydantic schemas, has_path / is_bipartite / topological_sort, result handles and read_result,
#       bounded messages, shorter missing-node errors, waypoint question) + distances_from and neighborhood.
#   v4: run_python's description: the tool functions return plain values (get_neighbors(4) -> [2, 5]), errors
#       raise ValueError, don't redefine them. (Code-mode tools now really do this: code_value in sandbox/runner.py.)
#   v5: CODE_HINT, a strategy hint added to the system prompt only when run_python is on and code_hint is set
#       (its use is logged as code_hint in run_start); the combine task family's question wording.
#   v6: EMPTY_REPLY, its own nudge for a reply with no text and no tool call (qwen3.5 found answers and then went
#       silent; qwen3 said "let me call get_neighbors" only in its thinking); run_python's description shows what
#       graph_info() returns; a redefined tool function is restored with a note (sandbox/runner.py).
#   v7: compaction (harness/compaction.py): near the context limit, older tool results and run_python code are
#       shortened with SHORTENED_RESULT / SHORTENED_CODE. Runs that never get near the limit see the same text as v6.
#   v8: M4. connected_components_count and triangle_count answers come with evidence (the components / the
#       triangles, answer types number_with_components / number_with_triangles); the `any` answer type for questions
#       the router can't place; MORE_TOOLS (the more_tools escape hatch, only with --tools hybrid / agent). Tool
#       exposure by task (--tools task) changes which tools are shown, not their text.
#   v9: per-step output cap (models.OLLAMA_NUM_PREDICT) with its own nudge THOUGHT_TOO_LONG; a run_python list
#       result of 20+ numbers also gets a handle (result_handle) so it can be submitted without copying; run_python's
#       description says the tool functions need no import; skills (SKILL_PREFIX + harness/skills/<task>.md), only
#       with eval.runner --skills. Rejections of missing / partial count evidence say how to send a long list by
#       handle (verifiers.LONG_LIST_HOW).
PROMPT_VERSION = "v9"

SYSTEM_PROMPT = (
    "You answer questions about a graph G. You cannot see G directly; use the tools to inspect it. "
    "When you know the answer, call submit_answer. "
    "If the question cannot be answered with the available tools, call cannot_answer instead of guessing."
)

# Sent when the model replies with plain text instead of calling a tool.
NUDGE = (
    "You did not call a tool. Use the tools to inspect G, call submit_answer with your final answer, "
    "or call cannot_answer if the question cannot be answered."
)

# Sent when the reply is empty: no text and no tool call (often the plan was only in the model's thinking).
EMPTY_REPLY = (
    "Your reply was empty: no text and no tool call. If you already know the answer, call submit_answer now. "
    "Otherwise make your next tool call, or call cannot_answer if the question cannot be answered."
)

# Sent when a reply was cut off by the output cap (models.OLLAMA_NUM_PREDICT): the model reasoned for too long,
# usually trying to work the answer out in its head instead of with the tools.
THOUGHT_TOO_LONG = (
    "Your reply was cut off: you thought for too long without calling a tool. Don't work it out in your head: "
    "make one tool call or one run_python call now, or call submit_answer if you already know the answer."
)

# Compaction (harness/compaction.py): what an older tool result / older code becomes when the context gets full.
SHORTENED_RESULT = "(shortened; call the tool again for the full result)"
SHORTENED_CODE = "# (older code shortened to save space)"

# Sent when the plain text looks like a tool call the model wrote out by hand (it was not executed).
FORMAT_ERROR = (
    "Your last message contains a tool call written as plain text, so it was NOT executed. "
    "Do not write tool calls in your reply; call the tool through the function-calling interface."
)

# Strategy hint ("skill"), added to the system prompt when run_python is offered and AgentConfig.code_hint is on.
# Without it the model never chose code when a direct tool existed (0/5 in the code_B runs), even where a loop
# over many nodes would be faster and safer than dozens of tool calls.
CODE_HINT = (
    "How to work: use a direct tool for a single lookup. When the answer needs many lookups (a value for every "
    "neighbor, every node, or every pair of nodes) or a comparison over many results, write ONE run_python call "
    "that calls the tool functions in a loop and puts the answer in `result`: it is faster and avoids mistakes "
    "in long lists. You can combine both: a tool call to find the candidates, then code to check them all."
)

# Router (harness/router.py): the system prompt of the LLM router, a menu of the tasks. Not part of the agent's
# prompt, so PROMPT_VERSION doesn't change with it; router runs log their own ROUTER_VERSION.
ROUTER_VERSION = "r3"  # r2: optional fields as plain integers, confidence second (r1: 44 format misses)
# r3: confidence optional, a node pair in the other pair's fields accepted (r2: 17 missing confidences)
ROUTER_TASKS = {
    "node_count": "how many nodes G has",
    "edge_count": "how many edges G has",
    "node_degree": "how many neighbors (edges) one node has: a number",
    "connected_nodes": "which nodes are the neighbors of one node: a list",
    "edge_existence": "whether two nodes are directly connected by an edge",
    "connectivity": "whether two nodes are connected by any path (possibly through other nodes)",
    "connected_components_count": "how many connected components (separate pieces) G has",
    "cycle_check": "whether G contains a cycle",
    "shortest_path": "a shortest path (route) between two nodes",
    "mst": "a minimum spanning tree (or forest) of G",
    "shortest_path_via": "a shortest route from one node to another that must pass through a third node",
    "hop_max_degree": "among the nodes within k hops of a node, the one with the highest degree",
    "common_neighbors_max": "the node sharing the most neighbors with a given node",
    "triangle_count": "how many triangles include a node",
}
ROUTER_PROMPT = (
    "You route questions about a graph G to the task that answers them. Call the `route` tool once with the "
    "task and the node ids the question mentions, in the right fields. Tasks:\n"
    + "\n".join(f"- {name}: {text}" for name, text in ROUTER_TASKS.items())
    + "\n- unknown: anything else, or a question that matches none of these exactly. "
    "If you are not sure, say so with a low confidence."
)

# Skills (harness/skills/): a task's playbook follows the system prompt under this heading.
SKILL_PREFIX = "\n\nHow to answer this kind of question:\n"

# The more_tools escape hatch (AgentConfig.more_tools): the agent sees one task's tools, or none, and asks for more.
MORE_TOOLS_DESCRIPTION = ("Add more graph tools to your tool list, one category at a time: {categories}. "
                          "Use it when the tools you have are not enough.")
