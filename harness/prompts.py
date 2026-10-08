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
PROMPT_VERSION = "v6"

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
