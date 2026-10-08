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
PROMPT_VERSION = "v3"

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

# Sent when the plain text looks like a tool call the model wrote out by hand (it was not executed).
FORMAT_ERROR = (
    "Your last message contains a tool call written as plain text, so it was NOT executed. "
    "Do not write tool calls in your reply; call the tool through the function-calling interface."
)
