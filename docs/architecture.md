# Harness architecture (as built)

How the graph harness works **today**. `CLAUDE.md` has the plan and the milestones; this file describes the code
as it is, and is updated when the architecture changes. The other files in `docs/` describe the paper's original
prompting pipeline (`graph_reasoning/`), which the harness does not use.

Last updated: 2026-10-08 (M3 steps 1–5: … MCP server, `run_python` sandbox; task `shortest_path_via` + trajectory analysis).

## One question, end to end

```
eval/runner.py                                   harness/ (the agent never imports eval/)
──────────────                                   ────────
load question + graph (eval/tasks.py)
look up the task (harness/tasks.py) ──────────▶  Task: question template, answer type, verifier
                                                 │
run_agent(question, graph, answer_type, verify) ▶ harness/loop.py
                                                 │  messages = [system prompt, question]
                                                 │  tools    = graph tools + submit_answer + cannot_answer
                                                 │  repeat up to max_steps:
                                                 │    reply = call_model(messages, tools)      models.py (LiteLLM)
                                                 │    drop the model's hidden thinking from the history
                                                 │    tool written as text? → rescue it        parsing.py
                                                 │    same reply 3× in a row? → stop (loop_detected)
                                                 │    for each tool call:
                                                 │      arguments not JSON → error back to the model
                                                 │      graph tool  → run_tool (validated)     tools/graph_tools.py
                                                 │                    → big values → handles   tools/handles.py
                                                 │      read_result → page through a handle
                                                 │      submit_answer → handles resolved
                                                 │                    → shape check            answers.py
                                                 │                    → verifier               verifiers.py
                                                 │                    → pass: done | fail: error back, try again
                                                 │      cannot_answer → done
                                                 │    no tool call → nudge / format error      prompts.py
                                                 │  every step logged                          trace.py (JSONL)
                                                 ▼
RunResult(answer, status, evidence, steps, tokens, rescued, rejected, ...)
grade against the reference (eval/tasks.py)  ← independent of the harness verifiers
write one row to results.jsonl
```

## Modules

### `harness/` (the agent; never imports `eval/` or `graph_reasoning/`)
| file | job |
|---|---|
| `loop.py` | `run_agent`: the agent loop. `AgentConfig` (model, max steps, rescue, tools offered, handle limit), `RunResult`. |
| `models.py` | `call_model`: one LiteLLM call, returns the reply plus tokens and latency under `extra`. |
| `tools/graph_tools.py` | The graph tools. `TOOLS`: name → function + Pydantic args model + description. `GRAPH_TOOLS` (the schemas the model sees) is generated from it; `run_tool` validates with it. |
| `tools/handles.py` | Result handles: `HandleStore` (one per run), `read_result`. |
| `tools/mcp_server.py` | The same tools over MCP, for other harnesses (Claude Code, generic agents). One process per graph, stdio. |
| `sandbox/` | `run_python`: the `run_python` tool schema and how replies are shown (`__init__.py`), the Docker backend (`docker.py`), the code that runs inside the container (`runner.py`), the image (`Dockerfile`). |
| `brief.py` | `brief()`: short versions of values repeated back to the model (first items + size). |
| `answers.py` | Answer types (Pydantic types) → `submit_answer` schema and `check_answer`; `cannot_answer`; evidence fields. |
| `verifiers.py` | Checks of the final answer, in plain Python. Evidence only, never re-solving (see below). |
| `tasks.py` | Task registry: question template, answer type, verifier per task. No ground truth. |
| `parsing.py` | Spotting and rescuing tool calls the model wrote as text. |
| `prompts.py` | Every piece of text the harness says to the model, and `PROMPT_VERSION` (logged in every `run_start`; bump it, with a changelog line, whenever model-facing text changes). |
| `trace.py` | Append-only JSONL trace, one line per event. |
| `ask.py` | One question about one graph from the command line. |

### `eval/` (measurement; may import `harness/`, never the reverse)
| file | job |
|---|---|
| `tasks.py` | Dataset loading (questions, graphs, dev/test split), generated questions for tasks not in the dataset (`GENERATED`, fixed seed per graph), reference answers, **grading** (written independently of the verifiers). |
| `runner.py` | Runs the agent over dataset questions; one row per answer in `results.jsonl`, traces in `traces.jsonl`; progress with ETA. |
| `analysis/report.py` | Tables (accuracy, 95% bootstrap CI, strict accuracy, checked share, tokens, time) and the accuracy-vs-tokens plot. |
| `analysis/trajectory.py` | How a run got to its answer (multi-step tasks): ideal / extra calls / alternative path / right calls but wrong / wrong path, plus detour flags. |

## Key designs

**Tools** (`graph_tools.py`). Each tool is a plain function `tool(graph, **args) -> dict` that never raises:
bad input returns `{"error": "..."}`. Arguments are a Pydantic model (`NodeArgs`, `EdgeArgs`, `PathArgs`, `NoArgs`):
it generates the schema the model reads and validates what the model sends (node ids accept `"3"`, reject `true`;
unknown arguments are errors). Tools return **evidence** with their answer where they can (a path, a cycle, a
2-coloring or an odd cycle, a topological order), so checkers can verify answers later.

Current tools: `get_neighbors`, `degree`, `has_edge`, `graph_info`, `shortest_path`, `connected_components`,
`has_cycle`, `minimum_spanning_tree`, `has_path`, `is_bipartite`, `topological_sort`, `distances_from` (nodes in
layers by distance) and `neighborhood` (nodes within k hops), plus `read_result` once a handle exists. The last two
are building blocks for multi-step questions (farthest node, k-hop counts): one call instead of dozens. New tools are
appended, so the text of earlier tools stays the same.

**Answers** (`answers.py`). `submit_answer` is built per answer type: `number`, `yes_no`, `node_list`,
`edge_list`, and `yes_no_with_path` / `yes_no_with_cycle`, where a "yes" carries the path or cycle as an extra
field (evidence) that goes to the verifier. Shape errors go back to the model.

**Verifiers** (`verifiers.py`, per task in `tasks.py`). Code, not a model. They check the evidence in an answer
and never re-solve the question (that would be an oracle). Full: shortest_path, mst, shortest_path_via. Partial: connectivity and
cycle_check (a "yes" needs a real path / cycle; a "no" is accepted unseen), connected_nodes (listed nodes must be
real neighbors; a missing one can't be seen). None: counts, degree, edge existence.

**Result handles** (`handles.py`). A value with more than 200 numbers is stored per run; the model sees
`{"handle": "result_1", "length": 9932, "first_5": [...]}`. A list of big items (components) gets one handle per
item. Once a handle exists, `read_result(handle, offset, limit)` is offered and `submit_answer` also accepts a
handle string (`"result_1"`, or `["result_1"]`, which models send when the schema says "array"). Before that, the
tool list is unchanged, so the current dataset (max 98 numbers) never sees handles.

**MCP server** (`tools/mcp_server.py`, official `mcp` SDK, low-level `Server`). Other harnesses get exactly our
tools: the tool list is built from `GRAPH_TOOLS` (same names, descriptions and schemas), calls go through
`run_tool` (same validation and errors, flagged `is_error`), big results become handles. The SDK's own argument
validation is left off so errors match our loop. stdio transport: the client launches one server process per
question with `--graph <file>`, so handles never leak between questions. Differences from our loop: `read_result`
is listed from the start; `submit_answer` isn't served yet (decided in M6). Our loop does not use MCP: it calls
`run_tool` directly.

```
graph_tools.TOOLS ──┬── our loop: run_tool() in-process
                    └── mcp_server.py ── stdio ── Claude Code / LangChain / any MCP client
```

**`run_python` sandbox** (`harness/sandbox/`, off by default; `AgentConfig(python="tools" | "networkx")`, runner
`--python`). The model's code runs in a Docker container, one per question, started on the first call (~0.3 s) and
reused (~0.3 ms per call; variables persist). Inside, our graph tools are plain functions with the same arguments and
results (`get_neighbors(4)`); in `"tools"` mode networkx can't be imported, so code composes our tools instead of
bypassing them; `"networkx"` mode also gives `G` and `nx` (the escape hatch, for comparison). The model gets back
`result` (big values become handles), printed output (shortened), and errors as "line N: Error: message".
Lockdown: no network, read-only filesystem except a small /tmp, 512 MB memory, 1 CPU, 64 processes, no
capabilities, non-root, 10 s per call (then the container is killed and restarted, and the reply says variables are
lost). Mounted: only the graph file and `harness/` (read-only), never `eval/` or `results/`, which hold reference
answers. The image pins the project's networkx/pydantic versions; its tag is a hash of the Dockerfile.
Backends are swappable: a WASM backend (no Docker needed) is the plan for a public release.

```
loop ── run_python(code) ──▶ docker exec (warm container) ──▶ runner.py: exec(code) with tool functions
     ◀── {"result" | handle, "stdout", "error"} ◀──────────────────────────────┘
```

**Nothing the model reads grows with the graph.** Tool results go through result handles; everything else that
repeats a value back (a wrong answer, a cycle, an exception text) goes through `harness/brief.py` (`brief()`: the
first items and the size), and errors describe G by count and id range, never by listing nodes.
`tests/test_message_size.py` checks all of these on a 10,000-node graph.

**Robustness in the loop.** Text tool calls are rescued (lenient) or answered with a format error (strict),
broken JSON arguments and crashing tools come back as errors, the model's hidden thinking is not re-sent, and
identical replies 3× in a row end the run.

**Logging.** Every run writes `run_start`, `model_call` (content, tool calls, thinking, tokens, latency),
`tool_call` (arguments, result as the model saw it), `rescued_tool_call`, `verifier_rejected`, `nudge` and
`run_end` to `traces.jsonl`. `results.jsonl` has one row per answer. Every reported number comes from these files.

## Adding things

- **A tool:** a function in `graph_tools.py` returning a dict (errors as `{"error": ...}`), an args model, an entry
  appended to `TOOLS`, tests against networkx in `tests/test_graph_tools.py`. The schema is generated.
- **A task:** an entry in `harness/tasks.py` (template, answer type, verifier or `None`), parameter parsing,
  reference answer and grading in `eval/tasks.py`.
- **A verifier:** `verify(graph, params, answer, **evidence) -> error message or None` in `verifiers.py`, tests
  that valid answers pass and each kind of broken answer is caught.

## Not built yet

Router and skills (M4), interactive CLI (M4.5),
escalation (M5), benchmark adapters and baselines (M6).
