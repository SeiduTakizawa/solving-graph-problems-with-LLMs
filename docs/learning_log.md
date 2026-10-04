# Learning log: harness engineering

One entry per milestone or experiment round: what was built, what broke, what it taught.

---

## 2026-10-04: First agent loop (pre-M1)

**Model:** `ollama_chat/qwen3:8b` via LiteLLM, local, default temperature.
**Graph:** `data/graphs/er/small/0.txt` (7 nodes, 5 edges, a forest with no triangles).

### What was built
- `harness/hello_agent.py`: the agent loop: ask the model, run the tools it calls, feed results back, repeat.
- `harness/tools/graph_tools.py`: `get_neighbors`, `count_edges`, plus `run_tool`, which never raises.
- `harness/trace.py`: append-only JSONL trace (`run_start`, `model_call`, `tool_call`, `nudge`, `run_end`)
  with tokens and latency per call.
- Tests with a scripted fake model (loop) and against networkx on all 300 dataset graphs (tools).

### Key ideas
- **An agent is a loop over a message list.** The model has no memory; every call re-sends the whole history.
  Seen in a trace: prompt tokens went 249 → 407 between step 1 and step 2, so each extra step makes
  every later step more expensive.
- **The model never sees the graph.** It only sees what tools return (design principle 1).
- **Tool descriptions are the model's only manual.** A schema that doesn't match the function confuses it;
  a test now checks that schemas and function signatures agree.
- **Tools never crash the agent.** Errors go back to the model as `{"error": ...}`. The `EdgeView` bug
  (returning `graph.edges()` instead of a count) crashed the whole run before this safety net existed.

### Failures found by poking the naive loop
| Experiment | What the model did | Why | Lesson |
|---|---|---|---|
| "degree of node 99" (no such node) | Read the tool error, then submitted `0` | `submit_answer` only accepts an integer, so there was no way to say "invalid question" | The answer schema shapes behaviour; give an honest exit |
| "how many edges" with only `get_neighbors` | Submitted `0` immediately, no tool calls | No suitable tool; cheap models guess instead of doing tedious work | Tool coverage matters more than model effort |
| "find the biggest clique" | Said honestly "tools insufficient", was nudged twice, then guessed `3` (true: 2) | The generic nudge ("use tools or submit") pressured it into guessing | **The harness itself can cause hallucinations** |
| "neighbors of node 10990" | Wrote `submit_answer {"answer": 0} </tool_call>` as plain text, 9 times, until `max_steps` | qwen3's tool-call format leaked into its text; the generic nudge didn't say what was wrong | Feedback must be specific; loops must be detected |

### Fixes
1. **`cannot_answer(reason)` tool:** an honest exit, logged as status `cannot_answer`, not as a wrong answer.
   The system prompt says to use it instead of guessing.
2. **Specific nudges:** a plain-text reply that looks like a hand-written tool call (`<tool_call>` tags, or a
   tool name followed by `{`) gets `FORMAT_ERROR` ("your tool call was not executed..."); other text gets
   `NUDGE`, which now also offers `cannot_answer`. Prose that only *mentions* tool names is not flagged.
3. **Loop detection:** the same reply (text + tool calls, ignoring call ids) 3 times in a row ends the run
   with status `loop_detected`.
4. `run_agent` returns a `RunResult(answer, status, messages, reason)`. Statuses: `submitted`,
   `cannot_answer`, `max_steps`, `loop_detected`.

### Results after the fixes (same questions, real model)
| Question | Before | After |
|---|---|---|
| biggest clique | guessed `3` after 2 nudges | `cannot_answer` ("no clique tool") at step 1 |
| neighbors of node 10990 | text tool call ×9, `max_steps` | `cannot_answer` ("node does not exist") at step 2 |
| degree of node 4 | `2` ✅ | `2` ✅ |
| degree of node 99 | `0` | 4 runs: `cannot_answer` 2×, `0` 2× (**improved, not solved**) |

**Repeated runs matter:** the node-99 question gave different outcomes across runs. One run is not a result
(`CLAUDE.md`: at least 3 runs per configuration).

### Open issues
- **Per-task answer schemas** (design principle 4): `submit_answer` takes one integer, but "neighbors" or
  "biggest clique" need a list of nodes. This needs task families, so it goes in M2.
- Node 99 still sometimes gets `0`. Possible next steps: a clearer error from the tool, or a verifier
  that rejects answers about nodes that don't exist.
- No `temperature` is set (model default). Decide before running real experiments.
- `FORMAT_ERROR` hasn't been seen to recover a real run yet: after the fixes, qwen3 called tools properly.
