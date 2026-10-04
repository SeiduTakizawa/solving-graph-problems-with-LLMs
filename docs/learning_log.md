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

---

## 2026-10-04: First experiment, node degree (small graphs, dev)

**Setup:** `uv run python -m eval.runner --size small --split dev --n 100 --runs 3 --name degree_small_dev_v1`.
100 node-degree questions from dev graphs 0–49 (2 per graph), asked as "What is the degree of node X?",
with no edge list in the prompt. Tools: `get_neighbors`, `count_edges`. Model `ollama_chat/qwen3:8b`.
Results in `results/harness_runs/degree_small_dev_v1/`. Stopped early: runs 1 and 2 complete, run 3 has 12/100.

**Dev/test split:** graphs 0–49 are dev (build and tune), 50–99 are test (only for final results).
The split is by graph, so no test graph is ever seen during development.

### Results
| Run | Accuracy | Notes |
|---|---|---|
| 1 | 99% (99/100) | 1 `loop_detected` |
| 2 | 100% (100/100) | |
| 3 | 100% (12/12, incomplete) | |

- 0 submitted-but-wrong answers. The only failure was a formatting problem, not a reasoning one.
- Per question: 2 steps (`get_neighbors` → `submit_answer`), ~930 prompt + ~420 completion tokens, ~4.3 s.

### The one failure (run 1, graph 14, node 3)
`get_neighbors(3)` → `[0, 1, 2, 4, 5, 6]`, then the model wrote `submit_answer {"answer": 6} </tool_call>`
**as text**, three times, despite `FORMAT_ERROR`, and was stopped by loop detection. **The answer, 6, was correct.**
- Loop detection worked: stopped at step 4 instead of 10.
- `FORMAT_ERROR` did not help: first real evidence that the specific message doesn't fix this qwen3 failure.
- Open design question: rescue tool calls written as text (lenient: better accuracy) or count them as
  failures (strict: measures the model more honestly)? If rescued, log it as a separate event so both
  numbers can be reported.

### Takeaways
- Node degree on small graphs is easy for tools + a cheap model: one tool call does it. The interesting cases
  will be harder tasks and larger graphs.
- Runner fixes: progress printing is now flushed (the log file stayed empty while running), and the summary
  marks incomplete runs.

---

## 2026-10-04: All 5 M1 tools, plus a first look at a code-writing agent

### Tools
`get_neighbors`, `graph_info` (replaces `count_edges`), `shortest_path`, `connected_components`, `has_cycle`,
each tested against networkx on all 300 dataset graphs. Notes from writing them:
- networkx signals "nothing found" by raising (`NetworkXNoPath`, `NetworkXNoCycle`); tools turn that into a
  normal answer (`{"reachable": False}`, `{"has_cycle": False}`), never an error.
- Bug caught by the tests: `graph.number_of_nodes` without `()` returns the method, not the number.

With the real model, each question went straight to the right tool: components → `connected_components`,
path length → `shortest_path`, cycle → `has_cycle`. But for the cycle question it submitted `0` for "no",
because `submit_answer` only accepts an integer. **Per-task answer types are now blocking.**

### Code-writing agent vs tool harness (one question, anecdotal)
"Is there a cycle in G?" on `data/graphs/er/small/0.txt`. Claude Code (`claude -p --model sonnet`, Sonnet 5.5),
allowed only to run the project's Python and read files, with the graph given as a file.

| | Our harness (qwen3:8b, local) | Claude Code (Sonnet 5.5) |
|---|---|---|
| Correct | yes (`has_cycle` → false, submitted as `0`) | yes ("No") |
| Model calls | 2 | 2 |
| Time | 3.0 s | 4.1 s |
| Input tokens | 1,093 | 38,660 (29,071 cache read + 9,585 cache write) |
| Output tokens | 276 | 196 |
| Cost | $0 (local) | $0.046 (reported API-equivalent) |

Sonnet wrote one script, `nx.is_forest(G)` plus node/edge counts, and got it right first try.
- **Overhead:** about 35× more input tokens, almost all from Claude Code's own system prompt and tool
  definitions, sent on every call (cached, but still there). A general harness pays this on every task.
- **My prediction was wrong on output tokens:** code was not more output-heavy; qwen3's thinking tokens
  dominate its output. No retries either: strong models rarely fumble a one-liner. The retry cost
  should show up with cheap models writing code, or with harder tasks.
- The two approaches put the knowledge in different places: in the model (Sonnet knows networkx) vs in
  the tools (`has_cycle` does it for qwen3).
- **Not a fair comparison:** different models, one run, a tiny graph. The real version is M6: same model,
  same tools, many questions.

---

## 2026-10-04: Per-task answer types

`run_agent(..., answer_type=...)` with `number`, `yes_no` or `node_list`. `submit_answer` is built per type
(the model sees e.g. `answer: boolean`), and `check_answer` validates the submitted value. A wrong shape goes
back to the model as an error ("must be true or false") instead of ending the run. In Python `True` is an
`int`, so `number` rejects booleans and `yes_no` rejects 0/1.

Real model: cycle → `False` (was `0`), path 0→5 → `[0, 2, 4, 5]`, components → `2`. Neighbors of node 4 lost
the correct `[2, 5]` to the text-tool-call bug again (3rd time overall, after graph 14 and node 10990).

**Open decision:** rescue tool calls written as text (lenient, logged as rescued, so both numbers can be
reported) or keep counting them as failures (strict).

---

## 2026-10-05: Task registry, has_edge ablation, thinking re-sent in the history

### Task registry
`harness/tasks.py` holds what the agent may use per task: question template, answer type, verifier.
`eval/tasks.py` holds what only the grader uses: dataset parameter parsing, reference answers, grading
(any shortest path counts, neighbors in any order, `True` never counts as `1`). The harness never imports
eval code, so no ground truth can leak into a run. Smoke test with qwen3:8b: 18/18 correct over all 9 tasks.

### Ablation: does a direct `has_edge` tool help? (`--without-tool has_edge`)
Same 5 `edge_existence` questions (small, dev), 1 run each. Results in
`results/harness_runs/has_edge_ablation_{without,with}/`.

| | without `has_edge` | with `has_edge` |
|---|---|---|
| correct | 5/5 | 5/5 |
| output tokens (mostly thinking) | 2,256 | **282** (8× fewer) |
| prompt tokens | 4,200 | 1,231 (partly inflated, see below) |
| time per question | 23.7 s | **3.1 s** (7.6× faster) |

Without the tool, qwen3 spends 1,500–2,700 thinking tokens working out that `get_neighbors` answers the
question, before its first call. With it: 275–287 every time. **A well-fitting tool cuts cost ~8× at the
same accuracy.** Small sample, but the gap is far larger than the noise.

### Bug found: the model's thinking was being re-sent
qwen3 replies carry their hidden thinking in `reasoning_content`, and the loop kept it in the history, so
it was sent back (and re-read) on every later step. Fix: pop it before appending the reply, and log it as
`reasoning` in the `model_call` event instead (it wasn't logged at all before).
Same question (edge 3–6, without `has_edge`): step-2 prompt **1,826 → 534 tokens**.
Note: all results before this fix (including the ablation above) have inflated prompt tokens on steps ≥ 2.
Output tokens and accuracy are unaffected.

**Lesson (context engineering):** what goes back into the history matters as much as what goes into the
prompt. Check every field of the model's reply before re-sending it.
