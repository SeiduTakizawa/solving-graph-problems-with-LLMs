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

---

## 2026-10-06: M2 so far: MST, evidence-based checkers, report script, dataset audit

### What was built
- **MST task.** New `edge_list` answer type, `minimum_spanning_tree` tool, `verify_mst`. The graphs are
  unweighted and some small ones are disconnected, so any spanning *forest* is a correct answer. The old
  pipeline's checker (`graph_reasoning/graph_algorithms.py`) compared against one exact edge set, so it marked
  some correct answers wrong. `eval/tasks.py` now grades with an independent spanning-forest check.
- **Evidence-based checkers.** A "yes" to connectivity or cycle_check now has to come with the path or cycle,
  sent as an extra `submit_answer` field that the loop passes to the verifier. The answer itself stays
  true/false, so grading did not change. `has_cycle` now returns the cycle it found. `connected_nodes` gets a
  partial check: every listed node must be a real neighbor, listed once.
- **Report script** `eval/analysis/report.py`: experiments side by side (and per task) as Markdown tables,
  with accuracy, a 95% bootstrap CI over questions (all runs of a question go in or out together), the share
  of answers a checker actually looked at, and tokens and time per question. `--plot` draws accuracy against
  tokens, with both axes from 0, so 99.5% vs 100% doesn't look like a big gap.

### Key idea: a verifier checks evidence, it never re-solves the question
If a verifier recomputes the answer with networkx and compares, it is an oracle: accuracy goes to ~100% and
the comparison with other harnesses means nothing. So a verifier may only check what the answer itself shows:

| checkable | how |
|---|---|
| shortest_path, mst | the answer *is* the evidence (real edges, right ends / no cycle, n − c edges) |
| connectivity "yes", cycle_check "yes" | the model must send the path / cycle |
| connected_nodes | partly: listed nodes must be real neighbors; a missing one can't be seen |
| counts, degree, edge_existence, any "no" | not checkable without solving again → unverified |

This is the same pattern coding agents use (tests and compilers as verifiers), and the same reason
"LLM-as-judge" is weaker: code gives a precise, trustworthy error message; a second model doesn't.

### Dataset audit: several tasks have (almost) one possible answer
Counted on the dev graphs (0–49):

| | small (5–10 nodes) | large (21–50 nodes) |
|---|---|---|
| edge_existence answer is "yes" | 97 / 100 | **100 / 100** |
| graph has a cycle | 47 / 50 | **50 / 50** |
| graph is connected (components = 1) | 47 / 50 | **50 / 50** |
| connectivity questions | 28 (14 yes / 14 no) | **0** |
| shortest path length | | avg 1.5, max 3 |

Large graphs are very dense (63–1,005 edges on 21–50 nodes), so paths are short and almost every pair is an
edge. A model that always answers "yes" / "1" scores ~100% on edge_existence, cycle_check and
components_count. These tasks still test whether the loop and tools work, but they can't tell a good model
from a lucky one. Only connectivity (small) is balanced.

Where large graphs *are* a real test: **connected_nodes** (avg 21, up to 45 neighbors to copy),
**node_degree** (counting up to 45, unchecked), and **mst** (avg 36, up to 49 edges to copy). That is where
copying mistakes, and the effect of the checkers, can show up.

**Takeaways for the thesis:**
- Report per-task accuracy next to a "majority answer" baseline, so a 100% on a one-answer task is read right.
- The M6 generator should balance yes/no answers and control density, so paths are long and not every pair
  is an edge.
- An ablation idea: a `degree` tool would make node_degree trivial on large graphs (like `has_edge` did for
  edge_existence: same accuracy, ~8× fewer tokens).

### Still open for M2
Real runs with qwen3:8b on the GPU machine, checkers on vs off, then the report and a closing entry:
small (all tasks), and large (at least mst, connected_nodes, node_degree). Timing: ~4 s/answer on the RTX 5070,
~680 questions per run, 3 runs, twice → ~4–5 h per size. A base M4 Mac (24 GB) fits the model but has
~5× less memory bandwidth (~120 vs ~670 GB/s), and token generation is bandwidth-bound, so there it's
~3–5× slower: fine for quick tests, not for the full runs.

---

## 2026-10-07: Pilot run, large graphs (M2)

**Setup:** `ollama_chat/qwen3:8b`, base M4 MacBook Air (24 GB), large dev graphs, all 9 tasks × 6 questions,
1 run, checkers on. Results in `results/harness_runs/m2_pilot_large_verify/`. The runner now prints progress
with time left and an estimated end time.

### Results
**54/54 correct (lenient), 43/54 strict.** The model picked the right tool on its first call in every question.

| task | time / answer | notes |
|---|---|---|
| node / edge / component counts, edge existence | 18–20 s | 2 steps: one tool, then submit |
| connected_nodes, shortest_path | 27 s | |
| cycle_check | 35 s | 5 of 6 sent back once by the checker (see below) |
| mst | 69 s | writes out up to 49 edges |
| node_degree | **133 s** (up to 223 s) | see below |

On this Mac that is ~30 s per answer on average, ~4.5× slower than the RTX 5070. The full large run
(650 questions × 2 runs × checkers on/off) would take ~32 h here; 20 questions per task ~8 h.

### Findings
1. **Text tool calls: 11/54 (20%) answers were rescued.** qwen3 writes `submit_answer {"answer": 12}` at the end
   of its text instead of calling the tool. Lenient mode parses and runs it; strict mode would fail them.
   Part of what makes a small model usable is the harness cleaning up its format mistakes, so report both.
2. **node_degree is slow because the model worries, not because it counts.** In the slow cases most of the time is
   step 1, *before any tool call* (e.g. 1,759 tokens, 123 s), spent debating whether G might be directed (degree =
   in + out?). Then it calls `graph_info` to check. The fact was one tool call away, but thinking happens before
   tool calls, so another tool can't fix it. **Lesson: facts the model needs to plan belong in the context up
   front; tools are for facts it needs to compute.** Only add facts that don't give away an answer (directedness
   yes, node count no).
3. **cycle_check: the model saw the cycle and decided not to send it.** `has_cycle` returned `[2, 1, 5, 3]`;
   thinking: "the user only asked if there's a cycle, so confirming true is sufficient". Our `submit_answer`
   description makes the evidence sound optional. The checker caught it every time (fixed on the 2nd try), at the
   cost of an extra round (~8 s, ~130 tokens).

### Next (proposed, not done yet)
- A. `submit_answer` description: evidence is *required* when answering true.
- B. System prompt line "G is an undirected graph" (from `graph.is_directed()`); measure on the same 6
  node_degree questions.
- C. Fewer text tool calls: clearer prompt or an example call; measure the strict score.
- Experiments: `degree` tool on/off; qwen3 thinking on/off; checkers on/off (the M2 comparison).
- Report: add strict accuracy and a majority-answer baseline; a `--compare` view (one row per task, one column
  per experiment).

### Comparing with Claude (open)
Two ways, and they answer different questions:
- **Same harness, different model** (Sonnet 5.5 through LiteLLM): needs API access (`ant auth login` or an API
  key, < $0.25 for 6 questions). A Claude.ai / Claude Code subscription can't be used from our own scripts.
- **Different harness** (Claude Code headless, `claude -p`, on the subscription) with our tools through an MCP
  server: this is the M6 baseline and the thesis's main comparison. Needs the MCP server from M3 (new dependency).

---

## 2026-10-07: A `degree` tool (ablation, large graphs)

`degree(node)` returns the degree (in + out for a directed graph, with both parts). Its description says how
directed graphs are handled without saying whether *this* G is directed, so the model doesn't need to know.
Same 6 node_degree questions as the pilot (large, dev, qwen3:8b on the M4 Mac, 1 run). Results in
`results/harness_runs/degree_tool_large_pilot/` vs `m2_pilot_large_verify/`.

| | without `degree` | with `degree` |
|---|---|---|
| correct | 6/6 | 6/6 |
| time / question | 133 s (max 223 s) | **17 s** (max 23 s), ~8× faster |
| output tokens (mostly thinking) | 1,988 | **264**, ~7.5× fewer |
| total tokens | 3,758 | 1,582 |
| rescued text tool calls | 4/6 | 1/6 |
| tools used | `graph_info` → `get_neighbors` (4), `get_neighbors` (2) | `degree` (6) |

In the pilot the model's first thought was "there's no function called degree, so get_neighbors is the way to
go", and the directed/undirected debate started from there. With the tool, the plan is obvious and the debate
never starts; the counting disappears too.

**Pattern (2nd time, after `has_edge`, also ~8×):** when a task has a tool that answers it directly, a small
model stops reasoning about *how* to answer. Same accuracy, a fraction of the cost. Move decisions the model is
bad at (what does "degree" mean here, counting a 45-item list) into tools.

**Note for experiments:** `degree` is now in the default tool set, so runs from here on are not directly
comparable with earlier node_degree runs; use `--without-tool degree` to reproduce the old setup. Baselines in M6
must get the same tool.

---

## 2026-10-07: M3 step 1, Pydantic for tool arguments and answers

Each tool's arguments are now a Pydantic model (`NodeArgs`, `EdgeArgs`, `PathArgs`, `NoArgs`), registered once in
`TOOLS` (function + args model + description). The schema the model sees (`GRAPH_TOOLS`) is generated from it, and
`run_tool` validates with it. Answer types work the same way (`ANSWER_MODELS`, strict types), and `check_answer`
validates with them. Before, every tool was described twice (function + hand-written JSON) with a test to keep
them in sync.

- **The prompt didn't change.** Generated schemas are byte-identical to the old hand-written ones (titles and
  `additionalProperties` trimmed, key order kept), so all earlier results stay comparable. Checked by dumping
  both and comparing.
- **Better errors for the model:** "Bad arguments for shortest_path: target: Field required", "verbose: Extra
  inputs are not permitted", "node: a node id must be an integer, not true/false".
- **Two behaviour fixes:** `"3"` is now accepted as node 3 (before, it was looked up as the string "3" and
  reported as a missing node, which is a confusing error); `true` as a node id is rejected (plain `int` in Pydantic
  turns `True` into 1, so node ids use a custom type that refuses bools).
- Smoke test, all 10 tasks on small graphs with qwen3:8b: 10/10, no argument errors.

**Lesson:** a schema the model reads and the validation the harness runs should come from the same definition;
two copies drift. And check what a library's "lax" mode accepts: Pydantic's `int` happily takes `True`.

---

## 2026-10-07: Cleanup before the long runs

- **The runner re-read the whole trace file after every question** to find that question's token counts
  (`_run_end`). The file grows with every question, so the work grew quadratically: on a 2,600-answer run, the
  last questions each re-read tens of MB. Now `run_agent` returns steps, tokens and model time in `RunResult`, and
  the runner reads them from there. **Lesson:** don't read back from the log what the code that wrote it already
  knew; the log is for later analysis, not for passing data between parts of the program.
- **One summary instead of two.** `runner.summarize` and `eval/analysis/report.py` computed the same tables with
  different columns (only one had strict accuracy, only the other had CIs). The runner now prints the report's
  tables, which gained a **strict** column (accuracy without rescued text tool calls). Report numbers were checked
  unchanged before/after; the pilot shows 100% lenient vs 79.6% strict.
- Small: `missing_node()` helper instead of 5 copies of the same error, `partial(task.verify, graph, params)`
  instead of a lambda with default arguments, `list(TOOLS)` instead of digging names out of the schemas.
