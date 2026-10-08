# Graph Harness — Project Context

## What this project is
MSc/diploma thesis of Vasileios Papadimitriou (University of Ioannina), supervised by Konstantinos Skianis.
We are building a **graph-specific, cost-aware agent harness**: a Python program that wraps an LLM and lets it
solve graph problems by calling validated graph tools instead of reasoning over a serialized graph in the prompt.

**Core research question:** can a graph-specific harness let a cheap model match or beat a frontier model running
in a general-purpose harness (e.g. Claude Code), at a fraction of the cost?
"Better" means the accuracy-vs-cost Pareto front, not only raw accuracy.

The full proposal is in `docs/thesis_proposal_options_ABC.pdf` (we are doing Option A; Option B,
process-level evaluation, may become one chapter).

## Prior work in this codebase
The original thesis code lives in this repo, next to the new harness. It builds on
Skianis et al. 2024, "Graph Reasoning with LLMs via Pseudo-code Prompting". Reuse, don't rewrite:
- ER graph + question generator `scripts/generate_dataset.py`, data in `data/graphs/` and
  `data/graphs_questions/` (10 problems: node count, edge count, node degree, neighbors, edge existence,
  connectivity, connected components count, cycle check, shortest path, MST; sizes small/medium/large).
  All graphs are undirected and unweighted, and 5 small graphs are disconnected (so "MST" = any spanning
  forest). Topological sort (pseudocode only) and bipartite check have no questions; they come with the
  scaled generator in M6.
- `data/pseudocodes/` → become harness **skills** (per-task playbooks)
- prompting methods in `graph_reasoning/` (none, CoT, build-a-graph, alg, 1-shot, 1-shot+pseudo)
  → **pure-prompting baselines**
- structured step-by-step output and `scripts/dfs_cycle_demo.py` → basis for trajectory logging / trace checking
The old code is OpenAI-only and partially run; treat it as a reference, not a dependency
(`harness/` must not import from `graph_reasoning/`).

Old pipeline notes:
- Run with `uv run python run_experiments.py ...`; real runs call the OpenAI API and cost money. Ask first.
- `scripts/generate_dataset.py` overwrites the committed dataset in `data/`. Don't run it without asking.
- `legacy/run_experiments_original.py` is the paper's original script, kept for reference only.
- `--all` skips configs that already have a summary in `results/experiments/`.

## Architecture
```
question → Router (cheap LLM: task family + skill) → Agent loop (cheap LLM ⇄ graph tools)
        → Verifier (plain Python) → pass: answer | fail: Escalate (stronger model reruns the loop)
```
Layers (bottom → top): execution (networkx etc.) → tool layer (MCP server) → model layer (LiteLLM)
→ harness core (router, loop, verifiers, skills, escalation) → eval + logging.

## Design principles (do not violate without asking)
1. **The graph never goes into the prompt.** It lives in Python memory; the model only sees tool results.
2. **Tools never raise.** Validate inputs (node exists, directed/undirected, types) and return
   `{"error": "<clear, actionable message>"}` so the model can self-correct.
3. **Result handles for compaction.** Large outputs are stored harness-side and returned as a short summary
   plus an ID (e.g. `result_17: path of 243 nodes, length 1180, first 5: [...]`). Tools accept handle IDs.
4. **Structured final answers.** The loop ends when the model calls `submit_answer` with a per-task schema.
5. **Verification is code, not the model.** Each task family has a programmatic verifier
   (valid path, proper coloring, valid topological order, MST weight, ...).
   Verifiers check evidence; they never recompute the answer and compare (that would be an oracle).
   Tasks whose answer can't be checked without recomputing it (counts, degree, "no" answers) run with
   `verify=None` and are reported as unverified.
6. **Show only relevant tools.** The router picks a task family; the agent sees only that family's tools.
7. **Budgets everywhere:** max steps, max cost, loop detection (identical repeated calls).
8. **Append-only message history** (keeps prompt caching effective).
9. **Model-agnostic.** All LLM calls go through LiteLLM; never hardcode a provider.
10. **Log everything.** Every run writes a JSONL trace: messages, tool calls, results, tokens, cost, latency.

## Stack
- Python 3.11+, managed with `uv`
- LiteLLM (model layer + `completion_cost`), Pydantic (schemas, structured outputs)
- Models for now are local via Ollama (no API keys yet): cheap = `ollama/qwen3:8b`,
  stronger = `ollama/qwen2.5-coder:14b` (RTX 5070, 12 GB). Dollar cost is 0 locally, so log tokens and
  latency on every call; API models (and the frontier baseline) come later.
- FastMCP / official MCP Python SDK (tool server — the same server is given to baselines like Claude Code)
- networkx (correctness reference), igraph or rustworkx (large graphs), OR-Tools (NP-hard heuristics)
- Docker sandbox for the `run_python` escape-hatch tool
- pytest
- Secrets only via `.env` (gitignored). Never print, log, or commit API keys.

## Target repo layout
New folders sit next to the existing `graph_reasoning/`, `data/`, `results/`, `scripts/`, `legacy/`.
```
harness/
  models.py       # LiteLLM wrapper, cost + token logging
  tools/          # graph tools + MCP server, result-handle store
  skills/         # playbooks (ported from old pseudocodes/)
  router.py       # task-family classification (structured output)
  cli.py          # interactive chat with a loaded graph (M4.5)
  loop.py         # agent loop (AgentConfig, RunResult, run_agent)
  answers.py      # answer types, submit_answer / cannot_answer, check_answer
  prompts.py      # all text the harness says to the model
  parsing.py      # spotting and rescuing tool calls written as text
  tasks.py        # task registry: question template, answer type, verifier (no ground truth)
  trace.py        # append-only JSONL trace
  ask.py          # one question about one graph, from the command line
  verifiers.py    # per-family programmatic checks
  escalation.py   # cascade logic
eval/
  tasks.py        # dataset parsing, reference answers, grading (never imported by harness/)
  generators/     # ported question generator, scaled to 10k+ nodes
  adapters/       # GABench, ProGraph, GrAlgoBench, GT Bench loaders
  baselines/      # pure prompting, generic ReAct+python, Claude Code headless, GraphTeam
  runner.py       # runs any system on any benchmark, writes JSONL
  analysis/       # tables, Pareto plots, ablations
tests/
docs/
```

## Milestones (work on ONE at a time; I will say which)
- **M0** Project setup: uv, layout, `.env` handling, LiteLLM smoke test.
- **M1** Bare agent loop + 5 tools (graph_info, get_neighbors, shortest_path, connected_components, has_cycle)
  + `submit_answer` + JSONL logging, run on ported generator questions.
- **M2** Verifiers for the 10 dataset problems (evidence-only, see principle 5); accuracy + cost report script.
- **M3** Full tool set as an MCP server, input validation, result handles, `run_python` sandbox.
- **M4** Router + skills (ported pseudocodes); per-family tool filtering.
  Router = task classification (picks the registry entry, so also the answer type and verifier) +
  parameter extraction (node ids). Keep it swappable and compare candidates on routing accuracy,
  latency and cost: small LLM with structured output, TypeSafe Jev (zero-shot classifier with typed
  output + confidence; candidate for the classification part only, not parameters; cloud API, new
  dependency), embeddings (`nomic-embed-text`), regex baseline. Low confidence → no verifier
  ("unverified") or ask the user, never a guessed verifier. Jev is a router candidate, never a verifier.
- **M4.5** Interactive CLI on top of the router: load any graph file, ask questions in plain English in a loop,
  show tool calls and whether the answer was verified. Later: follow-up questions that reuse earlier answers.
  For debugging and the thesis demo.
- **M5** Escalation / model cascade.
- **M6** Benchmark adapters + baselines (same model and same tools across harnesses).
- **M7** Experiments: size scaling, Pareto fronts, component ablations, repeated runs with confidence intervals.

## Current status (update when it changes)
- M0, M1 done. **M2 in progress:** verifiers for shortest_path, mst (full), connectivity / cycle_check (a "yes"
  carries the path / cycle as evidence), connected_nodes (partial); counts, degree, edge existence and any "no"
  are unverified by design. Report script `eval/analysis/report.py` done.
- Pilot on large graphs done (`results/harness_runs/m2_pilot_large_verify/`, findings in `docs/learning_log.md`).
- `degree` tool added (node_degree ~8× faster, same accuracy); it's in the default tool set now
  (`--without-tool degree` reproduces the old setup).
- **M3 started** (plan approved 2026-10-07: 1 Pydantic schemas, 2 fuller tool set, 3 result handles,
  4 MCP server, 5 `run_python` sandbox, 6 close-out). Step 1 done: tool arguments and answers are Pydantic
  models; schemas the model sees are generated from them (byte-identical to before).
  Step 2 done: `has_path`, `is_bipartite`, `topological_sort` (each returns evidence for its answer).
  Step 3 done: result handles (`harness/tools/handles.py`), tested on a 10,000-node graph.
  Step 4 done: MCP server (`harness/tools/mcp_server.py`, official `mcp` SDK, stdio). Claude Code demo pending:
  the `claude` CLI's login on the MacBook had expired.
- **`docs/architecture.md` describes the code as built; update it whenever the architecture changes.** Open decisions: `mcp`
  package as a dependency; Docker vs OrbStack/Colima for the sandbox (no Docker on the MacBook yet).
- M2 still open: fixes A/B from the pilot (evidence wording, "G is undirected" prompt line), then the M2 runs on the GPU
  machine (checkers on vs off; small all tasks, large at least mst / connected_nodes / node_degree), then the
  closing learning-log entry.
- Machines: GPU machine (RTX 5070) for real runs; MacBook Air M4 has Ollama + qwen3:8b but is ~4.5× slower
  (~30 s/answer), so only for quick tests. No Anthropic API access yet (see learning log, 2026-10-07).

## Experimental rules
- Compare harnesses with the **same model** and, where possible, the **same tools**.
- **Dev/test split:** design tools, prompts, and routing on dev tasks only. Never tune on test tasks.
- At least 3 runs per configuration; report bootstrap confidence intervals.
- Every metric must be reproducible from the JSONL logs.

## How to work with me
- Before a milestone, propose a short plan and wait for approval. No large refactors without asking.
- Ask before adding a new dependency.
- Every tool and verifier gets tests against networkx ground truth.
- Prefer small, readable modules over clever abstractions; this is research code that must be easy to change.
- Keep explanations casual; Greek phrases are welcome.
- **This is also a learning project in harness engineering.** I write the core pieces (loop, tools,
  verifiers) myself: give me a concept explanation, a skeleton with TODOs, and failing tests, then review
  my code. Claude writes the plumbing (config, logging boilerplate, test scaffolding).
  Each milestone ends with a short entry in `docs/learning_log.md`.
- Git: on the GPU machine push over SSH
  (`git push git@github.com:SeiduTakizawa/solving-graph-problems-with-LLMs.git main`); on the MacBook SSH has no
  key, so `git push origin main` over HTTPS (credentials via `gh`).

## Key references
- Skianis et al. 2024, pseudo-code prompting — arXiv:2409.17906
- GABench (agent benchmark, calls for graph-specific harnesses) — arXiv:2608.01684
- GraphChain (tool chaining, 45 networkx APIs) — arXiv:2511.00457, github.com/GraphChain651/GraphChain
- GTA / GT Bench — arXiv:2609.12265
- GrAlgoBench — arXiv:2602.06319
- GraphTeam (multi-agent graph baseline) — github.com/BUPT-GAMMA/GraphTeam
- ProGraph benchmark — github.com/BUPT-GAMMA/ProGraph
- Neo4j GDS Agent (MCP tools + skill) — github.com/neo4j-contrib/gds-agent
- mini-swe-agent (minimal loop reference) — github.com/SWE-agent/mini-swe-agent