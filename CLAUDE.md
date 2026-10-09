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
- Models are local via Ollama for now (no API keys yet). Cheap model so far: `ollama_chat/qwen3:8b`; the
  cheap and stronger models are being re-chosen by a pilot (see "Model candidates" below). Dollar cost is 0
  locally, so log tokens and latency on every call; API models (and the frontier baseline) come later.
- Official MCP Python SDK (`mcp`), not FastMCP: the tool server given to baselines like Claude Code
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
  Tool exposure ablation (decided 2026-10-09): compare (1) all tools, (2) router picks the family's tools,
  (3) agent picks: sees tool categories and calls `load_tools(category)`, (4) hybrid: router picks plus a
  `more_tools(category)` escape hatch (planned default). Measure accuracy, tokens/question, tool precision (process
  metrics) and escape-hatch use. Today all 13 default tools cost ~900 tokens per step (first call 1.3–1.7k tokens vs
  ~20k for Claude Code); the saving grows with the tool count. Caveats: an extra `load_tools` step resends the
  conversation, and changing the tool list mid-run breaks Ollama's prefix cache (`read_result` already does this).
- **M4.5** Interactive CLI on top of the router: load any graph file, ask questions in plain English in a loop,
  show tool calls and whether the answer was verified. Later: follow-up questions that reuse earlier answers.
  For debugging and the thesis demo.
- **M5** Escalation / model cascade.
- **M6** Benchmark adapters + baselines (same model and same tools across harnesses).
- **M7** Experiments: size scaling, Pareto fronts, component ablations, repeated runs with confidence intervals.

## Current status (update when it changes)
- M0, M1, **M2 done** (2026-10-07). Checkers for shortest_path, mst (full), connectivity / cycle_check (a "yes"
  carries the path / cycle as evidence), connected_nodes (partial); counts, degree, edge existence and any "no"
  are unverified by design. Report script `eval/analysis/report.py`.
  M2 closing run on the GPU (large, 9 tasks × 15 × 2 runs, checkers on/off, `m2_large_verify` / `m2_large_noverify`):
  100% lenient, 88–90% strict; all checker rejections were missing cycle evidence; rescued text tool calls
  (10–12%, ~half of connected_nodes) are the main error source. Fixes A/B were not applied first and are still open.
- `degree` tool added (node_degree ~8× faster, same accuracy); it's in the default tool set now
  (`--without-tool degree` reproduces the old setup).
- `distances_from` and `neighborhood` (building blocks for multi-step questions): on "farthest node" / "nodes
  within 2 hops" the model went from 1/4 to 4/4, one tool call each. Claude Code + Sonnet vs our harness on large
  graphs: 9/9 both, ~28× fewer input tokens for ours (`claude_code_vs_harness_large`).
- `PROMPT_VERSION` (`harness/prompts.py`, logged in every `run_start`); its changelog says what each version
  means. The GPU machine and the MacBook diverged on 2026-10-07/08 and were merged on 2026-10-08; the GPU side's
  original commits are on branch `backup-gpu-2026-10-07`.
- **M3 started** (plan approved 2026-10-07: 1 Pydantic schemas, 2 fuller tool set, 3 result handles,
  4 MCP server, 5 `run_python` sandbox, 6 close-out). Step 1 done: tool arguments and answers are Pydantic
  models; schemas the model sees are generated from them (byte-identical to before).
  Step 2 done: `has_path`, `is_bipartite`, `topological_sort` (each returns evidence for its answer).
  Step 3 done: result handles (`harness/tools/handles.py`), tested on a 10,000-node graph.
  Step 4 done: MCP server (`harness/tools/mcp_server.py`, official `mcp` SDK, stdio). Claude Code demo done
  (`results/harness_runs/claude_code_mcp_demo/`): correct, ~21.5k input tokens vs 1.3k in our harness.
  Step 5 done: `run_python` sandbox (`harness/sandbox/`, Docker; OrbStack on the MacBook, Docker Engine on the
  GPU machine). Off by default; runner `--python tools|networkx`. Tests skip without Docker.
  Code vs tools (2026-10-08, learning log): in code, tools return plain values and errors raise (`code_value`); a
  "combine" task family (`hop_max_degree`, `common_neighbors_max`, `triangle_count`) where code beats tool-by-tool
  calls (tools only 0–4/15, with code 8–12/15); runner `--code-only` / `--code-hint`; report `--code`.
  **Ollama's default 4k context silently cut qwen3.5's replies**: `models.py` now asks for 16k (`OLLAMA_NUM_CTX`) and
  flags prompts near the limit. Only the `combine_*` runs were affected; clean rerun `combine16k_*`.
  Compaction (`harness/compaction.py`): at 75% of the window older tool results and code are shortened; nothing
  left to shorten at 90% → status `context_full`. `PROMPT_VERSION` is now **v7** (see its changelog). Model calls retry transient errors (Ollama CUDA crashes).
- **`docs/architecture.md` describes the code as built; update it whenever the architecture changes.** Open decisions: `mcp`
  package as a dependency; Docker vs OrbStack/Colima for the sandbox (no Docker on the MacBook yet).
- **Next:** read `combine16k_*`, then M3 step 6 (close-out), the model pilot, and the M4 plan. Still open from M2: fixes A/B from the pilot (evidence wording, "G is undirected"
  prompt line); a run on small graphs was skipped (large only).
- **GDS Agent review** (2026-10-08): `docs/gds_agent_findings.md`: comparison, their public benchmark with frontier
  results and costs (a ready-made baseline), ranked ideas, and the tools they have that we don't (with evidence and
  verifier notes). Item 1 (process metrics: `eval/analysis/process.py`, report `--process`) is done.
- **M4 started** (2026-10-09): router (`harness/router.py`; written by Claude at the author's request: regex
  baseline + LLM router, `qwen3.5:9b`, thinking off, via a `route` tool call since Ollama ignores `format` with
  thinking off). Routing eval `eval/routing.py` on hand-written paraphrases (`data/routing/paraphrases.json`): dev
  regex 60%, LLM 94% (r3, after schema fixes seen on dev; test phrasings untouched). Wired into the runner
  (`--router oracle|regex|llm`, `--phrasing paraphrase`), tool exposure (`--tools all|task|hybrid|agent`, task
  families in `harness/tasks.py`, `more_tools` escape hatch), `any` answer type for unknown routes, count verifiers
  with evidence (components, triangles). Decisions, results, Jev: `docs/router.md`. Results: router + hybrid as
  accurate as oracle + all tools with 25–37% fewer tokens (`m4_*`, `ms_*`, 1 run).
  v9 fixes from those traces: per-step output cap (thinking spirals), code lists submittable by handle, skills for
  five tasks (`harness/skills/`, `--skills`), "no import needed". `PROMPT_VERSION` **v9**.
  Still M4: skills for the simple tasks (ported pseudocodes), 3-run exposure / skills ablations, a better router
  confidence signal (route twice for parameter-heavy tasks).
- Ideas noted: questions for the multi-step family (farthest node, k-hop counts) from the M6 generator; Brig
  (brig.sh, local microVM sandbox) as a candidate for sandboxing the Claude Code / Codex baselines in M6.
- Machines: GPU machine (RTX 5070) for real runs; MacBook Air M4 has Ollama + qwen3:8b but is ~4.5× slower
  (~30 s/answer), so only for quick tests. No Anthropic API access yet (see learning log, 2026-10-07).

## Sandbox setup (run_python)
- MacBook: OrbStack (installed 2026-10-08). Shells opened before the install may not have `docker` on PATH; the
  harness also finds `~/.orbstack/bin/docker`.
- GPU machine (Linux): Docker Engine, and let your user run it without sudo:
  ```
  sudo apt install docker.io            # or Docker's own repo for a newer version
  sudo usermod -aG docker $USER         # then log out and back in
  docker run --rm hello-world           # check
  ```
  The sandbox image builds itself on first use (~1 min), then starts in ~0.3 s.
  Done on the GPU machine: Docker 29 runs without sudo, and the sandbox tests pass (checked 2026-10-08).

## Model candidates (next GPU-machine session, decided 2026-10-08)
RTX 5070 = 12 GB VRAM: dense models up to ~14B fit fully on the GPU; MoE models with ~3B active parameters can run
with expert offload to RAM. Research notes and sources: `docs/learning_log.md` (2026-10-08, model choice).

| model | why | fits? |
|---|---|---|
| `qwen3.5:9b` | likely new default cheap model: successor to qwen3:8b, same memory (6.6–7.6 GB) | yes |
| `gemma4:12b` | a second family (Google), native function calling, 7.7–8 GB | yes |
| `gpt-oss:20b` | strong function calling; 14 GB download, sources disagree on fit | test: `ollama ps` must show 100% GPU |
| `qwen3:8b` | current baseline; rerun on the GPU with today's tool set (the Mac pilot used fewer tools) | yes |
| later: `qwen3.6:35b-a3b` | strongest local option (MoE, 3B active) for M5 escalation; needs llama.cpp `--n-cpu-moe` | offload only |
| API: Sonnet 5.5, Kimi K2.6 (OpenRouter) | strong closed / strong open baselines in our harness (Kimi K2 can't run on 12 GB) | n/a |

Pilot per model (54 questions, large dev graphs, ~5–10 min each on the 5070):
```
ollama pull qwen3.5:9b && ollama pull gemma4:12b && ollama pull gpt-oss:20b && ollama pull qwen3:8b
uv run python -m eval.runner --task all --size large --n 6 --runs 1 --model ollama_chat/qwen3.5:9b --name pilot_qwen35_9b
uv run python -m eval.runner --task all --size large --n 6 --runs 1 --model ollama_chat/gemma4:12b --name pilot_gemma4_12b
uv run python -m eval.runner --task all --size large --n 6 --runs 1 --model ollama_chat/gpt-oss:20b --name pilot_gptoss_20b
uv run python -m eval.runner --task all --size large --n 6 --runs 1 --model ollama_chat/qwen3:8b --name pilot_qwen3_8b_gpu
uv run python -m eval.analysis.report pilot_qwen35_9b pilot_gemma4_12b pilot_gptoss_20b pilot_qwen3_8b_gpu --by-task --plot pilot_models.png
```
Decide on: strict accuracy, rescued tool calls, verifier rejections, tokens and time per question. Then fix the
cheap model (and tool set) for the M2 runs and write it here. Note: these model names are ~Oct 2026 and the specs
come mostly from model pages and blogs; check `ollama.com/library` for newer small models first.

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