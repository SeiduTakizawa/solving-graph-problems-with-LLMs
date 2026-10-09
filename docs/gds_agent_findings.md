# GDS Agent: review and what to take from it (2026-10-08)

What Neo4j's GDS Agent does, how it compares with our harness, and what is worth implementing. Read on 2026-10-08:
the repo (281 commits, latest `6dc2b49`), the paper, and the public benchmark with its results.

- Repo: https://github.com/neo4j-contrib/gds-agent (MCP server + one skill)
- Paper: "GDS Agent for Graph Algorithmic Reasoning", arXiv 2508.20637
- Benchmark: https://github.com/brs96/gds-agent-benchmarks (questions, evaluation code, per-model results)

## 1. What it is, and isn't

A **tool server**, not a harness: ~60 Neo4j Graph Data Science algorithms over MCP, plus one skill (`SKILL.md`)
telling the agent how to use them. There's no loop, no budgets, no verification, no context management: those are
left to whatever harness runs it (Claude Code, Codex, Cursor, Gemini CLI). That makes it the **"frontier model +
general-purpose harness"** side of our research question almost exactly: a baseline, more than a design to copy.

How it works: the graph lives in Neo4j. The agent first *projects* an in-memory graph (`project_graph_cypher`), then
runs algorithms on it by `graphName`. Each algorithm has two modes: `stream` (return a table) and `mutate` (write the
result to the projected graph as a node property, for later tools to use).

## 2. Side by side

| | GDS Agent | our harness |
|---|---|---|
| Tools | ~60 algorithms, all listed at once | 13 graph tools (+ `run_python`); per-family filtering planned (M4) |
| Bad input | raw exception text, `Error: {e}` | Pydantic validation, actionable errors (`tools never raise`) |
| Large results | pandas table as text, capped at 500 rows / **100k chars (~25k tokens) per result** | handles (> 200 numbers), `brief()`, compaction at 75% |
| Composing results | `mutate` mode: results stored on the graph, then filtered/streamed back | handles (read only), `run_python` |
| Node references | names via `nodeIdentifierProperty` (e.g. "Paddington"), translated to ids | integer ids only |
| Verification | the skill *asks* the model to "verify key results using a genuinely different method" | code verifiers on evidence: free, enforced, never skipped |
| Guidance | one 22-line skill + a troubleshooting file | system prompt + `CODE_HINT`; skills in M4 |
| Code execution | whatever the host harness allows (their benchmark runs `claude -p --dangerously-skip-permissions`) | locked-down Docker sandbox |
| Evaluation | separate repo: tool precision/recall/F1, parameter match, answer match | JSONL traces, report, now the same process metrics |

## 3. Their benchmark: the most valuable find

`gds-agent-benchmarks` has questions **with expected tools, parameters and answers**, and **full results for Haiku
4.5, Sonnet 4, GPT-5 and GPT-4o**, run through Claude Code: answers, tool calls, turns, **tokens and dollar cost per
question** (`detailed_question_results.csv`, `results_<model>/*.json`). Examples of cost per question: 3.3k tokens /
$0.05 (articulation points), 65k tokens / $0.27 (personalised ArticleRank).

- **London Underground** (`ln`): 88 questions, 302 stations, 406 links (`dataset/london.json` in the gds-agent repo:
  `stations`, `connections` with `time`, `lines`). The CSV stores each question as 4 lines: question,
  expected tools (JSON list), expected parameters (JSON), expected answer.
- **Game of Thrones** (`got`): 12 questions, 2,565 nodes. **Citations**: a JSON question set (top-k ratios,
  longest citation chain, Adamic-Adar link prediction).
- Reported in the paper: GPT-5 best at **85.4%** answer accuracy; across models tool precision 0.77, recall 0.85,
  F1 0.75, parameter match 0.89, answer match 0.66. Their finding: **tool metrics barely correlate with answer
  accuracy**.

**What it gives us:** the London graph loads into networkx with no Neo4j. Running qwen3.5 + our harness on the same
questions compares a cheap model in a graph-specific harness with frontier models in a general harness, whose
tokens and cost are already measured. That is the thesis question, with the expensive baseline already paid for.

**Catch:** some answers depend on how GDS implements an algorithm (PageRank / ArticleRank scaling, Louvain/Leiden with
seed properties, label propagation, k-1 coloring). Those won't match networkx. Questions with exactly one correct
answer should: articulation points, WCC/SCC, bridges, k-core, triangle count, shortest paths, Yen's, MST, longest
path, BFS reachability, Jaccard node similarity. Rough guess: **55–60 of 88**; check question by question, and
recompute every reference answer with networkx, compared against theirs. Also: questions name stations, so we need
node names (section 6, item 6), and the paper notes models inject outside knowledge about London, which synthetic
graphs avoid.

## 4. Failure modes in the paper, and what we have for each

| their failure | what happens | ours |
|---|---|---|
| large outputs | the agent gives up when "output … would surpass number of allowed tokens" | handles, `brief()`, compaction, `context_full` |
| ignored output format | post-processes results in its own way instead of the asked format | `submit_answer` with a per-task schema |
| missing tool | "pretend to reason": a max-flow question with no max-flow tool answered by adding two equal shortest paths | `cannot_answer`, offered in every nudge |
| outside knowledge | injects facts about the real London network | synthetic ER graphs (a risk again for London, M6) |
| tool choice | precision 0.77 with ~60 tools on offer | 13 tools now; per-family filtering (M4) |

## 5. Lessons from their git history

- **Skills got shorter, not longer.** July 2026: a 109-line skill with a "question phrase → algorithm" table, a worked
  example and five reference files (`a866669`). Within a week cut to 22 lines plus troubleshooting (`92f40cc`,
  `e82c40d`), and the MCP `instructions` field and tool annotations were removed: "skills are the single guidance
  layer" (`03c7855`). For frontier models a minimal skill was enough. Our data says small models differ: `CODE_HINT`
  took qwen3.5 from 8 to 14 of 15 and hurt qwen3. **M4 needs a skill ablation** (none / minimal / prescriptive, per
  family and per model).
- **Progressive disclosure.** `SKILL.md` is short; `references/troubleshooting.md` is read only when an error happens.
- **Truncation messages were iterated on** (`998f515`): "a truncation warning means: do not retry the same call;
  instead re-run in mutate mode / narrow the algorithm (`nodes`, `topK`, cutoffs)". Small models do retry.
- **Node identifiers were a bug source**: exact matching instead of fuzzy, numeric ids sent as JSON numbers
  (`anyOf: string | number`), and validators against Cypher injection through `nodeIdentifierProperty` (they build
  queries with f-strings). We have no query strings, so no injection surface.
- **Tool calls run off the event loop** (`asyncio.to_thread`) so a client can call tools in parallel.

## 6. Worth implementing (ranked)

| # | what | why / evidence | effort | status |
|---|---|---|---|---|
| 1 | **Process metrics**: tool precision / recall / F1, parameter match, exact match, with their definitions | comparable to the paper; the Option B chapter | ~1 h | **done**: `eval/analysis/process.py`, `EXPECTED_CALLS` in `eval/tasks.py`, report `--process` |
| 2 | **Batch tools + handles as inputs**: `degree(nodes="result_1")`, `top_nodes(metric, among, k)` | their per-node tools take a `nodes` filter and return one table; our tools-only setup failed (2/15) on ~17 single `degree` calls per question | ½ day, plan first | open |
| 3 | **London adapter** on the subset networkx can answer | published frontier baselines with tokens and cost (section 3) | 1 day | open (M6 preview) |
| 4 | **Skill ablation in M4**, with progressive disclosure | section 5 | part of M4 | open |
| 5 | **Error-triggered hints**: error pattern → one-line hint, only when it happens | their troubleshooting file; our code fails 30–60% of the time (e.g. `KeyError: 'num_nodes'`) | ~1 h | open |
| 6 | **Node names**: `str \| int` node ids, "did you mean …?" on a miss | needed for any real-world benchmark; better than their exact match | ½ day | open |
| 7 | **"Don't retry" in truncation notes** (handles, stdout) | section 5 | 10 min | open |
| 8 | **Count self-verification in their traces**: how often do frontier models actually verify, and at what token cost? | a direct contrast with our code verifiers, from data already published | ~1 h | open (analysis only) |

From our process metrics (2026-10-08): on the 9 simple M2 tasks the cheap model is process-perfect (F1 1.00 over
270 answers). On the combine tasks recall is ~1 everywhere and **precision** separates the setups (tools only 0.11,
tools + code + hint 0.53). Unlike the paper, ours **does** predict the answer: in setup H, F1 0.67 for right answers
vs 0.18 for wrong ones (15 questions, 1 run).

## 7. Tools they have that we don't

What we have now: `get_neighbors`, `degree`, `has_edge`, `graph_info`, `shortest_path`, `has_path`,
`connected_components`, `has_cycle`, `minimum_spanning_tree`, `is_bipartite`, `topological_sort`, `distances_from`,
`neighborhood` (+ `read_result`, `run_python`).

What makes a tool interesting for us: (a) it **returns evidence a checker can verify without re-solving** (design
principle 5), (b) it works on our undirected, unweighted graphs today, (c) it is likely in the benchmarks planned for
M6 (check when writing the adapters).

### Tier 1: add next (deterministic, evidence-producing, small)

**Done (2026-10-09), opt-in:** `articulation_points`, `bridges`, `k_core(k)` (with `max_k`), `triangles(node)`,
`greedy_coloring` (`tests/test_extra_tools.py`; offered only with `--with-tool`, see `docs/architecture.md`).
No questions use them yet; they need question types (M6 generator or the London adapter). Still to do from this
tier: `clustering`, `k_shortest_paths`, `similar_nodes`.

| their tool | our version | evidence → verifier |
|---|---|---|
| `triangle_count` (with `nodes`) | `triangles(node)` listing the triangles | each listed triple is a triangle (partial: a missing one can't be seen, like `connected_nodes`) |
| `local_clustering_coefficient` | `clustering(node)` | the triangles + degree: the formula can be recomputed from the evidence |
| `articulation_points` | `articulation_points()` | each listed node: removing it increases the component count (partial: completeness unchecked) |
| `bridges` | `bridges()` | each listed edge: removing it disconnects its endpoints (partial) |
| `k_core_decomposition` | `core_number(node)` / `k_core(k)` | the k-core subgraph: every node in it has ≥ k neighbors inside it (a lower bound) |
| `k_1_coloring` | `coloring()` (greedy) | full check that it's a proper coloring (not that it's minimal) |
| `yens_shortest_paths` | `k_shortest_paths(source, target, k)` | each path is real and simple, lengths non-decreasing (partial on "these are the k shortest") |
| `breadth_first_search` / `depth_first_search` (`maxDepth`, `targetNodes`) | mostly covered by `distances_from`, `neighborhood`, `has_path` | an explicit DFS order only if a benchmark asks for one |
| `node_similarity` (Jaccard, `topK`) | `similar_nodes(node, k)` | the neighbor sets, so the scores can be recomputed |

### Tier 2: scores (common in benchmarks; no cheap evidence, so unverified)

`degree_centrality`, `closeness_centrality`, `harmonic_centrality`, `betweenness_centrality`, `pagerank`,
`eigenvector_centrality`, `HITS`, `article_rank`. The value is in **top-k and filtered queries** ("the 5 most
central nodes", "the score of node 4"), so give them `nodes` and `top_k` arguments instead of streaming every node.
Watch the conventions: GDS PageRank isn't normalized like networkx's, so answers from their benchmark won't match.

### Tier 3: once the M6 generator brings weights or direction

- **Weighted paths:** `find_shortest_path` / Dijkstra, `dijkstra_single_source_shortest_path`, `delta_stepping`,
  `bellman_ford` (negative weights, negative-cycle detection), `a_star` (needs coordinates), weighted MST.
- **`max_flow`**: the most interesting one. A **minimum cut is a certificate**: if the flow's value equals the cut's
  capacity, both are optimal (max-flow min-cut). The verifier can check optimality *without re-solving*, which is
  rare and fits principle 5 exactly.
- **`longest_path`** (DAGs), **`strongly_connected_components`** (directed), `all_pairs_shortest_paths` (the output
  is O(n²): handles only).
- **Steiner trees** (`minimum_directed_steiner_tree`, `prize_collecting_steiner_tree`): NP-hard, heuristic answers;
  OR-Tools territory (see Stack in `CLAUDE.md`).

The same certificate idea applies to what we have: `distances_from` layers are distance labels, and labels that
satisfy every edge prove a shortest path's length optimal, in O(E) checks, without re-running BFS. A possible
upgrade for the `shortest_path` verifier.

### Community detection: later, if at all

`louvain`, `leiden`, `label_propagation`, `speaker_listener_label_propagation`, `modularity_optimization` are
non-deterministic or seed-dependent, so "which community is node 4 in" has no single right answer.
`modularity_metric` and `conductance` (scoring a given partition) are deterministic and could be tools for
questions like "how good is this split".

### Skip

Node embeddings (`fast_rp`, `node2vec`, `hashgnn`, GraphSAGE), ML pipelines (node classification, link prediction,
regression), `k_means`, `HDBSCAN`, `CELF`, `random_walk` (random or learned output, not graph reasoning), and the
Neo4j plumbing (projection, sessions, `stream_*`, model catalog).

### Features they put on many tools

- **`nodes` filter + `topK` / `topN` / cutoffs:** return only what was asked. Covered by item 2 of section 6.
- **`mutate` mode:** a result stored on the graph and used by the next tool. Our version would be handles as
  inputs (item 2).
- **`orientation`** (natural / reverse / undirected) for directed graphs, and **`relationshipWeightProperty`** on
  every weighted algorithm. Add these when graphs get direction and weights (M6).
- **`maxDepth` / `targetNodes`** early stopping on traversals.

## 8. Beyond the benchmarks: tools for real-life questions

Questions people actually ask of real graphs (roads, power grids, social networks, payments, dependencies), and
the tool that answers each with evidence a checker can use. The pattern: a real question is rarely "run algorithm
X"; it is "what breaks if...", "what is the cheapest...", "who is connected to...". So the useful tools are
**what-if** and **constrained** versions of the algorithms we have.

| real question | domain | tool | evidence → verifier |
|---|---|---|---|
| Which single station / server / router failing cuts the network? | infrastructure, IT | `articulation_points`, `bridges` (done) | remove it, count components |
| What if node X (or edge u-v) goes down: still connected? how much longer is the route? | resilience, logistics | `what_if_removed(nodes, edges)` then any tool on the copy | the changed graph is the same graph minus the listed items |
| Shortest route that avoids X / must pass Y | navigation, routing | `shortest_path(..., avoid=[...], via=[...])` | a valid path that skips the avoided nodes (checkable) |
| Fastest / cheapest route (weights: km, minutes, price) | roads, flights | `weighted_shortest_path` (Dijkstra) | path is real + distance labels as an optimality certificate |
| A second and third route as a backup | navigation, networks | `k_shortest_paths` (Yen) | each path real and simple, lengths non-decreasing |
| How much can flow from A to B (pipes, bandwidth, trucks)? | utilities, supply chain | `max_flow` + `min_cut` | flow conserves at every node, cut capacity = flow value proves it optimal |
| Do tasks / packages have a circular dependency? an order to do them in? | builds, project plans | `topological_sort` (have), directed `has_cycle` | the order or the cycle |
| Schedule exams / meetings so conflicts never share a slot | timetabling, register allocation | `greedy_coloring` (done) | proper-coloring check |
| Pair people with tasks / drivers with riders | assignment, matching markets | `max_matching` (bipartite: Hopcroft–Karp) | the pairs share no node; a vertex cover of the same size proves maximum (König) |
| Where to put k warehouses / sensors / cameras to cover everything | facility location | `dominating_set` / `vertex_cover` (approximate) | every node / edge is covered (checkable; not optimal) |
| Cheapest cable / pipe layout connecting given sites | telecom, utilities | `steiner_tree` (approximate), `minimum_spanning_tree` (have) | it's a tree touching every site; total weight |
| Tight groups, fraud rings, bot clusters | social, finance | `k_core` (done), `triangles` (done), later communities | every core node has ≥ k neighbors inside |
| Money going around in a loop (A→B→C→A) | anti-money-laundering | directed cycles through a node, with amounts | the cycle is real |
| Who is most central / influential / a bottleneck? | social, transport | `top_k_centrality(measure, k)` (PageRank, betweenness) | none cheap: unverified, `top_k` keeps the output small |
| People you may know / products bought together | recommendations | `similar_nodes` (Jaccard), `common_neighbors` | the neighbor sets |
| Who can reach whom within 2 hops / 30 minutes | contact tracing, delivery zones | `neighborhood` / `distances_from` (have); weighted version | distance layers |
| Find the node called "King's Cross" | any real data | `find_node(name)` + node names in every result | an exact match from the node table |

Plumbing real graphs need before most of this matters: **node names and attributes** (real graphs have names, not
ids; see item 6 of section 6), **edge weights and direction** (every row above with "weighted" or "directed"), and
loading common formats (CSV edge lists, GraphML). The M6 generator and the London adapter bring the first two.

Best picks, by value for the thesis (each adds evidence-checkable answers, the harness's strong point):
1. `shortest_path` with `avoid` / `via` and `what_if_removed`: real questions, cheap, fully checkable.
2. `max_flow` + `min_cut`: the rare case where the evidence proves optimality without re-solving.
3. `max_matching` with a König cover: same, for assignment questions.
4. `weighted_shortest_path` with distance labels as a certificate, once graphs have weights.

