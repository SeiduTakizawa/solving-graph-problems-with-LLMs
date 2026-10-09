# Router (M4): design notes

Started 2026-10-09. Code: `harness/router.py`, eval: `eval/routing.py`, questions: `data/routing/paraphrases.json`,
tests: `tests/test_router.py`.

## Decisions
- **Route to a task** (14 classes + `unknown`), not a family: the answer type and the verifier are per task;
  families (for tool filtering) follow from the task.
- **`unknown` instead of a guess.** A route is accepted only if `check_route` passes: a real task, confidence
  ≥ `MIN_CONFIDENCE` (0.6 for now, to be set from the dev eval), exactly the task's parameters, nodes that exist
  in G. Otherwise: all tools, no verifier, reported as unverified.
- **Routers first:** regex (the floor) and an LLM (`qwen3.5:9b`, the agent model: no extra VRAM, no model swap
  on the 12 GB GPU). Later: smaller LLMs (`qwen3:4b`, `1.7b`), embeddings (`nomic-embed-text`), Jev.
- **The LLM router calls a `route` tool**, thinking off. Not Ollama's `format` (JSON schema) option: with qwen3.5
  and `think: false`, Ollama 0.30.11 ignores `format` and returns plain text (checked with a direct API call). With
  thinking on, `format` works but costs ~650 characters of reasoning and ~2.5 s. The tool call: ~1 s and ~450 tokens
  per route, 4/4 right in a smoke test (including an out-of-scope question → unknown).
- **Named parameters** (`node`, `u`, `v`, `source`, `target`, `via`, `k`), not "the numbers in order": a
  paraphrase can reorder them ("via 5, from 2 to 9").
- **Paraphrases, not templates.** The dataset's questions come from fixed templates, so a regex router scores
  ~100% on them and says nothing. `data/routing/paraphrases.json` has hand-written phrasings: 6 dev + 3 test per
  task, and out-of-scope questions (10 dev, 5 test). Test phrasings are held out like graphs 50–99.
- **Two kinds of errors:** a *safe miss* (in-scope → unknown: still answered, just unverified) and a *dangerous*
  one (wrong task or wrong parameters → a wrong verifier, or an out-of-scope question answered as a task). The
  eval reports them separately; the confidence threshold trades one for the other.

## First results (2026-10-09, dev paraphrases, 94 questions, 1 run each; `results/routing/`)

| router | right | in scope right | safe misses | dangerous (wrong + false known) | median latency | tokens/route |
|---|---|---|---|---|---|---|
| regex, dataset templates (sanity) | 14/14 | 100% | 0 | 0 | ~0 s | 0 |
| regex, paraphrases | 60% | 56% | 29 | 9 | ~0 s | 0 |
| LLM r1 (qwen3.5:9b) | 53% | 48% | 44 | 0 | 1.00 s | 954 |
| LLM r2 | 73% | 70% | 23 | 2 | 0.97 s | 932 |
| LLM r3 | **94%** | 94% | 4 | 2 | 0.85 s | 885 |

- The trap is real: the regex router is perfect on the templates and 60% on paraphrases (9 dangerous errors,
  e.g. "Route from 5 to 1 through 2" → connectivity). It was not tuned to the paraphrases: it is the baseline.
- The LLM's misses were almost all **format**, not classification: r1 filled every unused field with the string
  "None" (the schema said `anyOf [integer, null]`) and left out `confidence`. r2 shows the optional fields as plain
  integers and moves `confidence` up; r3 makes `confidence` optional and accepts a node pair in the other pair's
  fields. Same lesson as the agent's tools: the schema's shape matters as much as the prompt.
- r3's two dangerous errors are arguable phrasings: "Count the neighbours of node 0" → connected_nodes, and
  "Is G a tree?" → mst.
- Caveat: these fixes were made while looking at dev errors, so 94% is optimistic; the held-out test phrasings
  (`--split test`) give the honest number, once, when the router is frozen. One run each: the LLM isn't
  deterministic, so repeat runs are needed before comparing routers closely.
- After r3, 75 of 94 LLM routes came without a confidence (others: 1.0 ×11, 0.5 ×2, 0.9, 0.8), so the 0.6
  threshold rarely acts: confidence needs another signal (see below).

## End to end with the router (2026-10-09; 26 paraphrased questions, all 14 tasks, large dev, 1 run)

qwen3.5:9b, `--python tools --code-hint --phrasing paraphrase`, v8 (`results/harness_runs/m4_*`):

| setup | accuracy | tokens / q (incl. router) | time / q | graph tools at start | tool precision |
|---|---|---|---|---|---|
| oracle + all tools | 84.6% | 9,624 | 22.6 s | 13 | 0.81 |
| LLM router + task tools | 88.5% | 7,971 | 14.4 s | 2.5 | 0.87 |
| LLM router + hybrid | 88.5% | 6,066 | 12.5 s | 2.1 | 0.83 |

- No accuracy lost to routing and small tool families (the difference is noise at 26 questions), 37% fewer tokens
  and 45% less time with hybrid. `more_tools` was never called.
- Routed right 92% (24/26): the misses were shortest_path routed unknown (parameters in the wrong fields): safe.
- **A dangerous error, seen live:** in the hybrid run the router swapped `target` and `via` on "from 7 to 0 that
  passes through 16" (routed right in the other run: the LLM isn't deterministic). The verifier, checking the wrong
  question, rejected the agent's answer twice until loop detection. Grading still used the dataset's task.
- `cycle_check` "is the graph acyclic?" was wrong in all three: the phrasing had the opposite polarity, so a right
  answer was graded wrong. Fixed in the paraphrase set (r3 routing numbers came from the old set).
- Triangle evidence: the agent often forgot the triangle list and was sent back; it cost one question in two setups.
- Seen in a trace: one step of 14,530 thinking tokens (162 s) that hit the 16k window and came back empty; the
  `near_context_limit` flag only checks the prompt. Fix candidates: a per-step output cap, and flagging prompt +
  output near the window.

## Multi-step tasks with the router (2026-10-09; 5 paraphrased questions × 4 tasks, large dev, 1 run)

`ms_oracle_all` vs `ms_llm_hybrid` (qwen3.5:9b, code + tools + hint, v8):

| setup | accuracy | tokens / q | time / q | rejected | no answer |
|---|---|---|---|---|---|
| oracle + all tools | 65% (13/20) | 23,698 | 45.6 s | 6 | 5 |
| LLM router + hybrid | 80% (16/20) | 17,866 | 38.1 s | 2 | 4 |

Per task (before → after): shortest_path_via 5→5, hop_max_degree 5→5, common_neighbors_max 2→3, triangle_count
1→3 (one of the three via an unknown route, so with no evidence asked). Routing 19/20 right, the miss safe
(unknown). The oracle run compacted twice; the router run never. Overlapping CIs (45–85% vs 60–95%): promising,
not proven; needs 3 runs.
- triangle_count: 3 of 4 oracle failures involve the evidence check (the agent forgets or can't copy the triangle
  list); two of them also had a wrong count the check exposed ("answered 390 but listed 60"). Fix candidates: a
  hint to return the list from code and submit its handle, or the opt-in `triangles` tool in the family.
- common_neighbors_max: max steps, and answering the node itself: a skill should help.
- One shortest_path_via question took 178 s (still right): likely a thinking spiral; cap the output per step.

## v9 fixes and skills (2026-10-09; same 20 multi-step questions, router + hybrid, 1 run)

| run | accuracy | tokens / q | time / q |
|---|---|---|---|
| oracle + all tools (v8) | 65% | 23,698 | 45.6 s |
| router + hybrid (v8) | 80% | 17,866 | 38.1 s |
| router + hybrid + v9 fixes + skills (`ms_llm_hybrid_v9_skills`) | 85% | 15,877 | 30.0 s |

Per task with skills: shortest_path_via 5/5, hop_max_degree 5/5, common_neighbors_max **5/5** (was 2–3/5: the
playbook stopped "the node itself" and the step-limit runs), triangle_count 2/5. Both triangle misses had the
**right count** (390 and 730, checked against networkx): the verifier rejected them because the model kept the
list in its own variable, got no handle, and copied 2–5 triangles by hand. Lesson: an evidence check is only fair
if the evidence can be delivered. Fix: every rejection of missing or partial evidence now says how to send a long
list by handle (`verifiers.LONG_LIST_HOW`), and the playbook says to keep the list in `result` itself.
Rerun of the 5 triangle questions with that (`ms_triangles_v9b`): **5/5**, 9,546 tokens / q, 12.3 s (was 2/5,
36,843 tokens, 48.2 s). One question still needed five tries (a flat list, an empty one, an invented handle name)
before it sent the real handle. Skills and the v9 fixes were measured together: a skills-only ablation is still
to do, and all of this is 1 run.

## Open questions
- LLM confidence is self-reported and coarse (0.5 / 0.95 / 1 in the smoke test). Options: logprobs of the task
  (if Ollama exposes them), agreement over a few samples, or just calibrating the threshold on dev.
- Thinking on vs off for routing: an ablation, if routing accuracy turns out low.
- ~~Wiring into the runner~~ done: `--router`, `--tools`, `--phrasing paraphrase`, the `any` answer type for
  unknown routes (see `docs/architecture.md`, "Router and tool exposure").

## TypeSafe Jev (research, 2026-10-09; sources from public docs and press, not tested)
- A "System One" classifier from TypeSafe AI (launched 2026-09-15, generally available 2026-09-21, `jev-1.13.0`).
  API only (`POST https://api.typesafe.ai/v1/systemone`, PyPI `typesafe-sdk`, also a Pydantic AI model
  `typesafe:jev-latest`). Architecture and size are not disclosed.
- Typed questions (choice / score / yes-no) with label descriptions; returns probabilities per label, so it can't
  answer outside the label set. Confidence for a choice = (p_max − 1/n) / (1 − 1/n): how peaked, **not shown to be
  calibrated**; their docs say to set thresholds on your own data.
- Price $0.042 per 1M input tokens, output free; ~150 routing calls ≈ 90k tokens ≈ $0.004. Reported median
  latency ~76 ms (press, not the docs). No self-hosting; no training on user data, zero retention only for
  enterprise.
- Their own weakness list includes numbers, counting and literal reading: risky for labels that differ by a
  detail (`shortest_path` vs `shortest_path_via`, `hop_max_degree`). Classification only: parameters still need
  regex or the LLM.
- To test: an API key (the author's sign-up and approval), raw HTTP rather than a new dependency if possible,
  dev questions only, a reliability plot of confidence vs correctness.
- Comparable alternatives: zero-shot NLI (bart-large-mnli), GLiClass (local zero-shot), GLiNER (parameter
  extraction), SetFit (few-shot, ~8 examples per label), embedding nearest neighbour.
