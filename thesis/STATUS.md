# Thesis draft: status (2026-10-09)

Sources: `main.tex` (title page, abstracts, front matter), `chapters/*.tex`, `references.bib`.
Not compiled: no LaTeX on this machine (`latexmk`, `pdflatex`, `bibtex` not on PATH). Build with
`latexmk -pdf -interaction=nonstopmode main.tex` (pdflatex + bibtex; packages: geometry, booktabs, tabularx,
longtable, listings, tikz, todonotes, natbib, hyperref). The sources passed a script check (braces, environments,
labels/refs, cite keys, stray `_`/`$`/`#`), not a real compile, so expect a few fixes on the first build.

## Chapters

| chapter | status | notes |
|---|---|---|
| Abstract (English) | written | revise after M7 |
| Abstract (Greek) | TODO | for you to write |
| 1 Introduction | written | sub-questions not checked against the proposal (PDF missing) |
| 2 Background and related work | partial | GDS Agent detailed; GABench, GraphChain, GT Bench, GrAlgoBench, ProGraph, GraphTeam only one line each; Skianis et al. results missing |
| 3 Design | written | TikZ architecture figure; principle-5 caveat (BFS length check) flagged |
| 4 Implementation | written | as built on 2026-10-09 |
| 5 Experimental setup | written | dataset audit, configs A/B/H/C/NX, metrics, limitations |
| 6 Results so far | written | all experiments to date, each table with `% source:` |
| 7 Discussion and lessons | written | from the learning log |
| 8 Remaining work and timeline | written (plan) | timeline dates TODO |
| 9 Conclusion | partial (provisional) | |
| App. A Tools and task templates | written | generated from the code |
| App. B Prompt texts | written | generated from the code (PROMPT_VERSION v7) |

## Every \todo in the draft

1. main.tex: revise the abstract after the pilot, M4–M7.
2. main.tex: Greek abstract (needs babel greek / texlive-lang-greek).
3. ch1: check the sub-questions against `docs/thesis_proposal_options_ABC.pdf` (not in the repo).
4. ch2: expand GABench, GraphChain, GT Bench, GrAlgoBench, ProGraph, GraphTeam from the papers.
5. ch2: headline results of Skianis et al. 2024.
6. ch3: decide whether the BFS length check in `verify_shortest_path` / `verify_path_via` is OK under principle 5.
7. ch3: per-run dollar budget not implemented.
8. ch4: open decision, `mcp` package as a dependency.
9. ch4: open decision, Docker vs OrbStack/Colima on macOS.
10. ch5: majority-answer baseline column in the report.
11. ch5: fix and report the sampling temperature.
12. ch5: fix hosted models and API access.
13. ch6: rerun the combine setups with a second model at 16k.
14. ch8: scope of M3 step 6 (close-out).
15. ch8: timeline dates and submission deadline.
16. ch9: rewrite the conclusion after M7.

Bibliography: every entry except Skianis et al. 2024 has `note = {TODO: verify}` (titles, authors, years not in the
repo). ReAct, MCP, LiteLLM, NetworkX and Claude Code have no id or URL in the repo at all.

## Numbers I could not source (left out or marked TODO)

- Headline accuracies of Skianis et al. 2024 (not in the repo docs).
- Anything about GABench, GraphChain, GT Bench, GrAlgoBench, ProGraph, GraphTeam beyond the one-line notes in CLAUDE.md
  (e.g. GraphChain's "45 networkx APIs" is the only number used).
- The proposal PDF (`docs/thesis_proposal_options_ABC.pdf`) is referenced in CLAUDE.md but is not in the repository.
- Which machine the `code_*` runs ran on is not recorded; the draft does not state it.
- The multi-step probe numbers (distances_from / neighborhood) come from console output recorded in the learning log;
  those runs were not traced.

## Things that looked contradictory (worth a look)

- CLAUDE.md "Current status" and the learning log (2026-10-08) give the combine results as "tools only 0–4/15, with
  code 8–12/15"; those are from the first combine round, which was hit by the 4k context. The valid `combine16k_*`
  runs give A 2/15, B 8/15, H 14/15, C 14/15, NX 13/15 (qwen3.5:9b). `docs/gds_agent_findings.md` already uses the
  16k numbers (2/15; "8 to 14 of 15"). The learning log has no entry for the combine16k results yet.
- The learning log says the CODE_HINT "hurt qwen3"; that observation is from the invalid first round
  (combine_qwen3_* traces reach 4,034–4,077 prompt tokens), so the thesis does not use it as evidence.
- Principle 5 says verifiers never recompute, but `verify_shortest_path` and `verify_path_via` compute the optimal
  length with `nx.shortest_path_length`. The draft flags this rather than hiding it.
- `--model sonnet` resolved to `claude-sonnet-5-5` in `claude_code_vs_harness_large` but to `claude-sonnet-5` in
  `claude_code_mcp_demo`; the two Claude Code previews used different model versions.
- `harness/models.py` still has `DEFAULT_MODEL = "ollama_chat/qwen3:8b"` while the combine runs use qwen3.5:9b (pilot
  not done yet, so this is expected, just worth noting).
- `degree_small_dev_v1` is reported as "100 questions, 3 runs" by the report, but run 3 stopped at 12 questions
  (212 answers total).
