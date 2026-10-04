# CLAUDE.md

Code for the paper "Graph Reasoning with Large Language Models via Pseudo-code Prompting". See README.md for the layout.

## Commands
- Run anything with `uv run python ...`; add dependencies with `uv add <pkg>` (not pip) and commit `uv.lock`.
- Single run: `uv run python run_experiments.py --problem cycle_check --size small --method 1_shot_pseudo --model gpt-4o`

## Rules
- Real runs call the OpenAI API and cost money. Test by stubbing `graph_reasoning.experiment_runner.get_openai_response` and pointing `graph_reasoning.experiment_config.RESULTS_DIR` at a temp dir. Ask before any real run.
- `scripts/generate_dataset.py` overwrites the committed dataset in `data/`. Don't run it without asking.
- `legacy/run_experiments_original.py` is the paper's original script, kept for reference. Make changes in `graph_reasoning/`.
- Only `er` graphs exist in `data/graphs/`; `topological_sorting` (dag) and `bipartite` are skipped until their data is generated.
- `--all` skips any configuration that already has a summary in `results/experiments/`. Delete that file to re-run it.
- Answer parsing in `graph_reasoning/utils.py` is loose string matching (e.g. connectivity matches `'No'` inside "Node"). Keep this in mind when reading accuracy numbers.

## Git
- The `origin` remote is HTTPS with no stored credentials. Push over SSH: `git push git@github.com:SeiduTakizawa/solving-graph-problems-with-LLMs.git main`
