# Graph Reasoning via LLMs with Pseudocode Prompting

This is the official repository for the paper: "Graph Reasoning with Large Language Models via Pseudo-code Prompting"

## Project layout

```
run_experiments.py        Main CLI entry point
graph_reasoning/          Experiment pipeline (package)
  experiment_config.py      ExperimentConfig, path constants, batch configs
  file_io.py                Loading graphs/questions, saving results
  prompt_builder.py         Builds prompts per problem and method
  experiment_runner.py      Runs experiments and queries the model
  result_processor.py       Ground truth and answer validation
  graph_algorithms.py       Reference graph algorithms (ground truth)
  utils.py                  OpenAI client and answer-parsing helpers
data/
  graphs/                   Graphs as adjacency lists (er/{small,medium,large})
  graphs_questions/         Questions per graph (JSON, adj + edgelist forms)
  pseudocodes/              Pseudocode and 1-shot prompts per problem
results/
  exp_results/              Per-graph model answers and correctness
  experiments/              Per-run accuracy summaries (created on run)
scripts/
  generate_dataset.py       Regenerates data/graphs and data/graphs_questions
  visualize_graph.py        Draws a single graph
  dfs_cycle_demo.py         Traced DFS cycle-detection walkthrough
legacy/
  run_experiments_original.py   Original single-file script, kept for reference
docs/                     Pipeline, prompt and refactoring notes
```

## Setup

With [uv](https://docs.astral.sh/uv/) (installs the exact versions in `uv.lock`):

```
uv sync                     # add --extra viz for scripts/visualize_graph.py
export OPENAI_API_KEY=...
uv run python run_experiments.py --help
```

Or with pip (versions not pinned):

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # or: pip install -e '.[viz]'
```

## Run experiments

Run a single experiment:

`python3 run_experiments.py --problem cycle_check --size small --method 1_shot_pseudo --model gpt-4o`

Run all experiments:

`python3 run_experiments.py --all`

Methods: `none`, `cot`, `bag`, `alg`, `default_1_shot`, `1_shot_pseudo`. Add `--structured` for JSON step-by-step output.

## Citation

If you use our dataset or code from this repository in your work, please cite it as follows:

```bibtex
@misc{skianis2024graphreasoninglargelanguage,
      title={Graph Reasoning with Large Language Models via Pseudo-code Prompting},
      author={Konstantinos Skianis and Giannis Nikolentzos and Michalis Vazirgiannis},
      year={2024},
      eprint={2409.17906},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.17906},
}
```
