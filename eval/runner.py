"""Run the agent on dataset questions and record whether each answer is correct.

Example (100 node-degree questions from the dev graphs, 3 runs):
    uv run python -m eval.runner --size small --n 100 --runs 3

Writes to results/harness_runs/<name>/:
    traces.jsonl   every event of every run (from harness.trace)
    results.jsonl  one line per question per run: answer, ground truth, correct, status, tokens, latency
"""
import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import networkx as nx

from harness.hello_agent import MODEL, run_agent
from harness.trace import Trace

ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "data" / "graphs" / "er"
QUESTIONS_DIR = ROOT / "data" / "graphs_questions" / "er"
RESULTS_DIR = ROOT / "results" / "harness_runs"

# Graphs 0-49 are for developing the harness, 50-99 are kept untouched for final results.
SPLITS = {"dev": range(0, 50), "test": range(50, 100)}


def load_degree_tasks(size: str, split: str, n: int) -> list[dict]:
    """Node-degree tasks, spread over the graphs of the split (1st question of every graph, then 2nd, ...).

    The dataset's questions contain the whole edge list; only the node is kept, so the graph never
    enters the prompt.
    """
    per_graph = []
    for graph_id in SPLITS[split]:
        questions = json.loads((QUESTIONS_DIR / size / f"{graph_id}.txt").read_text())["edgelist"]["node_degree"]
        nodes = [int(re.search(r"degree of node (\d+)\?$", q).group(1)) for q in questions]
        per_graph.append([(graph_id, node) for node in nodes])

    tasks = []
    for round_ in range(max(len(p) for p in per_graph)):
        for pairs in per_graph:
            if round_ < len(pairs):
                graph_id, node = pairs[round_]
                tasks.append({"graph_id": graph_id, "node": node,
                              "question": f"What is the degree of node {node}?"})
    return tasks[:n]


def run(args) -> Path:
    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_degree_tasks(args.size, args.split, args.n)
    graphs = {}

    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as results:
        for run_no in range(1, args.runs + 1):
            for i, task in enumerate(tasks, 1):
                graph_id = task["graph_id"]
                if graph_id not in graphs:
                    graphs[graph_id] = nx.read_adjlist(GRAPHS_DIR / args.size / f"{graph_id}.txt", nodetype=int)
                graph = graphs[graph_id]
                truth = graph.degree(task["node"])

                trace = Trace(out_dir / "traces.jsonl")
                start = time.time()
                try:
                    result = run_agent(task["question"], graph, trace=trace)
                    answer, status = result.answer, result.status
                except Exception as e:  # e.g. Ollama not running; record it and keep going
                    answer, status = None, f"error: {e}"
                end = _run_end(trace)

                row = {
                    "run": run_no, "run_id": trace.run_id, "model": MODEL, "size": args.size, "split": args.split,
                    **task, "truth": truth, "answer": answer, "correct": answer == truth, "status": status,
                    "steps": end.get("steps"), "prompt_tokens": end.get("prompt_tokens"),
                    "completion_tokens": end.get("completion_tokens"),
                    "latency_s": round(time.time() - start, 2),
                }
                results.write(json.dumps(row) + "\n")
                results.flush()
                print(f"run {run_no} [{i}/{len(tasks)}] graph {graph_id} node {task['node']}: "
                      f"{status} answer={answer} truth={truth} {'✓' if row['correct'] else '✗'}", flush=True)
    return out_dir


def _run_end(trace: Trace) -> dict:
    """The run_end event of this trace's run (empty if the run crashed before it)."""
    if not trace.path.exists():
        return {}
    with trace.path.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event["run_id"] == trace.run_id and event["event"] == "run_end":
                return event
    return {}


def summarize(out_dir: Path) -> None:
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").open(encoding="utf-8")]
    runs = sorted({r["run"] for r in rows})
    per_run = {run_no: [r for r in rows if r["run"] == run_no] for run_no in runs}
    accuracies = [statistics.mean(r["correct"] for r in per_run[run_no]) for run_no in runs]
    n_questions = max(len(run_rows) for run_rows in per_run.values())

    print(f"\n=== {out_dir.name}: {n_questions} questions x {len(runs)} runs, model {rows[0]['model']}")
    for run_no, acc in zip(runs, accuracies):
        done = len(per_run[run_no])
        note = f" (incomplete: {done}/{n_questions} questions)" if done < n_questions else ""
        print(f"run {run_no}: accuracy {acc:.1%}{note}")
    if len(runs) > 1:
        print(f"mean accuracy {statistics.mean(accuracies):.1%} (std {statistics.stdev(accuracies):.1%})")

    print("\nhow runs ended:", dict(Counter(r["status"] for r in rows).most_common()))
    wrong = [r for r in rows if r["status"] == "submitted" and not r["correct"]]
    print(f"submitted but wrong: {len(wrong)}")

    done = [r for r in rows if r["prompt_tokens"] is not None]
    if done:
        print(f"\nper question: {statistics.mean(r['steps'] for r in done):.1f} steps, "
              f"{statistics.mean(r['prompt_tokens'] for r in done):.0f} prompt + "
              f"{statistics.mean(r['completion_tokens'] for r in done):.0f} completion tokens, "
              f"{statistics.mean(r['latency_s'] for r in done):.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agent on node-degree questions.")
    parser.add_argument("--size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--split", default="dev", choices=list(SPLITS))
    parser.add_argument("--n", type=int, default=100, help="number of questions")
    parser.add_argument("--runs", type=int, default=3, help="repeat every question this many times")
    parser.add_argument("--name", default=None, help="results folder name (default: degree_<size>_<split>_<time>)")
    parser.add_argument("--summary-only", action="store_true", help="only summarize an existing --name")
    args = parser.parse_args()
    args.name = args.name or f"degree_{args.size}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"

    summarize(RESULTS_DIR / args.name if args.summary_only else run(args))
