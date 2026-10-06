"""Run the agent on dataset questions and record whether each answer is correct.

Examples (dev graphs, 3 runs):
    uv run python -m eval.runner --task node_degree --size small --n 100
    uv run python -m eval.runner --task all --size small --n 30
    uv run python -m eval.runner --task shortest_path --no-verify     # ablation: no verifier

Writes to results/harness_runs/<name>/:
    traces.jsonl   every event of every run (from harness.trace)
    results.jsonl  one line per question per run: answer, reference answer, correct, status, tokens, latency
"""
import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from eval.tasks import SPLITS, is_correct, load_graph, load_tasks, reference_answer
from harness.loop import AgentConfig, run_agent
from harness.models import DEFAULT_MODEL
from harness.tasks import TASKS
from harness.tools.graph_tools import GRAPH_TOOLS
from harness.trace import Trace

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "harness_runs"


def run(args) -> Path:
    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    task_names = sorted(TASKS) if args.task == "all" else [args.task]
    items = [item for name in task_names for item in load_tasks(name, args.size, args.split, args.n)]
    graphs = {}
    config = AgentConfig(model=args.model, graph_tools=tuple(
        t["function"]["name"] for t in GRAPH_TOOLS if t["function"]["name"] not in args.without_tool))

    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as results:
        for run_no in range(1, args.runs + 1):
            for i, item in enumerate(items, 1):
                task, params, graph_id = TASKS[item["task"]], item["params"], item["graph_id"]
                if graph_id not in graphs:
                    graphs[graph_id] = load_graph(args.size, graph_id)
                graph = graphs[graph_id]
                verify = None
                if task.verify and not args.no_verify:
                    verify = (lambda answer, graph=graph, params=params, check=task.verify, **evidence:
                              check(graph, params, answer, **evidence))

                trace = Trace(out_dir / "traces.jsonl")
                start = time.time()
                try:
                    result = run_agent(item["question"], graph, answer_type=task.answer_type, verify=verify,
                                       config=config, trace=trace)
                    answer, status, rescued, rejected = result.answer, result.status, result.rescued, result.rejected
                    evidence = result.evidence
                except Exception as e:  # e.g. Ollama not running; record it and keep going
                    answer, status, rescued, rejected, evidence = None, f"error: {e}", 0, 0, None
                end = _run_end(trace)
                correct = is_correct(task.name, graph, params, answer)

                row = {
                    "run": run_no, "run_id": trace.run_id, "model": args.model, "size": args.size, "split": args.split,
                    **item, "answer_type": task.answer_type, "verified": verify is not None,
                    "without_tools": args.without_tool,
                    "reference": reference_answer(task.name, graph, params), "answer": answer, "evidence": evidence, "correct": correct,
                    "status": status, "rescued": rescued, "rejected": rejected,
                    "steps": end.get("steps"), "prompt_tokens": end.get("prompt_tokens"),
                    "completion_tokens": end.get("completion_tokens"),
                    "latency_s": round(time.time() - start, 2),
                }
                results.write(json.dumps(row) + "\n")
                results.flush()
                print(f"run {run_no} [{i}/{len(items)}] {task.name} graph {graph_id} {params}: "
                      f"{status} answer={answer} {'✓' if correct else '✗'}", flush=True)
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
    for r in rows:
        r.setdefault("task", "node_degree")  # results from before the task registry were all node degree
    runs = sorted({r["run"] for r in rows})
    per_run = {run_no: [r for r in rows if r["run"] == run_no] for run_no in runs}
    n_questions = max(len(run_rows) for run_rows in per_run.values())

    print(f"\n=== {out_dir.name}: {n_questions} questions x {len(runs)} runs, model {rows[0]['model']}")
    accuracies = [statistics.mean(r["correct"] for r in per_run[run_no]) for run_no in runs]
    for run_no, acc in zip(runs, accuracies):
        done = len(per_run[run_no])
        note = f" (incomplete: {done}/{n_questions} questions)" if done < n_questions else ""
        print(f"run {run_no}: accuracy {acc:.1%}{note}")
    if len(runs) > 1:
        print(f"mean accuracy {statistics.mean(accuracies):.1%} (std {statistics.stdev(accuracies):.1%})")

    # Per task. Lenient accuracy counts answers rescued from text; strict accuracy does not.
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    print(f"\n{'task':28} {'runs':>5} {'accuracy':>9} {'strict':>7} {'wrong':>6} {'no ans':>7} {'rescued':>8}"
          f" {'rejected':>9} {'tokens':>7} {'time':>6}")
    for task, task_rows in sorted(by_task.items()):
        done = [r for r in task_rows if r.get("prompt_tokens") is not None]
        print(f"{task:28} {len(task_rows):>5}"
              f" {statistics.mean(r['correct'] for r in task_rows):>9.1%}"
              f" {statistics.mean(r['correct'] and not r.get('rescued') for r in task_rows):>7.1%}"
              f" {sum(r['status'] == 'submitted' and not r['correct'] for r in task_rows):>6}"
              f" {sum(r['status'] != 'submitted' for r in task_rows):>7}"
              f" {sum(1 for r in task_rows if r.get('rescued')):>8}"
              f" {sum(r.get('rejected', 0) for r in task_rows):>9}"
              f" {statistics.mean(r['prompt_tokens'] + r['completion_tokens'] for r in done) if done else 0:>7.0f}"
              f" {statistics.mean(r['latency_s'] for r in done) if done else 0:>5.1f}s")
    print("\nwrong = submitted but incorrect, no ans = cannot_answer / loop_detected / max_steps / error,"
          "\nrescued = runs with a tool call rescued from text, rejected = answers sent back by the verifier,"
          "\ntokens = prompt + completion per question")
    print("\nhow runs ended:", dict(Counter(r["status"] for r in rows).most_common()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agent on dataset questions.")
    parser.add_argument("--task", default="node_degree", choices=sorted(TASKS) + ["all"])
    parser.add_argument("--size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--split", default="dev", choices=list(SPLITS))
    parser.add_argument("--n", type=int, default=100, help="number of questions per task")
    parser.add_argument("--runs", type=int, default=3, help="repeat every question this many times")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="any LiteLLM model name")
    parser.add_argument("--no-verify", action="store_true", help="run without verifiers (ablation)")
    parser.add_argument("--without-tool", action="append", default=[], metavar="TOOL",
                        choices=[t["function"]["name"] for t in GRAPH_TOOLS],
                        help="hide this graph tool from the agent (ablation); can be repeated")
    parser.add_argument("--name", default=None, help="results folder name (default: <task>_<size>_<split>_<time>)")
    parser.add_argument("--summary-only", action="store_true", help="only summarize an existing --name")
    args = parser.parse_args()
    args.name = args.name or f"{args.task}_{args.size}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"

    summarize(RESULTS_DIR / args.name if args.summary_only else run(args))
