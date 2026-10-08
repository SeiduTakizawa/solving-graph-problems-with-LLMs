"""Run the agent on dataset questions and record whether each answer is correct.

Examples (dev graphs, 3 runs):
    uv run python -m eval.runner --task node_degree --size small --n 100
    uv run python -m eval.runner --task all --size small --n 30
    uv run python -m eval.runner --task shortest_path --no-verify     # ablation: no verifier
    uv run python -m eval.runner --task shortest_path_via --python tools --code-only   # force code: no direct tools

Writes to results/harness_runs/<name>/:
    traces.jsonl   every event of every run (from harness.trace)
    results.jsonl  one line per question per run: answer, reference answer, correct, status, tokens, latency
"""
import argparse
import json
import time
from collections import Counter
from functools import partial
from pathlib import Path

from eval.analysis.report import RESULTS_DIR, load_rows, report
from eval.tasks import SPLITS, is_correct, load_graph, load_tasks, reference_answer
from harness.loop import AgentConfig, RunResult, run_agent
from harness.models import DEFAULT_MODEL
from harness.tasks import TASKS
from harness.tools.graph_tools import TOOLS
from harness.trace import Trace

def run(args) -> Path:
    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    task_names = sorted(TASKS) if args.task == "all" else [args.task]
    items = [item for name in task_names for item in load_tasks(name, args.size, args.split, args.n)]
    graphs = {}
    # --code-only offers no graph tool directly: run_python is the only way in (its code can still call the tools).
    graph_tools = () if args.code_only else tuple(name for name in TOOLS if name not in args.without_tool)
    config = AgentConfig(model=args.model, graph_tools=graph_tools, python=args.python, code_hint=args.code_hint)

    total, done, started = args.runs * len(items), 0, time.time()
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as results:
        for run_no in range(1, args.runs + 1):
            for i, item in enumerate(items, 1):
                task, params, graph_id = TASKS[item["task"]], item["params"], item["graph_id"]
                if graph_id not in graphs:
                    graphs[graph_id] = load_graph(args.size, graph_id)
                graph = graphs[graph_id]
                # verify(answer, **evidence): the task's verifier with this question's graph and parameters filled in
                verify = partial(task.verify, graph, params) if task.verify and not args.no_verify else None

                trace = Trace(out_dir / "traces.jsonl")
                start = time.time()
                try:
                    result = run_agent(item["question"], graph, answer_type=task.answer_type, verify=verify,
                                       config=config, trace=trace)
                except Exception as e:  # e.g. Ollama not running; record it and keep going
                    result = RunResult(None, f"error: {e}", [])
                correct = is_correct(task.name, graph, params, result.answer)
                crashed = result.status.startswith("error")  # no token counts for these

                row = {
                    "run": run_no, "run_id": trace.run_id, "model": args.model, "size": args.size, "split": args.split,
                    **item, "answer_type": task.answer_type, "verified": verify is not None,
                    "without_tools": args.without_tool, "python": args.python, "code_only": args.code_only, "code_hint": args.code_hint,
                    "reference": reference_answer(task.name, graph, params), "answer": result.answer,
                    "evidence": result.evidence, "correct": correct,
                    "status": result.status, "rescued": result.rescued, "rejected": result.rejected,
                    "compactions": result.compactions,
                    "steps": None if crashed else result.steps,
                    "prompt_tokens": None if crashed else result.prompt_tokens,
                    "completion_tokens": None if crashed else result.completion_tokens,
                    "latency_s": round(time.time() - start, 2),  # wall clock: model + tools + verifier
                }
                results.write(json.dumps(row) + "\n")
                results.flush()
                done += 1
                print(f"run {run_no} [{i}/{len(items)}] {task.name} graph {graph_id} {params}: "
                      f"{result.status} answer={_short(result.answer)} {'✓' if correct else '✗'}  "
                      f"{_progress(done, total, started)}", flush=True)
    return out_dir


def _progress(done: int, total: int, started: float) -> str:
    """e.g. "12/108, 2m 05s left, ends ~14:32": the estimate is the average time so far x answers left."""
    left = (time.time() - started) / done * (total - done)
    minutes, seconds = divmod(int(left), 60)
    hours, minutes = divmod(minutes, 60)
    left_text = f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"
    return f"| {done}/{total}, {left_text} left, ends ~{time.strftime('%H:%M', time.localtime(time.time() + left))}"


def _short(answer) -> str:
    """Long answers (a 49-edge MST) would flood the progress output."""
    text = json.dumps(answer)
    return text if len(text) <= 40 else text[:37] + "..."


def summarize(name: str) -> None:
    """The same tables as eval/analysis/report.py, plus how the runs ended."""
    text, _ = report([name], by_task=True)
    print("\n" + text)
    print("\nhow runs ended:", dict(Counter(r["status"] for r in load_rows(name)).most_common()))


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
                        choices=list(TOOLS),
                        help="hide this graph tool from the agent (ablation); can be repeated")
    parser.add_argument("--python", choices=["tools", "networkx"], default=None,
                        help="offer run_python (needs Docker): code that calls our tools, or also networkx")
    parser.add_argument("--code-hint", action="store_true",
                        help="tell the model when to combine tools and code (harness/prompts.py CODE_HINT); needs --python")
    parser.add_argument("--code-only", action="store_true",
                        help="offer only run_python, no graph tools directly (needs --python); forces code")
    parser.add_argument("--name", default=None, help="results folder name (default: <task>_<size>_<split>_<time>)")
    parser.add_argument("--summary-only", action="store_true", help="only summarize an existing --name")
    args = parser.parse_args()
    if (args.code_only or args.code_hint) and not args.python:
        parser.error("--code-only and --code-hint need --python tools or --python networkx")
    args.name = args.name or f"{args.task}_{args.size}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"

    if not args.summary_only:
        run(args)
    summarize(args.name)
