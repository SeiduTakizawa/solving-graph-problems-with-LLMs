"""Accuracy and cost report: one or more experiments side by side, with 95% bootstrap confidence intervals.

    uv run python -m eval.analysis.report degree_small_dev_v1
    uv run python -m eval.analysis.report with_verify no_verify --by-task
    uv run python -m eval.analysis.report with_verify no_verify --out report.md --plot pareto.png
    uv run python -m eval.analysis.report combine_qwen35_B combine_qwen35_C --code   # how run_python was used
    uv run python -m eval.analysis.report m2_large_verify --process --by-task   # right tools, right arguments?

Everything is computed from the results.jsonl files that eval/runner.py writes, so every number can be
reproduced from the logs. The tables are Markdown, so they paste straight into notes or the thesis.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from eval.analysis.process import process_metrics

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "harness_runs"


def load_rows(name: str) -> list[dict]:
    path = RESULTS_DIR / name / "results.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    for r in rows:  # results from before the task registry: all node degree, with the node as its own field
        r.setdefault("task", "node_degree")
        r.setdefault("params", {"node": r["node"]} if "node" in r else {})
    return rows


def question_key(row: dict) -> tuple:
    """The same question across runs: same task, same graph, same parameters."""
    return row["task"], row["graph_id"], json.dumps(row["params"], sort_keys=True)


def bootstrap_ci(rows: list[dict], resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for accuracy. Resamples questions (not single answers): the runs of one
    question are not independent, so a question and all its runs go in or out together."""
    by_question = defaultdict(list)
    for r in rows:
        by_question[question_key(r)].append(r["correct"])
    groups = list(by_question.values())
    rng = random.Random(seed)  # fixed seed: the same logs always give the same interval
    accuracies = []
    for _ in range(resamples):
        sample = [rng.choice(groups) for _ in groups]
        answers = [correct for group in sample for correct in group]
        accuracies.append(sum(answers) / len(answers))
    accuracies.sort()
    return accuracies[int(0.025 * resamples)], accuracies[int(0.975 * resamples) - 1]


def was_checked(row: dict) -> bool:
    """Whether a checker actually looked at this answer. Not the case when checkers were off, the task has
    none, the run ended without an answer, or the answer was a "no" (which the checkers accept unseen)."""
    if not row.get("verified") or row["status"] != "submitted":
        return False
    return not (row.get("answer_type", "").startswith("yes_no_with_") and row["answer"] is False)


# --- Code use (from traces.jsonl): did the model write code, did the code fail, did it mix code with tools? ---

ANSWER_TOOLS = {"submit_answer", "cannot_answer"}


def load_tool_calls(name: str) -> dict[str, list[dict]]:
    """run_id -> the tool_call and nudge events of that run, in order."""
    by_run = defaultdict(list)
    path = RESULTS_DIR / name / "traces.jsonl"
    if path.exists():
        for line in path.open(encoding="utf-8"):
            event = json.loads(line)
            if event["event"] in ("tool_call", "nudge"):
                by_run[event["run_id"]].append(event)
    return by_run


def code_use(rows: list[dict], events: dict[str, list[dict]]) -> dict:
    """The numbers of one code-use line. A code error is a run_python call whose reply has an "error"."""
    code_calls = tool_calls = code_errors = with_code = mixed = empty = 0
    for r in rows:
        run = events.get(r.get("run_id"), [])
        code = [e for e in run if e["event"] == "tool_call" and e["name"] == "run_python"]
        tools = [e for e in run if e["event"] == "tool_call" and e["name"] not in ANSWER_TOOLS | {"run_python"}]
        code_calls += len(code)
        tool_calls += len(tools)
        code_errors += sum(bool((e.get("result") or {}).get("error")) for e in code)
        with_code += bool(code)
        mixed += bool(code and tools)
        empty += sum(e["event"] == "nudge" and e.get("kind") == "empty_reply" for e in run)
    n = len(rows)
    return {"code_offered": any(r.get("python") for r in rows), "with_code": with_code / n, "mixed": mixed / n,
            "code_calls": code_calls / n, "code_errors": code_errors / code_calls if code_calls else 0.0,
            "tool_calls": tool_calls / n, "empty_replies": empty}


def code_line(label: list[str], c: dict) -> list:
    if not c["code_offered"]:
        return [*label, "not offered", "–", "–", "–", f"{c['tool_calls']:.1f}", c["empty_replies"]]
    return [*label, f"{c['with_code']:.0%}", f"{c['mixed']:.0%}", f"{c['code_calls']:.1f}", f"{c['code_errors']:.0%}",
            f"{c['tool_calls']:.1f}", c["empty_replies"]]


CODE_COLUMNS = ["used code", "code + tools", "code calls / q", "code errors", "tool calls / q", "empty replies"]
CODE_LEGEND = ("used code = questions with at least one run_python call; code + tools = questions that used both "
               "run_python and a graph tool; code errors = share of run_python calls that ended in an error; "
               "tool calls / q = graph tool calls (not run_python, submit_answer or cannot_answer); "
               "empty replies = replies with no text and no tool call (counted from v6 on, when they got their "
               "own nudge).")


# --- Process metrics (eval/analysis/process.py): the right tools with the right arguments? ---

def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def process_use(rows: list[dict], events: dict[str, list[dict]]) -> dict | None:
    """Mean tool precision / recall / F1 / params / exact over the answers whose task has expected calls."""
    scored = []
    for r in rows:
        calls = [e for e in events.get(r.get("run_id"), []) if e["event"] == "tool_call"]
        m = process_metrics(r["task"], r.get("params") or {}, calls)
        if m is not None:
            scored.append((r, m))
    if not scored:
        return None
    return {
        **{key: mean([m[key] for _, m in scored]) for key in ("precision", "recall", "f1", "params")},
        "exact": statistics.mean(m["exact"] for _, m in scored),
        # The GDS Agent paper found tool metrics hardly predict the answer: compare F1 of right and wrong answers.
        "f1_right": mean([m["f1"] for r, m in scored if r["correct"]]),
        "f1_wrong": mean([m["f1"] for r, m in scored if not r["correct"]]),
    }


def process_line(label: list[str], p: dict | None) -> list:
    if p is None:
        return [*label, *["–"] * len(PROCESS_COLUMNS)]
    show = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    return [*label, *(show(p[k]) for k in ("precision", "recall", "f1", "params")), f"{p['exact']:.0%}",
            show(p["f1_right"]), show(p["f1_wrong"])]


PROCESS_COLUMNS = ["tool precision", "tool recall", "tool F1", "params", "exact", "F1 if right", "F1 if wrong"]
PROCESS_LEGEND = ("Process metrics as in the GDS Agent benchmark (eval/analysis/process.py; expected calls per task in "
                  "eval/tasks.py): recall = expected calls made; precision = expected calls made / all tool calls "
                  "(repeats and exploring lower it); params = right arguments in the expected calls made directly; "
                  "exact = all expected calls and nothing else. run_python fills an expected call when its code calls "
                  "that tool.")


def total_tokens(row: dict) -> int:
    """The agent's tokens plus the router's (M4): routing is part of answering the question."""
    return (row["prompt_tokens"] + row["completion_tokens"] + (row.get("route_prompt_tokens") or 0)
            + (row.get("route_completion_tokens") or 0))


def route_use(rows: list[dict]) -> dict | None:
    """How the questions were routed (rows without a router are oracle runs from before M4)."""
    if not any(r.get("router") for r in rows):
        return None
    tools = [len(r["graph_tools_offered"]) for r in rows if r.get("graph_tools_offered") is not None]
    return {"router": rows[0].get("router"), "tools": rows[0].get("tools_mode"),
            "routed_right": statistics.mean(bool(r.get("route_right")) for r in rows),
            "unknown": sum(r.get("routed_task") == "unknown" for r in rows),
            "route_tokens": statistics.mean((r.get("route_prompt_tokens") or 0) + (r.get("route_completion_tokens") or 0)
                                            for r in rows),
            "route_time": statistics.mean(r.get("route_latency_s") or 0 for r in rows),
            "start_tools": statistics.mean(tools) if tools else None}


def route_line(label: list[str], u: dict | None) -> list:
    if u is None:
        return [*label, *["–"] * len(ROUTE_COLUMNS)]
    start = "all" if u["start_tools"] is None else f"{u['start_tools']:.1f}"
    return [*label, u["router"], u["tools"], f"{u['routed_right']:.0%}", u["unknown"], f"{u['route_tokens']:.0f}",
            f"{u['route_time']:.2f}s", start]


ROUTE_COLUMNS = ["router", "tools", "routed right", "unknown", "router tokens / q", "router time / q",
                 "graph tools at start"]
ROUTE_LEGEND = ("Routing (M4): routed right = the router's task and parameters match the dataset's (oracle: 100%); "
                "unknown = questions it couldn't place (run with all tools and no verifier); router tokens are "
                "included in tokens / q above; graph tools at start = how many graph tools the agent was shown "
                "before any more_tools call (all = every default tool).")


def summarize(rows: list[dict]) -> dict:
    """The numbers of one table line."""
    done = [r for r in rows if r.get("prompt_tokens") is not None]  # runs that crashed have no token counts
    low, high = bootstrap_ci(rows)
    return {
        "answers": len(rows),
        "questions": len({question_key(r) for r in rows}),
        "runs": len({r["run"] for r in rows}),
        "accuracy": statistics.mean(r["correct"] for r in rows),
        "strict": statistics.mean(r["correct"] and not r.get("rescued") for r in rows),
        "ci": (low, high),
        "checked": statistics.mean(was_checked(r) for r in rows),
        "no_answer": sum(r["status"] != "submitted" for r in rows),
        "rejected": sum(r.get("rejected", 0) for r in rows),
        "compacted": sum(bool(r.get("compactions")) for r in rows),
        "tokens": statistics.mean(total_tokens(r) for r in done) if done else 0,
        "time": statistics.mean(r["latency_s"] for r in done) if done else 0,
    }


def markdown_table(header: list[str], lines: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in line) + " |" for line in lines]
    return "\n".join(out)


def table_line(label: list[str], s: dict) -> list:
    low, high = s["ci"]
    return [*label, s["questions"], s["runs"], f"{s['accuracy']:.1%}", f"{low:.1%} – {high:.1%}", f"{s['strict']:.1%}",
            f"{s['checked']:.0%}", s["no_answer"], s["rejected"], s["compacted"], f"{s['tokens']:.0f}", f"{s['time']:.1f}s"]


COLUMNS = ["questions", "runs", "accuracy", "95% CI", "strict", "checked", "no answer", "rejected", "compacted",
           "tokens / q", "time / q"]
LEGEND = ("accuracy = share of all answers (every run) that were correct; 95% CI = bootstrap over questions; "
          "strict = accuracy without answers rescued from text tool calls; "
          "checked = answers a checker actually looked at; no answer = runs that ended without one "
          "(cannot_answer, loop, max steps, context full, error); rejected = answers a checker sent back; "
          "compacted = runs whose conversation was shortened to fit the context window; "
          "tokens / q = prompt + completion tokens per answer.")


def by_task_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    tasks = defaultdict(list)
    for r in rows:
        tasks[r["task"]].append(r)
    return sorted(tasks.items())


def report(names: list[str], by_task: bool, code: bool = False, process: bool = False,
           route: bool = False) -> tuple[str, dict]:
    experiments = {name: load_rows(name) for name in names}
    overview = [table_line([name, rows[0]["model"]], summarize(rows)) for name, rows in experiments.items()]
    text = ["## Experiments", "", markdown_table(["experiment", "model", *COLUMNS], overview)]

    if by_task:
        lines = []
        for name, rows in experiments.items():
            lines += [table_line([name, task], summarize(task_rows)) for task, task_rows in by_task_rows(rows)]
        text += ["", "## Per task", "", markdown_table(["experiment", "task", *COLUMNS], lines)]

    text += ["", LEGEND]

    if code:
        events = {name: load_tool_calls(name) for name in names}
        lines = [code_line([name], code_use(rows, events[name])) for name, rows in experiments.items()]
        text += ["", "## Code use", "", markdown_table(["experiment", *CODE_COLUMNS], lines)]
        if by_task:
            lines = [code_line([name, task], code_use(task_rows, events[name]))
                     for name, rows in experiments.items() for task, task_rows in by_task_rows(rows)]
            text += ["", markdown_table(["experiment", "task", *CODE_COLUMNS], lines)]
        text += ["", CODE_LEGEND]

    if process:
        events = {name: load_tool_calls(name) for name in names}
        lines = [process_line([name], process_use(rows, events[name])) for name, rows in experiments.items()]
        text += ["", "## Process", "", markdown_table(["experiment", *PROCESS_COLUMNS], lines)]
        if by_task:
            lines = [process_line([name, task], process_use(task_rows, events[name]))
                     for name, rows in experiments.items() for task, task_rows in by_task_rows(rows)]
            text += ["", markdown_table(["experiment", "task", *PROCESS_COLUMNS], lines)]
        text += ["", PROCESS_LEGEND]

    if route:
        lines = [route_line([name], route_use(rows)) for name, rows in experiments.items()]
        text += ["", "## Routing", "", markdown_table(["experiment", *ROUTE_COLUMNS], lines)]
        if by_task:
            lines = [route_line([name, task], route_use(task_rows))
                     for name, rows in experiments.items() for task, task_rows in by_task_rows(rows)]
            text += ["", markdown_table(["experiment", "task", *ROUTE_COLUMNS], lines)]
        text += ["", ROUTE_LEGEND]
    return "\n".join(text), {name: summarize(rows) for name, rows in experiments.items()}


def plot(summaries: dict, path: Path) -> None:
    """Accuracy against tokens per question, one point per experiment: up and to the left is better."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # write a file, no window
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("The plot needs matplotlib: run `uv sync --extra viz` first.")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, s in summaries.items():
        low, high = s["ci"]
        ax.errorbar(s["tokens"], s["accuracy"], yerr=[[s["accuracy"] - low], [high - s["accuracy"]]],
                    fmt="o", capsize=4, label=name)
    ax.set_xlabel("tokens per question (prompt + completion)")
    ax.set_ylabel("accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    # Both axes from 0: a zoomed-in axis makes 99.5% vs 100% look like a big gap.
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(s["tokens"] for s in summaries.values()) * 1.1)
    ax.legend(fontsize=8, loc="lower right")  # a legend, not labels next to the points: close points overlap
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy and cost report for one or more experiments.")
    parser.add_argument("names", nargs="+", help="experiment folders in results/harness_runs/")
    parser.add_argument("--by-task", action="store_true", help="also show one line per task")
    parser.add_argument("--code", action="store_true", help="also show how run_python was used (reads traces.jsonl)")
    parser.add_argument("--process", action="store_true",
                        help="also show tool precision / recall / F1 and parameter match (reads traces.jsonl)")
    parser.add_argument("--route", action="store_true", help="also show how questions were routed (M4)")
    parser.add_argument("--out", type=Path, help="also save the tables to this Markdown file")
    parser.add_argument("--plot", type=Path, help="save an accuracy-vs-tokens plot here (needs --extra viz)")
    args = parser.parse_args()

    text, summaries = report(args.names, args.by_task, args.code, args.process, args.route)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved {args.out}")
    if args.plot:
        plot(summaries, args.plot)
        print(f"Saved {args.plot}")
