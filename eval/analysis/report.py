"""Accuracy and cost report: one or more experiments side by side, with 95% bootstrap confidence intervals.

    uv run python -m eval.analysis.report degree_small_dev_v1
    uv run python -m eval.analysis.report with_verify no_verify --by-task
    uv run python -m eval.analysis.report with_verify no_verify --out report.md --plot pareto.png

Everything is computed from the results.jsonl files that eval/runner.py writes, so every number can be
reproduced from the logs. The tables are Markdown, so they paste straight into notes or the thesis.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

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
        "tokens": statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in done) if done else 0,
        "time": statistics.mean(r["latency_s"] for r in done) if done else 0,
    }


def markdown_table(header: list[str], lines: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in line) + " |" for line in lines]
    return "\n".join(out)


def table_line(label: list[str], s: dict) -> list:
    low, high = s["ci"]
    return [*label, s["questions"], s["runs"], f"{s['accuracy']:.1%}", f"{low:.1%} – {high:.1%}", f"{s['strict']:.1%}",
            f"{s['checked']:.0%}", s["no_answer"], s["rejected"], f"{s['tokens']:.0f}", f"{s['time']:.1f}s"]


COLUMNS = ["questions", "runs", "accuracy", "95% CI", "strict", "checked", "no answer", "rejected", "tokens / q", "time / q"]
LEGEND = ("accuracy = share of all answers (every run) that were correct; 95% CI = bootstrap over questions; "
          "strict = accuracy without answers rescued from text tool calls; "
          "checked = answers a checker actually looked at; no answer = runs that ended without one "
          "(cannot_answer, loop, max steps, error); rejected = answers a checker sent back; "
          "tokens / q = prompt + completion tokens per answer.")


def report(names: list[str], by_task: bool) -> tuple[str, dict]:
    experiments = {name: load_rows(name) for name in names}
    overview = [table_line([name, rows[0]["model"]], summarize(rows)) for name, rows in experiments.items()]
    text = ["## Experiments", "", markdown_table(["experiment", "model", *COLUMNS], overview)]

    if by_task:
        lines = []
        for name, rows in experiments.items():
            tasks = defaultdict(list)
            for r in rows:
                tasks[r["task"]].append(r)
            lines += [table_line([name, task], summarize(task_rows)) for task, task_rows in sorted(tasks.items())]
        text += ["", "## Per task", "", markdown_table(["experiment", "task", *COLUMNS], lines)]

    text += ["", LEGEND]
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
    parser.add_argument("--out", type=Path, help="also save the tables to this Markdown file")
    parser.add_argument("--plot", type=Path, help="save an accuracy-vs-tokens plot here (needs --extra viz)")
    args = parser.parse_args()

    text, summaries = report(args.names, args.by_task)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved {args.out}")
    if args.plot:
        plot(summaries, args.plot)
        print(f"Saved {args.plot}")
