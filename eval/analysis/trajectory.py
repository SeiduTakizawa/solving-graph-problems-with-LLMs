"""Trajectory analysis: did the model take the right *path* to its answer, not just get the right answer?

For a multi-step task we know which tool calls an ideal run makes. But a different route to the answer can be
just as valid (e.g. reading the way back from a path it already has), so a run is judged by its calls *and* its
answer, never by matching one expected sequence alone:

    ideal               exactly the needed calls, correct answer
    extra calls         the needed calls plus others, correct answer
    alternative path    not all needed calls, but a correct answer: read these, the reasoning may be sound
    right calls, wrong  the needed calls were made but the answer is wrong (e.g. the join went wrong)
    wrong path          not all needed calls and a wrong answer

plus flags for typical detours:

    shortcut     called shortest_path(source, target), which ignores the waypoint
    has_path leg used has_path for a leg (fine since 2026-10-08: it promises a shortest path; before, it only
                 promised "one such path", so runs before that relied on something the tool didn't guarantee)
    manual       explored by hand with get_neighbors

    uv run python -m eval.analysis.trajectory waypoint_small_dev
"""
import argparse
import json
from collections import Counter, defaultdict

from eval.analysis.report import RESULTS_DIR, load_rows
from harness.tools.graph_tools import TOOLS


def pair(a, b) -> frozenset:
    return frozenset((a, b))  # the graphs are undirected: shortest_path(7, 3) is as good as (3, 7)


def needed_calls(task: str, params: dict) -> list[tuple[str, frozenset]] | None:
    """The calls an ideal run makes, or None if the task has no known ideal path."""
    if task == "shortest_path_via":
        return [("shortest_path", pair(params["source"], params["via"])),
                ("shortest_path", pair(params["via"], params["target"]))]
    return None


def classify(task: str, params: dict, calls: list[tuple[str, dict]], correct: bool) -> dict:
    """`calls`: the run's graph tool calls in order, as (name, args); `correct`: whether the answer was right."""
    # has_path returns a shortest path too, so for the legs it counts as shortest_path.
    keyed = [("shortest_path" if name == "has_path" else name, pair(args.get("source"), args.get("target")))
             if name in ("shortest_path", "has_path") else (name, None) for name, args in calls]
    needed = needed_calls(task, params)
    made = set(keyed)
    if all(need in made for need in needed):
        if not correct:
            kind = "right calls, wrong"
        else:
            kind = "ideal" if len(keyed) == len(needed) else "extra calls"
    else:
        kind = "alternative path" if correct else "wrong path"
    legs = [pair(params["source"], params["via"]), pair(params["via"], params["target"])]
    return {
        "kind": kind,
        "shortcut": ("shortest_path", pair(params["source"], params["target"])) in made,
        "has_path_leg": any(name == "has_path" and pair(a.get("source"), a.get("target")) in legs
                            for name, a in calls),
        "manual": any(name == "get_neighbors" for name, _ in keyed),
        "calls": len(keyed),
    }


def graph_tool_calls(traces_path) -> dict[str, list[tuple[str, dict]]]:
    """run_id -> the graph tool calls of that run, in order (rescued text calls included, as they were run)."""
    calls = defaultdict(list)
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event["event"] == "tool_call" and event["name"] in TOOLS:
                calls[event["run_id"]].append((event["name"], event["args"]))
    return calls


def analyze(name: str) -> list[dict]:
    calls = graph_tool_calls(RESULTS_DIR / name / "traces.jsonl")
    out = []
    for row in load_rows(name):
        if needed_calls(row["task"], row["params"]) is None:
            continue
        run_calls = calls.get(row["run_id"], [])
        out.append({**classify(row["task"], row["params"], run_calls, row["correct"]), "correct": row["correct"],
                    "params": row["params"], "sequence": [f"{n}({', '.join(map(str, a.values()))})" for n, a in run_calls],
                    "answer": row["answer"], "status": row["status"]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="How did the model get to its answers?")
    parser.add_argument("name", help="experiment folder in results/harness_runs/")
    parser.add_argument("--runs", action="store_true", help="also list every run's tool calls")
    args = parser.parse_args()

    runs = analyze(args.name)
    if not runs:
        raise SystemExit("No runs of a task with a known ideal path (e.g. shortest_path_via) in this experiment.")
    print(f"{args.name}: {len(runs)} runs\n")
    print(f"{'path taken':18} {'runs':>5}")
    for kind in ("ideal", "extra calls", "alternative path", "right calls, wrong", "wrong path"):
        group = [r for r in runs if r["kind"] == kind]
        if group:
            print(f"{kind:18} {len(group):>5}")
    print(f"\ncorrect: {sum(r['correct'] for r in runs)}/{len(runs)} | flags: shortcut {sum(r['shortcut'] for r in runs)}, "
          f"has_path leg {sum(r['has_path_leg'] for r in runs)}, manual (get_neighbors) {sum(r['manual'] for r in runs)} | "
          f"graph tool calls per run: {sum(r['calls'] for r in runs) / len(runs):.1f}")
    print("most common sequences:")
    for seq, count in Counter(" > ".join(r["sequence"]) or "(no graph tools)" for r in runs).most_common(5):
        print(f"  {count:>3}x  {seq}")
    if args.runs:
        print()
        for r in runs:
            print(f"{'✓' if r['correct'] else '✗'} {r['kind']:18} {r['params']} -> {r['answer']}  |  {' > '.join(r['sequence'])}")
