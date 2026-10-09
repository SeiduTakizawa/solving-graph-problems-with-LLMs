"""Process metrics: did the model call the right tools with the right arguments, not only get the right answer?

Same definitions as the GDS Agent benchmark (github.com/brs96/gds-agent-benchmarks, evaluate_benchmark.py), so the
numbers can be put next to the ones in its paper (arXiv 2508.20637). The expected calls per task are in
eval/tasks.py (EXPECTED_CALLS); each expected call is a "slot" that one of several equivalent tools can fill.

  recall     = expected calls that were made / expected calls
  precision  = expected calls that were made / tool calls made (so repeating a call, or exploring, lowers it)
  F1         = 2PR / (P + R)
  params     = for each expected call that was made directly, the share of its arguments that were right
               (best matching call; mean over the expected calls)
  exact      = every expected call made and no unexpected call

Ours differ from theirs in three small, deliberate ways, all because of our setup:
  - run_python counts as one tool call; it makes an expected call when its code calls that tool function
    (`degree(n)` inside a loop counts as calling degree). Its arguments can't be read, so it adds no params score.
  - argument pairs (u, v) and (source, target) also match swapped: all our graphs are undirected.
  - params takes the best matching call, not the first: shortest_path_via makes the same tool call twice.
"""
import re

from eval.tasks import EXPECTED_CALLS

UNSCORED = {"submit_answer", "cannot_answer", "read_result"}  # bookkeeping, like their WORKFLOW_TOOLS
SYMMETRIC = [("u", "v"), ("source", "target")]


def tools_in_code(code: str) -> set[str]:
    """Tool functions a run_python snippet calls: `degree(` counts, `G.degree(` (networkx) does not."""
    return set(re.findall(r"(?<![.\w])([a-z_]+)\(", code or ""))


def fills(slot: dict, call: dict) -> bool:
    if call["name"] in slot:
        return True
    return call["name"] == "run_python" and bool(tools_in_code((call.get("args") or {}).get("code")) & set(slot))


def same(expected, actual) -> bool:
    return actual == expected or str(actual) == str(expected)  # "3" and 3 are the same node, as in theirs


def args_score(spec: dict, params: dict, args: dict) -> float:
    """Share of the expected arguments that are right; a symmetric pair also counts when swapped."""
    if not spec:
        return 1.0
    expected = {arg: params[key] for arg, key in spec.items()}
    right = {arg for arg, value in expected.items() if same(value, args.get(arg))}
    for a, b in SYMMETRIC:
        if a in expected and b in expected and same(expected[a], args.get(b)) and same(expected[b], args.get(a)):
            right |= {a, b}
    return len(right) / len(expected)


def process_metrics(task: str, params: dict, calls: list[dict]) -> dict | None:
    """calls: the run's tool_call events in order ({"name", "args"}). None if the task has no expected calls."""
    slots = EXPECTED_CALLS.get(task)
    if not slots:
        return None
    scored = [c for c in calls if c["name"] not in UNSCORED]
    made = [any(fills(slot, c) for c in scored) for slot in slots]
    recall = sum(made) / len(slots)
    precision = min(1.0, sum(made) / len(scored)) if scored else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    param_scores = []
    for slot in slots:
        direct = [c for c in scored if c["name"] in slot]
        if direct:
            param_scores.append(max(args_score(slot[c["name"]], params, c.get("args") or {}) for c in direct))
    unexpected = [c["name"] for c in scored if not any(fills(slot, c) for slot in slots)]
    return {"precision": precision, "recall": recall, "f1": f1,
            "params": sum(param_scores) / len(param_scores) if param_scores else None,
            "exact": all(made) and not unexpected, "calls": len(scored), "unexpected": unexpected}
