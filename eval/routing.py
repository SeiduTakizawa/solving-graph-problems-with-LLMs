"""Routing eval (M4): how well does a router turn a question into the right task and parameters?

    uv run python -m eval.routing --router regex                       # paraphrases, dev split
    uv run python -m eval.routing --router llm --model ollama_chat/qwen3.5:9b --name route_llm_qwen35
    uv run python -m eval.routing --router regex --set templates       # the dataset's own wording (a sanity check)

Questions: data/routing/paraphrases.json (hand-written phrasings, dev / test, plus out-of-scope "unknown" ones),
filled with real parameters from the dataset (dev graphs for dev phrasings, test graphs for test phrasings).
Each route goes through check_route with the question's graph, exactly as it would before a real run.

Outcomes per question:
  right         the task and the parameters are right (for out-of-scope questions: unknown)
  safe_miss     an in-scope question routed to unknown: the run still happens, with all tools and no verifier
  wrong         a known task that is wrong, or the right task with wrong parameters: the dangerous case (a wrong
                verifier, or a right verifier checking a different question)
  false_known   an out-of-scope question routed to a task: also dangerous (it is answered as something it isn't)
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from eval.tasks import ROOT, load_graph, load_tasks
from harness.models import DEFAULT_MODEL
from harness.prompts import ROUTER_VERSION
from harness.router import UNKNOWN, Route, check_route, llm_route, regex_route, task_params
from harness.tasks import TASKS

PARAPHRASES = ROOT / "data" / "routing" / "paraphrases.json"
RESULTS_DIR = ROOT / "results" / "routing"
SIZE = "small"  # node ids only need to exist; small graphs load fast
SYMMETRIC = {"edge_existence": ("u", "v"), "connectivity": ("source", "target")}  # undirected: order doesn't matter


def routing_questions(split: str = "dev", which: str = "paraphrases") -> list[dict]:
    """[{question, task, params, graph_id}], task UNKNOWN for out-of-scope questions (graph_id 0, no params)."""
    phrasings = json.loads(PARAPHRASES.read_text())
    items = []
    for task in TASKS:
        templates = phrasings[task][split] if which == "paraphrases" else [TASKS[task].question]
        examples = load_tasks(task, SIZE, split, n=len(templates)) if task_params(task) else []
        for i, template in enumerate(templates):
            example = examples[i % len(examples)] if examples else {"params": {}, "graph_id": None}
            items.append({"question": template.format(**example["params"]), "task": task,
                          "params": example["params"], "graph_id": example["graph_id"]})
    if which == "paraphrases":
        items += [{"question": q, "task": UNKNOWN, "params": {}, "graph_id": None} for q in phrasings[UNKNOWN][split]]
    return items


def same_params(task: str, expected: dict, got: dict) -> bool:
    if expected == got:
        return True
    if task in SYMMETRIC:
        a, b = SYMMETRIC[task]
        return {**expected, a: expected[b], b: expected[a]} == got
    return False


def outcome(item: dict, route: Route) -> str:
    if item["task"] == UNKNOWN:
        return "right" if not route.known else "false_known"
    if not route.known:
        return "safe_miss"
    return "right" if route.task == item["task"] and same_params(item["task"], item["params"], route.params) else "wrong"


def run(args) -> Path:
    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    items = routing_questions(args.split, args.set)
    graphs = {}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as results:
        for i, item in enumerate(items, 1):
            start = time.time()
            try:
                raw = regex_route(item["question"]) if args.router == "regex" else llm_route(item["question"], args.model)
            except NotImplementedError:
                raise SystemExit(f"{args.router}_route is still a TODO in harness/router.py")
            except Exception as e:  # e.g. Ollama not running; record it and keep going
                raw = Route(UNKNOWN, {}, 0.0, args.router, reason=f"error: {e}")
            gid = item["graph_id"]
            if gid is not None and gid not in graphs:
                graphs[gid] = load_graph(SIZE, gid)
            route = check_route(raw, graphs.get(gid), args.min_confidence)
            row = {**item, "router": args.router, "model": args.model if args.router == "llm" else None,
                   "router_version": ROUTER_VERSION, "split": args.split, "set": args.set,
                   "raw_task": raw.task, "raw_params": raw.params, "routed_task": route.task,
                   "routed_params": route.params, "confidence": raw.confidence, "reason": route.reason,
                   "outcome": outcome(item, route), "prompt_tokens": raw.prompt_tokens,
                   "completion_tokens": raw.completion_tokens, "latency_s": round(time.time() - start, 3)}
            results.write(json.dumps(row) + "\n")
            results.flush()
            print(f"[{i}/{len(items)}] {row['outcome']:11s} {item['task']:26s} -> {route.task:26s} {item['question']}")
    summarize(args.name)
    return out_dir


def summarize(name: str) -> dict:
    rows = [json.loads(line) for line in (RESULTS_DIR / name / "results.jsonl").open()]
    count = lambda kind: sum(r["outcome"] == kind for r in rows)
    in_scope = [r for r in rows if r["task"] != UNKNOWN]
    summary = {
        "questions": len(rows),
        "accuracy": count("right") / len(rows),
        "in_scope_right": sum(r["outcome"] == "right" for r in in_scope) / max(1, len(in_scope)),
        "safe_miss": count("safe_miss"), "wrong": count("wrong"), "false_known": count("false_known"),
        "latency_s_median": statistics.median(r["latency_s"] for r in rows),
        "tokens_per_route": statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
    }
    print(f"\n{name}: {summary['accuracy']:.0%} right ({len(rows)} questions; in scope {summary['in_scope_right']:.0%}) | "
          f"safe misses {summary['safe_miss']}, wrong {summary['wrong']}, false known {summary['false_known']} | "
          f"median {summary['latency_s_median']:.2f} s, {summary['tokens_per_route']:.0f} tokens/route")
    for r in rows:
        if r["outcome"] != "right":
            print(f"  {r['outcome']:11s} {r['task']:26s} -> {r['routed_task']:26s} {r['question']}"
                  + (f"  ({r['reason']})" if r["reason"] else ""))
    (RESULTS_DIR / name / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a router on the routing questions.")
    parser.add_argument("--router", required=True, choices=["regex", "llm"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM model for --router llm")
    parser.add_argument("--set", default="paraphrases", choices=["paraphrases", "templates"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="override MIN_CONFIDENCE in harness/router.py")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.min_confidence is None:
        from harness.router import MIN_CONFIDENCE
        args.min_confidence = MIN_CONFIDENCE
    args.name = args.name or f"route_{args.router}_{args.set}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"
    run(args)
