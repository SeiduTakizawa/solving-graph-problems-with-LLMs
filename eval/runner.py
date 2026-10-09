"""Run the agent on dataset questions and record whether each answer is correct.

Examples (dev graphs, 3 runs):
    uv run python -m eval.runner --task node_degree --size small --n 100
    uv run python -m eval.runner --task all --size small --n 30
    uv run python -m eval.runner --task shortest_path --no-verify     # ablation: no verifier
    uv run python -m eval.runner --task shortest_path_via --python tools --code-only   # force code: no direct tools
    uv run python -m eval.runner --task all --router llm --tools hybrid   # M4: routed questions, the task's tools

Routing (M4): --router oracle (default) takes the task and parameters from the dataset, as before. regex / llm
route the question text (harness/router.py); the answer type, the verifier and (with --tools task / hybrid) the
tools then come from the routed task, and grading still uses the dataset's task. An unknown route runs with all
tools, the `any` answer type and no verifier.

Writes to results/harness_runs/<name>/:
    traces.jsonl   every event of every run (from harness.trace)
    results.jsonl  one line per question per run: answer, reference answer, correct, status, tokens, latency
"""
import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path

from eval.analysis.report import RESULTS_DIR, load_rows, report
from eval.tasks import SPLITS, is_correct, load_graph, load_tasks, reference_answer
from eval.routing import PARAPHRASES, same_params
from harness.loop import AgentConfig, RunResult, run_agent
from harness.models import DEFAULT_MODEL
from harness.router import UNKNOWN, Route, check_route, llm_route, regex_route
from harness.skills import load_skill
from harness.tasks import TASKS
from harness.tools.graph_tools import DEFAULT_TOOLS, TOOLS
from harness.trace import Trace

def run(args) -> Path:
    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    task_names = sorted(TASKS) if args.task == "all" else [args.task]
    items = [item for name in task_names for item in load_tasks(name, args.size, args.split, args.n)]
    if args.phrasing == "paraphrase":
        items = paraphrased(items, args.split)
    graphs = {}
    # --code-only offers no graph tool directly: run_python is the only way in (its code can still call the tools).
    chosen = tuple(name for name in TOOLS
                   if (name in DEFAULT_TOOLS or name in args.with_tool) and name not in args.without_tool)
    base = AgentConfig(model=args.model, graph_tools=() if args.code_only else chosen,
                       code_tools=DEFAULT_TOOLS + tuple(args.with_tool) if args.code_only else None,
                       python=args.python, code_hint=args.code_hint)

    total, done, started = args.runs * len(items), 0, time.time()
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as results:
        for run_no in range(1, args.runs + 1):
            for i, item in enumerate(items, 1):
                task, params, graph_id = TASKS[item["task"]], item["params"], item["graph_id"]
                if graph_id not in graphs:
                    graphs[graph_id] = load_graph(args.size, graph_id)
                graph = graphs[graph_id]
                start = time.time()
                route = check_route(make_route(args, item), graph)
                routed = TASKS[route.task] if route.known else None
                # verify(answer, **evidence): the routed task's verifier with the graph and routed parameters filled in
                verify = (partial(routed.verify, graph, route.params)
                          if routed and routed.verify and not args.no_verify else None)
                answer_type = routed.answer_type if routed else "any"
                config = replace(base, **tool_exposure(args, routed),
                                 skill=load_skill(routed.name) if routed and args.skills else None)

                trace = Trace(out_dir / "traces.jsonl")
                try:
                    result = run_agent(item["question"], graph, answer_type=answer_type, verify=verify,
                                       config=config, trace=trace)
                except Exception as e:  # e.g. Ollama not running; record it and keep going
                    result = RunResult(None, f"error: {e}", [])
                correct = is_correct(task.name, graph, params, result.answer)
                crashed = result.status.startswith("error")  # no token counts for these

                row = {
                    "run": run_no, "run_id": trace.run_id, "model": args.model, "size": args.size, "split": args.split,
                    **item, "answer_type": task.answer_type, "verified": verify is not None,
                    "without_tools": args.without_tool, "with_tools": args.with_tool, "python": args.python, "code_only": args.code_only, "code_hint": args.code_hint,
                    "reference": reference_answer(task.name, graph, params), "answer": result.answer,
                    "evidence": result.evidence, "correct": correct,
                    "status": result.status, "rescued": result.rescued, "rejected": result.rejected,
                    "compactions": result.compactions,
                    "router": args.router, "tools_mode": args.tools, "phrasing": args.phrasing, "skill": config.skill is not None, "routed_task": route.task,
                    "routed_params": route.params, "route_reason": route.reason, "answer_type_used": answer_type,
                    "route_right": route.task == task.name and same_params(task.name, params, route.params),
                    "graph_tools_offered": None if config.graph_tools is None else list(config.graph_tools),
                    "route_prompt_tokens": route.prompt_tokens, "route_completion_tokens": route.completion_tokens,
                    "route_latency_s": route.latency_s,
                    "steps": None if crashed else result.steps,
                    "prompt_tokens": None if crashed else result.prompt_tokens,
                    "completion_tokens": None if crashed else result.completion_tokens,
                    "latency_s": round(time.time() - start, 2),  # wall clock: model + tools + verifier
                }
                results.write(json.dumps(row) + "\n")
                results.flush()
                done += 1
                print(f"run {run_no} [{i}/{len(items)}] {task.name} graph {graph_id} {params}: "
                      + ("" if args.router == "oracle" else f"routed {route.task}{'' if route.task == task.name else ' ✗'}, ")
                      + f"{result.status} answer={_short(result.answer)} {'✓' if correct else '✗'}  "
                      f"{_progress(done, total, started)}", flush=True)
    return out_dir


def paraphrased(items: list[dict], split: str) -> list[dict]:
    """The same questions in the hand-written phrasings (data/routing/paraphrases.json, this split), one per
    question in turn. The template's instructions after its first "?" are kept (e.g. "If several are tied, answer
    with the smallest node id."): they define the expected answer, so grading stays fair."""
    phrasings = json.loads(PARAPHRASES.read_text())
    out, seen = [], Counter()
    for item in items:
        template = TASKS[item["task"]].question
        options = phrasings[item["task"]][split]
        phrasing = options[seen[item["task"]] % len(options)]
        seen[item["task"]] += 1
        rest = template.split("? ", 1)[1] if "? " in template else ""
        question = phrasing.format(**item["params"])
        if rest:
            question += ("" if question.endswith(("?", ".", "!")) else ".") + " " + rest.format(**item["params"])
        out.append({**item, "question": question, "template_question": item["question"]})
    return out


def make_route(args, item: dict) -> Route:
    """The question's route before checking: the dataset's own task (oracle) or a router's guess from the text."""
    if args.router == "oracle":
        return Route(item["task"], item["params"], 1.0, "oracle")
    try:
        if args.router == "regex":
            return regex_route(item["question"])
        return llm_route(item["question"], args.router_model or args.model)
    except Exception as e:  # e.g. Ollama not running: route unknown, the run still happens
        return Route(UNKNOWN, {}, 0.0, args.router, reason=f"router error: {e}")


def tool_exposure(args, routed) -> dict:
    """Which graph tools the agent starts with (AgentConfig fields), per --tools:
    all: every default tool (as before M4); task: only the routed task's tools; hybrid: those plus more_tools;
    agent: none, only more_tools. An unknown route gets all tools in task / hybrid mode."""
    if args.code_only or args.tools == "all":
        return {}
    if args.tools == "agent":
        return {"graph_tools": (), "more_tools": True}
    if routed is None:
        return {}
    family = tuple(t for t in routed.tools if t not in args.without_tool)
    return {"graph_tools": family, "more_tools": args.tools == "hybrid"}


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
    text, _ = report([name], by_task=True, route=True)
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
    parser.add_argument("--with-tool", action="append", default=[], metavar="TOOL",
                        choices=[name for name in TOOLS if name not in DEFAULT_TOOLS],
                        help="also offer this opt-in tool (e.g. bridges, k_core); can be repeated")
    parser.add_argument("--python", choices=["tools", "networkx"], default=None,
                        help="offer run_python (needs Docker): code that calls our tools, or also networkx")
    parser.add_argument("--code-hint", action="store_true",
                        help="tell the model when to combine tools and code (harness/prompts.py CODE_HINT); needs --python")
    parser.add_argument("--code-only", action="store_true",
                        help="offer only run_python, no graph tools directly (needs --python); forces code")
    parser.add_argument("--router", default="oracle", choices=["oracle", "regex", "llm"],
                        help="oracle: task and parameters from the dataset (default); regex / llm: route the question")
    parser.add_argument("--phrasing", default="template", choices=["template", "paraphrase"],
                        help="ask the dataset's template questions, or the hand-written paraphrases (for routing)")
    parser.add_argument("--router-model", default=None, help="LiteLLM model for --router llm (default: --model)")
    parser.add_argument("--tools", default="all", choices=["all", "task", "hybrid", "agent"],
                        help="tool exposure: all default tools, the routed task's tools, those + more_tools, "
                             "or only more_tools (M4 ablation)")
    parser.add_argument("--skills", action="store_true",
                        help="add the routed task's playbook (harness/skills/<task>.md) to the system prompt")
    parser.add_argument("--name", default=None, help="results folder name (default: <task>_<size>_<split>_<time>)")
    parser.add_argument("--summary-only", action="store_true", help="only summarize an existing --name")
    args = parser.parse_args()
    if (args.code_only or args.code_hint) and not args.python:
        parser.error("--code-only and --code-hint need --python tools or --python networkx")
    args.name = args.name or f"{args.task}_{args.size}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"

    if not args.summary_only:
        run(args)
    summarize(args.name)
