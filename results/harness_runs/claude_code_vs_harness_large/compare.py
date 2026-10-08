"""Our harness (qwen3:8b + tools) vs Claude Code (Sonnet + python/networkx) on large dev graphs."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/thrig/Desktop/solving-graph-problems-with-LLMs")
sys.path.insert(0, str(ROOT))
from eval.tasks import GRAPHS_DIR, is_correct, load_graph, load_tasks
from harness.loop import run_agent
from harness.tasks import TASKS
from harness.trace import Trace, read_trace

OUT = Path(__file__).parent
PY = ROOT / ".venv/bin/python"
TASK_NAMES, N, SIZE = ["mst", "connected_nodes", "node_degree"], 3, "large"
FORMAT = {"number": "a single integer", "node_list": "a list of node ids, e.g. [1, 4, 7]",
          "edge_list": "a list of edges, each a pair of node ids, e.g. [[0, 2], [2, 5]]"}


def to_ints(x):
    if isinstance(x, list):
        return [to_ints(v) for v in x]
    if isinstance(x, str) and x.lstrip("-").isdigit():
        return int(x)
    return x


def ours(item, graph):
    task = TASKS[item["task"]]
    trace = Trace(OUT / "ours_traces.jsonl")
    verify = (lambda a, g=graph, p=item["params"], c=task.verify, **ev: c(g, p, a, **ev)) if task.verify else None
    start = time.time()
    r = run_agent(item["question"], graph, answer_type=task.answer_type, verify=verify, trace=trace)
    end = [e for e in read_trace(trace.path) if e["run_id"] == trace.run_id and e["event"] == "run_end"][0]
    return {"answer": r.answer, "status": r.status, "time_s": round(time.time() - start, 1), "steps": end["steps"],
            "input_tokens": end["prompt_tokens"], "output_tokens": end["completion_tokens"], "cost_usd": 0.0}


def claude(item, graph_file):
    task = TASKS[item["task"]]
    work = OUT / "cc_work"
    work.mkdir(exist_ok=True)
    (work / "graph.txt").write_text(graph_file.read_text())
    prompt = (f"The file graph.txt in the current directory is an undirected graph G in networkx adjacency-list "
              f"format; node ids are integers. {item['question']} You can run Python with networkx using: {PY}. "
              f"On the last line of your reply, output only JSON like {{\"answer\": ...}}, where the answer is "
              f"{FORMAT[task.answer_type]}.")
    start = time.time()
    p = subprocess.run(["claude", "-p", prompt, "--model", "sonnet", "--output-format", "json", "--max-turns", "10",
                        "--allowedTools", f"Bash({PY}:*)", "Read"], cwd=work, capture_output=True, text=True,
                       timeout=600)
    wall = round(time.time() - start, 1)
    d = json.loads(p.stdout)
    answer = None
    try:
        answer = to_ints(json.loads(d["result"].strip().splitlines()[-1])["answer"])
    except Exception:
        pass
    u = d.get("usage", {})
    return {"answer": answer, "status": "submitted" if answer is not None else "unparsed", "time_s": wall,
            "steps": d.get("num_turns"),
            "input_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0),
            "cache_read": u.get("cache_read_input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
            "cost_usd": d.get("total_cost_usd"), "model": list((d.get("modelUsage") or {}).keys())}


items = [i for t in TASK_NAMES for i in load_tasks(t, SIZE, "dev", N)]
with (OUT / "results.jsonl").open("a") as f:
    for system in ["ours", "claude"]:
        for item in items:
            graph = load_graph(SIZE, item["graph_id"])
            r = ours(item, graph) if system == "ours" else claude(item, GRAPHS_DIR / SIZE / f"{item['graph_id']}.txt")
            r.update(system=system, task=item["task"], graph_id=item["graph_id"], params=item["params"],
                     correct=is_correct(item["task"], graph, item["params"], r["answer"]))
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"{system:6} {item['task']:16} graph {item['graph_id']:>2} {'✓' if r['correct'] else '✗'} "
                  f"{r['time_s']:>6}s in={r['input_tokens']:>6} out={r['output_tokens']:>5} ${r['cost_usd']}",
                  flush=True)
