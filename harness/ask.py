"""Ask the agent one question about one graph and print the conversation.

    uv run python -m harness.ask "Is there a cycle in G?" --answer-type yes_no
    uv run python -m harness.ask "What is the degree of node 4?" --graph data/graphs/er/small/0.txt
"""
import argparse

import networkx as nx

from harness.answers import ANSWER_TYPES
from harness.loop import AgentConfig, run_agent
from harness.models import DEFAULT_MODEL
from harness.trace import Trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask the agent one question about a graph.")
    parser.add_argument("question", nargs="?", default="is there any cycles?")
    parser.add_argument("--answer-type", default="yes_no", choices=list(ANSWER_TYPES))
    parser.add_argument("--graph", default="data/graphs/er/large/0.txt", help="graph file (networkx adjacency list)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="any LiteLLM model name")
    args = parser.parse_args()

    graph = nx.read_adjlist(args.graph, nodetype=int)
    trace = Trace("results/harness_runs/ask.jsonl")
    result = run_agent(args.question, graph, answer_type=args.answer_type, config=AgentConfig(model=args.model),
                       trace=trace)

    for m in result.messages:
        print(f"[{m['role']}]", m.get("content") or "", m.get("tool_calls") or "")
    print(f"\nStatus: {result.status} | answer: {result.answer}" + (f" | reason: {result.reason}" if result.reason else ""))
    print(f"Trace: {trace.path} (run_id {trace.run_id})")
