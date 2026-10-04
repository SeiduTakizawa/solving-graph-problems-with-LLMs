"""Task registry: what the harness knows about each kind of question.

Only things the agent may use live here: how the question is asked, what shape the answer has, and the
verifier that checks it. Ground truth and grading live in eval/tasks.py and never reach the agent.
"""
from dataclasses import dataclass
from typing import Callable

from harness.verifiers import verify_shortest_path


@dataclass(frozen=True)
class Task:
    name: str
    question: str  # template, filled in with the task's params, e.g. "What is the degree of node {node}?"
    answer_type: str  # "number", "yes_no" or "node_list" (see hello_agent.ANSWER_TYPES)
    verify: Callable | None = None  # verify(graph, params, answer) -> error message or None

    def make_question(self, params: dict) -> str:
        return self.question.format(**params)


TASKS = {task.name: task for task in [
    Task("node_count", "How many nodes does G have?", "number"),
    Task("edge_count", "How many edges does G have?", "number"),
    Task("node_degree", "What is the degree of node {node}?", "number"),
    Task("connected_nodes", "Which nodes are the neighbors of node {node}?", "node_list"),
    Task("edge_existence", "Is there an edge between nodes {u} and {v}?", "yes_no"),
    Task("connectivity", "Is there a path between nodes {source} and {target}?", "yes_no"),
    Task("connected_components_count", "How many connected components does G have?", "number"),
    Task("cycle_check", "Is there a cycle in G?", "yes_no"),
    Task("shortest_path",
         "What is a shortest path from node {source} to node {target}? Answer with the list of nodes on the path.",
         "node_list", verify=verify_shortest_path),
]}
