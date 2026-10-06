"""The two tools that end a run: submit_answer (one per answer type) and cannot_answer."""
import json

# The shape of the final answer, per kind of question. submit_answer is built to match.
ANSWER_TYPES = {
    "number": {"schema": {"type": "integer"}, "description": "a whole number"},
    "yes_no": {"schema": {"type": "boolean"}, "description": "true or false"},
    "node_list": {"schema": {"type": "array", "items": {"type": "integer"}}, "description": "a list of node ids"},
    "edge_list": {"schema": {"type": "array",
                             "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
                  "description": "a list of edges, each a pair of node ids like [0, 2]"},
    # Yes/no answers that come with evidence for a "yes", so a verifier can check them. The answer itself is
    # still true/false; the evidence is an extra field of submit_answer that is passed on to the verifier.
    "yes_no_with_path": {"schema": {"type": "boolean"}, "description": "true or false",
                         "evidence": {"path": {"type": "array", "items": {"type": "integer"},
                                               "description": "If your answer is true: the path you found, "
                                                              "as a list of nodes from the first node to the second."}}},
    "yes_no_with_cycle": {"schema": {"type": "boolean"}, "description": "true or false",
                          "evidence": {"cycle": {"type": "array", "items": {"type": "integer"},
                                                 "description": "If your answer is true: the nodes of one cycle "
                                                                "in order, e.g. [0, 1, 2] for the cycle 0-1-2-0."}}},
}

ENDING_TOOL_NAMES = ["submit_answer", "cannot_answer"]


def make_submit_answer(answer_type: str) -> dict:
    """The submit_answer tool for one answer type. Ends the task; the loop handles it itself."""
    expected = ANSWER_TYPES[answer_type]
    evidence = expected.get("evidence", {})  # optional extra fields, e.g. the path behind a "yes"
    return {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": f"Submit your final answer, which must be {expected['description']}. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": expected["schema"], **evidence},
                "required": ["answer"],
            },
        },
    }


def check_answer(answer, answer_type: str) -> str | None:
    """None if the answer has the right shape, otherwise an error message for the model."""
    # Careful: in Python True/False are also ints, so a number must not be a bool and vice versa.
    is_int = lambda x: isinstance(x, int) and not isinstance(x, bool)
    ok = {
        "number": is_int(answer),
        "yes_no": isinstance(answer, bool),
        "yes_no_with_path": isinstance(answer, bool),
        "yes_no_with_cycle": isinstance(answer, bool),
        "node_list": isinstance(answer, list) and all(is_int(x) for x in answer),
        "edge_list": isinstance(answer, list) and all(
            isinstance(e, list) and len(e) == 2 and all(is_int(x) for x in e) for e in answer),
    }[answer_type]
    if ok:
        return None
    return (f"Invalid answer {json.dumps(answer)}: the answer must be {ANSWER_TYPES[answer_type]['description']}. "
            "Call submit_answer again with the right type.")


def evidence_fields(answer_type: str) -> list[str]:
    """The extra submit_answer fields of an answer type that go to the verifier (none for most types)."""
    return list(ANSWER_TYPES[answer_type].get("evidence", {}))


CANNOT_ANSWER = {
    "type": "function",
    "function": {
        "name": "cannot_answer",
        "description": "Use this when the question cannot be answered with the available tools, "
                       "or the question itself is invalid. This ends the task.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
}
