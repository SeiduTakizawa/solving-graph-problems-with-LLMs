"""The two tools that end a run: submit_answer (one per answer type) and cannot_answer."""
from typing import Annotated

from annotated_types import Len
from pydantic import StrictBool, StrictInt, TypeAdapter, ValidationError

from harness.brief import brief

# Pydantic type of each answer. It checks submitted answers and generates the schema the model sees.
# Strict types: in Python True/False are also ints, so a number must not be a bool and vice versa.
Edge = Annotated[list[StrictInt], Len(2, 2)]
ANSWER_MODELS = {
    "number": StrictInt,
    "yes_no": StrictBool,
    "node_list": list[StrictInt],
    "edge_list": list[Edge],
    "yes_no_with_path": StrictBool,
    "yes_no_with_cycle": StrictBool,
    "number_with_components": StrictInt,
    "number_with_triangles": StrictInt,
    # For a question the router couldn't place (harness/router.py): any of the shapes above, and no verifier.
    "any": StrictInt | StrictBool | list[StrictInt] | list[Edge],
}
_ADAPTERS = {name: TypeAdapter(t) for name, t in ANSWER_MODELS.items()}


def _schema(answer_type: str) -> dict:
    """The answer's JSON schema. Pydantic sorts the keys alphabetically; they are put back in the order the
    hand-written schemas had, so the text the model sees is byte-identical to earlier runs."""
    schema = _ADAPTERS[answer_type].json_schema()
    if "anyOf" in schema:  # "any": its options, each in the short form
        return {"anyOf": [{"type": "integer"}, {"type": "boolean"}, _schema("node_list"), _schema("edge_list")]}
    if "items" in schema and "items" in schema["items"]:  # edge_list: a list of [u, v] pairs
        schema["items"] = {"type": "array", "items": schema["items"]["items"],
                           "minItems": schema["items"]["minItems"], "maxItems": schema["items"]["maxItems"]}
    return {"type": schema["type"], "items": schema["items"]} if "items" in schema else schema


# The shape of the final answer, per kind of question. submit_answer is built to match.
ANSWER_TYPES = {
    "number": {"schema": _schema("number"), "description": "a whole number"},
    "yes_no": {"schema": _schema("yes_no"), "description": "true or false"},
    "node_list": {"schema": _schema("node_list"), "description": "a list of node ids"},
    "edge_list": {"schema": _schema("edge_list"), "description": "a list of edges, each a pair of node ids like [0, 2]"},
    # Yes/no answers that come with evidence for a "yes", so a verifier can check them. The answer itself is
    # still true/false; the evidence is an extra field of submit_answer that is passed on to the verifier.
    "yes_no_with_path": {"schema": _schema("yes_no_with_path"), "description": "true or false",
                         "evidence": {"path": {"type": "array", "items": {"type": "integer"},
                                               "description": "If your answer is true: the path you found, "
                                                              "as a list of nodes from the first node to the second."}}},
    "yes_no_with_cycle": {"schema": _schema("yes_no_with_cycle"), "description": "true or false",
                          "evidence": {"cycle": {"type": "array", "items": {"type": "integer"},
                                                 "description": "If your answer is true: the nodes of one cycle "
                                                                "in order, e.g. [0, 1, 2] for the cycle 0-1-2-0."}}},
    # Counts that come with what was counted (v8), so a verifier can check them without counting again.
    "number_with_components": {"schema": _schema("number_with_components"), "description": "a whole number",
                               "evidence": {"components": {"type": "array", "items": {"type": "array",
                                                                                      "items": {"type": "integer"}},
                                                           "description": "The components you counted: one list of "
                                                                          "nodes per component."}}},
    "number_with_triangles": {"schema": _schema("number_with_triangles"), "description": "a whole number",
                              "evidence": {"triangles": {"type": "array", "items": {"type": "array",
                                                                                    "items": {"type": "integer"}},
                                                         "description": "The triangles you counted, each as its three "
                                                                        "nodes, e.g. [[4, 1, 7], [4, 2, 9]]."}}},
    "any": {"schema": _schema("any"),
            "description": "a whole number, true or false, a list of node ids, or a list of edges"},
}

ENDING_TOOL_NAMES = ["submit_answer", "cannot_answer"]


HANDLE_SCHEMA = {"type": "string", "description": "or the handle of a stored result, e.g. \"result_1\""}


def make_submit_answer(answer_type: str, handles: bool = False) -> dict:
    """The submit_answer tool for one answer type. Ends the task; the loop handles it itself.

    handles=True (once a big result was stored): the answer, and any evidence, may also be a handle string.
    Without it the schema would say "array" and the model couldn't send "result_1" at all.
    """
    expected = ANSWER_TYPES[answer_type]
    evidence = expected.get("evidence", {})  # optional extra fields, e.g. the path behind a "yes"
    answer_schema = expected["schema"]
    if handles:
        answer_schema = {"anyOf": [answer_schema, HANDLE_SCHEMA]}
        evidence = {name: {"anyOf": [{k: v for k, v in schema.items() if k != "description"}, HANDLE_SCHEMA],
                           "description": schema["description"]} for name, schema in evidence.items()}
    return {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": f"Submit your final answer, which must be {expected['description']}. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": answer_schema, **evidence},
                "required": ["answer"],
            },
        },
    }


def check_answer(answer, answer_type: str) -> str | None:
    """None if the answer has the right shape, otherwise an error message for the model."""
    try:
        _ADAPTERS[answer_type].validate_python(answer)
        return None
    except ValidationError:
        return (f"Invalid answer {brief(answer)}: the answer must be {ANSWER_TYPES[answer_type]['description']}. "
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
