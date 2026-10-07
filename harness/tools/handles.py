"""Result handles: big tool outputs are stored harness-side; the model sees a short summary and an ID.

A 10,000-node graph can give a tool result of 20,000 numbers (an MST, a component list). Sent as-is it would
fill the context and be re-read on every later step. Instead:

    {"edges": [[0, 1], [0, 7], ...9,999 edges]}
becomes
    {"edges": {"handle": "result_1", "length": 9999, "first_5": [[0, 1], [0, 7], ...]}, "note": "..."}

The model can page through a stored value with read_result, and can submit a handle as its answer
("result_1"), so it never has to copy a huge list. One store per run: handles don't outlive the question.
"""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

HANDLE_LIMIT = 200  # values with more numbers than this are stored; the current dataset never gets close (max 98)
PREVIEW = 5  # items shown in the summary

NOTE = ("Large values were stored and replaced by a handle. Use read_result(handle, offset, limit) to read them, "
        "or pass the handle (e.g. \"result_1\") to submit_answer instead of copying the values.")


def size(value) -> int:
    """How many numbers (or other plain values) a value holds, e.g. 49 edges -> 98."""
    if isinstance(value, dict):
        return sum(size(v) for v in value.values())
    if isinstance(value, list):
        return sum(size(v) for v in value)
    return 1


class HandleStore:
    def __init__(self, limit: int = HANDLE_LIMIT):
        self.limit = limit
        self.values: dict[str, list] = {}

    def put(self, value: list) -> str:
        handle = f"result_{len(self.values) + 1}"
        self.values[handle] = value
        return handle

    def compact(self, result: dict) -> dict:
        """The result as the model should see it: every value holding more than `limit` numbers is stored and
        replaced by a summary with its handle."""
        compacted = {key: self.shrink(value) for key, value in result.items()}
        if compacted != result:
            compacted["note"] = NOTE
        return compacted

    def shrink(self, value):
        if not isinstance(value, list) or size(value) <= self.limit:
            return value
        # A list of big items (e.g. components of 9,000 nodes): each big item gets its own handle,
        # so even the summary stays small. Otherwise the whole list is stored as one handle.
        preview = value[:PREVIEW]
        if any(size(item) > self.limit for item in value):
            shrunk = [self.shrink(item) for item in value]
            if size(shrunk) <= self.limit:
                return shrunk
            preview = shrunk[:PREVIEW]  # too many big items: store the list, preview their summaries
        return {"handle": self.put(value), "length": len(value), f"first_{PREVIEW}": preview}

    def resolve(self, value):
        """A handle string becomes the stored value; anything else is returned unchanged.
        ["result_1"] counts too: when the schema says "array", models wrap the handle in one."""
        if isinstance(value, list) and len(value) == 1:
            value = value[0] if isinstance(value[0], str) and value[0] in self.values else value
        if isinstance(value, str) and value in self.values:
            return self.values[value]
        return value

    def read(self, args: dict) -> dict:
        """The read_result tool: a slice of a stored value."""
        try:
            a = ReadArgs.model_validate(args)
        except ValidationError as e:
            return {"error": f"Bad arguments for read_result: {e.errors()[0]['loc'][0]}: {e.errors()[0]['msg']}"}
        if a.handle not in self.values:
            known = sorted(self.values) or "none yet"
            return {"error": f"Unknown handle {a.handle}. Stored handles: {known}."}
        value = self.values[a.handle]
        items = value[a.offset:a.offset + a.limit]
        return {"handle": a.handle, "length": len(value), "offset": a.offset, "items": items,
                "more": a.offset + len(items) < len(value)}


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handle: str
    offset: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 50


READ_RESULT = {
    "type": "function",
    "function": {
        "name": "read_result",
        "description": "Read part of a large stored result by its handle (e.g. \"result_1\"): "
                       "up to `limit` items starting at `offset`.",
        "parameters": {
            "type": "object",
            "properties": {"handle": {"type": "string"},
                           "offset": {"type": "integer", "minimum": 0},
                           "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "required": ["handle"],
        },
    },
}
