"""Spotting and recovering tool calls the model wrote as plain text instead of calling them."""
import json
import re

from harness.answers import ENDING_TOOL_NAMES
from harness.tools.graph_tools import TOOLS

TOOL_NAMES = list(TOOLS) + ["read_result"] + ENDING_TOOL_NAMES


def looks_like_text_tool_call(content: str | None) -> bool:
    """True if a plain-text reply contains a hand-written tool call, e.g. 'submit_answer\\n{"answer": 0}'."""
    if not content:
        return False
    if "tool_call>" in content:  # leaked <tool_call> / </tool_call> tags
        return True
    names = "|".join(TOOL_NAMES)
    return re.search(rf"^\s*({names})\s*\(?\s*\{{", content, re.MULTILINE) is not None


def parse_text_tool_call(content: str | None) -> tuple[str, dict] | None:
    """Recover a tool call the model wrote as text instead of calling it. Returns (name, args) or None.

    Handles the two forms qwen3 produces:
      submit_answer\\n{"answer": [2, 5]}\\n</tool_call>                        (name, then arguments)
      <tool_call>{"name": "get_neighbors", "arguments": {"node": 4}}</tool_call>  (its own format)
    """
    if not content:
        return None
    decoder = json.JSONDecoder()

    def json_at(index):
        try:
            value, _ = decoder.raw_decode(content, index)
            return value
        except json.JSONDecodeError:
            return None

    for match in re.finditer(r"\{", content):
        value = json_at(match.start())
        if (isinstance(value, dict) and value.get("name") in TOOL_NAMES
                and isinstance(value.get("arguments"), dict)):
            return value["name"], value["arguments"]

    names = "|".join(TOOL_NAMES)
    for match in re.finditer(rf"^\s*({names})\s*\(?\s*(?=\{{)", content, re.MULTILINE):
        value = json_at(match.end())
        if isinstance(value, dict):
            return match.group(1), value
    return None
