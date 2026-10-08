"""Short versions of values that end up in messages to the model.

Rule: nothing the model reads may grow with the graph. Tool results are handled by result handles; this covers
everything else that repeats a value back (a wrong answer, a cycle, an exception): it shows the start and the size.
"""
import json

MAX_CHARS = 150


def brief(value, max_chars: int = MAX_CHARS) -> str:
    """e.g. '[[0, 1], [1, 2], [2, 3], ...] (9999 items)' instead of 9,999 edges."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= max_chars:
        return text
    if isinstance(value, list):
        shown = []
        for item in value:  # as many leading items as fit
            candidate = json.dumps(shown + [item], default=str)
            if len(candidate) > max_chars - 30:
                break
            shown.append(item)
        start = json.dumps(shown, default=str)[:-1]
        return f"{start}{', ' if shown else ''}...] ({len(value)} items)"
    return text[:max_chars] + f"... ({len(text)} characters)"
