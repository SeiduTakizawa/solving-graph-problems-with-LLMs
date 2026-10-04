"""Append-only JSONL trace: one line per event, so every run can be replayed and measured later."""
import json
import time
import uuid
from pathlib import Path


class Trace:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]  # groups the events of one run inside a shared file

    def log(self, event: str, **data) -> None:
        line = {"run_id": self.run_id, "time": time.time(), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")


def read_trace(path: str | Path) -> list[dict]:
    """Load all events from a trace file."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
